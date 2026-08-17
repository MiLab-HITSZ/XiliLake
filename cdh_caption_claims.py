# Copyright (c) 2026 MiLab. All rights reserved.
import re
from typing import Any, Dict, Iterable, List, Optional


NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "both": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}

_STOP_SHARED_TOKENS = {
    "a",
    "an",
    "the",
    "normal",
    "standard",
    "typical",
    "ordinary",
    "anomaly",
    "anomalous",
    "counterfactual",
    "commonsense",
    "color",
    "state",
    "style",
}

_CLAIM_SCHEMA_VERSION = "cdh_gen_claim_schema_v1"


def normalize_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = "".join(ch for ch in text if ch.isalnum() or ch.isspace())
    return " ".join(text.split())


def normalize_caption_text(text: str) -> str:
    words = normalize_text(text).split()
    return " ".join(NUMBER_WORDS.get(word, word) for word in words)


def split_caption_phrases(gt: str) -> List[str]:
    return [part.strip() for part in str(gt or "").split(",") if part.strip()]


def caption_phrase_match(pred_norm: str, phrase: str) -> bool:
    phrase_norm = normalize_caption_text(phrase)
    if not phrase_norm:
        return False
    if phrase_norm in pred_norm:
        return True
    if caption_numbered_head_match(pred_norm, phrase_norm):
        return True
    phrase_tokens = [tok for tok in phrase_norm.split() if tok not in {"a", "an", "the", "normal"}]
    pred_tokens = set(pred_norm.split())
    return bool(phrase_tokens) and all(tok in pred_tokens for tok in phrase_tokens)


_COUNT_HEAD_EQUIVALENTS = {
    "finger": {"finger", "fingers", "digit", "digits"},
    "fingers": {"finger", "fingers", "digit", "digits"},
    "digit": {"finger", "fingers", "digit", "digits"},
    "digits": {"finger", "fingers", "digit", "digits"},
}


def caption_head_variants(token: str) -> set[str]:
    token = str(token or "").strip()
    if not token:
        return set()
    variants = set(_COUNT_HEAD_EQUIVALENTS.get(token, {token}))
    if token.endswith("s") and len(token) > 1:
        variants.add(token[:-1])
    else:
        variants.add(token + "s")
    return variants


def caption_numbered_head_match(pred_norm: str, claim_norm: str) -> bool:
    claim_tokens = claim_norm.split()
    pred_tokens = pred_norm.split()
    number_positions = [(idx, tok) for idx, tok in enumerate(claim_tokens) if tok.isdigit()]
    if len(number_positions) != 1:
        return False

    number_idx, number = number_positions[0]
    stop = {"a", "an", "the", "normal", "visible", "fully", "clearly", "distinct"}
    content_after = [
        tok for tok in claim_tokens[number_idx + 1 :]
        if tok and not tok.isdigit() and tok not in stop
    ]
    content_anywhere = [
        tok for tok in claim_tokens
        if tok and not tok.isdigit() and tok not in stop
    ]
    head = (content_after or content_anywhere or [""])[-1]
    head_variants = caption_head_variants(head)
    if not head_variants:
        return False

    for idx, tok in enumerate(pred_tokens):
        if tok != number:
            continue
        window = pred_tokens[max(0, idx - 2) : min(len(pred_tokens), idx + 8)]
        if any(w in head_variants for w in window):
            return True
    return False


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, tuple):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    return [text] if text else []


def _dedupe(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
        key = normalize_caption_text(text)
        if not key or key in seen:
            continue
        out.append(text)
        seen.add(key)
    return out


def split_multiple_choice_options(options: Any) -> List[str]:
    raw_options = [str(option).strip() for option in options or [] if str(option).strip()]
    if not raw_options:
        return []
    joined = " • ".join(raw_options)
    marker = re.compile(r"(?<![A-Za-z0-9])([A-D])\s*(?:[.)]|:)\s*", flags=re.I)
    matches = list(marker.finditer(joined))
    if len(matches) <= len(raw_options):
        return raw_options
    output = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(joined)
        text = joined[match.end() : end].strip(" \t\r\n•;|")
        output.append(f"{match.group(1).upper()}. {text}")
    return output


def _option_text_for_side(item: Dict[str, Any], side: str) -> str:
    mc = item.get("multiple_choice") or {}
    gt_letter = str(mc.get(f"{side}_gt") or "").strip().upper()
    if not gt_letter:
        return ""
    raw_options = split_multiple_choice_options(mc.get("options") or [])
    joined = " • ".join(option for option in raw_options if option)
    marker = re.compile(r"(?<![A-Za-z0-9])([A-D])\s*(?:[.)]|:)\s*", flags=re.I)
    matches = list(marker.finditer(joined))
    for index, match in enumerate(matches):
        if match.group(1).upper() != gt_letter:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(joined)
        return joined[match.end() : end].strip(" \t\r\n•;|")
    for option in raw_options:
        match = re.match(r"^([A-D])(?:\.|\)|\s|-)+(.+)$", option, flags=re.I)
        if match and match.group(1).upper() == gt_letter:
            return match.group(2).strip()
    return ""


def _count_focus_from_direct_qa(item: Dict[str, Any]) -> str:
    question = str((item.get("direct_qa") or {}).get("question") or "").strip()
    match = re.match(
        r"^Does\s+(.+?)\s+have\s+(.+?)\s+and\s+not\s+(.+?)\?$",
        question,
        flags=re.I,
    )
    if not match:
        return ""
    subject = match.group(1).strip()
    positive = match.group(2).strip()
    negative = match.group(3).strip()
    number = r"(?:\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
    positive_noun = re.sub(rf"^{number}\s+", "", positive, flags=re.I).strip()
    negative_noun = re.sub(rf"^{number}\s+", "", negative, flags=re.I).strip()
    noun_phrase = positive_noun
    pluralizes_positive = (
        len(positive_noun.split()) == 1
        and len(negative_noun.split()) == 1
        and negative_noun.lower() == positive_noun.lower() + "s"
    )
    if len(negative_noun.split()) > len(positive_noun.split()) or pluralizes_positive:
        noun_phrase = negative_noun
    if not subject or not noun_phrase or noun_phrase == positive:
        return ""
    return f"How many {noun_phrase} does {subject} have?"


def build_caption_focus_question(item: Dict[str, Any]) -> str:
    captioning = item.get("captioning") or {}
    existing = str(captioning.get("question") or "").strip()
    existing_norm = normalize_text(existing)
    if existing and existing_norm not in {"describe this image", "write a short caption", "write one caption"}:
        return existing

    subcategory = str(item.get("subcategory") or "").strip()
    if str(item.get("category") or "").strip() == "Counting Anomalies":
        direct_focus = _count_focus_from_direct_qa(item)
        if direct_focus:
            return direct_focus

    neutral_by_subcategory = {
        "Animal Behavior": "What visible action or interaction is happening between the animals?",
        "Object Function": "What visible wearing, using, or functional relationship is shown?",
        "Spatial": "What visible spatial relation is shown in the scene?",
        "Size Scale": "What visible size or scale relationship is shown?",
        "Causality": "What visible cause-and-effect relation is shown?",
        "Color": "What visible color does the main object have?",
        "Material": "What visible material or substance state is shown?",
        "Temperature": "What apparent temperature is visibly shown?",
        "Physical State": "What physical state is visibly shown?",
        "Luminescence/Transparency": "What visible opacity, transparency, luminosity, or reflectiveness is shown?",
    }
    if subcategory in neutral_by_subcategory:
        return neutral_by_subcategory[subcategory]

    mc_question = str((item.get("multiple_choice") or {}).get("question") or "").strip()
    if mc_question and "unusual" not in normalize_text(mc_question):
        return mc_question
    if mc_question:
        return mc_question.replace("unusual", "visible").replace("Unusual", "Visible")

    return "What is the CDH-critical visible count, attribute, relation, state, or cause in this image?"


def _first_phrase(captioning: Dict[str, Any], side: str) -> str:
    phrases = split_caption_phrases(captioning.get(f"{side}_gt") or "")
    return phrases[0] if phrases else ""


def _explicit_critical(captioning: Dict[str, Any], side: str) -> str:
    critical = captioning.get("critical_claim")
    if isinstance(critical, dict):
        value = critical.get(side)
        if value is not None:
            return str(value).strip()
    return ""


def _critical_claim_for_side(item: Dict[str, Any], side: str) -> str:
    captioning = item.get("captioning") or {}
    explicit = _explicit_critical(captioning, side)
    option_text = _option_text_for_side(item, side)
    if explicit:
        if option_text and _should_prefer_option_critical(explicit, option_text):
            return option_text
        return explicit
    first = _first_phrase(captioning, side)
    if option_text and _should_prefer_option_critical(first, option_text):
        return option_text
    if first:
        return first
    return option_text


def _number_set(text: str) -> set[str]:
    return {tok for tok in normalize_caption_text(text).split() if tok.isdigit()}


def _first_number(text: str) -> str:
    return next(
        (tok for tok in normalize_caption_text(text).split() if tok.isdigit()),
        "",
    )


def _should_prefer_option_critical(first_phrase: str, option_text: str) -> bool:
    if not first_phrase:
        return bool(option_text)
    first_number = _first_number(first_phrase)
    option_number = _first_number(option_text)
    # Prefer MC when it corrects a numeric caption typo, e.g. "seven fingers" vs "6 fingers".
    if first_number and option_number and first_number != option_number:
        return True
    return False


def _shared_phrase_candidates(captioning: Dict[str, Any]) -> List[str]:
    cs_phrases = split_caption_phrases(captioning.get("commonsense_gt") or "")
    cf_phrases = split_caption_phrases(captioning.get("counterfactual_gt") or "")
    out: List[str] = []

    cs_norm = {normalize_caption_text(x): x for x in cs_phrases}
    for phrase in cf_phrases:
        key = normalize_caption_text(phrase)
        if key and key in cs_norm:
            out.append(phrase)

    # Also keep compact shared tokens when phrases differ only by normal/anomaly modifiers.
    exact_shared_token_sets = [set(normalize_caption_text(phrase).split()) for phrase in out]
    cs_tokens = set()
    cf_tokens = set()
    for phrase in cs_phrases:
        cs_tokens.update(tok for tok in normalize_caption_text(phrase).split() if tok not in _STOP_SHARED_TOKENS)
    for phrase in cf_phrases:
        cf_tokens.update(tok for tok in normalize_caption_text(phrase).split() if tok not in _STOP_SHARED_TOKENS)
    shared_tokens = sorted(cs_tokens & cf_tokens)
    for tok in shared_tokens:
        if len(tok) <= 2:
            continue
        if any(tok in token_set for token_set in exact_shared_token_sets):
            continue
        out.append(tok)
    return _dedupe(out)


def _content_tokens(text: str) -> List[str]:
    return [tok for tok in normalize_caption_text(text).split() if tok not in _STOP_SHARED_TOKENS]


def _explicit_forbidden(captioning: Dict[str, Any], side: str) -> List[str]:
    forbidden = captioning.get("forbidden_claims")
    if isinstance(forbidden, dict):
        return _as_list(forbidden.get(side))
    return []


def build_caption_claim_schema(item: Dict[str, Any]) -> Dict[str, Any]:
    captioning = item.get("captioning") or {}
    critical = {
        "commonsense": _critical_claim_for_side(item, "commonsense"),
        "counterfactual": _critical_claim_for_side(item, "counterfactual"),
    }
    prior_claim = str(captioning.get("prior_claim") or critical["commonsense"] or _first_phrase(captioning, "commonsense")).strip()

    explicit_shared = _as_list(captioning.get("shared_claims"))
    shared_claims = explicit_shared or _shared_phrase_candidates(captioning)

    cs_phrases = split_caption_phrases(captioning.get("commonsense_gt") or "")
    cf_phrases = split_caption_phrases(captioning.get("counterfactual_gt") or "")
    shared_norm = {normalize_caption_text(x) for x in shared_claims}
    shared_tokens = set()
    for claim in shared_claims:
        shared_tokens.update(_content_tokens(claim))

    def side_forbidden(side: str) -> List[str]:
        explicit = _explicit_forbidden(captioning, side)
        if explicit:
            return _dedupe(explicit)
        other_side = "counterfactual" if side == "commonsense" else "commonsense"
        source_phrases = cf_phrases if side == "commonsense" else cs_phrases
        values = [critical.get(other_side, "")]
        for phrase in source_phrases:
            if normalize_caption_text(phrase) in shared_norm:
                continue
            tokens = set(_content_tokens(phrase))
            if tokens and tokens.issubset(shared_tokens):
                continue
            values.append(phrase)
        if side == "counterfactual" and prior_claim:
            values.insert(0, prior_claim)
        return _dedupe(values)

    return {
        "schema_version": str(captioning.get("claim_schema_version") or _CLAIM_SCHEMA_VERSION),
        "critical_claim": critical,
        "prior_claim": prior_claim,
        "shared_claims": _dedupe(shared_claims),
        "forbidden_claims": {
            "commonsense": side_forbidden("commonsense"),
            "counterfactual": side_forbidden("counterfactual"),
        },
        "benchmark_views": {
            "cdh_gen": ["critical_claim", "prior_claim", "shared_claims"],
            "chair_like": ["forbidden_claims"],
            "pope_like": ["prior_claim"],
            "mme_like": ["critical_claim"],
            "mmvet_like": ["claim_coverage"],
        },
        "claim_source": "explicit_or_bootstrapped_from_mc_and_caption_gt",
    }
