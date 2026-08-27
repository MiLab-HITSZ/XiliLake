# Copyright (c) 2026 MiLab. All rights reserved.
import http.client
import base64
import hashlib
import json
import math
import os
import time
import ssl
import argparse
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm import tqdm

from cdh_caption_claims import build_caption_claim_schema, split_multiple_choice_options
from cdh_bench_loader import CDHBenchLoader

BASE_DIR = Path(__file__).resolve().parent
MITIGATION_NONE = "none"
MITIGATION_PROMPT_GROUNDING = "prompt_grounding"
MITIGATION_VISUAL_EVIDENCE = "visual_evidence"
MITIGATION_OPTION_ENTAILMENT = "option_entailment"
MITIGATION_CP_VBC = "cp_vbc"
MITIGATION_VCD = "vcd"
MITIGATION_PAI = "pai"
MITIGATION_MFCD = "mfcd"
MITIGATION_REVIS = "revis"
CP_VBC_MODE_CANDIDATE = "candidate"
CP_VBC_MODE_TOKEN = "token"
CP_VBC_LAMBDA_POLICY_FIXED = "fixed"
CP_VBC_LAMBDA_POLICY_BAYES_PATH = "bayes_path"
SUPPORTED_MITIGATIONS = (
    MITIGATION_NONE,
    MITIGATION_PROMPT_GROUNDING,
    MITIGATION_VISUAL_EVIDENCE,
    MITIGATION_OPTION_ENTAILMENT,
    MITIGATION_CP_VBC,
    MITIGATION_VCD,
    MITIGATION_PAI,
    MITIGATION_MFCD,
    MITIGATION_REVIS,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_subcategory(subcategory: str) -> str:
    return (subcategory or "").replace(" ", "_").replace("/", "_")


def _normalize_pair_id(pair_id: str) -> str:
    return (pair_id or "").replace(" ", "_")


def _image_path(images_root: str, subcategory: str, pair_id: str, side: str) -> str:
    sub_dir = _normalize_subcategory(subcategory)
    p_dir = _normalize_pair_id(pair_id)
    filename = "counterfactual.png" if side == "counterfactual" else "commonsense.png"
    return str(Path(images_root) / sub_dir / p_dir / filename)


def _evaluation_task_shard(
    item_index: int,
    task_index: int,
    side_index: int,
    task_count: int,
    shard_count: int,
) -> int:
    position = (int(item_index) * int(task_count) + int(task_index)) * 2 + int(
        side_index
    )
    return position % int(shard_count)


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _append_jsonl(path: str, record: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_slug(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return "model"
    keep = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)[:120]


def _hash_dict(d: Dict[str, Any]) -> str:
    raw = json.dumps(d, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def _strip_thinking(text: str) -> str:
    """如果文本包含 <think>...</think>，则返回 </think> 之后的内容。"""
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text.strip()


def _extract_first_letter(text: str) -> Optional[str]:
    text = _strip_thinking(text)
    t = text.strip().upper()
    patterns = [
        r'^([A-D])(?:\.|\)|$|\s)',
        r'(?:FINAL\s+ANSWER|FINAL|最终答案|答案)(?:\s*(?:IS|是|为|:|-))?\s*([A-D])',
        r'(?:ANSWER|答案)(?:IS|是|为)?\s*([A-D])',
        r'\s([A-D])(?:\.|\)|$|\s)',
    ]
    for p in patterns:
        m = re.search(p, t)
        if m:
            return m.group(1)
    
    m = re.search(r'\b([A-D])\b', t)
    if m:
        return m.group(1)
        
    return None


def _extract_first_int(text: str) -> Optional[int]:
    text = _strip_thinking(text)
    m = re.search(r'\d+', text)
    if m:
        try:
            return int(m.group())
        except:
            pass
    return None


def _extract_yes_no(text: str) -> Optional[str]:
    text = _strip_thinking(text)
    t = _normalize_text(text)
    if not t:
        return None
    
    words = t.split()
    if words:
        first = words[0]
        if first in ("yes", "y", "true", "是", "对"):
            return "yes"
        if first in ("no", "n", "false", "否", "不", "不是", "错"):
            return "no"
    
    if "yes" in words or "是" in t or "对" in t:
        return "yes"
    if "no" in words or "否" in t or "不" in t:
        return "no"
        
    return None


def _normalize_text(text: str) -> str:
    text = _strip_thinking(text)
    t = (text or "").strip().lower()
    t = "".join(ch for ch in t if ch.isalnum() or ch.isspace())
    t = " ".join(t.split())
    return t


_NUMBER_WORDS = {
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


def _normalize_caption_text(text: str) -> str:
    words = _normalize_text(text).split()
    return " ".join(_NUMBER_WORDS.get(word, word) for word in words)


def _caption_phrases(gt: str) -> List[str]:
    return [part.strip() for part in str(gt or "").split(",") if part.strip()]


def _caption_phrase_match(pred_norm: str, phrase: str) -> bool:
    phrase_norm = _normalize_caption_text(phrase)
    if not phrase_norm:
        return False
    if phrase_norm in pred_norm:
        return True
    if _caption_numbered_head_match(pred_norm, phrase_norm):
        return True
    tokens = [tok for tok in phrase_norm.split() if tok not in {"a", "an", "the", "normal"}]
    return bool(tokens) and all(tok in pred_norm.split() for tok in tokens)


_COUNT_HEAD_EQUIVALENTS = {
    "finger": {"finger", "fingers", "digit", "digits"},
    "fingers": {"finger", "fingers", "digit", "digits"},
    "digit": {"finger", "fingers", "digit", "digits"},
    "digits": {"finger", "fingers", "digit", "digits"},
}


def _caption_head_variants(token: str) -> set[str]:
    token = str(token or "").strip()
    if not token:
        return set()
    variants = set(_COUNT_HEAD_EQUIVALENTS.get(token, {token}))
    if token.endswith("s") and len(token) > 1:
        variants.add(token[:-1])
    else:
        variants.add(token + "s")
    return variants


def _caption_numbered_head_match(pred_norm: str, claim_norm: str) -> bool:
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
    head_variants = _caption_head_variants(head)
    if not head_variants:
        return False

    for idx, tok in enumerate(pred_tokens):
        if tok != number:
            continue
        window = pred_tokens[max(0, idx - 2) : min(len(pred_tokens), idx + 8)]
        if any(w in head_variants for w in window):
            return True
    return False


def _caption_forbidden_match(pred_norm: str, claim: str) -> bool:
    claim_norm = _normalize_caption_text(claim)
    if not claim_norm:
        return False
    return claim_norm in pred_norm or _caption_numbered_head_match(pred_norm, claim_norm)


def _caption_claim_rows(pred_norm: str, role: str, claims: List[str], strict: bool = False) -> List[Dict[str, Any]]:
    rows = []
    for claim in claims:
        text = str(claim or "").strip()
        if not text:
            continue
        match = _caption_forbidden_match(pred_norm, text) if strict else _caption_phrase_match(pred_norm, text)
        rows.append({"role": role, "claim": text, "matched": bool(match)})
    return rows


def _score_caption_details(
    pred: str,
    gt: str,
    min_coverage: float = 0.5,
    item: Optional[Dict[str, Any]] = None,
    side: str = "",
) -> Dict[str, Any]:
    pred_norm = _normalize_caption_text(pred)
    if item is not None and side in ("commonsense", "counterfactual"):
        schema = build_caption_claim_schema(item)
        critical_map = schema.get("critical_claim") or {}
        critical_claim = str(critical_map.get(side) or "").strip()
        prior_claim = str(schema.get("prior_claim") or "").strip()
        shared_claims = [str(x).strip() for x in schema.get("shared_claims") or [] if str(x).strip()]
        forbidden_map = schema.get("forbidden_claims") or {}
        forbidden_claims = [str(x).strip() for x in forbidden_map.get(side) or [] if str(x).strip()]

        critical_rows = _caption_claim_rows(pred_norm, "critical", [critical_claim])
        shared_rows = _caption_claim_rows(pred_norm, "shared", shared_claims)
        forbidden_rows = _caption_claim_rows(pred_norm, "forbidden", forbidden_claims, strict=True)
        prior_rows = _caption_claim_rows(pred_norm, "prior", [prior_claim], strict=True)

        required_rows = critical_rows + shared_rows
        required_matches = sum(1 for row in required_rows if row["matched"])
        shared_matches = sum(1 for row in shared_rows if row["matched"])
        critical_match = bool(critical_rows and critical_rows[0]["matched"])
        forbidden_matches = [row["claim"] for row in forbidden_rows if row["matched"]]
        prior_match = bool(prior_rows and prior_rows[0]["matched"])
        prior_attraction = bool(side == "counterfactual" and prior_match)
        coverage = (required_matches / len(required_rows)) if required_rows else 0.0
        shared_coverage = (shared_matches / len(shared_rows)) if shared_rows else None
        correct = bool(critical_match and not forbidden_matches)
        return {
            "schema_version": schema.get("schema_version"),
            "claim_source": schema.get("claim_source"),
            "critical_claim": critical_claim,
            "prior_claim": prior_claim,
            "shared_claims": shared_claims,
            "forbidden_claims": forbidden_claims,
            "claims": required_rows + forbidden_rows + prior_rows,
            "matched_claims": [row["claim"] for row in required_rows if row["matched"]],
            "forbidden_matches": forbidden_matches,
            "coverage": coverage,
            "claim_coverage": coverage,
            "shared_coverage": shared_coverage,
            "critical_match": critical_match,
            "critical_claim_match": critical_match,
            "prior_match": prior_match,
            "prior_attraction": prior_attraction,
            "forbidden_match": bool(forbidden_matches),
            "correct": correct,
            "min_coverage": None,
            "evaluator": "cdh_gen_claims_v1",
            "benchmark_views": schema.get("benchmark_views"),
        }

    phrases = _caption_phrases(gt)
    matched = [_caption_phrase_match(pred_norm, phrase) for phrase in phrases]
    coverage = (sum(1 for x in matched if x) / len(phrases)) if phrases else 0.0
    critical_match = bool(matched[0]) if matched else False
    correct = bool(critical_match and coverage >= float(min_coverage))
    return {
        "phrases": phrases,
        "matched": matched,
        "coverage": coverage,
        "critical_match": critical_match,
        "correct": correct,
        "min_coverage": float(min_coverage),
    }


def _score_caption(pred: str, gt: str) -> bool:
    return bool(_score_caption_details(pred, gt).get("correct"))


def _score_direct_qa(pred: str, gt: str) -> bool:
    gt = (gt or "").strip()
    if gt == "":
        return False
    gt_norm = _normalize_text(gt)
    pred_norm = _normalize_text(pred)
    if gt_norm in ("yes", "no"):
        p = _extract_yes_no(pred)
        return p == gt_norm
    if gt.isdigit():
        p = _extract_first_int(pred)
        return p is not None and str(p) == gt
    if gt_norm == pred_norm:
        return True
    return gt_norm in pred_norm


def _score_multiple_choice(pred: str, gt_letter: str) -> bool:
    gt_letter = (gt_letter or "").strip().upper()
    if gt_letter not in ("A", "B", "C", "D"):
        return False
    p = _extract_first_letter(pred)
    return p == gt_letter


def _task_fields(task: str) -> Tuple[str, str]:
    if task == "qa":
        return "direct_qa", "question"
    if task == "mc":
        return "multiple_choice", "question"
    if task == "caption":
        return "captioning", "question"
    raise ValueError(f"unknown task: {task}")


def _get_gt(item: Dict[str, Any], task: str, side: str) -> str:
    if task == "qa":
        return (item.get("direct_qa") or {}).get(f"{side}_gt") or ""
    if task == "mc":
        return (item.get("multiple_choice") or {}).get(f"{side}_gt") or ""
    if task == "caption":
        return (item.get("captioning") or {}).get(f"{side}_gt") or ""
    raise ValueError(f"unknown task: {task}")


def _get_question(item: Dict[str, Any], task: str) -> str:
    if task == "qa":
        return (item.get("direct_qa") or {}).get("question") or ""
    if task == "mc":
        return (item.get("multiple_choice") or {}).get("question") or ""
    if task == "caption":
        return (item.get("captioning") or {}).get("question") or "Describe this image."
    raise ValueError(f"unknown task: {task}")


def _get_options(item: Dict[str, Any]) -> List[str]:
    opts = (item.get("multiple_choice") or {}).get("options") or []
    if isinstance(opts, list):
        return split_multiple_choice_options(opts)
    return []


def _build_user_text(task: str, item: Dict[str, Any]) -> str:
    q = _get_question(item, task)
    if task == "mc":
        opts = _get_options(item)
        opts_text = "\n".join(str(o) for o in opts)
        return f"{q}\n{opts_text}\nAnswer with a single letter (A, B, C, or D)."
    if task == "qa":
        return f"{q}\nAnswer with yes or no."
    if task == "caption":
        return (
            f"{q}\n"
            "Write one concise visual caption. Explicitly state the task-focused visible count, attribute, material, state, relation, cause, or scale even if it seems ordinary. "
            "Use only what is visible in this single image; do not assume typical details."
        )
    raise ValueError(f"unknown task: {task}")


def _candidate_answers(task: str, item: Dict[str, Any]) -> List[Dict[str, str]]:
    if task == "qa":
        return [{"key": "yes", "text": " yes"}, {"key": "no", "text": " no"}]
    if task == "mc":
        candidates: List[Dict[str, str]] = []
        for opt in _get_options(item):
            m = re.match(r"\s*([A-D])(?:\.|\)|\s|$)", str(opt).strip(), flags=re.I)
            if not m:
                continue
            letter = m.group(1).upper()
            candidates.append({"key": letter, "text": f" {letter}"})
        if not candidates:
            candidates = [{"key": x, "text": f" {x}"} for x in ("A", "B", "C", "D")]
        return candidates
    raise ValueError(f"unknown task: {task}")


def _candidate_key_from_prediction(task: str, pred: str) -> Optional[str]:
    if task == "qa":
        return _extract_yes_no(pred)
    if task == "mc":
        return _extract_first_letter(pred)
    raise ValueError(f"unknown task: {task}")


def _candidate_key_from_model_prediction(
    spec: "ModelSpec", task: str, pred: str
) -> Optional[str]:
    model_identity = f"{spec.name} {spec.model}".lower()
    if "thinking" in model_identity and "</think>" not in str(pred):
        return None
    return _candidate_key_from_prediction(task, pred)


def _prediction_from_candidate_key(task: str, key: str) -> str:
    if task == "qa":
        return f"Final answer: {str(key).lower()}"
    if task == "mc":
        return f"Final answer: {str(key).upper()}"
    raise ValueError(f"unknown task: {task}")


def _build_candidate_scoring_prompt(task: str, item: Dict[str, Any]) -> str:
    q = _get_question(item, task)
    if task == "qa":
        return f"{q}\nAnswer with yes or no.\nFinal answer:"
    if task == "mc":
        opts = "\n".join(_get_options(item))
        return f"{q}\n{opts}\nAnswer with a single letter (A, B, C, or D).\nFinal answer:"
    raise ValueError(f"unknown task: {task}")


def _build_token_cp_vbc_prompt(task: str, item: Dict[str, Any]) -> str:
    if task == "mc":
        return _build_candidate_scoring_prompt(task, item)
    if task == "qa":
        return _build_candidate_scoring_prompt(task, item)
    if task == "caption":
        return _build_user_text(task, item)
    raise ValueError(f"unknown task: {task}")


def _build_visual_evidence_text(task: str, item: Dict[str, Any]) -> str:
    q = _get_question(item, task)
    lines = [
        "Inspect the image before answering the task.",
        "Use only visible evidence in this single image; do not add typical-but-unseen details.",
        "Do not classify the image; just describe task-relevant visible facts.",
        f"Question: {q}",
    ]
    if task == "mc":
        opts = _get_options(item)
        if opts:
            lines.append("Options to visually inspect one by one; do not choose yet:")
            lines.extend(str(o) for o in opts)
        lines.append("Focus on the main subject and property asked in the question, not incidental parts.")
    else:
        lines.append("If the question contains a negated clause such as 'not ...', inspect that clause as a separate visible claim.")
    lines.extend(
        [
            "Write 2-4 short bullets naming the concrete visual evidence needed to answer.",
            "Mention visible counts, attributes, relations, or materials only when they are relevant to the question.",
            "Do not provide the final yes/no or option letter in this step.",
        ]
    )
    return "\n".join(lines)


def _build_prompt_grounding_text(task: str, item: Dict[str, Any]) -> str:
    base = _build_user_text(task, item)
    if task == "qa":
        final_rule = "Output exactly one line: Final answer: yes or Final answer: no."
    elif task == "mc":
        final_rule = "Output exactly one line: Final answer: A/B/C/D."
    elif task == "caption":
        final_rule = "Output only one concise visual caption, with no explanation."
    else:
        raise ValueError(f"unknown task: {task}")
    return (
        "Ground your answer in this single image only.\n"
        "Before deciding, silently verify the task-focused visible count, identity, attribute, material, state, relation, cause, or scale.\n"
        "Treat normal commonsense defaults as hypotheses, not facts. If the image conflicts with what is typical, follow the image.\n"
        "Do not use paired examples, dataset labels, reference images, ground-truth claims, or any information outside the task prompt and image.\n\n"
        f"{base}\n{final_rule}"
    )


def _build_visual_evidence_answer_text(task: str, item: Dict[str, Any], evidence: str) -> str:
    base = _build_user_text(task, item)
    if task == "qa":
        answer_instruction = (
            "Output exactly one line: Final answer: yes or Final answer: no. "
            "Answer yes only when the full question is supported by the visual evidence, including any negated clause. "
            "Answer no when a required visible claim is contradicted or the negated clause is visible."
        )
    elif task == "mc":
        answer_instruction = (
            "Output exactly one line with one option letter only, for example: Final answer: B. "
            "Choose the option best supported by the visual evidence about the main subject and asked property."
        )
    elif task == "caption":
        answer_instruction = (
            "Output only one concise visual caption. "
            "State the task-focused visible count, attribute, material, state, relation, cause, or scale. "
            "Do not output option letters or explain the evidence."
        )
    else:
        raise ValueError(f"unknown task: {task}")
    return (
        "Use the visual evidence below as the anchor for the answer. "
        "Trust the visible image over common-sense expectations if they conflict.\n\n"
        f"Visual evidence:\n{(evidence or '').strip()}\n\n"
        f"{base}\n{answer_instruction}"
    )


def _build_option_entailment_text(task: str, item: Dict[str, Any]) -> str:
    q = _get_question(item, task)
    lines = [
        "Check the task as visual entailment against this single image.",
        "Use neutral labels only: supported, contradicted, or not visible.",
        "Do not use paired examples, dataset labels, or unstated assumptions.",
        f"Question: {q}",
    ]
    if task == "mc":
        opts = _get_options(item)
        if opts:
            lines.append("Options:")
            lines.extend(str(o) for o in opts)
        lines.extend(
            [
                "For each option, write one compact line:",
                "Option <letter>: supported/contradicted/not visible - brief visual reason.",
                "Do not provide the final option letter in this step.",
            ]
        )
    elif task == "qa":
        lines.extend(
            [
                "Break the question into the smallest visible claims, including any negated claim.",
                "For each claim, write one compact line:",
                "Claim <n>: supported/contradicted/not visible - brief visual reason.",
                "Do not provide the final yes/no answer in this step.",
            ]
        )
    else:
        raise ValueError(f"unknown task: {task}")
    return "\n".join(lines)


def _build_option_entailment_answer_text(task: str, item: Dict[str, Any], entailment: str) -> str:
    base = _build_user_text(task, item)
    if task == "mc":
        rule = (
            "Use the option checks below. Choose the option with the strongest visual support and no direct contradiction. "
            "Do not prefer an option just because it is typical. Output exactly one line: Final answer: A/B/C/D."
        )
    elif task == "qa":
        rule = (
            "Use the claim checks below. Answer yes only if the whole question is visually supported, including any negated clause. "
            "Answer no if any required part is contradicted, or if a negated claim is visible. "
            "Output exactly one line: Final answer: yes or Final answer: no."
        )
    else:
        raise ValueError(f"unknown task: {task}")
    return (
        "Make the final answer from the single-image visual entailment checks below.\n\n"
        f"Visual entailment checks:\n{(entailment or '').strip()}\n\n"
        f"{base}\n{rule}"
    )


def _b64_data_url(path: str) -> str:
    from PIL import Image
    import io
    img = Image.open(path).convert("RGB")
    img = img.resize((512, 512), Image.Resampling.LANCZOS)
    
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    backend: str
    model: str
    base_url: str
    api_key: Optional[str]
    temperature: float
    max_tokens: int
    models_root: str
    attn_implementation: str = ""
    quantization: str = "auto"
    dtype: str = "auto"


def _parse_models_arg(models_arg: str) -> List[Dict[str, Any]]:
    p = Path(models_arg)
    if p.exists() and p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    return json.loads(models_arg)


def _load_model_specs(models_arg: str) -> List[ModelSpec]:
    raw = _parse_models_arg(models_arg)
    if not isinstance(raw, list):
        raise ValueError("--models must be a JSON list or a path to a JSON file containing a list")
    specs: List[ModelSpec] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or entry.get("model") or "model")
        backend = str(entry.get("backend") or "").lower().strip()
        if backend in ("hf", "transformer", "local_transformers"):
            backend = "transformers"
        model = str(entry.get("model") or "")
        base_url = str(entry.get("base_url") or "").rstrip("/")
        if backend == "vllm" and not base_url:
            base_url = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
        api_key = entry.get("api_key")
        api_key_env = entry.get("api_key_env")
        if api_key is None and api_key_env:
            api_key = os.environ.get(str(api_key_env))
        temperature = float(entry.get("temperature", 0.0))
        max_tokens = int(entry.get("max_tokens", 4096))
        thinking_min_tokens = int(os.environ.get("CDH_THINKING_MIN_MAX_TOKENS", "4096"))
        if "thinking" in f"{name} {model}".lower() and max_tokens < thinking_min_tokens:
            max_tokens = thinking_min_tokens
        models_root = str(
            entry.get("models_root")
            or os.environ.get("XILILAKE_MODELS_ROOT")
            or os.environ.get("CDH_MODELS_ROOT")
            or (BASE_DIR / "models")
        )
        attn_implementation = str(entry.get("attn_implementation") or entry.get("attention_implementation") or "").strip()
        quantization = str(entry.get("quantization") or "auto").strip().lower()
        dtype = str(entry.get("dtype") or "auto").strip().lower()
        if quantization not in {"auto", "none", "4bit"}:
            raise ValueError(f"unknown quantization for model {name}: {quantization}")
        if dtype not in {"auto", "float16", "bfloat16", "float32"}:
            raise ValueError(f"unknown dtype for model {name}: {dtype}")
        if backend not in ("api", "vllm", "transformers"):
            raise ValueError(f"unknown backend for model {name}: {backend}")
        if backend in ("api", "vllm") and not base_url:
            raise ValueError(f"missing base_url for model {name}")
        if not model:
            raise ValueError(f"missing model for model {name}")
        specs.append(
            ModelSpec(
                name=name,
                backend=backend,
                model=model,
                base_url=base_url,
                api_key=str(api_key) if api_key is not None else None,
                temperature=temperature,
                max_tokens=max_tokens,
                models_root=models_root,
                attn_implementation=attn_implementation,
                quantization=quantization,
                dtype=dtype,
            )
        )
    return specs


def _parse_csv_list(raw: str) -> List[str]:
    if not raw:
        return []
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def _load_pair_manifest(path_value: str, split: str = "") -> List[str]:
    if not str(path_value or "").strip():
        return []
    path = Path(path_value)
    raw = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return [line.strip() for line in raw.splitlines() if line.strip()]

    if isinstance(payload, list):
        return [str(value).strip() for value in payload if str(value).strip()]
    if not isinstance(payload, dict):
        raise ValueError("pair manifest must be a JSON list, object, or text file")
    if split:
        manifests = payload.get("split_manifests")
        if not isinstance(manifests, dict) or split not in manifests:
            raise ValueError(f"pair manifest has no split {split!r}")
        payload = manifests[split]
    values = payload.get("pair_ids")
    if not isinstance(values, list):
        raise ValueError("pair manifest object must contain a pair_ids list")
    return [str(value).strip() for value in values if str(value).strip()]


def _write_progress(progress_file: str, payload: Dict[str, Any]) -> None:
    if not progress_file:
        return
    _write_json(progress_file, payload)


def _http_post_json(url_str: str, payload: Dict[str, Any], api_key: Optional[str], timeout_s: int) -> Dict[str, Any]:
    from urllib.parse import urlparse
    parsed = urlparse(url_str)
    host = parsed.hostname
    port = parsed.port
    path = parsed.path
    if parsed.query:
        path += "?" + parsed.query
    
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json'
    }
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    
    payload_json = json.dumps(payload).encode("utf-8")
    
    if parsed.scheme == "https":
        conn = http.client.HTTPSConnection(host, port if port else 443, timeout=timeout_s, context=ssl._create_unverified_context())
    else:
        conn = http.client.HTTPConnection(host, port if port else 80, timeout=timeout_s)
    
    try:
        conn.request("POST", path, body=payload_json, headers=headers)
        res = conn.getresponse()
        data = res.read().decode("utf-8")
        if res.status >= 400:
            raise RuntimeError(f"HTTP {res.status}: {data}")
        return json.loads(data)
    finally:
        conn.close()


def _call_openai_compat_chat(
    spec: ModelSpec,
    user_text: str,
    image_path: str,
    timeout_s: int,
    top_logprobs: int = 0,
) -> Tuple[str, Dict[str, Any]]:
    base_url = spec.base_url.rstrip("/")
    if base_url.endswith("/v1/chat/completions"):
        url = base_url
    elif base_url.endswith("/v1"):
        url = f"{base_url}/chat/completions"
    else:
        url = f"{base_url}/v1/chat/completions"
    
    content: List[Dict[str, Any]] = [
        {"type": "text", "text": user_text},
        {"type": "image_url", "image_url": {"url": _b64_data_url(image_path)}},
    ]

    payload = {
        "model": spec.model,
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
        "temperature": spec.temperature,
        "max_tokens": spec.max_tokens,
        "stream": False
    }
    if top_logprobs and top_logprobs > 0:
        payload["logprobs"] = True
        payload["top_logprobs"] = min(int(top_logprobs), 20)
    raw = _http_post_json(url, payload, spec.api_key, timeout_s=timeout_s)
    text = ""
    try:
        text = raw["choices"][0]["message"]["content"]
    except Exception:
        text = json.dumps(raw, ensure_ascii=False)
    return str(text), raw


_LOCAL_QWEN3_CACHE: Dict[str, Any] = {}
_JSPACE_LENS_CACHE: Dict[str, Dict[str, Any]] = {}
_OFFICIAL_REVIS_CACHE: Dict[str, Any] = {}
_OFFICIAL_REVIS_ASSET_CACHE: Dict[str, Any] = {}
_LOCAL_GENERIC_VLM_CACHE: Dict[str, Any] = {}


def _local_qwen3_bundle(spec: ModelSpec) -> Tuple[Any, Any]:
    try:
        from transformers import (
            AutoConfig,
            AutoProcessor,
            Qwen3VLForConditionalGeneration,
            Qwen3VLMoeForConditionalGeneration,
        )
    except Exception as e:
        raise RuntimeError(
            "local transformers backend requires torch+Pillow+transformers in the current Python environment"
        ) from e

    cache_key = (
        f"{spec.models_root}::{spec.model}::attn={spec.attn_implementation}"
        f"::quant={spec.quantization}::dtype={spec.dtype}"
    )
    bundle = _LOCAL_QWEN3_CACHE.get(cache_key)
    if bundle is not None:
        return bundle

    model_path = Path(spec.models_root) / spec.model
    load_id = str(model_path) if model_path.exists() else spec.model
    model_type = str(AutoConfig.from_pretrained(load_id).model_type)
    model_class = {
        "qwen3_vl": Qwen3VLForConditionalGeneration,
        "qwen3_vl_moe": Qwen3VLMoeForConditionalGeneration,
    }.get(model_type)
    if model_class is None:
        raise RuntimeError(
            f"Unsupported Qwen3-VL model_type={model_type!r} for {load_id}."
        )

    quant_config = None
    use_4bit = spec.quantization == "4bit" or (
        spec.quantization == "auto" and ("32B" in spec.model or "235B" in spec.model)
    )
    if use_4bit:
        try:
            import importlib.util
            import torch
            from transformers import BitsAndBytesConfig

            if importlib.util.find_spec("bitsandbytes") is not None:
                compute_dtype = torch.float16
                if spec.dtype == "bfloat16":
                    compute_dtype = torch.bfloat16
                elif spec.dtype == "float32":
                    compute_dtype = torch.float32
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
        except Exception:
            quant_config = None

    model_dtype: Any = "auto"
    if spec.dtype != "auto":
        import torch

        model_dtype = getattr(torch, spec.dtype)
    load_kwargs = {
        "dtype": model_dtype,
        "device_map": "auto",
        "quantization_config": quant_config,
    }
    if spec.attn_implementation:
        load_kwargs["attn_implementation"] = spec.attn_implementation
    model = model_class.from_pretrained(load_id, **load_kwargs)
    processor = AutoProcessor.from_pretrained(load_id)
    bundle = (model, processor)
    _LOCAL_QWEN3_CACHE[cache_key] = bundle
    return bundle


def _qwen3_vl_inputs(
    model: Any,
    processor: Any,
    user_text: str,
    image: Optional[Any],
) -> Dict[str, Any]:
    content = []
    if image is not None:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": user_text})
    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}


def _local_candidate_bundle(spec: ModelSpec) -> Tuple[Any, Any, str]:
    if "llava" not in f"{spec.name} {spec.model}".lower():
        model, processor = _local_qwen3_bundle(spec)
        return model, processor, "qwen3_vl"
    model_path = Path(spec.models_root) / spec.model
    load_id = str(model_path) if model_path.exists() else spec.model
    cache_key = str(Path(load_id).resolve()) if Path(load_id).exists() else load_id
    bundle = _LOCAL_GENERIC_VLM_CACHE.get(cache_key)
    if bundle is None:
        from official_revis_adapter import load_model_bundle

        bundle = load_model_bundle(load_id)
        _LOCAL_GENERIC_VLM_CACHE[cache_key] = bundle
    return bundle


def _candidate_vlm_inputs(
    model: Any,
    processor: Any,
    family: str,
    user_text: str,
    image: Optional[Any],
) -> Dict[str, Any]:
    if family == "qwen3_vl":
        return _qwen3_vl_inputs(model, processor, user_text, image)
    from official_revis_adapter import build_inputs

    return build_inputs(model, processor, family, user_text, image)


def _probe_layer_indices(num_layers: int) -> List[int]:
    if num_layers <= 0:
        return []
    raw = {0, num_layers // 4, num_layers // 2, (3 * num_layers) // 4, num_layers - 1}
    return sorted(idx for idx in raw if 0 <= idx < num_layers)


def _orthogonal_visual_residual_from_outputs(
    outputs_image: Any,
    outputs_prior: Any,
    state_index: Optional[int] = None,
) -> Tuple[Dict[str, Any], Any]:
    import torch
    import torch.nn.functional as F

    states_image = getattr(outputs_image, "hidden_states", None) or []
    states_prior = getattr(outputs_prior, "hidden_states", None) or []
    if not states_image or not states_prior:
        return {"available": False, "reason": "missing_hidden_states"}, None
    num_states = min(len(states_image), len(states_prior))
    if state_index is None:
        resolved_state_index = num_states - 1
    else:
        resolved_state_index = int(state_index)
        if resolved_state_index < 0:
            resolved_state_index = num_states + resolved_state_index
        resolved_state_index = max(0, min(resolved_state_index, num_states - 1))
    h_image = states_image[resolved_state_index][0, -1, :].float()
    h_prior = states_prior[resolved_state_index][0, -1, :].to(device=h_image.device).float()
    visual_delta = h_image - h_prior
    prior_norm_sq = torch.dot(h_prior, h_prior).clamp_min(1e-12)
    projection = torch.dot(visual_delta, h_prior) / prior_norm_sq * h_prior
    pure_visual = visual_delta - projection

    image_norm = float(torch.linalg.norm(h_image).detach().cpu().item())
    prior_norm = float(torch.linalg.norm(h_prior).detach().cpu().item())
    delta_norm = float(torch.linalg.norm(visual_delta).detach().cpu().item())
    projection_norm = float(torch.linalg.norm(projection).detach().cpu().item())
    pure_norm = float(torch.linalg.norm(pure_visual).detach().cpu().item())
    summary = {
        "available": True,
        "state_index": int(resolved_state_index),
        "decoder_layer_index": int(resolved_state_index - 1) if resolved_state_index > 0 else None,
        "layer": "last" if resolved_state_index == num_states - 1 else int(resolved_state_index - 1),
        "num_states": int(num_states),
        "image_norm": image_norm,
        "prior_norm": prior_norm,
        "visual_delta_norm": delta_norm,
        "prior_projection_norm": projection_norm,
        "pure_visual_norm": pure_norm,
        "pure_visual_ratio": float(pure_norm / delta_norm) if delta_norm else None,
        "relative_pure_visual": float(pure_norm / image_norm) if image_norm else None,
        "image_prior_cosine": float(F.cosine_similarity(h_image, h_prior, dim=0).detach().cpu().item()),
        "delta_prior_cosine": float(F.cosine_similarity(visual_delta, h_prior, dim=0).detach().cpu().item())
        if delta_norm and prior_norm
        else None,
    }
    return summary, pure_visual


def _salient_visual_patch_residual_from_outputs(
    model: Any,
    inputs_image: Dict[str, Any],
    outputs_image: Any,
    outputs_prior: Any,
    state_index: int,
) -> Tuple[Dict[str, Any], Any]:
    """Pool robust image-vs-degraded residuals from informative vision tokens."""
    import torch
    import torch.nn.functional as F

    states_image = getattr(outputs_image, "hidden_states", None) or []
    states_prior = getattr(outputs_prior, "hidden_states", None) or []
    if not states_image or not states_prior:
        return {"available": False, "reason": "missing_hidden_states"}, None
    resolved_state_index = max(0, min(int(state_index), min(len(states_image), len(states_prior)) - 1))
    image_states = states_image[resolved_state_index]
    prior_states = states_prior[resolved_state_index]
    if image_states.shape[1] != prior_states.shape[1]:
        return {"available": False, "reason": "unaligned_image_and_prior_sequences"}, None

    input_ids = inputs_image.get("input_ids")
    image_token_id = getattr(getattr(model, "config", None), "image_token_id", None)
    if input_ids is None or image_token_id is None:
        return {"available": False, "reason": "missing_image_token_mask"}, None
    mask = input_ids[0].to(device=image_states.device) == int(image_token_id)
    if int(mask.sum().detach().cpu().item()) < 4:
        return {"available": False, "reason": "too_few_image_tokens"}, None

    h_image_patches = image_states[0, mask, :].float()
    h_prior_patches = prior_states[0, mask.to(device=prior_states.device), :].to(
        device=h_image_patches.device
    ).float()
    patch_delta = h_image_patches - h_prior_patches
    salience = torch.linalg.norm(patch_delta, dim=-1)
    q50 = torch.quantile(salience, 0.50)
    q95 = torch.quantile(salience, 0.95)
    robust_salience = torch.clamp(salience, max=q95)
    weights = torch.clamp(robust_salience - q50, min=0.0)
    if float(weights.sum().detach().cpu().item()) <= 1e-12:
        weights = torch.ones_like(weights)
    weights = weights / weights.sum().clamp_min(1e-12)
    visual_delta = torch.sum(weights[:, None] * patch_delta, dim=0)

    h_image_last = image_states[0, -1, :].float()
    h_prior_last = prior_states[0, -1, :].to(device=h_image_last.device).float()
    prior_norm_sq = torch.dot(h_prior_last, h_prior_last).clamp_min(1e-12)
    projection = torch.dot(visual_delta, h_prior_last) / prior_norm_sq * h_prior_last
    pure_visual = visual_delta - projection
    pure_norm = float(torch.linalg.norm(pure_visual).detach().cpu().item())
    image_norm = float(torch.linalg.norm(h_image_last).detach().cpu().item())
    return {
        "available": True,
        "type": "salient_visual_patch_residual",
        "state_index": int(resolved_state_index),
        "decoder_layer_index": int(resolved_state_index - 1) if resolved_state_index > 0 else None,
        "num_image_tokens": int(mask.sum().detach().cpu().item()),
        "salience_q50": float(q50.detach().cpu().item()),
        "salience_q95": float(q95.detach().cpu().item()),
        "pooled_visual_delta_norm": float(torch.linalg.norm(visual_delta).detach().cpu().item()),
        "prior_projection_norm": float(torch.linalg.norm(projection).detach().cpu().item()),
        "pure_visual_norm": pure_norm,
        "relative_pure_visual": float(pure_norm / image_norm) if image_norm else None,
        "delta_prior_cosine": float(F.cosine_similarity(visual_delta, h_prior_last, dim=0).detach().cpu().item()),
    }, pure_visual


def _latent_steered_logp_from_outputs(
    model: Any,
    outputs_image: Any,
    outputs_prior: Any,
    latent_gamma: float,
) -> Tuple[Any, Dict[str, Any]]:
    import torch

    image_logp = torch.log_softmax(outputs_image.logits[0, -1, :].float(), dim=-1)
    summary, pure_visual = _orthogonal_visual_residual_from_outputs(outputs_image, outputs_prior)
    if pure_visual is None or float(latent_gamma) == 0.0:
        summary = {**summary, "latent_gamma": float(latent_gamma), "steered": False}
        return image_logp, summary

    states_image = getattr(outputs_image, "hidden_states", None) or []
    h_image = states_image[-1][0, -1, :]
    output_embeddings = model.get_output_embeddings()
    weight = getattr(output_embeddings, "weight", None)
    target_dtype = getattr(weight, "dtype", h_image.dtype)
    target_device = getattr(weight, "device", h_image.device)
    steered_hidden = h_image.float() + float(latent_gamma) * pure_visual
    steered_logits = output_embeddings(steered_hidden.to(device=target_device, dtype=target_dtype)).float()
    steered_logp = torch.log_softmax(steered_logits, dim=-1).to(device=image_logp.device)
    summary = {**summary, "latent_gamma": float(latent_gamma), "steered": True}
    return steered_logp, summary


def _revis_prior_correction_score(prior_logits: Any, prior_logp: Any, score_form: str) -> Tuple[Any, Dict[str, Any]]:
    import torch

    form = str(score_form or "logprob").strip().lower()
    logits = prior_logits.float()
    if form == "logprob":
        score = prior_logp.float()
    elif form == "centered_logit":
        score = logits - logits.mean()
    elif form == "zscore_logit":
        centered = logits - logits.mean()
        score = centered / centered.std(unbiased=False).clamp_min(1e-6)
    elif form == "positive_zscore_logit":
        centered = logits - logits.mean()
        score = torch.clamp(centered / centered.std(unbiased=False).clamp_min(1e-6), min=0.0)
    else:
        raise ValueError(f"unknown revis prior score form: {score_form}")

    top_id = int(torch.argmax(prior_logp).detach().cpu().item())
    return score.to(device=prior_logp.device), {
        "form": form,
        "top_token_id": top_id,
        "top_logprob": float(prior_logp[top_id].detach().cpu().item()),
        "top_probability": float(torch.exp(prior_logp[top_id]).detach().cpu().item()),
        "top_correction": float(score[top_id].detach().cpu().item()),
        "mean_correction": float(score.mean().detach().cpu().item()),
        "std_correction": float(score.std(unbiased=False).detach().cpu().item()),
        "max_correction": float(score.max().detach().cpu().item()),
        "min_correction": float(score.min().detach().cpu().item()),
    }


def _prior_token_subspace_steered_logp(
    model: Any,
    outputs_image: Any,
    prior_logp: Any,
    alpha: float,
    top_k: int,
) -> Tuple[Any, Dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    image_logp = torch.log_softmax(outputs_image.logits[0, -1, :].float(), dim=-1)
    states_image = getattr(outputs_image, "hidden_states", None) or []
    if not states_image:
        return image_logp, {"available": False, "reason": "missing_hidden_states", "alpha": float(alpha)}
    output_embeddings = model.get_output_embeddings()
    weight = getattr(output_embeddings, "weight", None)
    if weight is None:
        return image_logp, {"available": False, "reason": "missing_output_embedding_weight", "alpha": float(alpha)}
    if float(alpha) == 0.0:
        return image_logp, {"available": True, "steered": False, "alpha": 0.0}

    h_image = states_image[-1][0, -1, :].float()
    top_n = max(1, min(int(top_k), int(prior_logp.numel())))
    top_logp, top_ids = torch.topk(prior_logp.float(), k=top_n)
    top_probs = torch.softmax(top_logp, dim=0).to(device=weight.device, dtype=torch.float32)
    token_vecs = weight[top_ids.to(device=weight.device)].float()
    prior_direction = torch.sum(token_vecs * top_probs[:, None], dim=0).to(device=h_image.device)
    direction_norm = torch.linalg.norm(prior_direction).clamp_min(1e-12)
    unit_direction = prior_direction / direction_norm
    projection_coeff_raw = torch.dot(h_image, unit_direction)
    projection_coeff = torch.clamp(projection_coeff_raw, min=0.0)
    removed = float(alpha) * projection_coeff * unit_direction
    steered_hidden = h_image - removed

    target_dtype = getattr(weight, "dtype", h_image.dtype)
    target_device = getattr(weight, "device", h_image.device)
    steered_logits = output_embeddings(steered_hidden.to(device=target_device, dtype=target_dtype)).float()
    steered_logp = torch.log_softmax(steered_logits, dim=-1).to(device=image_logp.device)
    prior_top_id = int(top_ids[0].detach().cpu().item())
    image_top_id = int(torch.argmax(image_logp).detach().cpu().item())
    steered_top_id = int(torch.argmax(steered_logp).detach().cpu().item())
    summary = {
        "available": True,
        "steered": True,
        "type": "prior_token_embedding_subspace_suppression",
        "alpha": float(alpha),
        "top_k": int(top_n),
        "prior_top_token_id": prior_top_id,
        "prior_top_probability": float(torch.exp(prior_logp[prior_top_id]).detach().cpu().item()),
        "image_top_token_id": image_top_id,
        "steered_top_token_id": steered_top_id,
        "hidden_norm": float(torch.linalg.norm(h_image.detach()).detach().cpu().item()),
        "prior_direction_norm": float(direction_norm.detach().cpu().item()),
        "projection_coeff_raw": float(projection_coeff_raw.detach().cpu().item()),
        "projection_coeff_positive": float(projection_coeff.detach().cpu().item()),
        "removed_norm": float(torch.linalg.norm(removed.detach()).detach().cpu().item()),
        "image_prior_direction_cosine": float(F.cosine_similarity(h_image, prior_direction, dim=0).detach().cpu().item()),
        "prior_top_logp_before": float(image_logp[prior_top_id].detach().cpu().item()),
        "prior_top_logp_after": float(steered_logp[prior_top_id].detach().cpu().item()),
        "prior_top_logp_delta": float((steered_logp[prior_top_id] - image_logp[prior_top_id]).detach().cpu().item()),
        "top_token_ids": [int(x) for x in top_ids.detach().cpu().tolist()],
    }
    return steered_logp, summary


def _topk_union_token_ids(scores: List[Any], k: int) -> Any:
    import torch

    valid_scores = [score for score in scores if score is not None]
    if not valid_scores:
        raise ValueError("empty token score list")
    top_k = max(2, min(int(k), int(valid_scores[0].numel())))
    ids = set()
    for score in valid_scores:
        _, top_ids = torch.topk(score, k=top_k)
        ids.update(int(x) for x in top_ids.detach().cpu().tolist())
    return torch.tensor(sorted(ids), dtype=torch.long, device=valid_scores[0].device)


def _visual_attention_summary(model: Any, inputs_image: Dict[str, Any], outputs_image: Any) -> Dict[str, Any]:
    import torch

    attentions = getattr(outputs_image, "attentions", None)
    if not attentions:
        return {"available": False, "reason": "missing_attentions"}
    input_ids = inputs_image.get("input_ids")
    image_token_id = getattr(getattr(model, "config", None), "image_token_id", None)
    if input_ids is None or image_token_id is None:
        return {"available": False, "reason": "missing_input_ids_or_image_token_id"}
    image_positions = (input_ids[0] == int(image_token_id)).nonzero(as_tuple=False).flatten()
    if image_positions.numel() == 0:
        return {"available": False, "reason": "no_image_tokens", "image_token_id": int(image_token_id)}

    num_layers = len(attentions)
    layer_rows: List[Dict[str, Any]] = []
    for layer_idx in _probe_layer_indices(num_layers):
        attn = attentions[layer_idx]
        if attn is None:
            continue
        # Shape: [batch, heads, target_len, source_len].
        row = attn[0, :, -1, :].float()
        image_pos = image_positions.to(device=row.device)
        visual_by_head = row[:, image_pos].sum(dim=-1)
        total_by_head = row.sum(dim=-1).clamp_min(1e-12)
        visual_ratio = visual_by_head / total_by_head
        layer_rows.append(
            {
                "layer": int(layer_idx),
                "mean_visual_attention": float(visual_by_head.mean().detach().cpu().item()),
                "max_visual_attention": float(visual_by_head.max().detach().cpu().item()),
                "min_visual_attention": float(visual_by_head.min().detach().cpu().item()),
                "mean_visual_attention_ratio": float(visual_ratio.mean().detach().cpu().item()),
                "max_visual_attention_ratio": float(visual_ratio.max().detach().cpu().item()),
                "num_heads": int(row.shape[0]),
            }
        )

    last = layer_rows[-1] if layer_rows else None
    mean_over_layers = (
        sum(float(r["mean_visual_attention"]) for r in layer_rows) / len(layer_rows)
        if layer_rows
        else None
    )
    return {
        "available": True,
        "type": "decoder_last_token_to_image_tokens",
        "image_token_id": int(image_token_id),
        "num_image_tokens": int(image_positions.numel()),
        "image_token_start": int(image_positions[0].detach().cpu().item()),
        "image_token_end": int(image_positions[-1].detach().cpu().item()),
        "num_attention_layers": int(num_layers),
        "layers": layer_rows,
        "last_probe_layer": last,
        "mean_visual_attention_over_probe_layers": mean_over_layers,
    }


def _text_inertia_risk_summary(
    token_id: int,
    token_text: str,
    image_top: int,
    prior_top: int,
    image_logp: Any,
    prior_logp: Any,
    visual_attention: Dict[str, Any],
    visual_attention_max: float,
    logprob_margin: float,
    prior_logp_min: float,
    scope: str,
) -> Dict[str, Any]:
    token_id = int(token_id)
    token_clean = _normalize_text(str(token_text or "")).strip()
    text_stopwords = {
        "a", "an", "the", "in", "on", "of", "and", "or", "to", "with", "is", "are", "has", "have",
        "this", "that", "image", "photo", "picture", "visible", "visibly", "showing", "shows",
        "displays", "displaying", "including", "include",
    }
    token_is_count = bool(token_clean.isdigit() or token_clean in _NUMBER_WORDS)
    token_is_content = bool(re.search(r"[a-z0-9]", token_clean) and token_clean not in text_stopwords)
    scope = str(scope or "all").strip().lower()
    if scope == "count":
        scope_match = token_is_count
    elif scope == "content":
        scope_match = token_is_content
    else:
        scope_match = True
    image_value = float(image_logp[token_id].detach().cpu().item())
    prior_value = float(prior_logp[token_id].detach().cpu().item())
    visual_gain = float(image_value - prior_value)
    mean_visual_attention = None
    if isinstance(visual_attention, dict):
        raw_attn = visual_attention.get("mean_visual_attention_over_probe_layers")
        if isinstance(raw_attn, (int, float)):
            mean_visual_attention = float(raw_attn)

    image_prior_agree = bool(token_id == int(image_top) == int(prior_top))
    strong_prior = bool(prior_value >= float(prior_logp_min))
    low_visual_gain = bool(abs(visual_gain) <= float(logprob_margin))
    low_visual_attention = bool(
        mean_visual_attention is not None and mean_visual_attention <= float(visual_attention_max)
    )
    base_risk = bool(image_prior_agree and strong_prior and low_visual_gain and low_visual_attention)
    risk = bool(base_risk and scope_match)
    reasons: List[str] = []
    if image_prior_agree:
        reasons.append("image_top_equals_prior_top")
    if strong_prior:
        reasons.append("strong_text_prior")
    if low_visual_gain:
        reasons.append("low_image_over_prior_gain")
    if low_visual_attention:
        reasons.append("low_visual_attention")
    missing: List[str] = []
    if not image_prior_agree:
        missing.append("image_prior_disagree_or_token_not_top")
    if not strong_prior:
        missing.append("prior_logp_below_threshold")
    if not low_visual_gain:
        missing.append("image_prior_logp_gap_above_threshold")
    if mean_visual_attention is None:
        missing.append("visual_attention_unavailable")
    elif not low_visual_attention:
        missing.append("visual_attention_above_threshold")
    if not scope_match:
        missing.append(f"token_outside_{scope}_scope")
    return {
        "available": mean_visual_attention is not None,
        "token_id": token_id,
        "token_text": str(token_text or ""),
        "token_clean": token_clean,
        "token_is_count": token_is_count,
        "token_is_content": token_is_content,
        "scope": scope,
        "scope_match": scope_match,
        "base_risk": base_risk,
        "image_top": int(image_top),
        "prior_top": int(prior_top),
        "risk": risk,
        "reasons": reasons,
        "missing": missing,
        "image_logp": image_value,
        "prior_logp": prior_value,
        "image_minus_prior_logp": visual_gain,
        "mean_visual_attention": mean_visual_attention,
        "visual_attention_max": float(visual_attention_max),
        "logprob_margin": float(logprob_margin),
        "prior_logp_min": float(prior_logp_min),
    }


def _get_decoder_layers(model: Any) -> Any:
    paths = (
        ("model", "language_model", "layers"),
        ("language_model", "layers"),
        ("model", "layers"),
    )
    for path in paths:
        node = model
        ok = True
        for attr in path:
            if not hasattr(node, attr):
                ok = False
                break
            node = getattr(node, attr)
        if ok and hasattr(node, "__len__") and hasattr(node, "__getitem__"):
            return node
    raise RuntimeError("could not locate decoder layers for layer-wise REVIS steering")


def _resolve_revis_layer_index(model: Any, layer_index: int, layer_fraction: float) -> Tuple[int, int]:
    layers = _get_decoder_layers(model)
    num_layers = len(layers)
    if num_layers <= 0:
        raise RuntimeError("decoder layer list is empty")
    if int(layer_index) >= 0:
        idx = int(layer_index)
    else:
        frac = min(max(float(layer_fraction), 0.0), 1.0)
        idx = int(round(frac * float(num_layers - 1)))
    return max(0, min(idx, num_layers - 1)), int(num_layers)


def _layer_residual_steered_logp(
    model: Any,
    inputs_image: Dict[str, Any],
    layer_index: int,
    residual: Any,
    layer_gamma: float,
) -> Tuple[Any, Dict[str, Any]]:
    import torch

    if residual is None or float(layer_gamma) == 0.0:
        raise ValueError("layer residual steering requires a nonzero gamma and residual")

    layers = _get_decoder_layers(model)
    if int(layer_index) < 0 or int(layer_index) >= len(layers):
        raise ValueError(f"layer index out of range: {layer_index}")
    target_layer = layers[int(layer_index)]
    hook_state: Dict[str, Any] = {
        "layer_index": int(layer_index),
        "layer_gamma": float(layer_gamma),
        "applied": False,
    }

    def hook_fn(_module: Any, _inputs: Tuple[Any, ...], output: Any) -> Any:
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None
        if not hasattr(hidden, "dim") or hidden.dim() < 3:
            return output
        add = residual.to(device=hidden.device, dtype=hidden.dtype) * float(layer_gamma)
        hidden_new = hidden.clone()
        hidden_new[:, -1, :] = hidden_new[:, -1, :] + add
        hook_state.update(
            {
                "applied": True,
                "hidden_device": str(hidden.device),
                "hidden_dtype": str(hidden.dtype),
                "residual_norm": float(torch.linalg.norm(residual.detach().float().cpu()).item()),
                "added_norm": float(torch.linalg.norm((add.detach().float()).cpu()).item()),
            }
        )
        if rest is None:
            return hidden_new
        return (hidden_new, *rest)

    handle = target_layer.register_forward_hook(hook_fn)
    try:
        with torch.inference_mode():
            outputs = model(**inputs_image, output_hidden_states=False, use_cache=False)
            logp = torch.log_softmax(outputs.logits[0, -1, :].float(), dim=-1)
    finally:
        handle.remove()
    del outputs
    return logp, hook_state


def _layer_prior_subspace_steered_logp(
    model: Any,
    inputs_image: Dict[str, Any],
    prior_logp: Any,
    image_logp: Any,
    layer_index: int,
    alpha: float,
    top_k: int,
) -> Tuple[Any, Dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    if float(alpha) == 0.0:
        return image_logp, {"applied": False, "reason": "zero_alpha", "layer_alpha": 0.0}

    output_embeddings = model.get_output_embeddings()
    weight = getattr(output_embeddings, "weight", None)
    if weight is None:
        return image_logp, {"applied": False, "reason": "missing_output_embedding_weight", "layer_alpha": float(alpha)}

    layers = _get_decoder_layers(model)
    if int(layer_index) < 0 or int(layer_index) >= len(layers):
        raise ValueError(f"layer index out of range: {layer_index}")

    top_n = max(1, min(int(top_k), int(prior_logp.numel())))
    top_logp, top_ids = torch.topk(prior_logp.float(), k=top_n)
    top_probs = torch.softmax(top_logp, dim=0).to(device=weight.device, dtype=torch.float32)
    token_vecs = weight[top_ids.to(device=weight.device)].float()
    prior_direction_base = torch.sum(token_vecs * top_probs[:, None], dim=0)
    prior_top_id = int(top_ids[0].detach().cpu().item())
    hook_state: Dict[str, Any] = {
        "type": "layer_prior_token_embedding_subspace_suppression",
        "layer_index": int(layer_index),
        "layer_alpha": float(alpha),
        "top_k": int(top_n),
        "prior_top_token_id": prior_top_id,
        "prior_top_probability": float(torch.exp(prior_logp[prior_top_id]).detach().cpu().item()),
        "applied": False,
    }

    target_layer = layers[int(layer_index)]

    def hook_fn(_module: Any, _inputs: Tuple[Any, ...], output: Any) -> Any:
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None
        if not hasattr(hidden, "dim") or hidden.dim() < 3:
            return output
        direction = prior_direction_base.to(device=hidden.device, dtype=torch.float32)
        direction_norm = torch.linalg.norm(direction).clamp_min(1e-12)
        unit_direction = direction / direction_norm
        h_last = hidden[:, -1, :].float()
        projection_raw = torch.sum(h_last * unit_direction[None, :], dim=-1, keepdim=True)
        projection = torch.clamp(projection_raw, min=0.0)
        removed = float(alpha) * projection * unit_direction[None, :]
        hidden_new = hidden.clone()
        hidden_new[:, -1, :] = hidden_new[:, -1, :] - removed.to(device=hidden.device, dtype=hidden.dtype)
        hook_state.update(
            {
                "applied": True,
                "hidden_device": str(hidden.device),
                "hidden_dtype": str(hidden.dtype),
                "hidden_norm": float(torch.linalg.norm(h_last[0].detach().float().cpu()).item()),
                "prior_direction_norm": float(direction_norm.detach().cpu().item()),
                "projection_coeff_raw": float(projection_raw[0, 0].detach().cpu().item()),
                "projection_coeff_positive": float(projection[0, 0].detach().cpu().item()),
                "removed_norm": float(torch.linalg.norm(removed[0].detach().float().cpu()).item()),
                "hidden_prior_direction_cosine": float(F.cosine_similarity(h_last[0], direction, dim=0).detach().cpu().item()),
            }
        )
        if rest is None:
            return hidden_new
        return (hidden_new, *rest)

    handle = target_layer.register_forward_hook(hook_fn)
    try:
        with torch.inference_mode():
            outputs = model(**inputs_image, output_hidden_states=False, use_cache=False)
            steered_logp = torch.log_softmax(outputs.logits[0, -1, :].float(), dim=-1)
    finally:
        handle.remove()
    steered_top_id = int(torch.argmax(steered_logp).detach().cpu().item())
    hook_state.update(
        {
            "steered_top_token_id": steered_top_id,
            "prior_top_logp_before": float(image_logp[prior_top_id].detach().cpu().item()),
            "prior_top_logp_after": float(steered_logp[prior_top_id].detach().cpu().item()),
            "prior_top_logp_delta": float((steered_logp[prior_top_id] - image_logp[prior_top_id]).detach().cpu().item()),
            "top_token_ids": [int(x) for x in top_ids.detach().cpu().tolist()],
        }
    )
    del outputs
    return steered_logp.to(device=image_logp.device), hook_state


def _get_final_norm_module(model: Any) -> Optional[Any]:
    paths = (
        ("model", "language_model", "norm"),
        ("model", "language_model", "model", "norm"),
        ("language_model", "norm"),
        ("language_model", "model", "norm"),
        ("model", "norm"),
    )
    for path in paths:
        node = model
        ok = True
        for attr in path:
            if not hasattr(node, attr):
                ok = False
                break
            node = getattr(node, attr)
        if ok:
            return node
    return None


def _accelerate_execution_device(module: Any) -> Optional[Any]:
    """Find the real device behind direct or sequential Accelerate hooks."""

    root = getattr(module, "_hf_hook", None)
    pending = [root] if root is not None else []
    while pending:
        hook = pending.pop(0)
        device = getattr(hook, "execution_device", None)
        if device is not None:
            return device
        pending.extend(list(getattr(hook, "hooks", ()) or ()))
    return None


def _module_device_dtype(module: Any, fallback_device: Any, fallback_dtype: Any) -> Tuple[Any, Any]:
    param = None
    try:
        param = next(module.parameters(), None)
    except Exception:
        param = None
    if param is None:
        try:
            param = next(module.buffers(), None)
        except Exception:
            param = None
    if param is None:
        return fallback_device, fallback_dtype
    dtype = param.dtype if getattr(param, "dtype", None) is not None else fallback_dtype
    device = param.device
    if getattr(device, "type", None) == "meta":
        execution_device = _accelerate_execution_device(module)
        if execution_device is not None:
            device = execution_device
        else:
            device = fallback_device
    return device, dtype


def _apply_final_norm_for_lens(model: Any, hidden: Any) -> Tuple[Any, Dict[str, Any]]:
    norm = _get_final_norm_module(model)
    if norm is None:
        return hidden.float(), {"final_norm_found": False}
    device, dtype = _module_device_dtype(norm, hidden.device, hidden.dtype)
    x = hidden.to(device=device, dtype=dtype).reshape(1, 1, -1)
    y = norm(x)[0, 0, :].float().to(device=hidden.device)
    return y, {
        "final_norm_found": True,
        "final_norm_class": type(norm).__name__,
        "final_norm_device": str(device),
        "final_norm_dtype": str(dtype),
    }


def _logit_lens_logp_from_hidden(model: Any, hidden: Any) -> Tuple[Any, Dict[str, Any]]:
    import torch

    output_embeddings = model.get_output_embeddings()
    weight = getattr(output_embeddings, "weight", None)
    if output_embeddings is None or weight is None:
        raise RuntimeError("logit-lens readout requires output embedding weights")
    normed, norm_summary = _apply_final_norm_for_lens(model, hidden)
    target_device, target_dtype = _module_device_dtype(
        output_embeddings, normed.device, normed.dtype
    )
    with torch.inference_mode():
        logits = output_embeddings(normed.to(device=target_device, dtype=target_dtype)).float()
        if logits.dim() > 1:
            logits = logits.reshape(-1, logits.shape[-1])[-1]
        logp = torch.log_softmax(logits, dim=-1).to(device=hidden.device)
    return logp, {
        **norm_summary,
        "lens": "logit_lens_final_norm_unembed",
        "output_embedding_device": str(target_device),
        "output_embedding_dtype": str(target_dtype),
    }


def _load_jspace_lens(path: str) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    cache_key = str(Path(path).resolve())
    cached = _JSPACE_LENS_CACHE.get(cache_key)
    if cached is not None:
        return cached
    import torch

    payload = torch.load(cache_key, map_location="cpu", weights_only=True)
    if "J" not in payload:
        raise ValueError(f"{path} is not a JacobianLens checkpoint")
    lens = {
        "path": cache_key,
        "jacobians": {int(k): v.float() for k, v in payload["J"].items()},
        "source_layers": [int(x) for x in payload.get("source_layers", payload["J"].keys())],
        "n_prompts": int(payload.get("n_prompts", 0)),
        "d_model": int(payload.get("d_model", 0)),
    }
    _JSPACE_LENS_CACHE[cache_key] = lens
    return lens


def _jacobian_lens_logp_from_hidden(model: Any, hidden: Any, jacobian: Any) -> Tuple[Any, Dict[str, Any]]:
    import torch

    J = jacobian.to(device=hidden.device, dtype=torch.float32)
    transported = torch.matmul(hidden.float(), J.T)
    logp, meta = _logit_lens_logp_from_hidden(model, transported)
    return logp, {
        **meta,
        "lens": "fitted_jacobian_lens",
        "jacobian_device": str(J.device),
        "jacobian_dtype": str(J.dtype),
    }


def _lens_top_token_rows(tokenizer: Any, logp: Any, top_k: int) -> List[Dict[str, Any]]:
    import torch

    top_n = max(1, min(int(top_k), int(logp.numel())))
    top_logp, top_ids = torch.topk(logp.float(), k=top_n)
    probs = torch.softmax(top_logp, dim=0)
    rows: List[Dict[str, Any]] = []
    for rank, (tok_id, value, prob) in enumerate(zip(top_ids, top_logp, probs), start=1):
        token_id = int(tok_id.detach().cpu().item())
        rows.append(
            {
                "rank": int(rank),
                "token_id": token_id,
                "token": tokenizer.decode([token_id]),
                "lens_logp": float(value.detach().cpu().item()),
                "local_probability": float(prob.detach().cpu().item()),
            }
        )
    return rows


def _jspace_contrast_token_rows(
    tokenizer: Any,
    image_logp: Any,
    prior_logp: Any,
    top_k: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Find target-free visual/prior concepts from the two J-space readouts."""
    import torch

    vocab_size = int(image_logp.numel())
    keep = max(1, min(int(top_k), vocab_size))
    pool_k = max(64, min(vocab_size, keep * 32))
    visual_delta = image_logp.float() - prior_logp.float()
    prior_delta = -visual_delta
    pool_ids = torch.unique(
        torch.cat(
            [
                torch.topk(image_logp.float(), k=pool_k).indices,
                torch.topk(prior_logp.float(), k=pool_k).indices,
                torch.topk(visual_delta, k=pool_k).indices,
                torch.topk(prior_delta, k=pool_k).indices,
            ]
        )
    )

    def ranked(delta: Any, direction: str) -> List[Dict[str, Any]]:
        scores = delta[pool_ids]
        order = torch.argsort(scores, descending=True)
        rows: List[Dict[str, Any]] = []
        for idx in order:
            token_id = int(pool_ids[int(idx)].detach().cpu().item())
            token = tokenizer.decode([token_id])
            if not str(token).strip() or str(token).startswith("<|"):
                continue
            rows.append(
                {
                    "rank": len(rows) + 1,
                    "token_id": token_id,
                    "token": token,
                    "direction": direction,
                    "contrast_logp": float(delta[token_id].detach().cpu().item()),
                    "image_lens_logp": float(image_logp[token_id].detach().cpu().item()),
                    "prior_lens_logp": float(prior_logp[token_id].detach().cpu().item()),
                }
            )
            if len(rows) >= keep:
                break
        return rows

    return ranked(visual_delta, "visual_over_prior"), ranked(prior_delta, "prior_over_visual")


def _jspace_logitlens_summary_from_outputs(
    model: Any,
    tokenizer: Any,
    outputs_image: Any,
    outputs_prior: Any,
    layer_index: int,
    top_k: int,
    lens_jacobian: Optional[Any] = None,
    lens_path: str = "",
) -> Dict[str, Any]:
    states_image = getattr(outputs_image, "hidden_states", None) or []
    states_prior = getattr(outputs_prior, "hidden_states", None) or []
    if not states_image or not states_prior:
        return {"available": False, "reason": "missing_hidden_states"}
    num_states = min(len(states_image), len(states_prior))
    state_index = max(0, min(int(layer_index) + 1, num_states - 1))
    h_image = states_image[state_index][0, -1, :].float()
    h_prior = states_prior[state_index][0, -1, :].to(device=h_image.device).float()
    try:
        if lens_jacobian is not None:
            image_lens_logp, image_lens_meta = _jacobian_lens_logp_from_hidden(model, h_image, lens_jacobian)
            prior_lens_logp, prior_lens_meta = _jacobian_lens_logp_from_hidden(model, h_prior, lens_jacobian)
            lens_name = "fitted_jacobian_lens"
        else:
            image_lens_logp, image_lens_meta = _logit_lens_logp_from_hidden(model, h_image)
            prior_lens_logp, prior_lens_meta = _logit_lens_logp_from_hidden(model, h_prior)
            lens_name = "logit_lens_fallback"
    except Exception as e:
        return {"available": False, "reason": f"jspace_lens_failed: {e}"}
    image_rows = _lens_top_token_rows(tokenizer, image_lens_logp, top_k)
    prior_rows = _lens_top_token_rows(tokenizer, prior_lens_logp, top_k)
    visual_contrast_rows, prior_contrast_rows = _jspace_contrast_token_rows(
        tokenizer,
        image_lens_logp,
        prior_lens_logp,
        top_k,
    )
    image_ids = {int(row["token_id"]) for row in image_rows}
    prior_ids = {int(row["token_id"]) for row in prior_rows}
    return {
        "available": True,
        "type": "jspace_logitlens_readout",
        "decoder_layer_index": int(layer_index),
        "state_index": int(state_index),
        "num_states": int(num_states),
        "lens": lens_name,
        "lens_path": str(lens_path or ""),
        "image_lens": image_lens_meta,
        "prior_lens": prior_lens_meta,
        "image_top_tokens": image_rows,
        "prior_top_tokens": prior_rows,
        "visual_contrast_tokens": visual_contrast_rows,
        "prior_contrast_tokens": prior_contrast_rows,
        "shared_top_token_ids": sorted(image_ids & prior_ids),
        "prior_only_token_ids": sorted(prior_ids - image_ids),
        "image_only_token_ids": sorted(image_ids - prior_ids),
    }


def _layer_jspace_prior_steered_logp(
    model: Any,
    tokenizer: Any,
    inputs_image: Dict[str, Any],
    outputs_image: Any,
    outputs_prior: Any,
    image_logp: Any,
    prior_logp: Any,
    layer_index: int,
    alpha: float,
    gamma: float,
    top_k: int,
    lens_jacobian: Optional[Any] = None,
    lens_path: str = "",
    swap_alpha: float = 0.0,
) -> Tuple[Any, Dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    layers = _get_decoder_layers(model)
    if int(layer_index) < 0 or int(layer_index) >= len(layers):
        raise ValueError(f"layer index out of range: {layer_index}")

    output_embeddings = model.get_output_embeddings()
    weight = getattr(output_embeddings, "weight", None)
    if output_embeddings is None or weight is None:
        return image_logp, {
            "available": False,
            "applied": False,
            "reason": "missing_output_embedding_weight",
            "jspace_alpha": float(alpha),
            "jspace_gamma": float(gamma),
        }

    summary = _jspace_logitlens_summary_from_outputs(
        model,
        tokenizer,
        outputs_image,
        outputs_prior,
        layer_index=int(layer_index),
        top_k=int(top_k),
        lens_jacobian=lens_jacobian,
        lens_path=lens_path,
    )
    if not summary.get("available"):
        return image_logp, {
            **summary,
            "applied": False,
            "jspace_alpha": float(alpha),
            "jspace_gamma": float(gamma),
        }
    if float(alpha) == 0.0 and float(gamma) == 0.0 and float(swap_alpha) == 0.0:
        return image_logp, {
            **summary,
            "applied": False,
            "steered": False,
            "jspace_alpha": 0.0,
            "jspace_gamma": 0.0,
            "jspace_swap_alpha": float(swap_alpha),
        }

    prior_rows = summary.get("prior_contrast_tokens") or summary.get("prior_top_tokens") or []
    top_ids_list = [int(row["token_id"]) for row in prior_rows]
    if not top_ids_list:
        return image_logp, {
            **summary,
            "applied": False,
            "reason": "empty_prior_lens_top_tokens",
            "jspace_alpha": float(alpha),
            "jspace_gamma": float(gamma),
        }

    top_ids = torch.tensor(top_ids_list, dtype=torch.long, device=getattr(weight, "device", image_logp.device))
    if lens_jacobian is not None:
        J_for_vecs = lens_jacobian.to(device=getattr(weight, "device", image_logp.device), dtype=torch.float32)
        token_vecs = torch.matmul(weight[top_ids].float(), J_for_vecs)
    else:
        J_for_vecs = None
        token_vecs = weight[top_ids].float()
    target_rows = summary.get("visual_contrast_tokens") or []
    source_id_set = set(top_ids_list)
    target_ids_list = [
        int(row["token_id"])
        for row in target_rows
        if int(row["token_id"]) not in source_id_set
    ][: len(top_ids_list)]
    target_token_id = int(target_ids_list[0]) if target_ids_list else None
    target_vecs = None
    if target_ids_list and float(swap_alpha) != 0.0:
        target_ids = torch.tensor(
            target_ids_list,
            dtype=torch.long,
            device=getattr(weight, "device", image_logp.device),
        )
        if J_for_vecs is not None:
            target_vecs = torch.matmul(weight[target_ids].float(), J_for_vecs)
        else:
            target_vecs = weight[target_ids].float()
    state_index = int(summary.get("state_index"))
    visual_summary, pure_visual = _salient_visual_patch_residual_from_outputs(
        model,
        inputs_image,
        outputs_image,
        outputs_prior,
        state_index=state_index,
    )
    if pure_visual is None:
        visual_summary, pure_visual = _orthogonal_visual_residual_from_outputs(
            outputs_image,
            outputs_prior,
            state_index=state_index,
        )
    hook_state: Dict[str, Any] = {
        **summary,
        "type": "layer_jspace_logitlens_sparse_prior_suppression",
        "layer_index": int(layer_index),
        "jspace_alpha": float(alpha),
        "jspace_gamma": float(gamma),
        "jspace_swap_alpha": float(swap_alpha),
        "top_k": int(len(top_ids_list)),
        "prior_top_token_id": int(top_ids_list[0]),
        "prior_contrast_token_ids": top_ids_list,
        "visual_contrast_token_ids": target_ids_list,
        "target_token_id": target_token_id,
        "target_token": tokenizer.decode([target_token_id]) if target_token_id is not None else None,
        "prior_top_probability": float(torch.exp(prior_logp[int(top_ids_list[0])]).detach().cpu().item()),
        "visual_residual": visual_summary,
        "applied": False,
    }

    target_layer = layers[int(layer_index)]

    def hook_fn(_module: Any, _inputs: Tuple[Any, ...], output: Any) -> Any:
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None
        if not hasattr(hidden, "dim") or hidden.dim() < 3:
            return output

        h_last = hidden[:, -1, :].float()
        V = token_vecs.to(device=hidden.device, dtype=torch.float32)
        gram = torch.matmul(V, V.T)
        pinv = torch.linalg.pinv(gram.float())
        coords = torch.matmul(torch.matmul(pinv, V), h_last.T).T
        positive_coords = torch.clamp(coords, min=0.0)
        removed = float(alpha) * torch.matmul(positive_coords, V)
        target_add = torch.zeros_like(removed)
        swap_coordinates = None
        if target_vecs is not None and float(swap_alpha) != 0.0:
            V_target = target_vecs.to(device=hidden.device, dtype=torch.float32)
            pair_count = min(int(V.shape[0]), int(V_target.shape[0]))
            V_pair = torch.cat([V[:pair_count], V_target[:pair_count]], dim=0)
            pair_gram = torch.matmul(V_pair, V_pair.T)
            pair_coords = torch.matmul(
                torch.matmul(torch.linalg.pinv(pair_gram.float()), V_pair),
                h_last.T,
            ).T
            swapped_coords = pair_coords.clone()
            source_coords = pair_coords[:, :pair_count]
            visual_coords = pair_coords[:, pair_count:]
            amount = float(swap_alpha)
            swapped_coords[:, :pair_count] = (1.0 - amount) * source_coords + amount * visual_coords
            swapped_coords[:, pair_count:] = (1.0 - amount) * visual_coords + amount * source_coords
            target_add = torch.matmul(swapped_coords - pair_coords, V_pair)
            swap_coordinates = {
                "source": [float(x) for x in source_coords[0].detach().cpu().tolist()],
                "visual": [float(x) for x in visual_coords[0].detach().cpu().tolist()],
            }
        visual_add = torch.zeros_like(removed)
        if pure_visual is not None and float(gamma) != 0.0:
            visual_vec = pure_visual.to(device=hidden.device, dtype=torch.float32)
            if float(alpha) != 0.0 and V.numel() > 0:
                prior_component = torch.matmul(
                    torch.matmul(pinv, V),
                    visual_vec.reshape(-1, 1),
                ).reshape(-1)
                visual_vec = visual_vec - torch.matmul(prior_component, V)
            visual_add = float(gamma) * visual_vec.reshape(1, -1)
        hidden_new = hidden.clone()
        delta = -removed + target_add + visual_add
        hidden_new[:, -1, :] = hidden_new[:, -1, :] + delta.to(device=hidden.device, dtype=hidden.dtype)
        coord_values = [float(x) for x in coords[0].detach().cpu().tolist()]
        positive_coord_values = [float(x) for x in positive_coords[0].detach().cpu().tolist()]
        removed_norm = float(torch.linalg.norm(removed[0].detach().float().cpu()).item())
        visual_add_norm = float(torch.linalg.norm(visual_add[0].detach().float().cpu()).item())
        mean_cosine = None
        try:
            mean_cosine = float(F.cosine_similarity(h_last[0][None, :], V, dim=-1).detach().mean().cpu().item())
        except Exception:
            mean_cosine = None
        hook_state.update(
            {
                "applied": True,
                "steered": True,
                "hidden_device": str(hidden.device),
                "hidden_dtype": str(hidden.dtype),
                "hidden_norm": float(torch.linalg.norm(h_last[0].detach().float().cpu()).item()),
                "jspace_coordinates": coord_values,
                "jspace_positive_coordinates": positive_coord_values,
                "removed_norm": removed_norm,
                "target_add_norm": float(torch.linalg.norm(target_add[0].detach().float().cpu()).item()),
                "swap_coordinates": swap_coordinates,
                "visual_add_norm": visual_add_norm,
                "delta_norm": float(torch.linalg.norm(delta[0].detach().float().cpu()).item()),
                "mean_hidden_jspace_vector_cosine": mean_cosine,
            }
        )
        if rest is None:
            return hidden_new
        return (hidden_new, *rest)

    handle = target_layer.register_forward_hook(hook_fn)
    try:
        with torch.inference_mode():
            outputs = model(**inputs_image, output_hidden_states=False, use_cache=False)
            steered_logp = torch.log_softmax(outputs.logits[0, -1, :].float(), dim=-1)
    finally:
        handle.remove()
    steered_top_id = int(torch.argmax(steered_logp).detach().cpu().item())
    prior_top_id = int(top_ids_list[0])
    hook_state.update(
        {
            "steered_top_token_id": steered_top_id,
            "prior_top_logp_before": float(image_logp[prior_top_id].detach().cpu().item()),
            "prior_top_logp_after": float(steered_logp[prior_top_id].detach().cpu().item()),
            "prior_top_logp_delta": float((steered_logp[prior_top_id] - image_logp[prior_top_id]).detach().cpu().item()),
        }
    )
    del outputs
    return steered_logp.to(device=image_logp.device), hook_state


def _local_jacobian_prior_direction(
    model: Any,
    inputs_prior: Dict[str, Any],
    layer_index: int,
    token_id: int,
) -> Tuple[Any, Dict[str, Any]]:
    import torch

    layers = _get_decoder_layers(model)
    if int(layer_index) < 0 or int(layer_index) >= len(layers):
        raise ValueError(f"layer index out of range: {layer_index}")
    target_layer = layers[int(layer_index)]
    captured: Dict[str, Any] = {}

    def capture_hook(_module: Any, _inputs: Tuple[Any, ...], output: Any) -> Any:
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None
        if not hasattr(hidden, "detach") or not hasattr(hidden, "requires_grad_"):
            return output
        detached = hidden.detach().requires_grad_(True)
        captured["hidden"] = detached
        if rest is None:
            return detached
        return (detached, *rest)

    handle = target_layer.register_forward_hook(capture_hook)
    try:
        with torch.enable_grad():
            outputs = model(**inputs_prior, output_hidden_states=False, use_cache=False)
            if "hidden" not in captured:
                return None, {
                    "available": False,
                    "reason": "missing_captured_hidden",
                    "layer_index": int(layer_index),
                    "token_id": int(token_id),
                }
            target_logit = outputs.logits[0, -1, int(token_id)].float()
            grad = torch.autograd.grad(
                target_logit,
                captured["hidden"],
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )[0]
    finally:
        handle.remove()
    if grad is None:
        return None, {
            "available": False,
            "reason": "unused_captured_hidden",
            "layer_index": int(layer_index),
            "token_id": int(token_id),
        }
    direction = grad[0, -1, :].detach().float()
    summary = {
        "available": True,
        "type": "local_jacobian_prior_token_direction",
        "layer_index": int(layer_index),
        "token_id": int(token_id),
        "target_logit": float(target_logit.detach().cpu().item()),
        "direction_norm": float(torch.linalg.norm(direction.detach().float().cpu()).item()),
        "hidden_shape": [int(x) for x in captured["hidden"].shape],
    }
    del outputs, grad
    return direction, summary


def _layer_local_jacobian_prior_steered_logp(
    model: Any,
    tokenizer: Any,
    inputs_image: Dict[str, Any],
    inputs_prior: Dict[str, Any],
    outputs_image: Any,
    outputs_prior: Any,
    image_logp: Any,
    prior_logp: Any,
    layer_index: int,
    prior_token_id: int,
    alpha: float,
    gamma: float,
) -> Tuple[Any, Dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    if float(alpha) == 0.0 and float(gamma) == 0.0:
        return image_logp, {
            "available": True,
            "applied": False,
            "steered": False,
            "type": "layer_local_jacobian_prior_suppression",
            "jspace_alpha": 0.0,
            "jspace_gamma": 0.0,
        }

    direction, direction_summary = _local_jacobian_prior_direction(
        model,
        inputs_prior,
        layer_index=int(layer_index),
        token_id=int(prior_token_id),
    )
    if direction is None:
        return image_logp, {
            **direction_summary,
            "applied": False,
            "jspace_alpha": float(alpha),
            "jspace_gamma": float(gamma),
        }

    layers = _get_decoder_layers(model)
    target_layer = layers[int(layer_index)]
    state_index = int(layer_index) + 1
    visual_summary, pure_visual = _orthogonal_visual_residual_from_outputs(
        outputs_image,
        outputs_prior,
        state_index=state_index,
    )
    hook_state: Dict[str, Any] = {
        "available": True,
        "type": "layer_local_jacobian_prior_suppression",
        "lens": "context_local_jacobian",
        "layer_index": int(layer_index),
        "state_index": int(state_index),
        "jspace_alpha": float(alpha),
        "jspace_gamma": float(gamma),
        "prior_top_token_id": int(prior_token_id),
        "prior_top_token": tokenizer.decode([int(prior_token_id)]),
        "prior_top_probability": float(torch.exp(prior_logp[int(prior_token_id)]).detach().cpu().item()),
        "local_jacobian": direction_summary,
        "visual_residual": visual_summary,
        "applied": False,
    }

    def hook_fn(_module: Any, _inputs: Tuple[Any, ...], output: Any) -> Any:
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None
        if not hasattr(hidden, "dim") or hidden.dim() < 3:
            return output
        h_last = hidden[:, -1, :].float()
        direction_local = direction.to(device=hidden.device, dtype=torch.float32)
        direction_norm = torch.linalg.norm(direction_local).clamp_min(1e-12)
        unit_direction = direction_local / direction_norm
        projection_raw = torch.sum(h_last * unit_direction[None, :], dim=-1, keepdim=True)
        projection = torch.clamp(projection_raw, min=0.0)
        removed = float(alpha) * projection * unit_direction[None, :]
        visual_add = torch.zeros_like(removed)
        if pure_visual is not None and float(gamma) != 0.0:
            visual_vec = pure_visual.to(device=hidden.device, dtype=torch.float32)
            visual_vec = visual_vec - torch.dot(visual_vec, unit_direction) * unit_direction
            visual_add = float(gamma) * visual_vec.reshape(1, -1)
        delta = -removed + visual_add
        hidden_new = hidden.clone()
        hidden_new[:, -1, :] = hidden_new[:, -1, :] + delta.to(device=hidden.device, dtype=hidden.dtype)
        hook_state.update(
            {
                "applied": True,
                "steered": True,
                "hidden_device": str(hidden.device),
                "hidden_dtype": str(hidden.dtype),
                "hidden_norm": float(torch.linalg.norm(h_last[0].detach().float().cpu()).item()),
                "projection_coeff_raw": float(projection_raw[0, 0].detach().cpu().item()),
                "projection_coeff_positive": float(projection[0, 0].detach().cpu().item()),
                "removed_norm": float(torch.linalg.norm(removed[0].detach().float().cpu()).item()),
                "visual_add_norm": float(torch.linalg.norm(visual_add[0].detach().float().cpu()).item()),
                "delta_norm": float(torch.linalg.norm(delta[0].detach().float().cpu()).item()),
                "hidden_prior_gradient_cosine": float(
                    F.cosine_similarity(h_last[0], direction_local, dim=0).detach().cpu().item()
                ),
            }
        )
        if rest is None:
            return hidden_new
        return (hidden_new, *rest)

    handle = target_layer.register_forward_hook(hook_fn)
    try:
        with torch.inference_mode():
            outputs = model(**inputs_image, output_hidden_states=False, use_cache=False)
            steered_logp = torch.log_softmax(outputs.logits[0, -1, :].float(), dim=-1)
    finally:
        handle.remove()
    steered_top_id = int(torch.argmax(steered_logp).detach().cpu().item())
    hook_state.update(
        {
            "steered_top_token_id": steered_top_id,
            "prior_top_logp_before": float(image_logp[int(prior_token_id)].detach().cpu().item()),
            "prior_top_logp_after": float(steered_logp[int(prior_token_id)].detach().cpu().item()),
            "prior_top_logp_delta": float(
                (steered_logp[int(prior_token_id)] - image_logp[int(prior_token_id)]).detach().cpu().item()
            ),
        }
    )
    del outputs
    return steered_logp.to(device=image_logp.device), hook_state


def _layer_attention_prior_residual_steered_logp(
    model: Any,
    inputs_image: Dict[str, Any],
    inputs_prior: Dict[str, Any],
    image_logp: Any,
    prior_logp: Any,
    layer_index: int,
    alpha: float,
) -> Tuple[Any, Dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    if float(alpha) == 0.0:
        return image_logp, {"applied": False, "reason": "zero_alpha", "attention_prior_alpha": 0.0}

    layers = _get_decoder_layers(model)
    if int(layer_index) < 0 or int(layer_index) >= len(layers):
        raise ValueError(f"layer index out of range: {layer_index}")
    target_layer = layers[int(layer_index)]
    self_attn = getattr(target_layer, "self_attn", None)
    if self_attn is None:
        return image_logp, {
            "applied": False,
            "reason": "missing_self_attention_module",
            "attention_prior_alpha": float(alpha),
            "layer_index": int(layer_index),
        }

    prior_top_id = int(torch.argmax(prior_logp).detach().cpu().item())
    hook_state: Dict[str, Any] = {
        "type": "layer_attention_prior_residual_suppression",
        "layer_index": int(layer_index),
        "attention_prior_alpha": float(alpha),
        "prior_top_token_id": prior_top_id,
        "prior_top_probability": float(torch.exp(prior_logp[prior_top_id]).detach().cpu().item()),
        "applied": False,
    }
    captured: Dict[str, Any] = {}

    def capture_prior_hook(_module: Any, _inputs: Tuple[Any, ...], output: Any) -> Any:
        attn_output = output[0] if isinstance(output, tuple) else output
        if hasattr(attn_output, "dim") and attn_output.dim() >= 3:
            captured["prior_attention_residual"] = attn_output[:, -1, :].detach().float()
        return output

    handle = self_attn.register_forward_hook(capture_prior_hook)
    try:
        with torch.inference_mode():
            prior_outputs = model(**inputs_prior, output_hidden_states=False, use_cache=False)
    finally:
        handle.remove()
    del prior_outputs

    prior_attention_residual = captured.get("prior_attention_residual")
    if prior_attention_residual is None:
        return image_logp, {
            **hook_state,
            "applied": False,
            "reason": "missing_prior_attention_residual",
        }

    direction_base = prior_attention_residual[0].float()

    def suppress_image_hook(_module: Any, _inputs: Tuple[Any, ...], output: Any) -> Any:
        if isinstance(output, tuple):
            attn_output = output[0]
            rest = output[1:]
        else:
            attn_output = output
            rest = None
        if not hasattr(attn_output, "dim") or attn_output.dim() < 3:
            return output

        direction = direction_base.to(device=attn_output.device, dtype=torch.float32)
        direction_norm = torch.linalg.norm(direction).clamp_min(1e-12)
        unit_direction = direction / direction_norm
        h_last = attn_output[:, -1, :].float()
        projection_raw = torch.sum(h_last * unit_direction[None, :], dim=-1, keepdim=True)
        projection = torch.clamp(projection_raw, min=0.0)
        removed = float(alpha) * projection * unit_direction[None, :]
        attn_new = attn_output.clone()
        attn_new[:, -1, :] = attn_new[:, -1, :] - removed.to(device=attn_output.device, dtype=attn_output.dtype)
        hook_state.update(
            {
                "applied": True,
                "hidden_device": str(attn_output.device),
                "hidden_dtype": str(attn_output.dtype),
                "attention_output_norm": float(torch.linalg.norm(h_last[0].detach().float().cpu()).item()),
                "prior_attention_residual_norm": float(direction_norm.detach().cpu().item()),
                "projection_coeff_raw": float(projection_raw[0, 0].detach().cpu().item()),
                "projection_coeff_positive": float(projection[0, 0].detach().cpu().item()),
                "removed_norm": float(torch.linalg.norm(removed[0].detach().float().cpu()).item()),
                "attention_prior_residual_cosine": float(
                    F.cosine_similarity(h_last[0], direction, dim=0).detach().cpu().item()
                ),
            }
        )
        if rest is None:
            return attn_new
        return (attn_new, *rest)

    handle = self_attn.register_forward_hook(suppress_image_hook)
    try:
        with torch.inference_mode():
            outputs = model(**inputs_image, output_hidden_states=False, use_cache=False)
            steered_logp = torch.log_softmax(outputs.logits[0, -1, :].float(), dim=-1)
    finally:
        handle.remove()
    steered_top_id = int(torch.argmax(steered_logp).detach().cpu().item())
    hook_state.update(
        {
            "steered_top_token_id": steered_top_id,
            "prior_top_logp_before": float(image_logp[prior_top_id].detach().cpu().item()),
            "prior_top_logp_after": float(steered_logp[prior_top_id].detach().cpu().item()),
            "prior_top_logp_delta": float((steered_logp[prior_top_id] - image_logp[prior_top_id]).detach().cpu().item()),
        }
    )
    del outputs
    return steered_logp.to(device=image_logp.device), hook_state


def _layer_attention_prior_head_steered_logp(
    model: Any,
    inputs_image: Dict[str, Any],
    inputs_prior: Dict[str, Any],
    image_logp: Any,
    prior_logp: Any,
    layer_index: int,
    alpha: float,
    head_top_k: int,
) -> Tuple[Any, Dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    if float(alpha) == 0.0:
        return image_logp, {"applied": False, "reason": "zero_alpha", "attention_prior_alpha": 0.0}

    layers = _get_decoder_layers(model)
    if int(layer_index) < 0 or int(layer_index) >= len(layers):
        raise ValueError(f"layer index out of range: {layer_index}")
    target_layer = layers[int(layer_index)]
    self_attn = getattr(target_layer, "self_attn", None)
    o_proj = getattr(self_attn, "o_proj", None) if self_attn is not None else None
    if self_attn is None or o_proj is None:
        return image_logp, {
            "applied": False,
            "reason": "missing_self_attention_o_proj",
            "attention_prior_alpha": float(alpha),
            "layer_index": int(layer_index),
        }

    num_heads = int(getattr(getattr(self_attn, "config", None), "num_attention_heads", 0) or 0)
    head_dim = int(getattr(self_attn, "head_dim", 0) or 0)
    if num_heads <= 0 or head_dim <= 0:
        return image_logp, {
            "applied": False,
            "reason": "missing_head_shape",
            "attention_prior_alpha": float(alpha),
            "layer_index": int(layer_index),
        }

    prior_top_id = int(torch.argmax(prior_logp).detach().cpu().item())
    top_n_heads = max(1, min(int(head_top_k), num_heads))
    hook_state: Dict[str, Any] = {
        "type": "layer_attention_prior_head_suppression",
        "layer_index": int(layer_index),
        "attention_prior_alpha": float(alpha),
        "head_top_k": int(top_n_heads),
        "num_heads": int(num_heads),
        "head_dim": int(head_dim),
        "prior_top_token_id": prior_top_id,
        "prior_top_probability": float(torch.exp(prior_logp[prior_top_id]).detach().cpu().item()),
        "applied": False,
    }
    captured: Dict[str, Any] = {}

    def capture_prior_pre_hook(_module: Any, module_inputs: Tuple[Any, ...]) -> None:
        if not module_inputs:
            return None
        o_proj_input = module_inputs[0]
        if hasattr(o_proj_input, "dim") and o_proj_input.dim() >= 3:
            captured["prior_o_proj_input"] = o_proj_input[:, -1, :].detach().float()
        return None

    handle = o_proj.register_forward_pre_hook(capture_prior_pre_hook)
    try:
        with torch.inference_mode():
            prior_outputs = model(**inputs_prior, output_hidden_states=False, use_cache=False)
    finally:
        handle.remove()
    del prior_outputs

    prior_o_proj_input = captured.get("prior_o_proj_input")
    if prior_o_proj_input is None:
        return image_logp, {
            **hook_state,
            "applied": False,
            "reason": "missing_prior_o_proj_input",
        }
    if int(prior_o_proj_input.shape[-1]) != num_heads * head_dim:
        return image_logp, {
            **hook_state,
            "applied": False,
            "reason": "head_shape_mismatch",
            "o_proj_input_dim": int(prior_o_proj_input.shape[-1]),
        }

    prior_heads_base = prior_o_proj_input[0].reshape(num_heads, head_dim).float()

    def suppress_image_pre_hook(_module: Any, module_inputs: Tuple[Any, ...]) -> Tuple[Any, ...]:
        if not module_inputs:
            return module_inputs
        o_proj_input = module_inputs[0]
        if not hasattr(o_proj_input, "dim") or o_proj_input.dim() < 3:
            return module_inputs
        if int(o_proj_input.shape[-1]) != num_heads * head_dim:
            return module_inputs

        prior_heads = prior_heads_base.to(device=o_proj_input.device, dtype=torch.float32)
        direction_norm = torch.linalg.norm(prior_heads, dim=-1).clamp_min(1e-12)
        unit_direction = prior_heads / direction_norm[:, None]
        h_last = o_proj_input[:, -1, :].float().reshape(o_proj_input.shape[0], num_heads, head_dim)
        projection_raw = torch.sum(h_last * unit_direction[None, :, :], dim=-1)
        projection = torch.clamp(projection_raw, min=0.0)
        head_scores = projection[0] * direction_norm
        selected_scores, selected_heads = torch.topk(head_scores, k=top_n_heads)
        mask = torch.zeros(num_heads, device=o_proj_input.device, dtype=torch.float32)
        mask[selected_heads] = 1.0
        removed = float(alpha) * projection[:, :, None] * unit_direction[None, :, :] * mask[None, :, None]
        h_new = h_last - removed
        output_new = o_proj_input.clone()
        output_new[:, -1, :] = h_new.reshape(o_proj_input.shape[0], num_heads * head_dim).to(
            device=o_proj_input.device,
            dtype=o_proj_input.dtype,
        )
        selected_head_list = [int(x) for x in selected_heads.detach().cpu().tolist()]
        selected_score_list = [float(x) for x in selected_scores.detach().float().cpu().tolist()]
        selected_projection_list = [
            float(projection[0, idx].detach().cpu().item()) for idx in selected_head_list
        ]
        selected_removed_norms = [
            float(torch.linalg.norm(removed[0, idx].detach().float().cpu()).item()) for idx in selected_head_list
        ]
        cosine_values = F.cosine_similarity(h_last[0], prior_heads, dim=-1)
        hook_state.update(
            {
                "applied": True,
                "hidden_device": str(o_proj_input.device),
                "hidden_dtype": str(o_proj_input.dtype),
                "selected_heads": selected_head_list,
                "selected_head_scores": selected_score_list,
                "selected_projection_coeff_positive": selected_projection_list,
                "selected_removed_norms": selected_removed_norms,
                "removed_norm": float(torch.linalg.norm(removed[0].detach().float().cpu()).item()),
                "o_proj_input_norm": float(torch.linalg.norm(h_last[0].detach().float().cpu()).item()),
                "prior_o_proj_input_norm": float(torch.linalg.norm(prior_heads.detach().float().cpu()).item()),
                "mean_head_prior_cosine": float(cosine_values.detach().float().mean().cpu().item()),
                "max_head_prior_cosine": float(cosine_values.detach().float().max().cpu().item()),
            }
        )
        return (output_new, *module_inputs[1:])

    handle = o_proj.register_forward_pre_hook(suppress_image_pre_hook)
    try:
        with torch.inference_mode():
            outputs = model(**inputs_image, output_hidden_states=False, use_cache=False)
            steered_logp = torch.log_softmax(outputs.logits[0, -1, :].float(), dim=-1)
    finally:
        handle.remove()
    steered_top_id = int(torch.argmax(steered_logp).detach().cpu().item())
    hook_state.update(
        {
            "steered_top_token_id": steered_top_id,
            "prior_top_logp_before": float(image_logp[prior_top_id].detach().cpu().item()),
            "prior_top_logp_after": float(steered_logp[prior_top_id].detach().cpu().item()),
            "prior_top_logp_delta": float((steered_logp[prior_top_id] - image_logp[prior_top_id]).detach().cpu().item()),
        }
    )
    del outputs
    return steered_logp.to(device=image_logp.device), hook_state


def _layer_attention_visual_delta_steered_logp(
    model: Any,
    inputs_image: Dict[str, Any],
    inputs_prior: Dict[str, Any],
    image_logp: Any,
    prior_logp: Any,
    layer_index: int,
    alpha: float,
    head_top_k: int,
) -> Tuple[Any, Dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    if float(alpha) == 0.0:
        return image_logp, {"applied": False, "reason": "zero_alpha", "attention_visual_alpha": 0.0}

    layers = _get_decoder_layers(model)
    if int(layer_index) < 0 or int(layer_index) >= len(layers):
        raise ValueError(f"layer index out of range: {layer_index}")
    target_layer = layers[int(layer_index)]
    self_attn = getattr(target_layer, "self_attn", None)
    o_proj = getattr(self_attn, "o_proj", None) if self_attn is not None else None
    if self_attn is None or o_proj is None:
        return image_logp, {
            "applied": False,
            "reason": "missing_self_attention_o_proj",
            "attention_visual_alpha": float(alpha),
            "layer_index": int(layer_index),
        }

    num_heads = int(getattr(getattr(self_attn, "config", None), "num_attention_heads", 0) or 0)
    head_dim = int(getattr(self_attn, "head_dim", 0) or 0)
    if num_heads <= 0 or head_dim <= 0:
        return image_logp, {
            "applied": False,
            "reason": "missing_head_shape",
            "attention_visual_alpha": float(alpha),
            "layer_index": int(layer_index),
        }

    prior_top_id = int(torch.argmax(prior_logp).detach().cpu().item())
    top_n_heads = max(1, min(int(head_top_k), num_heads))
    hook_state: Dict[str, Any] = {
        "type": "layer_attention_visual_delta_boost",
        "layer_index": int(layer_index),
        "attention_visual_alpha": float(alpha),
        "head_top_k": int(top_n_heads),
        "num_heads": int(num_heads),
        "head_dim": int(head_dim),
        "prior_top_token_id": prior_top_id,
        "prior_top_probability": float(torch.exp(prior_logp[prior_top_id]).detach().cpu().item()),
        "applied": False,
    }
    captured: Dict[str, Any] = {}

    def capture_prior_pre_hook(_module: Any, module_inputs: Tuple[Any, ...]) -> None:
        if not module_inputs:
            return None
        o_proj_input = module_inputs[0]
        if hasattr(o_proj_input, "dim") and o_proj_input.dim() >= 3:
            captured["prior_o_proj_input"] = o_proj_input[:, -1, :].detach().float()
        return None

    handle = o_proj.register_forward_pre_hook(capture_prior_pre_hook)
    try:
        with torch.inference_mode():
            prior_outputs = model(**inputs_prior, output_hidden_states=False, use_cache=False)
    finally:
        handle.remove()
    del prior_outputs

    prior_o_proj_input = captured.get("prior_o_proj_input")
    if prior_o_proj_input is None:
        return image_logp, {
            **hook_state,
            "applied": False,
            "reason": "missing_prior_o_proj_input",
        }
    if int(prior_o_proj_input.shape[-1]) != num_heads * head_dim:
        return image_logp, {
            **hook_state,
            "applied": False,
            "reason": "head_shape_mismatch",
            "o_proj_input_dim": int(prior_o_proj_input.shape[-1]),
        }

    prior_heads_base = prior_o_proj_input[0].reshape(num_heads, head_dim).float()

    def boost_image_pre_hook(_module: Any, module_inputs: Tuple[Any, ...]) -> Tuple[Any, ...]:
        if not module_inputs:
            return module_inputs
        o_proj_input = module_inputs[0]
        if not hasattr(o_proj_input, "dim") or o_proj_input.dim() < 3:
            return module_inputs
        if int(o_proj_input.shape[-1]) != num_heads * head_dim:
            return module_inputs

        prior_heads = prior_heads_base.to(device=o_proj_input.device, dtype=torch.float32)
        image_heads = o_proj_input[:, -1, :].float().reshape(o_proj_input.shape[0], num_heads, head_dim)
        visual_delta = image_heads - prior_heads[None, :, :]
        delta_norm = torch.linalg.norm(visual_delta[0], dim=-1)
        selected_scores, selected_heads = torch.topk(delta_norm, k=top_n_heads)
        mask = torch.zeros(num_heads, device=o_proj_input.device, dtype=torch.float32)
        mask[selected_heads] = 1.0
        added = float(alpha) * visual_delta * mask[None, :, None]
        boosted_heads = image_heads + added
        output_new = o_proj_input.clone()
        output_new[:, -1, :] = boosted_heads.reshape(o_proj_input.shape[0], num_heads * head_dim).to(
            device=o_proj_input.device,
            dtype=o_proj_input.dtype,
        )
        selected_head_list = [int(x) for x in selected_heads.detach().cpu().tolist()]
        selected_score_list = [float(x) for x in selected_scores.detach().float().cpu().tolist()]
        selected_added_norms = [
            float(torch.linalg.norm(added[0, idx].detach().float().cpu()).item()) for idx in selected_head_list
        ]
        cosine_values = F.cosine_similarity(image_heads[0], prior_heads, dim=-1)
        hook_state.update(
            {
                "applied": True,
                "hidden_device": str(o_proj_input.device),
                "hidden_dtype": str(o_proj_input.dtype),
                "selected_heads": selected_head_list,
                "selected_delta_norms": selected_score_list,
                "selected_added_norms": selected_added_norms,
                "added_norm": float(torch.linalg.norm(added[0].detach().float().cpu()).item()),
                "visual_delta_norm": float(torch.linalg.norm(visual_delta[0].detach().float().cpu()).item()),
                "o_proj_input_norm": float(torch.linalg.norm(image_heads[0].detach().float().cpu()).item()),
                "prior_o_proj_input_norm": float(torch.linalg.norm(prior_heads.detach().float().cpu()).item()),
                "mean_head_image_prior_cosine": float(cosine_values.detach().float().mean().cpu().item()),
                "min_head_image_prior_cosine": float(cosine_values.detach().float().min().cpu().item()),
            }
        )
        return (output_new, *module_inputs[1:])

    handle = o_proj.register_forward_pre_hook(boost_image_pre_hook)
    try:
        with torch.inference_mode():
            outputs = model(**inputs_image, output_hidden_states=False, use_cache=False)
            steered_logp = torch.log_softmax(outputs.logits[0, -1, :].float(), dim=-1)
    finally:
        handle.remove()
    steered_top_id = int(torch.argmax(steered_logp).detach().cpu().item())
    hook_state.update(
        {
            "steered_top_token_id": steered_top_id,
            "prior_top_logp_before": float(image_logp[prior_top_id].detach().cpu().item()),
            "prior_top_logp_after": float(steered_logp[prior_top_id].detach().cpu().item()),
            "prior_top_logp_delta": float((steered_logp[prior_top_id] - image_logp[prior_top_id]).detach().cpu().item()),
        }
    )
    del outputs
    return steered_logp.to(device=image_logp.device), hook_state


def _layer_image_attention_boost_steered_logp(
    model: Any,
    inputs_image: Dict[str, Any],
    image_logp: Any,
    prior_logp: Any,
    layer_index: int,
    alpha: float,
    head_top_k: int,
    head_select: str,
    text_alpha: float = 0.0,
    text_top_k: int = 0,
) -> Tuple[Any, Dict[str, Any]]:
    import types

    import torch
    import torch.nn.functional as F

    if float(alpha) == 0.0 and float(text_alpha) == 0.0:
        return image_logp, {
            "applied": False,
            "reason": "zero_alpha",
            "image_attention_alpha": 0.0,
            "text_attention_alpha": 0.0,
        }

    layers = _get_decoder_layers(model)
    if int(layer_index) < 0 or int(layer_index) >= len(layers):
        raise ValueError(f"layer index out of range: {layer_index}")
    target_layer = layers[int(layer_index)]
    self_attn = getattr(target_layer, "self_attn", None)
    if self_attn is None:
        return image_logp, {
            "applied": False,
            "reason": "missing_self_attention_module",
            "image_attention_alpha": float(alpha),
            "layer_index": int(layer_index),
        }

    input_ids = inputs_image.get("input_ids")
    image_token_id = getattr(getattr(model, "config", None), "image_token_id", None)
    if input_ids is None or image_token_id is None:
        return image_logp, {
            "applied": False,
            "reason": "missing_input_ids_or_image_token_id",
            "image_attention_alpha": float(alpha),
            "layer_index": int(layer_index),
        }
    image_positions_base = (input_ids[0] == int(image_token_id)).nonzero(as_tuple=False).flatten()
    if image_positions_base.numel() == 0:
        return image_logp, {
            "applied": False,
            "reason": "no_image_tokens",
            "image_attention_alpha": float(alpha),
            "layer_index": int(layer_index),
            "image_token_id": int(image_token_id),
        }

    original_forward = self_attn.forward
    original_func = getattr(original_forward, "__func__", original_forward)
    forward_globals = getattr(original_func, "__globals__", {})
    apply_rotary_pos_emb_fn = forward_globals.get("apply_rotary_pos_emb")
    repeat_kv_fn = forward_globals.get("repeat_kv")
    if apply_rotary_pos_emb_fn is None or repeat_kv_fn is None:
        try:
            from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb, repeat_kv
        except Exception as e:
            return image_logp, {
                "applied": False,
                "reason": f"missing_qwen3_vl_attention_helpers: {e}",
                "image_attention_alpha": float(alpha),
                "layer_index": int(layer_index),
            }
        apply_rotary_pos_emb_fn = apply_rotary_pos_emb
        repeat_kv_fn = repeat_kv

    head_select = str(head_select or "low_visual").strip().lower()
    if head_select not in {"low_visual", "high_visual", "high_text", "all"}:
        raise ValueError(f"unknown image-attention head selection: {head_select}")
    prior_top_id = int(torch.argmax(prior_logp).detach().cpu().item())
    hook_state: Dict[str, Any] = {
        "type": "layer_image_text_route_attention_reallocation"
        if float(text_alpha) != 0.0
        else "layer_image_token_attention_boost",
        "layer_index": int(layer_index),
        "image_attention_alpha": float(alpha),
        "text_attention_alpha": float(text_alpha),
        "text_attention_top_k": int(text_top_k),
        "head_top_k": int(head_top_k),
        "head_select": head_select,
        "image_token_id": int(image_token_id),
        "num_image_tokens": int(image_positions_base.numel()),
        "image_token_start": int(image_positions_base[0].detach().cpu().item()),
        "image_token_end": int(image_positions_base[-1].detach().cpu().item()),
        "prior_top_token_id": prior_top_id,
        "prior_top_probability": float(torch.exp(prior_logp[prior_top_id]).detach().cpu().item()),
        "applied": False,
    }

    def boosted_forward(
        module_self: Any,
        hidden_states: Any,
        position_embeddings: Tuple[Any, Any],
        attention_mask: Optional[Any],
        past_key_values: Optional[Any] = None,
        cache_position: Optional[Any] = None,
        **kwargs: Any,
    ) -> Tuple[Any, Optional[Any]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, module_self.head_dim)

        query_states = module_self.q_norm(module_self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = module_self.k_norm(module_self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = module_self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_fn(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                module_self.layer_idx,
                cache_kwargs,
            )

        key_states = repeat_kv_fn(key_states, module_self.num_key_value_groups)
        value_states = repeat_kv_fn(value_states, module_self.num_key_value_groups)
        attn_logits = torch.matmul(query_states, key_states.transpose(2, 3)) * module_self.scaling
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_logits = attn_logits + causal_mask

        image_positions = image_positions_base.to(device=attn_logits.device)
        image_positions = image_positions[image_positions < attn_logits.shape[-1]]
        if image_positions.numel() == 0:
            hook_state.update({"applied": False, "reason": "image_positions_outside_attention_source"})
            attn_weights = F.softmax(attn_logits, dim=-1, dtype=torch.float32).to(query_states.dtype)
        else:
            base_weights = F.softmax(attn_logits, dim=-1, dtype=torch.float32).to(query_states.dtype)
            source_positions = torch.arange(attn_logits.shape[-1], device=attn_logits.device)
            valid_source_mask = torch.isfinite(attn_logits[0, 0, -1, :])
            image_source_mask = torch.zeros_like(valid_source_mask, dtype=torch.bool)
            image_source_mask[image_positions] = True
            text_positions = source_positions[valid_source_mask & ~image_source_mask]
            visual_mass_before = base_weights[0, :, -1, image_positions].sum(dim=-1).float()
            text_mass_before = (
                base_weights[0, :, -1, text_positions].sum(dim=-1).float()
                if text_positions.numel() > 0
                else torch.zeros_like(visual_mass_before)
            )
            num_heads = int(visual_mass_before.shape[0])
            if head_select == "all" or int(head_top_k) <= 0 or int(head_top_k) >= num_heads:
                selected_heads = torch.arange(num_heads, device=attn_logits.device)
            elif head_select == "low_visual":
                _, selected_heads = torch.topk(visual_mass_before, k=max(1, int(head_top_k)), largest=False)
            elif head_select == "high_text":
                _, selected_heads = torch.topk(text_mass_before, k=max(1, int(head_top_k)), largest=True)
            else:
                _, selected_heads = torch.topk(visual_mass_before, k=max(1, int(head_top_k)), largest=True)

            boosted_logits = attn_logits.clone()
            boosted_last = boosted_logits[:, selected_heads, -1, :].clone()
            if float(alpha) != 0.0:
                boosted_last[:, :, image_positions] = boosted_last[:, :, image_positions] + float(alpha)
            suppressed_text_mask = torch.zeros(
                (int(selected_heads.numel()), int(attn_logits.shape[-1])),
                device=attn_logits.device,
                dtype=torch.bool,
            )
            if float(text_alpha) != 0.0 and text_positions.numel() > 0:
                selected_text_weights = base_weights[0, selected_heads, -1, :][:, text_positions].float()
                if int(text_top_k) > 0 and int(text_top_k) < int(text_positions.numel()):
                    top_n_text = max(1, int(text_top_k))
                    for row_idx in range(int(selected_heads.numel())):
                        _, local_top = torch.topk(selected_text_weights[row_idx], k=top_n_text, largest=True)
                        suppress_positions = text_positions[local_top]
                        suppressed_text_mask[row_idx, suppress_positions] = True
                        boosted_last[:, row_idx, suppress_positions] = (
                            boosted_last[:, row_idx, suppress_positions] - float(text_alpha)
                        )
                else:
                    suppressed_text_mask[:, text_positions] = True
                    boosted_last[:, :, text_positions] = boosted_last[:, :, text_positions] - float(text_alpha)
            boosted_logits[:, selected_heads, -1, :] = boosted_last
            attn_weights = F.softmax(boosted_logits, dim=-1, dtype=torch.float32).to(query_states.dtype)
            visual_mass_after = attn_weights[0, :, -1, image_positions].sum(dim=-1).float()
            text_mass_after = (
                attn_weights[0, :, -1, text_positions].sum(dim=-1).float()
                if text_positions.numel() > 0
                else torch.zeros_like(visual_mass_before)
            )
            selected_head_list = [int(x) for x in selected_heads.detach().cpu().tolist()]
            selected_before_weights = base_weights[0, selected_heads, -1, :].float()
            selected_after_weights = attn_weights[0, selected_heads, -1, :].float()
            suppressed_mass_before = (selected_before_weights * suppressed_text_mask.float()).sum(dim=-1)
            suppressed_mass_after = (selected_after_weights * suppressed_text_mask.float()).sum(dim=-1)
            hook_state.update(
                {
                    "applied": True,
                    "hidden_device": str(hidden_states.device),
                    "hidden_dtype": str(hidden_states.dtype),
                    "num_heads": int(num_heads),
                    "selected_heads": selected_head_list,
                    "selected_visual_mass_before": [
                        float(visual_mass_before[idx].detach().cpu().item()) for idx in selected_head_list
                    ],
                    "selected_visual_mass_after": [
                        float(visual_mass_after[idx].detach().cpu().item()) for idx in selected_head_list
                    ],
                    "selected_text_mass_before": [
                        float(text_mass_before[idx].detach().cpu().item()) for idx in selected_head_list
                    ],
                    "selected_text_mass_after": [
                        float(text_mass_after[idx].detach().cpu().item()) for idx in selected_head_list
                    ],
                    "mean_visual_mass_before": float(visual_mass_before.detach().mean().cpu().item()),
                    "mean_visual_mass_after": float(visual_mass_after.detach().mean().cpu().item()),
                    "mean_text_mass_before": float(text_mass_before.detach().mean().cpu().item()),
                    "mean_text_mass_after": float(text_mass_after.detach().mean().cpu().item()),
                    "selected_mean_visual_mass_before": float(
                        visual_mass_before[selected_heads].detach().mean().cpu().item()
                    ),
                    "selected_mean_visual_mass_after": float(
                        visual_mass_after[selected_heads].detach().mean().cpu().item()
                    ),
                    "selected_mean_text_mass_before": float(
                        text_mass_before[selected_heads].detach().mean().cpu().item()
                    ),
                    "selected_mean_text_mass_after": float(
                        text_mass_after[selected_heads].detach().mean().cpu().item()
                    ),
                    "selected_suppressed_text_mass_before": float(
                        suppressed_mass_before.detach().mean().cpu().item()
                    ),
                    "selected_suppressed_text_mass_after": float(
                        suppressed_mass_after.detach().mean().cpu().item()
                    ),
                    "max_visual_mass_before": float(visual_mass_before.detach().max().cpu().item()),
                    "max_visual_mass_after": float(visual_mass_after.detach().max().cpu().item()),
                    "max_text_mass_before": float(text_mass_before.detach().max().cpu().item()),
                    "max_text_mass_after": float(text_mass_after.detach().max().cpu().item()),
                    "visual_mass_delta_mean": float(
                        (visual_mass_after - visual_mass_before).detach().mean().cpu().item()
                    ),
                    "text_mass_delta_mean": float(
                        (text_mass_after - text_mass_before).detach().mean().cpu().item()
                    ),
                    "num_text_tokens_in_attention": int(text_positions.numel()),
                    "num_suppressed_text_positions_per_head": [
                        int(suppressed_text_mask[row_idx].sum().detach().cpu().item())
                        for row_idx in range(int(selected_heads.numel()))
                    ][:16],
                    "valid_image_tokens_in_attention": int(image_positions.numel()),
                }
            )

        attn_weights = F.dropout(
            attn_weights,
            p=0.0 if not module_self.training else module_self.attention_dropout,
            training=module_self.training,
        )
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = module_self.o_proj(attn_output)
        return attn_output, attn_weights

    self_attn.forward = types.MethodType(boosted_forward, self_attn)
    try:
        with torch.inference_mode():
            outputs = model(**inputs_image, output_hidden_states=False, use_cache=False)
            steered_logp = torch.log_softmax(outputs.logits[0, -1, :].float(), dim=-1)
    finally:
        self_attn.forward = original_forward
    steered_top_id = int(torch.argmax(steered_logp).detach().cpu().item())
    hook_state.update(
        {
            "steered_top_token_id": steered_top_id,
            "prior_top_logp_before": float(image_logp[prior_top_id].detach().cpu().item()),
            "prior_top_logp_after": float(steered_logp[prior_top_id].detach().cpu().item()),
            "prior_top_logp_delta": float((steered_logp[prior_top_id] - image_logp[prior_top_id]).detach().cpu().item()),
        }
    )
    del outputs
    return steered_logp.to(device=image_logp.device), hook_state


def _hidden_delta_probe(
    model: Any,
    processor: Any,
    user_text: str,
    image_a: Optional[Any],
    image_b: Optional[Any],
    probe_type: str,
    label_a: str,
    label_b: str,
) -> Dict[str, Any]:
    import torch
    import torch.nn.functional as F

    inputs_a = _qwen3_vl_inputs(model, processor, user_text, image_a)
    inputs_b = _qwen3_vl_inputs(model, processor, user_text, image_b)
    with torch.inference_mode():
        outputs_a = model(**inputs_a, output_hidden_states=True, use_cache=False)
        outputs_b = model(**inputs_b, output_hidden_states=True, use_cache=False)

    orthogonal_summary, _ = _orthogonal_visual_residual_from_outputs(outputs_a, outputs_b)
    states_a = getattr(outputs_a, "hidden_states", None) or []
    states_b = getattr(outputs_b, "hidden_states", None) or []
    num_layers = min(len(states_a), len(states_b))
    layers: List[Dict[str, Any]] = []
    max_delta_layer: Optional[Dict[str, Any]] = None
    for layer_idx in _probe_layer_indices(num_layers):
        vec_a = states_a[layer_idx][0, -1, :].detach().float().cpu()
        vec_b = states_b[layer_idx][0, -1, :].detach().float().cpu()
        delta = vec_a - vec_b
        norm_a = float(torch.linalg.norm(vec_a).item())
        norm_b = float(torch.linalg.norm(vec_b).item())
        delta_norm = float(torch.linalg.norm(delta).item())
        cosine = float(F.cosine_similarity(vec_a, vec_b, dim=0).item())
        row = {
            "layer": int(layer_idx),
            f"{label_a}_norm": norm_a,
            f"{label_b}_norm": norm_b,
            "delta_norm": delta_norm,
            f"{label_a}_{label_b}_cosine": cosine,
            "relative_delta": float(delta_norm / norm_a) if norm_a else None,
        }
        layers.append(row)
        if max_delta_layer is None or delta_norm > float(max_delta_layer.get("delta_norm") or 0.0):
            max_delta_layer = row

    probe = {
        "type": probe_type,
        "num_layers": num_layers,
        f"{label_a}_prompt_tokens": int(inputs_a["input_ids"].shape[-1]) if "input_ids" in inputs_a else None,
        f"{label_b}_prompt_tokens": int(inputs_b["input_ids"].shape[-1]) if "input_ids" in inputs_b else None,
        "layers": layers,
        "max_delta_layer": max_delta_layer,
        "orthogonal_visual": orthogonal_summary,
    }
    del outputs_a, outputs_b, states_a, states_b
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return probe


def _visual_delta_probe(model: Any, processor: Any, user_text: str, image: Any) -> Dict[str, Any]:
    return _hidden_delta_probe(
        model=model,
        processor=processor,
        user_text=user_text,
        image_a=image,
        image_b=None,
        probe_type="image_delta",
        label_a="image",
        label_b="no_image",
    )


def _candidate_hidden_delta_qwen3_vl(
    model: Any,
    processor: Any,
    user_text: str,
    image: Any,
    candidate_text: str,
) -> Dict[str, Any]:
    probe = _hidden_delta_probe(
        model=model,
        processor=processor,
        user_text=f"{user_text}{candidate_text}",
        image_a=image,
        image_b=None,
        probe_type="candidate_image_delta",
        label_a="image",
        label_b="no_image",
    )
    layers = probe.get("layers") or []
    relative_values = [
        float(row.get("relative_delta"))
        for row in layers
        if isinstance(row.get("relative_delta"), (int, float))
    ]
    delta_values = [
        float(row.get("delta_norm"))
        for row in layers
        if isinstance(row.get("delta_norm"), (int, float))
    ]
    probe["max_relative_delta"] = max(relative_values) if relative_values else None
    probe["mean_relative_delta"] = (sum(relative_values) / len(relative_values)) if relative_values else None
    probe["max_delta_norm"] = max(delta_values) if delta_values else None
    orthogonal = probe.get("orthogonal_visual") or {}
    if isinstance(orthogonal, dict):
        probe["relative_pure_visual"] = orthogonal.get("relative_pure_visual")
        probe["pure_visual_ratio"] = orthogonal.get("pure_visual_ratio")
        probe["image_prior_cosine"] = orthogonal.get("image_prior_cosine")
    return probe


def _candidate_logprob_qwen3_vl(
    model: Any,
    processor: Any,
    user_text: str,
    image: Optional[Any],
    candidate_text: str,
    assistant_prefix: str = "",
    family: str = "qwen3_vl",
) -> Dict[str, Any]:
    import torch

    tokenizer = getattr(processor, "tokenizer", processor)
    prompt_inputs = _candidate_vlm_inputs(model, processor, family, user_text, image)
    input_device = prompt_inputs["input_ids"].device
    prefix_ids = tokenizer(
        assistant_prefix,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].to(input_device)
    cand_ids = tokenizer(
        candidate_text,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].to(input_device)
    if cand_ids.shape[-1] == 0:
        raise ValueError("empty candidate tokenization")

    prompt_ids = prompt_inputs["input_ids"]
    continuation_ids = torch.cat([prefix_ids, cand_ids], dim=-1)
    full_inputs: Dict[str, Any] = {}
    for key, value in prompt_inputs.items():
        if key == "input_ids":
            full_inputs[key] = torch.cat([prompt_ids, continuation_ids], dim=-1)
        elif key == "attention_mask":
            extension = torch.ones(
                (value.shape[0], continuation_ids.shape[-1]),
                dtype=value.dtype,
                device=value.device,
            )
            full_inputs[key] = torch.cat([value, extension], dim=-1)
        else:
            full_inputs[key] = value

    with torch.inference_mode():
        outputs = model(**full_inputs, use_cache=False)
        logits = outputs.logits
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        start = int(prompt_ids.shape[-1] + prefix_ids.shape[-1])
        token_logprobs = []
        for idx, token_id in enumerate(cand_ids[0]):
            pos = start + idx - 1
            token_logprobs.append(float(log_probs[0, pos, int(token_id)].detach().cpu().item()))

    total = float(sum(token_logprobs))
    count = len(token_logprobs)
    return {
        "text": candidate_text,
        "token_ids": [int(x) for x in cand_ids[0].detach().cpu().tolist()],
        "token_logprobs": token_logprobs,
        "logprob": total,
        "avg_logprob": total / count if count else None,
        "token_count": count,
    }


def _candidate_logprobs_shared_prefix(
    model: Any,
    processor: Any,
    user_text: str,
    image: Optional[Any],
    candidate_texts: Sequence[str],
    family: str = "qwen3_vl",
) -> List[Dict[str, Any]]:
    """Score one-token branches after a shared token prefix in one forward pass."""
    import torch

    tokenizer = getattr(processor, "tokenizer", processor)
    token_ids = [
        tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        for text in candidate_texts
    ]
    if not token_ids or any(ids.numel() == 0 for ids in token_ids):
        raise ValueError("empty candidate tokenization")

    common_length = 0
    shortest = min(int(ids.numel()) for ids in token_ids)
    while common_length < shortest:
        value = int(token_ids[0][common_length])
        if any(int(ids[common_length]) != value for ids in token_ids[1:]):
            break
        common_length += 1

    if any(int(ids.numel()) != common_length + 1 for ids in token_ids):
        return [
            _candidate_logprob_qwen3_vl(
                model, processor, user_text, image, text, family=family
            )
            for text in candidate_texts
        ]

    prompt_inputs = _candidate_vlm_inputs(model, processor, family, user_text, image)
    prompt_ids = prompt_inputs["input_ids"]
    device = prompt_ids.device
    prefix_ids = token_ids[0][:common_length].unsqueeze(0).to(device)
    full_inputs: Dict[str, Any] = {}
    for key, value in prompt_inputs.items():
        if key == "input_ids":
            full_inputs[key] = torch.cat([prompt_ids, prefix_ids], dim=-1)
        elif key == "attention_mask" and common_length:
            extension = torch.ones(
                (value.shape[0], common_length), dtype=value.dtype, device=value.device
            )
            full_inputs[key] = torch.cat([value, extension], dim=-1)
        else:
            full_inputs[key] = value

    with torch.inference_mode():
        outputs = model(**full_inputs, use_cache=False)
        log_probs = torch.log_softmax(outputs.logits.float(), dim=-1)

    prompt_length = int(prompt_ids.shape[-1])
    shared_logprobs = [
        float(
            log_probs[0, prompt_length + index - 1, int(prefix_ids[0, index])]
            .detach()
            .cpu()
            .item()
        )
        for index in range(common_length)
    ]
    branch_position = prompt_length + common_length - 1
    scores: List[Dict[str, Any]] = []
    for text, ids in zip(candidate_texts, token_ids):
        branch_id = int(ids[common_length])
        branch_logprob = float(
            log_probs[0, branch_position, branch_id].detach().cpu().item()
        )
        values = [*shared_logprobs, branch_logprob]
        total = float(sum(values))
        scores.append(
            {
                "text": text,
                "token_ids": [int(value) for value in ids.tolist()],
                "token_logprobs": values,
                "logprob": total,
                "avg_logprob": total / len(values),
                "token_count": len(values),
            }
        )
    return scores


def _softmax_over(values: List[float]) -> List[float]:
    import math

    if not values:
        return []
    m = max(values)
    exps = [math.exp(v - m) for v in values]
    z = sum(exps)
    return [v / z for v in exps] if z else [0.0 for _ in values]


def _entropy(probs: List[float]) -> Optional[float]:
    import math

    if not probs:
        return None
    return float(-sum(p * math.log(max(p, 1e-12)) for p in probs))


def _margin(sorted_rows: List[Dict[str, Any]], field: str) -> Optional[float]:
    if len(sorted_rows) < 2:
        return None
    top = sorted_rows[0].get(field)
    second = sorted_rows[1].get(field)
    if isinstance(top, (int, float)) and isinstance(second, (int, float)):
        return float(top - second)
    return None


def _minmax_unit(values: List[Optional[float]]) -> List[float]:
    numeric = [float(v) for v in values if isinstance(v, (int, float))]
    if not numeric:
        return [0.0 for _ in values]
    lo = min(numeric)
    hi = max(numeric)
    if hi <= lo:
        return [0.0 for _ in values]
    out: List[float] = []
    for value in values:
        if isinstance(value, (int, float)):
            out.append(float((float(value) - lo) / (hi - lo)))
        else:
            out.append(0.0)
    return out


def _append_token_to_inputs(inputs: Dict[str, Any], token_id: int) -> Dict[str, Any]:
    import torch

    out: Dict[str, Any] = {}
    for key, value in inputs.items():
        if key == "input_ids":
            token = torch.tensor([[int(token_id)]], dtype=value.dtype, device=value.device)
            out[key] = torch.cat([value, token], dim=-1)
        elif key == "attention_mask":
            token = torch.ones((value.shape[0], 1), dtype=value.dtype, device=value.device)
            out[key] = torch.cat([value, token], dim=-1)
        else:
            out[key] = value
    return out


def _top_token_rows(
    tokenizer: Any,
    image_logp: Any,
    prior_logp: Any,
    contrast_logp: Any,
    token_ids: Any,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for tok in token_ids.detach().cpu().tolist():
        tok_id = int(tok)
        rows.append(
            {
                "token_id": tok_id,
                "token": tokenizer.decode([tok_id]),
                "logp_image": float(image_logp[tok_id].detach().cpu().item()),
                "logp_prior": float(prior_logp[tok_id].detach().cpu().item()),
                "score": float(contrast_logp[tok_id].detach().cpu().item()),
            }
        )
    return rows


def _token_margin(rows: List[Dict[str, Any]], field: str) -> Optional[float]:
    if len(rows) < 2:
        return None
    return float(rows[0][field] - rows[1][field])


def _hidden_delta_summary_from_outputs(outputs_a: Any, outputs_b: Any, label_a: str, label_b: str) -> Dict[str, Any]:
    import torch
    import torch.nn.functional as F

    states_a = getattr(outputs_a, "hidden_states", None) or []
    states_b = getattr(outputs_b, "hidden_states", None) or []
    num_layers = min(len(states_a), len(states_b))
    rows: List[Dict[str, Any]] = []
    relative_values: List[float] = []
    delta_values: List[float] = []
    for layer_idx in _probe_layer_indices(num_layers):
        vec_a = states_a[layer_idx][0, -1, :].detach().float().cpu()
        vec_b = states_b[layer_idx][0, -1, :].detach().float().cpu()
        delta = vec_a - vec_b
        norm_a = float(torch.linalg.norm(vec_a).item())
        norm_b = float(torch.linalg.norm(vec_b).item())
        delta_norm = float(torch.linalg.norm(delta).item())
        relative_delta = float(delta_norm / norm_a) if norm_a else None
        if relative_delta is not None:
            relative_values.append(relative_delta)
        delta_values.append(delta_norm)
        rows.append(
            {
                "layer": int(layer_idx),
                f"{label_a}_norm": norm_a,
                f"{label_b}_norm": norm_b,
                "delta_norm": delta_norm,
                "relative_delta": relative_delta,
                f"{label_a}_{label_b}_cosine": float(F.cosine_similarity(vec_a, vec_b, dim=0).item()),
            }
        )
    return {
        "num_layers": num_layers,
        "layers": rows,
        "max_relative_delta": max(relative_values) if relative_values else None,
        "mean_relative_delta": (sum(relative_values) / len(relative_values)) if relative_values else None,
        "max_delta_norm": max(delta_values) if delta_values else None,
    }


def _call_local_qwen3_vl_cp_vbc_token(
    spec: ModelSpec,
    task: str,
    item: Dict[str, Any],
    image_path: str,
    baseline_pred: str,
    lambda_text: float,
    prior_margin_threshold: float,
    contrast_margin_threshold: float,
    risk_mode: str,
    visual_conflict_image_margin: float,
    absorption_image_margin_max: float,
    token_top_k: int,
    trace_tokens_limit: int,
    lambda_policy: str,
    path_lambda_low: float,
    path_lambda_high: float,
    path_lambda_steps: int,
    path_stability_threshold: float,
    path_margin_threshold: float,
    path_prior_relief_min: float,
) -> Tuple[str, Dict[str, Any], int]:
    try:
        import torch
        from PIL import Image
    except Exception as e:
        raise RuntimeError("token cp_vbc requires torch+Pillow in the current Python environment") from e

    model, processor = _local_qwen3_bundle(spec)
    tokenizer = getattr(processor, "tokenizer", processor)
    image = Image.open(image_path).convert("RGB")
    image = image.resize((512, 512), Image.Resampling.LANCZOS)
    prompt = _build_token_cp_vbc_prompt(task, item)
    inputs_image = _qwen3_vl_inputs(model, processor, prompt, image)
    inputs_prior = _qwen3_vl_inputs(model, processor, prompt, None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    eos_ids = set(eos_token_id if isinstance(eos_token_id, list) else [eos_token_id]) if eos_token_id is not None else set()
    generated: List[int] = []
    steps: List[Dict[str, Any]] = []
    risk_count = 0
    override_token_count = 0
    t0 = time.time()
    lambda_policy = str(lambda_policy or CP_VBC_LAMBDA_POLICY_FIXED).strip().lower()
    path_steps = max(2, int(path_lambda_steps))
    path_lambdas = [
        float(path_lambda_low)
        + (float(path_lambda_high) - float(path_lambda_low)) * step / (path_steps - 1)
        for step in range(path_steps)
    ]

    for step_idx in range(max(1, int(spec.max_tokens))):
        with torch.inference_mode():
            image_out = model(**inputs_image, use_cache=False)
            prior_out = model(**inputs_prior, use_cache=False)
            image_logp = torch.log_softmax(image_out.logits[0, -1, :].float(), dim=-1)
            prior_logp = torch.log_softmax(prior_out.logits[0, -1, :].float(), dim=-1)
            static_score = image_logp - float(lambda_text) * prior_logp

        top_k = max(2, min(int(token_top_k), int(image_logp.numel())))
        _, allowed_ids = torch.topk(image_logp, k=top_k)
        allowed = allowed_ids.detach().cpu().tolist()
        allowed_set = set(int(x) for x in allowed)

        image_rows = _top_token_rows(tokenizer, image_logp, prior_logp, image_logp, allowed_ids)
        image_rows.sort(key=lambda r: r["logp_image"], reverse=True)
        prior_rows = sorted(image_rows, key=lambda r: r["logp_prior"], reverse=True)
        contrast_rows = sorted(
            _top_token_rows(tokenizer, image_logp, prior_logp, static_score, allowed_ids),
            key=lambda r: r["score"],
            reverse=True,
        )

        image_top = int(image_rows[0]["token_id"])
        prior_top = int(prior_rows[0]["token_id"])
        static_contrast_top = int(contrast_rows[0]["token_id"])
        image_margin = _token_margin(image_rows, "logp_image")
        prior_margin = _token_margin(prior_rows, "logp_prior")
        static_contrast_margin = _token_margin(contrast_rows, "score")
        path_top = None
        path_margin = None
        path_stability = 0.0
        path_prior_relief = None
        path_probabilities = None
        if lambda_policy == CP_VBC_LAMBDA_POLICY_BAYES_PATH:
            allowed_image_logp = image_logp[allowed_ids]
            allowed_prior_logp = prior_logp[allowed_ids]
            path_probabilities = torch.zeros_like(allowed_image_logp)
            path_top_counts: Dict[int, int] = {}
            for path_lambda in path_lambdas:
                probabilities = torch.softmax(
                    allowed_image_logp - path_lambda * allowed_prior_logp,
                    dim=-1,
                )
                path_probabilities += probabilities / path_steps
                top_position = int(torch.argmax(probabilities).detach().cpu().item())
                top_token = int(allowed_ids[top_position].detach().cpu().item())
                path_top_counts[top_token] = path_top_counts.get(top_token, 0) + 1
            ranked_path = torch.topk(path_probabilities, k=min(2, int(path_probabilities.numel())))
            path_top_position = int(ranked_path.indices[0].detach().cpu().item())
            path_top = int(allowed_ids[path_top_position].detach().cpu().item())
            path_stability = path_top_counts.get(path_top, 0) / path_steps
            if int(ranked_path.values.numel()) > 1:
                path_margin = float((ranked_path.values[0] - ranked_path.values[1]).detach().cpu().item())
            path_prior_relief = float((prior_logp[image_top] - prior_logp[path_top]).detach().cpu().item())

        contrast_top = path_top if lambda_policy == CP_VBC_LAMBDA_POLICY_BAYES_PATH else static_contrast_top
        high_prior = prior_margin is None or prior_margin >= float(prior_margin_threshold)
        if lambda_policy == CP_VBC_LAMBDA_POLICY_BAYES_PATH:
            enough_static_contrast = bool(
                path_stability >= float(path_stability_threshold)
                and (path_margin is None or path_margin >= float(path_margin_threshold))
            )
            enough_prior_relief = bool(
                path_prior_relief is not None and path_prior_relief >= float(path_prior_relief_min)
            )
        else:
            enough_static_contrast = (
                static_contrast_margin is None
                or static_contrast_margin >= float(contrast_margin_threshold)
            )
            enough_prior_relief = True
        visual_conflict_risk = bool(
            image_top != prior_top
            and contrast_top == image_top
            and (image_margin is None or image_margin >= float(visual_conflict_image_margin))
        )
        prior_absorption_risk = bool(
            image_top == prior_top
            and contrast_top != image_top
            and high_prior
            and (image_margin is None or image_margin <= float(absorption_image_margin_max))
        )
        dynamic_risk = enough_static_contrast and enough_prior_relief and (
            visual_conflict_risk or prior_absorption_risk
        )
        active = bool(str(risk_mode) == "static" or dynamic_risk)
        if lambda_policy == CP_VBC_LAMBDA_POLICY_BAYES_PATH:
            path_active = bool(active and enough_static_contrast and enough_prior_relief and path_top is not None)
            lambda_eff = sum(path_lambdas) / len(path_lambdas) if path_active else 0.0
            next_id = int(path_top) if path_active else image_top
        else:
            lambda_eff = float(lambda_text) if active else 0.0
            final_score = image_logp - lambda_eff * prior_logp
            mask = torch.full_like(final_score, float("-inf"))
            mask[list(allowed_set)] = final_score[list(allowed_set)]
            next_id = int(torch.argmax(mask).detach().cpu().item())
        if dynamic_risk:
            risk_count += 1
        if next_id != image_top:
            override_token_count += 1
        if len(steps) < max(0, int(trace_tokens_limit)):
            steps.append(
                {
                    "step": step_idx,
                    "lambda_eff": lambda_eff,
                    "dynamic_risk": dynamic_risk,
                    "visual_conflict_risk": visual_conflict_risk,
                    "prior_absorption_risk": prior_absorption_risk,
                    "image_top": image_top,
                    "image_top_token": tokenizer.decode([image_top]),
                    "prior_top": prior_top,
                    "prior_top_token": tokenizer.decode([prior_top]),
                    "static_contrast_top": static_contrast_top,
                    "static_contrast_top_token": tokenizer.decode([static_contrast_top]),
                    "path_top": path_top,
                    "path_top_token": tokenizer.decode([path_top]) if path_top is not None else None,
                    "path_stability": path_stability,
                    "path_margin": path_margin,
                    "path_prior_relief": path_prior_relief,
                    "final_token_id": next_id,
                    "final_token": tokenizer.decode([next_id]),
                    "image_margin": image_margin,
                    "prior_margin": prior_margin,
                    "static_contrast_margin": static_contrast_margin,
                    "top_image_tokens": image_rows[:5],
                    "top_contrast_tokens": contrast_rows[:5],
                }
            )
        if next_id in eos_ids:
            break
        generated.append(next_id)
        inputs_image = _append_token_to_inputs(inputs_image, next_id)
        inputs_prior = _append_token_to_inputs(inputs_prior, next_id)

    pred = tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    raw = {
        "backend": "local_transformers_qwen3_vl",
        "model": spec.model,
        "mitigation": MITIGATION_CP_VBC,
        "cp_vbc_mode": CP_VBC_MODE_TOKEN,
        "lambda_text": float(lambda_text),
        "lambda_policy": lambda_policy,
        "path_lambda_low": float(path_lambda_low),
        "path_lambda_high": float(path_lambda_high),
        "path_lambda_steps": int(path_lambda_steps),
        "path_stability_threshold": float(path_stability_threshold),
        "path_margin_threshold": float(path_margin_threshold),
        "path_prior_relief_min": float(path_prior_relief_min),
        "risk_mode": str(risk_mode),
        "token_top_k": int(token_top_k),
        "prompt": prompt,
        "baseline_pred": baseline_pred,
        "generated_token_count": len(generated),
        "risk_token_count": risk_count,
        "override_token_count": override_token_count,
        "steps": steps,
    }
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return str(pred), raw, int((time.time() - t0) * 1000)


def _call_local_qwen3_vl_cp_vbc(
    spec: ModelSpec,
    task: str,
    item: Dict[str, Any],
    image_path: str,
    baseline_pred: str,
    lambda_text: float,
    prior_margin_threshold: float,
    contrast_margin_threshold: float,
    risk_mode: str,
    visual_conflict_image_margin: float,
    absorption_image_margin_max: float,
    lambda_policy: str,
    path_lambda_low: float,
    path_lambda_high: float,
    path_lambda_steps: int,
    path_stability_threshold: float,
    path_margin_threshold: float,
    path_prior_relief_min: float,
) -> Tuple[str, Dict[str, Any], int]:
    try:
        import torch
        from PIL import Image
    except Exception as e:
        raise RuntimeError("cp_vbc requires torch+Pillow in the current Python environment") from e

    model, processor, family = _local_candidate_bundle(spec)
    image = Image.open(image_path).convert("RGB")
    if family == "qwen3_vl":
        image = image.resize((512, 512), Image.Resampling.LANCZOS)
    scoring_prompt = _build_candidate_scoring_prompt(task, item)
    candidates = _candidate_answers(task, item)
    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    image_scores = _candidate_logprobs_shared_prefix(
        model,
        processor,
        scoring_prompt,
        image,
        [cand["text"] for cand in candidates],
        family=family,
    )
    prior_scores = _candidate_logprobs_shared_prefix(
        model,
        processor,
        scoring_prompt,
        None,
        [cand["text"] for cand in candidates],
        family=family,
    )
    for cand, image_score, prior_score in zip(candidates, image_scores, prior_scores):
        rows.append(
            {
                "key": cand["key"],
                "text": cand["text"],
                "logp_image": image_score["logprob"],
                "avg_logp_image": image_score["avg_logprob"],
                "logp_prior": prior_score["logprob"],
                "avg_logp_prior": prior_score["avg_logprob"],
                "image_token_count": image_score["token_count"],
                "prior_token_count": prior_score["token_count"],
                "image_token_ids": image_score["token_ids"],
                "prior_token_ids": prior_score["token_ids"],
            }
        )

    image_probs = _softmax_over([float(r["logp_image"]) for r in rows])
    prior_probs = _softmax_over([float(r["logp_prior"]) for r in rows])
    for row in rows:
        row["visual_gain"] = float(row["logp_image"] - float(lambda_text) * row["logp_prior"])
        row["avg_visual_gain"] = float(row["avg_logp_image"] - float(lambda_text) * row["avg_logp_prior"])
    contrast_probs = _softmax_over([float(r["visual_gain"]) for r in rows])
    for row, p_img, p_prior, p_contrast in zip(rows, image_probs, prior_probs, contrast_probs):
        row["p_image_candidates"] = p_img
        row["p_prior_candidates"] = p_prior
        row["p_contrast_candidates"] = p_contrast

    by_image = sorted(rows, key=lambda r: float(r["logp_image"]), reverse=True)
    by_prior = sorted(rows, key=lambda r: float(r["logp_prior"]), reverse=True)
    by_contrast = sorted(rows, key=lambda r: float(r["visual_gain"]), reverse=True)
    baseline_key = _candidate_key_from_model_prediction(spec, task, baseline_pred)
    static_contrast_key = str(by_contrast[0]["key"]) if by_contrast else None
    image_key = str(by_image[0]["key"]) if by_image else None
    prior_key = str(by_prior[0]["key"]) if by_prior else None
    prior_margin = _margin(by_prior, "logp_prior")
    image_margin = _margin(by_image, "logp_image")
    static_contrast_margin = _margin(by_contrast, "visual_gain")
    lambda_policy = str(lambda_policy or CP_VBC_LAMBDA_POLICY_FIXED).strip().lower()
    path_probabilities: Optional[List[float]] = None
    path_key: Optional[str] = None
    path_margin: Optional[float] = None
    path_stability = 0.0
    path_top_counts: Dict[str, int] = {}
    path_lambdas: List[float] = []
    if lambda_policy == CP_VBC_LAMBDA_POLICY_BAYES_PATH:
        path_steps = max(2, int(path_lambda_steps))
        path_lambdas = [
            float(path_lambda_low)
            + (float(path_lambda_high) - float(path_lambda_low)) * step / (path_steps - 1)
            for step in range(path_steps)
        ]
        path_probabilities = [0.0 for _ in rows]
        path_top_counts = {str(row["key"]): 0 for row in rows}
        for path_lambda in path_lambdas:
            probabilities = _softmax_over(
                [float(row["logp_image"]) - path_lambda * float(row["logp_prior"]) for row in rows]
            )
            if probabilities:
                top_index = max(range(len(probabilities)), key=lambda index: probabilities[index])
                path_top_counts[str(rows[top_index]["key"])] += 1
            for index, probability in enumerate(probabilities):
                path_probabilities[index] += float(probability) / path_steps
        if path_probabilities:
            ranked_indices = sorted(range(len(rows)), key=lambda index: path_probabilities[index], reverse=True)
            path_key = str(rows[ranked_indices[0]]["key"])
            path_stability = path_top_counts.get(path_key, 0) / path_steps
            if len(ranked_indices) > 1:
                path_margin = float(
                    path_probabilities[ranked_indices[0]] - path_probabilities[ranked_indices[1]]
                )
            for row, probability in zip(rows, path_probabilities):
                row["p_path_candidates"] = probability

    contrast_key = path_key if lambda_policy == CP_VBC_LAMBDA_POLICY_BAYES_PATH else static_contrast_key
    high_prior = prior_margin is None or prior_margin >= float(prior_margin_threshold)
    if lambda_policy == CP_VBC_LAMBDA_POLICY_BAYES_PATH:
        enough_static_contrast = bool(
            path_stability >= float(path_stability_threshold)
            and (path_margin is None or path_margin >= float(path_margin_threshold))
        )
    else:
        enough_static_contrast = static_contrast_margin is None or static_contrast_margin >= float(contrast_margin_threshold)
    prior_relief = None
    if baseline_key and contrast_key:
        rows_by_key = {str(row["key"]): row for row in rows}
        if baseline_key in rows_by_key and contrast_key in rows_by_key:
            prior_relief = float(
                rows_by_key[baseline_key]["logp_prior"] - rows_by_key[contrast_key]["logp_prior"]
            )
    enough_prior_relief = bool(
        baseline_key is None
        or lambda_policy != CP_VBC_LAMBDA_POLICY_BAYES_PATH
        or (prior_relief is not None and prior_relief >= float(path_prior_relief_min))
    )
    visual_conflict_risk = bool(
        image_key
        and prior_key
        and contrast_key
        and image_key != prior_key
        and contrast_key == image_key
        and (image_margin is None or image_margin >= float(visual_conflict_image_margin))
    )
    prior_absorption_risk = bool(
        baseline_key
        and prior_key
        and image_key
        and contrast_key
        and baseline_key == prior_key
        and image_key == prior_key
        and contrast_key != baseline_key
        and high_prior
        and (image_margin is None or image_margin <= float(absorption_image_margin_max))
    )
    dynamic_risk = enough_static_contrast and enough_prior_relief and (
        visual_conflict_risk or prior_absorption_risk
    )
    path_lambda_mean = (
        sum(path_lambdas) / len(path_lambdas)
        if path_lambdas
        else float(lambda_text)
    )
    lambda_eff = path_lambda_mean if (str(risk_mode) == "static" or dynamic_risk) else 0.0

    for row in rows:
        row["visual_gain_dynamic"] = float(row["logp_image"] - lambda_eff * row["logp_prior"])
        row["avg_visual_gain_dynamic"] = float(row["avg_logp_image"] - lambda_eff * row["avg_logp_prior"])
        row["lambda_eff"] = lambda_eff

    by_dynamic = sorted(rows, key=lambda r: float(r["visual_gain_dynamic"]), reverse=True)
    if lambda_policy == CP_VBC_LAMBDA_POLICY_BAYES_PATH and lambda_eff > 0 and path_probabilities is not None:
        dynamic_probs = path_probabilities
    else:
        dynamic_probs = _softmax_over([float(r["visual_gain_dynamic"]) for r in rows])
    for row, p_dynamic in zip(rows, dynamic_probs):
        row["p_dynamic_candidates"] = p_dynamic
    if lambda_policy == CP_VBC_LAMBDA_POLICY_BAYES_PATH and lambda_eff > 0:
        final_contrast_key = path_key
        dynamic_contrast_margin = path_margin
    else:
        final_contrast_key = str(by_dynamic[0]["key"]) if by_dynamic else None
        dynamic_contrast_margin = _margin(by_dynamic, "visual_gain_dynamic")
    baseline_missing = baseline_key is None
    enough_dynamic_contrast = (
        enough_static_contrast
        if lambda_policy == CP_VBC_LAMBDA_POLICY_BAYES_PATH
        else (dynamic_contrast_margin is None or dynamic_contrast_margin >= float(contrast_margin_threshold))
    )
    should_override = bool(
        lambda_eff > 0
        and final_contrast_key
        and (baseline_missing or final_contrast_key != baseline_key)
        and high_prior
        and enough_prior_relief
        and enough_dynamic_contrast
    )
    final_key = final_contrast_key if should_override and final_contrast_key else baseline_key
    pred = _prediction_from_candidate_key(task, final_key) if final_key else baseline_pred
    raw = {
        "backend": f"local_transformers_{family}",
        "model": spec.model,
        "family": family,
        "image_preprocessing": "fixed_512" if family == "qwen3_vl" else "model_native",
        "mitigation": MITIGATION_CP_VBC,
        "lambda_text": float(lambda_text),
        "lambda_eff": lambda_eff,
        "lambda_policy": lambda_policy,
        "path_lambda_low": float(path_lambda_low),
        "path_lambda_high": float(path_lambda_high),
        "path_lambda_steps": int(path_lambda_steps),
        "path_stability_threshold": float(path_stability_threshold),
        "path_margin_threshold": float(path_margin_threshold),
        "path_prior_relief_min": float(path_prior_relief_min),
        "risk_mode": str(risk_mode),
        "prior_margin_threshold": float(prior_margin_threshold),
        "contrast_margin_threshold": float(contrast_margin_threshold),
        "visual_conflict_image_margin": float(visual_conflict_image_margin),
        "absorption_image_margin_max": float(absorption_image_margin_max),
        "scoring_prompt": scoring_prompt,
        "baseline_pred": baseline_pred,
        "baseline_key": baseline_key,
        "image_top": image_key,
        "prior_top": prior_key,
        "static_contrast_top": static_contrast_key,
        "path_top": path_key,
        "path_stability": path_stability,
        "path_margin": path_margin,
        "path_top_counts": path_top_counts,
        "prior_relief": prior_relief,
        "enough_prior_relief": enough_prior_relief,
        "contrast_top": final_contrast_key,
        "final_key": final_key,
        "overridden": should_override,
        "dynamic_risk": dynamic_risk,
        "visual_conflict_risk": visual_conflict_risk,
        "prior_absorption_risk": prior_absorption_risk,
        "prior_margin": prior_margin,
        "image_margin": image_margin,
        "static_contrast_margin": static_contrast_margin,
        "contrast_margin": dynamic_contrast_margin,
        "image_entropy": _entropy(image_probs),
        "prior_entropy": _entropy(prior_probs),
        "static_contrast_entropy": _entropy(contrast_probs),
        "contrast_entropy": _entropy(dynamic_probs),
        "candidates": rows,
    }
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pred, raw, int((time.time() - t0) * 1000)


def _call_local_qwen3_vl_revis_candidate(
    spec: ModelSpec,
    task: str,
    item: Dict[str, Any],
    image_path: str,
    baseline_pred: str,
    lambda_prior: float,
    lambda_hidden: float,
    gate: str,
    prior_margin_threshold: float,
    contrast_margin_threshold: float,
    visual_conflict_image_margin: float,
    absorption_image_margin_max: float,
    absmax_min: float,
    hidden_margin_threshold: float,
) -> Tuple[str, Dict[str, Any], int]:
    try:
        import torch
        from PIL import Image
    except Exception as e:
        raise RuntimeError("revis requires torch+Pillow in the current Python environment") from e

    if task not in ("qa", "mc"):
        raise ValueError("candidate REVIS currently supports qa and mc")

    model, processor, family = _local_candidate_bundle(spec)
    image = Image.open(image_path).convert("RGB")
    if family == "qwen3_vl":
        image = image.resize((512, 512), Image.Resampling.LANCZOS)
    scoring_prompt = _build_candidate_scoring_prompt(task, item)
    prompt_probe = _visual_delta_probe(model, processor, scoring_prompt, image)
    prompt_layers = prompt_probe.get("layers") or []
    prompt_relative_values = [
        float(row.get("relative_delta"))
        for row in prompt_layers
        if isinstance(row.get("relative_delta"), (int, float))
    ]
    prompt_absmax = max(prompt_relative_values) if prompt_relative_values else None
    candidates = _candidate_answers(task, item)
    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    for cand in candidates:
        image_score = _candidate_logprob_qwen3_vl(model, processor, scoring_prompt, image, cand["text"])
        prior_score = _candidate_logprob_qwen3_vl(model, processor, scoring_prompt, None, cand["text"])
        hidden_probe = _candidate_hidden_delta_qwen3_vl(model, processor, scoring_prompt, image, cand["text"])
        hidden_signal = hidden_probe.get("relative_pure_visual")
        if not isinstance(hidden_signal, (int, float)):
            hidden_signal = hidden_probe.get("max_relative_delta")
        rows.append(
            {
                "key": cand["key"],
                "text": cand["text"],
                "logp_image": image_score["logprob"],
                "avg_logp_image": image_score["avg_logprob"],
                "logp_prior": prior_score["logprob"],
                "avg_logp_prior": prior_score["avg_logprob"],
                "candidate_visual_delta": hidden_signal,
                "candidate_pure_visual_ratio": hidden_probe.get("pure_visual_ratio"),
                "candidate_image_prior_cosine": hidden_probe.get("image_prior_cosine"),
                "candidate_visual_delta_mean": hidden_probe.get("mean_relative_delta"),
                "candidate_delta_norm": hidden_probe.get("max_delta_norm"),
                "candidate_probe": hidden_probe,
                "image_token_count": image_score["token_count"],
                "prior_token_count": prior_score["token_count"],
                "image_token_ids": image_score["token_ids"],
                "prior_token_ids": prior_score["token_ids"],
            }
        )

    image_probs = _softmax_over([float(r["logp_image"]) for r in rows])
    prior_probs = _softmax_over([float(r["logp_prior"]) for r in rows])
    hidden_gains = _minmax_unit(
        [
            float(r["candidate_visual_delta"])
            if isinstance(r.get("candidate_visual_delta"), (int, float))
            else None
            for r in rows
        ]
    )
    for row, hidden_gain in zip(rows, hidden_gains):
        row["hidden_gain"] = hidden_gain
        row["prior_corrected_score"] = float(row["logp_image"] - float(lambda_prior) * row["logp_prior"])
        row["revis_static_score"] = float(row["prior_corrected_score"] + float(lambda_hidden) * hidden_gain)
        row["avg_prior_corrected_score"] = float(
            row["avg_logp_image"] - float(lambda_prior) * row["avg_logp_prior"]
        )

    prior_corrected_probs = _softmax_over([float(r["prior_corrected_score"]) for r in rows])
    revis_static_probs = _softmax_over([float(r["revis_static_score"]) for r in rows])
    for row, p_img, p_prior, p_pc, p_revis in zip(
        rows, image_probs, prior_probs, prior_corrected_probs, revis_static_probs
    ):
        row["p_image_candidates"] = p_img
        row["p_prior_candidates"] = p_prior
        row["p_prior_corrected_candidates"] = p_pc
        row["p_revis_candidates"] = p_revis

    by_image = sorted(rows, key=lambda r: float(r["logp_image"]), reverse=True)
    by_prior = sorted(rows, key=lambda r: float(r["logp_prior"]), reverse=True)
    by_prior_corrected = sorted(rows, key=lambda r: float(r["prior_corrected_score"]), reverse=True)
    by_hidden = sorted(rows, key=lambda r: float(r["hidden_gain"]), reverse=True)
    by_revis_static = sorted(rows, key=lambda r: float(r["revis_static_score"]), reverse=True)
    baseline_key = _candidate_key_from_model_prediction(spec, task, baseline_pred)
    image_key = str(by_image[0]["key"]) if by_image else None
    prior_key = str(by_prior[0]["key"]) if by_prior else None
    prior_corrected_key = str(by_prior_corrected[0]["key"]) if by_prior_corrected else None
    hidden_key = str(by_hidden[0]["key"]) if by_hidden else None
    revis_static_key = str(by_revis_static[0]["key"]) if by_revis_static else None
    prior_margin = _margin(by_prior, "logp_prior")
    image_margin = _margin(by_image, "logp_image")
    prior_corrected_margin = _margin(by_prior_corrected, "prior_corrected_score")
    hidden_margin = _margin(by_hidden, "hidden_gain")
    revis_static_margin = _margin(by_revis_static, "revis_static_score")
    high_prior = prior_margin is None or prior_margin >= float(prior_margin_threshold)
    enough_static_contrast = (
        revis_static_margin is None or revis_static_margin >= float(contrast_margin_threshold)
    )
    enough_absmax = prompt_absmax is None or prompt_absmax >= float(absmax_min)
    visual_conflict_risk = bool(
        image_key
        and prior_key
        and revis_static_key
        and image_key != prior_key
        and revis_static_key == image_key
        and (image_margin is None or image_margin >= float(visual_conflict_image_margin))
    )
    prior_absorption_risk = bool(
        baseline_key
        and prior_key
        and image_key
        and revis_static_key
        and baseline_key == prior_key
        and image_key == prior_key
        and revis_static_key != baseline_key
        and high_prior
        and (image_margin is None or image_margin <= float(absorption_image_margin_max))
    )
    hidden_rescue_risk = bool(
        baseline_key
        and hidden_key
        and revis_static_key
        and hidden_key == revis_static_key
        and hidden_key != baseline_key
        and high_prior
        and (hidden_margin is None or hidden_margin >= float(hidden_margin_threshold))
    )
    dynamic_risk = bool(
        enough_absmax
        and enough_static_contrast
        and (visual_conflict_risk or prior_absorption_risk or hidden_rescue_risk)
    )
    gate = str(gate or "dynamic").strip().lower()
    if gate == "always":
        active = bool(enough_absmax)
        gate_reason = "always_gate" if active else "absmax_below_threshold"
    elif gate == "dynamic":
        active = bool(dynamic_risk)
        if visual_conflict_risk:
            gate_reason = "visual_conflict_supported"
        elif prior_absorption_risk:
            gate_reason = "prior_absorption_supported"
        elif hidden_rescue_risk:
            gate_reason = "hidden_rescue_supported"
        elif not enough_absmax:
            gate_reason = "absmax_below_threshold"
        elif not enough_static_contrast:
            gate_reason = "contrast_margin_below_threshold"
        else:
            gate_reason = "no_revis_risk"
    else:
        raise ValueError(f"unknown revis gate: {gate}")

    for row in rows:
        lambda_prior_eff = float(lambda_prior) if active else 0.0
        lambda_hidden_eff = float(lambda_hidden) if active else 0.0
        row["revis_dynamic_score"] = float(
            row["logp_image"] - lambda_prior_eff * row["logp_prior"] + lambda_hidden_eff * row["hidden_gain"]
        )
        row["lambda_prior_eff"] = lambda_prior_eff
        row["lambda_hidden_eff"] = lambda_hidden_eff

    by_revis_dynamic = sorted(rows, key=lambda r: float(r["revis_dynamic_score"]), reverse=True)
    revis_dynamic_probs = _softmax_over([float(r["revis_dynamic_score"]) for r in rows])
    for row, p_dynamic in zip(rows, revis_dynamic_probs):
        row["p_revis_dynamic_candidates"] = p_dynamic
    revis_dynamic_key = str(by_revis_dynamic[0]["key"]) if by_revis_dynamic else None
    revis_dynamic_margin = _margin(by_revis_dynamic, "revis_dynamic_score")
    baseline_missing = baseline_key is None
    should_override = bool(
        active
        and revis_dynamic_key
        and (baseline_missing or revis_dynamic_key != baseline_key)
        and high_prior
        and (revis_dynamic_margin is None or revis_dynamic_margin >= float(contrast_margin_threshold))
    )
    final_key = revis_dynamic_key if should_override and revis_dynamic_key else baseline_key
    pred = _prediction_from_candidate_key(task, final_key) if final_key else baseline_pred
    raw = {
        "backend": "local_transformers_qwen3_vl",
        "model": spec.model,
        "mitigation": MITIGATION_REVIS,
        "revis_mode": "candidate_hidden_delta",
        "lambda_prior": float(lambda_prior),
        "lambda_hidden": float(lambda_hidden),
        "gate": gate,
        "gate_reason": gate_reason,
        "single_image_or_no_image_contrast": True,
        "prior_margin_threshold": float(prior_margin_threshold),
        "contrast_margin_threshold": float(contrast_margin_threshold),
        "visual_conflict_image_margin": float(visual_conflict_image_margin),
        "absorption_image_margin_max": float(absorption_image_margin_max),
        "absmax_min": float(absmax_min),
        "hidden_margin_threshold": float(hidden_margin_threshold),
        "scoring_prompt": scoring_prompt,
        "baseline_pred": baseline_pred,
        "baseline_key": baseline_key,
        "image_top": image_key,
        "prior_top": prior_key,
        "prior_corrected_top": prior_corrected_key,
        "hidden_top": hidden_key,
        "static_top": revis_static_key,
        "contrast_top": revis_dynamic_key,
        "final_key": final_key,
        "overridden": should_override,
        "active": active,
        "dynamic_risk": dynamic_risk,
        "visual_conflict_risk": visual_conflict_risk,
        "prior_absorption_risk": prior_absorption_risk,
        "hidden_rescue_risk": hidden_rescue_risk,
        "prompt_absmax_relative_delta": prompt_absmax,
        "prompt_probe": prompt_probe,
        "prior_margin": prior_margin,
        "image_margin": image_margin,
        "prior_corrected_margin": prior_corrected_margin,
        "hidden_margin": hidden_margin,
        "static_margin": revis_static_margin,
        "contrast_margin": revis_dynamic_margin,
        "image_entropy": _entropy(image_probs),
        "prior_entropy": _entropy(prior_probs),
        "prior_corrected_entropy": _entropy(prior_corrected_probs),
        "static_entropy": _entropy(revis_static_probs),
        "contrast_entropy": _entropy(revis_dynamic_probs),
        "candidates": rows,
    }
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pred, raw, int((time.time() - t0) * 1000)


def _call_local_qwen3_vl_revis_token(
    spec: ModelSpec,
    task: str,
    item: Dict[str, Any],
    image_path: str,
    baseline_pred: str,
    lambda_prior: float,
    lambda_hidden: float,
    gate: str,
    prior_margin_threshold: float,
    contrast_margin_threshold: float,
    visual_conflict_image_margin: float,
    absorption_image_margin_max: float,
    absmax_min: float,
    prior_source: str,
    prior_degrade_mode: str,
    prior_score_form: str,
    prior_inertia_gate: str,
    prior_inertia_prob_min: float,
    prior_inertia_logprob_margin: float,
    prior_subspace_alpha: float,
    prior_subspace_top_k: int,
    layer_prior_subspace_alpha: float,
    layer_prior_subspace_top_k: int,
    layer_prior_subspace_index: int,
    layer_prior_subspace_fraction: float,
    attention_prior_alpha: float,
    attention_prior_layer_index: int,
    attention_prior_layer_fraction: float,
    attention_prior_head_top_k: int,
    attention_visual_alpha: float,
    attention_visual_layer_index: int,
    attention_visual_layer_fraction: float,
    attention_visual_head_top_k: int,
    image_attention_alpha: float,
    image_attention_layer_index: int,
    image_attention_layer_fraction: float,
    image_attention_head_top_k: int,
    image_attention_head_select: str,
    image_attention_text_alpha: float,
    image_attention_text_top_k: int,
    jspace_alpha: float,
    jspace_gamma: float,
    jspace_top_k: int,
    jspace_layer_index: int,
    jspace_layer_fraction: float,
    jspace_probe: str,
    jspace_lens: str,
    jspace_lens_path: str,
    jspace_swap_alpha: float,
    latent_gamma: float,
    layer_gamma: float,
    layer_index: int,
    layer_fraction: float,
    attention_probe: str,
    text_inertia_mode: str,
    text_inertia_visual_attention_max: float,
    text_inertia_logprob_margin: float,
    text_inertia_prior_logp_min: float,
    text_inertia_penalty: float,
    text_inertia_scope: str,
    token_top_k: int,
    trace_tokens_limit: int,
) -> Tuple[str, Dict[str, Any], int]:
    try:
        import torch
        from PIL import Image
    except Exception as e:
        raise RuntimeError("token REVIS requires torch+Pillow in the current Python environment") from e

    model, processor = _local_qwen3_bundle(spec)
    tokenizer = getattr(processor, "tokenizer", processor)
    image = Image.open(image_path).convert("RGB")
    image = image.resize((512, 512), Image.Resampling.LANCZOS)
    prior_source = str(prior_source or "no_image").strip().lower()
    if prior_source not in {"no_image", "degraded_image"}:
        raise ValueError(f"unknown REVIS prior source: {prior_source}")
    prior_image = _make_vcd_degraded_image(image, prior_degrade_mode) if prior_source == "degraded_image" else None
    prompt = _build_token_cp_vbc_prompt(task, item)
    inputs_image = _qwen3_vl_inputs(model, processor, prompt, image)
    inputs_prior = _qwen3_vl_inputs(model, processor, prompt, prior_image)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    eos_ids = set(eos_token_id if isinstance(eos_token_id, list) else [eos_token_id]) if eos_token_id is not None else set()
    generated: List[int] = []
    steps: List[Dict[str, Any]] = []
    risk_count = 0
    override_token_count = 0
    active_token_count = 0
    t0 = time.time()
    gate = str(gate or "dynamic").strip().lower()
    if gate not in {"dynamic", "always"}:
        raise ValueError(f"unknown revis gate: {gate}")
    prior_inertia_gate = str(prior_inertia_gate or "none").strip().lower()
    if prior_inertia_gate not in {"none", "agreement"}:
        raise ValueError(f"unknown prior-inertia gate: {prior_inertia_gate}")
    prior_subspace_enabled = float(prior_subspace_alpha) != 0.0
    layer_prior_subspace_enabled = float(layer_prior_subspace_alpha) != 0.0
    attention_prior_enabled = float(attention_prior_alpha) != 0.0
    attention_visual_enabled = float(attention_visual_alpha) != 0.0
    image_attention_enabled = float(image_attention_alpha) != 0.0 or float(image_attention_text_alpha) != 0.0
    jspace_probe = str(jspace_probe or "none").strip().lower()
    if jspace_probe not in {"none", "summary"}:
        raise ValueError(f"unknown jspace probe mode: {jspace_probe}")
    jspace_lens = str(jspace_lens or "logit_lens").strip().lower()
    if jspace_lens not in {"logit_lens", "local_jacobian", "fitted_jacobian"}:
        raise ValueError(f"unknown jspace lens: {jspace_lens}")
    if jspace_lens == "fitted_jacobian" and not str(jspace_lens_path or "").strip():
        raise ValueError("fitted_jacobian J-space mode requires --revis-jspace-lens-path")
    fitted_jspace_lens = _load_jspace_lens(str(jspace_lens_path or "")) if jspace_lens == "fitted_jacobian" else None
    jspace_intervention_enabled = (
        float(jspace_alpha) != 0.0
        or float(jspace_gamma) != 0.0
        or float(jspace_swap_alpha) != 0.0
    )
    jspace_probe_enabled = jspace_probe != "none"
    layer_steering_enabled = float(layer_gamma) != 0.0
    resolved_layer_index = None
    num_decoder_layers = None
    resolved_prior_subspace_layer_index = None
    num_prior_subspace_layers = None
    resolved_attention_prior_layer_index = None
    num_attention_prior_layers = None
    resolved_attention_visual_layer_index = None
    num_attention_visual_layers = None
    resolved_image_attention_layer_index = None
    num_image_attention_layers = None
    resolved_jspace_layer_index = None
    num_jspace_layers = None
    if layer_steering_enabled:
        resolved_layer_index, num_decoder_layers = _resolve_revis_layer_index(
            model,
            layer_index=int(layer_index),
            layer_fraction=float(layer_fraction),
        )
    if layer_prior_subspace_enabled:
        resolved_prior_subspace_layer_index, num_prior_subspace_layers = _resolve_revis_layer_index(
            model,
            layer_index=int(layer_prior_subspace_index),
            layer_fraction=float(layer_prior_subspace_fraction),
        )
    if attention_prior_enabled:
        resolved_attention_prior_layer_index, num_attention_prior_layers = _resolve_revis_layer_index(
            model,
            layer_index=int(attention_prior_layer_index),
            layer_fraction=float(attention_prior_layer_fraction),
        )
    if attention_visual_enabled:
        resolved_attention_visual_layer_index, num_attention_visual_layers = _resolve_revis_layer_index(
            model,
            layer_index=int(attention_visual_layer_index),
            layer_fraction=float(attention_visual_layer_fraction),
        )
    if image_attention_enabled:
        resolved_image_attention_layer_index, num_image_attention_layers = _resolve_revis_layer_index(
            model,
            layer_index=int(image_attention_layer_index),
            layer_fraction=float(image_attention_layer_fraction),
        )
    if jspace_intervention_enabled or jspace_probe_enabled:
        resolved_jspace_layer_index, num_jspace_layers = _resolve_revis_layer_index(
            model,
            layer_index=int(jspace_layer_index),
            layer_fraction=float(jspace_layer_fraction),
        )
    attention_probe = str(attention_probe or "none").strip().lower()
    attention_probe_enabled = attention_probe != "none"
    text_inertia_mode = str(text_inertia_mode or "none").strip().lower()
    if text_inertia_mode not in {"none", "trace", "suppress"}:
        raise ValueError(f"unknown text-inertia mode: {text_inertia_mode}")
    text_inertia_enabled = text_inertia_mode != "none"

    for step_idx in range(max(1, int(spec.max_tokens))):
        trace_this_step = len(steps) < max(0, int(trace_tokens_limit))
        need_attention = bool((attention_probe_enabled or text_inertia_enabled) and trace_this_step)
        with torch.inference_mode():
            image_out = model(
                **inputs_image,
                output_hidden_states=True,
                output_attentions=need_attention,
                use_cache=False,
            )
            prior_out = model(**inputs_prior, output_hidden_states=True, use_cache=False)
            image_logits = image_out.logits[0, -1, :].float()
            prior_logits = prior_out.logits[0, -1, :].float()
            image_logp = torch.log_softmax(image_logits, dim=-1)
            prior_logp = torch.log_softmax(prior_logits, dim=-1)
            prior_correction, prior_correction_summary = _revis_prior_correction_score(
                prior_logits,
                prior_logp,
                prior_score_form,
            )
            latent_logp_static, latent_summary_static = _latent_steered_logp_from_outputs(
                model,
                image_out,
                prior_out,
                latent_gamma=float(latent_gamma),
            )
            prior_subspace_logp_static, prior_subspace_summary_static = _prior_token_subspace_steered_logp(
                model,
                image_out,
                prior_logp,
                alpha=float(prior_subspace_alpha),
                top_k=int(prior_subspace_top_k),
            )

        attention_summary = _visual_attention_summary(model, inputs_image, image_out) if need_attention else {}
        layer_summary_static: Dict[str, Any] = {}
        layer_hook_static: Dict[str, Any] = {}
        layer_logp_static = None
        layer_prior_subspace_hook_static: Dict[str, Any] = {}
        layer_prior_subspace_logp_static = None
        attention_prior_hook_static: Dict[str, Any] = {}
        attention_prior_logp_static = None
        attention_visual_hook_static: Dict[str, Any] = {}
        attention_visual_logp_static = None
        image_attention_hook_static: Dict[str, Any] = {}
        image_attention_logp_static = None
        jspace_hook_static: Dict[str, Any] = {}
        jspace_logp_static = None
        if layer_steering_enabled and resolved_layer_index is not None:
            layer_summary_static, layer_residual_static = _orthogonal_visual_residual_from_outputs(
                image_out,
                prior_out,
                state_index=int(resolved_layer_index) + 1,
            )
            if layer_residual_static is not None:
                layer_logp_static, layer_hook_static = _layer_residual_steered_logp(
                    model,
                    inputs_image,
                    layer_index=int(resolved_layer_index),
                    residual=layer_residual_static,
                    layer_gamma=float(layer_gamma),
                )
        if layer_prior_subspace_enabled and resolved_prior_subspace_layer_index is not None:
            layer_prior_subspace_logp_static, layer_prior_subspace_hook_static = _layer_prior_subspace_steered_logp(
                model,
                inputs_image,
                prior_logp,
                image_logp,
                layer_index=int(resolved_prior_subspace_layer_index),
                alpha=float(layer_prior_subspace_alpha),
                top_k=int(layer_prior_subspace_top_k),
            )
        if attention_prior_enabled and resolved_attention_prior_layer_index is not None:
            if int(attention_prior_head_top_k) > 0:
                attention_prior_logp_static, attention_prior_hook_static = _layer_attention_prior_head_steered_logp(
                    model,
                    inputs_image,
                    inputs_prior,
                    image_logp,
                    prior_logp,
                    layer_index=int(resolved_attention_prior_layer_index),
                    alpha=float(attention_prior_alpha),
                    head_top_k=int(attention_prior_head_top_k),
                )
            else:
                attention_prior_logp_static, attention_prior_hook_static = _layer_attention_prior_residual_steered_logp(
                    model,
                    inputs_image,
                    inputs_prior,
                    image_logp,
                    prior_logp,
                    layer_index=int(resolved_attention_prior_layer_index),
                    alpha=float(attention_prior_alpha),
                )
        if attention_visual_enabled and resolved_attention_visual_layer_index is not None:
            attention_visual_logp_static, attention_visual_hook_static = _layer_attention_visual_delta_steered_logp(
                model,
                inputs_image,
                inputs_prior,
                image_logp,
                prior_logp,
                layer_index=int(resolved_attention_visual_layer_index),
                alpha=float(attention_visual_alpha),
                head_top_k=int(attention_visual_head_top_k),
            )
        if image_attention_enabled and resolved_image_attention_layer_index is not None:
            image_attention_logp_static, image_attention_hook_static = _layer_image_attention_boost_steered_logp(
                model,
                inputs_image,
                image_logp,
                prior_logp,
                layer_index=int(resolved_image_attention_layer_index),
                alpha=float(image_attention_alpha),
                head_top_k=int(image_attention_head_top_k),
                head_select=str(image_attention_head_select),
                text_alpha=float(image_attention_text_alpha),
                text_top_k=int(image_attention_text_top_k),
            )
        if (jspace_intervention_enabled or jspace_probe_enabled) and resolved_jspace_layer_index is not None:
            jspace_jacobian = None
            if fitted_jspace_lens is not None:
                jspace_jacobian = (fitted_jspace_lens.get("jacobians") or {}).get(int(resolved_jspace_layer_index))
                if jspace_jacobian is None:
                    jspace_hook_static = {
                        "available": False,
                        "reason": "selected_layer_not_in_fitted_lens",
                        "lens_path": str(jspace_lens_path or ""),
                        "requested_layer": int(resolved_jspace_layer_index),
                        "source_layers": fitted_jspace_lens.get("source_layers"),
                    }
            static_jspace_alpha = (
                float(jspace_alpha)
                if (jspace_intervention_enabled and jspace_lens in {"logit_lens", "fitted_jacobian"})
                else 0.0
            )
            static_jspace_gamma = (
                float(jspace_gamma)
                if (jspace_intervention_enabled and jspace_lens in {"logit_lens", "fitted_jacobian"})
                else 0.0
            )
        if (jspace_intervention_enabled or jspace_probe_enabled) and resolved_jspace_layer_index is not None and not jspace_hook_static:
            jspace_logp_static, jspace_hook_static = _layer_jspace_prior_steered_logp(
                model,
                tokenizer,
                inputs_image,
                image_out,
                prior_out,
                image_logp,
                prior_logp,
                layer_index=int(resolved_jspace_layer_index),
                alpha=static_jspace_alpha,
                gamma=static_jspace_gamma,
                top_k=int(jspace_top_k),
                lens_jacobian=jspace_jacobian,
                lens_path=str(jspace_lens_path or ""),
                swap_alpha=float(jspace_swap_alpha) if jspace_lens == "fitted_jacobian" else 0.0,
            )

        if jspace_logp_static is not None and jspace_intervention_enabled and jspace_lens in {"logit_lens", "fitted_jacobian"}:
            steered_logp_static = jspace_logp_static
        elif image_attention_logp_static is not None:
            steered_logp_static = image_attention_logp_static
        elif attention_visual_logp_static is not None:
            steered_logp_static = attention_visual_logp_static
        elif attention_prior_logp_static is not None:
            steered_logp_static = attention_prior_logp_static
        elif layer_logp_static is not None:
            steered_logp_static = layer_logp_static
        elif layer_prior_subspace_logp_static is not None:
            steered_logp_static = layer_prior_subspace_logp_static
        elif prior_subspace_enabled:
            steered_logp_static = prior_subspace_logp_static
        else:
            steered_logp_static = latent_logp_static
        static_score = steered_logp_static - float(lambda_prior) * prior_correction

        hidden_summary = _hidden_delta_summary_from_outputs(image_out, prior_out, "image", "no_image")
        step_absmax = layer_summary_static.get("relative_pure_visual") if layer_summary_static else None
        if not isinstance(step_absmax, (int, float)):
            step_absmax = latent_summary_static.get("relative_pure_visual")
        if not isinstance(step_absmax, (int, float)):
            step_absmax = hidden_summary.get("max_relative_delta")
        hidden_strength = min(max(float(step_absmax or 0.0), 0.0), 1.0)
        hidden_risk = step_absmax is None or float(step_absmax) >= float(absmax_min)

        top_k = max(2, min(int(token_top_k), int(image_logp.numel())))
        allowed_ids = _topk_union_token_ids([image_logp, prior_logp, static_score], top_k)
        allowed_set = set(int(x) for x in allowed_ids.detach().cpu().tolist())
        image_rows = _top_token_rows(tokenizer, image_logp, prior_logp, image_logp, allowed_ids)
        image_rows.sort(key=lambda r: r["logp_image"], reverse=True)
        prior_rows = sorted(image_rows, key=lambda r: r["logp_prior"], reverse=True)
        static_rows = sorted(
            _top_token_rows(tokenizer, image_logp, prior_logp, static_score, allowed_ids),
            key=lambda r: r["score"],
            reverse=True,
        )

        image_top = int(image_rows[0]["token_id"])
        prior_top = int(prior_rows[0]["token_id"])
        static_top = int(static_rows[0]["token_id"])
        image_margin = _token_margin(image_rows, "logp_image")
        prior_margin = _token_margin(prior_rows, "logp_prior")
        static_margin = _token_margin(static_rows, "score")
        high_prior = prior_margin is None or prior_margin >= float(prior_margin_threshold)
        enough_static_contrast = static_margin is None or static_margin >= float(contrast_margin_threshold)
        prior_top_probability = float(torch.exp(prior_logp[prior_top]).detach().cpu().item())
        image_prior_top_logprob_gap = float((image_logp[image_top] - prior_logp[image_top]).detach().cpu().item())
        visual_conflict_risk = bool(
            image_top != prior_top
            and static_top == image_top
            and (image_margin is None or image_margin >= float(visual_conflict_image_margin))
        )
        prior_absorption_risk = bool(
            image_top == prior_top
            and static_top != image_top
            and high_prior
            and (image_margin is None or image_margin <= float(absorption_image_margin_max))
        )
        prior_inertia_risk = bool(
            prior_inertia_gate == "agreement"
            and image_top == prior_top
            and high_prior
            and prior_top_probability >= float(prior_inertia_prob_min)
            and abs(image_prior_top_logprob_gap) <= float(prior_inertia_logprob_margin)
            and (image_margin is None or image_margin <= float(absorption_image_margin_max))
        )
        dynamic_risk = bool(
            hidden_risk
            and enough_static_contrast
            and (visual_conflict_risk or prior_absorption_risk or prior_inertia_risk)
        )
        if gate == "always":
            active = bool(hidden_risk)
            gate_reason = "always_gate" if active else "absmax_below_threshold"
        else:
            active = bool(dynamic_risk)
            if active and visual_conflict_risk:
                gate_reason = "visual_conflict_supported"
            elif active and prior_absorption_risk:
                gate_reason = "prior_absorption_supported"
            elif active and prior_inertia_risk:
                gate_reason = "prior_inertia_supported"
            elif not hidden_risk:
                gate_reason = "absmax_below_threshold"
            elif not enough_static_contrast:
                gate_reason = "contrast_margin_below_threshold"
            elif visual_conflict_risk or prior_absorption_risk:
                gate_reason = "risk_below_dynamic_gate"
            else:
                gate_reason = "no_revis_risk"
        lambda_eff = float(lambda_prior) * (1.0 + float(lambda_hidden) * hidden_strength) if active else 0.0
        latent_gamma_eff = float(latent_gamma) * (1.0 + float(lambda_hidden) * hidden_strength) if active else 0.0
        layer_gamma_eff = float(layer_gamma) * (1.0 + float(lambda_hidden) * hidden_strength) if active else 0.0
        prior_subspace_alpha_eff = (
            float(prior_subspace_alpha) * (1.0 + float(lambda_hidden) * hidden_strength)
            if active
            else 0.0
        )
        layer_prior_subspace_alpha_eff = (
            float(layer_prior_subspace_alpha) * (1.0 + float(lambda_hidden) * hidden_strength)
            if active
            else 0.0
        )
        attention_prior_alpha_eff = (
            float(attention_prior_alpha) * (1.0 + float(lambda_hidden) * hidden_strength)
            if active
            else 0.0
        )
        attention_visual_alpha_eff = (
            float(attention_visual_alpha) * (1.0 + float(lambda_hidden) * hidden_strength)
            if active
            else 0.0
        )
        image_attention_alpha_eff = (
            float(image_attention_alpha) * (1.0 + float(lambda_hidden) * hidden_strength)
            if active
            else 0.0
        )
        image_attention_text_alpha_eff = (
            float(image_attention_text_alpha) * (1.0 + float(lambda_hidden) * hidden_strength)
            if active
            else 0.0
        )
        jspace_alpha_eff = (
            float(jspace_alpha) * (1.0 + float(lambda_hidden) * hidden_strength)
            if active
            else 0.0
        )
        jspace_gamma_eff = (
            float(jspace_gamma) * (1.0 + float(lambda_hidden) * hidden_strength)
            if active
            else 0.0
        )
        layer_summary_final = layer_summary_static
        layer_hook_final = layer_hook_static
        layer_prior_subspace_hook_final = layer_prior_subspace_hook_static
        attention_prior_hook_final = attention_prior_hook_static
        attention_visual_hook_final = attention_visual_hook_static
        image_attention_hook_final = image_attention_hook_static
        jspace_hook_final = jspace_hook_static
        prior_subspace_summary_final = prior_subspace_summary_static
        if active and jspace_intervention_enabled and resolved_jspace_layer_index is not None:
            if jspace_lens == "local_jacobian":
                steered_logp_final, jspace_hook_final = _layer_local_jacobian_prior_steered_logp(
                    model,
                    tokenizer,
                    inputs_image,
                    inputs_prior,
                    image_out,
                    prior_out,
                    image_logp,
                    prior_logp,
                    layer_index=int(resolved_jspace_layer_index),
                    prior_token_id=int(prior_top),
                    alpha=jspace_alpha_eff,
                    gamma=jspace_gamma_eff,
                )
                if jspace_hook_static:
                    jspace_hook_final["logit_lens_probe"] = jspace_hook_static
            elif (
                jspace_alpha_eff != float(jspace_alpha)
                or jspace_gamma_eff != float(jspace_gamma)
            ):
                steered_logp_final, jspace_hook_final = _layer_jspace_prior_steered_logp(
                    model,
                    tokenizer,
                    inputs_image,
                    image_out,
                    prior_out,
                    image_logp,
                    prior_logp,
                    layer_index=int(resolved_jspace_layer_index),
                    alpha=jspace_alpha_eff,
                    gamma=jspace_gamma_eff,
                    top_k=int(jspace_top_k),
                    lens_jacobian=jspace_jacobian if jspace_lens == "fitted_jacobian" else None,
                    lens_path=str(jspace_lens_path or "") if jspace_lens == "fitted_jacobian" else "",
                    swap_alpha=float(jspace_swap_alpha) if jspace_lens == "fitted_jacobian" else 0.0,
                )
            elif jspace_logp_static is not None:
                steered_logp_final = jspace_logp_static
            else:
                steered_logp_final = image_logp
                jspace_hook_final = {"applied": False, "reason": "missing_jspace_static"}
            latent_summary_final = {
                **latent_summary_static,
                "latent_gamma": 0.0,
                "steered": False,
                "superseded_by_jspace": True,
            }
            layer_hook_final = {
                **layer_hook_static,
                "applied": False,
                "layer_gamma": 0.0,
                "superseded_by_jspace": True,
            } if layer_hook_static else {}
            layer_prior_subspace_hook_final = {
                **layer_prior_subspace_hook_static,
                "layer_alpha": 0.0,
                "applied": False,
                "superseded_by_jspace": True,
            } if layer_prior_subspace_hook_static else {}
            attention_prior_hook_final = {
                **attention_prior_hook_static,
                "applied": False,
                "attention_prior_alpha": 0.0,
                "superseded_by_jspace": True,
            } if attention_prior_hook_static else {}
            attention_visual_hook_final = {
                **attention_visual_hook_static,
                "applied": False,
                "attention_visual_alpha": 0.0,
                "superseded_by_jspace": True,
            } if attention_visual_hook_static else {}
            image_attention_hook_final = {
                **image_attention_hook_static,
                "applied": False,
                "image_attention_alpha": 0.0,
                "text_attention_alpha": 0.0,
                "superseded_by_jspace": True,
            } if image_attention_hook_static else {}
            prior_subspace_summary_final = {
                **prior_subspace_summary_static,
                "alpha": 0.0,
                "steered": False,
                "superseded_by_jspace": True,
            }
        elif active and image_attention_enabled and resolved_image_attention_layer_index is not None:
            if (
                image_attention_alpha_eff != float(image_attention_alpha)
                or image_attention_text_alpha_eff != float(image_attention_text_alpha)
            ):
                steered_logp_final, image_attention_hook_final = _layer_image_attention_boost_steered_logp(
                    model,
                    inputs_image,
                    image_logp,
                    prior_logp,
                    layer_index=int(resolved_image_attention_layer_index),
                    alpha=image_attention_alpha_eff,
                    head_top_k=int(image_attention_head_top_k),
                    head_select=str(image_attention_head_select),
                    text_alpha=image_attention_text_alpha_eff,
                    text_top_k=int(image_attention_text_top_k),
                )
            elif image_attention_logp_static is not None:
                steered_logp_final = image_attention_logp_static
            else:
                steered_logp_final = image_logp
                image_attention_hook_final = {"applied": False, "reason": "missing_image_attention_static"}
            latent_summary_final = {
                **latent_summary_static,
                "latent_gamma": 0.0,
                "steered": False,
                "superseded_by_image_attention_boost": True,
            }
            layer_hook_final = {
                **layer_hook_static,
                "applied": False,
                "layer_gamma": 0.0,
                "superseded_by_image_attention_boost": True,
            } if layer_hook_static else {}
            layer_prior_subspace_hook_final = {
                **layer_prior_subspace_hook_static,
                "layer_alpha": 0.0,
                "applied": False,
                "superseded_by_image_attention_boost": True,
            } if layer_prior_subspace_hook_static else {}
            attention_prior_hook_final = {
                **attention_prior_hook_static,
                "applied": False,
                "attention_prior_alpha": 0.0,
                "superseded_by_image_attention_boost": True,
            } if attention_prior_hook_static else {}
            attention_visual_hook_final = {
                **attention_visual_hook_static,
                "applied": False,
                "attention_visual_alpha": 0.0,
                "superseded_by_image_attention_boost": True,
            } if attention_visual_hook_static else {}
            prior_subspace_summary_final = {
                **prior_subspace_summary_static,
                "alpha": 0.0,
                "steered": False,
                "superseded_by_image_attention_boost": True,
            }
        elif active and attention_visual_enabled and resolved_attention_visual_layer_index is not None:
            if attention_visual_alpha_eff != float(attention_visual_alpha):
                steered_logp_final, attention_visual_hook_final = _layer_attention_visual_delta_steered_logp(
                    model,
                    inputs_image,
                    inputs_prior,
                    image_logp,
                    prior_logp,
                    layer_index=int(resolved_attention_visual_layer_index),
                    alpha=attention_visual_alpha_eff,
                    head_top_k=int(attention_visual_head_top_k),
                )
            elif attention_visual_logp_static is not None:
                steered_logp_final = attention_visual_logp_static
            else:
                steered_logp_final = image_logp
                attention_visual_hook_final = {"applied": False, "reason": "missing_attention_visual_static"}
            latent_summary_final = {
                **latent_summary_static,
                "latent_gamma": 0.0,
                "steered": False,
                "superseded_by_attention_visual_delta": True,
            }
            layer_hook_final = {
                **layer_hook_static,
                "applied": False,
                "layer_gamma": 0.0,
                "superseded_by_attention_visual_delta": True,
            } if layer_hook_static else {}
            layer_prior_subspace_hook_final = {
                **layer_prior_subspace_hook_static,
                "layer_alpha": 0.0,
                "applied": False,
                "superseded_by_attention_visual_delta": True,
            } if layer_prior_subspace_hook_static else {}
            attention_prior_hook_final = {
                **attention_prior_hook_static,
                "applied": False,
                "attention_prior_alpha": 0.0,
                "superseded_by_attention_visual_delta": True,
            } if attention_prior_hook_static else {}
            prior_subspace_summary_final = {
                **prior_subspace_summary_static,
                "alpha": 0.0,
                "steered": False,
                "superseded_by_attention_visual_delta": True,
            }
        elif active and attention_prior_enabled and resolved_attention_prior_layer_index is not None:
            if attention_prior_alpha_eff != float(attention_prior_alpha):
                if int(attention_prior_head_top_k) > 0:
                    steered_logp_final, attention_prior_hook_final = _layer_attention_prior_head_steered_logp(
                        model,
                        inputs_image,
                        inputs_prior,
                        image_logp,
                        prior_logp,
                        layer_index=int(resolved_attention_prior_layer_index),
                        alpha=attention_prior_alpha_eff,
                        head_top_k=int(attention_prior_head_top_k),
                    )
                else:
                    steered_logp_final, attention_prior_hook_final = _layer_attention_prior_residual_steered_logp(
                        model,
                        inputs_image,
                        inputs_prior,
                        image_logp,
                        prior_logp,
                        layer_index=int(resolved_attention_prior_layer_index),
                        alpha=attention_prior_alpha_eff,
                    )
            elif attention_prior_logp_static is not None:
                steered_logp_final = attention_prior_logp_static
            else:
                steered_logp_final = image_logp
                attention_prior_hook_final = {"applied": False, "reason": "missing_attention_prior_static"}
            latent_summary_final = {
                **latent_summary_static,
                "latent_gamma": 0.0,
                "steered": False,
                "superseded_by_attention_prior_residual": True,
            }
            layer_hook_final = {
                **layer_hook_static,
                "applied": False,
                "layer_gamma": 0.0,
                "superseded_by_attention_prior_residual": True,
            } if layer_hook_static else {}
            layer_prior_subspace_hook_final = {
                **layer_prior_subspace_hook_static,
                "layer_alpha": 0.0,
                "applied": False,
                "superseded_by_attention_prior_residual": True,
            } if layer_prior_subspace_hook_static else {}
            prior_subspace_summary_final = {
                **prior_subspace_summary_static,
                "alpha": 0.0,
                "steered": False,
                "superseded_by_attention_prior_residual": True,
            }
        elif active and layer_steering_enabled and resolved_layer_index is not None:
            if layer_gamma_eff != float(layer_gamma):
                _, layer_residual_final = _orthogonal_visual_residual_from_outputs(
                    image_out,
                    prior_out,
                    state_index=int(resolved_layer_index) + 1,
                )
                if layer_residual_final is not None:
                    steered_logp_final, layer_hook_final = _layer_residual_steered_logp(
                        model,
                        inputs_image,
                        layer_index=int(resolved_layer_index),
                        residual=layer_residual_final,
                        layer_gamma=layer_gamma_eff,
                    )
                else:
                    steered_logp_final = image_logp
                    layer_hook_final = {"applied": False, "reason": "missing_layer_residual"}
            elif layer_logp_static is not None:
                steered_logp_final = layer_logp_static
            else:
                steered_logp_final = image_logp
                layer_hook_final = {"applied": False, "reason": "missing_layer_logp_static"}
            latent_summary_final = {**latent_summary_static, "latent_gamma": 0.0, "steered": False, "superseded_by_layer_steering": True}
            prior_subspace_summary_final = {
                **prior_subspace_summary_static,
                "alpha": 0.0,
                "steered": False,
                "superseded_by_layer_steering": True,
            }
            layer_prior_subspace_hook_final = {
                **layer_prior_subspace_hook_static,
                "layer_alpha": 0.0,
                "applied": False,
                "superseded_by_layer_steering": True,
            } if layer_prior_subspace_hook_static else {}
        elif active and layer_prior_subspace_enabled and resolved_prior_subspace_layer_index is not None:
            if layer_prior_subspace_alpha_eff != float(layer_prior_subspace_alpha):
                steered_logp_final, layer_prior_subspace_hook_final = _layer_prior_subspace_steered_logp(
                    model,
                    inputs_image,
                    prior_logp,
                    image_logp,
                    layer_index=int(resolved_prior_subspace_layer_index),
                    alpha=layer_prior_subspace_alpha_eff,
                    top_k=int(layer_prior_subspace_top_k),
                )
            elif layer_prior_subspace_logp_static is not None:
                steered_logp_final = layer_prior_subspace_logp_static
            else:
                steered_logp_final = image_logp
                layer_prior_subspace_hook_final = {"applied": False, "reason": "missing_layer_prior_subspace_static"}
            latent_summary_final = {
                **latent_summary_static,
                "latent_gamma": 0.0,
                "steered": False,
                "superseded_by_layer_prior_subspace": True,
            }
            prior_subspace_summary_final = {
                **prior_subspace_summary_static,
                "alpha": 0.0,
                "steered": False,
                "superseded_by_layer_prior_subspace": True,
            }
        elif active and prior_subspace_enabled:
            if prior_subspace_alpha_eff != float(prior_subspace_alpha):
                steered_logp_final, prior_subspace_summary_final = _prior_token_subspace_steered_logp(
                    model,
                    image_out,
                    prior_logp,
                    alpha=prior_subspace_alpha_eff,
                    top_k=int(prior_subspace_top_k),
                )
            else:
                steered_logp_final = prior_subspace_logp_static
            latent_summary_final = {
                **latent_summary_static,
                "latent_gamma": 0.0,
                "steered": False,
                "superseded_by_prior_subspace": True,
            }
        elif active and latent_gamma_eff != float(latent_gamma):
            latent_logp_final, latent_summary_final = _latent_steered_logp_from_outputs(
                model,
                image_out,
                prior_out,
                latent_gamma=latent_gamma_eff,
            )
            steered_logp_final = latent_logp_final
        elif active:
            steered_logp_final = latent_logp_static
            latent_summary_final = latent_summary_static
        else:
            steered_logp_final = image_logp
            latent_summary_final = {**latent_summary_static, "latent_gamma": 0.0, "steered": False}
            layer_hook_final = {**layer_hook_static, "applied": False, "layer_gamma": 0.0} if layer_hook_static else {}
            layer_prior_subspace_hook_final = {
                **layer_prior_subspace_hook_static,
                "applied": False,
                "layer_alpha": 0.0,
            } if layer_prior_subspace_hook_static else {}
            attention_prior_hook_final = {
                **attention_prior_hook_static,
                "applied": False,
                "attention_prior_alpha": 0.0,
            } if attention_prior_hook_static else {}
            attention_visual_hook_final = {
                **attention_visual_hook_static,
                "applied": False,
                "attention_visual_alpha": 0.0,
            } if attention_visual_hook_static else {}
            image_attention_hook_final = {
                **image_attention_hook_static,
                "applied": False,
                "image_attention_alpha": 0.0,
                "text_attention_alpha": 0.0,
            } if image_attention_hook_static else {}
            jspace_hook_final = {
                **jspace_hook_static,
                "applied": False,
                "jspace_alpha": 0.0,
                "jspace_gamma": 0.0,
            } if jspace_hook_static else {}
            prior_subspace_summary_final = {
                **prior_subspace_summary_static,
                "alpha": 0.0,
                "steered": False,
            }
        final_score = steered_logp_final - lambda_eff * prior_correction
        text_inertia_summary: Dict[str, Any] = {}
        if text_inertia_enabled:
            text_inertia_summary = _text_inertia_risk_summary(
                token_id=image_top,
                token_text=tokenizer.decode([image_top]),
                image_top=image_top,
                prior_top=prior_top,
                image_logp=image_logp,
                prior_logp=prior_logp,
                visual_attention=attention_summary,
                visual_attention_max=text_inertia_visual_attention_max,
                logprob_margin=text_inertia_logprob_margin,
                prior_logp_min=text_inertia_prior_logp_min,
                scope=text_inertia_scope,
            )
            text_inertia_summary["mode"] = text_inertia_mode
            text_inertia_summary["suppression_penalty"] = float(text_inertia_penalty)
            if text_inertia_mode == "suppress" and text_inertia_summary.get("risk"):
                final_score = final_score.clone()
                final_score[int(image_top)] = final_score[int(image_top)] - float(text_inertia_penalty)
                text_inertia_summary["suppressed"] = True
            else:
                text_inertia_summary["suppressed"] = False
        mask = torch.full_like(final_score, float("-inf"))
        mask[list(allowed_set)] = final_score[list(allowed_set)]
        next_id = int(torch.argmax(mask).detach().cpu().item())
        if dynamic_risk:
            risk_count += 1
        if active:
            active_token_count += 1
        if next_id != image_top:
            override_token_count += 1
        if len(steps) < max(0, int(trace_tokens_limit)):
            final_rows = sorted(
                _top_token_rows(tokenizer, image_logp, prior_logp, final_score, allowed_ids),
                key=lambda r: r["score"],
                reverse=True,
            )
            steps.append(
                {
                    "step": step_idx,
                    "gate": gate,
                    "gate_reason": gate_reason,
                    "lambda_eff": lambda_eff,
                    "latent_gamma_eff": latent_gamma_eff,
                    "layer_gamma_eff": layer_gamma_eff,
                    "prior_subspace_alpha_eff": prior_subspace_alpha_eff,
                    "layer_prior_subspace_alpha_eff": layer_prior_subspace_alpha_eff,
                    "attention_prior_alpha_eff": attention_prior_alpha_eff,
                    "attention_visual_alpha_eff": attention_visual_alpha_eff,
                    "image_attention_alpha_eff": image_attention_alpha_eff,
                    "image_attention_text_alpha_eff": image_attention_text_alpha_eff,
                    "jspace_alpha_eff": jspace_alpha_eff,
                    "jspace_gamma_eff": jspace_gamma_eff,
                    "hidden_strength": hidden_strength,
                    "hidden_risk": hidden_risk,
                    "dynamic_risk": dynamic_risk,
                    "visual_conflict_risk": visual_conflict_risk,
                    "prior_absorption_risk": prior_absorption_risk,
                    "prior_inertia_risk": prior_inertia_risk,
                    "image_top": image_top,
                    "image_top_token": tokenizer.decode([image_top]),
                    "prior_top": prior_top,
                    "prior_top_token": tokenizer.decode([prior_top]),
                    "prior_top_probability": prior_top_probability,
                    "image_prior_top_logprob_gap": image_prior_top_logprob_gap,
                    "static_top": static_top,
                    "static_top_token": tokenizer.decode([static_top]),
                    "final_token_id": next_id,
                    "final_token": tokenizer.decode([next_id]),
                    "image_margin": image_margin,
                    "prior_margin": prior_margin,
                    "static_margin": static_margin,
                    "hidden_delta": hidden_summary,
                    "prior_correction": prior_correction_summary,
                    "prior_token_subspace": prior_subspace_summary_final,
                    "orthogonal_visual": latent_summary_final,
                    "layer_orthogonal_visual": layer_summary_final,
                    "layer_hook": layer_hook_final,
                    "layer_prior_token_subspace": layer_prior_subspace_hook_final,
                    "attention_prior_residual": attention_prior_hook_final,
                    "attention_visual_delta": attention_visual_hook_final,
                    "image_attention_boost": image_attention_hook_final,
                    "jspace_prior_concept": jspace_hook_final,
                    "visual_attention": attention_summary,
                    "text_inertia": text_inertia_summary,
                    "top_image_tokens": image_rows[:5],
                    "top_static_tokens": static_rows[:5],
                    "top_final_tokens": final_rows[:5],
                }
            )
        if next_id in eos_ids:
            del image_out, prior_out
            break
        generated.append(next_id)
        inputs_image = _append_token_to_inputs(inputs_image, next_id)
        inputs_prior = _append_token_to_inputs(inputs_prior, next_id)
        del image_out, prior_out

    pred = tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    raw = {
        "backend": "local_transformers_qwen3_vl",
        "model": spec.model,
        "mitigation": MITIGATION_REVIS,
        "revis_mode": "token_hidden_delta",
        "lambda_prior": float(lambda_prior),
        "lambda_hidden": float(lambda_hidden),
        "gate": gate,
        "prior_margin_threshold": float(prior_margin_threshold),
        "contrast_margin_threshold": float(contrast_margin_threshold),
        "visual_conflict_image_margin": float(visual_conflict_image_margin),
        "absorption_image_margin_max": float(absorption_image_margin_max),
        "absmax_min": float(absmax_min),
        "prior_source": prior_source,
        "prior_degrade_mode": str(prior_degrade_mode),
        "prior_score_form": str(prior_score_form),
        "prior_inertia_gate": str(prior_inertia_gate),
        "prior_inertia_prob_min": float(prior_inertia_prob_min),
        "prior_inertia_logprob_margin": float(prior_inertia_logprob_margin),
        "prior_subspace_alpha": float(prior_subspace_alpha),
        "prior_subspace_top_k": int(prior_subspace_top_k),
        "layer_prior_subspace_alpha": float(layer_prior_subspace_alpha),
        "layer_prior_subspace_top_k": int(layer_prior_subspace_top_k),
        "layer_prior_subspace_index": int(layer_prior_subspace_index),
        "layer_prior_subspace_fraction": float(layer_prior_subspace_fraction),
        "resolved_layer_prior_subspace_index": resolved_prior_subspace_layer_index,
        "num_layer_prior_subspace_layers": num_prior_subspace_layers,
        "attention_prior_alpha": float(attention_prior_alpha),
        "attention_prior_layer_index": int(attention_prior_layer_index),
        "attention_prior_layer_fraction": float(attention_prior_layer_fraction),
        "attention_prior_head_top_k": int(attention_prior_head_top_k),
        "resolved_attention_prior_layer_index": resolved_attention_prior_layer_index,
        "num_attention_prior_layers": num_attention_prior_layers,
        "attention_visual_alpha": float(attention_visual_alpha),
        "attention_visual_layer_index": int(attention_visual_layer_index),
        "attention_visual_layer_fraction": float(attention_visual_layer_fraction),
        "attention_visual_head_top_k": int(attention_visual_head_top_k),
        "resolved_attention_visual_layer_index": resolved_attention_visual_layer_index,
        "num_attention_visual_layers": num_attention_visual_layers,
        "image_attention_alpha": float(image_attention_alpha),
        "image_attention_layer_index": int(image_attention_layer_index),
        "image_attention_layer_fraction": float(image_attention_layer_fraction),
        "image_attention_head_top_k": int(image_attention_head_top_k),
        "image_attention_head_select": str(image_attention_head_select),
        "image_attention_text_alpha": float(image_attention_text_alpha),
        "image_attention_text_top_k": int(image_attention_text_top_k),
        "resolved_image_attention_layer_index": resolved_image_attention_layer_index,
        "num_image_attention_layers": num_image_attention_layers,
        "jspace_alpha": float(jspace_alpha),
        "jspace_gamma": float(jspace_gamma),
        "jspace_top_k": int(jspace_top_k),
        "jspace_layer_index": int(jspace_layer_index),
        "jspace_layer_fraction": float(jspace_layer_fraction),
        "jspace_probe": jspace_probe,
        "jspace_lens": jspace_lens,
        "jspace_lens_path": str(jspace_lens_path or ""),
        "jspace_swap_alpha": float(jspace_swap_alpha),
        "resolved_jspace_layer_index": resolved_jspace_layer_index,
        "num_jspace_layers": num_jspace_layers,
        "latent_gamma": float(latent_gamma),
        "layer_gamma": float(layer_gamma),
        "layer_index": int(layer_index),
        "layer_fraction": float(layer_fraction),
        "resolved_layer_index": resolved_layer_index,
        "num_decoder_layers": num_decoder_layers,
        "attention_probe": attention_probe,
        "text_inertia_mode": text_inertia_mode,
        "text_inertia_visual_attention_max": float(text_inertia_visual_attention_max),
        "text_inertia_logprob_margin": float(text_inertia_logprob_margin),
        "text_inertia_prior_logp_min": float(text_inertia_prior_logp_min),
        "text_inertia_penalty": float(text_inertia_penalty),
        "text_inertia_scope": str(text_inertia_scope),
        "token_top_k": int(token_top_k),
        "token_candidate_set": "union(image_topk,prior_topk,jspace_or_image_attention_or_attention_or_hidden_steered_topk)",
        "prompt": prompt,
        "baseline_pred": baseline_pred,
        "generated_token_count": len(generated),
        "risk_token_count": risk_count,
        "active_token_count": active_token_count,
        "override_token_count": override_token_count,
        "single_image_or_no_image_contrast": True,
        "single_image_degraded_prior": prior_source == "degraded_image",
        "steps": steps,
    }
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return str(pred), raw, int((time.time() - t0) * 1000)


def _make_vcd_degraded_image(image: Any, mode: str) -> Any:
    from PIL import Image, ImageFilter, ImageEnhance

    mode = str(mode or "blur_downsample").strip().lower()
    if mode == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=4.0))
    if mode == "downsample":
        small = image.resize((max(1, image.width // 8), max(1, image.height // 8)), Image.Resampling.BICUBIC)
        return small.resize(image.size, Image.Resampling.BICUBIC)
    if mode == "low_contrast":
        return ImageEnhance.Contrast(image).enhance(0.35).filter(ImageFilter.GaussianBlur(radius=1.5))
    if mode == "blur_downsample":
        small = image.resize((max(1, image.width // 8), max(1, image.height // 8)), Image.Resampling.BICUBIC)
        return small.resize(image.size, Image.Resampling.BICUBIC).filter(ImageFilter.GaussianBlur(radius=2.0))
    raise ValueError(f"unknown vcd degrade mode: {mode}")


def _vcd_candidate_score(logp_image: float, logp_degraded: float, alpha: float) -> float:
    """Candidate-level analogue of official VCD token-logit contrast."""
    return float((1.0 + float(alpha)) * float(logp_image) - float(alpha) * float(logp_degraded))


def _mfcd_candidate_score(
    logp_image: float,
    logp_low_freq: float,
    logp_high_freq: float,
    alpha_low: float,
    alpha_high: float,
) -> float:
    """Candidate-level analogue of official MFCD multi-frequency contrast."""
    return float(
        (1.0 + float(alpha_low) + float(alpha_high)) * float(logp_image)
        - float(alpha_low) * float(logp_low_freq)
        - float(alpha_high) * float(logp_high_freq)
    )


def _pai_candidate_score(logp_image: float, logp_no_image: float, guidance_scale: float) -> float:
    """Candidate-level analogue of the official PAI CFG logits processor."""
    return float(
        float(guidance_scale) * float(logp_image)
        - (float(guidance_scale) - 1.0) * float(logp_no_image)
    )


def _candidate_plausibility_mask(
    rows: List[Dict[str, Any]],
    logp_key: str,
    beta: float,
) -> Dict[str, bool]:
    """Apply the official adaptive plausibility constraint over finite candidates."""
    if not rows:
        return {}
    beta = float(beta)
    if beta <= 0.0:
        return {str(row["key"]): True for row in rows}
    max_logp = max(float(row[logp_key]) for row in rows)
    log_beta = math.log(min(beta, 1.0))
    return {
        str(row["key"]): float(row[logp_key]) >= max_logp + log_beta
        for row in rows
    }


def _call_local_qwen3_vl_vcd_candidate(
    spec: ModelSpec,
    task: str,
    item: Dict[str, Any],
    image_path: str,
    baseline_pred: str,
    lambda_degrade: float,
    plausibility_beta: float,
    gate: str,
    contrast_margin_threshold: float,
    image_margin_threshold: float,
    degrade_mode: str,
) -> Tuple[str, Dict[str, Any], int]:
    try:
        import torch
        from PIL import Image
    except Exception as e:
        raise RuntimeError("vcd requires torch+Pillow in the current Python environment") from e

    if task not in ("qa", "mc"):
        raise ValueError("candidate VCD currently supports qa and mc")

    model, processor, family = _local_candidate_bundle(spec)
    image = Image.open(image_path).convert("RGB")
    if family == "qwen3_vl":
        image = image.resize((512, 512), Image.Resampling.LANCZOS)
    degraded = _make_vcd_degraded_image(image, degrade_mode)
    scoring_prompt = _build_candidate_scoring_prompt(task, item)
    candidates = _candidate_answers(task, item)
    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    for cand in candidates:
        image_score = _candidate_logprob_qwen3_vl(
            model, processor, scoring_prompt, image, cand["text"], family=family
        )
        degraded_score = _candidate_logprob_qwen3_vl(
            model, processor, scoring_prompt, degraded, cand["text"], family=family
        )
        score = _vcd_candidate_score(
            image_score["logprob"], degraded_score["logprob"], lambda_degrade
        )
        rows.append(
            {
                "key": cand["key"],
                "text": cand["text"],
                "logp_image": image_score["logprob"],
                "avg_logp_image": image_score["avg_logprob"],
                "logp_degraded": degraded_score["logprob"],
                "avg_logp_degraded": degraded_score["avg_logprob"],
                "vcd_score": score,
                "avg_vcd_score": _vcd_candidate_score(
                    image_score["avg_logprob"], degraded_score["avg_logprob"], lambda_degrade
                ),
                "image_token_count": image_score["token_count"],
                "degraded_token_count": degraded_score["token_count"],
                "image_token_ids": image_score["token_ids"],
                "degraded_token_ids": degraded_score["token_ids"],
            }
        )

    image_probs = _softmax_over([float(r["logp_image"]) for r in rows])
    degraded_probs = _softmax_over([float(r["logp_degraded"]) for r in rows])
    contrast_probs = _softmax_over([float(r["vcd_score"]) for r in rows])
    for row, p_img, p_deg, p_contrast in zip(rows, image_probs, degraded_probs, contrast_probs):
        row["p_image_candidates"] = p_img
        row["p_degraded_candidates"] = p_deg
        row["p_vcd_candidates"] = p_contrast

    by_image = sorted(rows, key=lambda r: float(r["logp_image"]), reverse=True)
    by_degraded = sorted(rows, key=lambda r: float(r["logp_degraded"]), reverse=True)
    plausible = _candidate_plausibility_mask(rows, "logp_image", plausibility_beta)
    for row in rows:
        row["plausible"] = plausible[str(row["key"])]
    plausible_rows = [row for row in rows if row["plausible"]] or rows
    by_contrast = sorted(plausible_rows, key=lambda r: float(r["vcd_score"]), reverse=True)
    baseline_key = _candidate_key_from_model_prediction(spec, task, baseline_pred)
    image_key = str(by_image[0]["key"]) if by_image else None
    degraded_key = str(by_degraded[0]["key"]) if by_degraded else None
    contrast_key = str(by_contrast[0]["key"]) if by_contrast else None
    image_margin = _margin(by_image, "logp_image")
    degraded_margin = _margin(by_degraded, "logp_degraded")
    contrast_margin = _margin(by_contrast, "vcd_score")
    visual_conflict = bool(
        image_key
        and degraded_key
        and contrast_key
        and image_key != degraded_key
        and contrast_key == image_key
        and (image_margin is None or image_margin >= float(image_margin_threshold))
    )
    enough_contrast = contrast_margin is None or contrast_margin >= float(contrast_margin_threshold)
    gate = str(gate or "visual_conflict").strip().lower()
    if gate == "always":
        should_override = bool(contrast_key and contrast_key != baseline_key and enough_contrast)
        gate_reason = "always_gate"
    elif gate == "visual_conflict":
        should_override = bool(visual_conflict and contrast_key and contrast_key != baseline_key and enough_contrast)
        gate_reason = "visual_conflict_supported" if visual_conflict else "no_degraded_view_visual_conflict"
    else:
        raise ValueError(f"unknown vcd gate: {gate}")

    final_key = contrast_key if should_override and contrast_key else baseline_key
    pred = _prediction_from_candidate_key(task, final_key) if final_key else baseline_pred
    raw = {
        "backend": "local_transformers_candidate_vlm",
        "model": spec.model,
        "family": family,
        "mitigation": MITIGATION_VCD,
        "vcd_mode": "candidate",
        "lambda_degrade": float(lambda_degrade),
        "alpha": float(lambda_degrade),
        "plausibility_beta": float(plausibility_beta),
        "official_score_form": "(1+alpha)*logp_image-alpha*logp_degraded",
        "degrade_mode": str(degrade_mode),
        "gate": gate,
        "gate_reason": gate_reason,
        "single_image_derived_view": True,
        "scoring_prompt": scoring_prompt,
        "baseline_pred": baseline_pred,
        "baseline_key": baseline_key,
        "image_top": image_key,
        "degraded_top": degraded_key,
        "contrast_top": contrast_key,
        "final_key": final_key,
        "overridden": should_override,
        "visual_conflict": visual_conflict,
        "image_margin": image_margin,
        "degraded_margin": degraded_margin,
        "contrast_margin": contrast_margin,
        "contrast_margin_threshold": float(contrast_margin_threshold),
        "image_margin_threshold": float(image_margin_threshold),
        "image_entropy": _entropy(image_probs),
        "degraded_entropy": _entropy(degraded_probs),
        "contrast_entropy": _entropy(contrast_probs),
        "candidates": rows,
    }
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pred, raw, int((time.time() - t0) * 1000)


def _call_local_qwen3_vl_pai_candidate(
    spec: ModelSpec,
    task: str,
    item: Dict[str, Any],
    image_path: str,
    baseline_pred: str,
    guidance_scale: float,
    plausibility_beta: float,
    contrast_margin_threshold: float,
) -> Tuple[str, Dict[str, Any], int]:
    try:
        import torch
        from PIL import Image
    except Exception as e:
        raise RuntimeError("pai requires torch+Pillow in the current Python environment") from e

    if task not in ("qa", "mc"):
        raise ValueError("candidate PAI currently supports qa and mc")
    if float(guidance_scale) < 1.0:
        raise ValueError("PAI guidance scale must be at least 1")

    model, processor, family = _local_candidate_bundle(spec)
    image = Image.open(image_path).convert("RGB")
    if family == "qwen3_vl":
        image = image.resize((512, 512), Image.Resampling.LANCZOS)
    scoring_prompt = _build_candidate_scoring_prompt(task, item)
    candidates = _candidate_answers(task, item)
    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    for cand in candidates:
        image_score = _candidate_logprob_qwen3_vl(
            model, processor, scoring_prompt, image, cand["text"], family=family
        )
        prior_score = _candidate_logprob_qwen3_vl(
            model, processor, scoring_prompt, None, cand["text"], family=family
        )
        pai_score = _pai_candidate_score(
            image_score["logprob"], prior_score["logprob"], guidance_scale
        )
        rows.append(
            {
                "key": cand["key"],
                "text": cand["text"],
                "logp_image": image_score["logprob"],
                "avg_logp_image": image_score["avg_logprob"],
                "logp_no_image": prior_score["logprob"],
                "avg_logp_no_image": prior_score["avg_logprob"],
                "pai_score": pai_score,
                "avg_pai_score": _pai_candidate_score(
                    image_score["avg_logprob"], prior_score["avg_logprob"], guidance_scale
                ),
                "image_token_count": image_score["token_count"],
                "no_image_token_count": prior_score["token_count"],
                "image_token_ids": image_score["token_ids"],
                "no_image_token_ids": prior_score["token_ids"],
            }
        )

    image_probs = _softmax_over([float(row["logp_image"]) for row in rows])
    prior_probs = _softmax_over([float(row["logp_no_image"]) for row in rows])
    pai_probs = _softmax_over([float(row["pai_score"]) for row in rows])
    plausible = _candidate_plausibility_mask(rows, "logp_image", plausibility_beta)
    for row, p_image, p_prior, p_pai in zip(rows, image_probs, prior_probs, pai_probs):
        row["p_image_candidates"] = p_image
        row["p_no_image_candidates"] = p_prior
        row["p_pai_candidates"] = p_pai
        row["plausible"] = plausible[str(row["key"])]

    plausible_rows = [row for row in rows if row["plausible"]] or rows
    by_image = sorted(rows, key=lambda row: float(row["logp_image"]), reverse=True)
    by_prior = sorted(rows, key=lambda row: float(row["logp_no_image"]), reverse=True)
    by_pai = sorted(plausible_rows, key=lambda row: float(row["pai_score"]), reverse=True)
    baseline_key = _candidate_key_from_model_prediction(spec, task, baseline_pred)
    pai_key = str(by_pai[0]["key"]) if by_pai else None
    pai_margin = _margin(by_pai, "pai_score")
    enough_contrast = pai_margin is None or pai_margin >= float(contrast_margin_threshold)
    should_override = bool(pai_key and pai_key != baseline_key and enough_contrast)
    final_key = pai_key if should_override else baseline_key
    pred = _prediction_from_candidate_key(task, final_key) if final_key else baseline_pred
    raw = {
        "backend": "local_transformers_candidate_vlm",
        "model": spec.model,
        "family": family,
        "mitigation": MITIGATION_PAI,
        "pai_mode": "candidate_cfg",
        "guidance_scale": float(guidance_scale),
        "plausibility_beta": float(plausibility_beta),
        "contrast_margin_threshold": float(contrast_margin_threshold),
        "official_score_form": "gamma*logp_image-(gamma-1)*logp_no_image",
        "attention_intervention": False,
        "single_image_or_no_image_contrast": True,
        "scoring_prompt": scoring_prompt,
        "baseline_pred": baseline_pred,
        "baseline_key": baseline_key,
        "image_top": str(by_image[0]["key"]) if by_image else None,
        "no_image_top": str(by_prior[0]["key"]) if by_prior else None,
        "contrast_top": pai_key,
        "final_key": final_key,
        "overridden": should_override,
        "contrast_margin": pai_margin,
        "image_entropy": _entropy(image_probs),
        "no_image_entropy": _entropy(prior_probs),
        "contrast_entropy": _entropy(pai_probs),
        "candidates": rows,
    }
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pred, raw, int((time.time() - t0) * 1000)


def _make_mfcd_high_frequency_image(image: Any, mode: str) -> Any:
    from PIL import ImageChops, ImageEnhance, ImageFilter, ImageOps

    mode = str(mode or "high_pass").strip().lower()
    if mode == "edges":
        edges = image.convert("L").filter(ImageFilter.FIND_EDGES)
        return ImageOps.autocontrast(edges).convert("RGB")
    if mode == "high_pass":
        blurred = image.filter(ImageFilter.GaussianBlur(radius=3.0))
        high = ImageChops.subtract(image, blurred, scale=2.0, offset=128)
        return ImageEnhance.Contrast(high).enhance(1.8)
    if mode == "sharpen":
        sharpened = ImageEnhance.Sharpness(image).enhance(3.0)
        return sharpened.filter(ImageFilter.EDGE_ENHANCE_MORE)
    raise ValueError(f"unknown mfcd high-frequency mode: {mode}")


def _call_local_qwen3_vl_mfcd_candidate(
    spec: ModelSpec,
    task: str,
    item: Dict[str, Any],
    image_path: str,
    baseline_pred: str,
    lambda_low: float,
    lambda_high: float,
    plausibility_beta: float,
    gate: str,
    contrast_margin_threshold: float,
    image_margin_threshold: float,
    high_margin_threshold: float,
    low_mode: str,
    high_mode: str,
) -> Tuple[str, Dict[str, Any], int]:
    try:
        import torch
        from PIL import Image
    except Exception as e:
        raise RuntimeError("mfcd requires torch+Pillow in the current Python environment") from e

    if task not in ("qa", "mc"):
        raise ValueError("candidate MFCD currently supports qa and mc")

    model, processor, family = _local_candidate_bundle(spec)
    image = Image.open(image_path).convert("RGB")
    if family == "qwen3_vl":
        image = image.resize((512, 512), Image.Resampling.LANCZOS)
    low_freq = _make_vcd_degraded_image(image, low_mode)
    high_freq = _make_mfcd_high_frequency_image(image, high_mode)
    scoring_prompt = _build_candidate_scoring_prompt(task, item)
    candidates = _candidate_answers(task, item)
    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    for cand in candidates:
        image_score = _candidate_logprob_qwen3_vl(
            model, processor, scoring_prompt, image, cand["text"], family=family
        )
        low_score = _candidate_logprob_qwen3_vl(
            model, processor, scoring_prompt, low_freq, cand["text"], family=family
        )
        high_score = _candidate_logprob_qwen3_vl(
            model, processor, scoring_prompt, high_freq, cand["text"], family=family
        )
        low_contrast = _vcd_candidate_score(
            image_score["logprob"], low_score["logprob"], lambda_low
        )
        mfcd_score = _mfcd_candidate_score(
            image_score["logprob"],
            low_score["logprob"],
            high_score["logprob"],
            lambda_low,
            lambda_high,
        )
        rows.append(
            {
                "key": cand["key"],
                "text": cand["text"],
                "logp_image": image_score["logprob"],
                "avg_logp_image": image_score["avg_logprob"],
                "logp_low_freq": low_score["logprob"],
                "avg_logp_low_freq": low_score["avg_logprob"],
                "logp_high_freq": high_score["logprob"],
                "avg_logp_high_freq": high_score["avg_logprob"],
                "low_contrast_score": low_contrast,
                "avg_low_contrast_score": _vcd_candidate_score(
                    image_score["avg_logprob"], low_score["avg_logprob"], lambda_low
                ),
                "mfcd_score": mfcd_score,
                "avg_mfcd_score": _mfcd_candidate_score(
                    image_score["avg_logprob"],
                    low_score["avg_logprob"],
                    high_score["avg_logprob"],
                    lambda_low,
                    lambda_high,
                ),
                "image_token_count": image_score["token_count"],
                "low_freq_token_count": low_score["token_count"],
                "high_freq_token_count": high_score["token_count"],
                "image_token_ids": image_score["token_ids"],
                "low_freq_token_ids": low_score["token_ids"],
                "high_freq_token_ids": high_score["token_ids"],
            }
        )

    image_probs = _softmax_over([float(r["logp_image"]) for r in rows])
    low_probs = _softmax_over([float(r["logp_low_freq"]) for r in rows])
    high_probs = _softmax_over([float(r["logp_high_freq"]) for r in rows])
    low_contrast_probs = _softmax_over([float(r["low_contrast_score"]) for r in rows])
    mfcd_probs = _softmax_over([float(r["mfcd_score"]) for r in rows])
    for row, p_img, p_low, p_high, p_low_contrast, p_mfcd in zip(
        rows, image_probs, low_probs, high_probs, low_contrast_probs, mfcd_probs
    ):
        row["p_image_candidates"] = p_img
        row["p_low_freq_candidates"] = p_low
        row["p_high_freq_candidates"] = p_high
        row["p_low_contrast_candidates"] = p_low_contrast
        row["p_mfcd_candidates"] = p_mfcd

    by_image = sorted(rows, key=lambda r: float(r["logp_image"]), reverse=True)
    by_low = sorted(rows, key=lambda r: float(r["logp_low_freq"]), reverse=True)
    by_high = sorted(rows, key=lambda r: float(r["logp_high_freq"]), reverse=True)
    plausible = _candidate_plausibility_mask(rows, "logp_image", plausibility_beta)
    for row in rows:
        row["plausible"] = plausible[str(row["key"])]
    plausible_rows = [row for row in rows if row["plausible"]] or rows
    by_low_contrast = sorted(plausible_rows, key=lambda r: float(r["low_contrast_score"]), reverse=True)
    by_mfcd = sorted(plausible_rows, key=lambda r: float(r["mfcd_score"]), reverse=True)
    baseline_key = _candidate_key_from_model_prediction(spec, task, baseline_pred)
    image_key = str(by_image[0]["key"]) if by_image else None
    low_key = str(by_low[0]["key"]) if by_low else None
    high_key = str(by_high[0]["key"]) if by_high else None
    low_contrast_key = str(by_low_contrast[0]["key"]) if by_low_contrast else None
    mfcd_key = str(by_mfcd[0]["key"]) if by_mfcd else None
    image_margin = _margin(by_image, "logp_image")
    low_margin = _margin(by_low, "logp_low_freq")
    high_margin = _margin(by_high, "logp_high_freq")
    low_contrast_margin = _margin(by_low_contrast, "low_contrast_score")
    mfcd_margin = _margin(by_mfcd, "mfcd_score")
    enough_contrast = mfcd_margin is None or mfcd_margin >= float(contrast_margin_threshold)
    image_margin_ok = image_margin is None or image_margin >= float(image_margin_threshold)
    high_margin_ok = high_margin is None or high_margin >= float(high_margin_threshold)
    visual_conflict = bool(
        image_key
        and low_key
        and mfcd_key
        and image_key != low_key
        and mfcd_key == image_key
        and image_margin_ok
    )
    frequency_rescue = bool(
        baseline_key
        and image_key
        and low_key
        and high_key
        and mfcd_key
        and baseline_key == image_key
        and image_key == low_key
        and high_key != baseline_key
        and mfcd_key == high_key
        and high_margin_ok
    )
    baseline_missing = baseline_key is None
    gate = str(gate or "frequency_conflict").strip().lower()
    if gate == "always":
        should_override = bool(mfcd_key and (baseline_missing or mfcd_key != baseline_key) and enough_contrast)
        gate_reason = "always_gate"
    elif gate == "visual_conflict":
        should_override = bool(
            visual_conflict and mfcd_key and (baseline_missing or mfcd_key != baseline_key) and enough_contrast
        )
        gate_reason = "visual_conflict_supported" if visual_conflict else "no_low_frequency_visual_conflict"
    elif gate == "frequency_conflict":
        should_override = bool(
            (visual_conflict or frequency_rescue)
            and mfcd_key
            and (baseline_missing or mfcd_key != baseline_key)
            and enough_contrast
        )
        gate_reason = (
            "visual_conflict_supported"
            if visual_conflict
            else "high_frequency_rescue_supported"
            if frequency_rescue
            else "no_frequency_conflict"
        )
    else:
        raise ValueError(f"unknown mfcd gate: {gate}")

    final_key = mfcd_key if should_override and mfcd_key else baseline_key
    pred = _prediction_from_candidate_key(task, final_key) if final_key else baseline_pred
    raw = {
        "backend": "local_transformers_candidate_vlm",
        "model": spec.model,
        "family": family,
        "mitigation": MITIGATION_MFCD,
        "mfcd_mode": "candidate_frequency",
        "lambda_low": float(lambda_low),
        "lambda_high": float(lambda_high),
        "alpha_low": float(lambda_low),
        "alpha_high": float(lambda_high),
        "plausibility_beta": float(plausibility_beta),
        "official_score_form": "(1+alpha_low+alpha_high)*logp_image-alpha_low*logp_low-alpha_high*logp_high",
        "low_mode": str(low_mode),
        "high_mode": str(high_mode),
        "gate": gate,
        "gate_reason": gate_reason,
        "single_image_derived_view": True,
        "scoring_prompt": scoring_prompt,
        "baseline_pred": baseline_pred,
        "baseline_key": baseline_key,
        "image_top": image_key,
        "low_freq_top": low_key,
        "high_freq_top": high_key,
        "low_contrast_top": low_contrast_key,
        "contrast_top": mfcd_key,
        "final_key": final_key,
        "overridden": should_override,
        "visual_conflict": visual_conflict,
        "frequency_rescue": frequency_rescue,
        "image_margin": image_margin,
        "low_freq_margin": low_margin,
        "high_freq_margin": high_margin,
        "low_contrast_margin": low_contrast_margin,
        "contrast_margin": mfcd_margin,
        "contrast_margin_threshold": float(contrast_margin_threshold),
        "image_margin_threshold": float(image_margin_threshold),
        "high_margin_threshold": float(high_margin_threshold),
        "image_entropy": _entropy(image_probs),
        "low_freq_entropy": _entropy(low_probs),
        "high_freq_entropy": _entropy(high_probs),
        "low_contrast_entropy": _entropy(low_contrast_probs),
        "contrast_entropy": _entropy(mfcd_probs),
        "candidates": rows,
    }
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pred, raw, int((time.time() - t0) * 1000)


def _call_local_qwen3_vl_chat(
    spec: ModelSpec,
    user_text: str,
    image_path: str,
    trace_generation: str = "none",
    top_logprobs: int = 0,
    trace_tokens_limit: int = 64,
    trace_hidden_states: bool = False,
    visual_probe: str = "none",
) -> Tuple[str, Dict[str, Any]]:
    try:
        import torch
        from PIL import Image
    except Exception as e:
        raise RuntimeError(
            "local transformers backend requires torch+Pillow in the current Python environment"
        ) from e

    model, processor = _local_qwen3_bundle(spec)
    image = Image.open(image_path).convert("RGB")
    image = image.resize((512, 512), Image.Resampling.LANCZOS)
    visual_probe_payload: Optional[Dict[str, Any]] = None
    visual_probe_error: Optional[str] = None
    if visual_probe != "none":
        try:
            if visual_probe == "image_delta":
                visual_probe_payload = _visual_delta_probe(model, processor, user_text, image)
            else:
                raise ValueError(f"unknown visual probe: {visual_probe}")
        except Exception as e:
            visual_probe_error = str(e)

    inputs = _qwen3_vl_inputs(model, processor, user_text, image)
    gen_kwargs: Dict[str, Any] = {"max_new_tokens": int(spec.max_tokens)}
    if float(spec.temperature) and float(spec.temperature) > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = float(spec.temperature)
    else:
        gen_kwargs["do_sample"] = False
    wants_generation_trace = trace_generation != "none" or int(top_logprobs or 0) > 0 or trace_hidden_states
    if wants_generation_trace:
        gen_kwargs["return_dict_in_generate"] = True
    if int(top_logprobs or 0) > 0:
        gen_kwargs["output_scores"] = True
    if trace_hidden_states:
        gen_kwargs["output_hidden_states"] = True

    with torch.inference_mode():
        generation = model.generate(**inputs, **gen_kwargs)
    generated_ids = generation.sequences if hasattr(generation, "sequences") else generation
    input_ids = inputs.get("input_ids")
    if input_ids is None:
        out_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        raw_minimal: Dict[str, Any] = {"backend": "local_transformers_qwen3_vl", "model": spec.model}
        if visual_probe_payload is not None:
            raw_minimal["visual_probe"] = visual_probe_payload
        if visual_probe_error:
            raw_minimal["visual_probe_error"] = visual_probe_error
        return str(out_text), raw_minimal

    trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, generated_ids)]
    out_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    raw: Dict[str, Any] = {"backend": "local_transformers_qwen3_vl", "model": spec.model}
    if visual_probe_payload is not None:
        raw["visual_probe"] = visual_probe_payload
    if visual_probe_error:
        raw["visual_probe_error"] = visual_probe_error
    if wants_generation_trace:
        token_ids = [int(x) for x in trimmed[0].detach().cpu().tolist()] if trimmed else []
        tokenizer = getattr(processor, "tokenizer", processor)
        trace: Dict[str, Any] = {
            "generated_token_count": len(token_ids),
            "generated_token_ids": token_ids[: max(0, int(trace_tokens_limit))],
            "generated_tokens": [tokenizer.decode([tok]) for tok in token_ids[: max(0, int(trace_tokens_limit))]],
        }
        scores = getattr(generation, "scores", None)
        if scores and int(top_logprobs or 0) > 0:
            log_probs = torch.log_softmax(scores[0][0].detach().float().cpu(), dim=-1)
            values, ids = torch.topk(log_probs, k=min(int(top_logprobs), int(log_probs.numel())))
            trace["first_token_top_logprobs"] = [
                {
                    "token_id": int(tok_id),
                    "token": tokenizer.decode([int(tok_id)]),
                    "logprob": float(logprob),
                    "prob": float(logprob.exp()),
                }
                for logprob, tok_id in zip(values, ids)
            ]
        hidden_states = getattr(generation, "hidden_states", None)
        if hidden_states:
            try:
                last_step = hidden_states[-1]
                last_layer = last_step[-1]
                last_token = last_layer[0, -1, :].detach().float().cpu()
                trace["hidden_state_summary"] = {
                    "generation_steps": len(hidden_states),
                    "layers": len(last_step),
                    "last_layer_last_token_norm": float(torch.linalg.norm(last_token).item()),
                    "last_layer_last_token_mean": float(last_token.mean().item()),
                    "last_layer_last_token_std": float(last_token.std().item()),
                }
            except Exception as e:
                trace["hidden_state_summary_error"] = str(e)
        raw["generation_trace"] = trace
    return str(out_text), raw


def _call_model(
    spec: ModelSpec,
    user_text: str,
    image_path: str,
    timeout_s: int,
    trace_generation: str = "none",
    top_logprobs: int = 0,
    trace_tokens_limit: int = 64,
    trace_hidden_states: bool = False,
    visual_probe: str = "none",
) -> Tuple[str, Dict[str, Any]]:
    if spec.backend in ("api", "vllm"):
        return _call_openai_compat_chat(
            spec,
            user_text=user_text,
            image_path=image_path,
            timeout_s=timeout_s,
            top_logprobs=top_logprobs,
        )
    if spec.backend == "transformers":
        if "llava" in f"{spec.name} {spec.model}".lower():
            return _call_local_generic_vlm_chat(spec, user_text=user_text, image_path=image_path)
        return _call_local_qwen3_vl_chat(
            spec,
            user_text=user_text,
            image_path=image_path,
            trace_generation=trace_generation,
            top_logprobs=top_logprobs,
            trace_tokens_limit=trace_tokens_limit,
            trace_hidden_states=trace_hidden_states,
            visual_probe=visual_probe,
        )
    raise ValueError(f"unknown backend: {spec.backend}")


def _call_local_generic_vlm_chat(
    spec: ModelSpec,
    user_text: str,
    image_path: str,
) -> Tuple[str, Dict[str, Any]]:
    import torch
    from PIL import Image

    model, processor, family = _local_candidate_bundle(spec)
    image = Image.open(image_path).convert("RGB")
    if family == "qwen3_vl":
        image = image.resize((512, 512), Image.Resampling.LANCZOS)
    inputs = _candidate_vlm_inputs(model, processor, family, user_text, image)
    generation_kwargs: Dict[str, Any] = {
        "max_new_tokens": int(spec.max_tokens),
        "do_sample": bool(float(spec.temperature) > 0),
    }
    if float(spec.temperature) > 0:
        generation_kwargs["temperature"] = float(spec.temperature)
    with torch.inference_mode():
        output = model.generate(**inputs, **generation_kwargs)
    generated = output[0, inputs["input_ids"].shape[1] :]
    text = processor.decode(generated, skip_special_tokens=True).strip()
    return text, {
        "backend": "local_transformers_generic_vlm",
        "model": spec.model,
        "family": family,
        "image_preprocessing": "fixed_512" if family == "qwen3_vl" else "model_native",
    }


def _call_local_official_revis(
    spec: ModelSpec,
    user_text: str,
    image_path: str,
    vector_file: str,
    calibration_file: str,
    repo_root: str,
    steering: bool,
    alpha_visual: float = 0.0,
    risk_gamma: float = 1.0,
) -> Tuple[str, Dict[str, Any]]:
    import torch
    from PIL import Image

    from official_revis_adapter import (
        attach_official_hook,
        build_inputs,
        load_model_bundle,
        resolve_revis_paper_alpha,
    )

    model_path = Path(spec.models_root) / spec.model
    load_id = str(model_path) if model_path.exists() else spec.model
    cache_key = str(Path(load_id).resolve()) if Path(load_id).exists() else load_id
    bundle = _OFFICIAL_REVIS_CACHE.get(cache_key)
    if bundle is None:
        bundle = load_model_bundle(load_id)
        _OFFICIAL_REVIS_CACHE[cache_key] = bundle
    model, processor, family = bundle
    resolved_alpha = (
        float(alpha_visual)
        if float(alpha_visual) > 0
        else resolve_revis_paper_alpha(family)
    )

    asset_key = f"{Path(vector_file).resolve()}::{Path(calibration_file).resolve()}"
    assets = _OFFICIAL_REVIS_ASSET_CACHE.get(asset_key)
    if assets is None:
        calibration = json.loads(Path(calibration_file).read_text(encoding="utf-8"))
        vectors = torch.load(vector_file, map_location="cpu", weights_only=True)
        assets = (calibration, vectors)
        _OFFICIAL_REVIS_ASSET_CACHE[asset_key] = assets
    calibration, vectors = assets
    selected = calibration.get("selected") or calibration
    if not isinstance(selected, dict) or selected.get("layer") is None or selected.get("tau") is None:
        raise ValueError(f"invalid official REVIS calibration file: {calibration_file}")
    layer = int(selected["layer"])
    tau = float(selected["tau"])
    image = Image.open(image_path).convert("RGB")
    inputs = build_inputs(model, processor, family, user_text, image)
    hook = None
    if steering:
        hook = attach_official_hook(
            model,
            vectors,
            layer,
            tau,
            Path(repo_root),
            alpha_visual=resolved_alpha,
            risk_gamma=float(risk_gamma),
        )
    try:
        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": int(spec.max_tokens),
            "do_sample": bool(float(spec.temperature) > 0),
        }
        if float(spec.temperature) > 0:
            generation_kwargs["temperature"] = float(spec.temperature)
        with torch.inference_mode():
            output = model.generate(**inputs, **generation_kwargs)
    finally:
        if hook is not None:
            hook.detach()
    generated = output[0, inputs["input_ids"].shape[1] :]
    text = processor.decode(generated, skip_special_tokens=True).strip()
    observation = dict(getattr(hook, "observation", {})) if hook is not None else {}
    if observation.get("positions"):
        observation["trigger_rate"] = (
            observation["triggered_positions"] / observation["positions"]
        )
        observation["risk_mean"] = observation["risk_sum"] / observation["positions"]
    return text, {
        "backend": "official_revis_compat",
        "model": spec.model,
        "family": family,
        "image_preprocessing": "model_native",
        "steering": bool(steering),
        "vector_file": vector_file,
        "calibration_file": calibration_file,
        "official_parameters": {
            "layer": layer,
            "vector_index": layer + 1,
            "tau_low": tau,
            "alpha_visual": resolved_alpha,
            "risk_gamma": float(risk_gamma),
            "profile": "paper_family" if float(alpha_visual) <= 0 else "explicit",
        },
        "mechanism": "official token-level risk-gated visual-vector injection",
        "gate_observation": observation,
    }


def _call_model_with_retry(
    spec: ModelSpec,
    user_text: str,
    image_path: str,
    timeout_s: int,
    retry: int,
    trace_generation: str = "none",
    top_logprobs: int = 0,
    trace_tokens_limit: int = 64,
    trace_hidden_states: bool = False,
    visual_probe: str = "none",
) -> Tuple[str, str, Dict[str, Any], int]:
    dt_ms = 0
    pred_text = ""
    raw_resp: Dict[str, Any] = {}
    status = "ok"
    for attempt in range(retry + 1):
        t0 = time.time()
        status = "ok"
        try:
            pred_text, raw_resp = _call_model(
                spec,
                user_text=user_text,
                image_path=image_path,
                timeout_s=timeout_s,
                trace_generation=trace_generation,
                top_logprobs=top_logprobs,
                trace_tokens_limit=trace_tokens_limit,
                trace_hidden_states=trace_hidden_states,
                visual_probe=visual_probe,
            )
            dt_ms = int((time.time() - t0) * 1000)
            break
        except Exception as e:
            status = "error"
            pred_text = str(e)
            raw_resp = {}
            dt_ms = int((time.time() - t0) * 1000)
            if attempt < retry:
                time.sleep(2 ** attempt)
                continue
    return status, pred_text, raw_resp, dt_ms


def _score(task: str, pred: str, gt: str) -> bool:
    if task == "qa":
        return _score_direct_qa(pred, gt)
    if task == "mc":
        return _score_multiple_choice(pred, gt)
    if task == "caption":
        return _score_caption(pred, gt)
    raise ValueError(f"unknown task: {task}")


def _collect_existing_keys(results_jsonl: str) -> set[Tuple[str, str, str]]:
    existing = set()
    for rec in _read_jsonl(results_jsonl):
        if rec.get("status") != "ok":
            continue
        pid = str(rec.get("pair_id") or "")
        task = str(rec.get("task") or "")
        side = str(rec.get("side") or "")
        if pid and task and side:
            existing.add((pid, task, side))
    return existing


def _load_official_revis_baseline_cache(
    path_value: str,
) -> Dict[Tuple[str, str, str], Tuple[str, Dict[str, Any]]]:
    if not str(path_value or "").strip():
        return {}
    rows = _read_jsonl(path_value)
    cached: Dict[Tuple[str, str, str], Tuple[str, Dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (
            str(row.get("pair_id") or ""),
            str(row.get("task") or ""),
            str(row.get("side") or ""),
        )
        revis = row.get("revis")
        raw = row.get("raw")
        if not all(key) or not isinstance(revis, dict) or not isinstance(raw, dict):
            continue
        prediction = revis.get("baseline_prediction")
        baseline_raw = raw.get("baseline_raw")
        if prediction is None or not isinstance(baseline_raw, dict):
            continue
        if key in cached:
            raise ValueError(f"duplicate official REVIS baseline cache key: {key}")
        cached[key] = (str(prediction), dict(baseline_raw))
    return cached


def _aggregate_metrics(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    total = 0
    correct = 0
    cf_total = 0
    cf_correct = 0
    cs_total = 0
    cs_correct = 0
    cf_errors = 0
    cf_commonsense_errors = 0
    caption_coverage_sum = 0.0
    caption_coverage_count = 0
    caption_critical_correct = 0
    caption_claim_coverage_sum = 0.0
    caption_claim_coverage_count = 0
    caption_prior_attraction_count = 0
    caption_prior_attraction_total = 0
    caption_forbidden_match_count = 0
    caption_forbidden_match_total = 0
    for r in records:
        if r.get("status") != "ok":
            continue
        total += 1
        if r.get("correct") is True:
            correct += 1
        caption_eval = r.get("caption_eval")
        if isinstance(caption_eval, dict):
            caption_coverage_sum += float(caption_eval.get("coverage") or 0.0)
            caption_coverage_count += 1
            if caption_eval.get("claim_coverage") is not None:
                caption_claim_coverage_sum += float(caption_eval.get("claim_coverage") or 0.0)
                caption_claim_coverage_count += 1
            if caption_eval.get("critical_match") is True:
                caption_critical_correct += 1
            if caption_eval.get("forbidden_match") is not None:
                caption_forbidden_match_total += 1
                if caption_eval.get("forbidden_match") is True:
                    caption_forbidden_match_count += 1
            if r.get("side") == "counterfactual" and caption_eval.get("prior_attraction") is not None:
                caption_prior_attraction_total += 1
                if caption_eval.get("prior_attraction") is True:
                    caption_prior_attraction_count += 1
        side = r.get("side")
        if side == "counterfactual":
            cf_total += 1
            if r.get("correct") is True:
                cf_correct += 1
            else:
                cf_errors += 1
                if r.get("commonsense_error") is True:
                    cf_commonsense_errors += 1
        elif side == "commonsense":
            cs_total += 1
            if r.get("correct") is True:
                cs_correct += 1

    cf_acc = (cf_correct / cf_total) if cf_total else None
    cs_acc = (cs_correct / cs_total) if cs_total else None
    gap = (cs_acc - cf_acc) if (cs_acc is not None and cf_acc is not None) else None
    ccr = (cf_commonsense_errors / cf_errors) if cf_errors else None
    rpd = ((cs_acc - cf_acc) / cs_acc) if (cs_acc is not None and cf_acc is not None and cs_acc not in (0, None)) else None
    return {
        "n_total": total,
        "n_cf": cf_total,
        "n_cs": cs_total,
        "CF_Acc": cf_acc,
        "CS_Acc": cs_acc,
        "Gap": gap,
        "CCR": ccr,
        "RPD": rpd,
        "Caption_Coverage": (caption_coverage_sum / caption_coverage_count) if caption_coverage_count else None,
        "Caption_Critical_Acc": (caption_critical_correct / caption_coverage_count) if caption_coverage_count else None,
        "Claim_Coverage": (caption_claim_coverage_sum / caption_claim_coverage_count) if caption_claim_coverage_count else None,
        "Critical_Claim_Acc": (caption_critical_correct / caption_coverage_count) if caption_coverage_count else None,
        "Prior_Attraction_Rate": (caption_prior_attraction_count / caption_prior_attraction_total) if caption_prior_attraction_total else None,
        "Forbidden_Claim_Rate": (caption_forbidden_match_count / caption_forbidden_match_total) if caption_forbidden_match_total else None,
    }


def _build_summary(all_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_task: Dict[str, List[Dict[str, Any]]] = {}
    for r in all_records:
        by_task.setdefault(str(r.get("task")), []).append(r)

    out: Dict[str, Any] = {"overall": {}, "by_category": {}, "by_subcategory": {}}

    for task, recs in by_task.items():
        out["overall"][task] = _aggregate_metrics(recs)

        cat_map: Dict[str, List[Dict[str, Any]]] = {}
        sub_map: Dict[str, List[Dict[str, Any]]] = {}
        for rr in recs:
            cat = str(rr.get("category") or "Unknown")
            sub = str(rr.get("subcategory") or "Unknown")
            cat_map.setdefault(cat, []).append(rr)
            sub_key = f"{cat} / {sub}"
            sub_map.setdefault(sub_key, []).append(rr)

        out["by_category"][task] = {k: _aggregate_metrics(v) for k, v in sorted(cat_map.items(), key=lambda x: x[0])}
        out["by_subcategory"][task] = {k: _aggregate_metrics(v) for k, v in sorted(sub_map.items(), key=lambda x: x[0])}

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default=str(BASE_DIR / "CDH-Bench.revised.strict.jsonl"))
    ap.add_argument("--images-root", default=str(BASE_DIR / "images"))
    ap.add_argument("--output-dir", default=str(BASE_DIR / "result"))
    ap.add_argument("--models", default="")
    ap.add_argument("--tasks", default="qa,mc")
    ap.add_argument(
        "--sides",
        default="commonsense,counterfactual",
        help="Comma-separated image sides to evaluate: commonsense,counterfactual",
    )
    ap.add_argument("--categories", default="", help="Comma-separated categories to evaluate")
    ap.add_argument("--subcategories", default="", help="Comma-separated subcategories to evaluate")
    ap.add_argument("--timeout-s", type=int, default=300)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--limit-per-subcategory", type=int, default=0, help="Limit the number of pairs per subcategory after filtering")
    ap.add_argument(
        "--pair-manifest",
        default="",
        help="Optional text/JSON pair list, or an exposure-ledger JSON used with --pair-split.",
    )
    ap.add_argument(
        "--pair-split",
        default="",
        help="Name under pair-manifest.split_manifests, such as cpr_unseen or mitigation_unseen.",
    )
    ap.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Deterministically partition the full (pair, task, side) work list before resume filtering.",
    )
    ap.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based shard selected from --shard-count.",
    )
    ap.add_argument("--parallel", type=int, default=1, help="Number of parallel API requests")
    ap.add_argument("--retry", type=int, default=3, help="Number of retries for failed requests")
    ap.add_argument("--progress-file", default="", help="Write machine-readable progress to this JSON file")
    ap.add_argument(
        "--mitigation",
        choices=SUPPORTED_MITIGATIONS,
        default=os.environ.get("CDH_EVAL_MITIGATION", MITIGATION_NONE),
        help="Optional single-image mitigation strategy. prompt_grounding is a one-pass fair prompt baseline; visual_evidence anchors on visible facts; option_entailment verifies MC options; cp_vbc applies commonsense-prior Bayes correction; vcd/mfcd run fair candidate view contrasts for QA/MC; revis applies risk-gated image/no-image hidden-delta steering.",
    )
    ap.add_argument(
        "--cp-vbc-lambda",
        type=float,
        default=float(os.environ.get("CDH_CP_VBC_LAMBDA", "0.6")),
        help="Text-prior subtraction weight for cp_vbc. A conservative 0.6 default avoids full-prior overcorrection.",
    )
    ap.add_argument(
        "--cp-vbc-lambda-policy",
        choices=(CP_VBC_LAMBDA_POLICY_FIXED, CP_VBC_LAMBDA_POLICY_BAYES_PATH),
        default=os.environ.get("CDH_CP_VBC_LAMBDA_POLICY", CP_VBC_LAMBDA_POLICY_FIXED),
        help="fixed uses one subtraction weight; bayes_path marginalizes candidate posteriors over a lambda interval.",
    )
    ap.add_argument(
        "--cp-vbc-path-lambda-low",
        type=float,
        default=float(os.environ.get("CDH_CP_VBC_PATH_LAMBDA_LOW", "0.8")),
        help="Lower endpoint of the Bayesian lambda path.",
    )
    ap.add_argument(
        "--cp-vbc-path-lambda-high",
        type=float,
        default=float(os.environ.get("CDH_CP_VBC_PATH_LAMBDA_HIGH", "1.2")),
        help="Upper endpoint of the Bayesian lambda path.",
    )
    ap.add_argument(
        "--cp-vbc-path-lambda-steps",
        type=int,
        default=int(os.environ.get("CDH_CP_VBC_PATH_LAMBDA_STEPS", "17")),
        help="Number of lambda values used to marginalize the candidate posterior.",
    )
    ap.add_argument(
        "--cp-vbc-path-stability",
        type=float,
        default=float(os.environ.get("CDH_CP_VBC_PATH_STABILITY", "0.5")),
        help="Minimum fraction of lambda values for which the path proposal remains top-ranked.",
    )
    ap.add_argument(
        "--cp-vbc-path-margin",
        type=float,
        default=float(os.environ.get("CDH_CP_VBC_PATH_MARGIN", "0.1")),
        help="Minimum posterior-mean probability margin for a Bayesian path override.",
    )
    ap.add_argument(
        "--cp-vbc-path-prior-relief",
        type=float,
        default=float(os.environ.get("CDH_CP_VBC_PATH_PRIOR_RELIEF", "0.0")),
        help="Minimum no-image log-probability relief from baseline to path proposal.",
    )
    ap.add_argument(
        "--cp-vbc-tasks",
        default=os.environ.get("CDH_CP_VBC_TASKS", "mc"),
        help="Comma-separated tasks where cp_vbc is allowed. Use caption with --cp-vbc-mode token for open generation.",
    )
    ap.add_argument(
        "--cp-vbc-mode",
        choices=(CP_VBC_MODE_CANDIDATE, CP_VBC_MODE_TOKEN),
        default=os.environ.get("CDH_CP_VBC_MODE", CP_VBC_MODE_CANDIDATE),
        help="candidate reranks finite answers; token applies risk-gated prior subtraction during generation.",
    )
    ap.add_argument(
        "--cp-vbc-risk-mode",
        choices=("dynamic", "static"),
        default=os.environ.get("CDH_CP_VBC_RISK_MODE", "dynamic"),
        help="dynamic subtracts the text prior only when prior-conflict risk is detected; static always applies subtraction.",
    )
    ap.add_argument(
        "--cp-vbc-prior-margin",
        type=float,
        default=float(os.environ.get("CDH_CP_VBC_PRIOR_MARGIN", "0.0")),
        help="Minimum text-only candidate margin required before cp_vbc can override.",
    )
    ap.add_argument(
        "--cp-vbc-contrast-margin",
        type=float,
        default=float(os.environ.get("CDH_CP_VBC_CONTRAST_MARGIN", "0.75")),
        help="Minimum contrastive candidate margin required before cp_vbc can override.",
    )
    ap.add_argument(
        "--cp-vbc-visual-conflict-image-margin",
        type=float,
        default=float(os.environ.get("CDH_CP_VBC_VISUAL_CONFLICT_IMAGE_MARGIN", "0.5")),
        help="Minimum image-only candidate margin for dynamic visual-conflict risk.",
    )
    ap.add_argument(
        "--cp-vbc-absorption-image-margin-max",
        type=float,
        default=float(os.environ.get("CDH_CP_VBC_ABSORPTION_IMAGE_MARGIN_MAX", "10.0")),
        help="Maximum image candidate margin for dynamic prior-absorption risk.",
    )
    ap.add_argument(
        "--cp-vbc-token-top-k",
        type=int,
        default=int(os.environ.get("CDH_CP_VBC_TOKEN_TOP_K", "50")),
        help="For token-mode cp_vbc, rerank only image-conditioned top-k tokens at each generation step.",
    )
    ap.add_argument(
        "--vcd-tasks",
        default=os.environ.get("CDH_VCD_TASKS", "qa,mc"),
        help="Comma-separated tasks where VCD-style candidate contrast is allowed. Current implementation supports qa,mc.",
    )
    ap.add_argument(
        "--vcd-lambda",
        type=float,
        default=float(os.environ.get("CDH_VCD_LAMBDA", "1.0")),
        help="Weight for subtracting degraded-view logprob in VCD-style candidate scoring.",
    )
    ap.add_argument(
        "--vcd-beta",
        type=float,
        default=float(os.environ.get("CDH_VCD_BETA", "0.1")),
        help="Adaptive plausibility threshold relative to the original-image candidate maximum.",
    )
    ap.add_argument(
        "--vcd-degrade-mode",
        choices=("blur_downsample", "blur", "downsample", "low_contrast"),
        default=os.environ.get("CDH_VCD_DEGRADE_MODE", "blur_downsample"),
        help="Single-image derived degraded view used by VCD-style contrast.",
    )
    ap.add_argument(
        "--vcd-gate",
        choices=("visual_conflict", "always"),
        default=os.environ.get("CDH_VCD_GATE", "visual_conflict"),
        help="visual_conflict only overrides when original and degraded image candidates disagree and contrast agrees with original.",
    )
    ap.add_argument(
        "--vcd-contrast-margin",
        type=float,
        default=float(os.environ.get("CDH_VCD_CONTRAST_MARGIN", "0.0")),
        help="Minimum VCD contrast margin required before override.",
    )
    ap.add_argument(
        "--vcd-image-margin",
        type=float,
        default=float(os.environ.get("CDH_VCD_IMAGE_MARGIN", "0.0")),
        help="Minimum original-image candidate margin for visual_conflict gate.",
    )
    ap.add_argument(
        "--pai-tasks",
        default=os.environ.get("CDH_PAI_TASKS", "qa,mc"),
        help="Comma-separated tasks where the PAI CFG candidate adapter is allowed.",
    )
    ap.add_argument(
        "--pai-guidance-scale",
        type=float,
        default=float(os.environ.get("CDH_PAI_GUIDANCE_SCALE", "1.1")),
        help="Official PAI classifier-free guidance scale gamma.",
    )
    ap.add_argument(
        "--pai-beta",
        type=float,
        default=float(os.environ.get("CDH_PAI_BETA", "0.1")),
        help="Official PAI adaptive plausibility threshold.",
    )
    ap.add_argument(
        "--pai-contrast-margin",
        type=float,
        default=float(os.environ.get("CDH_PAI_CONTRAST_MARGIN", "0.0")),
        help="Minimum finite-candidate PAI score margin required for override.",
    )
    ap.add_argument(
        "--mfcd-tasks",
        default=os.environ.get("CDH_MFCD_TASKS", "qa,mc"),
        help="Comma-separated tasks where MFCD-style candidate frequency contrast is allowed. Current implementation supports qa,mc.",
    )
    ap.add_argument(
        "--mfcd-lambda-low",
        type=float,
        default=float(os.environ.get("CDH_MFCD_LAMBDA_LOW", "1.0")),
        help="Weight for subtracting the low-frequency/degraded-view logprob in MFCD-style candidate scoring.",
    )
    ap.add_argument(
        "--mfcd-lambda-high",
        type=float,
        default=float(os.environ.get("CDH_MFCD_LAMBDA_HIGH", "0.5")),
        help="Weight for subtracting the high-frequency/edge-view logprob in MFCD-style candidate scoring.",
    )
    ap.add_argument(
        "--mfcd-beta",
        type=float,
        default=float(os.environ.get("CDH_MFCD_BETA", "0.3")),
        help="Adaptive plausibility threshold relative to the original-image candidate maximum.",
    )
    ap.add_argument(
        "--mfcd-low-mode",
        choices=("blur_downsample", "blur", "downsample", "low_contrast"),
        default=os.environ.get("CDH_MFCD_LOW_MODE", "blur_downsample"),
        help="Single-image derived low-frequency view used by MFCD-style contrast.",
    )
    ap.add_argument(
        "--mfcd-high-mode",
        choices=("high_pass", "edges", "sharpen"),
        default=os.environ.get("CDH_MFCD_HIGH_MODE", "high_pass"),
        help="Single-image derived high-frequency view used by MFCD-style contrast.",
    )
    ap.add_argument(
        "--mfcd-gate",
        choices=("visual_conflict", "frequency_conflict", "always"),
        default=os.environ.get("CDH_MFCD_GATE", "frequency_conflict"),
        help="frequency_conflict also permits a high-frequency rescue when original and low-frequency views agree with the baseline but the high-frequency view supports the contrastive top.",
    )
    ap.add_argument(
        "--mfcd-contrast-margin",
        type=float,
        default=float(os.environ.get("CDH_MFCD_CONTRAST_MARGIN", "0.0")),
        help="Minimum MFCD contrast margin required before override.",
    )
    ap.add_argument(
        "--mfcd-image-margin",
        type=float,
        default=float(os.environ.get("CDH_MFCD_IMAGE_MARGIN", "0.0")),
        help="Minimum original-image candidate margin for visual_conflict gate.",
    )
    ap.add_argument(
        "--mfcd-high-margin",
        type=float,
        default=float(os.environ.get("CDH_MFCD_HIGH_MARGIN", "0.0")),
        help="Minimum high-frequency candidate margin for frequency_rescue gate.",
    )
    ap.add_argument(
        "--revis-tasks",
        default=os.environ.get("CDH_REVIS_TASKS", "qa,mc,caption"),
        help="Comma-separated tasks where REVIS-style sparse steering is allowed.",
    )
    ap.add_argument(
        "--revis-mode",
        choices=("official", "auto", "candidate", "token"),
        default=os.environ.get("CDH_REVIS_MODE", "auto"),
        help="official uses the released dataset vector, calibrated layer/risk threshold, and token-level hook for every task. Legacy auto/candidate/token modes are retained for supplementary diagnostics.",
    )
    ap.add_argument(
        "--revis-official-vector-file",
        default=os.environ.get("CDH_REVIS_OFFICIAL_VECTOR_FILE", ""),
        help="Model-specific vector file produced with the pinned official REVIS extraction equation.",
    )
    ap.add_argument(
        "--revis-official-calibration-file",
        default=os.environ.get("CDH_REVIS_OFFICIAL_CALIBRATION_FILE", ""),
        help="Model-specific official backward-search layer and percentile threshold artifact.",
    )
    ap.add_argument(
        "--revis-official-repo-root",
        default=os.environ.get("CDH_REVIS_OFFICIAL_REPO_ROOT", "downloads/hallucination_refs/REVIS"),
        help="Pinned official REVIS repository snapshot used for PCA and the steering hook.",
    )
    ap.add_argument(
        "--revis-official-alpha",
        type=float,
        default=float(os.environ.get("CDH_REVIS_OFFICIAL_ALPHA", "0.0")),
        help="Official REVIS steering intensity. Zero selects the paper's family profile: Qwen=1.6, LLaVA=1.1. Use 2.5 only for the released entry-point default sensitivity run.",
    )
    ap.add_argument(
        "--revis-official-risk-gamma",
        type=float,
        default=float(os.environ.get("CDH_REVIS_OFFICIAL_RISK_GAMMA", "1.0")),
        help="Official REVIS risk cosine scale; the released implementation uses 1.0.",
    )
    ap.add_argument(
        "--revis-official-baseline-results",
        default=os.environ.get("CDH_REVIS_OFFICIAL_BASELINE_RESULTS", ""),
        help="Optional prior official-REVIS results.jsonl whose deterministic unsteered predictions are reused while changing only alpha.",
    )
    ap.add_argument(
        "--revis-lambda-prior",
        type=float,
        default=float(os.environ.get("CDH_REVIS_LAMBDA_PRIOR", "0.6")),
        help="Weight for subtracting no-image/text-prior logits or candidate logprob in REVIS-style steering.",
    )
    ap.add_argument(
        "--revis-lambda-hidden",
        type=float,
        default=float(os.environ.get("CDH_REVIS_LAMBDA_HIDDEN", "0.5")),
        help="Hidden-delta scaling for REVIS. Candidate mode adds hidden gain; token mode scales the sparse prior-subtraction strength.",
    )
    ap.add_argument(
        "--revis-gate",
        choices=("dynamic", "always"),
        default=os.environ.get("CDH_REVIS_GATE", "dynamic"),
        help="dynamic activates only under prior-attraction risk; always applies hidden-delta-gated steering whenever absmax passes.",
    )
    ap.add_argument(
        "--revis-prior-margin",
        type=float,
        default=float(os.environ.get("CDH_REVIS_PRIOR_MARGIN", "0.0")),
        help="Minimum no-image prior margin for REVIS prior-attraction risk.",
    )
    ap.add_argument(
        "--revis-contrast-margin",
        type=float,
        default=float(os.environ.get("CDH_REVIS_CONTRAST_MARGIN", "0.75")),
        help="Minimum REVIS contrast margin before override/steering is accepted.",
    )
    ap.add_argument(
        "--revis-visual-conflict-image-margin",
        type=float,
        default=float(os.environ.get("CDH_REVIS_VISUAL_CONFLICT_IMAGE_MARGIN", "0.5")),
        help="Minimum image-conditioned margin for REVIS visual-conflict risk.",
    )
    ap.add_argument(
        "--revis-absorption-image-margin-max",
        type=float,
        default=float(os.environ.get("CDH_REVIS_ABSORPTION_IMAGE_MARGIN_MAX", "10.0")),
        help="Maximum image-conditioned margin for REVIS prior-absorption risk.",
    )
    ap.add_argument(
        "--revis-absmax-min",
        type=float,
        default=float(os.environ.get("CDH_REVIS_ABSMAX_MIN", "0.0")),
        help="Minimum max relative hidden delta between image and no-image passes before REVIS can activate.",
    )
    ap.add_argument(
        "--revis-prior-source",
        choices=("no_image", "degraded_image"),
        default=os.environ.get("CDH_REVIS_PRIOR_SOURCE", "no_image"),
        help="Token-mode REVIS contrast source: text-only/no-image prior or a degraded view derived from the same image.",
    )
    ap.add_argument(
        "--revis-prior-degrade-mode",
        default=os.environ.get("CDH_REVIS_PRIOR_DEGRADE_MODE", "blur_downsample"),
        help="Image degradation mode when --revis-prior-source degraded_image.",
    )
    ap.add_argument(
        "--revis-prior-score",
        choices=("logprob", "centered_logit", "zscore_logit", "positive_zscore_logit"),
        default=os.environ.get("CDH_REVIS_PRIOR_SCORE", "logprob"),
        help="Token-mode REVIS prior term. Logit forms subtract the no-image prior energy directly instead of subtracting normalized logprob.",
    )
    ap.add_argument(
        "--revis-prior-inertia-gate",
        choices=("none", "agreement"),
        default=os.environ.get("CDH_REVIS_PRIOR_INERTIA_GATE", "none"),
        help="Additional token-mode gate for prior absorption when image and no-image distributions agree too strongly.",
    )
    ap.add_argument(
        "--revis-prior-inertia-prob-min",
        type=float,
        default=float(os.environ.get("CDH_REVIS_PRIOR_INERTIA_PROB_MIN", "0.5")),
        help="Minimum no-image top-token probability for prior-inertia risk.",
    )
    ap.add_argument(
        "--revis-prior-inertia-logprob-margin",
        type=float,
        default=float(os.environ.get("CDH_REVIS_PRIOR_INERTIA_LOGPROB_MARGIN", "0.25")),
        help="Maximum absolute image-vs-prior logprob gap at the shared top token for prior-inertia risk.",
    )
    ap.add_argument(
        "--revis-prior-subspace-alpha",
        type=float,
        default=float(os.environ.get("CDH_REVIS_PRIOR_SUBSPACE_ALPHA", "0.0")),
        help="Token-mode hidden-level suppression strength for the prior-token embedding subspace before lm_head.",
    )
    ap.add_argument(
        "--revis-prior-subspace-top-k",
        type=int,
        default=int(os.environ.get("CDH_REVIS_PRIOR_SUBSPACE_TOP_K", "8")),
        help="Number of prior top tokens used to form the hidden-level prior-token embedding subspace.",
    )
    ap.add_argument(
        "--revis-layer-prior-subspace-alpha",
        type=float,
        default=float(os.environ.get("CDH_REVIS_LAYER_PRIOR_SUBSPACE_ALPHA", "0.0")),
        help="Decoder-layer hidden suppression strength for the prior-token embedding subspace before the critical token reaches lm_head.",
    )
    ap.add_argument(
        "--revis-layer-prior-subspace-top-k",
        type=int,
        default=int(os.environ.get("CDH_REVIS_LAYER_PRIOR_SUBSPACE_TOP_K", "1")),
        help="Number of prior top tokens used for decoder-layer prior-token subspace suppression.",
    )
    ap.add_argument(
        "--revis-layer-prior-subspace-index",
        type=int,
        default=int(os.environ.get("CDH_REVIS_LAYER_PRIOR_SUBSPACE_INDEX", "-1")),
        help="Decoder layer index for prior-token subspace suppression. Use -1 to select by --revis-layer-prior-subspace-fraction.",
    )
    ap.add_argument(
        "--revis-layer-prior-subspace-fraction",
        type=float,
        default=float(os.environ.get("CDH_REVIS_LAYER_PRIOR_SUBSPACE_FRACTION", "0.5")),
        help="Decoder depth fraction for prior-token subspace suppression when --revis-layer-prior-subspace-index is -1.",
    )
    ap.add_argument(
        "--revis-attention-prior-alpha",
        type=float,
        default=float(os.environ.get("CDH_REVIS_ATTENTION_PRIOR_ALPHA", "0.0")),
        help="Decoder self-attention residual suppression strength using the no-image/degraded prior-source attention residual.",
    )
    ap.add_argument(
        "--revis-attention-prior-layer-index",
        type=int,
        default=int(os.environ.get("CDH_REVIS_ATTENTION_PRIOR_LAYER_INDEX", "-1")),
        help="Decoder layer index for attention-prior residual suppression. Use -1 to select by --revis-attention-prior-layer-fraction.",
    )
    ap.add_argument(
        "--revis-attention-prior-layer-fraction",
        type=float,
        default=float(os.environ.get("CDH_REVIS_ATTENTION_PRIOR_LAYER_FRACTION", "0.9")),
        help="Decoder depth fraction for attention-prior residual suppression when --revis-attention-prior-layer-index is -1.",
    )
    ap.add_argument(
        "--revis-attention-prior-head-top-k",
        type=int,
        default=int(os.environ.get("CDH_REVIS_ATTENTION_PRIOR_HEAD_TOP_K", "0")),
        help="If positive, suppress only the top-k prior-dominant attention heads before self-attention o_proj instead of the whole attention residual.",
    )
    ap.add_argument(
        "--revis-attention-visual-alpha",
        type=float,
        default=float(os.environ.get("CDH_REVIS_ATTENTION_VISUAL_ALPHA", "0.0")),
        help="Decoder attention head visual-delta boost strength using image minus no-image/degraded o_proj inputs.",
    )
    ap.add_argument(
        "--revis-attention-visual-layer-index",
        type=int,
        default=int(os.environ.get("CDH_REVIS_ATTENTION_VISUAL_LAYER_INDEX", "-1")),
        help="Decoder layer index for attention visual-delta boost. Use -1 to select by --revis-attention-visual-layer-fraction.",
    )
    ap.add_argument(
        "--revis-attention-visual-layer-fraction",
        type=float,
        default=float(os.environ.get("CDH_REVIS_ATTENTION_VISUAL_LAYER_FRACTION", "0.9")),
        help="Decoder depth fraction for attention visual-delta boost when --revis-attention-visual-layer-index is -1.",
    )
    ap.add_argument(
        "--revis-attention-visual-head-top-k",
        type=int,
        default=int(os.environ.get("CDH_REVIS_ATTENTION_VISUAL_HEAD_TOP_K", "4")),
        help="Number of highest visual-delta attention heads to boost before self-attention o_proj.",
    )
    ap.add_argument(
        "--revis-image-attention-alpha",
        type=float,
        default=float(os.environ.get("CDH_REVIS_IMAGE_ATTENTION_ALPHA", "0.0")),
        help="Pre-softmax attention-logit boost applied from the current token to image tokens in a selected decoder layer.",
    )
    ap.add_argument(
        "--revis-image-attention-layer-index",
        type=int,
        default=int(os.environ.get("CDH_REVIS_IMAGE_ATTENTION_LAYER_INDEX", "-1")),
        help="Decoder layer index for image-token attention boost. Use -1 to select by --revis-image-attention-layer-fraction.",
    )
    ap.add_argument(
        "--revis-image-attention-layer-fraction",
        type=float,
        default=float(os.environ.get("CDH_REVIS_IMAGE_ATTENTION_LAYER_FRACTION", "0.9")),
        help="Decoder depth fraction for image-token attention boost when --revis-image-attention-layer-index is -1.",
    )
    ap.add_argument(
        "--revis-image-attention-head-top-k",
        type=int,
        default=int(os.environ.get("CDH_REVIS_IMAGE_ATTENTION_HEAD_TOP_K", "4")),
        help="Number of attention heads selected for image-token attention boost. Use 0 for all heads.",
    )
    ap.add_argument(
        "--revis-image-attention-head-select",
        choices=("low_visual", "high_visual", "high_text", "all"),
        default=os.environ.get("CDH_REVIS_IMAGE_ATTENTION_HEAD_SELECT", "low_visual"),
        help="Head selection strategy for image-token attention boost.",
    )
    ap.add_argument(
        "--revis-image-attention-text-alpha",
        type=float,
        default=float(os.environ.get("CDH_REVIS_IMAGE_ATTENTION_TEXT_ALPHA", "0.0")),
        help="Pre-softmax attention-logit penalty applied to high-attention non-image text/prefix source positions on selected heads.",
    )
    ap.add_argument(
        "--revis-image-attention-text-top-k",
        type=int,
        default=int(os.environ.get("CDH_REVIS_IMAGE_ATTENTION_TEXT_TOP_K", "16")),
        help="Number of non-image text/prefix source positions suppressed per selected head. Use 0 to suppress all non-image sources.",
    )
    ap.add_argument(
        "--revis-jspace-alpha",
        type=float,
        default=float(os.environ.get("CDH_REVIS_JSPACE_ALPHA", "0.0")),
        help="Sparse logit-lens/J-space prior concept suppression strength at a decoder layer.",
    )
    ap.add_argument(
        "--revis-jspace-gamma",
        type=float,
        default=float(os.environ.get("CDH_REVIS_JSPACE_GAMMA", "0.0")),
        help="Orthogonal image-minus-prior residual injection strength paired with J-space prior suppression.",
    )
    ap.add_argument(
        "--revis-jspace-top-k",
        type=int,
        default=int(os.environ.get("CDH_REVIS_JSPACE_TOP_K", "4")),
        help="Number of prior-source logit-lens concepts used for sparse J-space suppression.",
    )
    ap.add_argument(
        "--revis-jspace-layer-index",
        type=int,
        default=int(os.environ.get("CDH_REVIS_JSPACE_LAYER_INDEX", "-1")),
        help="Decoder layer index for J-space prior concept suppression. Use -1 to select by --revis-jspace-layer-fraction.",
    )
    ap.add_argument(
        "--revis-jspace-layer-fraction",
        type=float,
        default=float(os.environ.get("CDH_REVIS_JSPACE_LAYER_FRACTION", "0.5")),
        help="Decoder depth fraction for J-space prior concept suppression when --revis-jspace-layer-index is -1.",
    )
    ap.add_argument(
        "--revis-jspace-probe",
        choices=("none", "summary"),
        default=os.environ.get("CDH_REVIS_JSPACE_PROBE", "none"),
        help="Record a logit-lens J-space readout at the selected layer without requiring an intervention.",
    )
    ap.add_argument(
        "--revis-jspace-lens",
        choices=("logit_lens", "local_jacobian", "fitted_jacobian"),
        default=os.environ.get("CDH_REVIS_JSPACE_LENS", "logit_lens"),
        help="Use logit-lens fallback vectors or a context-local Jacobian direction for J-space prior suppression.",
    )
    ap.add_argument(
        "--revis-jspace-lens-path",
        default=os.environ.get("CDH_REVIS_JSPACE_LENS_PATH", ""),
        help="Path to a fitted JacobianLens checkpoint used when --revis-jspace-lens fitted_jacobian.",
    )
    ap.add_argument(
        "--revis-jspace-swap-alpha",
        type=float,
        default=float(os.environ.get("CDH_REVIS_JSPACE_SWAP_ALPHA", "0.0")),
        help="When using a fitted J-lens, move prior concept mass toward the top image-only J-space concept.",
    )
    ap.add_argument(
        "--revis-latent-gamma",
        type=float,
        default=float(os.environ.get("CDH_REVIS_LATENT_GAMMA", "0.0")),
        help="Weight for REVIS-style orthogonal visual residual steering at the final decoder hidden state in token mode.",
    )
    ap.add_argument(
        "--revis-layer-gamma",
        type=float,
        default=float(os.environ.get("CDH_REVIS_LAYER_GAMMA", "0.0")),
        help="Weight for REVIS-style orthogonal visual residual injection at an intermediate decoder layer in token mode.",
    )
    ap.add_argument(
        "--revis-layer-index",
        type=int,
        default=int(os.environ.get("CDH_REVIS_LAYER_INDEX", "-1")),
        help="Decoder layer index for layer-wise REVIS token steering. Use -1 to select by --revis-layer-fraction.",
    )
    ap.add_argument(
        "--revis-layer-fraction",
        type=float,
        default=float(os.environ.get("CDH_REVIS_LAYER_FRACTION", "0.5")),
        help="Decoder depth fraction for layer-wise REVIS token steering when --revis-layer-index is -1.",
    )
    ap.add_argument(
        "--revis-attention-probe",
        choices=("none", "summary"),
        default=os.environ.get("CDH_REVIS_ATTENTION_PROBE", "none"),
        help="Record a lightweight decoder attention summary from the current token to image tokens in token-mode REVIS.",
    )
    ap.add_argument(
        "--revis-text-inertia-mode",
        choices=("none", "trace", "suppress"),
        default=os.environ.get("CDH_REVIS_TEXT_INERTIA_MODE", "none"),
        help="Trace or suppress text-inertia tokens using image/no-image logits plus visual attention. Suppression is experimental.",
    )
    ap.add_argument(
        "--revis-text-inertia-visual-attention-max",
        type=float,
        default=float(os.environ.get("CDH_REVIS_TEXT_INERTIA_VISUAL_ATTENTION_MAX", "0.05")),
        help="Maximum mean visual attention for a token to be treated as text-inertia risk.",
    )
    ap.add_argument(
        "--revis-text-inertia-logprob-margin",
        type=float,
        default=float(os.environ.get("CDH_REVIS_TEXT_INERTIA_LOGPROB_MARGIN", "0.25")),
        help="Maximum absolute image-vs-prior logprob gap for text-inertia risk.",
    )
    ap.add_argument(
        "--revis-text-inertia-prior-logp-min",
        type=float,
        default=float(os.environ.get("CDH_REVIS_TEXT_INERTIA_PRIOR_LOGP_MIN", "-0.2")),
        help="Minimum no-image prior logprob for text-inertia risk.",
    )
    ap.add_argument(
        "--revis-text-inertia-penalty",
        type=float,
        default=float(os.environ.get("CDH_REVIS_TEXT_INERTIA_PENALTY", "2.0")),
        help="Logit/logprob penalty applied to a text-inertia token when --revis-text-inertia-mode suppress is enabled.",
    )
    ap.add_argument(
        "--revis-text-inertia-scope",
        choices=("all", "content", "count"),
        default=os.environ.get("CDH_REVIS_TEXT_INERTIA_SCOPE", "all"),
        help="Token scope for text-inertia risk. Use count for count-focused diagnostic suppression.",
    )
    ap.add_argument(
        "--revis-hidden-margin",
        type=float,
        default=float(os.environ.get("CDH_REVIS_HIDDEN_MARGIN", "0.0")),
        help="Minimum hidden-gain margin for candidate-mode hidden rescue.",
    )
    ap.add_argument(
        "--revis-token-top-k",
        type=int,
        default=int(os.environ.get("CDH_REVIS_TOKEN_TOP_K", "50")),
        help="For token-mode REVIS, rerank only image-conditioned top-k tokens at each generation step.",
    )
    ap.add_argument(
        "--generation-trace",
        choices=("none", "summary"),
        default=os.environ.get("CDH_EVAL_GENERATION_TRACE", "none"),
        help="Record lightweight generation traces for local transformers inference.",
    )
    ap.add_argument(
        "--top-logprobs",
        type=int,
        default=int(os.environ.get("CDH_EVAL_TOP_LOGPROBS", "0")),
        help="Record top-k first-token logprobs when supported. Use with vLLM OpenAI API or local transformers.",
    )
    ap.add_argument(
        "--trace-tokens-limit",
        type=int,
        default=int(os.environ.get("CDH_EVAL_TRACE_TOKENS_LIMIT", "64")),
        help="Maximum number of generated tokens to keep in a trace.",
    )
    ap.add_argument(
        "--trace-hidden-states",
        action="store_true",
        default=os.environ.get("CDH_EVAL_TRACE_HIDDEN_STATES", "0") == "1",
        help="For local transformers only, request hidden states and store a compact summary. This is memory intensive.",
    )
    ap.add_argument(
        "--visual-probe",
        choices=("none", "image_delta"),
        default=os.environ.get("CDH_EVAL_VISUAL_PROBE", "none"),
        help="For local transformers only, compare hidden states for the same prompt with image vs no image.",
    )
    args = ap.parse_args()

    if args.shard_count < 1:
        raise SystemExit("shard-count must be at least 1")
    if not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("shard-index must be in [0, shard-count)")

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    for t in tasks:
        if t not in ("qa", "mc", "caption"):
            raise SystemExit(f"invalid task: {t}")
    sides = _parse_csv_list(args.sides)
    if not sides or any(side not in {"commonsense", "counterfactual"} for side in sides):
        raise SystemExit("invalid sides: use commonsense and/or counterfactual")
    sides = list(dict.fromkeys(sides))
    cp_vbc_tasks = set(_parse_csv_list(args.cp_vbc_tasks))
    for t in cp_vbc_tasks:
        if t not in ("qa", "mc", "caption"):
            raise SystemExit(f"invalid cp_vbc task: {t}")
    if args.cp_vbc_mode == CP_VBC_MODE_CANDIDATE and "caption" in cp_vbc_tasks:
        raise SystemExit("caption cp_vbc requires --cp-vbc-mode token")
    if args.cp_vbc_lambda_policy == CP_VBC_LAMBDA_POLICY_BAYES_PATH:
        if args.cp_vbc_path_lambda_high < args.cp_vbc_path_lambda_low:
            raise SystemExit("cp-vbc Bayesian path requires lambda-high >= lambda-low")
        if args.cp_vbc_path_lambda_steps < 2:
            raise SystemExit("cp-vbc Bayesian path requires at least two lambda steps")
        if not 0.0 <= args.cp_vbc_path_stability <= 1.0:
            raise SystemExit("cp-vbc path stability must be in [0, 1]")
    vcd_tasks = set(_parse_csv_list(args.vcd_tasks))
    for t in vcd_tasks:
        if t not in ("qa", "mc"):
            raise SystemExit(f"invalid vcd task for candidate implementation: {t}")
    pai_tasks = set(_parse_csv_list(args.pai_tasks))
    for t in pai_tasks:
        if t not in ("qa", "mc"):
            raise SystemExit(f"invalid pai task for candidate implementation: {t}")
    if args.pai_guidance_scale < 1.0:
        raise SystemExit("PAI guidance scale must be at least 1")
    mfcd_tasks = set(_parse_csv_list(args.mfcd_tasks))
    for t in mfcd_tasks:
        if t not in ("qa", "mc"):
            raise SystemExit(f"invalid mfcd task for candidate implementation: {t}")
    revis_tasks = set(_parse_csv_list(args.revis_tasks))
    for t in revis_tasks:
        if t not in ("qa", "mc", "caption"):
            raise SystemExit(f"invalid revis task: {t}")
    if args.revis_mode == "candidate" and "caption" in revis_tasks:
        raise SystemExit("caption REVIS requires --revis-mode auto or --revis-mode token")
    if args.mitigation == MITIGATION_REVIS and args.revis_mode == "official":
        if not args.revis_official_vector_file:
            raise SystemExit("official REVIS requires --revis-official-vector-file")
        if not args.revis_official_calibration_file:
            raise SystemExit("official REVIS requires --revis-official-calibration-file")
        for required_path in (
            args.revis_official_vector_file,
            args.revis_official_calibration_file,
            args.revis_official_repo_root,
        ):
            if not Path(required_path).exists():
                raise SystemExit(f"official REVIS path does not exist: {required_path}")
        if args.revis_official_baseline_results and not Path(
            args.revis_official_baseline_results
        ).is_file():
            raise SystemExit(
                "official REVIS baseline-results path does not exist: "
                f"{args.revis_official_baseline_results}"
            )

    loader = CDHBenchLoader(args.jsonl)
    items = loader.data
    categories = set(_parse_csv_list(args.categories))
    subcategories = set(_parse_csv_list(args.subcategories))
    if categories:
        items = [item for item in items if str(item.get("category") or "") in categories]
    if subcategories:
        items = [item for item in items if str(item.get("subcategory") or "") in subcategories]
    pair_manifest = set(_load_pair_manifest(args.pair_manifest, args.pair_split))
    if pair_manifest:
        items = [item for item in items if str(item.get("pair_id") or "") in pair_manifest]
    if args.limit_per_subcategory and args.limit_per_subcategory > 0:
        per_subcategory_counts: Dict[str, int] = {}
        limited_items: List[Dict[str, Any]] = []
        for item in items:
            sub = str(item.get("subcategory") or "")
            current = per_subcategory_counts.get(sub, 0)
            if current >= args.limit_per_subcategory:
                continue
            per_subcategory_counts[sub] = current + 1
            limited_items.append(item)
        items = limited_items
    if args.limit and args.limit > 0:
        items = items[: args.limit]
    models_arg = str(args.models or "").strip()
    if not models_arg:
        base_url = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
        temperature = float(os.environ.get("CDH_EVAL_TEMPERATURE", "0.0"))
        max_tokens = int(os.environ.get("CDH_EVAL_MAX_TOKENS", "4096"))
        specs = [
            ModelSpec(
                name="Qwen3-VL-2B-Instruct", backend="transformers", model="Qwen3-VL-2B-Instruct",
                base_url=base_url, temperature=temperature, max_tokens=max_tokens, models_root=str(BASE_DIR / "models")
            )
        ]
    else:
        specs = _load_model_specs(models_arg)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_lock = threading.Lock()
    progress_lock = threading.Lock()
    official_revis_baseline_cache = _load_official_revis_baseline_cache(
        args.revis_official_baseline_results
    )

    # 1. Collect all evaluation tasks from all models
    global_eval_tasks = []
    for spec in specs:
        config_hash = _hash_dict({
            "model": spec.model,
            "backend": spec.backend,
            "base_url": spec.base_url,
            "temp": spec.temperature,
            "max_tokens": spec.max_tokens,
            "quantization": spec.quantization,
            "dtype": spec.dtype,
            "mitigation": args.mitigation,
            "prompt_grounding_version": "prompt_grounding_v1",
            "vcd_tasks": args.vcd_tasks,
            "vcd_lambda": args.vcd_lambda,
            "vcd_beta": args.vcd_beta,
            "vcd_degrade_mode": args.vcd_degrade_mode,
            "vcd_gate": args.vcd_gate,
            "vcd_contrast_margin": args.vcd_contrast_margin,
            "vcd_image_margin": args.vcd_image_margin,
            "pai_tasks": args.pai_tasks,
            "pai_guidance_scale": args.pai_guidance_scale,
            "pai_beta": args.pai_beta,
            "pai_contrast_margin": args.pai_contrast_margin,
            "mfcd_tasks": args.mfcd_tasks,
            "mfcd_lambda_low": args.mfcd_lambda_low,
            "mfcd_lambda_high": args.mfcd_lambda_high,
            "mfcd_beta": args.mfcd_beta,
            "mfcd_low_mode": args.mfcd_low_mode,
            "mfcd_high_mode": args.mfcd_high_mode,
            "mfcd_gate": args.mfcd_gate,
            "mfcd_contrast_margin": args.mfcd_contrast_margin,
            "mfcd_image_margin": args.mfcd_image_margin,
            "mfcd_high_margin": args.mfcd_high_margin,
            "revis_tasks": args.revis_tasks,
            "revis_mode": args.revis_mode,
            "revis_official_vector_file": args.revis_official_vector_file,
            "revis_official_calibration_file": args.revis_official_calibration_file,
            "revis_official_repo_root": args.revis_official_repo_root,
            "revis_official_alpha": args.revis_official_alpha,
            "revis_official_risk_gamma": args.revis_official_risk_gamma,
            "revis_official_baseline_results": args.revis_official_baseline_results,
            "revis_lambda_prior": args.revis_lambda_prior,
            "revis_lambda_hidden": args.revis_lambda_hidden,
            "revis_gate": args.revis_gate,
            "revis_prior_margin": args.revis_prior_margin,
            "revis_contrast_margin": args.revis_contrast_margin,
            "revis_visual_conflict_image_margin": args.revis_visual_conflict_image_margin,
            "revis_absorption_image_margin_max": args.revis_absorption_image_margin_max,
            "revis_absmax_min": args.revis_absmax_min,
            "revis_prior_source": args.revis_prior_source,
            "revis_prior_degrade_mode": args.revis_prior_degrade_mode,
            "revis_prior_score": args.revis_prior_score,
            "revis_prior_inertia_gate": args.revis_prior_inertia_gate,
            "revis_prior_inertia_prob_min": args.revis_prior_inertia_prob_min,
            "revis_prior_inertia_logprob_margin": args.revis_prior_inertia_logprob_margin,
            "revis_prior_subspace_alpha": args.revis_prior_subspace_alpha,
            "revis_prior_subspace_top_k": args.revis_prior_subspace_top_k,
            "revis_layer_prior_subspace_alpha": args.revis_layer_prior_subspace_alpha,
            "revis_layer_prior_subspace_top_k": args.revis_layer_prior_subspace_top_k,
            "revis_layer_prior_subspace_index": args.revis_layer_prior_subspace_index,
            "revis_layer_prior_subspace_fraction": args.revis_layer_prior_subspace_fraction,
            "revis_attention_prior_alpha": args.revis_attention_prior_alpha,
            "revis_attention_prior_layer_index": args.revis_attention_prior_layer_index,
            "revis_attention_prior_layer_fraction": args.revis_attention_prior_layer_fraction,
            "revis_attention_prior_head_top_k": args.revis_attention_prior_head_top_k,
            "revis_attention_visual_alpha": args.revis_attention_visual_alpha,
            "revis_attention_visual_layer_index": args.revis_attention_visual_layer_index,
            "revis_attention_visual_layer_fraction": args.revis_attention_visual_layer_fraction,
            "revis_attention_visual_head_top_k": args.revis_attention_visual_head_top_k,
            "revis_image_attention_alpha": args.revis_image_attention_alpha,
            "revis_image_attention_layer_index": args.revis_image_attention_layer_index,
            "revis_image_attention_layer_fraction": args.revis_image_attention_layer_fraction,
            "revis_image_attention_head_top_k": args.revis_image_attention_head_top_k,
            "revis_image_attention_head_select": args.revis_image_attention_head_select,
            "revis_image_attention_text_alpha": args.revis_image_attention_text_alpha,
            "revis_image_attention_text_top_k": args.revis_image_attention_text_top_k,
            "revis_jspace_alpha": args.revis_jspace_alpha,
            "revis_jspace_gamma": args.revis_jspace_gamma,
            "revis_jspace_top_k": args.revis_jspace_top_k,
            "revis_jspace_layer_index": args.revis_jspace_layer_index,
            "revis_jspace_layer_fraction": args.revis_jspace_layer_fraction,
            "revis_jspace_probe": args.revis_jspace_probe,
            "revis_jspace_lens": args.revis_jspace_lens,
            "revis_jspace_lens_path": args.revis_jspace_lens_path,
            "revis_jspace_swap_alpha": args.revis_jspace_swap_alpha,
            "revis_latent_gamma": args.revis_latent_gamma,
            "revis_layer_gamma": args.revis_layer_gamma,
            "revis_layer_index": args.revis_layer_index,
            "revis_layer_fraction": args.revis_layer_fraction,
            "revis_attention_probe": args.revis_attention_probe,
            "revis_text_inertia_mode": args.revis_text_inertia_mode,
            "revis_text_inertia_visual_attention_max": args.revis_text_inertia_visual_attention_max,
            "revis_text_inertia_logprob_margin": args.revis_text_inertia_logprob_margin,
            "revis_text_inertia_prior_logp_min": args.revis_text_inertia_prior_logp_min,
            "revis_text_inertia_penalty": args.revis_text_inertia_penalty,
            "revis_text_inertia_scope": args.revis_text_inertia_scope,
            "revis_hidden_margin": args.revis_hidden_margin,
            "revis_token_top_k": args.revis_token_top_k,
            "cp_vbc_lambda": args.cp_vbc_lambda,
            "cp_vbc_lambda_policy": args.cp_vbc_lambda_policy,
            "cp_vbc_path_lambda_low": args.cp_vbc_path_lambda_low,
            "cp_vbc_path_lambda_high": args.cp_vbc_path_lambda_high,
            "cp_vbc_path_lambda_steps": args.cp_vbc_path_lambda_steps,
            "cp_vbc_path_stability": args.cp_vbc_path_stability,
            "cp_vbc_path_margin": args.cp_vbc_path_margin,
            "cp_vbc_path_prior_relief": args.cp_vbc_path_prior_relief,
            "cp_vbc_tasks": args.cp_vbc_tasks,
            "cp_vbc_mode": args.cp_vbc_mode,
            "cp_vbc_risk_mode": args.cp_vbc_risk_mode,
            "cp_vbc_prior_margin": args.cp_vbc_prior_margin,
            "cp_vbc_contrast_margin": args.cp_vbc_contrast_margin,
            "cp_vbc_visual_conflict_image_margin": args.cp_vbc_visual_conflict_image_margin,
            "cp_vbc_absorption_image_margin_max": args.cp_vbc_absorption_image_margin_max,
            "cp_vbc_token_top_k": args.cp_vbc_token_top_k,
            "limit_per_subcategory": args.limit_per_subcategory,
            "pair_manifest": args.pair_manifest,
            "pair_split": args.pair_split,
            "pair_manifest_count": len(pair_manifest),
            "shard_count": args.shard_count,
            "shard_index": args.shard_index,
            "sides": sides,
            "generation_trace": args.generation_trace,
            "top_logprobs": args.top_logprobs,
            "trace_hidden_states": args.trace_hidden_states,
            "visual_probe": args.visual_probe,
        })
        model_dir = output_dir / _safe_slug(spec.name)
        model_dir.mkdir(parents=True, exist_ok=True)
        results_path = str(model_dir / "results.jsonl")
        _write_json(
            str(model_dir / "run_config.json"),
            {
                "name": spec.name,
                "backend": spec.backend,
                "model": spec.model,
                "base_url": spec.base_url,
                "temperature": spec.temperature,
                "max_tokens": spec.max_tokens,
                "models_root": spec.models_root,
                "attn_implementation": spec.attn_implementation,
                "quantization": spec.quantization,
                "dtype": spec.dtype,
                "jsonl": args.jsonl,
                "images_root": args.images_root,
                "tasks": tasks,
                "categories": sorted(categories),
                "subcategories": sorted(subcategories),
                "limit": args.limit,
                "limit_per_subcategory": args.limit_per_subcategory,
                "pair_manifest": args.pair_manifest,
                "pair_split": args.pair_split,
                "pair_manifest_count": len(pair_manifest),
                "shard_count": args.shard_count,
                "shard_index": args.shard_index,
                "sides": sides,
                "parallel": args.parallel,
                "retry": args.retry,
                "timeout_s": args.timeout_s,
                "mitigation": args.mitigation,
                "prompt_grounding_version": "prompt_grounding_v1",
                "vcd_tasks": args.vcd_tasks,
                "vcd_lambda": args.vcd_lambda,
                "vcd_beta": args.vcd_beta,
                "vcd_degrade_mode": args.vcd_degrade_mode,
                "vcd_gate": args.vcd_gate,
                "vcd_contrast_margin": args.vcd_contrast_margin,
                "vcd_image_margin": args.vcd_image_margin,
                "pai_tasks": args.pai_tasks,
                "pai_guidance_scale": args.pai_guidance_scale,
                "pai_beta": args.pai_beta,
                "pai_contrast_margin": args.pai_contrast_margin,
                "mfcd_tasks": args.mfcd_tasks,
                "mfcd_lambda_low": args.mfcd_lambda_low,
                "mfcd_lambda_high": args.mfcd_lambda_high,
                "mfcd_beta": args.mfcd_beta,
                "mfcd_low_mode": args.mfcd_low_mode,
                "mfcd_high_mode": args.mfcd_high_mode,
                "mfcd_gate": args.mfcd_gate,
                "mfcd_contrast_margin": args.mfcd_contrast_margin,
                "mfcd_image_margin": args.mfcd_image_margin,
                "mfcd_high_margin": args.mfcd_high_margin,
                "revis_tasks": args.revis_tasks,
                "revis_mode": args.revis_mode,
                "revis_official_vector_file": args.revis_official_vector_file,
                "revis_official_calibration_file": args.revis_official_calibration_file,
                "revis_official_repo_root": args.revis_official_repo_root,
                "revis_official_alpha": args.revis_official_alpha,
                "revis_official_risk_gamma": args.revis_official_risk_gamma,
                "revis_official_baseline_results": args.revis_official_baseline_results,
                "revis_official_baseline_cache_rows": len(official_revis_baseline_cache),
                "revis_lambda_prior": args.revis_lambda_prior,
                "revis_lambda_hidden": args.revis_lambda_hidden,
                "revis_gate": args.revis_gate,
                "revis_prior_margin": args.revis_prior_margin,
                "revis_contrast_margin": args.revis_contrast_margin,
                "revis_visual_conflict_image_margin": args.revis_visual_conflict_image_margin,
                "revis_absorption_image_margin_max": args.revis_absorption_image_margin_max,
                "revis_absmax_min": args.revis_absmax_min,
                "revis_prior_source": args.revis_prior_source,
                "revis_prior_degrade_mode": args.revis_prior_degrade_mode,
                "revis_prior_score": args.revis_prior_score,
                "revis_prior_inertia_gate": args.revis_prior_inertia_gate,
                "revis_prior_inertia_prob_min": args.revis_prior_inertia_prob_min,
                "revis_prior_inertia_logprob_margin": args.revis_prior_inertia_logprob_margin,
                "revis_prior_subspace_alpha": args.revis_prior_subspace_alpha,
                "revis_prior_subspace_top_k": args.revis_prior_subspace_top_k,
                "revis_layer_prior_subspace_alpha": args.revis_layer_prior_subspace_alpha,
                "revis_layer_prior_subspace_top_k": args.revis_layer_prior_subspace_top_k,
                "revis_layer_prior_subspace_index": args.revis_layer_prior_subspace_index,
                "revis_layer_prior_subspace_fraction": args.revis_layer_prior_subspace_fraction,
                "revis_attention_prior_alpha": args.revis_attention_prior_alpha,
                "revis_attention_prior_layer_index": args.revis_attention_prior_layer_index,
                "revis_attention_prior_layer_fraction": args.revis_attention_prior_layer_fraction,
                "revis_attention_prior_head_top_k": args.revis_attention_prior_head_top_k,
                "revis_attention_visual_alpha": args.revis_attention_visual_alpha,
                "revis_attention_visual_layer_index": args.revis_attention_visual_layer_index,
                "revis_attention_visual_layer_fraction": args.revis_attention_visual_layer_fraction,
                "revis_attention_visual_head_top_k": args.revis_attention_visual_head_top_k,
                "revis_image_attention_alpha": args.revis_image_attention_alpha,
                "revis_image_attention_layer_index": args.revis_image_attention_layer_index,
                "revis_image_attention_layer_fraction": args.revis_image_attention_layer_fraction,
                "revis_image_attention_head_top_k": args.revis_image_attention_head_top_k,
                "revis_image_attention_head_select": args.revis_image_attention_head_select,
                "revis_image_attention_text_alpha": args.revis_image_attention_text_alpha,
                "revis_image_attention_text_top_k": args.revis_image_attention_text_top_k,
                "revis_jspace_alpha": args.revis_jspace_alpha,
                "revis_jspace_gamma": args.revis_jspace_gamma,
                "revis_jspace_top_k": args.revis_jspace_top_k,
                "revis_jspace_layer_index": args.revis_jspace_layer_index,
                "revis_jspace_layer_fraction": args.revis_jspace_layer_fraction,
                "revis_jspace_probe": args.revis_jspace_probe,
                "revis_jspace_lens": args.revis_jspace_lens,
                "revis_jspace_lens_path": args.revis_jspace_lens_path,
                "revis_jspace_swap_alpha": args.revis_jspace_swap_alpha,
                "revis_latent_gamma": args.revis_latent_gamma,
                "revis_layer_gamma": args.revis_layer_gamma,
                "revis_layer_index": args.revis_layer_index,
                "revis_layer_fraction": args.revis_layer_fraction,
                "revis_attention_probe": args.revis_attention_probe,
                "revis_text_inertia_mode": args.revis_text_inertia_mode,
                "revis_text_inertia_visual_attention_max": args.revis_text_inertia_visual_attention_max,
                "revis_text_inertia_logprob_margin": args.revis_text_inertia_logprob_margin,
                "revis_text_inertia_prior_logp_min": args.revis_text_inertia_prior_logp_min,
                "revis_text_inertia_penalty": args.revis_text_inertia_penalty,
                "revis_text_inertia_scope": args.revis_text_inertia_scope,
                "revis_hidden_margin": args.revis_hidden_margin,
                "revis_token_top_k": args.revis_token_top_k,
                "cp_vbc_lambda": args.cp_vbc_lambda,
                "cp_vbc_lambda_policy": args.cp_vbc_lambda_policy,
                "cp_vbc_path_lambda_low": args.cp_vbc_path_lambda_low,
                "cp_vbc_path_lambda_high": args.cp_vbc_path_lambda_high,
                "cp_vbc_path_lambda_steps": args.cp_vbc_path_lambda_steps,
                "cp_vbc_path_stability": args.cp_vbc_path_stability,
                "cp_vbc_path_margin": args.cp_vbc_path_margin,
                "cp_vbc_path_prior_relief": args.cp_vbc_path_prior_relief,
                "cp_vbc_tasks": args.cp_vbc_tasks,
                "cp_vbc_mode": args.cp_vbc_mode,
                "cp_vbc_risk_mode": args.cp_vbc_risk_mode,
                "cp_vbc_prior_margin": args.cp_vbc_prior_margin,
                "cp_vbc_contrast_margin": args.cp_vbc_contrast_margin,
                "cp_vbc_visual_conflict_image_margin": args.cp_vbc_visual_conflict_image_margin,
                "cp_vbc_absorption_image_margin_max": args.cp_vbc_absorption_image_margin_max,
                "cp_vbc_token_top_k": args.cp_vbc_token_top_k,
                "generation_trace": args.generation_trace,
                "top_logprobs": args.top_logprobs,
                "trace_tokens_limit": args.trace_tokens_limit,
                "trace_hidden_states": args.trace_hidden_states,
                "visual_probe": args.visual_probe,
            },
        )
        
        existing_keys = _collect_existing_keys(results_path)
        
        for item_index, item in enumerate(items):
            pair_id = str(item.get("pair_id") or "")
            category = str(item.get("category") or "")
            subcategory = str(item.get("subcategory") or "")
            for task_index, t_type in enumerate(tasks):
                for side in sides:
                    side_index = 0 if side == "commonsense" else 1
                    if _evaluation_task_shard(
                        item_index,
                        task_index,
                        side_index,
                        len(tasks),
                        args.shard_count,
                    ) != args.shard_index:
                        continue
                    if (pair_id, t_type, side) in existing_keys:
                        continue
                    global_eval_tasks.append({
                        "spec": spec,
                        "item": item,
                        "task": t_type,
                        "side": side,
                        "pair_id": pair_id,
                        "category": category,
                        "subcategory": subcategory,
                        "results_path": results_path,
                        "config_hash": config_hash,
                        "model_dir": model_dir
                    })

    global_eval_tasks.sort(
        key=lambda task: (
            str(task["spec"].name),
            str(task["pair_id"]),
            str(task["task"]),
            str(task["side"]),
        )
    )
    total_tasks = len(global_eval_tasks)
    completed_tasks = 0
    started_at = _utc_now_iso()
    progress_payload: Dict[str, Any] = {
        "status": "running",
        "phase": "initializing",
        "started_at": started_at,
        "completed": completed_tasks,
        "total": total_tasks,
        "percent": 100.0 if total_tasks == 0 else 0.0,
        "tasks": tasks,
        "categories": sorted(categories),
        "subcategories": sorted(subcategories),
        "models": [spec.name for spec in specs],
        "mitigation": args.mitigation,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "vcd_tasks": args.vcd_tasks,
        "vcd_lambda": args.vcd_lambda,
        "vcd_beta": args.vcd_beta,
        "vcd_degrade_mode": args.vcd_degrade_mode,
        "vcd_gate": args.vcd_gate,
        "vcd_contrast_margin": args.vcd_contrast_margin,
        "vcd_image_margin": args.vcd_image_margin,
        "pai_tasks": args.pai_tasks,
        "pai_guidance_scale": args.pai_guidance_scale,
        "pai_beta": args.pai_beta,
        "pai_contrast_margin": args.pai_contrast_margin,
        "mfcd_tasks": args.mfcd_tasks,
        "mfcd_lambda_low": args.mfcd_lambda_low,
        "mfcd_lambda_high": args.mfcd_lambda_high,
        "mfcd_beta": args.mfcd_beta,
        "mfcd_low_mode": args.mfcd_low_mode,
        "mfcd_high_mode": args.mfcd_high_mode,
        "mfcd_gate": args.mfcd_gate,
        "mfcd_contrast_margin": args.mfcd_contrast_margin,
        "mfcd_image_margin": args.mfcd_image_margin,
        "mfcd_high_margin": args.mfcd_high_margin,
        "revis_tasks": args.revis_tasks,
        "revis_mode": args.revis_mode,
        "revis_official_vector_file": args.revis_official_vector_file,
        "revis_official_calibration_file": args.revis_official_calibration_file,
        "revis_official_repo_root": args.revis_official_repo_root,
        "revis_official_alpha": args.revis_official_alpha,
        "revis_official_risk_gamma": args.revis_official_risk_gamma,
        "revis_official_baseline_results": args.revis_official_baseline_results,
        "revis_official_baseline_cache_rows": len(official_revis_baseline_cache),
        "revis_lambda_prior": args.revis_lambda_prior,
        "revis_lambda_hidden": args.revis_lambda_hidden,
        "revis_gate": args.revis_gate,
        "revis_prior_margin": args.revis_prior_margin,
        "revis_contrast_margin": args.revis_contrast_margin,
        "revis_visual_conflict_image_margin": args.revis_visual_conflict_image_margin,
        "revis_absorption_image_margin_max": args.revis_absorption_image_margin_max,
        "revis_absmax_min": args.revis_absmax_min,
        "revis_prior_source": args.revis_prior_source,
        "revis_prior_degrade_mode": args.revis_prior_degrade_mode,
        "revis_prior_score": args.revis_prior_score,
        "revis_prior_inertia_gate": args.revis_prior_inertia_gate,
        "revis_prior_inertia_prob_min": args.revis_prior_inertia_prob_min,
        "revis_prior_inertia_logprob_margin": args.revis_prior_inertia_logprob_margin,
        "revis_prior_subspace_alpha": args.revis_prior_subspace_alpha,
        "revis_prior_subspace_top_k": args.revis_prior_subspace_top_k,
        "revis_layer_prior_subspace_alpha": args.revis_layer_prior_subspace_alpha,
        "revis_layer_prior_subspace_top_k": args.revis_layer_prior_subspace_top_k,
        "revis_layer_prior_subspace_index": args.revis_layer_prior_subspace_index,
        "revis_layer_prior_subspace_fraction": args.revis_layer_prior_subspace_fraction,
        "revis_attention_prior_alpha": args.revis_attention_prior_alpha,
        "revis_attention_prior_layer_index": args.revis_attention_prior_layer_index,
        "revis_attention_prior_layer_fraction": args.revis_attention_prior_layer_fraction,
        "revis_attention_prior_head_top_k": args.revis_attention_prior_head_top_k,
        "revis_attention_visual_alpha": args.revis_attention_visual_alpha,
        "revis_attention_visual_layer_index": args.revis_attention_visual_layer_index,
        "revis_attention_visual_layer_fraction": args.revis_attention_visual_layer_fraction,
        "revis_attention_visual_head_top_k": args.revis_attention_visual_head_top_k,
        "revis_image_attention_alpha": args.revis_image_attention_alpha,
        "revis_image_attention_layer_index": args.revis_image_attention_layer_index,
        "revis_image_attention_layer_fraction": args.revis_image_attention_layer_fraction,
        "revis_image_attention_head_top_k": args.revis_image_attention_head_top_k,
        "revis_image_attention_head_select": args.revis_image_attention_head_select,
        "revis_image_attention_text_alpha": args.revis_image_attention_text_alpha,
        "revis_image_attention_text_top_k": args.revis_image_attention_text_top_k,
        "revis_jspace_alpha": args.revis_jspace_alpha,
        "revis_jspace_gamma": args.revis_jspace_gamma,
        "revis_jspace_top_k": args.revis_jspace_top_k,
        "revis_jspace_layer_index": args.revis_jspace_layer_index,
        "revis_jspace_layer_fraction": args.revis_jspace_layer_fraction,
        "revis_jspace_probe": args.revis_jspace_probe,
        "revis_jspace_lens": args.revis_jspace_lens,
        "revis_jspace_lens_path": args.revis_jspace_lens_path,
        "revis_jspace_swap_alpha": args.revis_jspace_swap_alpha,
        "revis_latent_gamma": args.revis_latent_gamma,
        "revis_layer_gamma": args.revis_layer_gamma,
        "revis_layer_index": args.revis_layer_index,
        "revis_layer_fraction": args.revis_layer_fraction,
        "revis_attention_probe": args.revis_attention_probe,
        "revis_text_inertia_mode": args.revis_text_inertia_mode,
        "revis_text_inertia_visual_attention_max": args.revis_text_inertia_visual_attention_max,
        "revis_text_inertia_logprob_margin": args.revis_text_inertia_logprob_margin,
        "revis_text_inertia_prior_logp_min": args.revis_text_inertia_prior_logp_min,
        "revis_text_inertia_penalty": args.revis_text_inertia_penalty,
        "revis_text_inertia_scope": args.revis_text_inertia_scope,
        "revis_hidden_margin": args.revis_hidden_margin,
        "revis_token_top_k": args.revis_token_top_k,
        "cp_vbc_lambda": args.cp_vbc_lambda,
        "cp_vbc_lambda_policy": args.cp_vbc_lambda_policy,
        "cp_vbc_path_lambda_low": args.cp_vbc_path_lambda_low,
        "cp_vbc_path_lambda_high": args.cp_vbc_path_lambda_high,
        "cp_vbc_path_lambda_steps": args.cp_vbc_path_lambda_steps,
        "cp_vbc_path_stability": args.cp_vbc_path_stability,
        "cp_vbc_path_margin": args.cp_vbc_path_margin,
        "cp_vbc_path_prior_relief": args.cp_vbc_path_prior_relief,
        "cp_vbc_tasks": args.cp_vbc_tasks,
        "cp_vbc_mode": args.cp_vbc_mode,
        "cp_vbc_risk_mode": args.cp_vbc_risk_mode,
        "cp_vbc_prior_margin": args.cp_vbc_prior_margin,
        "cp_vbc_contrast_margin": args.cp_vbc_contrast_margin,
        "cp_vbc_visual_conflict_image_margin": args.cp_vbc_visual_conflict_image_margin,
        "cp_vbc_absorption_image_margin_max": args.cp_vbc_absorption_image_margin_max,
        "cp_vbc_token_top_k": args.cp_vbc_token_top_k,
        "generation_trace": args.generation_trace,
        "visual_probe": args.visual_probe,
        "message": f"Prepared {total_tasks} tasks",
        "last_result": None,
    }
    _write_progress(args.progress_file, progress_payload)

    def process_task(task_ctx: Dict[str, Any]) -> Dict[str, Any]:
        spec = task_ctx["spec"]
        item = task_ctx["item"]
        task = task_ctx["task"]
        side = task_ctx["side"]
        pair_id = task_ctx["pair_id"]
        category = task_ctx["category"]
        subcategory = task_ctx["subcategory"]
        results_path = task_ctx["results_path"]
        config_hash = task_ctx["config_hash"]

        img_path = _image_path(args.images_root, subcategory, pair_id, side)
        if not os.path.exists(img_path):
            rec = {
                "ts": _utc_now_iso(), "run": config_hash, "model_name": spec.name,
                "backend": spec.backend, "model": spec.model, "pair_id": pair_id,
                "category": category, "subcategory": subcategory, "task": task,
                "side": side, "image_path": img_path, "status": "missing_image",
            }
            with file_lock:
                _append_jsonl(results_path, rec)
            return rec
        paired_side = "counterfactual" if side == "commonsense" else "commonsense"

        user_text = _build_user_text(task, item)
        gt = _get_gt(item, task, side)
        cs_gt = _get_gt(item, task, "commonsense")
        cf_gt = _get_gt(item, task, "counterfactual")

        dt_ms = 0
        evidence_text = ""
        evidence_raw: Optional[Dict[str, Any]] = None
        evidence_latency_ms: Optional[int] = None
        answer_latency_ms: Optional[int] = None
        effective_mitigation = args.mitigation
        if args.mitigation == MITIGATION_OPTION_ENTAILMENT and task != "mc":
            effective_mitigation = MITIGATION_NONE
        if args.mitigation == MITIGATION_CP_VBC and spec.backend != "transformers":
            effective_mitigation = MITIGATION_NONE
        if args.mitigation == MITIGATION_CP_VBC and task not in cp_vbc_tasks:
            effective_mitigation = MITIGATION_NONE
        if args.mitigation == MITIGATION_VCD and spec.backend != "transformers":
            effective_mitigation = MITIGATION_NONE
        if args.mitigation == MITIGATION_VCD and task not in vcd_tasks:
            effective_mitigation = MITIGATION_NONE
        if args.mitigation == MITIGATION_PAI and spec.backend != "transformers":
            effective_mitigation = MITIGATION_NONE
        if args.mitigation == MITIGATION_PAI and task not in pai_tasks:
            effective_mitigation = MITIGATION_NONE
        if args.mitigation == MITIGATION_MFCD and spec.backend != "transformers":
            effective_mitigation = MITIGATION_NONE
        if args.mitigation == MITIGATION_MFCD and task not in mfcd_tasks:
            effective_mitigation = MITIGATION_NONE
        if args.mitigation == MITIGATION_REVIS and spec.backend != "transformers":
            effective_mitigation = MITIGATION_NONE
        if args.mitigation == MITIGATION_REVIS and task not in revis_tasks:
            effective_mitigation = MITIGATION_NONE

        if effective_mitigation == MITIGATION_PROMPT_GROUNDING:
            user_text = _build_prompt_grounding_text(task, item)

        if effective_mitigation in (MITIGATION_VISUAL_EVIDENCE, MITIGATION_OPTION_ENTAILMENT):
            if effective_mitigation == MITIGATION_OPTION_ENTAILMENT:
                evidence_prompt = _build_option_entailment_text(task, item)
            else:
                evidence_prompt = _build_visual_evidence_text(task, item)
            evidence_status, evidence_text, evidence_raw_resp, evidence_dt = _call_model_with_retry(
                spec,
                user_text=evidence_prompt,
                image_path=img_path,
                timeout_s=args.timeout_s,
                retry=args.retry,
                trace_generation=args.generation_trace,
                top_logprobs=args.top_logprobs,
                trace_tokens_limit=args.trace_tokens_limit,
                trace_hidden_states=args.trace_hidden_states,
                visual_probe=args.visual_probe,
            )
            evidence_raw = evidence_raw_resp if evidence_status == "ok" else None
            evidence_latency_ms = evidence_dt
            dt_ms += evidence_dt
            if evidence_status != "ok":
                status = evidence_status
                pred_text = evidence_text
                raw_resp: Dict[str, Any] = {}
            else:
                if effective_mitigation == MITIGATION_OPTION_ENTAILMENT:
                    user_text = _build_option_entailment_answer_text(task, item, evidence_text)
                else:
                    user_text = _build_visual_evidence_answer_text(task, item, evidence_text)
                status, pred_text, answer_raw_resp, answer_dt = _call_model_with_retry(
                    spec,
                    user_text=user_text,
                    image_path=img_path,
                    timeout_s=args.timeout_s,
                    retry=args.retry,
                    trace_generation=args.generation_trace,
                    top_logprobs=args.top_logprobs,
                    trace_tokens_limit=args.trace_tokens_limit,
                    trace_hidden_states=args.trace_hidden_states,
                    visual_probe=args.visual_probe,
                )
                answer_latency_ms = answer_dt
                dt_ms += answer_dt
                raw_resp = {
                    "mitigation": args.mitigation,
                    "effective_mitigation": effective_mitigation,
                    "evidence_raw": evidence_raw_resp,
                    "answer_raw": answer_raw_resp if status == "ok" else None,
                }
        elif effective_mitigation == MITIGATION_CP_VBC:
            baseline_status, baseline_pred, baseline_raw_resp, baseline_dt = _call_model_with_retry(
                spec,
                user_text=user_text,
                image_path=img_path,
                timeout_s=args.timeout_s,
                retry=args.retry,
                trace_generation=args.generation_trace,
                top_logprobs=args.top_logprobs,
                trace_tokens_limit=args.trace_tokens_limit,
                trace_hidden_states=args.trace_hidden_states,
                visual_probe=args.visual_probe,
            )
            dt_ms += baseline_dt
            answer_latency_ms = baseline_dt
            if baseline_status != "ok":
                status = baseline_status
                pred_text = baseline_pred
                raw_resp = {}
            else:
                try:
                    if args.cp_vbc_mode == CP_VBC_MODE_TOKEN:
                        cp_pred, cp_raw_resp, cp_dt = _call_local_qwen3_vl_cp_vbc_token(
                            spec,
                            task=task,
                            item=item,
                            image_path=img_path,
                            baseline_pred=baseline_pred,
                            lambda_text=args.cp_vbc_lambda,
                            prior_margin_threshold=args.cp_vbc_prior_margin,
                            contrast_margin_threshold=args.cp_vbc_contrast_margin,
                            risk_mode=args.cp_vbc_risk_mode,
                            visual_conflict_image_margin=args.cp_vbc_visual_conflict_image_margin,
                            absorption_image_margin_max=args.cp_vbc_absorption_image_margin_max,
                            token_top_k=args.cp_vbc_token_top_k,
                            trace_tokens_limit=args.trace_tokens_limit,
                            lambda_policy=args.cp_vbc_lambda_policy,
                            path_lambda_low=args.cp_vbc_path_lambda_low,
                            path_lambda_high=args.cp_vbc_path_lambda_high,
                            path_lambda_steps=args.cp_vbc_path_lambda_steps,
                            path_stability_threshold=args.cp_vbc_path_stability,
                            path_margin_threshold=args.cp_vbc_path_margin,
                            path_prior_relief_min=args.cp_vbc_path_prior_relief,
                        )
                    else:
                        cp_pred, cp_raw_resp, cp_dt = _call_local_qwen3_vl_cp_vbc(
                            spec,
                            task=task,
                            item=item,
                            image_path=img_path,
                            baseline_pred=baseline_pred,
                            lambda_text=args.cp_vbc_lambda,
                            prior_margin_threshold=args.cp_vbc_prior_margin,
                            contrast_margin_threshold=args.cp_vbc_contrast_margin,
                            risk_mode=args.cp_vbc_risk_mode,
                            visual_conflict_image_margin=args.cp_vbc_visual_conflict_image_margin,
                            absorption_image_margin_max=args.cp_vbc_absorption_image_margin_max,
                            lambda_policy=args.cp_vbc_lambda_policy,
                            path_lambda_low=args.cp_vbc_path_lambda_low,
                            path_lambda_high=args.cp_vbc_path_lambda_high,
                            path_lambda_steps=args.cp_vbc_path_lambda_steps,
                            path_stability_threshold=args.cp_vbc_path_stability,
                            path_margin_threshold=args.cp_vbc_path_margin,
                            path_prior_relief_min=args.cp_vbc_path_prior_relief,
                        )
                    dt_ms += cp_dt
                    status = "ok"
                    pred_text = cp_pred
                    raw_resp = {
                        "mitigation": args.mitigation,
                        "effective_mitigation": effective_mitigation,
                        "baseline_raw": baseline_raw_resp,
                        "cp_vbc": cp_raw_resp,
                    }
                except Exception as e:
                    status = "error"
                    pred_text = str(e)
                    raw_resp = {}
        elif effective_mitigation == MITIGATION_VCD:
            baseline_status, baseline_pred, baseline_raw_resp, baseline_dt = _call_model_with_retry(
                spec,
                user_text=user_text,
                image_path=img_path,
                timeout_s=args.timeout_s,
                retry=args.retry,
                trace_generation=args.generation_trace,
                top_logprobs=args.top_logprobs,
                trace_tokens_limit=args.trace_tokens_limit,
                trace_hidden_states=args.trace_hidden_states,
                visual_probe=args.visual_probe,
            )
            dt_ms += baseline_dt
            answer_latency_ms = baseline_dt
            if baseline_status != "ok":
                status = baseline_status
                pred_text = baseline_pred
                raw_resp = {}
            else:
                try:
                    vcd_pred, vcd_raw_resp, vcd_dt = _call_local_qwen3_vl_vcd_candidate(
                        spec,
                        task=task,
                        item=item,
                        image_path=img_path,
                        baseline_pred=baseline_pred,
                        lambda_degrade=args.vcd_lambda,
                        plausibility_beta=args.vcd_beta,
                        gate=args.vcd_gate,
                        contrast_margin_threshold=args.vcd_contrast_margin,
                        image_margin_threshold=args.vcd_image_margin,
                        degrade_mode=args.vcd_degrade_mode,
                    )
                    dt_ms += vcd_dt
                    status = "ok"
                    pred_text = vcd_pred
                    raw_resp = {
                        "mitigation": args.mitigation,
                        "effective_mitigation": effective_mitigation,
                        "baseline_raw": baseline_raw_resp,
                        "vcd": vcd_raw_resp,
                    }
                except Exception as e:
                    status = "error"
                    pred_text = str(e)
                    raw_resp = {}
        elif effective_mitigation == MITIGATION_PAI:
            baseline_status, baseline_pred, baseline_raw_resp, baseline_dt = _call_model_with_retry(
                spec,
                user_text=user_text,
                image_path=img_path,
                timeout_s=args.timeout_s,
                retry=args.retry,
                trace_generation=args.generation_trace,
                top_logprobs=args.top_logprobs,
                trace_tokens_limit=args.trace_tokens_limit,
                trace_hidden_states=args.trace_hidden_states,
                visual_probe=args.visual_probe,
            )
            dt_ms += baseline_dt
            answer_latency_ms = baseline_dt
            if baseline_status != "ok":
                status = baseline_status
                pred_text = baseline_pred
                raw_resp = {}
            else:
                try:
                    pai_pred, pai_raw_resp, pai_dt = _call_local_qwen3_vl_pai_candidate(
                        spec,
                        task=task,
                        item=item,
                        image_path=img_path,
                        baseline_pred=baseline_pred,
                        guidance_scale=args.pai_guidance_scale,
                        plausibility_beta=args.pai_beta,
                        contrast_margin_threshold=args.pai_contrast_margin,
                    )
                    dt_ms += pai_dt
                    status = "ok"
                    pred_text = pai_pred
                    raw_resp = {
                        "mitigation": args.mitigation,
                        "effective_mitigation": effective_mitigation,
                        "baseline_raw": baseline_raw_resp,
                        "pai": pai_raw_resp,
                    }
                except Exception as e:
                    status = "error"
                    pred_text = str(e)
                    raw_resp = {}
        elif effective_mitigation == MITIGATION_MFCD:
            baseline_status, baseline_pred, baseline_raw_resp, baseline_dt = _call_model_with_retry(
                spec,
                user_text=user_text,
                image_path=img_path,
                timeout_s=args.timeout_s,
                retry=args.retry,
                trace_generation=args.generation_trace,
                top_logprobs=args.top_logprobs,
                trace_tokens_limit=args.trace_tokens_limit,
                trace_hidden_states=args.trace_hidden_states,
                visual_probe=args.visual_probe,
            )
            dt_ms += baseline_dt
            answer_latency_ms = baseline_dt
            if baseline_status != "ok":
                status = baseline_status
                pred_text = baseline_pred
                raw_resp = {}
            else:
                try:
                    mfcd_pred, mfcd_raw_resp, mfcd_dt = _call_local_qwen3_vl_mfcd_candidate(
                        spec,
                        task=task,
                        item=item,
                        image_path=img_path,
                        baseline_pred=baseline_pred,
                        lambda_low=args.mfcd_lambda_low,
                        lambda_high=args.mfcd_lambda_high,
                        plausibility_beta=args.mfcd_beta,
                        gate=args.mfcd_gate,
                        contrast_margin_threshold=args.mfcd_contrast_margin,
                        image_margin_threshold=args.mfcd_image_margin,
                        high_margin_threshold=args.mfcd_high_margin,
                        low_mode=args.mfcd_low_mode,
                        high_mode=args.mfcd_high_mode,
                    )
                    dt_ms += mfcd_dt
                    status = "ok"
                    pred_text = mfcd_pred
                    raw_resp = {
                        "mitigation": args.mitigation,
                        "effective_mitigation": effective_mitigation,
                        "baseline_raw": baseline_raw_resp,
                        "mfcd": mfcd_raw_resp,
                    }
                except Exception as e:
                    status = "error"
                    pred_text = str(e)
                    raw_resp = {}
        elif effective_mitigation == MITIGATION_REVIS:
            if args.revis_mode == "official":
                baseline_dt_start = time.time()
                cache_key = (pair_id, task, side)
                cached_baseline = official_revis_baseline_cache.get(cache_key)
                try:
                    if cached_baseline is not None:
                        baseline_pred, baseline_raw_resp = cached_baseline
                        baseline_raw_resp = {
                            **baseline_raw_resp,
                            "reused_from": args.revis_official_baseline_results,
                        }
                    else:
                        baseline_pred, baseline_raw_resp = _call_local_official_revis(
                            spec,
                            user_text=user_text,
                            image_path=img_path,
                            vector_file=args.revis_official_vector_file,
                            calibration_file=args.revis_official_calibration_file,
                            repo_root=args.revis_official_repo_root,
                            steering=False,
                            alpha_visual=args.revis_official_alpha,
                            risk_gamma=args.revis_official_risk_gamma,
                        )
                    baseline_status = "ok"
                except Exception as error:
                    baseline_status = "error"
                    baseline_pred = str(error)
                    baseline_raw_resp = {}
                baseline_dt = int((time.time() - baseline_dt_start) * 1000)
            else:
                baseline_status, baseline_pred, baseline_raw_resp, baseline_dt = _call_model_with_retry(
                    spec,
                    user_text=user_text,
                    image_path=img_path,
                    timeout_s=args.timeout_s,
                    retry=args.retry,
                    trace_generation=args.generation_trace,
                    top_logprobs=args.top_logprobs,
                    trace_tokens_limit=args.trace_tokens_limit,
                    trace_hidden_states=args.trace_hidden_states,
                    visual_probe=args.visual_probe,
                )
            dt_ms += baseline_dt
            answer_latency_ms = baseline_dt
            if baseline_status != "ok":
                status = baseline_status
                pred_text = baseline_pred
                raw_resp = {}
            else:
                try:
                    revis_mode = str(args.revis_mode or "auto").strip().lower()
                    if revis_mode == "auto":
                        revis_mode = "token" if task == "caption" else "candidate"
                    if revis_mode == "official":
                        revis_started = time.time()
                        revis_pred, revis_raw_resp = _call_local_official_revis(
                            spec,
                            user_text=user_text,
                            image_path=img_path,
                            vector_file=args.revis_official_vector_file,
                            calibration_file=args.revis_official_calibration_file,
                            repo_root=args.revis_official_repo_root,
                            steering=True,
                            alpha_visual=args.revis_official_alpha,
                            risk_gamma=args.revis_official_risk_gamma,
                        )
                        revis_dt = int((time.time() - revis_started) * 1000)
                    elif revis_mode == "candidate":
                        revis_pred, revis_raw_resp, revis_dt = _call_local_qwen3_vl_revis_candidate(
                            spec,
                            task=task,
                            item=item,
                            image_path=img_path,
                            baseline_pred=baseline_pred,
                            lambda_prior=args.revis_lambda_prior,
                            lambda_hidden=args.revis_lambda_hidden,
                            gate=args.revis_gate,
                            prior_margin_threshold=args.revis_prior_margin,
                            contrast_margin_threshold=args.revis_contrast_margin,
                            visual_conflict_image_margin=args.revis_visual_conflict_image_margin,
                            absorption_image_margin_max=args.revis_absorption_image_margin_max,
                            absmax_min=args.revis_absmax_min,
                            hidden_margin_threshold=args.revis_hidden_margin,
                        )
                    elif revis_mode == "token":
                        revis_pred, revis_raw_resp, revis_dt = _call_local_qwen3_vl_revis_token(
                            spec,
                            task=task,
                            item=item,
                            image_path=img_path,
                            baseline_pred=baseline_pred,
                            lambda_prior=args.revis_lambda_prior,
                            lambda_hidden=args.revis_lambda_hidden,
                            gate=args.revis_gate,
                            prior_margin_threshold=args.revis_prior_margin,
                            contrast_margin_threshold=args.revis_contrast_margin,
                            visual_conflict_image_margin=args.revis_visual_conflict_image_margin,
                            absorption_image_margin_max=args.revis_absorption_image_margin_max,
                            absmax_min=args.revis_absmax_min,
                            prior_source=args.revis_prior_source,
                            prior_degrade_mode=args.revis_prior_degrade_mode,
                            prior_score_form=args.revis_prior_score,
                            prior_inertia_gate=args.revis_prior_inertia_gate,
                            prior_inertia_prob_min=args.revis_prior_inertia_prob_min,
                            prior_inertia_logprob_margin=args.revis_prior_inertia_logprob_margin,
                            prior_subspace_alpha=args.revis_prior_subspace_alpha,
                            prior_subspace_top_k=args.revis_prior_subspace_top_k,
                            layer_prior_subspace_alpha=args.revis_layer_prior_subspace_alpha,
                            layer_prior_subspace_top_k=args.revis_layer_prior_subspace_top_k,
                            layer_prior_subspace_index=args.revis_layer_prior_subspace_index,
                            layer_prior_subspace_fraction=args.revis_layer_prior_subspace_fraction,
                            attention_prior_alpha=args.revis_attention_prior_alpha,
                            attention_prior_layer_index=args.revis_attention_prior_layer_index,
                            attention_prior_layer_fraction=args.revis_attention_prior_layer_fraction,
                            attention_prior_head_top_k=args.revis_attention_prior_head_top_k,
                            attention_visual_alpha=args.revis_attention_visual_alpha,
                            attention_visual_layer_index=args.revis_attention_visual_layer_index,
                            attention_visual_layer_fraction=args.revis_attention_visual_layer_fraction,
                            attention_visual_head_top_k=args.revis_attention_visual_head_top_k,
                            image_attention_alpha=args.revis_image_attention_alpha,
                            image_attention_layer_index=args.revis_image_attention_layer_index,
                            image_attention_layer_fraction=args.revis_image_attention_layer_fraction,
                            image_attention_head_top_k=args.revis_image_attention_head_top_k,
                            image_attention_head_select=args.revis_image_attention_head_select,
                            image_attention_text_alpha=args.revis_image_attention_text_alpha,
                            image_attention_text_top_k=args.revis_image_attention_text_top_k,
                            jspace_alpha=args.revis_jspace_alpha,
                            jspace_gamma=args.revis_jspace_gamma,
                            jspace_top_k=args.revis_jspace_top_k,
                            jspace_layer_index=args.revis_jspace_layer_index,
                            jspace_layer_fraction=args.revis_jspace_layer_fraction,
                            jspace_probe=args.revis_jspace_probe,
                            jspace_lens=args.revis_jspace_lens,
                            jspace_lens_path=args.revis_jspace_lens_path,
                            jspace_swap_alpha=args.revis_jspace_swap_alpha,
                            latent_gamma=args.revis_latent_gamma,
                            layer_gamma=args.revis_layer_gamma,
                            layer_index=args.revis_layer_index,
                            layer_fraction=args.revis_layer_fraction,
                            attention_probe=args.revis_attention_probe,
                            text_inertia_mode=args.revis_text_inertia_mode,
                            text_inertia_visual_attention_max=args.revis_text_inertia_visual_attention_max,
                            text_inertia_logprob_margin=args.revis_text_inertia_logprob_margin,
                            text_inertia_prior_logp_min=args.revis_text_inertia_prior_logp_min,
                            text_inertia_penalty=args.revis_text_inertia_penalty,
                            text_inertia_scope=args.revis_text_inertia_scope,
                            token_top_k=args.revis_token_top_k,
                            trace_tokens_limit=args.trace_tokens_limit,
                        )
                    else:
                        raise ValueError(f"unknown revis mode: {revis_mode}")
                    revis_raw_resp["baseline_prediction"] = baseline_pred
                    revis_raw_resp["steered_prediction"] = revis_pred
                    dt_ms += revis_dt
                    status = "ok"
                    pred_text = revis_pred
                    raw_resp = {
                        "mitigation": args.mitigation,
                        "effective_mitigation": effective_mitigation,
                        "baseline_raw": baseline_raw_resp,
                        "revis": revis_raw_resp,
                    }
                except Exception as e:
                    status = "error"
                    pred_text = str(e)
                    raw_resp = {}
        else:
            status, pred_text, raw_resp, dt_ms = _call_model_with_retry(
                spec,
                user_text=user_text,
                image_path=img_path,
                timeout_s=args.timeout_s,
                retry=args.retry,
                trace_generation=args.generation_trace,
                top_logprobs=args.top_logprobs,
                trace_tokens_limit=args.trace_tokens_limit,
                trace_hidden_states=args.trace_hidden_states,
                visual_probe=args.visual_probe,
            )
        
        correct = False
        commonsense_error = False
        caption_eval = None
        if status == "ok":
            if task == "caption":
                caption_eval = _score_caption_details(pred_text, gt, item=item, side=side)
                correct = bool(caption_eval.get("correct"))
            else:
                correct = _score(task, pred_text, gt)
            if side == "counterfactual" and (not correct):
                if task == "caption" and isinstance(caption_eval, dict):
                    commonsense_error = bool(caption_eval.get("prior_attraction"))
                else:
                    commonsense_error = _score(task, pred_text, cs_gt)

        rec = {
            "ts": _utc_now_iso(), "run": config_hash, "model_name": spec.name,
            "backend": spec.backend, "model": spec.model, "pair_id": pair_id,
            "cv_group": item.get("cv_group"),
            "category": category, "subcategory": subcategory, "task": task,
            "side": side, "image_path": img_path, "status": status,
            "paired_side": paired_side,
            "latency_ms": dt_ms, "question": _get_question(item, task),
            "prompt": item.get(f"{side}_prompt"), "gt": gt, "cf_gt": cf_gt,
            "cs_gt": cs_gt, "pred": pred_text,
            "mitigation": args.mitigation,
            "effective_mitigation": effective_mitigation,
            "visual_evidence": evidence_text if effective_mitigation in (MITIGATION_VISUAL_EVIDENCE, MITIGATION_OPTION_ENTAILMENT) else None,
            "option_entailment": evidence_text if effective_mitigation == MITIGATION_OPTION_ENTAILMENT else None,
            "prompt_grounding": user_text if effective_mitigation == MITIGATION_PROMPT_GROUNDING else None,
            "cp_vbc": (raw_resp or {}).get("cp_vbc") if effective_mitigation == MITIGATION_CP_VBC and status == "ok" else None,
            "vcd": (raw_resp or {}).get("vcd") if effective_mitigation == MITIGATION_VCD and status == "ok" else None,
            "pai": (raw_resp or {}).get("pai") if effective_mitigation == MITIGATION_PAI and status == "ok" else None,
            "mfcd": (raw_resp or {}).get("mfcd") if effective_mitigation == MITIGATION_MFCD and status == "ok" else None,
            "revis": (raw_resp or {}).get("revis") if effective_mitigation == MITIGATION_REVIS and status == "ok" else None,
            "caption_eval": caption_eval,
            "caption_claims": build_caption_claim_schema(item) if task == "caption" else None,
            "evidence_latency_ms": evidence_latency_ms,
            "answer_latency_ms": answer_latency_ms,
            "correct": bool(correct) if status == "ok" else None,
            "commonsense_error": bool(commonsense_error) if (status == "ok" and side == "counterfactual") else None,
            "raw": raw_resp if status == "ok" else None,
        }
        with file_lock:
            _append_jsonl(results_path, rec)
        return rec

    def mark_progress(rec: Optional[Dict[str, Any]], phase: str) -> None:
        nonlocal completed_tasks
        with progress_lock:
            completed_tasks += 1
            percent = 100.0 if total_tasks == 0 else round((completed_tasks / total_tasks) * 100, 2)
            progress_payload.update(
                {
                    "status": "running",
                    "phase": phase,
                    "completed": completed_tasks,
                    "total": total_tasks,
                    "percent": percent,
                    "message": f"{completed_tasks}/{total_tasks} tasks completed",
                    "last_result": {
                        "pair_id": rec.get("pair_id"),
                        "task": rec.get("task"),
                        "side": rec.get("side"),
                        "status": rec.get("status"),
                        "model_name": rec.get("model_name"),
                        "category": rec.get("category"),
                        "subcategory": rec.get("subcategory"),
                    } if rec else None,
                }
            )
            _write_progress(args.progress_file, progress_payload)

    # 2. Split tasks into API (parallel) and local (sequential)
    api_tasks = [t for t in global_eval_tasks if t["spec"].backend in ("api", "vllm")]
    local_tasks = [t for t in global_eval_tasks if t["spec"].backend == "transformers"]

    # 3. Run API tasks in parallel
    if api_tasks:
        print(f"Starting parallel API evaluation for {len(api_tasks)} tasks across {len(specs)} models...")
        progress_payload.update({"phase": "api_eval", "message": f"Running {len(api_tasks)} API tasks"})
        _write_progress(args.progress_file, progress_payload)
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = [executor.submit(process_task, t) for t in api_tasks]
            for future in tqdm(as_completed(futures), total=len(api_tasks), desc="API Eval Progress"):
                rec = future.result()
                mark_progress(rec, phase="api_eval")

    # 4. Run local tasks sequentially
    if local_tasks:
        print(f"Starting sequential local evaluation for {len(local_tasks)} tasks...")
        progress_payload.update({"phase": "local_eval", "message": f"Running {len(local_tasks)} local tasks"})
        _write_progress(args.progress_file, progress_payload)
        for t in tqdm(local_tasks, desc="Local Eval Progress"):
            rec = process_task(t)
            mark_progress(rec, phase="local_eval")

    # 5. Build summary for each model
    progress_payload.update({"phase": "summarizing", "message": "Building summaries"})
    _write_progress(args.progress_file, progress_payload)
    for spec in specs:
        model_dir = output_dir / _safe_slug(spec.name)
        results_path = model_dir / "results.jsonl"
        if results_path.exists():
            records = _read_jsonl(str(results_path))
            summary = _build_summary(records)
            (model_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    progress_payload.update(
        {
            "status": "completed",
            "phase": "completed",
            "completed": completed_tasks,
            "total": total_tasks,
            "percent": 100.0,
            "ended_at": _utc_now_iso(),
            "message": "Evaluation completed",
        }
    )
    _write_progress(args.progress_file, progress_payload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
