# Copyright (c) 2026 MiLab. All rights reserved.
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import ssl
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse
import http.client


TAXONOMY_NAMES = {
    "T1": "temporal_inconsistency",
    "T2": "representational_contradiction",
    "T3": "physiological_implausibility",
    "T4": "treatment_logic_mismatch",
    "T5": "documentation_code_conflict",
}
TRACK_SOURCE_TYPES = {
    "end_to_end": "joint_anomaly_analysis_strict",
    "oracle_assisted": "oracle_anchored_reasoning",
}
TRACK_DISPLAY_NAMES = {
    "end_to_end": "end_to_end_auditing",
    "oracle_assisted": "oracle_assisted_reasoning",
}
TAXONOMY_ALIASES = {
    "temporal_inconsistency": [
        "temporal_inconsistency",
        "temporal inconsistency",
        "timeline inconsistency",
        "timestamp order",
        "time-order",
        "时序不一致",
        "时间不一致",
    ],
    "representational_contradiction": [
        "representational_contradiction",
        "representational contradiction",
        "representation contradiction",
        "conflicting representation",
        "表征矛盾",
        "表示矛盾",
    ],
    "physiological_implausibility": [
        "physiological_implausibility",
        "physiological implausibility",
        "physiologically implausible",
        "implausible vital",
        "生理不合理",
        "生理异常",
    ],
    "treatment_logic_mismatch": [
        "treatment_logic_mismatch",
        "treatment logic mismatch",
        "treatment mismatch",
        "management mismatch",
        "治疗逻辑不一致",
        "治疗不匹配",
    ],
    "documentation_code_conflict": [
        "documentation_code_conflict",
        "documentation-evidence conflict",
        "documentation evidence conflict",
        "diagnosis evidence conflict",
        "documentation conflict",
        "文档证据冲突",
        "诊断证据冲突",
    ],
}
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "but",
    "by",
    "case",
    "for",
    "from",
    "has",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "record",
    "that",
    "the",
    "this",
    "to",
    "was",
    "with",
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    display_name: str
    backend: str
    model: str
    base_url: str
    api_key: Optional[str]
    temperature: float
    max_tokens: int


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value or "").strip())
    return re.sub(r"_+", "_", slug).strip("_")[:120] or "model"


def read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_model_specs(models_arg: str) -> List[ModelSpec]:
    source = Path(models_arg)
    raw = json.loads(source.read_text(encoding="utf-8")) if source.exists() else json.loads(models_arg)
    specs: List[ModelSpec] = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("model") or "model")
        api_key_env = str(item.get("api_key_env") or "").strip()
        specs.append(
            ModelSpec(
                name=name,
                display_name=str(item.get("display_name") or item.get("selected_model_name") or name),
                backend=str(item.get("backend") or "api").lower(),
                model=str(item.get("model") or ""),
                base_url=str(item.get("base_url") or "").rstrip("/"),
                api_key=os.environ.get(api_key_env) if api_key_env else None,
                temperature=float(item.get("temperature", 0.0) or 0.0),
                max_tokens=int(item.get("max_tokens", 1024) or 1024),
            )
        )
    if not specs:
        raise ValueError("No usable model configuration was provided")
    return specs


def http_post_json(url: str, payload: Dict[str, Any], api_key: Optional[str], timeout_s: int) -> Dict[str, Any]:
    parsed = urlparse(url)
    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = json.dumps(payload).encode("utf-8")
    if parsed.scheme == "https":
        conn = http.client.HTTPSConnection(
            parsed.hostname,
            parsed.port or 443,
            timeout=timeout_s,
            context=ssl._create_unverified_context(),
        )
    else:
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=timeout_s)
    try:
        conn.request("POST", path, body=body, headers=headers)
        response = conn.getresponse()
        data = response.read().decode("utf-8", errors="replace")
        if response.status >= 400:
            raise RuntimeError(f"HTTP {response.status}: {data}")
        parsed_data = json.loads(data)
        if not isinstance(parsed_data, dict):
            raise RuntimeError("Model endpoint returned a non-object JSON response")
        return parsed_data
    finally:
        conn.close()


def call_chat(spec: ModelSpec, system_prompt: str, user_prompt: str, timeout_s: int) -> Tuple[str, Dict[str, Any]]:
    base = spec.base_url.rstrip("/")
    url = base if base.endswith("/chat/completions") else (
        base + "/chat/completions" if base.endswith("/v1") else base + "/v1/chat/completions"
    )
    raw = http_post_json(
        url,
        {
            "model": spec.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": spec.temperature,
            "max_tokens": spec.max_tokens,
            "stream": False,
        },
        spec.api_key,
        timeout_s,
    )
    try:
        answer = raw["choices"][0]["message"]["content"]
    except Exception:
        answer = json.dumps(raw, ensure_ascii=False)
    return str(answer), raw


def load_upstream_module(dataset_dir: Path):
    repo_root = dataset_dir.parent.parent
    ult_dir = repo_root / "ult"
    if not ult_dir.exists():
        raise FileNotFoundError(f"EHRPerturb evaluator code was not found at {ult_dir}")
    sys.path.insert(0, str(ult_dir))
    import eval_runner  # type: ignore

    return eval_runner


def balanced_manifest_rows(dataset_dir: Path, taxonomy_id: str, max_cases: int) -> List[Dict[str, Any]]:
    manifest_path = dataset_dir / "_manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"EHRPerturb manifest was not found: {manifest_path}")
    rows = [row for row in read_jsonl(manifest_path) if str(row.get("taxonomy")) == taxonomy_id]
    difficulty_order = ["easy", "medium", "hard"]
    buckets = {
        difficulty: sorted(
            [row for row in rows if str(row.get("difficulty")) == difficulty],
            key=lambda row: (str(row.get("output_case_key")), str(row.get("slot_id"))),
        )
        for difficulty in difficulty_order
    }
    selected: List[Dict[str, Any]] = []
    index = 0
    while any(index < len(bucket) for bucket in buckets.values()):
        for difficulty in difficulty_order:
            bucket = buckets[difficulty]
            if index < len(bucket):
                selected.append(bucket[index])
                if max_cases > 0 and len(selected) >= max_cases:
                    return selected
        index += 1
    return selected


def load_tasks(
    dataset_dir: Path,
    taxonomy_id: str,
    max_cases: int,
    track: str = "end_to_end",
) -> List[Dict[str, Any]]:
    source_task_type = TRACK_SOURCE_TYPES.get(track)
    if not source_task_type:
        raise ValueError(f"Unsupported EHRPerturb track: {track}")
    upstream = load_upstream_module(dataset_dir)
    tasks: List[Dict[str, Any]] = []
    for manifest_row in balanced_manifest_rows(dataset_dir, taxonomy_id, max_cases):
        case_dir = dataset_dir / str(manifest_row.get("output_case_key")) / str(manifest_row.get("slot_id"))
        case_path = case_dir / "case_corrupted.json"
        qa_path = case_dir / "qa_tasks_v2.jsonl"
        if not case_path.exists() or not qa_path.exists():
            raise FileNotFoundError(f"Incomplete EHRPerturb case: {case_dir}")
        case_obj = read_json(case_path)
        for qa_item in read_jsonl(qa_path):
            source_type = str(qa_item.get("task_type") or "")
            if source_type != source_task_type:
                continue
            task = upstream._task_from_qa_item(qa_item, case_obj, case_path)
            if not isinstance(task, dict):
                continue
            task["_case_path"] = str(case_path)
            task["_difficulty"] = str(manifest_row.get("difficulty") or "")
            task["_slot_id"] = str(manifest_row.get("slot_id") or "")
            task["_system_prompt"] = upstream._system_prompt(str(task.get("question_type")))
            task["_user_prompt"] = upstream._user_prompt(task)
            tasks.append(task)
    if not tasks:
        raise ValueError(f"No runnable EHRPerturb tasks found for {taxonomy_id} on track {track}")
    return tasks


def flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(flatten_text(item) for item in value.values())
    if isinstance(value, list):
        return " ".join(flatten_text(item) for item in value)
    return str(value)


def text_tokens(value: Any) -> List[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9][a-z0-9._/%+-]*", flatten_text(value).lower())
        if len(token) > 1 and token not in STOP_WORDS
    ]


def token_similarity(left: Any, right: Any) -> float:
    left_tokens = text_tokens(left)
    right_tokens = text_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    left_counts: Dict[str, int] = {}
    right_counts: Dict[str, int] = {}
    for token in left_tokens:
        left_counts[token] = left_counts.get(token, 0) + 1
    for token in right_tokens:
        right_counts[token] = right_counts.get(token, 0) + 1
    overlap = sum(min(count, right_counts.get(token, 0)) for token, count in left_counts.items())
    precision = overlap / len(left_tokens)
    recall = overlap / len(right_tokens)
    return 0.4 * precision + 0.6 * recall


def score_from_similarity(similarity: float) -> int:
    if similarity >= 0.42:
        return 5
    if similarity >= 0.30:
        return 4
    if similarity >= 0.20:
        return 3
    if similarity >= 0.12:
        return 2
    if similarity >= 0.06:
        return 1
    return 0


def classification_label(expected: Dict[str, Any]) -> str:
    classification = expected.get("classification")
    if isinstance(classification, dict) and classification.get("label"):
        return str(classification.get("label"))
    return str(expected.get("taxonomy") or expected.get("gold_taxonomy") or "")


def predicted_taxonomy(answer: str) -> Optional[str]:
    lowered = answer.lower()
    matches: List[Tuple[int, str]] = []
    for taxonomy, aliases in TAXONOMY_ALIASES.items():
        positions = [lowered.find(alias.lower()) for alias in aliases if alias.lower() in lowered]
        if positions:
            matches.append((min(positions), taxonomy))
    return min(matches)[1] if matches else None


def answer_detects_anomaly(answer: str) -> bool:
    lowered = answer.lower()
    negative_phrases = [
        "no clinically meaningful inconsistency",
        "no meaningful inconsistency",
        "no internal inconsistency",
        "no inconsistency is present",
        "record is internally consistent",
        "没有临床意义的不一致",
        "未发现内部不一致",
        "记录内部一致",
    ]
    if any(phrase in lowered for phrase in negative_phrases):
        return False
    positive_phrases = [
        "inconsisten",
        "contradict",
        "implausib",
        "mismatch",
        "conflict",
        "anomal",
        "不一致",
        "矛盾",
        "冲突",
        "异常",
        "不合理",
    ]
    return any(phrase in lowered for phrase in positive_phrases)


def evidence_anchors(expected: Dict[str, Any], oracle: Dict[str, Any]) -> List[Dict[str, Any]]:
    for candidate in [
        oracle.get("gold_evidence_anchors"),
        expected.get("gold_evidence_anchors"),
        expected.get("evidence_anchors"),
    ]:
        if isinstance(candidate, list):
            return [item for item in candidate if isinstance(item, dict)]
    localization = expected.get("localization") if isinstance(expected.get("localization"), dict) else {}
    anchors: List[Dict[str, Any]] = []
    primary = localization.get("primary_location")
    if isinstance(primary, dict):
        anchors.append(primary)
    for item in localization.get("related_locations") or []:
        if isinstance(item, dict):
            anchors.append(item)
    return anchors


def anchor_strength(answer: str, anchor: Dict[str, Any]) -> int:
    lowered = answer.lower()
    strength = 0
    section = str(anchor.get("section_title") or "").strip().lower()
    timestamp = str(anchor.get("timestamp") or "").strip().lower()
    field = str(anchor.get("field_or_value") or "").strip().lower()
    text_anchor = str(anchor.get("text_anchor") or "").strip().lower()
    if section and section in lowered:
        strength += 1
    if timestamp and timestamp not in {"none", "null"} and timestamp in lowered:
        strength += 2
    if field and field not in {"timestamp", "none", "null"} and field in lowered:
        strength += 2
    if text_anchor:
        compact_anchor = re.sub(r"\s+", " ", text_anchor)
        if compact_anchor in re.sub(r"\s+", " ", lowered):
            strength += 2
        else:
            distinctive = [token for token in text_tokens(text_anchor) if len(token) >= 4]
            if distinctive and sum(token in lowered for token in distinctive) >= min(2, len(distinctive)):
                strength += 1
    return strength


def localization_score(answer: str, anchors: List[Dict[str, Any]]) -> Tuple[int, List[int]]:
    if not anchors:
        return 0, []
    strengths = [anchor_strength(answer, anchor) for anchor in anchors]
    primary = strengths[0]
    related = strengths[1:]
    if primary >= 3 and related and all(score >= 2 for score in related):
        return 5, strengths
    if primary >= 3 and any(score >= 2 for score in related):
        return 4, strengths
    if primary >= 3:
        return 3, strengths
    if primary >= 1:
        return 2, strengths
    if any(score >= 1 for score in related):
        return 1, strengths
    return 0, strengths


def explanation_gold(expected: Dict[str, Any]) -> Any:
    explanation = expected.get("explanation")
    return explanation if isinstance(explanation, dict) else {}


def repair_gold(expected: Dict[str, Any]) -> Any:
    repair = expected.get("repair")
    return repair if isinstance(repair, dict) else {}


def repair_score(answer: str, expected: Dict[str, Any]) -> Tuple[int, float]:
    similarity = token_similarity(answer, repair_gold(expected))
    score = score_from_similarity(similarity)
    action_terms = [
        "revert",
        "restore",
        "correct",
        "change",
        "update",
        "amend",
        "remove",
        "replace",
        "verify",
        "review",
        "修正",
        "恢复",
        "更改",
        "核实",
    ]
    if not any(term in answer.lower() for term in action_terms):
        score = min(score, 2)
    return score, similarity


def proxy_score(task: Dict[str, Any], answer: str) -> Dict[str, Any]:
    question_type = str(task.get("question_type") or "")
    expected = task.get("expected_output") if isinstance(task.get("expected_output"), dict) else {}
    oracle = task.get("oracle_input") if isinstance(task.get("oracle_input"), dict) else {}
    anchors = evidence_anchors(expected, oracle)
    loc_score, anchor_strengths = localization_score(answer, anchors)
    exp_similarity = token_similarity(answer, explanation_gold(expected))
    exp_score = score_from_similarity(exp_similarity)
    rep_score, rep_similarity = repair_score(answer, expected)
    taxonomy_gold = classification_label(expected)
    taxonomy_pred = predicted_taxonomy(answer)
    target_match = loc_score >= 3 or (loc_score >= 2 and exp_score >= 3)

    if question_type == "anchored_ehr_reasoning_review":
        anchor_adherence = loc_score >= 2 or exp_score >= 3
        taxonomy_used = taxonomy_pred == taxonomy_gold or any(
            alias.lower() in answer.lower() for alias in TAXONOMY_ALIASES.get(taxonomy_gold, [])
        )
        overall_success = bool(target_match and anchor_adherence and taxonomy_used and exp_score >= 4 and rep_score >= 4)
        relaxed_success = bool(target_match and anchor_adherence and taxonomy_used and exp_score >= 3 and rep_score >= 3)
        return {
            "target_match": target_match,
            "anchor_adherence": anchor_adherence,
            "taxonomy_used": taxonomy_used,
            "predicted_taxonomy": taxonomy_pred,
            "localization_score": loc_score,
            "explanation_score": exp_score,
            "repair_score": rep_score,
            "overall_success": overall_success,
            "relaxed_success": relaxed_success,
            "detection_correct": None,
            "classification_correct": None,
            "diagnostics": {
                "anchor_strengths": anchor_strengths,
                "explanation_similarity": round(exp_similarity, 6),
                "repair_similarity": round(rep_similarity, 6),
            },
        }

    detection_correct = answer_detects_anomaly(answer)
    classification_correct = bool(target_match and taxonomy_pred == taxonomy_gold)
    overall_success = bool(
        target_match
        and detection_correct
        and classification_correct
        and loc_score >= 4
        and exp_score >= 4
        and rep_score >= 4
    )
    relaxed_success = bool(
        target_match
        and detection_correct
        and classification_correct
        and loc_score >= 3
        and exp_score >= 3
        and rep_score >= 3
    )
    return {
        "target_match": target_match,
        "detection_correct": detection_correct,
        "classification_correct": classification_correct,
        "predicted_taxonomy": taxonomy_pred,
        "localization_score": loc_score,
        "explanation_score": exp_score,
        "repair_score": rep_score,
        "overall_success": overall_success,
        "relaxed_success": relaxed_success,
        "anchor_adherence": None,
        "taxonomy_used": None,
        "diagnostics": {
            "anchor_strengths": anchor_strengths,
            "explanation_similarity": round(exp_similarity, 6),
            "repair_similarity": round(rep_similarity, 6),
        },
    }


def mean_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def rate_or_none(values: Iterable[Optional[bool]]) -> Optional[float]:
    clean = [bool(value) for value in values if value is not None]
    return sum(1 for value in clean if value) / len(clean) if clean else None


def aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [record for record in records if record.get("status") == "ok"]
    scored = [record for record in ok if record.get("correct") is not None]
    latencies = [float(record.get("latency_ms") or 0) for record in ok if record.get("latency_ms") is not None]
    return {
        "n_total": len(records),
        "n_ok": len(ok),
        "n_scored": len(scored),
        "accuracy": rate_or_none(record.get("correct") for record in scored),
        "response_rate": len(ok) / len(records) if records else None,
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else None,
        "target_match_rate": rate_or_none(record.get("target_match") for record in ok),
        "detection_accuracy": rate_or_none(record.get("detection_correct") for record in ok),
        "classification_accuracy": rate_or_none(record.get("classification_correct") for record in ok),
        "anchor_adherence_rate": rate_or_none(record.get("anchor_adherence") for record in ok),
        "taxonomy_usage_rate": rate_or_none(record.get("taxonomy_used") for record in ok),
        "localization_score": mean_or_none(
            float(record.get("localization_score")) / 5.0
            if record.get("localization_score") is not None
            else None
            for record in ok
        ),
        "explanation_score": mean_or_none(
            float(record.get("explanation_score")) / 5.0
            if record.get("explanation_score") is not None
            else None
            for record in ok
        ),
        "repair_score": mean_or_none(
            float(record.get("repair_score")) / 5.0 if record.get("repair_score") is not None else None
            for record in ok
        ),
        "overall_success_rate": rate_or_none(record.get("overall_success") for record in ok),
        "relaxed_success_rate": rate_or_none(record.get("relaxed_success") for record in ok),
    }


def build_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_task: Dict[str, List[Dict[str, Any]]] = {}
    by_category: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    by_subcategory: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for row in records:
        task = str(row.get("task") or "ehr_audit")
        category = str(row.get("category") or "EHRPerturb")
        subcategory = str(row.get("subcategory") or "EHRPerturb")
        by_task.setdefault(task, []).append(row)
        by_category.setdefault(task, {}).setdefault(category, []).append(row)
        by_subcategory.setdefault(task, {}).setdefault(f"{category} / {subcategory}", []).append(row)
    return {
        "scoring_mode": "deterministic_gold_proxy",
        "overall": {task: aggregate(rows) for task, rows in by_task.items()},
        "by_category": {
            task: {key: aggregate(rows) for key, rows in sorted(bucket.items())}
            for task, bucket in by_category.items()
        },
        "by_subcategory": {
            task: {key: aggregate(rows) for key, rows in sorted(bucket.items())}
            for task, bucket in by_subcategory.items()
        },
    }


def task_display_name(question_type: str) -> str:
    return (
        "end_to_end_auditing"
        if question_type == "holistic_ehr_problem_review"
        else "oracle_assisted_reasoning"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="EHRPerturb benchmark adapter")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--tasks", default="qa")
    parser.add_argument("--categories", default="")
    parser.add_argument("--dimensions", default="")
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--retry", type=int, default=1)
    parser.add_argument("--progress-file", default="")
    parser.add_argument("--taxonomy", required=True, choices=sorted(TAXONOMY_NAMES))
    parser.add_argument("--track", default="end_to_end", choices=sorted(TRACK_SOURCE_TYPES))
    parser.add_argument("--benchmark-name", default="EHRPerturb")
    parser.add_argument("--dimension-label", default="EHR consistency auditing")
    parser.add_argument("--benchmark-url", default="https://github.com/wealone1/EHRPerturb_release")
    parser.add_argument("--max-cases", type=int, default=15)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset).resolve()
    tasks = load_tasks(dataset_dir, args.taxonomy, max(0, args.max_cases), args.track)
    specs = load_model_specs(args.models)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    total = len(tasks) * len(specs)
    completed = 0

    progress = {
        "status": "running",
        "phase": "initializing",
        "started_at": utc_now_iso(),
        "completed": 0,
        "total": total,
        "percent": 100.0 if total == 0 else 0.0,
        "tasks": [TRACK_DISPLAY_NAMES[args.track]],
        "categories": [args.dimension_label],
        "subcategories": [args.benchmark_name],
        "models": [spec.name for spec in specs],
        "message": f"Prepared {total} EHRPerturb tasks",
        "last_result": None,
    }
    if args.progress_file:
        write_json(Path(args.progress_file), progress)

    def run_one(spec: ModelSpec, task: Dict[str, Any]) -> Dict[str, Any]:
        status = "error"
        answer = ""
        raw: Any = None
        latency_ms = 0
        error = ""
        for attempt in range(args.retry + 1):
            started = time.time()
            try:
                answer, raw = call_chat(
                    spec,
                    str(task.get("_system_prompt") or ""),
                    str(task.get("_user_prompt") or ""),
                    args.timeout_s,
                )
                latency_ms = int((time.time() - started) * 1000)
                status = "ok"
                break
            except Exception as exc:
                latency_ms = int((time.time() - started) * 1000)
                error = str(exc)
                if attempt < args.retry:
                    time.sleep(2**attempt)
        scoring = proxy_score(task, answer) if status == "ok" else {}
        expected = task.get("expected_output") if isinstance(task.get("expected_output"), dict) else {}
        task_name = task_display_name(str(task.get("question_type") or ""))
        return {
            "ts": utc_now_iso(),
            "run": hashlib.sha256((spec.model + args.benchmark_name).encode()).hexdigest()[:12],
            "model_name": spec.display_name,
            "backend": spec.backend,
            "model": spec.model,
            "pair_id": str(task.get("parent_sample_id") or task.get("qa_sample_id") or ""),
            "category": args.dimension_label,
            "subcategory": args.benchmark_name,
            "task": task_name,
            "side": "sample",
            "image_path": "",
            "status": status,
            "latency_ms": latency_ms,
            "question": str(task.get("question") or ""),
            "prompt": str(task.get("_user_prompt") or ""),
            "gt": json.dumps(expected, ensure_ascii=False),
            "pred": answer if status == "ok" else error,
            "model_answer": answer,
            "correct": scoring.get("overall_success") if status == "ok" else None,
            "taxonomy_id": args.taxonomy,
            "taxonomy_name": TAXONOMY_NAMES[args.taxonomy],
            "difficulty": task.get("_difficulty"),
            "slot_id": task.get("_slot_id"),
            "source_file": task.get("_case_path"),
            "benchmark_url": args.benchmark_url,
            **scoring,
            "raw": raw,
        }

    for spec in specs:
        model_dir = output_root / safe_slug(spec.name)
        model_dir.mkdir(parents=True, exist_ok=True)
        result_path = model_dir / "results.jsonl"
        if result_path.exists():
            result_path.unlink()
        write_json(
            model_dir / "run_config.json",
            {
                "name": spec.name,
                "display_name": spec.display_name,
                "backend": spec.backend,
                "model": spec.model,
                "base_url": spec.base_url,
                "benchmark_name": args.benchmark_name,
                "benchmark_url": args.benchmark_url,
                "dimension_label": args.dimension_label,
                "dataset": str(dataset_dir),
                "taxonomy_id": args.taxonomy,
                "taxonomy_name": TAXONOMY_NAMES[args.taxonomy],
                "track": args.track,
                "tasks": [TRACK_DISPLAY_NAMES[args.track]],
                "max_cases": args.max_cases,
                "adapter": "ehrperturb_official_data",
                "scoring_mode": "deterministic_gold_proxy",
            },
        )

        max_workers = max(1, int(args.parallel or 1)) if spec.backend == "api" else 1
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_one, spec, task) for task in tasks]
            for future in as_completed(futures):
                record = future.result()
                append_jsonl(result_path, record)
                completed += 1
                progress.update(
                    {
                        "phase": "ehrperturb_eval",
                        "completed": completed,
                        "total": total,
                        "percent": round((completed / total) * 100, 2) if total else 100.0,
                        "message": f"{completed}/{total} EHRPerturb tasks completed",
                        "last_result": {
                            "pair_id": record.get("pair_id"),
                            "task": record.get("task"),
                            "status": record.get("status"),
                            "model_name": record.get("model_name"),
                            "category": record.get("category"),
                            "subcategory": record.get("subcategory"),
                        },
                    }
                )
                if args.progress_file:
                    write_json(Path(args.progress_file), progress)

        model_records = read_jsonl(result_path)
        write_json(model_dir / "summary.json", build_summary(model_records))

    progress.update(
        {
            "status": "completed",
            "phase": "completed",
            "completed": completed,
            "total": total,
            "percent": 100.0,
            "ended_at": utc_now_iso(),
            "message": "EHRPerturb evaluation completed",
        }
    )
    if args.progress_file:
        write_json(Path(args.progress_file), progress)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
