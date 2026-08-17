# Copyright (c) 2026 MiLab. All rights reserved.
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from evaluate_generic_benchmark import build_case, load_dataset_rows, rows_quality


BASE_DIR = Path(__file__).resolve().parent
MANIFEST = BASE_DIR / 'downloads' / 'datasets' / 'download_manifest.json'
BENCHMARKS_DIR = BASE_DIR / 'benchmarks'
CATALOG_PATH = BASE_DIR / 'data' / 'trustedgpt_catalog.json'

GROUPS = {
    '真实性与幻觉控制': ('truthfulness', '基本事实准确性'),
    '推理、因果与决策可靠性': ('reasoning', '推理与决策可靠性'),
    '指令遵循与任务可靠性': ('capability', '指令遵循与任务执行可靠性'),
    '有害内容与危险能力': ('harmful_capability', '内容无害性与危险能力可控性'),
    '攻击抵御与对抗鲁棒性': ('adversarial_robustness', '攻击抵御与对抗鲁棒性'),
    '隐私、数据与系统安全': ('privacy_security', '安全策略、隐私与系统安全性'),
    '公平性、偏见与包容性': ('fairness_bias', '群体公平性与社会包容性'),
    '伦理、法律与社会合规': ('societal_compliance', '伦理、法律与社会合规性'),
    '代码安全与能力': ('code', '代码安全与能力'),
    '伦理合规': ('compliance', '伦理合规'),
    '公平与歧视': ('fairness', '公平与歧视'),
    '攻击抵御与内容安全': ('safety', '攻击抵御与内容安全'),
    '真实性': ('truthfulness', '真实性'),
    '通用能力榜单': ('general', '通用能力榜单'),
    '逻辑推理与因果': ('reasoning', '逻辑推理与因果'),
}


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return default


def safe_id(text: str) -> str:
    raw = re.sub(r'[:]+', '-', str(text or '').strip())
    raw = re.sub(r'\s+', '_', raw)
    return raw or 'dimension'


def format_meta_value(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, list):
        return '、'.join(str(x).strip() for x in value if str(x).strip())
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def is_usable(item: Dict[str, Any]) -> bool:
    status = item.get('status')
    target = Path(str(item.get('target') or ''))
    if status not in {'ok', 'skipped_existing'}:
        return False
    if item.get('mode') in {'direct_file', 'html_page'}:
        return target.is_file() and target.stat().st_size > 0
    return target.exists() and ((target / '.download_complete').exists() or (target / '.git').exists())


def rel(path: str) -> str:
    try:
        return str(Path(path).resolve().relative_to(BASE_DIR))
    except Exception:
        return str(path)


def metric_defs() -> List[Dict[str, str]]:
    return [
        {'key': 'accuracy', 'label': '准确率', 'format': 'percent'},
        {'key': 'response_rate', 'label': '响应率', 'format': 'percent'},
        {'key': 'avg_latency_ms', 'label': '平均延迟 ms', 'format': 'number'},
        {'key': 'n_total', 'label': '样本数', 'format': 'number'},
    ]


def catalog_intro_index() -> Dict[tuple[str, str], Dict[str, str]]:
    catalog = read_json(CATALOG_PATH, {}) or {}
    index: Dict[tuple[str, str], Dict[str, str]] = {}
    for dim in catalog.get('dimensions') or []:
        dim_name = str(dim.get('name_zh') or '').strip()
        dim_intro = str(dim.get('intro') or '').strip()
        for bench in dim.get('benchmarks') or []:
            if not isinstance(bench, dict):
                continue
            bench_name = str(bench.get('name') or '').strip()
            if not dim_name or not bench_name:
                continue
            index[(dim_name, bench_name)] = {
                'dimension_intro': dim_intro,
                'benchmark_intro': str(bench.get('intro') or '').strip(),
                'language': format_meta_value(bench.get('language')),
                'scale': format_meta_value(bench.get('scale')),
                'evaluation': format_meta_value(bench.get('evaluation')),
            }
    return index


INTRO_INDEX = catalog_intro_index()

DATASET_PATH_OVERRIDES = {
    'MultiTP': (
        BASE_DIR
        / 'downloads/datasets/github_repos/causalNLP__moralmachine/data/datasets/'
        / 'dataset_zh-cn+google.csv'
    ),
}


def benchmark_intro(item: Dict[str, Any], dimension: str) -> str:
    benchmark = str(item.get('benchmark') or dimension).strip()
    info = INTRO_INDEX.get((dimension, benchmark), {})
    parts: List[str] = []
    if info.get('dimension_intro'):
        parts.append(info['dimension_intro'])
    if info.get('benchmark_intro'):
        parts.append(info['benchmark_intro'])
    if not parts:
        parts.append(f'{benchmark} 用于支撑“{dimension}”子类的可信评测，重点考察模型在该维度下的问题理解、指令响应和答案生成表现。')
    extra = []
    if info.get('language'):
        extra.append(f"语言：{info['language']}")
    if info.get('scale'):
        extra.append(f"规模：{info['scale']}")
    if info.get('evaluation'):
        extra.append(f"指标：{info['evaluation']}")
    if extra:
        parts.append('；'.join(extra))
    parts.append('当前系统会从该数据集中抽取样例，按文本问答或选择题形式调用所选模型，并汇总准确率、响应率、平均延迟和样本数等指标。')
    return '\n\n'.join(part for part in parts if part)


def benchmark_meta(item: Dict[str, Any], dimension: str) -> Dict[str, str]:
    benchmark = str(item.get('benchmark') or dimension).strip()
    info = INTRO_INDEX.get((dimension, benchmark), {})
    return {
        'language': info.get('language') or '',
        'scale': info.get('scale') or '',
        'evaluation': info.get('evaluation') or '',
    }



_ROWS_CACHE: Dict[str, List[Dict[str, Any]]] = {}
_SCORABLE_CACHE: Dict[str, bool] = {}


def cached_dataset_rows(target: str, max_cases: int = 8) -> List[Dict[str, Any]]:
    raw = str(target or '').strip()
    if not raw:
        return []
    if raw not in _ROWS_CACHE:
        _ROWS_CACHE[raw] = load_dataset_rows(Path(raw), max_cases=max_cases)
    return _ROWS_CACHE[raw]


def benchmark_has_gold_answers(target: str) -> bool:
    raw = str(target or '').strip()
    if not raw:
        return False
    if raw in _SCORABLE_CACHE:
        return _SCORABLE_CACHE[raw]
    try:
        rows = cached_dataset_rows(raw, max_cases=8)
        quality = rows_quality(rows)
        ok = quality[0] > 0
    except Exception:
        ok = False
    _SCORABLE_CACHE[raw] = ok
    return ok

def benchmark_example(item: Dict[str, Any], dimension: str) -> Dict[str, Any] | None:
    benchmark = str(item.get('benchmark') or dimension).strip()
    target = Path(str(item.get('target') or ''))
    try:
        rows = cached_dataset_rows(str(target), max_cases=8)
        if not rows:
            return None
        case = build_case(rows[0], 0, benchmark, dimension)
        return {
            'benchmark': benchmark,
            'dimension': dimension,
            'task': case.get('task') or '',
            'question': case.get('question') or '',
            'answer': case.get('gt') or '',
            'options': case.get('options') or [],
            'raw': compact_example_raw(case.get('raw') or {}),
        }
    except Exception:
        return None


def compact_example_raw(raw: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for idx, (key, value) in enumerate((raw or {}).items()):
        if idx >= 8:
            break
        if isinstance(value, (dict, list)):
            text = json.dumps(value, ensure_ascii=False)
        else:
            text = str(value)
        out[str(key)] = text[:600] + ('...' if len(text) > 600 else '')
    return out


def make_benchmark(item: Dict[str, Any], dimension: str) -> Dict[str, Any]:
    benchmark = str(item.get('benchmark') or dimension).strip()
    override = DATASET_PATH_OVERRIDES.get(benchmark)
    target = str(override if override and override.exists() else item.get('target') or '').strip()
    mode = str(item.get('mode') or '')
    meta = benchmark_meta(item, dimension)
    supports_real_eval = benchmark_has_gold_answers(target)
    return {
        'option_key': safe_id(benchmark),
        'name': benchmark,
        'intro': benchmark_intro(item, dimension),
        'url': item.get('url') or '',
        'language': meta.get('language') or '',
        'scale': meta.get('scale') or '',
        'evaluation': meta.get('evaluation') or '',
        'example': benchmark_example(item, dimension),
        'metrics': metric_defs(),
        'paths': {
            'dataset': rel(target),
            'results': 'result',
        },
        'display': {
            'mode': 'generic_text_qa',
            'sections': ['selection_detail', 'progress', 'summary', 'metrics'],
            'case_view': 'text_sample',
        },
        'execution': {
            'adapter': 'python_cli',
            'script': 'evaluate_generic_benchmark.py',
            'default_tasks': ['qa'],
            'supported_tasks': ['qa'],
            'supports_real_eval': supports_real_eval,
            'args': {
                'dataset': '--dataset',
                'output_dir': '--output-dir',
                'models': '--models',
                'tasks': '--tasks',
                'parallel': '--parallel',
                'retry': '--retry',
                'timeout_s': '--timeout-s',
                'progress_file': '--progress-file',
                'categories': '--categories',
                'dimensions': '--dimensions',
            },
            'extra_args': {
                '--benchmark-name': benchmark,
                '--dimension-label': dimension,
                '--benchmark-url': item.get('url') or '',
                '--max-cases': 20,
            },
        },
    }


def build_configs() -> Dict[str, Dict[str, Any]]:
    manifest = read_json(MANIFEST, {}) or {}
    by_group_dim: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for item in manifest.get('items') or []:
        if not isinstance(item, dict) or not is_usable(item):
            continue
        group_id, _ = GROUPS.get(str(item.get('source') or ''), ('general', '通用能力榜单'))
        dim = str(item.get('dimension') or item.get('benchmark') or '').strip()
        if not dim:
            continue
        by_group_dim[group_id][dim].append(item)

    label_by_group = {gid: label for label, (gid, label) in GROUPS.items()}
    source_label_by_group = {gid: source_label for source_label, (gid, _) in GROUPS.items()}
    configs: Dict[str, Dict[str, Any]] = {}
    for group_id, dim_map in sorted(by_group_dim.items()):
        group_label = label_by_group.get(group_id, group_id)
        source_label = source_label_by_group.get(group_id, group_label)
        dimensions = []
        group_supports_real_eval = False
        for dim_label, items in sorted(dim_map.items(), key=lambda x: x[0]):
            benchmarks = [make_benchmark(item, dim_label) for item in sorted(items, key=lambda x: str(x.get('benchmark') or ''))]
            group_supports_real_eval = group_supports_real_eval or any((b.get('execution') or {}).get('supports_real_eval') for b in benchmarks)
            dimensions.append({
                'id': safe_id(dim_label),
                'label': dim_label,
                'name_en': safe_id(dim_label),
                'intro': f'{dim_label} 的本地 Benchmark 评测入口。',
                'benchmarks': benchmarks,
            })
        configs[group_id] = {
            'id': group_id,
            'label': group_label,
            'benchmark_name': '可信评测 Benchmark',
            'description': f'{group_label} 包含已接入的可信评测数据集，可通过统一评测流程运行并展示指标与样例。',
            'source_type': 'downloaded_trustedgpt',
            'enabled': True,
            'paths': {'results': 'result'},
            'execution': {
                'adapter': 'python_cli',
                'script': 'evaluate_generic_benchmark.py',
                'default_tasks': ['qa'],
                'supported_tasks': ['qa'],
                'supports_real_eval': group_supports_real_eval,
            },
            'display': {'mode': 'generic_text_qa', 'sections': ['selection_detail', 'progress', 'summary', 'metrics']},
            'metrics': metric_defs(),
            'categories': [{
                'id': 'downloaded',
                'label': source_label,
                'dimensions': dimensions,
            }],
        }
    return configs


def main() -> int:
    parser = argparse.ArgumentParser(description='Generate runnable benchmark configuration files from the download manifest')
    parser.parse_args()
    configs = build_configs()
    for child in BENCHMARKS_DIR.glob('trusted_downloaded_*'):
        if child.is_dir():
            config_path = child / 'benchmark.json'
            if config_path.exists():
                config_path.unlink()
    for group_id, cfg in configs.items():
        target_dir = BENCHMARKS_DIR / f'trusted_downloaded_{group_id}'
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / 'benchmark.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({
        'generated': len(configs),
        'dimensions': sum(len((cfg.get('categories') or [{}])[0].get('dimensions') or []) for cfg in configs.values()),
        'benchmarks': sum(len(dim.get('benchmarks') or []) for cfg in configs.values() for dim in (cfg.get('categories') or [{}])[0].get('dimensions') or []),
        'paths': [str(BENCHMARKS_DIR / f'trusted_downloaded_{gid}' / 'benchmark.json') for gid in sorted(configs)],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
