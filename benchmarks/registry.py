# Copyright (c) 2026 MiLab. All rights reserved.
from __future__ import annotations

import json
import copy
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return default


def merge_nested_dicts(*sources: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key, value in source.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = merge_nested_dicts(merged[key], value)
            else:
                merged[key] = copy.deepcopy(value)
    return merged


def load_benchmark_configs(base_dir: Path) -> List[Dict[str, Any]]:
    root = base_dir / 'benchmarks'
    configs: List[Dict[str, Any]] = []
    if not root.exists():
        return configs
    for path in sorted(root.glob('*/benchmark.json')):
        cfg = read_json(path, {}) or {}
        if not cfg or not cfg.get('enabled', True):
            continue
        cfg['_config_path'] = str(path)
        cfg['_benchmark_dir'] = str(path.parent)
        configs.append(cfg)
    return configs


def make_dimension_id(benchmark_id: str, category_id: str, dimension_id: str) -> str:
    if benchmark_id == 'cdh_hallucination':
        # Keep legacy ID format so existing result filters continue to work.
        return f'cdh::{category_id}::{dimension_id}'
    safe_b = re.sub(r'[:\s]+', '-', str(benchmark_id))
    safe_c = str(category_id).replace('::', '--')
    safe_d = str(dimension_id).replace('::', '--')
    return f'benchmark::{safe_b}::{safe_c}::{safe_d}'


def make_benchmark_option_id(benchmark_id: str, category_id: str, dimension_id: str, option_key: str = 'default') -> str:
    safe_b = re.sub(r'[:\s]+', '-', str(benchmark_id))
    safe_c = str(category_id).replace('::', '--')
    safe_d = str(dimension_id).replace('::', '--')
    safe_o = re.sub(r'[:\s]+', '-', str(option_key or 'default'))
    return f'benchopt::{safe_b}::{safe_c}::{safe_d}::{safe_o}'


def parse_benchmark_option_id(option_id: str) -> Optional[Tuple[str, str, str, str]]:
    raw = str(option_id or '')
    parts = raw.split('::', 4)
    if len(parts) != 5 or parts[0] != 'benchopt':
        return None
    return (
        parts[1],
        parts[2].replace('--', '::'),
        parts[3].replace('--', '::'),
        parts[4],
    )


def parse_dimension_id(dimension_id: str) -> Optional[Tuple[str, str, str]]:
    raw = str(dimension_id or '')
    parts = raw.split('::', 3)
    if len(parts) == 3 and parts[0] == 'cdh':
        return 'cdh_hallucination', parts[1], parts[2]
    if len(parts) == 4 and parts[0] == 'benchmark':
        return parts[1], parts[2].replace('--', '::'), parts[3].replace('--', '::')
    return None


def catalog_groups_from_configs(configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = []
    for cfg in configs:
        benchmark_id = str(cfg.get('id') or '').strip()
        if not benchmark_id:
            continue
        group = {
            'id': benchmark_id,
            'label': cfg.get('label') or benchmark_id,
            'source': cfg.get('source_type') or 'benchmark',
            'description': cfg.get('description') or '',
            'display': cfg.get('display') or {},
            'execution': cfg.get('execution') or {},
            'metrics': cfg.get('metrics') or [],
            'dimensions': [],
        }
        for cat in cfg.get('categories') or []:
            category_id = str(cat.get('id') or cat.get('label') or '').strip()
            if not category_id:
                continue
            category_label = cat.get('label') or category_id
            for dim in cat.get('dimensions') or []:
                dim_id = str(dim.get('id') or dim.get('label') or '').strip()
                if not dim_id:
                    continue
                dimension_uid = make_dimension_id(benchmark_id, category_id, dim_id)
                benchmark_defs = dim.get('benchmarks')
                benchmark_rows: List[Dict[str, Any]] = []
                dimension_execution = merge_nested_dicts(cfg.get('execution') or {}, dim.get('execution') or {})
                if isinstance(benchmark_defs, list) and benchmark_defs:
                    for idx, bench in enumerate(benchmark_defs):
                        if not isinstance(bench, dict):
                            continue
                        option_key = str(bench.get('option_key') or bench.get('id') or bench.get('name') or f'option_{idx + 1}').strip()
                        benchmark_execution = merge_nested_dicts(dimension_execution, bench.get('execution') or {})
                        benchmark_rows.append({
                            'id': make_benchmark_option_id(benchmark_id, category_id, dim_id, option_key),
                            'benchmark_id': benchmark_id,
                            'name': bench.get('name') or cfg.get('benchmark_name') or cfg.get('dataset_name') or group['label'],
                            'intro': bench.get('intro') or dim.get('intro') or cfg.get('description') or '',
                            'implemented': bool(benchmark_execution.get('supports_real_eval')),
                            'display': bench.get('display') or dim.get('display') or cfg.get('display') or {},
                            'execution': benchmark_execution,
                            'metrics': bench.get('metrics') or dim.get('metrics') or cfg.get('metrics') or [],
                            'paths': bench.get('paths') or dim.get('paths') or cfg.get('paths') or {},
                            'download': bench.get('download') or dim.get('download') or cfg.get('download') or {},
                            'url': bench.get('url') or '',
                            'example': bench.get('example') or dim.get('example') or cfg.get('example'),
                            'language': bench.get('language') or dim.get('language') or cfg.get('language') or '',
                            'scale': bench.get('scale') or dim.get('scale') or cfg.get('scale') or '',
                            'time': bench.get('time') or dim.get('time') or cfg.get('time') or '',
                            'source': bench.get('source') or dim.get('source') or cfg.get('source') or '',
                            'evaluation': bench.get('evaluation') or dim.get('evaluation') or cfg.get('evaluation') or '',
                        })
                if not benchmark_rows:
                    benchmark_name = cfg.get('benchmark_name') or cfg.get('dataset_name') or group['label']
                    benchmark_rows = [{
                        'id': make_benchmark_option_id(benchmark_id, category_id, dim_id),
                        'benchmark_id': benchmark_id,
                        'name': benchmark_name,
                        'intro': dim.get('intro') or cfg.get('description') or '',
                        'implemented': bool((cfg.get('execution') or {}).get('supports_real_eval')),
                        'display': dim.get('display') or cfg.get('display') or {},
                        'execution': dimension_execution,
                        'metrics': dim.get('metrics') or cfg.get('metrics') or [],
                        'paths': dim.get('paths') or cfg.get('paths') or {},
                        'download': dim.get('download') or cfg.get('download') or {},
                        'url': dim.get('url') or '',
                        'example': dim.get('example') or cfg.get('example'),
                        'language': dim.get('language') or cfg.get('language') or '',
                        'scale': dim.get('scale') or cfg.get('scale') or '',
                        'time': dim.get('time') or cfg.get('time') or '',
                        'source': dim.get('source') or cfg.get('source') or '',
                        'evaluation': dim.get('evaluation') or cfg.get('evaluation') or '',
                    }]
                group['dimensions'].append({
                    'id': dimension_uid,
                    'label': dim.get('label') or dim_id,
                    'name_en': dim.get('name_en') or dim_id,
                    'category': category_id,
                    'category_label': category_label,
                    'benchmark_id': benchmark_id,
                    'benchmark_label': group['label'],
                    'source_type': cfg.get('source_type') or benchmark_id,
                    'implemented': bool((cfg.get('execution') or {}).get('supports_real_eval')),
                    'metrics': cfg.get('metrics') or [],
                    'display': cfg.get('display') or {},
                    'execution': dimension_execution,
                    'paths': cfg.get('paths') or {},
                    'download': cfg.get('download') or {},
                    'intro': dim.get('intro') or cfg.get('description') or '',
                    'benchmarks': benchmark_rows,
                })
        groups.append(group)
    return groups


def cdh_scope_from_dimension_ids(dimension_ids: List[str]) -> Tuple[List[str], List[str]]:
    categories: set[str] = set()
    subcategories: set[str] = set()
    for dim_id in dimension_ids:
        parsed = parse_dimension_id(dim_id)
        if not parsed:
            continue
        benchmark_id, category_id, dimension_id = parsed
        if benchmark_id == 'cdh_hallucination':
            categories.add(category_id)
            subcategories.add(dimension_id)
    return sorted(categories), sorted(subcategories)
