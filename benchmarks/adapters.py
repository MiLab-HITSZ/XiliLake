# Copyright (c) 2026 MiLab. All rights reserved.
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .registry import load_benchmark_configs, make_benchmark_option_id, merge_nested_dicts, parse_dimension_id


def benchmark_config_map(base_dir: Path) -> Dict[str, Dict[str, Any]]:
    return {str(cfg.get('id') or ''): cfg for cfg in load_benchmark_configs(base_dir) if str(cfg.get('id') or '').strip()}


def benchmark_option_index(base_dir: Path) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for cfg in load_benchmark_configs(base_dir):
        benchmark_id = str(cfg.get('id') or '').strip()
        if not benchmark_id:
            continue
        for cat in cfg.get('categories') or []:
            category_id = str(cat.get('id') or cat.get('label') or '').strip()
            if not category_id:
                continue
            for dim in cat.get('dimensions') or []:
                dimension_id = str(dim.get('id') or dim.get('label') or '').strip()
                if not dimension_id:
                    continue
                benchmark_defs = dim.get('benchmarks')
                if isinstance(benchmark_defs, list) and benchmark_defs:
                    rows = benchmark_defs
                else:
                    rows = [{}]
                for idx, bench in enumerate(rows):
                    if not isinstance(bench, dict):
                        continue
                    option_key = str(bench.get('option_key') or bench.get('id') or bench.get('name') or f'option_{idx + 1}' if benchmark_defs else 'default').strip()
                    option_id = make_benchmark_option_id(benchmark_id, category_id, dimension_id, option_key)
                    execution = merge_nested_dicts(
                        cfg.get('execution') or {},
                        dim.get('execution') or {},
                        bench.get('execution') or {},
                    )
                    index[option_id] = {
                        'id': option_id,
                        'benchmark_id': benchmark_id,
                        'category_id': category_id,
                        'dimension_id': dimension_id,
                        'config': cfg,
                        'dimension': dim,
                        'category': cat,
                        'benchmark': bench,
                        'execution': execution,
                        'display': bench.get('display') or dim.get('display') or cfg.get('display') or {},
                        'metrics': bench.get('metrics') or dim.get('metrics') or cfg.get('metrics') or [],
                        'paths': bench.get('paths') or dim.get('paths') or cfg.get('paths') or {},
                        'download': bench.get('download') or dim.get('download') or cfg.get('download') or {},
                    }
    return index


def resolve_real_benchmark_run(base_dir: Path, dimension_ids: List[str], benchmark_option_ids: List[str]) -> Optional[Dict[str, Any]]:
    option_index = benchmark_option_index(base_dir)
    selected_options = [option_index[opt_id] for opt_id in benchmark_option_ids if opt_id in option_index]
    if not selected_options:
        for dim_id in dimension_ids:
            parsed = parse_dimension_id(dim_id)
            if not parsed:
                continue
            benchmark_id, category_id, dimension_id = parsed
            default_option_id = make_benchmark_option_id(benchmark_id, category_id, dimension_id)
            if default_option_id in option_index:
                selected_options.append(option_index[default_option_id])
    selected_options = [row for row in selected_options if (row.get('execution') or {}).get('supports_real_eval')]
    if not selected_options:
        return None

    benchmark_ids = sorted({str(row.get('benchmark_id') or '') for row in selected_options if str(row.get('benchmark_id') or '').strip()})
    if len(benchmark_ids) != 1:
        raise ValueError('一次只能启动一个可评测 Benchmark。')
    benchmark_id = benchmark_ids[0]
    option = selected_options[0]

    category_ids: List[str] = []
    dimension_scope_ids: List[str] = []
    for dim_id in dimension_ids:
        parsed = parse_dimension_id(dim_id)
        if not parsed:
            continue
        parsed_benchmark_id, category_id, subdimension_id = parsed
        if parsed_benchmark_id != benchmark_id:
            continue
        if category_id not in category_ids:
            category_ids.append(category_id)
        if subdimension_id not in dimension_scope_ids:
            dimension_scope_ids.append(subdimension_id)

    if not category_ids:
        category_ids = [str(option.get('category_id') or '')]
    if not dimension_scope_ids:
        dimension_scope_ids = [str(option.get('dimension_id') or '')]

    return {
        'benchmark_id': benchmark_id,
        'benchmark_option_id': option.get('id'),
        'config': option.get('config') or {},
        'option': option,
        'execution': option.get('execution') or {},
        'display': option.get('display') or {},
        'metrics': option.get('metrics') or [],
        'paths': option.get('paths') or {},
        'category_ids': category_ids,
        'dimension_ids': dimension_scope_ids,
        'benchmark_option_ids': benchmark_option_ids,
    }


def _resolve_path(path_value: str, benchmark_dir: Path, base_dir: Path) -> str:
    raw = str(path_value or '').strip()
    if not raw:
        return ''
    path = Path(raw)
    if path.is_absolute():
        return str(path)
    local = (benchmark_dir / raw).resolve()
    if local.exists():
        return str(local)
    return str((base_dir / raw).resolve())


def _append_arg(cmd: List[str], flag: str, value: Any) -> None:
    if not flag or value is None or value == '' or value == []:
        return
    cmd.extend([flag, str(value)])


def build_eval_command(
    base_dir: Path,
    resolved_run: Dict[str, Any],
    python_bin: str,
    models_cfg: List[Dict[str, Any]],
    payload: Dict[str, Any],
    progress_file: str,
    result_dir: Path,
) -> List[str]:
    config = resolved_run.get('config') or {}
    benchmark_dir = Path(config.get('_benchmark_dir') or base_dir)
    execution = resolved_run.get('execution') or {}
    if str(execution.get('adapter') or 'python_cli') != 'python_cli':
        raise ValueError(f"暂不支持的 benchmark adapter: {execution.get('adapter')}")

    script_raw = str(execution.get('script') or '').strip()
    if not script_raw:
        raise ValueError('Benchmark 未配置执行脚本')
    script_path = _resolve_path(script_raw, benchmark_dir, base_dir)
    paths = resolved_run.get('paths') or {}
    arg_map = execution.get('args') or {}

    cmd = [python_bin, script_path]
    fixed_args = execution.get('fixed_args') or []
    if isinstance(fixed_args, list):
        cmd.extend([str(item) for item in fixed_args if str(item).strip()])

    dataset_path = _resolve_path(str(paths.get('dataset') or ''), benchmark_dir, base_dir)
    images_path = _resolve_path(str(paths.get('images') or ''), benchmark_dir, base_dir)
    output_root = _resolve_path(str(paths.get('results') or ''), benchmark_dir, base_dir) or str(result_dir.parent)

    _append_arg(cmd, str(arg_map.get('dataset') or ''), dataset_path)
    _append_arg(cmd, str(arg_map.get('images') or ''), images_path)
    _append_arg(cmd, str(arg_map.get('output_dir') or ''), output_root)
    _append_arg(cmd, str(arg_map.get('models') or ''), json.dumps(models_cfg, ensure_ascii=False))
    _append_arg(cmd, str(arg_map.get('tasks') or ''), ','.join(payload.get('tasks') or []))
    _append_arg(cmd, str(arg_map.get('parallel') or ''), payload.get('parallel'))
    _append_arg(cmd, str(arg_map.get('retry') or ''), payload.get('retry'))
    _append_arg(cmd, str(arg_map.get('timeout_s') or ''), payload.get('timeout_s'))
    _append_arg(cmd, str(arg_map.get('progress_file') or ''), progress_file)

    category_ids = resolved_run.get('category_ids') or []
    dimension_ids = resolved_run.get('dimension_ids') or []
    if category_ids:
        _append_arg(cmd, str(arg_map.get('categories') or ''), ','.join(category_ids))
    if dimension_ids:
        _append_arg(cmd, str(arg_map.get('dimensions') or ''), ','.join(dimension_ids))

    extra_args = execution.get('extra_args') or {}
    if isinstance(extra_args, dict):
        for flag, value in extra_args.items():
            _append_arg(cmd, str(flag), value)

    return cmd


def download_declared_files(base_dir: Path, benchmark_id: Optional[str] = None) -> List[Dict[str, Any]]:
    import hashlib
    import stat
    from urllib.request import urlopen

    def collect_entries() -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for cfg in load_benchmark_configs(base_dir):
            current_id = str(cfg.get('id') or '').strip()
            if benchmark_id and current_id != benchmark_id:
                continue
            benchmark_dir = Path(cfg.get('_benchmark_dir') or base_dir)
            sources = [cfg.get('download') or {}]
            for cat in cfg.get('categories') or []:
                for dim in cat.get('dimensions') or []:
                    sources.append(dim.get('download') or {})
                    for bench in dim.get('benchmarks') or []:
                        if isinstance(bench, dict):
                            sources.append(bench.get('download') or {})
            for download_cfg in sources:
                files = download_cfg.get('files') or []
                for item in files:
                    if not isinstance(item, dict):
                        continue
                    url = str(item.get('url') or '').strip()
                    rel_path = str(item.get('path') or '').strip()
                    if not url or not rel_path:
                        continue
                    key = (current_id, rel_path, url)
                    if key in seen:
                        continue
                    seen.add(key)
                    entries.append({'benchmark_id': current_id, 'benchmark_dir': benchmark_dir, **item})
        return entries

    rows: List[Dict[str, Any]] = []
    for item in collect_entries():
        current_id = str(item.get('benchmark_id') or '').strip()
        benchmark_dir = Path(item.get('benchmark_dir') or base_dir)
        url = str(item.get('url') or '').strip()
        rel_path = str(item.get('path') or '').strip()
        target = (benchmark_dir / rel_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(url, timeout=60) as resp:
            data = resp.read()
        sha256 = str(item.get('sha256') or '').strip().lower()
        if sha256:
            digest = hashlib.sha256(data).hexdigest().lower()
            if digest != sha256:
                raise ValueError(f'{current_id}: {rel_path} sha256 校验失败')
        target.write_bytes(data)
        if item.get('executable'):
            target.chmod(target.stat().st_mode | stat.S_IXUSR)
        rows.append({'benchmark_id': current_id, 'path': str(target), 'url': url, 'bytes': len(data)})
    return rows
