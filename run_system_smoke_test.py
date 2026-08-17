# Copyright (c) 2026 MiLab. All rights reserved.
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from benchmarks.adapters import build_eval_command, resolve_real_benchmark_run
from web_backend import BASE_DIR, build_summary_from_records, build_trust_catalog


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_slug(value: str) -> str:
    text = ''.join(ch if ch.isalnum() or ch in '-_.' else '_' for ch in str(value or '').strip())
    return text[:160] or 'item'


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def append_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def selected_catalog_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for group in build_trust_catalog().get('groups') or []:
        for dimension in group.get('dimensions') or []:
            benchmark = next(
                (item for item in dimension.get('benchmarks') or [] if item.get('implemented')),
                None,
            )
            if not benchmark:
                continue
            execution_id = str(benchmark.get('execution_option_id') or benchmark.get('id') or '').strip()
            if not execution_id:
                continue
            rows.append({
                'group_id': group.get('id') or '',
                'group_label': group.get('label') or '',
                'dimension': dimension,
                'benchmark': benchmark,
                'execution_id': execution_id,
            })
    return rows


def result_material(record: Dict[str, Any]) -> str:
    question = str(record.get('question') or '').strip()
    prompt = str(record.get('prompt') or '').strip()
    if prompt and prompt != question:
        return prompt
    raw = record.get('case_raw')
    if not isinstance(raw, dict):
        return ''
    preferred = [
        'context', 'passage', 'article', 'document', 'story', 'scenario', 'source',
        'input', 'instruction', 'description', 'code', 'prompt', 'messages',
    ]
    blocks: List[str] = []
    for key in preferred:
        value = raw.get(key)
        if value in (None, '', [], {}):
            continue
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
        if str(text).strip() and str(text).strip() != question:
            blocks.append(f'{key}:\n{text}')
    return '\n\n'.join(blocks)


def error_record(selection: Dict[str, Any], index: int, error: str) -> Dict[str, Any]:
    dimension = selection['dimension']
    benchmark = selection['benchmark']
    dimension_id = str(dimension.get('id') or '')
    if dimension_id.startswith('cdh::'):
        parts = dimension_id.split('::', 2)
        category = parts[1] if len(parts) > 1 else str(dimension.get('label') or '')
        subcategory = parts[2] if len(parts) > 2 else str(benchmark.get('name') or '')
    else:
        category = str(dimension.get('label') or '')
        subcategory = str(benchmark.get('name') or dimension.get('label') or '')
    example = benchmark.get('example') if isinstance(benchmark.get('example'), dict) else {}
    return {
        'ts': utc_now_iso(),
        'run': 'system-smoke',
        'model_name': '',
        'backend': 'api',
        'model': '',
        'pair_id': f'smoke_{index:03d}_error',
        'category': category,
        'subcategory': subcategory,
        'task': 'qa',
        'side': 'sample',
        'image_path': '',
        'status': 'error',
        'latency_ms': 0,
        'question': str(example.get('question') or dimension.get('intro') or ''),
        'prompt': '',
        'material': '',
        'gt': str(example.get('answer') or ''),
        'pred': error,
        'model_answer': '',
        'correct': None,
        'raw': {'smoke_error': error},
    }


def annotate_records(
    rows: List[Dict[str, Any]],
    selection: Dict[str, Any],
    index: int,
    model: str,
    display_name: str,
) -> List[Dict[str, Any]]:
    dimension = selection['dimension']
    benchmark = selection['benchmark']
    annotated: List[Dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        original_pair_id = str(row.get('pair_id') or 'case_1')
        row['source_pair_id'] = original_pair_id
        row['pair_id'] = f'smoke_{index:03d}_{safe_slug(original_pair_id)}'
        row['pair_name'] = row.get('pair_name') or original_pair_id
        row['model_name'] = display_name
        row['model'] = model
        row['taxonomy_group_id'] = selection['group_id']
        row['taxonomy_group'] = selection['group_label']
        row['dimension_id'] = dimension.get('id') or ''
        row['dimension_label'] = dimension.get('label') or ''
        row['benchmark_id'] = selection['execution_id']
        row['benchmark_name'] = benchmark.get('name') or ''
        row['smoke_test'] = True
        row['material'] = row.get('material') or result_material(row)
        if not str(dimension.get('id') or '').startswith('cdh::'):
            row['category'] = dimension.get('label') or row.get('category') or ''
            row['subcategory'] = benchmark.get('name') or row.get('subcategory') or ''
        annotated.append(row)
    return annotated


def main() -> int:
    parser = argparse.ArgumentParser(description='Run one real model case for every evaluation subcategory')
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--display-name', default='Qwen3-VL-2B-Instruct')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--progress-file', default='')
    parser.add_argument('--python-bin', default=sys.executable)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--timeout-s', type=int, default=180)
    parser.add_argument('--max-tokens', type=int, default=192)
    parser.add_argument('--api-key-env', default='')
    args = parser.parse_args()

    selections = selected_catalog_rows()
    if not selections:
        raise RuntimeError('No evaluable subcategories were found')

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / 'results.jsonl'
    result_path.touch()

    work_root = BASE_DIR / 'runtime' / 'system_smoke' / hashlib.sha1(utc_now_iso().encode()).hexdigest()[:10]
    work_root.mkdir(parents=True, exist_ok=True)
    progress_path = Path(args.progress_file).resolve() if args.progress_file else None
    lock = threading.Lock()
    completed = 0
    failures: List[Dict[str, Any]] = []
    dimension_ids = [str(row['dimension'].get('id') or '') for row in selections]
    benchmark_ids = [row['execution_id'] for row in selections]
    run_config = {
        'selected_model_name': args.display_name,
        'selected_backend_mode': 'local_vllm',
        'model': args.model,
        'smoke_all': True,
        'smoke_cases_per_dimension': 1,
        'trust_dimensions': dimension_ids,
        'benchmark_ids': benchmark_ids,
        'result_selections': [
            {'dimension_id': dimension_id, 'benchmark_id': benchmark_id}
            for dimension_id, benchmark_id in zip(dimension_ids, benchmark_ids)
        ],
        'created_at': utc_now_iso(),
    }
    write_json(output_dir / 'run_config.json', run_config)

    progress: Dict[str, Any] = {
        'status': 'running',
        'phase': 'system_smoke',
        'started_at': utc_now_iso(),
        'completed': 0,
        'total': len(selections),
        'percent': 0.0,
        'tasks': ['system_smoke'],
        'categories': sorted({row['group_label'] for row in selections}),
        'subcategories': [str(row['dimension'].get('label') or '') for row in selections],
        'models': [args.display_name],
        'message': f'Prepared {len(selections)} subcategory smoke checks',
        'last_result': None,
        'failure_count': 0,
    }
    if progress_path:
        write_json(progress_path, progress)

    model_config = [{
        'name': 'current',
        'display_name': args.display_name,
        'backend': 'api',
        'model': args.model,
        'base_url': args.base_url,
        'api_key_env': args.api_key_env,
        'temperature': 0.0,
        'max_tokens': args.max_tokens,
        'models_root': str(BASE_DIR / 'models'),
    }]

    def run_selection(index: int, selection: Dict[str, Any]) -> Dict[str, Any]:
        dimension = selection['dimension']
        benchmark = selection['benchmark']
        dimension_id = str(dimension.get('id') or '')
        item_root = work_root / f'{index:03d}_{safe_slug(dimension.get("label") or dimension_id)}'
        output_root = item_root / 'output'
        child_progress = item_root / 'progress.json'
        item_root.mkdir(parents=True, exist_ok=True)
        try:
            resolved = resolve_real_benchmark_run(BASE_DIR, [dimension_id], [selection['execution_id']])
            if not resolved:
                raise RuntimeError('Benchmark execution could not be resolved')
            resolved = copy.deepcopy(resolved)
            resolved['paths'] = dict(resolved.get('paths') or {})
            resolved['paths']['results'] = str(output_root)
            execution = resolved.get('execution') or {}
            supported = [str(item) for item in execution.get('supported_tasks') or ['qa']]
            tasks = ['qa'] if 'qa' in supported else supported[:1]
            payload = {
                'tasks': tasks,
                'parallel': 1,
                'retry': 0,
                'timeout_s': args.timeout_s,
                'mitigation': 'none',
            }
            command = build_eval_command(
                BASE_DIR,
                resolved,
                args.python_bin,
                model_config,
                payload,
                str(child_progress),
                output_root / 'current',
            )
            script_name = Path(str(execution.get('script') or '')).name
            if script_name == 'evaluate_cdh_bench.py':
                command.extend(['--limit', '1'])
            else:
                command.extend(['--max-cases', '1'])
            log_path = item_root / 'runner.log'
            with log_path.open('w', encoding='utf-8') as log_handle:
                process = subprocess.run(
                    command,
                    cwd=str(BASE_DIR),
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    timeout=max(args.timeout_s + 120, 300),
                    check=False,
                )
            child_results = read_jsonl(output_root / 'current' / 'results.jsonl')
            if process.returncode != 0:
                tail = log_path.read_text(encoding='utf-8', errors='replace')[-4000:]
                raise RuntimeError(f'runner exited with {process.returncode}: {tail}')
            if not child_results:
                raise RuntimeError('runner completed without a result record')
            if not any(row.get('status') == 'ok' for row in child_results):
                errors = [str(row.get('pred') or row.get('error') or '') for row in child_results]
                raise RuntimeError('all model calls failed: ' + ' | '.join(errors[:3]))
            records = annotate_records(child_results, selection, index, args.model, args.display_name)
            return {'ok': True, 'records': records, 'selection': selection, 'index': index}
        except Exception as exc:
            record = error_record(selection, index, str(exc))
            record = annotate_records([record], selection, index, args.model, args.display_name)[0]
            return {'ok': False, 'records': [record], 'selection': selection, 'index': index, 'error': str(exc)}

    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = {
            pool.submit(run_selection, index, selection): (index, selection)
            for index, selection in enumerate(selections, start=1)
        }
        for future in as_completed(futures):
            outcome = future.result()
            selection = outcome['selection']
            with lock:
                append_jsonl(result_path, outcome['records'])
                completed += 1
                if not outcome['ok']:
                    failures.append({
                        'dimension_id': selection['dimension'].get('id') or '',
                        'dimension': selection['dimension'].get('label') or '',
                        'benchmark': selection['benchmark'].get('name') or '',
                        'error': outcome.get('error') or 'unknown error',
                    })
                progress.update({
                    'completed': completed,
                    'percent': round(completed / len(selections) * 100.0, 2),
                    'message': f'{completed}/{len(selections)} subcategories checked',
                    'failure_count': len(failures),
                    'last_result': {
                        'pair_id': outcome['records'][0].get('pair_id'),
                        'task': outcome['records'][0].get('task'),
                        'status': outcome['records'][0].get('status'),
                        'model_name': args.display_name,
                        'category': selection['group_label'],
                        'subcategory': selection['dimension'].get('label') or '',
                    },
                })
                if progress_path:
                    write_json(progress_path, progress)
                print(
                    f'[{completed:02d}/{len(selections)}] '
                    f'{selection["group_label"]} / {selection["dimension"].get("label")} '
                    f'-> {"ok" if outcome["ok"] else "failed"}',
                    flush=True,
                )

    records = read_jsonl(result_path)
    summary = build_summary_from_records(records)
    summary['smoke_test'] = {
        'total_dimensions': len(selections),
        'completed_dimensions': completed,
        'successful_dimensions': len(selections) - len(failures),
        'failed_dimensions': len(failures),
        'result_records': len(records),
        'failures': failures,
    }
    write_json(output_dir / 'summary.json', summary)
    run_config.update({
        'completed_at': utc_now_iso(),
        'smoke_successful_dimensions': len(selections) - len(failures),
        'smoke_failed_dimensions': len(failures),
    })
    write_json(output_dir / 'run_config.json', run_config)
    progress.update({
        'status': 'completed',
        'phase': 'completed',
        'ended_at': utc_now_iso(),
        'completed': len(selections),
        'percent': 100.0,
        'message': f'Smoke check completed: {len(selections) - len(failures)}/{len(selections)} successful',
        'failure_count': len(failures),
    })
    if progress_path:
        write_json(progress_path, progress)
    shutil.rmtree(work_root, ignore_errors=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
