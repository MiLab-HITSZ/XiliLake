# Copyright (c) 2026 MiLab. All rights reserved.
from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.adapters import download_declared_files
from benchmarks.registry import load_benchmark_configs


BASE_DIR = Path(__file__).resolve().parent


def list_downloads(base_dir: Path) -> list[dict]:
    rows: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for cfg in load_benchmark_configs(base_dir):
        benchmark_id = str(cfg.get('id') or '').strip()
        sources = [cfg.get('download') or {}]
        for cat in cfg.get('categories') or []:
            for dim in cat.get('dimensions') or []:
                sources.append(dim.get('download') or {})
                for bench in dim.get('benchmarks') or []:
                    if isinstance(bench, dict):
                        sources.append(bench.get('download') or {})
        for source in sources:
            for item in source.get('files') or []:
                if not isinstance(item, dict):
                    continue
                path = item.get('path') or ''
                url = item.get('url') or ''
                key = (benchmark_id, path, url)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({
                    'benchmark_id': benchmark_id,
                    'path': path,
                    'url': url,
                })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description='下载 benchmark 配置中声明的脚本或资源文件')
    parser.add_argument('--benchmark', dest='benchmark_id', default='', help='仅下载指定 benchmark id')
    parser.add_argument('--list', action='store_true', help='仅列出可下载文件，不执行下载')
    args = parser.parse_args()

    if args.list:
        print(json.dumps(list_downloads(BASE_DIR), ensure_ascii=False, indent=2))
        return 0

    rows = download_declared_files(BASE_DIR, args.benchmark_id or None)
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
