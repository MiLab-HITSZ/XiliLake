#!/usr/bin/env python3
# Copyright (c) 2026 MiLab. All rights reserved.
"""Build a runnable SafetyBench set from the official labeled dev splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / 'downloads/datasets/huggingface/thu-coai__SafetyBench'
DEFAULT_OUTPUT = Path(__file__).with_name('safetybench_labeled_dev.jsonl')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open('w', encoding='utf-8') as target:
        for source_file in sorted(args.source.glob('dev_*.json')):
            payload = json.loads(source_file.read_text(encoding='utf-8'))
            for category, rows in payload.items():
                for row in rows if isinstance(rows, list) else []:
                    options = row.get('options') or []
                    answer = row.get('answer')
                    if not row.get('question') or not options or not isinstance(answer, int):
                        continue
                    target.write(json.dumps({
                        'question': row['question'],
                        'options': options,
                        'answer': chr(ord('A') + answer),
                        'category': category,
                        'language': source_file.stem.removeprefix('dev_'),
                    }, ensure_ascii=False) + '\n')
                    count += 1
    print(json.dumps({'output': str(args.output), 'rows': count}, ensure_ascii=False))


if __name__ == '__main__':
    main()
