#!/usr/bin/env python3
# Copyright (c) 2026 MiLab. All rights reserved.
"""Build a runnable SAFE binary-classification set across all axes."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / 'downloads/datasets/github_repos/xiaoqiao__EvalSafetyLLM/SAFEDataset'
DEFAULT_OUTPUT = Path(__file__).with_name('safe_test.jsonl')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open('w', encoding='utf-8') as target:
        for source_file in sorted(args.source.glob('*/test.csv')):
            axis = source_file.parent.name
            with source_file.open(encoding='utf-8-sig', newline='') as source:
                for row in csv.DictReader(source):
                    text = html.unescape(str(row.get('text') or '')).strip()
                    label = str(row.get('label') or '').strip()
                    if not text or label not in {'0', '1'}:
                        continue
                    target.write(json.dumps({
                        'question': f'Classify the following text on the SAFE {axis} axis.\n\n{text}',
                        'options': ['0', '1'],
                        'answer': label,
                        'axis': axis,
                    }, ensure_ascii=False) + '\n')
                    count += 1
    print(json.dumps({'output': str(args.output), 'rows': count}, ensure_ascii=False))


if __name__ == '__main__':
    main()
