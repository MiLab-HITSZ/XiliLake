#!/usr/bin/env python3
# Copyright (c) 2026 MiLab. All rights reserved.
"""Combine every official CMMLU test subject into one runnable JSONL file."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / 'downloads/datasets/github_repos/haonan-li__CMMLU/data/test'
DEFAULT_OUTPUT = Path(__file__).with_name('cmmlu_test.jsonl')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open('w', encoding='utf-8') as target:
        for source_file in sorted(args.source.glob('*.csv')):
            with source_file.open(encoding='utf-8-sig', newline='') as source:
                for row in csv.DictReader(source):
                    options = [str(row.get(letter) or '').strip() for letter in 'ABCD']
                    record = {
                        'question': str(row.get('Question') or '').strip(),
                        'options': options,
                        'answer': str(row.get('Answer') or '').strip().upper(),
                        'subject': source_file.stem,
                    }
                    if record['question'] and all(options) and record['answer'] in 'ABCD':
                        target.write(json.dumps(record, ensure_ascii=False) + '\n')
                        count += 1
    print(json.dumps({'output': str(args.output), 'rows': count}, ensure_ascii=False))


if __name__ == '__main__':
    main()
