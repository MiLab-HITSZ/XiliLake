#!/usr/bin/env python3
# Copyright (c) 2026 MiLab. All rights reserved.
"""Build a runnable SafetyBench set from the official labeled dev splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / 'downloads/datasets/huggingface/thu-coai__SafetyBench'
DEFAULT_ENGLISH_OUTPUT = Path(__file__).with_name('safetybench_dev_english.jsonl')
DEFAULT_CHINESE_OUTPUT = Path(__file__).with_name('safetybench_dev_chinese.jsonl')


def prepared_rows(source_file: Path, language: str):
    payload = json.loads(source_file.read_text(encoding='utf-8'))
    for category, rows in payload.items():
        for row in rows if isinstance(rows, list) else []:
            options = row.get('options') or []
            answer = row.get('answer')
            if not row.get('question') or not options or not isinstance(answer, int):
                continue
            yield {
                'question': row['question'],
                'options': options,
                'answer': chr(ord('A') + answer),
                'category': category,
                'language': language,
            }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--english-output', type=Path, default=DEFAULT_ENGLISH_OUTPUT)
    parser.add_argument('--chinese-output', type=Path, default=DEFAULT_CHINESE_OUTPUT)
    args = parser.parse_args()
    counts = {}
    for language, source_name, output_path in [
        ('English', 'dev_en.json', args.english_output),
        ('Chinese', 'dev_zh.json', args.chinese_output),
    ]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with output_path.open('w', encoding='utf-8') as target:
            for row in prepared_rows(args.source / source_name, language):
                target.write(json.dumps(row, ensure_ascii=False) + '\n')
                count += 1
        counts[language] = count
    print(json.dumps({
        'english_output': str(args.english_output),
        'chinese_output': str(args.chinese_output),
        'rows': counts,
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
