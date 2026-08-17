#!/usr/bin/env python3
# Copyright (c) 2026 MiLab. All rights reserved.
"""Flatten Natural Instructions tasks while retaining each task definition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / 'downloads/datasets/github_repos/allenai__natural-instructions/tasks'
DEFAULT_OUTPUT = Path(__file__).with_name('natural_instructions.jsonl')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open('w', encoding='utf-8') as target:
        for source_file in sorted(args.source.glob('*.json')):
            task = json.loads(source_file.read_text(encoding='utf-8'))
            definitions = task.get('Definition') or []
            definition = '\n'.join(str(item).strip() for item in definitions if str(item).strip())
            for instance in task.get('Instances') or []:
                outputs = instance.get('output') or []
                if isinstance(outputs, str):
                    outputs = [outputs]
                answer = str(outputs[0]).strip() if outputs else ''
                question = str(instance.get('input') or '').strip()
                if not question or not answer:
                    continue
                target.write(json.dumps({
                    'instruction': definition,
                    'question': question,
                    'answer': answer,
                    'acceptable_answers': outputs,
                    'task_name': source_file.stem,
                }, ensure_ascii=False) + '\n')
                count += 1
    print(json.dumps({'output': str(args.output), 'rows': count}, ensure_ascii=False))


if __name__ == '__main__':
    main()
