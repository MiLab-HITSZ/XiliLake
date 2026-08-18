#!/usr/bin/env python3
# Copyright (c) 2026 MiLab. All rights reserved.
"""Flatten Natural Instructions tasks while retaining each task definition."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / 'downloads/datasets/github_repos/allenai__natural-instructions/tasks'
DEFAULT_OUTPUT = Path(__file__).with_name('natural_instructions.jsonl.gz')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source_files = sorted(args.source.glob('*.json'))
    if not source_files:
        raise SystemExit(f'Natural-Instructions task files not found: {args.source}')
    newest_input = max([Path(__file__).stat().st_mtime, *(path.stat().st_mtime for path in source_files)])
    if args.output.exists() and args.output.stat().st_size > 0 and args.output.stat().st_mtime >= newest_input:
        print(json.dumps({'output': str(args.output), 'status': 'reused'}, ensure_ascii=False))
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    target_stream = (
        gzip.open(args.output, 'wt', encoding='utf-8', compresslevel=1)
        if args.output.suffix == '.gz'
        else args.output.open('w', encoding='utf-8')
    )
    with target_stream as target:
        for source_file in source_files:
            task = json.loads(source_file.read_text(encoding='utf-8'))
            definitions = task.get('Definition') or []
            definition = '\n'.join(str(item).strip() for item in definitions if str(item).strip())
            for instance in task.get('Instances') or []:
                outputs = instance.get('output') or []
                if isinstance(outputs, str):
                    outputs = [outputs]
                answer = str(outputs[0]).strip() if outputs else ''
                instance_input = str(instance.get('input') or '').strip()
                if not definition or not instance_input or not answer:
                    continue
                question = f"Task instruction:\n{definition}\n\nInput:\n{instance_input}"
                target.write(json.dumps({
                    'instruction': definition,
                    'input': instance_input,
                    'question': question,
                    'answer': answer,
                    'acceptable_answers': outputs,
                    'task_name': source_file.stem,
                }, ensure_ascii=False) + '\n')
                count += 1
    print(json.dumps({'output': str(args.output), 'rows': count}, ensure_ascii=False))


if __name__ == '__main__':
    main()
