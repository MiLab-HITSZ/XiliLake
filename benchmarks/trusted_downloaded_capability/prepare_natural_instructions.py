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
DEFAULT_GENERAL_OUTPUT = Path(__file__).with_name('natural_instructions_general.jsonl.gz')
DEFAULT_CHINESE_OUTPUT = Path(__file__).with_name('natural_instructions_chinese.jsonl.gz')


def task_languages(task: dict) -> list[str]:
    languages: list[str] = []
    for field in ('Input_language', 'Output_language', 'Instruction_language'):
        values = task.get(field) or []
        if isinstance(values, str):
            values = [values]
        languages.extend(str(value).strip() for value in values if str(value).strip())
    return list(dict.fromkeys(languages))


def includes_chinese(languages: list[str]) -> bool:
    return any(
        'chinese' in language.casefold() or language.casefold() in {'zh', 'cmn'}
        for language in languages
    )


def open_output(path: Path):
    return (
        gzip.open(path, 'wt', encoding='utf-8', compresslevel=1)
        if path.suffix == '.gz'
        else path.open('w', encoding='utf-8')
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--general-output', type=Path, default=DEFAULT_GENERAL_OUTPUT)
    parser.add_argument('--chinese-output', type=Path, default=DEFAULT_CHINESE_OUTPUT)
    args = parser.parse_args()
    source_files = sorted(args.source.glob('*.json'))
    if not source_files:
        raise SystemExit(f'Natural-Instructions task files not found: {args.source}')
    newest_input = max([Path(__file__).stat().st_mtime, *(path.stat().st_mtime for path in source_files)])
    output_paths = [args.general_output, args.chinese_output]
    if all(path.exists() and path.stat().st_size > 0 and path.stat().st_mtime >= newest_input for path in output_paths):
        print(json.dumps({'outputs': [str(path) for path in output_paths], 'status': 'reused'}, ensure_ascii=False))
        return
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    counts = {'general': 0, 'chinese': 0}
    task_counts = {'general': 0, 'chinese': 0}
    with open_output(args.general_output) as general_target, open_output(args.chinese_output) as chinese_target:
        for source_file in source_files:
            task = json.loads(source_file.read_text(encoding='utf-8'))
            languages = task_languages(task)
            split = 'chinese' if includes_chinese(languages) else 'general'
            target = chinese_target if split == 'chinese' else general_target
            task_counts[split] += 1
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
                    'source_languages': languages,
                    'language': 'Chinese' if split == 'chinese' else 'General (non-Chinese)',
                }, ensure_ascii=False) + '\n')
                counts[split] += 1
    print(json.dumps({
        'general_output': str(args.general_output),
        'chinese_output': str(args.chinese_output),
        'rows': counts,
        'tasks': task_counts,
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
