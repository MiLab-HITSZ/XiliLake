#!/usr/bin/env python3
# Copyright (c) 2026 MiLab. All rights reserved.
"""Build separate English and Chinese LogiQA evaluation views."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / 'downloads/datasets/github_repos/lgw863__LogiQA-dataset'
DEFAULT_ENGLISH_OUTPUT = Path(__file__).with_name('logiqa_english.jsonl')
DEFAULT_CHINESE_OUTPUT = Path(__file__).with_name('logiqa_chinese.jsonl')


def read_rows(path: Path, language: str):
    blocks = re.split(r'\n\s*\n', path.read_text(encoding='utf-8', errors='replace').strip())
    for index, block in enumerate(blocks):
        lines = block.splitlines()
        answer = str(lines[0] if lines else '').strip().upper()
        body_lines = lines[1:]
        markers: list[tuple[int, re.Match[str]]] = []
        search_from = 0
        for label in 'ABCD':
            marker = None
            for line_index in range(search_from, len(body_lines)):
                line = body_lines[line_index]
                separator = r'(?:[.．、，,]|\s+)' if language == 'Chinese' else r'[.．、，,]'
                marker = re.search(rf'^\s*{label}{separator}', line, flags=re.IGNORECASE)
                if marker is not None:
                    markers.append((line_index, marker))
                    search_from = line_index + 1
                    break
            if marker is None and language == 'English':
                for line_index in range(search_from, len(body_lines)):
                    marker = re.search(rf'\b{label}\.', body_lines[line_index])
                    if marker is not None:
                        markers.append((line_index, marker))
                        search_from = line_index + 1
                        break
            if marker is None and language == 'English':
                for line_index in range(search_from, len(body_lines)):
                    marker = re.search(rf'^\s*{label}\s+', body_lines[line_index])
                    if marker is not None:
                        markers.append((line_index, marker))
                        search_from = line_index + 1
                        break
            if marker is None:
                break
        if answer not in {'A', 'B', 'C', 'D'} or len(markers) != 4:
            continue
        question = '\n'.join(body_lines[:markers[0][0]]).strip()
        options = []
        for option_index, (line_index, marker) in enumerate(markers):
            next_line = markers[option_index + 1][0] if option_index + 1 < len(markers) else len(body_lines)
            first_line = body_lines[line_index][:marker.start()] + body_lines[line_index][marker.end():]
            option_text = '\n'.join([first_line, *body_lines[line_index + 1:next_line]]).strip()
            options.append(f'{chr(ord("A") + option_index)}. {option_text}')
        if question and all(option[3:].strip() for option in options):
            yield {
                'id': index,
                'question': question,
                'options': options,
                'answer': answer,
                'language': language,
                'source_split': path.name,
            }


def write_rows(source: Path, output: Path, language: str) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open('w', encoding='utf-8') as target:
        for row in read_rows(source, language):
            target.write(json.dumps(row, ensure_ascii=False) + '\n')
            count += 1
    if count == 0:
        output.unlink(missing_ok=True)
        raise RuntimeError(f'No complete LogiQA rows were read from {source}')
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--english-output', type=Path, default=DEFAULT_ENGLISH_OUTPUT)
    parser.add_argument('--chinese-output', type=Path, default=DEFAULT_CHINESE_OUTPUT)
    args = parser.parse_args()
    counts = {
        'English': write_rows(args.source / 'Eval.txt', args.english_output, 'English'),
        'Chinese': write_rows(args.source / 'zh_eval.txt', args.chinese_output, 'Chinese'),
    }
    print(json.dumps({
        'english_output': str(args.english_output),
        'chinese_output': str(args.chinese_output),
        'rows': counts,
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
