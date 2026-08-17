#!/usr/bin/env python3
# Copyright (c) 2026 MiLab. All rights reserved.
"""Build the runnable multilingual XSafety refusal profile from the local release."""

from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree


LANGUAGE_LABELS = {
    'ar': 'Arabic',
    'bn': 'Bengali',
    'de': 'German',
    'en': 'English',
    'fr': 'French',
    'hi': 'Hindi',
    'ja': 'Japanese',
    'ru': 'Russian',
    'sp': 'Spanish',
    'zh': 'Chinese',
}


def clean_prompt(value: str) -> str:
    return str(value or '').replace('\ufeff', '').strip().strip('"').strip()


def read_csv_prompts(path: Path) -> list[str]:
    prompts: list[str] = []
    with path.open('r', encoding='utf-8-sig', errors='replace', newline='') as stream:
        for row in csv.reader(stream):
            if not row:
                continue
            prompt = clean_prompt(row[0])
            if prompt:
                prompts.append(prompt)
    return prompts


def read_xlsx_prompts(path: Path) -> list[str]:
    namespace = {'x': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(path) as archive:
        shared: list[str] = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            root = ElementTree.fromstring(archive.read('xl/sharedStrings.xml'))
            for item in root.findall('x:si', namespace):
                shared.append(''.join(node.text or '' for node in item.iterfind('.//x:t', namespace)))
        sheet_name = sorted(
            name for name in archive.namelist()
            if re.fullmatch(r'xl/worksheets/sheet\d+\.xml', name)
        )[0]
        sheet = ElementTree.fromstring(archive.read(sheet_name))
        prompts: list[str] = []
        for row in sheet.findall('.//x:sheetData/x:row', namespace):
            first_cell = row.find('x:c', namespace)
            if first_cell is None:
                continue
            value = first_cell.find('x:v', namespace)
            if value is None or value.text is None:
                inline = first_cell.find('x:is/x:t', namespace)
                prompt = inline.text if inline is not None else ''
            elif first_cell.get('t') == 's':
                index = int(value.text)
                prompt = shared[index] if 0 <= index < len(shared) else ''
            else:
                prompt = value.text
            prompt = clean_prompt(prompt)
            if prompt:
                prompts.append(prompt)
        return prompts


def category_name(path: Path) -> str:
    name = re.sub(r'_n$|_en$', '', path.stem, flags=re.IGNORECASE)
    name = name.replace('_', ' ')
    if name.lower().startswith('crimes and illegal activitie'):
        return 'Crimes And Illegal Activities'
    return name


def build_rows(source_root: Path) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    seen: set[tuple[str, str, str]] = set()
    for language_code, language in LANGUAGE_LABELS.items():
        language_dir = source_root / language_code
        for path in sorted(language_dir.iterdir()):
            normalized_stem = re.sub(r'[^a-z]', '', path.stem.lower())
            if not path.is_file() or normalized_stem in {'commonsense', 'commensense'}:
                continue
            if path.suffix.lower() == '.csv':
                prompts = read_csv_prompts(path)
            elif path.suffix.lower() == '.xlsx':
                prompts = read_xlsx_prompts(path)
            else:
                continue
            category = category_name(path)
            for index, prompt in enumerate(prompts):
                key = (language_code, category, prompt)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({
                    'id': f'{language_code}-{path.stem}-{index}',
                    'language': language,
                    'language_code': language_code,
                    'category': category,
                    'question': prompt,
                    'answer': 'SAFE_REFUSAL',
                    'source': 'XSafety multilingual safety release',
                })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--source',
        default='downloads/datasets/github_repos/Jarviswang94__Multilingual_safety_benchmark',
    )
    parser.add_argument(
        '--output',
        default='benchmarks/custom_privacy/xsafety_multilingual_refusal.jsonl',
    )
    parser.add_argument(
        '--privacy-output',
        default='benchmarks/custom_privacy/xsafety_privacy_refusal.jsonl',
    )
    args = parser.parse_args()
    source_root = Path(args.source).resolve()
    output_path = Path(args.output).resolve()
    privacy_output_path = Path(args.privacy_output).resolve()
    all_rows = build_rows(source_root)
    privacy_rows = [row for row in all_rows if str(row.get('category')) == 'Privacy And Property']
    general_rows = [row for row in all_rows if str(row.get('category')) != 'Privacy And Property']
    if len(all_rows) < 1000 or len(privacy_rows) != 2000:
        raise RuntimeError(
            f'XSafety rows are unexpectedly incomplete: total={len(all_rows)}, privacy={len(privacy_rows)}'
        )

    for path, rows in [(output_path, general_rows), (privacy_output_path, privacy_rows)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(
        f'Prepared {len(general_rows)} non-privacy XSafety rows at {output_path}; '
        f'{len(privacy_rows)} privacy rows at {privacy_output_path}'
    )


if __name__ == '__main__':
    main()
