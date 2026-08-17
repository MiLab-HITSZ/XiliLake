#!/usr/bin/env python3
# Copyright (c) 2026 MiLab. All rights reserved.
"""Export the live evaluation catalog as a Markdown reference."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from web_backend import build_trust_catalog  # noqa: E402


OUTPUT_PATH = ROOT / 'docs' / 'current_taxonomy_full.md'


def cell(value: Any) -> str:
    return ' '.join(str(value or '').split()).replace('|', '\\|')


def source_link(benchmark: dict[str, Any]) -> str:
    url = str(benchmark.get('url') or '').strip()
    return f"[原文/仓库]({url})" if url else '本地多模态数据'


def audit_verdict(benchmark: dict[str, Any]) -> str:
    override = benchmark.get('taxonomy_override') or {}
    reason = str(override.get('reason') or '').strip() if isinstance(override, dict) else ''
    return f"已校正：{cell(reason)}" if reason else '符合'


def main() -> None:
    catalog = build_trust_catalog()
    groups_by_id = {str(group.get('id') or ''): group for group in catalog.get('groups') or []}
    lines = [
        '# 当前完整评测 Benchmark 分类方案',
        '',
        '> 本文件由当前系统 catalog 自动导出，反映网页实际使用的评测领域、大类、子类和 Benchmark 挂载关系。',
        '',
        '## 总览',
        '',
        f"- 导出时间：{datetime.now(timezone.utc).strftime('%Y-%m-%d UTC')}",
        f"- 评测领域数量：{catalog.get('total_domains', 0)}",
        f"- 大类数量：{catalog.get('total_groups', len(groups_by_id))}",
        f"- 子类数量：{catalog.get('total_dimensions', 0)}",
        f"- Benchmark 数量：{catalog.get('total_benchmarks', 0)}",
        f"- 可评测子类 / Benchmark：{catalog.get('evaluable_dimensions', 0)} / {catalog.get('evaluable_benchmarks', 0)}",
        '',
        '## 评测领域概览',
        '',
        '| 序号 | 评测领域 | 大类数 | 子类数 | Benchmark 数 | 领域介绍 |',
        '| ---: | --- | ---: | ---: | ---: | --- |',
    ]
    for domain_index, domain in enumerate(catalog.get('domains') or [], 1):
        lines.append(
            f"| {domain_index} | {cell(domain.get('label'))} | {domain.get('group_count', 0)} | "
            f"{domain.get('dimension_count', 0)} | {domain.get('benchmark_count', 0)} | "
            f"{cell(domain.get('description'))} |"
        )

    lines.extend([
        '',
        '## 大类概览',
        '',
        '| 评测领域 | 大类 | 子类数 | Benchmark 数 | 可评测 Benchmark 数 | 大类介绍 |',
        '| --- | --- | ---: | ---: | ---: | --- |',
    ])
    for domain in catalog.get('domains') or []:
        for group_id in domain.get('group_ids') or []:
            group = groups_by_id[group_id]
            dimensions = group.get('dimensions') or []
            benchmarks = [bench for dim in dimensions for bench in (dim.get('benchmarks') or [])]
            lines.append(
                f"| {cell(domain.get('label'))} | {cell(group.get('label'))} | {len(dimensions)} | "
                f"{len(benchmarks)} | {sum(bool(bench.get('implemented')) for bench in benchmarks)} | "
                f"{cell(group.get('description'))} |"
            )

    lines.extend(['', '## 分类明细', ''])
    for domain_index, domain in enumerate(catalog.get('domains') or [], 1):
        lines.extend([
            f"## {domain_index}. {domain.get('label')}",
            '',
            f"**领域介绍：** {domain.get('description') or ''}",
            '',
        ])
        for group_index, group_id in enumerate(domain.get('group_ids') or [], 1):
            group = groups_by_id[group_id]
            dimensions = group.get('dimensions') or []
            benchmark_count = sum(len(dim.get('benchmarks') or []) for dim in dimensions)
            lines.extend([
                f"### {domain_index}.{group_index} {group.get('label')}",
                '',
                f"**大类介绍：** {group.get('description') or ''}",
                '',
                f"**规模：** {len(dimensions)} 个子类，{benchmark_count} 个 Benchmark。",
                '',
            ])
            for dimension_index, dim in enumerate(dimensions, 1):
                benchmarks = dim.get('benchmarks') or []
                evaluable_count = sum(bool(bench.get('implemented')) for bench in benchmarks)
                lines.extend([
                    f"#### {domain_index}.{group_index}.{dimension_index} {dim.get('label')}",
                    '',
                    f"- 分类小标题：{dim.get('category_label') or group.get('label') or ''}",
                    f"- 子类介绍：{dim.get('intro') or ''}",
                    f"- Benchmark 数量：{len(benchmarks)}（可评测 {evaluable_count}）",
                    '',
                    '| Benchmark | 状态 | 分类核验 | 原文或官方仓库 | 评测内容简介 |',
                    '| --- | --- | --- | --- | --- |',
                ])
                for benchmark in benchmarks:
                    status = '可评测' if benchmark.get('implemented') else '资料展示'
                    lines.append(
                        f"| {cell(benchmark.get('name') or 'Benchmark')} | {status} | "
                        f"{audit_verdict(benchmark)} | {source_link(benchmark)} | "
                        f"{cell(benchmark.get('intro'))} |"
                    )
                lines.append('')

    OUTPUT_PATH.write_text('\n'.join(lines).rstrip() + '\n', encoding='utf-8')
    print(f'Exported {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
