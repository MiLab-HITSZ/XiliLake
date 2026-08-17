# Copyright (c) 2026 MiLab. All rights reserved.
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse
from urllib.request import urlopen

from web_backend import build_trust_catalog

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DEST = BASE_DIR / 'downloads' / 'datasets'
DEFAULT_HF_ENDPOINT = 'https://hf-mirror.com'
DIRECT_SUFFIXES = ('.json', '.jsonl', '.csv', '.tsv', '.txt', '.zip', '.tar', '.gz', '.bz2', '.xz', '.parquet')


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return default


def safe_slug(text: str) -> str:
    raw = re.sub(r'[^0-9A-Za-z._-]+', '-', str(text or '').strip())
    raw = re.sub(r'-+', '-', raw).strip('-')
    return raw or 'item'


def canonical_source_key(url: str) -> str:
    parsed = urlparse(str(url or '').strip().lower())
    parts = [p for p in parsed.path.split('/') if p]
    if parsed.netloc == 'raw.githubusercontent.com' and len(parts) >= 2:
        return f'github:{parts[0]}/{parts[1]}'
    if parsed.netloc == 'github.com' and len(parts) >= 2:
        return f'github:{parts[0]}/{parts[1].removesuffix(".git")}'
    if parsed.netloc == 'huggingface.co' and len(parts) >= 3 and parts[0] == 'datasets':
        return f'huggingface:{parts[1]}/{parts[2]}'
    return re.sub(r'[\s\-_./:%?=&]+', '', str(url or '').strip().lower())


def collect_catalog_sources() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: set[str] = set()
    catalog = build_trust_catalog()
    for group in catalog.get('groups') or []:
        for dim in group.get('dimensions') or []:
            dim_name = str(dim.get('label') or dim.get('name_en') or '').strip()
            for bench in dim.get('benchmarks') or []:
                if not isinstance(bench, dict) or bench.get('virtual'):
                    continue
                url = str(bench.get('url') or '').strip()
                if not url:
                    continue
                plan_key = canonical_source_key(url)
                if plan_key in seen:
                    continue
                seen.add(plan_key)
                rows.append({
                    'dimension': dim_name,
                    'benchmark': str(bench.get('name') or dim_name),
                    'source': str(group.get('label') or group.get('id') or ''),
                    'url': url,
                })
    return rows


def github_blob_raw(url: str) -> str | None:
    parsed = urlparse(url)
    if parsed.netloc != 'github.com':
        return None
    parts = [p for p in parsed.path.split('/') if p]
    if len(parts) >= 5 and parts[2] == 'blob':
        user, repo, _, branch = parts[:4]
        rest = '/'.join(parts[4:])
        return f'https://raw.githubusercontent.com/{user}/{repo}/{branch}/{rest}'
    return None


def github_repo_clone_url(url: str) -> tuple[str, str] | None:
    parsed = urlparse(url)
    if parsed.netloc != 'github.com':
        return None
    parts = [p for p in parsed.path.split('/') if p]
    if len(parts) < 2:
        return None
    user, repo = parts[0], parts[1]
    repo = repo.removesuffix('.git')
    return f'https://github.com/{user}/{repo}.git', f'{user}__{repo}'


def github_tree_checkout(url: str) -> tuple[str | None, str | None]:
    parsed = urlparse(url)
    if parsed.netloc != 'github.com':
        return None, None
    parts = [p for p in parsed.path.split('/') if p]
    if len(parts) >= 5 and parts[2] == 'tree':
        return parts[3], '/'.join(parts[4:])
    return None, None


def huggingface_repo_info(url: str) -> tuple[str, str] | None:
    parsed = urlparse(url)
    if parsed.netloc not in {'huggingface.co', 'hf-mirror.com'}:
        return None
    parts = [p for p in parsed.path.split('/') if p]
    if len(parts) < 3 or parts[0] != 'datasets':
        return None
    org, name = parts[1], parts[2]
    return f'{org}/{name}', f'{org}__{name}'


def huggingface_clone_url(repo_id: str, hf_endpoint: str) -> str:
    endpoint = str(hf_endpoint or DEFAULT_HF_ENDPOINT).rstrip('/')
    return f'{endpoint}/datasets/{repo_id}'


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def download_direct(url: str, target: Path, timeout: int = 120) -> int:
    ensure_parent(target)
    with urlopen(url, timeout=timeout) as resp:
        data = resp.read()
    target.write_bytes(data)
    return len(data)


def clone_or_update_git(repo_url: str, target: Path, timeout: int, branch: str | None = None, sparse_path: str | None = None) -> None:
    env = dict(os.environ)
    env.setdefault('GIT_LFS_SKIP_SMUDGE', '1')
    if sparse_path and target.exists() and not (target / '.download_complete').exists():
        shutil.rmtree(target)
    if target.exists() and (target / '.git').exists():
        try:
            subprocess.run(['git', '-C', str(target), 'remote', 'set-url', 'origin', repo_url], check=True, timeout=10, env=env)
            if sparse_path:
                subprocess.run(['git', '-C', str(target), 'sparse-checkout', 'init', '--cone'], check=False, timeout=30, env=env)
                subprocess.run(['git', '-C', str(target), 'sparse-checkout', 'set', sparse_path], check=False, timeout=60, env=env)
            subprocess.run(['git', '-C', str(target), 'pull', '--ff-only'], check=True, timeout=timeout, env=env)
            return
        except subprocess.CalledProcessError:
            shutil.rmtree(target)
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    cmd = ['git', 'clone', '--depth', '1']
    if branch:
        cmd.extend(['--branch', branch])
    if sparse_path:
        cmd.extend(['--filter=blob:none', '--sparse'])
    cmd.extend([repo_url, str(target)])
    subprocess.run(cmd, check=True, timeout=timeout, env=env)
    if sparse_path:
        subprocess.run(['git', '-C', str(target), 'sparse-checkout', 'set', sparse_path], check=True, timeout=timeout, env=env)


def ensure_hf_remote(plan: Dict[str, Any], target: Path, hf_endpoint: str) -> None:
    if plan.get('mode') != 'huggingface_dataset' or not (target / '.git').exists():
        return
    repo_id, _ = huggingface_repo_info(str(plan.get('url') or '')) or ('', '')
    if not repo_id:
        return
    env = dict(os.environ)
    env.setdefault('GIT_LFS_SKIP_SMUDGE', '1')
    subprocess.run(
        ['git', '-C', str(target), 'remote', 'set-url', 'origin', huggingface_clone_url(repo_id, hf_endpoint)],
        check=False,
        timeout=10,
        env=env,
    )


def snapshot_huggingface(repo_id: str, target: Path, timeout: int, hf_endpoint: str) -> None:
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        raise RuntimeError(f'huggingface_hub 不可用: {e}')
    target.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('HF_ENDPOINT', hf_endpoint.rstrip('/'))
    snapshot_download(repo_id=repo_id, repo_type='dataset', local_dir=str(target), local_dir_use_symlinks=False)


def save_html(url: str, target: Path, timeout: int = 120) -> int:
    ensure_parent(target)
    with urlopen(url, timeout=timeout) as resp:
        data = resp.read()
    target.write_bytes(data)
    return len(data)


def planned_target(row: Dict[str, Any], dest: Path) -> Dict[str, Any]:
    url = str(row.get('url') or '').strip()
    raw_github = github_blob_raw(url)
    if raw_github:
        filename = Path(urlparse(raw_github).path).name or f"{safe_slug(row.get('benchmark'))}.dat"
        return {'mode': 'direct_file', 'url': raw_github, 'target': dest / 'github_files' / filename}
    hf = huggingface_repo_info(url)
    if hf:
        _, repo_slug = hf
        return {'mode': 'huggingface_dataset', 'url': url, 'target': dest / 'huggingface' / repo_slug}
    gh = github_repo_clone_url(url)
    if gh:
        _, repo_slug = gh
        return {'mode': 'git_repo', 'url': url, 'target': dest / 'github_repos' / repo_slug}
    if any(url.lower().split('?', 1)[0].endswith(ext) for ext in DIRECT_SUFFIXES):
        filename = Path(urlparse(url).path).name or f"{safe_slug(row.get('benchmark'))}.dat"
        return {'mode': 'direct_file', 'url': url, 'target': dest / 'files' / filename}
    host = safe_slug(urlparse(url).netloc or 'page')
    filename = safe_slug(row.get('benchmark') or row.get('dimension') or 'page') + '.html'
    return {'mode': 'html_page', 'url': url, 'target': dest / 'pages' / host / filename}


def execute_plan(row: Dict[str, Any], plan: Dict[str, Any], skip_existing: bool, timeout: int, hf_endpoint: str) -> Dict[str, Any]:
    target = Path(plan['target'])
    result = {
        'dimension': row.get('dimension'),
        'benchmark': row.get('benchmark'),
        'source': row.get('source'),
        'url': row.get('url'),
        'mode': plan['mode'],
        'target': str(target),
        'status': 'pending',
    }
    if skip_existing and is_completed_target(plan['mode'], target):
        ensure_hf_remote(plan, target, hf_endpoint)
        result['status'] = 'skipped_existing'
        if plan['mode'] == 'huggingface_dataset':
            result['mirror'] = hf_endpoint.rstrip('/')
        return result
    try:
        if plan['mode'] == 'direct_file':
            result['bytes'] = download_direct(plan['url'], target, timeout=timeout)
        elif plan['mode'] == 'git_repo':
            repo_url, _ = github_repo_clone_url(plan['url']) or ('', '')
            if not repo_url:
                raise RuntimeError('无法解析 GitHub 仓库地址')
            branch, sparse_path = github_tree_checkout(plan['url'])
            clone_or_update_git(repo_url, target, timeout=timeout, branch=branch, sparse_path=sparse_path)
        elif plan['mode'] == 'huggingface_dataset':
            repo_id, _ = huggingface_repo_info(plan['url']) or ('', '')
            if not repo_id:
                raise RuntimeError('无法解析 HuggingFace 数据集地址')
            result['mirror'] = hf_endpoint.rstrip('/')
            clone_or_update_git(huggingface_clone_url(repo_id, hf_endpoint), target, timeout=timeout)
        else:
            result['bytes'] = save_html(plan['url'], target, timeout=timeout)
        mark_completed_target(plan['mode'], target)
        result['status'] = 'ok'
    except subprocess.TimeoutExpired as e:
        result['status'] = 'timeout'
        result['error'] = f'timeout after {e.timeout}s'
    except TimeoutError as e:
        result['status'] = 'timeout'
        result['error'] = str(e)
    except Exception as e:
        err = str(e)
        if 'timed out' in err.lower() or 'timeout' in err.lower():
            result['status'] = 'timeout'
        else:
            result['status'] = 'error'
        result['error'] = err
    return result


def is_completed_target(mode: str, target: Path) -> bool:
    if mode in {'direct_file', 'html_page'}:
        return target.is_file() and target.stat().st_size > 0
    marker = target / '.download_complete'
    if marker.exists():
        return True
    if (target / '.git').exists():
        try:
            subprocess.run(
                ['git', '-C', str(target), 'rev-parse', '--verify', 'HEAD'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
                timeout=5,
            )
            return True
        except Exception:
            return False
    return False


def mark_completed_target(mode: str, target: Path) -> None:
    if mode in {'git_repo', 'huggingface_dataset'}:
        (target / '.download_complete').write_text('ok\n', encoding='utf-8')


def main() -> int:
    parser = argparse.ArgumentParser(description='按 benchmark 链接批量下载本地数据集/源码/页面')
    parser.add_argument('--dest', default=str(DEFAULT_DEST), help='下载目标目录')
    parser.add_argument('--limit', type=int, default=0, help='仅处理前 N 个条目，0 表示全部')
    parser.add_argument('--list', action='store_true', help='仅列出下载计划')
    parser.add_argument('--skip-existing', action='store_true', help='目标已存在时跳过')
    parser.add_argument('--workers', type=int, default=8, help='并发下载线程数')
    parser.add_argument('--timeout', type=int, default=90, help='单个 benchmark 下载超时时间（秒）')
    parser.add_argument('--hf-endpoint', default=os.environ.get('HF_ENDPOINT', DEFAULT_HF_ENDPOINT), help='HuggingFace 镜像地址')
    parser.add_argument('--strict', action='store_true', help='存在失败或超时时返回非 0')
    parser.add_argument('--dimension', default='', help='仅处理名称包含该关键字的维度')
    parser.add_argument('--benchmark', default='', help='仅处理名称包含该关键字的 benchmark')
    args = parser.parse_args()

    dest = Path(args.dest).resolve()
    rows = collect_catalog_sources()
    if args.dimension:
        rows = [row for row in rows if args.dimension in str(row.get('dimension') or '')]
    if args.benchmark:
        rows = [row for row in rows if args.benchmark in str(row.get('benchmark') or '')]
    if args.limit > 0:
        rows = rows[:args.limit]

    plans = [{**row, **planned_target(row, dest)} for row in rows]
    if args.list:
        printable = [{**row, 'target': str(row.get('target') or '')} for row in plans]
        print(json.dumps(printable, ensure_ascii=False, indent=2))
        return 0

    dest.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []
    max_workers = max(1, int(args.workers or 1))
    hf_endpoint = str(args.hf_endpoint or DEFAULT_HF_ENDPOINT).rstrip('/')
    os.environ['HF_ENDPOINT'] = hf_endpoint
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(execute_plan, row, row, args.skip_existing, max(1, int(args.timeout or 1)), hf_endpoint) for row in plans]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda row: (str(row.get('dimension') or ''), str(row.get('benchmark') or ''), str(row.get('url') or '')))
    manifest = {
        'generated_at': datetime.utcnow().isoformat() + 'Z',
        'dest': str(dest),
        'hf_endpoint': hf_endpoint,
        'count': len(results),
        'ok': sum(1 for row in results if row.get('status') == 'ok'),
        'skipped_existing': sum(1 for row in results if row.get('status') == 'skipped_existing'),
        'timeout': sum(1 for row in results if row.get('status') == 'timeout'),
        'error': sum(1 for row in results if row.get('status') == 'error'),
        'items': results,
    }
    (dest / 'download_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if args.strict and any(row.get('status') in {'error', 'timeout'} for row in results):
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
