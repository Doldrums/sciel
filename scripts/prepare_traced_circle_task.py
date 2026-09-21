"""Prepare the pinned repeat-trace Circle task using source bytes only.

No evaluator, generated candidate, optimizer, runtime probe or model is invoked.
The canonical task remains untouched. The output is a fresh task identity, not
a migrated measurement or a checkpoint that can resume a historical study.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from sciel.engine.repeat_support import validate_circle_task, traced_execution_policy
from sciel.eval.registry import load_task
from sciel.hypothesis.circle_source_family import builder_for_profile


def prepare(canonical_root, target, *, execution_profile='paired_restart_trace/1'):
    """Copy a validated canonical snapshot and replace only its initial source.

    Build in a sibling temporary directory and validate both task identities
    before publishing. A target containing anything is never overwritten.
    """
    if execution_profile not in ('paired_restart_trace/1', 'paired_restart_trace/2'):
        raise ValueError('traced task requires an explicit supported execution profile')
    builder = builder_for_profile(execution_profile)
    canonical_root, target = Path(canonical_root).resolve(), Path(target).absolute()
    canonical = load_task(str(canonical_root))
    validate_circle_task(canonical)
    if target.is_symlink() or (target.exists() and (not target.is_dir() or any(target.iterdir()))):
        raise ValueError('traced task target must be absent or an empty directory')
    if target.resolve() == canonical_root or canonical_root in target.resolve().parents:
        raise ValueError('traced task target must be outside the canonical task')
    paths = [Path(name) for name in ('task.yaml', 'evaluator.py', 'initial_program.py')]
    assets = canonical_root / 'assets'
    if assets.exists():
        if assets.is_symlink() or not assets.is_dir():
            raise ValueError('canonical assets must be a regular directory')
        for path in sorted(assets.rglob('*')):
            if path.is_symlink() or (not path.is_file() and not path.is_dir()):
                raise ValueError('canonical assets must contain regular files and directories')
            if path.is_file():
                paths.append(path.relative_to(canonical_root))
    payload = {}
    for rel in paths:
        path = canonical_root / rel
        if path.is_symlink() or not path.is_file():
            raise ValueError('canonical task files must be regular files')
        payload[rel] = path.read_bytes()
    source = builder.source_for().encode('utf-8')
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.sciel_traced_task_', dir=target.parent) as temporary:
        staged = Path(temporary) / 'task'
        staged.mkdir()
        for rel, raw in payload.items():
            path = staged / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        # Revalidate the copied bytes, not just the source path read earlier.
        validate_circle_task(load_task(str(staged)))
        (staged / 'initial_program.py').write_bytes(source)
        final = load_task(str(staged))
        validate_circle_task(final, initial_source_policy=builder.OPERATION)
        files = {rel.as_posix(): hashlib.sha256((staged / rel).read_bytes()).hexdigest()
                 for rel in paths}
        policy = traced_execution_policy(builder.OPERATION)
        if target.exists():
            if target.is_symlink() or not target.is_dir() or any(target.iterdir()):
                raise ValueError('traced task target changed before publication')
            target.rmdir()  # Only an empty directory; never remove existing data.
        os.rename(staged, target)
    return {'schema': 'sciel-prepared-traced-circle-task/1', 'task_root': str(target.resolve()),
        'task': final.qualified_name, 'task_hash': final.task_hash,
        'canonical_task_hash': canonical.task_hash, 'initial_source_policy': builder.OPERATION,
        **policy, 'files': files, 'executed': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--canonical-task', type=Path, default=ROOT / 'tasks/oe/circle_packing')
    parser.add_argument('--output', '--out', dest='output', type=Path, required=True)
    parser.add_argument('--execution-profile', choices=('paired_restart_trace/1', 'paired_restart_trace/2'),
        default='paired_restart_trace/1')
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.canonical_task, args.output, execution_profile=args.execution_profile),
                     sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
