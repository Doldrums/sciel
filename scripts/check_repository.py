#!/usr/bin/env python3
"""Enforce the maintained-file allowlist and repository size limits."""

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
MARKDOWN = {'README.md', 'CONTRIBUTING.md'}
MAX_LINES = 500


def git(*arguments, **kwargs):
    return subprocess.run(['git', '-C', str(ROOT), *arguments],
                          capture_output=True, check=False, **kwargs)


def main():
    listing = git('ls-files', '--cached', '--others', '--exclude-standard', '-z')
    if listing.returncode:
        raise SystemExit(listing.stderr.decode())
    names = sorted({name.decode() for name in listing.stdout.split(b'\0') if name})
    existing = [name for name in names if (ROOT / name).exists() or (ROOT / name).is_symlink()]
    checked = git('check-ignore', '--no-index', '--stdin', '-z',
                  input=b'\0'.join(name.encode() for name in existing) + b'\0')
    if checked.returncode not in (0, 1):
        raise SystemExit(checked.stderr.decode())
    ignored = {name.decode() for name in checked.stdout.split(b'\0') if name}
    failures = []
    for name in existing:
        path = ROOT / name
        if name in ignored:
            failures.append(f'{name}: tracked file is outside the allowlist')
        if path.is_symlink() or not path.is_file():
            failures.append(f'{name}: source entries must be regular files')
            continue
        if path.suffix.lower() in {'.md', '.markdown', '.rst'} and name not in MARKDOWN:
            failures.append(f'{name}: documentation belongs in README.md or CONTRIBUTING.md')
        try:
            lines = len(path.read_text(encoding='utf-8').splitlines())
        except UnicodeError:
            failures.append(f'{name}: binary artifacts do not belong in the source tree')
            continue
        if lines > MAX_LINES:
            failures.append(f'{name}: {lines} lines exceeds {MAX_LINES}')
    for required in MARKDOWN:
        if required not in existing:
            failures.append(f'{required}: required repository documentation is missing')
    for failure in failures:
        print(failure)
    if failures:
        return 1
    print(f'Checked {len(existing)} maintained files; all repository constraints passed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
