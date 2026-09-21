"""Helper transport must not block if a candidate swaps an output file."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

from sciel.eval.repeat_io import RepeatInfrastructureError, _read_json


def test_rejects_duplicate_keys_and_non_object_output(tmp_path):
    output = tmp_path / 'output.json'
    output.write_text('{"value": 1, "value": 2}')
    with pytest.raises(ValueError, match='duplicate JSON key'):
        _read_json(output)
    output.write_text('[]')
    with pytest.raises(RepeatInfrastructureError, match='must be an object'):
        _read_json(output)


@pytest.mark.skipif(not hasattr(os, 'mkfifo'), reason='requires POSIX FIFO')
def test_file_replaced_by_fifo_between_stat_and_open_does_not_block(tmp_path):
    output = tmp_path / 'output.json'
    output.write_text('{}')
    source = textwrap.dedent('''
        import os
        from pathlib import Path
        import sys
        from sciel.eval import repeat_io

        path = Path(sys.argv[1])
        original_open = os.open

        def swap_then_open(target, flags, *args, **kwargs):
            path.unlink()
            os.mkfifo(path)
            return original_open(target, flags, *args, **kwargs)

        repeat_io.os.open = swap_then_open
        try:
            repeat_io._read_json(path)
        except repeat_io.RepeatInfrastructureError:
            sys.exit(0)
        raise AssertionError('non-regular helper output was accepted')
    ''')
    package_root = str(Path(__file__).resolve().parents[2] / 'src')
    subprocess.run(
        [sys.executable, '-c', source, str(output)],
        env={**os.environ, 'PYTHONPATH': package_root},
        check=True, timeout=5, capture_output=True,
    )
