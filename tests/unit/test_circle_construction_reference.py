"""Check construction examples and preserve the historical full-helper guard."""
from pathlib import Path

import numpy as np
import pytest

from sciel.hypothesis import circle_restart_body as builder


@pytest.mark.parametrize('name', ['indexed_array_example', 'comprehension_example'])
def test_documented_construction_is_admissible_and_builds_the_stated_local_array(name):
    reference = builder.interface_description()['code_contract']['construction']
    body = reference[name] + 'return values\n'
    assert builder.validate_body(body) == body
    # Execute only this tiny, repository-authored fragment, never a solver or
    # a returned model body. The four-element value is not a valid restart.
    namespace = {'np': np}
    exec(compile(builder._HEADER + builder._indented(body), '<construction-reference>', 'exec'), namespace)
    arguments = [np.zeros(78) for _ in range(4)]
    values = namespace['propose_restart'](*arguments, 0)
    np.testing.assert_array_equal(values, np.arange(4, dtype=np.float64))
    assert values.dtype == np.float64
    assert all(not np.shares_memory(values, array) for array in arguments)
    assert all(not array.any() for array in arguments)


@pytest.mark.parametrize('body', [
    'centers = []\ncenters.append([0.25, 0.5])\nreturn ordinary_params\n',
    'centers = np.zeros((26, 2))\ncenters.append([0.5, 0.25])\nreturn ordinary_params\n',
])
def test_append_responses_remain_outside_the_legacy_full_helper_guard(body):
    # Typed-body admission must not silently widen the legacy full-helper policy.
    with pytest.raises(ValueError, match='array attribute is outside'):
        builder.proposals.validate_helper(builder._HEADER + builder._indented(body))
    assert builder._sha(Path(builder.proposals.__file__).read_bytes()) == builder.PINNED_PROPOSALS_SHA256
    assert builder.proposals.PINNED_BASELINE_SHA256 == builder.PINNED_BASELINE_SHA256
