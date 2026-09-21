"""Exported reflection proofs must retain the exact reconstructed request.

The extra reply is synthetic saved data. No model, solver or evaluator runs.
"""
import copy
import hashlib
import json

import pytest

from sciel.engine.external_research_v2 import validate
from sciel.knowledge_base.catalogs import canonical_json
from test_external_reflection import saved, job_and_plan, receipt, admit


@pytest.fixture(scope='module')
def extended(saved):
    job, plan = job_and_plan(saved)
    result = admit(job, plan, receipt(saved))
    return result['extended_capsule'], saved[2], plan


@pytest.mark.parametrize('which', ['historical', 'new'])
def test_import_cannot_accept_replaced_reflection_request_commitment(extended, which):
    capsule, config, plan = extended
    altered = copy.deepcopy(capsule)
    operations = [op for op in altered['operations'] if op['kind'] == 'reflection']
    operation = operations[0] if which == 'historical' else operations[-1]
    if which == 'historical':
        assert operation['journal'] != operations[-1]['journal']
    rows = [json.loads(line) for line in altered['journals'][operation['journal']]['journal'].splitlines()]
    terminal = rows[operation['event_seq'] - 1]
    starts = [row for row in rows if row.get('phase') == 'aux_start'
              and row.get('kind') == 'reflection' and row.get('reference') == terminal['through']
              and row['event_seq'] < terminal['event_seq']]
    request = starts[-1]
    if which == 'new':
        assert request['prompt_sha256'] == plan['prompt_sha256']
    request['prompt_sha256'] = hashlib.sha256(
        (plan['prompt'] + '\nSynthetic adversarial alteration of the request.').encode()).hexdigest()
    altered['journals'][operation['journal']]['journal'] = ''.join(canonical_json(row) + '\n' for row in rows)
    # Every observation, response, presentation projection and terminal state
    # still matches. Admission must independently authenticate request bytes.
    assert altered['terminal'] == capsule['terminal']
    with pytest.raises(ValueError, match='prompt'):
        validate(altered, config=config, metric=altered['metric'])
