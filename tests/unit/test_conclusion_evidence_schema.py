"""Initial C and repairs share the portable evidence-bound wire, not admission."""
from copy import deepcopy
import json

import pytest

from sciel.hypothesis import conclusion_refs as refs
from test_conclusion_source_refs import evidence, entry
from test_conclusion_repair_schema import matches
from test_conclusion_format_repair import setup
from test_research_transport_split import two_jobs


def test_initial_wire_shares_exact_proven_repair_constraints_except_name():
    data = evidence()
    initial = refs.evidence_response_format(data)
    repair = refs.repair_response_format(data)
    assert initial['json_schema']['name'] == 'sciel_conclusion_evidence_refs_v2'
    initial['json_schema']['name'] = repair['json_schema']['name']
    assert initial == repair
    assert 'maxItems' not in json.dumps(initial) and 'maxLength' not in json.dumps(initial)
    review = initial['json_schema']['schema']['properties']['mechanism_review']
    assert not matches(review, [entry(source_refs=[], assessment='directly_supported')])
    assert not matches(review, [entry(claim_ref='foreign')])
    assert not matches(review, [entry(source_refs=['foreign'])])
    assert not matches(review, [entry(claim_ref=None, source_refs=[],
        evidence_kind='forecast', assessment='contradicted')])


def test_prepared_parallel_requests_use_each_detached_evidence_after_live_mutation():
    c, model, events, charges = two_jobs()
    c.queue[1]['hypothesis']['text'] = 'Another specific source change.'
    jobs = c.prepare_conclusions()
    frozen = [deepcopy(j['evidence']) for j in jobs]
    c.knowledge._experiments.clear()
    requests = [c.prepare_aux('conclusion', j['prompt'], j['evidence']['id'],
        conclusion_evidence=j['evidence']) for j in jobs]
    for request, data in zip(requests, frozen):
        assert request['options']['response_format'] == refs.evidence_response_format(data)
        request['options']['response_format']['json_schema']['schema']['properties'].clear()
        assert request['options']['response_format'] == refs.evidence_response_format(data)
    assert requests[0]['options'] != requests[1]['options']
    starts = [e for e in events if e['phase'] == 'aux_start']
    assert [e['response_format'] for e in starts] == [r['options']['response_format'] for r in requests]
    assert len(requests) == c.aux_calls == 2 and model.calls == charges == []


@pytest.mark.parametrize('kind,reference,options', [
    ('conclusion', 'r1', {}),
    ('conclusion', 'wrong', {'conclusion_evidence': evidence()}),
    ('reflection', 'r1', {'conclusion_evidence': evidence()}),
    ('conclusion', 'r1', {'conclusion_evidence': evidence(), 'repair_evidence': evidence()}),
])
def test_missing_or_misbound_evidence_cannot_reserve_or_charge(kind, reference, options):
    c, model, _, charges = setup([])
    with pytest.raises(ValueError, match='evidence'):
        c.prepare_aux(kind, 'unissued prompt', reference, **options)
    assert c.aux_calls == 0 and not model.calls and not charges
