"""Exact scalar observations survive retained-card budgets without metric heuristics."""
from copy import deepcopy
from retained_observation_fixtures import retained_fixture
import json
from types import SimpleNamespace

import pytest

from sciel.methods.models import content_hash
from sciel.engine.research_cells_context import SCHEMA, compact_context
from sciel.research_tools.observation_cards import format_context
from sciel.research_tools.retained_observations import retained_bundle
from sciel.research_tools.result_projection import verify_projection



def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(',', ':'), allow_nan=False)


def frozen_case():
    fixture = retained_fixture(summaries=True)
    artifacts = fixture['artifacts']
    assert all(content_hash(artifact) == ident for ident, artifact in artifacts.items())
    cards = fixture['actual_context']['cards']
    measurements = [card for card in cards if card['source']['receipt_id']]
    records = [(card['source']['receipt_id'], artifacts[card['source']['receipt_id']]['payload'])
               for card in reversed(measurements)]
    lookup = dict(records)

    class SavedIndex:
        records = lookup
        receipt_revisions = {card['source']['receipt_id']: card['revision_id'] for card in measurements}

        def receipt(self, ident, tool_id):
            result = lookup[ident]
            assert result['tool_id'] == tool_id
            return deepcopy(result)

        def proposal(self, ident):
            return deepcopy(artifacts[ident]['payload'])

    terminal = deepcopy(cards[-1])
    evidence = {'records': records, 'index': SavedIndex(),
                'proposal': artifacts[terminal['decision']['proposal_id']]['payload']}
    lifecycle = SimpleNamespace(config={'max_context_chars': 10000}, _state={'methods': {}})
    return fixture, lifecycle, terminal, evidence


def wrapper(record):
    receipt = record['receipt']
    return {'output': receipt.get('output'), 'result': record['result'],
            'replay_policy': record['replay_policy'], 'reason': receipt.get('reason'),
            'diagnostic': receipt.get('diagnostic')}


def values(value):
    # Expand representation only; original arrays stay atomic.
    if value.get('schema') == 'sciel-tool-result-projection/4':
        def visit(item, path=''):
            if type(item) is dict and item:
                return {key: child for k, v in item.items()
                        for key, child in visit(v, path+'/'+k.replace('~', '~0').replace('/', '~1')).items()}
            return {path: item}
        return visit(value['fields'])
    return {field['path']: field['value'] for field in value.get('fields', [])}


@pytest.mark.parametrize('cold', [False, True])
def test_actual_three_receipts_keep_all_retrospective_magnitudes_and_terminal(cold):
    fixture, lifecycle, terminal, evidence = frozen_case()
    before = encoded(fixture)
    assert len(encoded(wrapper(evidence['records'][-1][1]))) > 10000
    if cold:
        # Actual durable JSON roundtrip; source/receipt values remain immutable.
        terminal = json.loads(encoded(terminal))
        evidence['records'] = json.loads(encoded(evidence['records']))
    cards, omissions = retained_bundle(lifecycle, None, terminal, evidence, budget=10000)
    envelope = {'schema': SCHEMA, 'cards': cards, 'omitted_count': sum(omissions.values()),
                'omission_reasons': omissions}
    assert len(encoded(envelope)) <= 10000 and len(format_context(cards)) <= 10000
    assert compact_context(envelope, max_chars=10000) == envelope
    assert len(cards) == 4 and not omissions
    original_cards = {card['source']['receipt_id']: card for card in fixture['actual_context']['cards']}
    for card in cards:
        ident = card['source']['receipt_id']
        assert card['source'] == original_cards[ident]['source']
        assert card['observations']['resources'] == original_cards[ident]['observations']['resources']
        if ident:
            assert card['comparison_assessment'] == original_cards[ident]['comparison_assessment']
    latest = evidence['records'][-1]
    actual = next(card for card in cards if card['source']['receipt_id'] == latest[0])
    projection = json.loads(actual['observations']['output_excerpt'])
    verify_projection(projection, wrapper(latest[1]))
    got = values(projection)
    summaries = latest[1]['receipt']['output']['observations']['late_gain_summary']
    for cutoff, summary in summaries.items():
        for key, value in summary.items():
            assert got.get('/output/observations/late_gain_summary/'+cutoff+'/'+key) == value
    decision = cards[-1]['decision']
    assert decision['next_decision'] == terminal['decision']['next_decision']
    assert decision['omitted_chars']['interpretation'] < decision['original_chars']['interpretation']
    assert encoded(fixture) == before
