"""Worker-owned normalized source history; inert Engine evaluation only."""
import copy
import hashlib
import json

import pytest

from sciel.engine import checkpoint
from sciel.engine.db import normalized_hash
from sciel.hypothesis import circle_restart_body as body
from test_circle_generation_engine import BODIES, build as single_build, config as single_config, response
from test_circle_pair_engine import build as pair_build, config as pair_config, candidate, paired, actions


ALIAS = BODIES[0].rstrip('\n') + '   \n'


def build(path, mode, *, resume_replies=False):
    options = dict(budget=2, max_iterations=3, pop_cap=1, grid_bins=1)
    if mode == 'single':
        replies = [response(BODIES[0]), response(ALIAS), response(BODIES[1])]
        factory, config = single_build, single_config(**options)
    else:
        replies = [paired(), paired(candidate(ALIAS), candidate(BODIES[1]))]
        factory, config = pair_build, pair_config(**options)
    # Even the vulnerable alias path has a declared fake measurement; failures
    # below identify an extra dispatch rather than an unrelated missing fixture.
    engine = factory(path, replies[1:] if resume_replies else replies, cfg=config,
                     scores={BODIES[0]: .4, ALIAS: .41})
    return engine


@pytest.mark.parametrize('mode', ['single', 'pair'])
@pytest.mark.parametrize('resume', [False, True])
def test_measured_rejected_child_alias_is_not_evaluated_again(mode, resume, tmp_path):
    engine = build(tmp_path/'run', mode)
    earlier = []
    if resume:
        engine._stop_at = 1
        assert engine.run()['reason'] == 'paused'
        earlier = list(engine.fixture_evaluated)
        assert normalized_hash(body.source_for(BODIES[0])) not in engine.db.code_hashes(0)
        restored = build(tmp_path/'run', mode, resume_replies=True)
        checkpoint.load(restored, str(tmp_path/'run/checkpoint.json'))
        assert len(restored.circle_generation.seen_proofs) == 1
        engine = restored
    engine.run()
    assert body.source_for(ALIAS) not in engine.fixture_evaluated
    assert earlier + engine.fixture_evaluated == [body.source_for(), body.source_for(BODIES[0]), body.source_for(BODIES[1])]
    assert engine.circle_generation.evaluations == 2
    assert len(engine.circle_generation.seen_proofs) == 2
    assert engine.research.knowledge.experiments[0]['score'] == .4
    assert engine.research.knowledge.experiments[-1]['source_sha256'] == hashlib.sha256(body.source_for(BODIES[1]).encode()).hexdigest()
    if mode == 'single':
        repeated = engine.research.knowledge.experiments[1]
        assert 'known_normalized_source' in repeated['failure']
        assert repeated['score'] is repeated['source_sha256'] is repeated['actual_delta'] is None
    else:
        group = engine.research.knowledge.experiments[1]['proposal_group']
        assert group['entries'][0]['gate']['reason'] == 'known_normalized_source'
        assert group['selected_index'] == 1
        assert engine.research.knowledge.experiments[0]['proposal_group']['entries'][1]['disposition'] == 'eligible_unselected'


@pytest.mark.parametrize('mode', ['single', 'pair'])
def test_evicted_measured_child_alias_is_still_consumed(mode, tmp_path):
    options = dict(budget=3, max_iterations=3, pop_cap=1, grid_bins=1)
    if mode == 'single':
        engine = single_build(tmp_path/'run', [response(), response(BODIES[1]), response(ALIAS)],
            cfg=single_config(**options), scores={ALIAS: .41})
    else:
        fresh = 'return best_params + uniform_noise * 0.75\n'
        engine = pair_build(tmp_path/'run', [paired(), paired(candidate(BODIES[1]), candidate(fresh)),
            paired(candidate(ALIAS), candidate('return np.random.choice(26)\n'))],
            cfg=pair_config(**options), scores={ALIAS: .41})
    engine.run()
    assert normalized_hash(body.source_for(BODIES[0])) not in engine.db.code_hashes(0)
    assert body.source_for(ALIAS) not in engine.fixture_evaluated
    assert engine.fixture_evaluated == [body.source_for(), body.source_for(BODIES[0]), body.source_for(BODIES[1])]
    assert engine.circle_generation.evaluations == 2 and len(engine.circle_generation.seen_proofs) == 2
    final = engine.research.knowledge.experiments[-1]
    assert final['score'] is final['source_sha256'] is final['actual_delta'] is None


@pytest.mark.parametrize('side', ['parent_body', 'target_body'])
def test_pair_replay_requires_prior_proof_derived_history_even_when_gates_unchanged(tmp_path, side):
    fresh = 'return best_params + uniform_noise * 0.75\n'
    engine = pair_build(tmp_path/'run', [paired(), paired(candidate(BODIES[1]), candidate(fresh))],
        cfg=pair_config(budget=2, max_iterations=2, pop_cap=1, grid_bins=1), scores={BODIES[0]: .4})
    engine.run()
    log = tmp_path/'run/sciel_events.jsonl'
    saved = json.loads((tmp_path/'run/checkpoint.json').read_text())
    anchor = saved['provenance']['journal_anchor']
    engine.circle_generation.assert_journal(log, anchor)
    prefix = log.read_bytes()[:anchor['byte_offset']]
    rows = [json.loads(line) for line in prefix.splitlines()]
    groups = [row for row in rows if row.get('phase') == 'circle_generation' and row.get('action') == 'proposal_group']
    assert len(groups) == 2
    required = normalized_hash(body.source_for(engine.circle_generation.seen_proofs[0][side]))
    assert required in groups[1]['selection_snapshot']['code_hashes']
    # Neither second-request proposal matches this old source. Removing its
    # hash cannot change their gate results, so generic gate replay alone would
    # accept the edit after the outer journal anchor was recomputed.
    from sciel.engine.circle_pair_selection import assess
    before = assess(paired(candidate(BODIES[1]), candidate(fresh)), groups[1]['selection_snapshot'])
    groups[1]['selection_snapshot']['code_hashes'].remove(required)
    after = assess(paired(candidate(BODIES[1]), candidate(fresh)), groups[1]['selection_snapshot'])
    assert before['gates'] == after['gates'] and before['evidence'] == after['evidence']
    changed = b''.join((json.dumps(row, separators=(',', ':'))+'\n').encode() for row in rows)
    tampered = tmp_path/'rehash.jsonl';tampered.write_bytes(changed)
    new_anchor = {**anchor, 'byte_offset': len(changed), 'prefix_sha256': hashlib.sha256(changed).hexdigest()}
    with pytest.raises(ValueError, match='saved selection history'):
        engine.circle_generation.assert_journal(tampered, new_anchor)


def test_proof_history_does_not_mutate_snapshot_or_consume_unselected_source(tmp_path):
    engine = build(tmp_path/'run', 'pair')
    engine._stop_at = 1
    assert engine.run()['reason'] == 'paused'
    original = copy.deepcopy(engine.circle_generation.seen_proofs)
    codes = engine.db.code_hashes(0); before = set(codes)
    exact, normalized = engine.circle_generation._source_history(original, codes)
    assert normalized_hash(body.source_for(BODIES[0])) in normalized
    assert normalized_hash(body.source_for(BODIES[1])) not in normalized
    assert hashlib.sha256(body.source_for(BODIES[1]).encode()).hexdigest() not in exact
    assert codes == before and original == engine.circle_generation.seen_proofs
