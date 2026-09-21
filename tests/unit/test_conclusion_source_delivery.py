"""C-only full-source delivery and retention; fake responses, no source execution."""
from copy import deepcopy
import json
from types import SimpleNamespace as NS

import pytest

from sciel.engine import research as research_module
from sciel.engine.research import _FrozenPayload, digest
from sciel.engine.validation import ValidationResult
from sciel.hypothesis import conclusion_refs, conclusion_source, prompts
from sciel.hypothesis.conclusion_feedback import conclusion_feedback, validate_conclusions, reflection_review_context
from sciel.hypothesis.observed_facts import observed_comparison
from sciel.hypothesis.rendering import decode_evidence, serialize_record
from sciel.knowledge_base.catalogs import canonical_json
from test_research_injection import controller, begin_fixture


def setup(**settings):
    return controller(structured_auxiliary=True, conclusion_source_refs=True,
                      reflection_interval=0, **settings)


def observe(c, db, parent, source, *, status='evaluated', score=.6):
    ident, _ = begin_fixture(c, c.context(db, parent, 0, c.attempts+1))
    raw = NS(text='HYPOTHESIS_JSON: ' + json.dumps(dict(
        hypothesis='The edit changes a source expression.', predicted_low=0., predicted_high=.2,
        reasoning='Its mechanism and general benefit remain uncertain.', confidence=.5)),
        cost=.01, model='fake', prompt_tokens=10, completion_tokens=10)
    assert c.proposal(ident, raw, source, ValidationResult(source, None, [], None, []))
    record = c.resolve(ident, source=source, status=status,
                       score=score if status == 'evaluated' else None,
                       failure=None if status == 'evaluated' else 'inert evaluator failure')
    return record


def review(job, *, lessons=False):
    evidence = job['evidence']
    catalog = conclusion_refs.catalog(evidence)
    selected = [row for row in catalog['sources'] if row['field'].startswith('conclusion_source.')][-1]
    return json.dumps(dict(verdict=evidence['forecast_verdict'], explanation='One measured case only.',
        lessons=([dict(rule='The source edit improved this measured case only.', polarity='positive',
            scope='branch', branch=evidence['branch'], evidence_ids=[evidence['id']])] if lessons else []),
        mechanism_review=[dict(claim_ref=catalog['claims'][0]['id'], source_refs=[selected['id']],
            evidence_kind='source_effect', assessment='unknown', direct_effect='This exact source text is present.',
            untested_downstream='No semantic truth or future benefit is established.')]))


def apply(c, job, **kw):
    raw = review(job, **kw)
    result, repair = c.decode_conclusion(job, raw)
    assert result is not None and repair is None
    c.apply_conclusion(job, result, raw)
    return result


@pytest.mark.parametrize('status,score', [('evaluated', .6), ('evaluated', .2), ('eval_failed', 0.)])
def test_resolve_to_full_c_reference_and_cold_resume_never_bloats_ordinary_memory(status, score):
    c, db, parent, _ = setup()
    source = 'def run_search():\n' + ''.join(f'    inert_{i} = {i}\n' for i in range(700)) + '    return late_result\n'
    record = observe(c, db, parent, source, status=status, score=score)
    before = deepcopy(c.knowledge.state())
    assert 'late_result' not in record['change_excerpt']
    job, = c.prepare_conclusions()
    assert c._conclusion_sources == {} and c.queue == []
    projection = job['evidence']['conclusion_source']
    assert projection['mode'] == 'full' and projection['coverage']['complete']
    assert projection['diff'] == conclusion_source.full_diff(parent.code, source)
    sent = decode_evidence(json.JSONDecoder().raw_decode(job['prompt'].split('EVIDENCE_JSON:\n')[1])[0])
    assert sent['conclusion_source'] == projection and 'late_result' in sent['conclusion_source']['diff']
    assert sent['source_sha256'] == digest(source)
    apply(c, job, lessons=status == 'evaluated' and score > parent.score)
    c.finish_conclusions(reflect=False)
    # Record storage stays byte-identical, including old6000 and code_fragment rules.
    assert c.knowledge.experiments == before['experiments']
    for lesson in c.knowledge.lessons:
        assert 'conclusion_source' not in canonical_json(lesson)
        assert 'source_evidence' not in canonical_json(lesson)
    feedback = conclusion_feedback(c.conclusions, c.knowledge._evidence(), parent.id, max_chars=16000)
    assert feedback and 'candidate_source' not in canonical_json(feedback)
    assert 'parent_source"' not in canonical_json(feedback)
    reflections = reflection_review_context(c.conclusions, c.knowledge._evidence(), [record['id']], max_chars=32000)
    assert 'candidate_source' not in canonical_json(reflections)
    saved = deepcopy(c.state())
    restored, _, _, _ = setup()
    restored.load_state(saved)
    assert restored.state() == saved
    assert c.knowledge.experiments[0]['actual_delta'] == (score-.5 if status == 'evaluated' else None)


@pytest.mark.parametrize('mutate', [
    lambda job: job['source_evidence'].__setitem__('candidate_source', 'different source'),
    lambda job: job['source_evidence']['projection'].__setitem__('diff', '+forged\n'),
    lambda job: job['evidence']['conclusion_source'].__setitem__('source_sha256', 'a'*64),
    lambda job: job.pop('source_evidence'),
])
def test_detached_c_job_and_cold_conclusion_reject_tampered_sources(mutate):
    c, db, parent, _ = setup()
    observe(c, db, parent, 'def run_search():\n    return .6\n')
    job, = c.prepare_conclusions()
    bad = dict(job)
    mutate(bad)
    with pytest.raises(ValueError):
        c.decode_conclusion(_FrozenPayload.from_dict(bad), review(job))
    apply(c, job)
    saved = c.state()
    saved['conclusions'][job['record']['id']]['source_evidence']['candidate_source'] += '# altered'
    restored, _, _, _ = setup()
    with pytest.raises(ValueError, match='source'):
        restored.load_state(saved)


def test_evicted_records_still_deliver_detached_c_and_pairs_retire_with_conclusion():
    c, db, parent, _ = setup(max_experiments=1)
    records = [observe(c, db, parent, f'def run_search():\n    return {i}\n') for i in range(3)]
    assert len(c.knowledge.experiments) == 1 and len(c._conclusion_sources) == 3
    first, = c.prepare_conclusions(limit=1)
    assert len(c._conclusion_sources) == 2
    rest = c.prepare_conclusions()
    assert not c._conclusion_sources
    for job in [first, *rest]:
        assert job['evidence']['conclusion_source']['source_sha256'] == job['record']['source_sha256']
        apply(c, job)
    c.finish_conclusions(reflect=False)
    assert list(c.conclusions) == [records[-1]['id']]
    assert len(c.state()['conclusions']) == 1


def test_actual_escaped_c_evidence_budget_reserves_catalog_and_initializer(monkeypatch):
    c, db, parent, _ = setup()
    source = 'def run_search():\n' + ''.join(f'    inert_{i} = "界🙂{i}"\n' for i in range(3000))
    observe(c, db, parent, source)
    limit = 52000
    monkeypatch.setattr(research_module, 'MAX_CONTEXT_CHARS', limit)
    monkeypatch.setattr(prompts, 'MAX_CONTEXT_CHARS', limit)
    # An actual serialized initializer allocation, not a raw-character estimate.
    monkeypatch.setattr(c, '_initializer_context', lambda ids, available: {'inert': '界'*80})
    monkeypatch.setattr(prompts, 'initializer_block', lambda value: canonical_json(value))
    job, = c.prepare_conclusions()
    view = job['evidence']
    projection = view['conclusion_source']
    assert projection['mode'] == 'windows'
    assert projection['coverage']['omitted_changed_chars'] > 0
    assert projection['coverage']['omitted_diff_ranges']
    used = (len(serialize_record(view)) + len(canonical_json(observed_comparison(view)))
        + len(conclusion_refs.instruction(view)) + len(conclusion_source.RULES)
        + len(canonical_json({'inert': '界'*80})))
    assert used <= limit
    assert len(conclusion_source.full_diff(parent.code, source)) < len(canonical_json({'diff': conclusion_source.full_diff(parent.code, source)}))
    for window in projection['windows']:
        full = conclusion_source.full_diff(parent.code, source)
        assert full[window['start']:window['end']] == window['text']
    apply(c, job)
    restored, _, _, _ = setup()
    restored.load_state(c.state())


def test_legacy_queue_does_not_gain_reconstructed_source_or_changed_catalog():
    from test_mechanism_review import controller as legacy
    c, _, _, _ = legacy()
    record = deepcopy(c.queue[0])
    catalog = conclusion_refs.catalog(record)
    job, = c.prepare_conclusions()
    assert 'conclusion_source' not in job['evidence'] and 'source_evidence' not in job
    assert conclusion_refs.catalog(job['evidence']) == catalog
    assert c.knowledge.experiments == [record]


def test_failed_source_mismatch_is_rejected_before_admission():
    c, db, parent, _ = setup()
    ident, _ = begin_fixture(c, c.context(db, parent, 0, 1))
    c.pending[ident].update(hypothesis={'text': 'fixture'}, source_sha256=digest('bound source'))
    before = c.knowledge.state()
    with pytest.raises(ValueError, match='source identities'):
        c.resolve(ident, status='eval_failed', source='unbound source', failure='fixture')
    assert c.knowledge.state() == before and ident in c.pending and not c.queue
