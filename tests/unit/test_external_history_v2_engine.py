"""Journal-authenticated nonempty memory; all providers/measurements are inert.

The real worker/controller write the source history. Numeric receipts are fixed
geometry fixtures; no candidate, optimizer, runtime or external API is executed.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest

from sciel.config import apply_defaults
from sciel.engine import external_research as external
from sciel.engine import checkpoint
from sciel.engine.circle_generation import CircleGeneration
from sciel.engine.loop import Engine
from sciel.engine.llm.client import LLMResponse
from sciel.engine.research import ResearchController
from sciel.engine.types import Snapshot
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from sciel.hypothesis import circle_optimizer_traced_body as builder
from sciel.hypothesis.measurement import metric_context
from sciel.hypothesis.rendering import decode_evidence
from sciel.knowledge_base.catalogs import canonical_json, canonical_sha
from sciel.knowledge_base.paired_comparisons import build_comparison
from sciel.run_log import RunLog

from test_comparison_review_flow import review_proposal
from test_circle_intervention_engine import task_copy
from test_optimizer_traced_engine import OutcomeDraws
from test_research_measurement_state import circle_metric
from test_research_paired_comparisons import ADAPTER, LIMITS, receipt
from test_restart_prediction_engine import config as engine_config, spec


BODIES = [f"return best_params + uniform_noise * {i / 4}\n"
          for i in range(1, 5)]
RULES = ("Gain.", "Regression.")
HYPOTHESIS = "Fixture restart."


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def research_config():
    cfg = engine_config()
    cfg['research'] = dict(enabled=True, model='fixture',
        restart_predictions=True, structured_auxiliary=True, comparison_reviews=True,
        batch_only_lessons=True, max_lessons=8, max_experiments=8, max_fragments=4,
        reflection_interval=1, max_aux_calls=4)
    return apply_defaults(cfg)['research']


def parent():
    return NS(id='p1', parent_id=None, island=0, code=builder.source_for(),
              score=.5, cases=cases(.5), card='')


def cases(score):
    # Ordinary fake evaluator measures only its scalar score. Physical packing
    # facts belong to the separately certified, fixed paired receipts below.
    return {}


def snapshot(f):
    context = f.c.context(NS(_all={'p1': f.parent}), f.parent, 0, f.c.attempts + 1)
    return Snapshot(island=0, op='circle_generate', model='fixture',
        iteration_no=f.c.attempts + 1, parent_id='p1', parent_code=f.parent.code,
        parent_score=f.parent.score, task_prompt='oe/circle_packing',
        research_context=context, kb_lessons=context['lesson_texts'])


class Model:
    models = ['fixture']
    weights = [1.]
    max_retries = 1

    def __init__(self, *, generation_start=0):
        self.index = generation_start
        self.calls = []
        self.reflection = None

    def choose_model(self, rng):
        return 'fixture'

    def generate(self, prompt, model, temperature=None, *, response_format=None):
        name = response_format['json_schema']['name']
        if name.startswith('sciel_conclusion_'):
            value = dict(verdict='CONFIRMED', explanation='The fixture delta lies inside its declared interval.',
                lessons=[], mechanism_review=[dict(evidence_kind='unavailable', claim_quote=HYPOTHESIS,
                    source_quotes=[], assessment='unknown', forecast_ref=None,
                    direct_effect='The numeric fixture establishes no causal mechanism.',
                    untested_downstream='Generalization remains untested.')])
            role = 'C'
        elif name.startswith('sciel_reflection_'):
            assert self.reflection is not None, 'unexpected reflection replay'
            value = copy.deepcopy(self.reflection)
            self.reflection = None
            role = 'R'
        else:
            assert self.index < len(BODIES), 'unexpected generation replay'
            value = dict(hypothesis=dict(hypothesis=HYPOTHESIS, predicted_low=-.2,
                predicted_high=.2, reasoning='Fixture only.',
                confidence=.5, restart_predictions=spec()), body=BODIES[self.index])
            self.index += 1
            role = 'G'
        raw = json.dumps(value)
        self.calls.append(dict(role=role, prompt=prompt, text=raw,
                               response_format=copy.deepcopy(response_format)))
        return LLMResponse(raw, model, .001, 10, 10, finish_reason='stop')


def controller(path, config, *, model=None, metric=None):
    model = Model() if model is None else model
    log = RunLog(str(path / 'research.jsonl'))
    log.run_start(scope='inert external-history-v2 fixture')
    charges = []
    c = ResearchController(config, model, log, charges.append,
                           metric=circle_metric() if metric is None else metric)
    return NS(c=c, worker=CircleGeneration(c, execution_profile=builder.EXECUTION_PROFILE),
              parent=parent(), model=model, log=log, charges=charges, operations=[], groups=[])


def comparison(record, index):
    control = digest(builder.source_for())
    radii = (.02, .02, .02) if index == 0 else (.02, .005, .02)
    pairs = [dict(seed=seed, comparator=receipt(control, seed, .01),
                  challenger=receipt(record['source_sha256'], seed, radius))
             for seed, radius in zip((71, 72, 73), radii)]
    return build_comparison(namespace='history-v2-inert', batch_id=f'b{index + 1}', branch='p1',
        origin_experiment_id=record['id'], comparator_source_sha256=control,
        challenger_source_sha256=record['source_sha256'], adapter_identity=ADAPTER,
        limits=LIMITS, pairs=pairs, provenance={'fixture': 'saved arrays only'},
        task_metric=record['metric'])


def make_source(path, *, rules=RULES):
    task = task_copy(path / 'task', initial=builder.source_for())
    f = controller(path, research_config(), metric=metric_context(task))
    for index, score in enumerate((.6, .4)):
        outcome = CascadeOutcome(EvalResult(True, score, cases(score), timing=.01), score, 'full', False)
        result = f.worker.run(snapshot(f), f.model, lambda source: outcome, base_temp=.2)
        assert result.ok
        record = f.c.knowledge.experiments[-1]
        f.operations.append(('ordinary', record['id']))
        f.c.complete_step(reflect=False)
        f.operations.append(('conclusion', record['id']))
        f.operations.append(('step_end', None))
        group = comparison(record, index)
        f.c.import_comparison(group)
        f.groups.append(group)
        f.operations.append(('comparison', group['id']))
        lesson = dict(rule=rules[index], polarity='positive', scope='branch',
                      evidence_ids=['r000001a1'])
        f.model.reflection = dict(lessons=[lesson], retire_ids=[] if index == 0 else ['l000001'],
                                 comparison_reviews=[review_proposal(group)])
        f.c.complete_step()
        f.operations.append(('reflection', index + 1))
        assert f.c.reflections == index + 1
    f.state = f.c.state()
    f.journal = Path(f.log.path).read_text()
    f.log.close()
    return f


def capsule_for(f):
    from sciel.engine import external_research_v2 as history2
    events = [json.loads(line) for line in f.journal.splitlines()]
    operations = []
    for kind, ident in f.operations:
        if kind == 'ordinary':
            operations.append(dict(kind='generation', journal=0, experiment_id=ident))
        elif kind == 'step_end':
            operations.append(dict(kind='step_end'))
        elif kind == 'comparison':
            operations.append(dict(kind=kind, comparison_id=ident))
        else:
            field = 'experiment_id' if kind == 'conclusion' else 'through'
            event, = [row for row in events if row.get('phase') == kind and row.get(field) == ident]
            operation = dict(kind=kind, journal=0, event_seq=event['event_seq'])
            if kind == 'reflection':
                operation['interval'] = 1
            operations.append(operation)
    return dict(schema=history2.SCHEMA, profile=builder.EXECUTION_PROFILE, metric=f.c.metric,
        root_parent=dict(id='p1', source_sha256=digest(builder.source_for())),
        policy=history2.policy_identity(), journals=[dict(manifest_sha256=canonical_sha(
            {'fixture': 'inert native/controller history'}), journal=f.journal)],
        comparison_catalog={group['id']: copy.deepcopy(group) for group in f.groups},
        operations=operations, terminal=dict(knowledge_sha256=canonical_sha(f.state['knowledge']),
            conclusions_sha256=canonical_sha(f.state['conclusions']),
            comparison_reviews_sha256=canonical_sha(f.state['comparison_reviews'])))


def import_fixture(path, capsule, *, generation_start=2, auxiliary=False):
    cfg = research_config()
    cfg[external.CONFIG_KEY] = canonical_sha(capsule)
    cfg.update(max_aux_calls=2 if auxiliary else 0, reflection_interval=1 if auxiliary else 0)
    return controller(path, cfg, model=Model(generation_start=generation_start), metric=capsule['metric'])


def import_capsule(f, capsule):
    return f.worker.import_external_history(capsule,
        expected_sha256=canonical_sha(capsule), parent=f.parent)


def rewrite_journal(capsule, mutate):
    changed = copy.deepcopy(capsule)
    events = [json.loads(line) for line in changed['journals'][0]['journal'].splitlines()]
    mutate(events)
    changed['journals'][0]['journal'] = ''.join(json.dumps(row) + '\n' for row in events)
    return changed


def test_source_fixture_retires_exposed_lesson_without_new_credit(tmp_path):
    f = make_source(tmp_path / 'source')
    lesson, = f.c.knowledge.lessons
    assert lesson['id'] == 'l000002' and lesson['rule'] == RULES[1]
    assert lesson['exposure_ids'] == []
    assert lesson['refute_ids'] == ['r000002a1'] and lesson['confidence'] == 1 / 3
    old, = lesson['inherited_exposures']
    assert old['from_lesson_id'] == 'l000001' and old['from_rule'] == RULES[0]
    assert old['experiment']['injected_ids'] == ['l000001']
    assert len(f.state['comparison_reviews']) == 2
    assert [call['role'] for call in f.model.calls] == ['G', 'C', 'R', 'G', 'C', 'R']
    second = [call for call in f.model.calls if call['role'] == 'G'][1]
    assert RULES[0] in second['prompt']


def test_reducer_reconstructs_authored_retirement_exposure_and_review(tmp_path):
    from sciel.engine import external_research_v2 as history2
    f = make_source(tmp_path / 'source')
    capsule = capsule_for(f)
    original = copy.deepcopy(capsule)
    restored = history2.validate(capsule, config=f.c.config, metric=f.c.metric)
    assert restored['knowledge'].state() == f.state['knowledge']
    assert restored['conclusions'] == f.state['conclusions']
    assert restored['comparison_reviews'] == f.state['comparison_reviews']
    assert restored['imported_ids'] == ['r000001a1', 'r000002a1'] and restored['floor'] == 2
    assert set(restored['generation']) == set(restored['imported_ids'])
    assert capsule == original


def test_large_consolidated_lesson_is_explicitly_omitted_without_exposure_credit(tmp_path):
    large = 'Long retained interpretation. ' * 20
    source = make_source(tmp_path / 'source', rules=(RULES[0], large))
    capsule = capsule_for(source)
    f = import_fixture(tmp_path / 'fresh', capsule)
    import_capsule(f, capsule)
    context = snapshot(f).research_context
    assert context['injected_ids'] == []
    omitted, = context['lesson_presentation']['omitted']
    assert omitted['id'] == 'l000002' and omitted['reason'] == 'card_char_limit'
    assert omitted['chars'] > context['lesson_presentation']['max_card_chars'] == 16000
    before = f.c.knowledge.lessons
    outcome = CascadeOutcome(EvalResult(True, .55, cases(.55), timing=.01), .55, 'full', False)
    f.worker.run(snapshot(f), f.model, lambda code: outcome, base_temp=.2)
    assert f.c.knowledge.lessons == before
    assert large not in f.model.calls[0]['prompt']
    f.log.close()


def test_public_import_and_next_generation_preserve_old_credit_once(tmp_path):
    source = make_source(tmp_path / 'source')
    capsule = capsule_for(source)
    f = import_fixture(tmp_path / 'fresh', capsule)
    assert import_capsule(f, capsule) is True
    assert f.c.knowledge.state() == source.state['knowledge']
    assert f.c.state()['comparison_reviews'] == source.state['comparison_reviews']
    assert all(getattr(f.c, key) == 0 for key in (
        'attempts', 'outcomes', 'valid_hypotheses', 'aux_calls', 'aux_cost',
        'reflections', 'reflection_changes', 'reflected_through'))
    assert f.model.calls == f.charges == []
    assert [row.get('phase') for row in rows(f.log.path)] == [None, external.EVENT]
    assert import_capsule(f, capsule) is False and len(rows(f.log.path)) == 2

    context = snapshot(f).research_context
    assert context['injected_ids'] == ['l000002']
    assert context['comparison_reviews']['presented_count'] > 0
    before = copy.deepcopy(f.c.knowledge.lessons[0])
    outcome = CascadeOutcome(EvalResult(True, .55, cases(.55), timing=.01), .55, 'full', False)
    f.worker.run(snapshot(f), f.model, lambda code: outcome, base_temp=.2)
    f.c.complete_step()
    lesson, = f.c.knowledge.lessons
    assert lesson['exposure_ids'] == ['r000003a1']
    assert lesson['inherited_exposures'] == before['inherited_exposures']
    assert lesson['support_ids'] == ['r000003a1'] and lesson['refute_ids'] == ['r000002a1']
    assert lesson['confidence'] == .5
    assert f.c.attempts == f.c.outcomes == f.worker.evaluations == 1
    assert f.c.aux_calls == f.c.aux_cost == f.c.reflections == f.c.reflected_through == 0
    assert len(f.model.calls) == 1 and f.worker.known_cost == .001
    assert f.charges == []  # Engine accounts for G; this callback owns auxiliary calls only.
    assert RULES[1] in f.model.calls[0]['prompt']
    f.worker.assert_journal(f.log.path, f.log.anchor())
    restored = import_fixture(tmp_path / 'restored', capsule)
    restored.c.load_state(f.c.state())
    restored.worker.load_state(f.worker.state())
    restored.worker.assert_journal(f.log.path, f.log.anchor())
    assert restored.c.state() == f.c.state()
    assert restored.model.calls == restored.charges == []
    f.log.close()
    restored.log.close()


@pytest.mark.parametrize('damage', ['writer_reply', 'writer_index', 'context', 'exposure',
                                   'retirement', 'lost_counterexample', 'truncated_writer'])
def test_nonempty_history_tampering_is_rejected_atomically(tmp_path, damage):
    source = make_source(tmp_path / 'source')
    capsule = capsule_for(source)
    if damage in ('writer_index', 'lost_counterexample'):
        changed = copy.deepcopy(capsule)
        state = copy.deepcopy(source.state['knowledge'])
        lesson, = state['lessons']
        if damage == 'writer_index':
            authoring = lesson['authoring']
            old, = authoring['writer_refs']
            node = authoring['catalog'].pop(old)
            node['proposal_index'] = 1
            new = canonical_sha(node)
            authoring['catalog'][new] = node
            authoring['writer_refs'] = [new]
        else:
            lesson['inherited_exposures'] = []
            lesson['refute_ids'] = []
            lesson['confidence'] = .5
        changed['terminal']['knowledge_sha256'] = canonical_sha(state)
    else:
        def mutate(events):
            if damage == 'context':
                event = next(e for e in events if e.get('phase') == 'reflection_context' and e.get('through') == 2)
                event['lesson_ids'] = []
            elif damage == 'exposure':
                request = next(e for e in events if e.get('action') == 'request_start' and e.get('experiment_id') == 'r000002a1')
                request['prompt'] = request['prompt'].replace(RULES[0], 'A fabricated earlier rule.')
                request['prompt_sha256'] = digest(request['prompt'])
                attempt = next(e for e in events if e.get('phase') == 'attempt' and e.get('experiment_id') == 'r000002a1')
                attempt['prompt_sha256'] = request['prompt_sha256']
            else:
                event = next(e for e in events if e.get('phase') == 'aux_result'
                             and e.get('kind') == 'reflection' and e.get('reference') == 2)
                if damage == 'truncated_writer':
                    event['response_truncated'] = True
                    return
                parsed = json.loads(event['response'])
                if damage == 'writer_reply':
                    parsed['lessons'][0]['rule'] = 'A different fabricated writer.'
                else:
                    parsed['retire_ids'] = ['l999999']
                event['response'] = json.dumps(parsed)
                event['response_sha256'] = digest(event['response'])
        changed = rewrite_journal(capsule, mutate)
    target = import_fixture(tmp_path / 'target', changed)
    before = target.c.state()
    with pytest.raises(ValueError):
        import_capsule(target, changed)
    assert target.c.state() == before
    assert target.model.calls == target.charges == []
    assert len(rows(target.log.path)) == 1
    target.log.close()


def local_retirement(path):
    source = make_source(path / 'source')
    capsule = capsule_for(source)
    f = import_fixture(path / 'fresh', capsule, auxiliary=True)
    import_capsule(f, capsule)
    outcome = CascadeOutcome(EvalResult(True, .55, cases(.55), timing=.01), .55, 'full', False)
    f.worker.run(snapshot(f), f.model, lambda code: outcome, base_temp=.2)
    f.c.complete_step(reflect=False)
    proposed = review_proposal(source.groups[-1])
    proposed['interpretation'] = 'A local reflection retains the saved mixed result without a causal claim.'
    f.model.reflection = dict(lessons=[dict(rule='Retain both observed advice outcomes.',
        polarity='positive', scope='branch', evidence_ids=['r000001a1'])],
        retire_ids=['l000002'], comparison_reviews=[proposed])
    f.c.complete_step()
    return f, source, capsule


def test_local_reflection_replaces_foreign_review_and_retires_foreign_lesson(tmp_path):
    f, source, capsule = local_retirement(tmp_path)
    lesson, = f.c.knowledge.lessons
    assert lesson['id'] == 'l000003' and lesson['exposure_ids'] == []
    assert lesson['support_ids'] == ['r000003a1'] and lesson['refute_ids'] == ['r000002a1']
    assert {item['from_lesson_id'] for item in lesson['inherited_exposures']} == {'l000001', 'l000002'}
    assert lesson['confidence'] == .5
    assert f.c.reflections == f.c.reflected_through == f.c.outcomes == 1
    assert f.c.aux_calls == 2 and len(f.charges) == 2
    card = f.c.state()['comparison_reviews'][source.groups[-1]['id']]
    assert card['writer']['reference'] == 1
    assert card != source.state['comparison_reviews'][source.groups[-1]['id']]
    f.worker.assert_journal(f.log.path, f.log.anchor())
    restored = import_fixture(tmp_path / 'restored', capsule, auxiliary=True)
    restored.c.load_state(f.c.state())
    restored.worker.load_state(f.worker.state())
    restored.worker.assert_journal(f.log.path, f.log.anchor())
    assert restored.c.state() == f.c.state()
    assert restored.model.calls == restored.charges == []
    f.log.close()
    restored.log.close()


@pytest.mark.parametrize('damage', ['writer', 'writer_reanchored', 'lost_counterexample', 'review_prose'])
def test_checkpoint_cannot_invent_local_writer_or_drop_counterexample(tmp_path, damage):
    f, source, capsule = local_retirement(tmp_path)
    state = f.c.state()
    lesson, = state['knowledge']['lessons']
    if damage.startswith('writer'):
        data = lesson['authoring']
        old, = data['writer_refs']
        writer = data['catalog'].pop(old)
        writer['raw_response_sha256'] = '0' * 64
        new = canonical_sha(writer)
        data['catalog'][new] = writer
        data['writer_refs'] = [new]
    elif damage == 'lost_counterexample':
        lesson['inherited_exposures'] = [item for item in lesson['inherited_exposures']
                                       if item['experiment']['id'] != 'r000002a1']
        lesson['refute_ids'] = []
        lesson['confidence'] = 2 / 3
    else:
        state['comparison_reviews'][source.groups[-1]['id']]['unverified_interpretation']['interpretation'] = 'Fabricated causal success.'
    restored = import_fixture(tmp_path / 'restored', capsule, auxiliary=True)
    # The altered state is internally consistent: the separately anchored
    # journal must reject it instead of accepting rehashed local structures.
    restored.c.load_state(state)
    restored.worker.load_state(f.worker.state())
    journal_path, anchor = f.log.path, f.log.anchor()
    if damage == 'writer_reanchored':
        # A fabricated state and recomputed outer anchor must still agree with
        # the actual raw writer response. Rehashing a host transition alone is
        # insufficient to invent a new model author.
        events = rows(journal_path)
        event = next(row for row in reversed(events) if 'external_history_transition' in row)
        event['external_history_transition']['lessons_sha256'] = canonical_sha(state['knowledge']['lessons'])
        raw = ''.join(json.dumps(row) + '\n' for row in events).encode()
        journal_path = tmp_path / 'reanchored_writer.jsonl'
        journal_path.write_bytes(raw)
        anchor = {**anchor, 'byte_offset': len(raw), 'prefix_sha256': hashlib.sha256(raw).hexdigest()}
    with pytest.raises(ValueError, match='durable transition|admitted response|admission'):
        restored.worker.assert_journal(journal_path, anchor)
    assert restored.model.calls == restored.charges == []
    f.log.close()
    restored.log.close()
