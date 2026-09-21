"""Small authored measurement histories for knowledge delivery regressions."""
import hashlib

from sciel.hypothesis.measurement import metric_context
from sciel.knowledge_base import KnowledgeBase


def record(kb, number, delta=.1, injected=(), *, parent='p1', measured_context=False,
           hypothesis=None):
    ident = f'r{number:06d}a1'
    source = hashlib.sha256(ident.encode()).hexdigest()
    score = .5 + delta if delta is not None else None
    value = dict(id=ident, branch='p1', parent_id=parent, parent_score=.5,
                 score=score, actual_delta=score-.5 if score is not None else None,
                 status='evaluated' if score is not None else 'eval_failed',
                 injected_ids=list(injected), source_sha256=source if score is not None else None,
                 proposed_sha256=source if score is not None else None, hypothesis=hypothesis)
    if measured_context:
        value.update(metric=metric_context(), parent_cases=None, evaluation=None)
    kb.record(value)
    return kb.experiments[-1]


def positive_exposure_history():
    kb = KnowledgeBase()
    source = record(kb, 1)
    lid, = kb.add_lessons([dict(rule='The authored source improved the measured score.',
        polarity='positive', scope='branch', evidence_ids=[source['id']])],
        allowed_evidence_ids=[source['id']])
    for number, delta in [(75, -.1), (80, None), (86, -.2), (89, None),
                          (90, -.1), (92, None), (99, -.3)]:
        record(kb, number, delta, [lid])
    return kb


def negative_source_history():
    kb = KnowledgeBase()
    ids = [record(kb, number, -.1, parent=parent)['id']
           for number, parent in [(1, 'p49'), (2, 'p37')]]
    kb.add_lessons([dict(rule='The model claims mixed results; inspect the measurements.',
        polarity='negative', scope='branch', evidence_ids=ids)], allowed_evidence_ids=ids)
    return kb


def overlapping_reflection_history():
    kb = KnowledgeBase()
    records = [record(kb, number, delta) for number, delta in [(2, .1), (4, .2), (5, -.1), (6, None)]]
    rule = 'Keep measured source support separate from later advice exposure.'
    lesson = dict(id='l1', rule=rule, scope='branch', branch='p1', polarity='positive',
        evidence_ids=[records[0]['id'], records[1]['id']], observed_exposures=dict(
            positive=1, negative=1, zero=0, unmeasured=1, total=3,
            shown_ids=[r['id'] for r in records[1:]], omitted_count=0,
            representatives=[dict(experiment_id=r['id'], attribution=[dict(
                kind='direct', from_lesson_id='l1', from_rule=rule)]) for r in records[1:]]))
    return dict(experiments=records, lessons=[lesson])


def outside_forecast_history():
    kb = KnowledgeBase()
    hypothesis = dict(text='The authored mutation makes a small score change.',
        predicted_low=-.001, predicted_high=.001,
        reasoning='A bounded forecast before the declared measurement.', confidence=.5)
    record(kb, 1, -.002, measured_context=True, hypothesis=hypothesis)
    observed = record(kb, 2, .001855718577133958, measured_context=True, hypothesis=hypothesis)
    return observed, kb.state()
