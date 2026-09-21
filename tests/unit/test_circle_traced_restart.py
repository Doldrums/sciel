"""Same-invocation tracing with fixed helpers/fake solver/clock, no search."""
import ast
import copy
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from sciel.hypothesis import circle_restart_body as original
from sciel.hypothesis import circle_traced_restart as traced
from sciel.hypothesis.restart_trace import validate_trace
from test_circle_proposals import Clock


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def namespace(body='return None\n'):
    source = traced.source_for(body)
    ns = {}
    exec(compile(source, '<inert-traced-scaffold>', 'exec'), ns)
    clock = Clock()
    ns['time'] = SimpleNamespace(monotonic=clock)
    ns['SCIEL_RESTART_TRACE_BINDING'] = dict(source_sha256=sha(source), seed=71,
        runtime_sha256='a'*64, invocation_id='b'*32)
    calls = []
    base = ns['RestartTrace']
    class Spy(base):
        def begin_helper(self, inputs):
            calls.append(('begin_helper', copy.deepcopy(inputs)))
            return super().begin_helper(inputs)
        def helper_end(self, **kwargs):
            calls.append(('helper_end', copy.deepcopy(kwargs)))
            return super().helper_end(**kwargs)
        def optimizer_enter(self, **kwargs):
            calls.append(('optimizer_enter', copy.deepcopy(kwargs)))
            return super().optimizer_enter(**kwargs)
        def optimizer_exit(self, **kwargs):
            calls.append(('optimizer_exit', copy.deepcopy(kwargs)))
            return super().optimizer_exit(**kwargs)
    ns['RestartTrace'] = Spy
    def forbidden(*args, **kwargs):
        pytest.fail('real optimizer is forbidden')
    ns['minimize'] = forbidden
    return ns, clock, calls


def trace(ns):
    value = ns['SCIEL_RESTART_TRACE']
    binding = ns['SCIEL_RESTART_TRACE_BINDING']
    assert validate_trace(value, source_sha256=binding['source_sha256'], seed=binding['seed'],
        runtime_sha256=binding['runtime_sha256'],
        scaffold_sha256=ns['SCIEL_RESTART_TRACE_SCAFFOLD_SHA256']) == value
    return value


def test_proof_reverses_all_instrumentation_and_common_scaffold_is_shared():
    built = traced.build('return ordinary_params.copy()\n')
    raw = built['code'].encode()
    for patch in reversed(built['proof']['patches']):
        start = patch['sequential_byte_start']
        after, before = patch['after'].encode(), patch['before'].encode()
        assert raw[start:start+len(after)] == after
        assert sha(patch['after']) == patch['after_sha256']
        assert sha(patch['before']) == patch['before_sha256']
        raw = raw[:start] + before + raw[start+len(after):]
    assert raw.decode() == original.source_for('return ordinary_params.copy()\n')
    assert sha(raw.decode()) == built['proof']['untraced_source_sha256']
    assert sha(built['code']) == built['proof']['traced_source_sha256']
    assert traced.build()['proof']['scaffold_sha256'] == built['proof']['scaffold_sha256']
    assert sha(original.source_for()) == '3089f4e36bf787275ade5c3c24d4836ad8dc26e409185c0679c3377f4ce9c5f9'


def test_equations_rng_optimizer_and_single_helper_call_are_unchanged():
    sources = [original.source_for(), traced.source_for()]
    trees = [ast.parse(s) for s in sources]
    for name in ('objective', 'objective_jac', 'packing_constraints', 'packing_jacobian',
                 'is_feasible', 'check_deadline', 'timed', 'callback'):
        definitions = [next(n for n in ast.walk(t) if isinstance(n, ast.FunctionDef) and n.name == name)
                       for t in trees]
        assert ast.dump(definitions[0]) == ast.dump(definitions[1])
    for name in ('minimize', 'np.random.uniform'):
        calls = [[n for n in ast.walk(t) if isinstance(n, ast.Call) and ast.unparse(n.func) == name] for t in trees]
        assert len(calls[0]) == len(calls[1]) == 1
        assert ast.dump(calls[0][0]) == ast.dump(calls[1][0])
    assert sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'propose_restart'
               for n in ast.walk(trees[1])) == 1


@pytest.mark.parametrize('body,origin', [('return None\n', 'saved_ordinary_fallback'),
    ('return ordinary_params.copy()\n', 'accepted_return')])
def test_identical_guesses_keep_distinct_origins_and_full_rng_sequence(body, origin):
    ns, _, calls = namespace(body)
    guesses = []
    def fake(*args, **kwargs):
        guesses.append(args[1].copy())
        return SimpleNamespace(x=ns['initial_parameters'](), success=True)
    ns['minimize'] = fake
    saved = np.random.get_state()
    try:
        np.random.seed(71)
        ns['solve_packing'](max_restarts=9)
        end_rng = np.random.get_state()
        expected_rng = np.random.RandomState(71)
        initial = ns['initial_parameters']()
        for guess in guesses:
            expected = initial + expected_rng.uniform(-.06, .06, 78)
            expected[52:] = np.maximum(expected[52:], .001)
            expected[:52] = np.clip(expected[:52], .001, .999)
            np.testing.assert_array_equal(guess, expected)
        assert all(np.array_equal(a, b) for a, b in zip(end_rng, expected_rng.get_state()))
    finally:
        np.random.set_state(saved)
    ends = [data for phase, data in calls if phase == 'helper_end']
    assert len(ends) == 3 and all(e['origin'] == origin for e in ends)
    assert len(guesses) == 9 and ns['LAST_STATS']['completed_starts'] == 9
    trace(ns)


def test_mutated_argument_none_retains_saved_ordinary_and_post_mutation_facts():
    ns, _, calls = namespace()
    count = 0
    def helper(*args):
        nonlocal count
        count += 1
        for value in args[:4]: value[:] = -999.
        return None
    ns['propose_restart'] = helper
    ns['minimize'] = lambda *a, **k: SimpleNamespace(x=ns['initial_parameters'](), success=True)
    ns['solve_packing'](max_restarts=1)
    before = next(data for phase, data in calls if phase == 'begin_helper')
    after = next(data for phase, data in calls if phase == 'helper_end')
    assert count == 1 and after['raw_kind'] == 'none'
    np.testing.assert_array_equal(after['effective'], before['ordinary'])
    assert all(np.all(value == -999.) for value in after['post_inputs'].values())
    assert after['origin'] == 'saved_ordinary_fallback'
    trace(ns)


@pytest.mark.parametrize('failure', ['normalization', 'helper'])
def test_failure_trace_does_not_fabricate_effective_or_optimizer_start(failure):
    ns, _, calls = namespace()
    def helper(*args):
        if failure == 'helper': raise RuntimeError('fixed fixture error')
        return np.zeros(77)
    ns['propose_restart'] = helper
    with pytest.raises((RuntimeError, ValueError)):
        ns['solve_packing'](max_restarts=1)
    end, = [data for phase, data in calls if phase == 'helper_end']
    assert end['error_stage'] == failure and end['effective'] is None and end['origin'] is None
    assert not any(phase.startswith('optimizer_') for phase, _ in calls)
    assert ns['LAST_STATS']['proposal_errors'] == 1
    trace(ns)


def test_helper_deadline_can_improve_without_optimizer_entry():
    ns, clock, calls = namespace()
    def late(best, initial, noise, ordinary, index):
        value = initial.copy(); value[52:77] = .02
        clock.now = .05
        return value
    ns['propose_restart'] = late
    _, _, total = ns['solve_packing'](budget_seconds=.02)
    assert total == pytest.approx(.5)
    assert [p for p, _ in calls] == ['begin_helper', 'helper_end']
    assert calls[-1][1]['incumbent_sum'] == pytest.approx(.25)
    assert ns['LAST_STATS']['started_starts'] == 0
    trace(ns)


@pytest.mark.parametrize('returned', [False, True])
def test_optimizer_deadline_distinguishes_partial_from_returned_start(returned):
    ns, clock, calls = namespace()
    def fake(objective, guess, **kwargs):
        value = ns['initial_parameters'](); value[52:77] = .02
        clock.now = .05
        if not returned: objective(value)
        return SimpleNamespace(x=value, success=True)
    ns['minimize'] = fake
    _, _, total = ns['solve_packing'](budget_seconds=.02)
    assert total == pytest.approx(.5)
    assert sum(p == 'optimizer_enter' for p, _ in calls) == 1
    assert sum(p == 'optimizer_exit' for p, _ in calls) == int(returned)
    assert ns['LAST_STATS']['completed_starts'] == int(returned)
    trace(ns)


def test_deadline_before_helper_has_no_fabricated_invocation_or_optimizer():
    ns, _, calls = namespace()
    ns['solve_packing'](budget_seconds=0.)
    assert calls == [] and ns['LAST_STATS']['proposal_attempts'] == 0
    trace(ns)


def test_missing_binding_fails_before_any_execution():
    ns, _, calls = namespace()
    ns['SCIEL_RESTART_TRACE_BINDING'] = None
    with pytest.raises(ValueError, match='binding'):
        ns['solve_packing']()
    assert calls == [] and ns['SCIEL_RESTART_TRACE'] is None


def test_argument_copy_failure_preserves_attempt_error_and_no_helper_call():
    ns, _, calls = namespace()
    identity = {**ns['SCIEL_RESTART_TRACE_BINDING'],
        'scaffold_sha256': ns['SCIEL_RESTART_TRACE_SCAFFOLD_SHA256']}
    recorder = ns['_TraceRecorder'](identity)
    recorder.begin_start(0, incumbent_sum=.25, improvements=0)
    class BadCopy:
        def copy(self):
            raise MemoryError('fixed copy failure')
    stats = ns['_proposal_stats']()
    with pytest.raises(MemoryError, match='fixed copy failure'):
        ns['_proposal_invoke'](BadCopy(), np.zeros(78), np.zeros(78), np.zeros(78),
                               0, stats, recorder, (.25, 0))
    assert stats['proposal_attempts'] == stats['proposal_errors'] == 1
    assert calls == []
    value = recorder.finish('error', incumbent_sum=.25, improvements=0)
    assert value['total_scheduled_calls'] == 0
    assert validate_trace(value) == value


@pytest.mark.parametrize('method', ['begin_helper', 'helper_end', 'finish'])
@pytest.mark.parametrize('helper_fails', [False, True])
def test_recorder_fault_does_not_replace_helper_result_or_exception(method, helper_fails):
    ns, _, calls = namespace()
    def fail(*args, **kwargs):
        raise ValueError('inert recorder defect')
    setattr(ns['RestartTrace'], method, fail)
    count = 0
    def helper(*args):
        nonlocal count
        count += 1
        if helper_fails: raise RuntimeError('original helper failure')
        return None
    ns['propose_restart'] = helper
    ns['minimize'] = lambda *a, **k: SimpleNamespace(x=ns['initial_parameters'](), success=True)
    if helper_fails:
        with pytest.raises(RuntimeError, match='original helper failure'):
            ns['solve_packing'](max_restarts=1)
    else:
        assert ns['solve_packing'](max_restarts=1)[2] == pytest.approx(.25)
    assert count == 1
    assert ns['SCIEL_RESTART_TRACE']['schema'] == 'sciel-restart-trace-unavailable/1'
    assert ns['LAST_STATS']['proposal_errors'] == int(helper_fails)


def test_returned_parameters_are_captured_before_consider_with_one_attribute_read():
    import base64
    ns, _, _ = namespace()
    value = ns['initial_parameters']()
    value[52:77] = .02
    seen = []
    class Returned:
        @property
        def success(self):
            seen.append('success')
            return True
        @property
        def x(self):
            seen.append('x')
            return value
    ns['minimize'] = lambda *a, **k: Returned()
    ns['solve_packing'](max_restarts=1)
    actual = trace(ns)
    call, = actual['calls']
    vector = actual['vectors'][call['optimizer_result']['ref']]
    assert seen == ['success', 'x']
    assert base64.b64decode(vector['data_base64']) == value.tobytes()
    assert call['phases']['optimizer_returned']['sum'] == pytest.approx(.25)
    assert call['phases']['terminal']['sum'] == pytest.approx(.5)
    value[:] = -999.
    assert base64.b64decode(vector['data_base64']) != value.tobytes()


@pytest.mark.parametrize('failing_field', ['success', 'x'])
def test_returned_boundary_survives_result_attribute_error(failing_field):
    ns, _, _ = namespace()
    seen = []
    class Returned:
        @property
        def success(self):
            seen.append('success')
            if failing_field == 'success':
                raise RuntimeError('original success error')
            return True
        @property
        def x(self):
            seen.append('x')
            raise RuntimeError('original x error')
    ns['minimize'] = lambda *a, **k: Returned()
    with pytest.raises(RuntimeError, match='original ' + failing_field):
        ns['solve_packing'](max_restarts=1)
    actual = trace(ns)
    call, = actual['calls']
    assert ns['LAST_STATS']['completed_starts'] == 1
    assert call['phases']['optimizer_returned'] is not None
    assert call['optimizer_result'] == {'ref': None, 'reason': 'array_unavailable'}
    assert call['terminal'] == 'error'
    assert seen == (['success'] if failing_field == 'success' else ['success', 'x'])


def test_return_capture_error_does_not_replace_consider_or_numerical_result():
    ns, _, _ = namespace()
    def fail(*args, **kwargs):
        raise ValueError('capture defect')
    ns['RestartTrace'].optimizer_result = fail
    value = ns['initial_parameters'](); value[52:77] = .02
    ns['minimize'] = lambda *a, **k: SimpleNamespace(x=value, success=True)
    assert ns['solve_packing'](max_restarts=1)[2] == pytest.approx(.5)
    assert ns['SCIEL_RESTART_TRACE']['schema'] == 'sciel-restart-trace-unavailable/1'
