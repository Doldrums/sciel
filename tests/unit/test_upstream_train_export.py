"""Real new native/tools/oracle/export path; SDK, tool and numeric leaves synthetic."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import socket
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sciel.adaptation.dataset import build_dataset, verify_dataset
from sciel.adaptation.ancestry import NativeAncestry
from sciel.adaptation.auxiliary_ancestry import donor_auxiliary_ids
from sciel.adaptation.evaluation_evidence import retained_context
from sciel.adaptation.export import export_native
from sciel.engine.factory import build_engine
from sciel.engine import checkpoint
from sciel.engine.llm.local import LocalOpenAILLM
from sciel.eval import _runner, cascade, oracle_evidence as O
from sciel.eval.facade import EvalResult
from sciel.research_tools.controller import PROMPT_MARKER
from test_research_tool_controller import Executor, IMAGE
from test_upstream_local_port import local_config

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('upstream_oracle_trial', ROOT/'tests/fixtures/upstream_oracle_trial.py')
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def native_tools_train(tmp_path, monkeypatch, *, full_research=False):
    def no_socket(*args, **kwargs):
        raise AssertionError('synthetic TRAIN fixture forbids network connections')
    monkeypatch.setattr(socket.socket, 'connect', no_socket)
    monkeypatch.setattr(socket.socket, 'connect_ex', no_socket)
    E = O.task_module()
    entropy = iter(bytes([i]) * 32 for i in range(1, 5))
    monkeypatch.setattr(E.panels.secrets, 'token_bytes', lambda n: next(entropy))
    study = tmp_path/'synthetic-only-study'
    created = E.panels.create_study(study)
    panel = created['manifest']['panels']['train-0']
    oracle = dict(schema='sciel-oracle-launch/1', study_root=str(study),
        manifest_sha256=created['manifest_sha256'], panel='train-0', panel_sha256=panel['sha256'],
        image_id=IMAGE)
    binding = dict(image_id=IMAGE, assets_sha256=E.runtime.runtime_assets_sha256(), base_image=E.runtime.BASE_IMAGE)
    monkeypatch.setattr(E.runtime, 'verify_runtime_image', lambda image: binding)
    sources, model_calls, stages = {}, [], []

    def trial(source, instance, image, *, algorithm_seed, container_name, journal_path):
        position = panel['algorithm_seeds'].index(algorithm_seed)
        value, raw = fixture.synthetic_trial(E, position, source=source, name=container_name,
            selected=panel, objective_value=1.0/(sources.get(source.decode(), 0) + 1))
        Path(journal_path).write_bytes(raw)
        Path(journal_path).chmod(0o600)
        for key in ('panel_position', 'instance_index', 'instance_sha256'):
            value.pop(key)
        return value

    monkeypatch.setattr(E.runtime, 'run_container_trial', trial)
    monkeypatch.setattr(_runner, '_load', lambda _: E)
    monkeypatch.setattr(_runner, '_apply_limits', lambda *a, **kw: None)

    def stage(evaluator, name, source, *, oracle_environment, **kwargs):
        stages.append((name, deepcopy(oracle_environment)))
        output = Path(oracle_environment['SCIEL_ORACLE_FIXED_INVOCATION']).parent/'fixture-output.json'
        with monkeypatch.context() as env:
            for key, value in oracle_environment.items():
                env.setenv(key, value)
            args = ['synthetic-in-process', evaluator, name, source, str(output)]
            if name == 'confirm':
                args.append(str(kwargs['n']))
            _runner.main(args)
        result = EvalResult.from_json(json.loads(output.read_text()))
        output.unlink()
        return result

    monkeypatch.setattr(cascade, 'run_stage', stage)
    backend = Executor()
    monkeypatch.setattr('sciel.research_tools.library.DockerExecutor', lambda _: backend)
    sdk = Mock()

    def generate(**options):
        model_calls.append(options)
        prompt = options['messages'][0]['content']
        if prompt.startswith(PROMPT_MARKER):
            text = json.dumps(dict(action='create', name='Public fixture check', description='Synthetic test only',
                code="def run(payload):\n    return dict(schema='sciel-tool-result/1', status='success', observations={}, diagnostics={}, error=None)\n",
                parents=[], dependencies=[], arguments_json='{}', replay_policy='exact_observations'))
        elif prompt.startswith('Research protocol') and '\nEVIDENCE_JSON:\n' in prompt:
            from sciel.hypothesis.rendering import decode_evidence
            from _mechanism_fixtures import unknown_mechanism
            data = decode_evidence(json.loads(prompt.split('EVIDENCE_JSON:\n')[1].split('\n')[0]))
            if data.get('schema') == 'sciel-reflection-presentation/9':
                lessons = [] if any(l['unverified_interpretation']=='Synthetic retained observation' for l in data['lessons']) else [dict(rule='Synthetic retained observation',
                    polarity='positive', scope='global', evidence_ids=list(data['observation_catalog'])[:2])]
                text = json.dumps(dict(lessons=lessons, retire_ids=[]))
            else:
                text = json.dumps(dict(verdict=data['forecast_verdict'],
                    mechanism_review=unknown_mechanism(data), explanation='Synthetic observation only.',
                    lessons=[dict(rule='Synthetic branch observation', polarity='positive',
                        scope='branch', evidence_ids=[data['id']])]))
        else:
            value = len(sources) + 1
            source = f'# public fixture {value}\ndef search_algorithm(oracle):\n    oracle.query({value}.0, 0.0)\ndef run_search(oracle):\n    search_algorithm(oracle)\n'
            sources[source] = value
            hypothesis = dict(hypothesis='Synthetic controlled improvement', predicted_low=0., predicted_high=1.,
                reasoning='Fixture only', confidence=.5)
            text = 'HYPOTHESIS_JSON: ' + json.dumps(hypothesis) + '\n```python\n' + source + '```'
        return SimpleNamespace(model=options['model'], choices=[SimpleNamespace(finish_reason='stop',
            message=SimpleNamespace(content=text, refusal=None))], usage=SimpleNamespace(prompt_tokens=50, completion_tokens=20))

    sdk.chat.completions.create.side_effect = generate
    monkeypatch.setattr('openai.OpenAI', Mock(return_value=sdk))
    monkeypatch.setattr('sciel.engine.factory.load_env', Mock(side_effect=AssertionError('no credentials')))
    # Deterministic synthetic tokenizer leaf: no model assets/download. The real
    # budget/projection code still binds this identity and complete prompt hash.
    monkeypatch.setattr(LocalOpenAILLM, 'count_prompt_tokens', lambda self, text: 8 + len(text.encode()) // 4)
    monkeypatch.setattr(LocalOpenAILLM, 'prompt_tokenizer_identity', property(lambda self: {'fixture':'utf8-quarter/1'}))
    llm = local_config()['llm']
    llm.update(max_tokens=1536)
    llm['local_inference']['learner_by_island'] = {'0':'learner-0', '1':'learner-1'}
    count = 4 if full_research else 3
    cfg = dict(task=O.TASK, oracle_evaluation=oracle, confirm_records=True,
        budget=None, max_iterations=count, schedule_horizon=20, random_seed=23, num_islands=2,
        inflight_cap=1, dup_cap=1, schedule_policy='round_robin', milestone_fracs=[],
        portfolio_seed=False, dynamic_islands=False, embed_grid=False, embed_niche=False,
        idea_island=False, crossover=False, kb_free=False, c2_event_bets=False,
        research=dict(enabled=True, max_aux_calls=16 if full_research else 0,
            reflection_interval=2, generation_context_tokens=16384,
            reflection_context_tokens=16384),
        research_tools=dict(enabled=True, image_id=IMAGE, max_scientific_steps=1, max_requests=1,
            max_tokens=3072, context_tokens=16384, proactive_every_attempts=100,
            attempt_stall_threshold=100, trigger_sources=['search_attempt_stall'], shared_observations=True,
            max_prompt_chars=16000, max_context_chars=4096, interval=100),
        op_fixed='full_rewrite', policy={'enabled':full_research, 'mode_bandit':full_research}, llm=llm,
        candidate_novelty=dict(enabled=True, max_resamples=1, neighbors=2),
        parallel_search=dict(enabled=True, workers=1, capacity_dir=str(tmp_path/'capacity'),
            host_llm_slots=1, provider_slots=1, host_evaluator_slots=1,
            max_inflight_per_island=1, max_inflight_per_parent=1))
    run = tmp_path/'run'
    engine = build_engine(cfg, str(ROOT/'tasks'), str(run))
    try:
        result = engine.run()
        assert result['iteration_no'] == count
    finally:
        engine.close()
    return run, model_calls, backend, stages


def test_real_full_tools_train_export_and_portable_verified_rows(tmp_path, monkeypatch):
    run, calls, backend, stages = native_tools_train(tmp_path, monkeypatch)
    before = (len(calls), len(backend.calls), len(stages))
    out = tmp_path/'export'
    report = export_native(run, out)
    assert report['eligible_count'] == 3, report
    episodes = rows(out/'episodes.jsonl')
    assert {e['owner'] for e in episodes} == {'learner-0', 'learner-1'}
    assert all(e['evaluation_receipt']['reached'] == 'confirm' for e in episodes)
    assert all(e['evaluation_receipt']['result']['score'] > e['parent_score'] for e in episodes)
    nodes = [node for e in episodes for node in e['provenance']['ancestry_resolution']['auxiliary_bindings']]
    assert {'llm_call','tool_version','tool_invocation','research_card'} <= {n['kind'] for n in nodes}
    assert any(donor_auxiliary_ids(e['provenance']['ancestry_resolution'], e['owner']) for e in episodes)
    context = retained_context(out/'export.json', report)
    data = tmp_path/'dataset'
    manifest = build_dataset(episodes, data, owner='learner-0', evaluation_context=context)
    assert verify_dataset(data/'train.jsonl', data/'manifest.json')[1] == manifest
    assert export_native(run, out) == report
    assert before == (len(calls), len(backend.calls), len(stages))


def test_full_research_conclusion_reflection_and_mode_bandit_export(tmp_path, monkeypatch):
    run, calls, backend, stages = native_tools_train(tmp_path, monkeypatch, full_research=True)
    report = export_native(run, tmp_path/'export')
    assert report['eligible_count'] == 4, report
    cp = json.loads((run/'checkpoint.json').read_bytes())
    assert cp['config']['policy']['mode_bandit'] is True
    assert sum(map(len, cp['controller']['mode_bandit']['events'].values())) == 4
    traces = rows(run/'artifacts/llm_calls.jsonl')
    assert {'G', 'C', 'R', 'research_tool'} <= {t['inference_spec']['role'] for t in traces if t['event']=='response'}
    nodes = [n for e in rows(tmp_path/'export/episodes.jsonl')
             for n in e['provenance']['ancestry_resolution']['auxiliary_bindings']]
    assert {'lesson_revision', 'research_card', 'tool_invocation'} <= {n['kind'] for n in nodes}
    assert {'C', 'R'} <= {n['identity']['role'] for n in nodes if n['kind']=='llm_call'}
    before = (len(calls), len(backend.calls), len(stages))
    reopened = build_engine(cp['config'], str(ROOT/'tasks'), str(run))
    try:
        checkpoint.load(reopened, str(run/'checkpoint.json'))
        assert reopened.iteration_no == 4
        assert reopened.research_tools.state() == cp['research_tools']
        assert reopened.research.state() == cp['research']
        assert reopened.controller.state() == cp['controller']
    finally:
        reopened.close()
    assert before == (len(calls), len(backend.calls), len(stages))


@pytest.mark.parametrize('tamper', ['tool_receipt', 'planner_prompt', 'confirm_sidecar'])
def test_export_rejects_changed_retained_producers_or_full_confirm_evidence(tmp_path, monkeypatch, tamper):
    run, calls, backend, stages = native_tools_train(tmp_path, monkeypatch)
    before = (len(calls), len(backend.calls), len(stages))
    events = rows(run/'sciel_events.jsonl')
    if tamper == 'tool_receipt':
        invocation = next(e for e in events if e.get('phase')=='invocation_result')
        path = run/'research_tools/artifacts'/f"{invocation['receipt_id']}.json"
        value = json.loads(path.read_bytes())
        value['payload']['receipt']['output']['observations']['unmeasured'] = True
        path.write_text(json.dumps(value))
    elif tamper == 'planner_prompt':
        trace = next(t for t in rows(run/'artifacts/llm_calls.jsonl')
                     if t['event']=='request' and t['inference_spec']['role']=='research_tool')
        path = run/'artifacts/llm_calls'/f"{trace['call_id']}.prompt.txt"
        path.write_text(path.read_text()+'\nInvented additional tool input.')
    else:
        # Each generated candidate retains a successful saved score, but its
        # changed confirmation evidence cannot fall back to that score alone.
        candidates = [e for e in events if e.get('event')=='iteration']
        for candidate in candidates:
            path = run/O.relative_evaluation(candidate['meta']['evaluation_id'])/'sidecar.json'
            value = json.loads(path.read_bytes())
            value['terminal_stage'] = 'full'
            path.write_text(json.dumps(value))
    report = export_native(run, tmp_path/'rejected-export')
    assert report['eligible_count'] == 0, report
    assert before == (len(calls), len(backend.calls), len(stages))
    reasons = ' '.join(e['reason'] for e in report['exclusions'])
    assert ('immutable invocation' if tamper=='tool_receipt' else
            'prompt' if tamper=='planner_prompt' else 'oracle sidecar commitment differs') in reasons


def test_current_card_text_and_future_invocation_cannot_authorize_delivery(tmp_path, monkeypatch):
    run, _, _, _ = native_tools_train(tmp_path, monkeypatch)
    events = rows(run/'sciel_events.jsonl')
    traces = rows(run/'artifacts/llm_calls.jsonl')
    reader = NativeAncestry(run, json.loads((run/'checkpoint.json').read_bytes()), events,
        {(t['call_id'], t['event']):t for t in traces})
    event = next(e for e in events if e.get('event')=='iteration')
    card = deepcopy(event['meta']['research_tools']['cards'][0])
    card['interpretation'] = 'Unmeasured task improvement'
    with pytest.raises(ValueError, match='exact receipt projection'):
        reader.tools.card(card, reader._generation(event)[3], frozenset())
    invocation = next(e for e in events if e.get('phase')=='invocation_result')
    with pytest.raises(ValueError, match='one committed binding'):
        reader.tools.invocation(invocation['receipt_id'], reader.tools.position(invocation)-1, frozenset())
