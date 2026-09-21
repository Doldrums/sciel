"""Real Engine, archived history and inert model/evaluators across a restart."""
import copy
import json
from pathlib import Path
import sys

from sciel.engine import checkpoint
from sciel.engine.loop import Engine
from sciel.engine.llm.client import LLMResponse
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from sciel.hypothesis import circle_optimizer_traced_body as builder
from sciel.hypothesis.rendering import decode_evidence
from sciel.knowledge_base.catalogs import canonical_sha
from test_external_research import capsule
from test_circle_intervention_engine import task_copy
from test_optimizer_traced_engine import OutcomeDraws
from test_restart_prediction_engine import config, spec


def test_imported_history_survives_real_engine_checkpoint_and_next_segment(capsule, tmp_path, monkeypatch):
    draws = OutcomeDraws(monkeypatch)
    task = task_copy(tmp_path/'task', initial=builder.source_for())
    cfg = config()
    cfg['research'] = copy.deepcopy(capsule['conclusion']['state']['config'])
    cfg['research'].update(external_history_sha256=canonical_sha(capsule),
                           max_aux_calls=0, reflection_interval=0)
    folder = tmp_path/'run'
    old = capsule['source']['state']['knowledge']['experiments'][0]
    bodies = ['# fixture_step = 1\nreturn best_params + uniform_noise * 0.25\n',
              '# fixture_step = 2\nreturn best_params + uniform_noise * 0.5\n']
    evaluated = []

    def evaluate(_task, path, **kwargs):
        source = Path(path).read_text(); code = builder.body_of(source)
        index = 0 if code == builder.DEFAULT_BODY else bodies.index(code) + 1
        score = old['parent_score'] + index * .0001
        evaluated.append(source)
        cases = dict(sum_radii=score * 2.635983, target_ratio=score, validity=1., combined_score=score)
        return CascadeOutcome(EvalResult(True, score, cases, timing=.01), score, 'full', False)
    monkeypatch.setattr('sciel.engine.loop.cascade_evaluate', evaluate)

    class Model:
        models = ['fixture']; weights = [1.]; max_retries = 1
        def __init__(self, index): self.index = index; self.calls = []
        def choose_model(self, rng): return self.models[0]
        def generate(self, prompt, model, temperature=None, *, response_format=None):
            assert response_format['json_schema']['schema']['properties'].get('hypothesis')
            raw, _ = json.JSONDecoder().raw_decode(prompt.split('RESEARCH_CONTEXT_JSON:', 1)[1].lstrip())
            context = decode_evidence(raw)
            self.calls.append(context)
            value = dict(hypothesis=dict(hypothesis='Change the synthetic restart input.',
                predicted_low=-.1, predicted_high=.1, reasoning='Inert fixture only.', confidence=.5,
                restart_predictions=spec()), body=bodies[self.index])
            return LLMResponse(json.dumps(value), model, .001, 10, 10, finish_reason='stop')

    first_model = Model(0)
    first = Engine(cfg, task, str(folder), first_model, python=sys.executable)
    original_seed = first.seed
    def seed_and_import():
        original_seed()
        parent = first.db.get('p1')
        first.circle_generation.import_external_history(capsule,
            expected_sha256=canonical_sha(capsule), parent=parent)
        assert first.research.attempts == first.research.outcomes == first.research.aux_calls == 0
    first.seed = seed_and_import
    first._stop_at = 1
    first.run()
    assert len(first_model.calls) == 1 and len(draws.calls) == 9
    feedback = first_model.calls[0]['conclusion_feedback']
    assert feedback['presented_count'] == 1 and feedback['omitted_count'] == 0
    card, = feedback['cards']
    assert card['experiment_id'] == old['id']
    assert card['unverified_interpretation']['mechanism_review'] == capsule['reflection']['state']['conclusions'][old['id']]['llm']['mechanism_review']
    assert [r['id'] for r in first.research.knowledge.experiments] == ['r000001a1', 'r000002a1']
    assert first.research.outcomes == first.circle_generation.evaluations == 1
    assert first.research.aux_calls == 0 and first.research.aux_cost == 0
    assert first.research.conclusions['r000001a1'] == capsule['reflection']['state']['conclusions']['r000001a1']

    second_model = Model(1)
    second = Engine(cfg, task, str(folder), second_model, python=sys.executable)
    checkpoint.load(second, str(folder/'checkpoint.json'))
    second.run()
    assert len(second_model.calls) == 1 and len(draws.calls) == 15
    assert second.research.attempts == second.research.outcomes == second.circle_generation.evaluations == 2
    assert second.research.knowledge.experiments[-1]['id'] == 'r000003a1'
    assert evaluated == [builder.source_for(), *(builder.source_for(b) for b in bodies)]
    saved = json.loads((folder/'checkpoint.json').read_text())
    third = Engine(cfg, task, str(folder), Model(0), python=sys.executable)
    checkpoint.load(third, str(folder/'checkpoint.json'))
    third.circle_generation.assert_journal(third.log.path, saved['provenance']['journal_anchor'])
    assert third.research.state() == second.research.state()
    third.repeat_promotion.close(); third.log.close()
