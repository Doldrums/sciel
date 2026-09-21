"""Same-schema policy/source drift cannot resume an incompatible forecast run."""
import json
from pathlib import Path

import pytest

from sciel.hypothesis import restart_predictions as predictions
from sciel.hypothesis.measurement import metric_context
from sciel.hypothesis.rendering import presentation_identity
from sciel.knowledge_base import restart_prediction_context as contexts
from sciel.knowledge_base.catalogs import canonical_json
from test_restart_prediction_delivery import controller_fixture
import test_real_restart_memory_delivery as existing


@pytest.mark.parametrize('change', ['policy', 'evaluator_source', 'context_source', 'research_source'])
def test_same_schema_prediction_drift_refuses_restore_without_mutation_or_aux(change, monkeypatch):
    controller, parent, db, _, _, _ = controller_fixture(metric_context())
    existing._history(controller, parent, db)
    saved = json.loads(canonical_json(controller.state()))
    original = canonical_json(saved)
    restored, _, _, sink, events, charges = controller_fixture(metric_context())
    if change == 'policy':
        monkeypatch.setattr(predictions, 'POLICY', {**predictions.POLICY, 'scope': 'A different scope'})
    else:
        target = (Path(predictions.__file__).resolve().parents[1] / 'engine/research.py'
                  if change == 'research_source' else
                  Path((predictions if change == 'evaluator_source' else contexts).__file__).resolve())
        read = Path.read_bytes

        def altered(path):
            value = read(path)
            return value + b'\n# Inert source drift; never written.\n' if path.resolve() == target else value

        monkeypatch.setattr(Path, 'read_bytes', altered)
    current = presentation_identity(restart_predictions=True)['restart_predictions']
    assert current['schema'] == saved['presentation']['restart_predictions']['schema']
    assert current != saved['presentation']['restart_predictions']
    before = canonical_json(restored.state())
    with pytest.raises(ValueError, match='presentation mismatch'):
        restored.load_state(saved)
    assert canonical_json(restored.state()) == before and canonical_json(saved) == original
    assert sink.prompts == sink.conclusion_prompts == events == charges == []
