"""Explicit synthetic R/G on saved sources, then synthetic paired receipts."""
from test_evaluated_detached_study import generated, host, prepared, saved, component, pilot_artifacts
from test_evaluated_study_review import no_execution
from sciel.hypothesis.evaluated_study_review import source_catalog


def test_model_authorship_forecast_agreement_is_separate_from_refuted_quality(generated):
    bundle={key:generated[key] for key in ('capsule','author_dependencies','author_authority')}
    catalog=source_catalog(bundle);facts=catalog['facts']
    assert facts['author:kind']['value']=='component_target_G'
    assert facts['quality:verdict']['value']=='REFUTED'
    assert facts['forecast:paired:status']['value']=='DECLARED'
    assert facts['forecast:paired:expected']['value'] is False
    assert facts['forecast:paired:observed']['value'] is False
    assert facts['forecast:paired:verdict']['value']=='CONFIRMED'
    assert facts['native:forecast_status']['value']=='NOT_DECLARED'
    assert facts['restart:challenger:17:forecast_status']['value']=='DECLARED'
    assert facts['restart:comparator:17:forecast_status']['value']=='NOT_DECLARED'
    assert any(key.startswith('restart:challenger:17:') and key.endswith(':expected') for key in facts)
    assert not any(key.startswith('restart:comparator:17:') and key.endswith(':expected') for key in facts)
