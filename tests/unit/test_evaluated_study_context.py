"""Whole-study prompt delivery from authored negative evidence; no draw/model."""
import copy

import pytest

from sciel.knowledge_base import evaluated_study_context as context
from sciel.knowledge_base.catalogs import canonical_json, canonical_sha
import evaluated_study_fixture as adapter


@pytest.fixture(autouse=True)
def no_execution(monkeypatch):
    def forbidden(*args,**kwargs):
        pytest.fail('Context building attempted provider or numerical execution.')
    for name in ('subprocess.Popen','sciel.eval.repeat_draw.draw','sciel.eval.repeat_draw.adapter_identity',
                 'sciel.campaign.budget.BudgetedLLM.generate'):
        monkeypatch.setattr(name,forbidden)


@pytest.fixture(scope='module')
def imported():
    return adapter.build()


def arguments(imported,**extra):
    return dict(author_dependencies=imported['author_dependencies'],author_authority=imported['author_authority'],
                branch=imported['capsule']['receiving_root']['id'],**extra)


def test_authored_study_negative_quality_and_matched_control_are_delivered(imported):
    value=context.build_context(imported['capsule'],**arguments(imported))
    assert value is not None and value['presented_count']==1
    assert len(context.block(value))<=16000
    card,=value['cards']
    assert card['author']['kind']=='HOST'
    assert card['strict_quality']['verdict']=='REFUTED'
    assert card['native']['observation_status']=='UNMEASURED'
    assert card['native']['forecast_status']=='NOT_DECLARED'
    assert card['paired_forecast'] is None
    assert all(row['forecast'] is None for row in card['restart_forecasts'])
    obs=card['optimizer_observations']
    rows=[dict(zip(obs['matched_columns'],r)) for r in obs['matched_inputs']]
    row=next(r for r in rows if r['seed']==3084817015 and r['restart_index']==8)
    assert row['comparator_gain']==adapter.COMPARATOR_GAIN
    assert row['challenger_gain']==adapter.CHALLENGER_GAIN
    assert card['coverage']['unissued']==2
    assert context.validate_context(value,imported['capsule'],**arguments(imported))==value


@pytest.mark.parametrize('extra',[{'enabled':False},{'max_items':0},{'max_chars':0},{'max_chars':1},{'branch':'foreign-root'}])
def test_disabled_empty_or_foreign_delivery_has_no_facts_or_prose(imported,extra):
    kwargs=arguments(imported);kwargs.update(extra)
    assert context.build_context(imported['capsule'],**kwargs) is None
    assert context.block(None)==''


def test_budget_omits_whole_study_instead_of_dropping_loss_or_control(imported):
    value=context.build_context(imported['capsule'],**arguments(imported,max_chars=2400))
    assert value is None or (value['cards']==[] and value['presented_count']==0 and value['omitted_count']==1)
    if value is not None:
        assert len(context.block(value))<=2400


def test_rehashed_projection_cannot_hide_counterexample_or_rename_author(imported):
    original=context.build_context(imported['capsule'],**arguments(imported))
    for mutation in ('quality','author','matched'):
        value=copy.deepcopy(original);card=value['cards'][0]
        if mutation=='quality':card['strict_quality']['verdict']='CONFIRMED'
        elif mutation=='author':card['author']['kind']='component_target_G'
        else:card['optimizer_observations']['matched_inputs']=[]
        value['context_sha256']=canonical_sha({k:v for k,v in value.items() if k!='context_sha256'})
        with pytest.raises(ValueError,match='projection differs'):
            context.validate_context(value,imported['capsule'],**arguments(imported))


def test_complete_optional_body_appendices_never_clip_or_mutate_sources(imported):
    before=canonical_json(imported)
    value=context.build_context(imported['capsule'],**arguments(imported,max_chars=32000))
    card,=value['cards']
    from sciel.hypothesis import circle_optimizer_traced_body as builder
    for digest,item in card['source_bodies'].items():
        assert item['body']==builder.body_of(imported['capsule']['source_registry'][digest])
    assert set(card['source_bodies']) | {row['source_sha256'] for row in card['body_omissions']} == set(imported['capsule']['source_registry'])
    assert canonical_json(imported)==before
