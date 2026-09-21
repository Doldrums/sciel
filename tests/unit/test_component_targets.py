"""Pure component-target contract: saved archived G data, never new execution."""
import copy
from types import SimpleNamespace as NS

import pytest

from sciel.hypothesis import component_targets as targets
from sciel.hypothesis import comparison_targets as legacy
from sciel.hypothesis import generated_paired_review as reviews
from sciel.knowledge_base.catalogs import canonical_json, canonical_sha
from test_generated_paired_reflection import no_runtime


def shape_target():
    return dict(schema=targets.SCHEMA, component_sha256='a'*64,
        receiving_root={'id':'root', 'source_sha256':'b'*64}, policy_sha256='c'*64,
        motivating_evidence=[{'comparison_id':'comparison:example', 'fact_id':'quality:strict_verdict'}],
        base_source_sha256='d'*64, comparator_source_sha256='e'*64,
        change='Change one bounded insertion decision; outcome remains unmeasured.',
        quality_gate=legacy.QUALITY_GATE, diagnostics=[])


@pytest.fixture(scope='module')
def actual(no_runtime):
    from test_generated_paired_integration import pilot_artifacts, component
    value = component.__wrapped__(pilot_artifacts.__wrapped__())
    catalogs = reviews._catalogs(value)
    groups = {g['branch']:g for g in value['comparisons']}
    reviewed = groups['on_off']['id']
    target = dict(schema=targets.SCHEMA, component_sha256=canonical_sha(value),
        receiving_root=copy.deepcopy(value['root_parent']), policy_sha256=canonical_sha(value['policy']),
        motivating_evidence=[dict(comparison_id=reviewed, fact_id='quality:strict_verdict')],
        base_source_sha256=groups['off_control']['challenger_source_sha256'],
        comparator_source_sha256=groups['off_control']['comparator_source_sha256'],
        change='A synthetic bounded change to OFF, tested prospectively against control.',
        quality_gate=legacy.QUALITY_GATE, diagnostics=[])
    return NS(component=value,catalogs=catalogs,groups=groups,reviewed=reviewed,target=target)


def test_shape_validation_detaches_without_claiming_evidence_authority():
    value=shape_target()
    result=targets.validate_target_shape(value)
    assert result==value and result is not value
    result['receiving_root']['id']='elsewhere'
    result['motivating_evidence'][0]['fact_id']='unknown'
    assert value==shape_target()
    assert 'verdict' not in result and 'generation_allowed' not in result


@pytest.mark.parametrize('key,value',[
    ('schema',legacy.SCHEMA), ('schema',True), ('component_sha256','A'*64),
    ('policy_sha256',False), ('receiving_root',{'id':True,'source_sha256':'b'*64}),
    ('receiving_root',{'id':'root','source_sha256':'b'*64,'extra':None}),
    ('base_source_sha256',123), ('comparator_source_sha256','bad'),
    ('change',''), ('change','x'*1025), ('quality_gate','mean_gain'),
    ('diagnostics',[{'observable':'total_cost','direction':'decrease'}]),
    ('diagnostics',[{'observable':'candidate_elapsed_seconds','direction':'decrease'}]*2),
    ('motivating_evidence',[]), ('motivating_evidence',[{'comparison_id':'x','fact_id':'y'}]*5),
    ('motivating_evidence',[{'comparison_id':'x','fact_id':'y','value':'true'}]),
])
def test_shape_rejects_malformed_or_relaxed_contract(key,value):
    target=shape_target();target[key]=value
    with pytest.raises(ValueError): targets.validate_target_shape(target)


def test_missing_extra_and_legacy_targets_do_not_silently_upgrade():
    for mutation in ('missing','extra','legacy'):
        target=shape_target()
        if mutation=='missing': target.pop('receiving_root')
        elif mutation=='extra': target['permission_to_run']=True
        else: target={k:v for k,v in target.items() if k in ('schema','base_source_sha256','comparator_source_sha256','change','quality_gate','diagnostics')}
        with pytest.raises(ValueError): targets.validate_target_shape(target)
    with pytest.raises(ValueError): legacy.validate_target(shape_target())


@pytest.mark.parametrize('refs',[
    None, (), [], [{'comparison_id':'x','fact_id':'y'}]*2,
    [{'comparison_id':False,'fact_id':'y'}], [{'comparison_id':'x','fact_id':' '}],
    [{'comparison_id':'x','fact_id':'z'*257}], [{'comparison_id':'x'*129,'fact_id':'y'}],
    [{'comparison_id':'x','fact_id':'y','comparison_sha256':'a'*64}],
])
def test_reference_shape_rejects_missing_duplicate_or_unbounded_refs(refs):
    with pytest.raises(ValueError): targets.validate_references_shape(refs)


def test_same_fact_id_in_different_groups_is_not_a_duplicate(actual):
    refs=[dict(comparison_id=actual.groups[name]['id'],fact_id='quality:strict_verdict') for name in ('on_off','on_control')]
    result=targets._resolve_references(refs,actual.catalogs,actual.reviewed)
    assert [row['value'] for row in result]==['INCONCLUSIVE','REFUTED']
    assert [row['comparison_sha256'] for row in result]==[canonical_sha(actual.groups[n]) for n in ('on_off','on_control')]


def test_actual_off_to_control_is_valid_under_new_component_scope(actual):
    before=canonical_json(actual.component)
    result=targets.validate_target(actual.target,component=actual.component,comparison_id=actual.reviewed)
    pair_sources={actual.groups['on_off'][k] for k in ('comparator_source_sha256','challenger_source_sha256')}
    assert result['base_source_sha256'] in pair_sources
    assert result['comparator_source_sha256'] not in pair_sources
    assert result==actual.target and result is not actual.target
    assert canonical_json(actual.component)==before
    assert 'quality' not in result and 'candidate_source_sha256' not in result
    # The historical selected-pair target still rejects this contrast.
    old={k:copy.deepcopy(result[k]) for k in ('base_source_sha256','comparator_source_sha256','change','quality_gate','diagnostics')}
    old['schema']=legacy.SCHEMA
    with pytest.raises(ValueError): legacy.validate_target(old,sources=pair_sources)


def test_base_and_comparator_may_coincide_without_fabricating_candidate(actual):
    target=copy.deepcopy(actual.target)
    target['comparator_source_sha256']=target['base_source_sha256']
    assert targets._validate_target(target,actual.component,actual.catalogs,actual.reviewed)==target


def test_complete_authenticated_parent_is_available_without_its_own_measured_pair(actual):
    component=copy.deepcopy(actual.component)
    component['comparisons']=[component['comparisons'][0]]  # Both actual Gs still have their ON/OFF pair.
    target=copy.deepcopy(actual.target)
    target['component_sha256']=canonical_sha(component)
    assert targets.validate_target(target,component=component,comparison_id=actual.reviewed)==target
    wire=targets.json_schema(component)
    assert target['comparator_source_sha256'] in wire['properties']['comparator_source_sha256']['enum']


@pytest.mark.parametrize('mutation',['component','policy','root','base','comparator','unanchored','unknownfact','metadata','unknown_group','missing_review'])
def test_actual_target_bindings_and_per_group_reference_membership(actual,mutation):
    target=copy.deepcopy(actual.target);reviewed=actual.reviewed
    if mutation=='component': target['component_sha256']='f'*64
    elif mutation=='policy': target['policy_sha256']='f'*64
    elif mutation=='root': target['receiving_root']['id']='another-root'
    elif mutation in ('base','comparator'): target[mutation+'_source_sha256']='f'*64
    elif mutation=='unanchored': target['motivating_evidence'][0]['comparison_id']=actual.groups['on_control']['id']
    elif mutation=='unknownfact': target['motivating_evidence'][0]['fact_id']='restart:invented:observed'
    elif mutation=='metadata': target['motivating_evidence'][0]['fact_id']='comparison_id'
    elif mutation=='unknown_group': target['motivating_evidence'][0]['comparison_id']='comparison:unknown'
    else: reviewed=None
    with pytest.raises(ValueError): targets._validate_target(target,actual.component,actual.catalogs,reviewed)


def test_foreign_fact_in_union_is_not_fact_of_reviewed_group(actual):
    own=actual.catalogs[actual.reviewed]
    foreign=actual.catalogs[actual.groups['on_control']['id']]
    fact=next(k for k in own['facts'] if k not in foreign['facts'])
    refs=[dict(comparison_id=actual.groups['on_control']['id'],fact_id=fact)]
    with pytest.raises(ValueError): targets._resolve_references(refs,actual.catalogs)


def test_public_reference_resolution_preserves_unknown_not_false(actual):
    refs=[dict(comparison_id=actual.reviewed,fact_id='restart:challenger:763606949:result_changed:observed')]
    rows=targets.resolve_references(refs,component=actual.component,comparison_id=actual.reviewed)
    assert rows[0]['value']=='unknown'
    assert targets.validate_references(refs,component=actual.component,comparison_id=actual.reviewed)==refs
    rows[0]['value']='false'
    assert actual.catalogs[actual.reviewed]['facts'][refs[0]['fact_id']]=='unknown'


@pytest.mark.parametrize('mutation',['source','pairfacts','unsupported_host'])
def test_public_validation_authenticates_whole_component_not_only_self_digest(actual,mutation):
    component=copy.deepcopy(actual.component)
    if mutation=='source': component['proposals'][0]['source']+='\n# unauthenticated source'
    elif mutation=='pairfacts': component['comparisons'][0]['derived']['outcome']='improved'
    else: component['schema']='sciel-host-only-paired-study/1'
    target=copy.deepcopy(actual.target);target['component_sha256']=canonical_sha(component)
    with pytest.raises(ValueError): targets.validate_target(target,component=component,comparison_id=actual.reviewed)


def test_dynamic_wire_contains_exact_component_bindings_and_union_only(actual):
    wire=targets.json_schema(actual.component);properties=wire['properties']
    assert properties['component_sha256']['enum']==[canonical_sha(actual.component)]
    assert properties['policy_sha256']['enum']==[canonical_sha(actual.component['policy'])]
    assert properties['receiving_root']['properties']['id']['enum']==[actual.component['root_parent']['id']]
    assert actual.target['comparator_source_sha256'] in properties['comparator_source_sha256']['enum']
    ids=properties['motivating_evidence']['items']['properties']['fact_id']['enum']
    assert 'comparison_id' not in ids and 'quality:strict_verdict' in ids
    assert set(wire['required'])==set(actual.target)
    assert targets.json_schema()['properties']['schema']['enum']==[targets.SCHEMA]
    assert 'execution authority' in targets.identity()['scope']


def test_references_remain_data_without_extra_native_measurements(actual):
    before=canonical_json(actual.component)
    for name,group in actual.groups.items():
        rows=targets._resolve_references([dict(comparison_id=group['id'],fact_id='quality:strict_verdict')],actual.catalogs,group['id'])
        assert set(rows[0])=={'comparison_id','comparison_sha256','fact_id','value'}
    assert canonical_json(actual.component)==before
