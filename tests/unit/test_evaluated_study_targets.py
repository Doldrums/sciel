"""No model hashes: exact target compilation from authenticated HOST evidence."""
import copy
import pytest
from sciel.hypothesis import evaluated_study_targets as targets
from test_evaluated_study_review import bundle, no_execution, proposal_for


@pytest.mark.parametrize('field',['schema','study_sha256','author_authority_sha256','policy_sha256',
    'comparison_id','comparison_sha256','receiving_root','quality_gate','motivating_evidence','extra'])
def test_bound_target_rejects_rewritten_or_extra_host_metadata(bundle,field):
    target=targets.compile_target(proposal_for(bundle)['next_hypothesis'],bundle=bundle)
    bad=copy.deepcopy(target)
    if field=='receiving_root':bad[field]['id']='different-root'
    elif field=='motivating_evidence':bad[field][0]['value']='CONFIRMED'
    else:bad[field]='a'*64
    with pytest.raises(ValueError):targets.validate_target(bad,bundle=bundle)


def test_same_base_and_comparator_supported_without_changing_old_pair(bundle):
    proposal=proposal_for(bundle)['next_hypothesis'];proposal['base_source_sha256']=proposal['comparator_source_sha256']
    target=targets.compile_target(proposal,bundle=bundle)
    assert target['base_source_sha256']==target['comparator_source_sha256']
    assert targets.validate_target(target,bundle=bundle)==target


@pytest.mark.parametrize('refs',[[],['quality:verdict']*2,['a b'],[True],['x'*257],['quality:verdict']*5])
def test_reference_shape_is_bounded_and_unique(refs):
    with pytest.raises(ValueError):targets.validate_references(refs)
