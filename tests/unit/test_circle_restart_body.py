"""Typed-body source/transport tests; synthetic arrays and fake optimizer only."""
import ast
import copy
from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sciel.hypothesis import circle_restart_body as body
from sciel.hypothesis import circle_proposals as old
from sciel.hypothesis.parsing import parse_hypothesis, parse_hypothesis_value


HYP = {'hypothesis':'Change the center neighborhood', 'predicted_low':-.01,
       'predicted_high':.02, 'reasoning':'A testable new initialization may help or lose.', 'confidence':.4}


def wire(code='return best_params.copy()', hypothesis=None):
    return json.dumps({'hypothesis':HYP if hypothesis is None else hypothesis, 'body':code})


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def test_strict_nested_response_retains_exact_body_and_is_immutable():
    code='x = best_params.copy()\nx[:52] += uniform_noise[:52] * 0.5\nreturn x'
    value=body.parse_response(wire(code))
    assert value.body==code and value.hypothesis==parse_hypothesis_value(HYP)
    assert body.parse_hypothesis_response(wire(code))==value.hypothesis
    assert body.parse_response_hypothesis(wire(code))==value.hypothesis
    with pytest.raises(FrozenInstanceError): value.body='return None'
    with pytest.raises(FrozenInstanceError): value.hypothesis.text='changed'


def test_invalid_body_preserves_valid_forecast_and_both_errors_are_independent():
    result=body.decode_response(wire('return incumbent._params'))
    assert result['proposal'] is None and result['hypothesis']==parse_hypothesis_value(HYP)
    assert result['error']=='invalid_body' and result['body_error']
    bad={**HYP,'predicted_low':True}
    result=body.decode_response(wire('return time.monotonic()',bad))
    assert result['hypothesis'] is result['proposal'] is None
    assert result['error']=='invalid_hypothesis' and result['body_error']
    result=body.decode_response(wire('return ordinary_params',bad))
    assert result['error']=='invalid_hypothesis' and result['body_error'] is None


@pytest.mark.parametrize('code,line,message,excerpt', [
    ('    new_params = np.copy(best_params)\n    return new_params\n', 1,
     'unexpected indent', '    new_params = np.copy(best_params)'),
    ('x = ordinary_params.copy()\nreturn (\n', 2, 'was never closed', 'return ('),
    ('x = ordinary_params.copy()\nbreak\nreturn x\n', 2, 'outside loop', 'break'),
])
def test_rejected_syntax_reports_submitted_line_without_repair_or_quality(code,line,message,excerpt):
    result=body.decode_response(wire(code))
    assert result['proposal'] is None and result['hypothesis']==parse_hypothesis_value(HYP)
    assert result['error']=='invalid_body'
    assert message in result['body_error'] and f'body line {line}, column' in result['body_error']
    assert repr(excerpt) in result['body_error'] and 'untrusted' in result['body_error']
    with pytest.raises(ValueError,match='invalid body syntax'):
        body.source_for(code)
    assert json.loads(wire(code))['body']==code


def test_body_indentation_contract_rejects_global_indent_but_preserves_nested_suites():
    code='x = ordinary_params.copy()\nif restart_index > 0:\n    x[:52] += uniform_noise[:52] / 2\nreturn x\n'
    assert body.body_of(body.source_for(code))==code
    indented=''.join('    '+line for line in code.splitlines(keepends=True))
    assert body.parse_response(wire(indented)) is None
    for text in (body.interface_description()['body_indentation'],body.response_instruction(),
                 body.response_format()['json_schema']['schema']['properties']['body']['description']):
        assert 'column 1' in text and 'compiler adds' in text


def test_syntax_diagnostic_bounds_model_text_and_preserves_unicode_line_position():
    code='# α\n    '+ 'x'*2000+' = 1\nreturn None\n'
    result=body.decode_response(wire(code))
    assert 'body line 2' in result['body_error'] and len(result['body_error'])<450
    assert 'x'*161 not in result['body_error']


@pytest.mark.parametrize('code,column', [('é = 1; break\nreturn None\n',8),
                                        ('x = "α"; break\nreturn None\n',10)])
def test_contextual_syntax_error_column_counts_source_characters_not_utf8_bytes(code,column):
    result=body.decode_response(wire(code))
    assert f'body line 1, column {column}' in result['body_error']


@pytest.mark.parametrize('raw', [None, b'{}', '', 'null', '[]', '{}',
    '```json\n'+wire()+'\n```', 'HYPOTHESIS_JSON:'+wire(), 'before '+wire(), wire()+' after',
    wire()+wire(), '{"hypothesis":null,"hypothesis":null,"body":"return None"}',
    '{"hypothesis":{"reasoning":"one","reasoning":"two"},"body":"return None"}',
    json.dumps({'hypothesis':HYP,'body':'return None','extra':1}), json.dumps({'hypothesis':HYP}),
    '{"hypothesis":NaN,"body":"return None"}', '{"hypothesis":Infinity,"body":"return None"}',
])
def test_envelope_has_no_fallback_or_duplicate_unknown_field_tolerance(raw):
    result=body.decode_response(raw)
    assert result=={'proposal':None,'hypothesis':None,'error':'invalid_json_envelope','body_error':None}


@pytest.mark.parametrize('field,value', [('hypothesis',''),('hypothesis','   '),('hypothesis',False),
    ('reasoning',None),('confidence',True),('confidence',-.1),('confidence',1.1),
    ('predicted_low',True),('predicted_low',float('inf')),('predicted_low',float('nan')),
    ('predicted_low',.5),('predicted_high',-.5)])
def test_forecast_truth_table_reuses_same_legacy_value_rules(field,value):
    hypothesis={**HYP,field:value}
    assert parse_hypothesis_value(hypothesis) is None
    assert body.parse_response(wire(hypothesis=hypothesis)) is None


def test_hypothesis_values_accept_both_signs_and_do_not_invent_body_measurements():
    for low,high in [(-2,-1),(0,0),(-1,1),(1,2)]:
        h={**HYP,'predicted_low':low,'predicted_high':high}
        parsed=body.parse_response(wire(hypothesis=h))
        assert parsed.hypothesis.predicted_low==float(low)
        assert parsed.hypothesis.predicted_high==float(high)
    assert parse_hypothesis_value({**HYP,'delta':0}) is None
    assert body.parse_hypothesis_response(wire(None))==parse_hypothesis_value(HYP)


def test_response_schema_detached_exact_nested_fields_and_bounds():
    f=body.response_format(); schema=f['json_schema']['schema']
    assert f['type']=='json_schema' and f['json_schema']['strict'] is True
    assert set(schema['required'])==set(schema['properties'])=={'hypothesis','body'}
    assert schema['additionalProperties'] is False
    h=schema['properties']['hypothesis']
    assert h['additionalProperties'] is False and set(h['required'])==set(h['properties'])==set(HYP)
    assert h['properties']['confidence']['minimum']==0 and h['properties']['confidence']['maximum']==1
    assert schema['properties']['body']['maxLength']==body.MAX_BODY_BYTES
    assert 'pattern' not in schema['properties']['body']
    schema['properties'].clear()
    assert set(body.response_format()['json_schema']['schema']['properties'])=={'hypothesis','body'}


@pytest.mark.parametrize('code', [body.DEFAULT_BODY,'return best_params.copy()',
    'x = ordinary_params.copy()\r\nx[:52] += uniform_noise[:52]\r\nreturn x\r\n',
    '# Preserve α and two  spaces.\nx = best_params.copy()\nreturn x\n',
    'x = "a\u2028b"\nreturn ordinary_params',
    'x = best_params.copy()\nfor i in range(26):\n    x[2*i] += abs(uniform_noise[2*i])\nreturn x',
    'x = np.array([v for v in ordinary_params])\nreturn x + sum(v for v in uniform_noise) / 78',
])
def test_exact_body_roundtrip_final_lf_crlf_unicode_and_algorithms(code):
    source=body.source_for(code)
    canonical=body.body_of(source)
    assert canonical==(code if code.endswith('\n') else code+'\n')
    assert body.source_for(canonical)==source
    assert body.validate_body(code)==code


@pytest.mark.parametrize('code', [None, True, '', 'return None\r',
    'def propose_restart():\n    return None', 'import numpy as np\nreturn None',
    'return incumbent', 'return initial', 'return noise', 'return default_guess',
    'incumbent = best_params\nreturn incumbent', 'return best_params._params',
    'return _relocation_vector(best_params)', 'return time.monotonic()',
    'return np.random.uniform(size=78)', 'return np.linalg.solve(best_params, uniform_noise)',
    'global LAST_STATS\nreturn None', 'return globals()', 'return getattr(best_params,"shape")',
    'try:\n    return best_params\nexcept Exception:\n    return None',
    'break\nreturn None', 'return [v for np in best_params]',
    '#x\n'*512+'return None', '#'+('x'*8191), 'return None\n#\ud800',
])
def test_body_rejects_old_object_namespace_external_helpers_and_escapes(code):
    with pytest.raises(ValueError): body.validate_body(code)


def test_boundary_byte_limit_still_has_canonical_source_inverse():
    prefix='return None\n#'
    code=prefix+'x'*(body.MAX_BODY_BYTES-len(prefix)-1)
    assert len(code.encode())==body.MAX_BODY_BYTES-1
    assert len(body.body_of(body.source_for(code)).encode())==body.MAX_BODY_BYTES
    with pytest.raises(ValueError,match='structural final LF'):
        body.validate_body(code+'x')


def test_every_directed_synthetic_body_edit_rederives_exactly_and_reconstructs_source():
    bodies=[body.DEFAULT_BODY,'return best_params.copy()',
            'x = ordinary_params.copy()\nx[:52] += uniform_noise[:52] / 2\nreturn x\n']
    for parent in bodies:
        for target in bodies:
            if parent==target: continue
            source=body.source_for(parent); result=body.apply(source,target); p=result['proof']
            assert p['schema']=='source-bound-edit/1' and p['operation']==body.OPERATION==body.CONTRACT
            assert p['parent_body']==body.body_of(source) and p['target_body']==target
            assert p['target_body_sha256']==sha(target.encode())
            assert p['assembly']['append_final_lf']==(not target.endswith('\n'))
            assert p['parent_source_sha256']==sha(source.encode())
            assert p['child_source_sha256']==sha(result['code'].encode())
            a,b=p['before']['byte_span']; c,d=p['after']['byte_span']
            raw,child=source.encode(),result['code'].encode()
            assert raw[a:b]==p['before']['text'].encode() and child[c:d]==p['after']['text'].encode()
            assert raw[:a]==child[:c] and raw[b:]==child[d:]
            assert child[:c]+raw[a:b]+child[d:]==raw
            assert sha(raw[:a])==p['unchanged_prefix']['sha256'] and sha(raw[b:])==p['unchanged_suffix']['sha256']
            assert body.apply(body.source_for(p['parent_body']),p['target_body'])==result


def test_final_lf_alias_does_not_bypass_noop_and_has_same_directed_plan():
    with pytest.raises(ValueError,match='unchanged'):
        body.apply(body.source_for(),'return None')
    a=body.apply(body.source_for(),'return best_params')
    b=body.apply(body.source_for(),'return best_params\n')
    assert a['code']==b['code'] and a['proof']['plan_key']==b['proof']['plan_key']
    assert a['proof']['target_body_sha256']!=b['proof']['target_body_sha256']
    with pytest.raises(ValueError,match='unchanged'):
        body.apply(a['code'],'return best_params\n')


def test_parent_family_and_dependency_pin_are_strict(monkeypatch):
    parent=body.source_for()
    for changed in [parent+'#modified\n',parent.replace('ftol": 1e-7','ftol": 1e-8'),
                    parent.replace('    best_params = incumbent','    best_params = initial')]:
        with pytest.raises(ValueError): body.body_of(changed)
    monkeypatch.setattr(body,'PINNED_PROPOSALS_SHA256','0'*64)
    with pytest.raises(ValueError,match='pinned dependency'): body.source_for()


def test_kernel_rng_and_options_identical_to_pinned_scaffold():
    sources=[old.source_for(),body.source_for(),body.source_for('return best_params + uniform_noise')]
    trees=[ast.parse(x) for x in sources]
    for name in ('solve_packing','objective','packing_constraints','packing_jacobian','check_deadline','callback','run_packing','_proposal_invoke'):
        functions=[next(n for n in ast.walk(t) if isinstance(n,ast.FunctionDef) and n.name==name) for t in trees]
        assert len({ast.dump(n) for n in functions})==1
    assert sha(sources[0].encode())==body.PINNED_SCAFFOLD_SHA256
    assert sources[1]!=sources[0]


def test_writer_contract_contains_no_kernel_or_old_object_access():
    info=body.interface_description(); text=body.response_instruction()
    assert set(info['arrays'])=={'best_params','initial_params','uniform_noise','ordinary_params'}
    assert all(v['shape']==[78] and v['dtype']=='float64' for v in info['arrays'].values())
    for forbidden in ('class Incumbent','incumbent._params','def solve_packing','def propose_restart','_relocation_vector','LAST_STATS'):
        assert forbidden not in text
    assert 'not capped at1' in text and 'overlaps/out-of-bounds' in text
    assert 'signature' in text and 'not a function definition' in text
    info['arrays'].clear()
    assert len(body.interface_description()['arrays'])==4


def test_module_bindings_are_concrete_and_match_the_actual_inert_body_validator():
    info=body.interface_description(); code=info['code_contract']
    assert code['module_bindings']=={'np':'numpy','math':'math'}
    assert code['unbound_module_names']==['numpy']
    assert 'np.maximum' in code['numpy_calls'] and 'math.sqrt' in code['math_calls']
    assert all(name.startswith('np.') for name in code['numpy_calls']+code['numpy_values'])
    assert all(name.startswith('math.') for name in code['math_calls']+code['math_values'])
    assert 'abs' in code['builtins'] and all('.' not in name for name in code['builtins'])
    # New synthetic source only: no historical model response is rewritten or executed.
    valid='x = ordinary_params.copy()\nx[52:] = np.maximum(x[52:], 0.0)\nx[0] += math.sqrt(abs(uniform_noise[0]))\nreturn x'
    assert body.validate_body(valid)==valid
    with pytest.raises(ValueError):body.validate_body('return numpy.maximum(ordinary_params, 0.0)')
    instruction=body.response_instruction()
    assert 'np.maximum' in instruction and 'not numpy' in instruction
    code['module_bindings'].clear()
    assert body.interface_description()['code_contract']['module_bindings']=={'np':'numpy','math':'math'}


def test_future_contract_distinguishes_schedule_and_ordinary_none_fallback():
    info=body.interface_description()
    assert info['restart_index']['invoked_only_when']=='restart_index % 4 == 0'
    assert 'always true inside' in info['restart_index']['call_schedule']
    assert 'other three restarts skip' in info['restart_index']['call_schedule']
    assert all(array['nullable'] is False for array in info['arrays'].values())
    assert 'never None' in info['meanings']['best_params']
    assert 'not the unperturbed incumbent' in info['return']
    ns=namespace(body.DEFAULT_BODY)
    inputs=[np.full(78,0.1),np.full(78,0.2),np.full(78,0.01),np.full(78,0.11)]
    result=ns['_proposal_invoke'](*inputs,4,ns['_proposal_stats']())
    np.testing.assert_array_equal(result,inputs[-1])
    assert not np.array_equal(result,inputs[0])


def test_default_transformation_matches_pinned_kernel_ast_without_executing_it():
    source=body.source_for(); tree=ast.parse(source)
    solve=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='solve_packing')
    loop=next(n for n in ast.walk(solve) if isinstance(n,ast.For))
    start=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Assign)
               and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='guess')
    declared=body.interface_description()['default_transformation']

    class ArrayNames(ast.NodeTransformer):
        def visit_Attribute(self,node):
            if ast.unparse(node)=='incumbent._params':
                return ast.Name(id='best_params',ctx=ast.Load())
            return self.generic_visit(node)
        def visit_Name(self,node):
            return ast.Name(id={'guess':'ordinary_params','noise':'uniform_noise'}.get(node.id,node.id),ctx=node.ctx)
        def visit_BinOp(self,node):
            if ast.unparse(node)=='2 * N_CIRCLES':
                return ast.Constant(value=52)
            return self.generic_visit(node)

    actual=ast.Module(body=copy.deepcopy(loop.body[start:start+3]),type_ignores=[])
    expected=ast.parse(declared['readonly_array_equivalent'])
    assert ast.dump(ArrayNames().visit(actual))==ast.dump(expected)
    noise=loop.body[start-1]
    assert ast.unparse(noise)=='noise = np.random.uniform(-0.06, 0.06, initial.shape)'
    assert declared['existing_noise']=={'distribution':'uniform','low':-.06,'high':.06,
        'shape':[78],'draws_per_restart':1,'drawn_before_body':True}
    assert next(n.value.value for n in tree.body if isinstance(n,ast.Assign)
                and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='N_CIRCLES')==26
    assert declared['control_source_sha256']==sha(source.encode())=='3089f4e36bf787275ade5c3c24d4836ad8dc26e409185c0679c3377f4ce9c5f9'
    assert declared['common_scaffold_sha256']==sha(old.source_for().encode())==body.PINNED_SCAFFOLD_SHA256
    assert 'Read-only' in declared['scope'] and 'already a baseline component' in declared['scope']
    assert 'not establish novelty' in declared['scope']


def test_index_zero_facts_match_initialization_and_invocation_order_without_solver():
    tree=ast.parse(body.source_for()); solve=next(n for n in tree.body
        if isinstance(n,ast.FunctionDef) and n.name=='solve_packing')
    initial=next(n for n in solve.body if isinstance(n,ast.Assign)
                 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='initial')
    incumbent=next(n for n in solve.body if isinstance(n,ast.Assign)
                   and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='incumbent')
    assert ast.unparse(initial)=='initial = initial_parameters()'
    assert ast.unparse(incumbent)=='incumbent = Incumbent(initial)'
    init=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Incumbent').body[1]
    assert isinstance(init,ast.FunctionDef) and init.name=='__init__'
    assert any(ast.unparse(n)=='self._params = _parameters(initial).copy()' for n in init.body)
    loop=next(n for n in ast.walk(solve) if isinstance(n,ast.For))
    helper=next(n for n in loop.body if isinstance(n,ast.If) and ast.unparse(n.test)=='_ % 4 == 0')
    call=next(n for n in ast.walk(helper) if isinstance(n,ast.Call)
              and isinstance(n.func,ast.Name) and n.func.id=='_proposal_invoke')
    optimizer=next(n for n in ast.walk(solve) if isinstance(n,ast.Call)
                   and isinstance(n.func,ast.Name) and n.func.id=='minimize')
    assert ast.unparse(loop.iter)=='range(max_restarts)'
    assert [ast.unparse(arg) for arg in call.args]==['incumbent._params','initial','noise','guess','_','stats']
    assert initial.lineno<incumbent.lineno<loop.lineno<helper.lineno<optimizer.lineno
    declared=body.interface_description()['default_transformation']
    assert 'equal values before any SLSQP solve' in declared['index_zero']
    assert 'already been applied' in declared['index_zero']
    assert 'even if the body mutates' in declared['none_fallback']
    assert 'without these automatic clamps' in declared['none_fallback']


def test_declared_default_facts_fail_closed_on_dependency_drift_and_are_detached(monkeypatch):
    info=body.interface_description(); expected=copy.deepcopy(info['default_transformation'])
    info['default_transformation']['existing_noise']['low']=-1
    assert body.interface_description()['default_transformation']==expected
    monkeypatch.setattr(body,'PINNED_SCAFFOLD_SHA256','0'*64)
    with pytest.raises(ValueError,match='pinned dependency'):
        body.interface_description()


@pytest.mark.parametrize('inject',[False,True])
@pytest.mark.parametrize('policy',[None,'first_eligible_of_two'])
def test_exact_baseline_facts_reach_actual_body_generation_prompt(inject,policy):
    from sciel.config import apply_defaults
    from sciel.engine.circle_generation import CircleGeneration
    from sciel.engine.db import Database
    from sciel.engine.llm.client import LLMResponse
    from sciel.engine.research import ResearchController
    from sciel.engine.types import Snapshot
    from sciel.hypothesis import circle_proposal_pair
    def forbidden(*args,**kwargs): pytest.fail('provider or evaluator is forbidden')
    calls=[]
    wire_module=body if policy is None else circle_proposal_pair
    def capture(prompt,model,temperature=None,*,response_format=None):
        calls.append({'prompt':prompt,'response_format':response_format})
        return LLMResponse('invalid JSON fixture',model,0.,0,0,finish_reason='stop')
    llm=SimpleNamespace(models=['fake'],max_retries=1,generate=capture)
    config=apply_defaults({'research':{'enabled':True,'max_aux_calls':0,
        'reflection_interval':0,'inject_lessons':inject}})['research']
    controller=ResearchController(config,llm,SimpleNamespace(research=lambda **event:None),forbidden)
    db=Database(1); source=body.source_for()
    parent=db.add(source,.8,{'sum_radii':2.108},0,'seed')
    snap=Snapshot(0,'circle_generate','fake',0,parent.id,source,.8)
    snap.research_context=controller.context(db,parent,0,1)
    generation=CircleGeneration(controller,proposal_policy=policy)
    generation.run(snap,llm,forbidden)
    assert len(calls)==1 and calls[0]['response_format']==wire_module.response_format()
    prompt=calls[0]['prompt']
    interface_text=wire_module.response_instruction().split('\n',1)[1]
    interface=json.loads(interface_text)
    assert interface==body.interface_description() and prompt.count(interface_text)==1
    current=json.JSONDecoder().raw_decode(prompt.split('CURRENT_RESTART_BODY_JSON:\n',1)[1])[0]
    assert 'interface' not in current
    assert current['interface_reference']=={'contract':body.CONTRACT,
        'interface_sha256':sha(interface_text.encode()),
        'location':'Complete typed array interface in the response instruction.'}
    declared=interface['default_transformation']
    assert declared==body.interface_description()['default_transformation']
    assert current['body']==body.DEFAULT_BODY and current['source_sha256']==declared['control_source_sha256']
    assert '0.001' in prompt and '0.999' in prompt and 'equal values before any SLSQP solve' in prompt
    assert 'class Incumbent' not in prompt and 'incumbent._params' not in prompt
    context=json.JSONDecoder().raw_decode(prompt.split('RESEARCH_CONTEXT_JSON:\n',1)[1])[0]
    assert context['parent_score']==.8 and context['parent_cases']=={'sum_radii':2.108}
    assert context['parent_source_sha256']==current['source_sha256']
    assert controller.attempts==controller.outcomes==1 and controller.aux_calls==0
    assert generation.evaluations==0


def test_interface_schema_identity_changes_leave_control_bytes_and_old_dependency_pins_exact():
    identity=body.identity()
    assert sha(body.source_for().encode())==identity['control_source_sha256']=='3089f4e36bf787275ade5c3c24d4836ad8dc26e409185c0679c3377f4ce9c5f9'
    assert sha(Path(old.__file__).read_bytes())==body.PINNED_PROPOSALS_SHA256
    assert sha(old.source_for().encode())==body.PINNED_SCAFFOLD_SHA256
    assert identity['interface_sha256']!='7785086ad59286421e9d2c583354bb4083751f54354ff1d69d3bc235bdcd885c'
    # The future wire description now also states the zero-base-indent rule.
    assert identity['response_format_sha256']!='6e690537e693161ac042f0cf7cbce7622dd75ab65cba506a0600d8af63a2c931'
    assert identity['interface_sha256']!='ad7da4008066f07decdc2e7284c253893770156a0ab79b22a8be5f8a9ee4d2af'
    # This factual API extension does not alter the existing native response schema.
    assert identity['response_format_sha256']=='ece5d31cb6feb4198d1114fa82c527b044c759889bb9a4b82876e1cd0ad2e5dd'
    # A future prompt identity cannot silently resume an older prompt contract.
    from sciel.engine.circle_generation import CircleGeneration
    generation=CircleGeneration(SimpleNamespace(config={}));state=generation.state()
    restored=CircleGeneration(SimpleNamespace(config={}));restored.load_state(state)
    assert restored.state()==state
    for field,old_value in [('interface_sha256','7785086ad59286421e9d2c583354bb4083751f54354ff1d69d3bc235bdcd885c'),
                            ('interface_sha256','ad7da4008066f07decdc2e7284c253893770156a0ab79b22a8be5f8a9ee4d2af'),
                            ('source_sha256','2b2234c3915cf5aaafb99e2378c683c934912b3dac57fb7ced9c78939867bec8'),
                            ('response_format_sha256','6e690537e693161ac042f0cf7cbce7622dd75ab65cba506a0600d8af63a2c931')]:
        legacy=copy.deepcopy(state);legacy['identity']['builder'][field]=old_value
        with pytest.raises(ValueError,match='checkpoint'):restored.load_state(legacy)


def namespace(code):
    # Definitions are loaded, but actual optimizer is never called.
    ns={}; exec(compile(body.source_for(code),'<fake-only-body-scaffold>','exec'),ns)
    ns['time']=SimpleNamespace(monotonic=lambda:0.)
    def forbidden(*args,**kwargs): pytest.fail('actual optimizer forbidden')
    ns['minimize']=forbidden
    return ns


def test_synthetic_handoff_uses_detached_alias_arrays_without_clipping_or_rng():
    code='best_params[:] = -2\ninitial_params[:] = 3\nuniform_noise[:] = 4\nordinary_params[:] = 5\nreturn best_params'
    ns=namespace(code); inputs=[np.arange(78,dtype=float)+i for i in range(4)]
    copies=[x.copy() for x in inputs]; saved=np.random.get_state()
    stats=ns['_proposal_stats'](); result=ns['_proposal_invoke'](*inputs,0,stats)
    np.testing.assert_array_equal(result,np.full(78,-2.))
    for a,b in zip(inputs,copies): np.testing.assert_array_equal(a,b)
    assert all(np.array_equal(a,b) for a,b in zip(saved,np.random.get_state()))
    assert stats['proposal_attempts']==stats['proposal_accepted']==1


def test_none_alias_mutation_cannot_change_ordinary_fallback():
    ns=namespace('ordinary_params[:] = -7\nreturn None')
    inputs=[np.arange(78,dtype=float)+i for i in range(4)]
    value=ns['_proposal_invoke'](*inputs,0,ns['_proposal_stats']())
    np.testing.assert_array_equal(value,inputs[-1]); assert not np.shares_memory(value,inputs[-1])


def test_fake_solve_calls_body_on_indices_zero_four_with_original_draw_shape():
    ns=namespace('return ordinary_params')
    seen=[]; original=ns['propose_restart']
    def record(*args): seen.append(args[-1]); return original(*args)
    ns['propose_restart']=record
    def fake(fn,guess,**kwargs):
        assert guess.shape==(78,) and kwargs['method']=='SLSQP'
        return SimpleNamespace(x=ns['initial_parameters'](),success=True)
    ns['minimize']=fake
    saved=np.random.get_state()
    try: ns['solve_packing'](max_restarts=5)
    finally: np.random.set_state(saved)
    assert seen==[0,4] and ns['LAST_STATS']['proposal_accepted']==2


def test_legacy_hypothesis_transport_has_exact_same_scope_after_value_extraction():
    raw=json.dumps(HYP)
    expected=parse_hypothesis_value(HYP)
    for accepted in ['HYPOTHESIS_JSON:'+raw,'intro\nHYPOTHESIS_JSON: '+raw+'\n```python\npass\n```',raw+'\n```python\npass\n```']:
        assert parse_hypothesis(accepted)==expected
    for rejected in [raw,'```json\n'+raw+'\n```',wire(),
                     'HYPOTHESIS_JSON:'+json.dumps({**HYP,'delta':0}),
                     'HYPOTHESIS_JSON:'+raw+'\nREFLECTION_JSON:{}']:
        assert parse_hypothesis(rejected) is None


def test_legacy_fenced_responses_stay_rejected_and_interface_errors_remain():
    for source in ("return np.random.choice(ordinary_params)\n",
                   "return best_params.tolist()\n"):
        text = '```python\n' + body._HEADER + body._indented(source) + '\n```'
        original = text
        assert body.parse_response(text) is None
        assert parse_hypothesis(text) is None
        # Diagnostic extraction only: never assembled or executed as a candidate.
        function = text.split('```python\n', 1)[1].rsplit('\n```', 1)[0]
        diagnostic_body = ''.join(line[4:] + '\n' for line in function.splitlines()[1:])
        with pytest.raises(ValueError):
            body.validate_body(diagnostic_body)
        assert text == original
