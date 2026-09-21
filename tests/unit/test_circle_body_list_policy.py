"""Static syntax/provenance tests only; no helper, optimizer or evaluator runs."""
import ast
import json
from pathlib import Path

import pytest

from sciel.hypothesis import circle_body_list_policy as policy
from sciel.hypothesis import circle_proposals as legacy

ROOT=Path(__file__).resolve().parents[2]
HEADER='def propose_restart(incumbent, initial, noise, default_guess, restart_index):\n'


def helper(body):
    return HEADER+''.join('    '+line+'\n' for line in body.splitlines())


@pytest.mark.parametrize('declaration',[
    'values = []','values = [1, 2]','values = [i for i in range(3)]',
    'values = list()','values = list(range(3))'])
def test_fresh_top_level_lists_admit_standalone_append_preserving_exact_bytes(declaration):
    text=helper(declaration+'\nfor i in range(3):\n    if i > 0:\n        values.append(i)\nreturn None')
    assert policy.validate_helper(text)==text
    with pytest.raises(ValueError,match='attribute'):legacy.validate_helper(text)


def test_local_append_does_not_rewrite_whitespace_comments_or_crlf():
    text=helper('values=[]  # literal layout preserved\nvalues.append( 1 )\nreturn None').replace('\n','\r\n')
    expected=text.strip('\r\n')+'\n'
    assert policy.validate_helper(text)==expected


@pytest.mark.parametrize('body',[
    'values = []\nalias = values\nalias.append(1)',
    'incumbent.append(1)',
    'incumbent = []\nincumbent.append(1)',
    'values = incumbent\nvalues.append(1)',
    'values = []\nvalues = np.zeros(3)\nvalues.append(1)',
    'values = []\nif restart_index:\n    values = []\nvalues.append(1)',
    'values = []\nfor values in [1]:\n    pass\nvalues.append(1)',
    'values = []\nvalues += [1]\nvalues.append(1)',
    'if restart_index:\n    values = []\nvalues.append(1)',
    'values.append(1)\nvalues = []',
    'values = []\nmethod = values.append',
    'values = []\nresult = values.append(1)',
    'values = []\nother = [values.append(i) for i in range(3)]',
    'values = []\nvalues.append(value=1)',
    'values = []\nvalues.append()',
    'values = []\nvalues.append(1, 2)',
    'values = []\nvalues.append(*[1])',
    'values = []\nvalues.append(**{})',
    'values = []\nvalues.append = 1',
    'values = []\nvalues[0].append(1)',
    'values = []\nvalues.copy().append(1)',
    'values = []\n(values if restart_index else []).append(1)',
    'values = alias = []\nvalues.append(1)',
    'values = list(values=[1])\nvalues.append(1)',
    'values = []\nother = [values for values in range(3)]\nvalues.append(1)',
    'np.append([1], 2)',
    'list.append([1], 2)',
])
def test_receiver_rebinding_aliases_escapes_and_wrong_calls_rejected(body):
    with pytest.raises(ValueError):policy.validate_helper(helper(body+'\nreturn None'))


@pytest.mark.parametrize('extra',[
    'values.extend([1])','values.insert(0, 1)','values.clear()',
    'np.random.uniform(0, 1)','np.bool_(1)','open("secret")','eval("1")',
    'values.append.__call__(1)','import os','while True:\n    pass',
    'def nested():\n    return 1','np = []','x = incumbent.__class__',
    'x = globals()','propose_restart(incumbent, initial, noise, default_guess, restart_index)',
])
def test_new_append_does_not_broaden_other_legacy_restrictions(extra):
    text=helper('values = []\nvalues.append(1)\n'+extra+'\nreturn None')
    with pytest.raises(ValueError):policy.validate_helper(text)


def test_numpy_bool_dtype_is_already_legacy_supported():
    text=helper('mask = np.ones(26, dtype=np.bool_)\nreturn None')
    assert policy.validate_helper(text)==legacy.validate_helper(text)==text


@pytest.mark.parametrize('text',[
    helper('return None'),helper('x = [float(i) for i in range(4)]\nreturn None'),
    helper('x = np.copy(incumbent)\nx[:2] = 0.1\nreturn x'),
])
def test_legacy_accepted_helpers_delegate_exactly(text,monkeypatch):
    called=[];old=legacy.validate_helper
    def observe(value):called.append(value);return old(value)
    monkeypatch.setattr(legacy,'validate_helper',observe)
    assert policy.validate_helper(text)==text and called==[text]


@pytest.mark.parametrize('text',[None,1,'not python',HEADER+'    break\n',HEADER+'    return "\ud800"\n'])
def test_unextended_failures_keep_legacy_error_type_and_message(text):
    with pytest.raises(ValueError) as before:legacy.validate_helper(text)
    with pytest.raises(ValueError) as after:policy.validate_helper(text)
    assert str(before.value)==str(after.value)


def test_append_error_carries_original_ast_location_for_body_diagnostic():
    text=helper('values = incumbent\nvalues.append(1)\nreturn None')
    with pytest.raises(policy.LocalListPolicyError) as caught:policy.validate_helper(text)
    node=caught.value.node
    assert isinstance(node,ast.Attribute) and node.attr=='append' and node.lineno==3 and node.col_offset==4


def test_legacy_pin_and_policy_bytes_participate_in_identity(monkeypatch):
    identity=policy.identity()
    assert identity['version']==policy.POLICY_VERSION
    assert identity['legacy_guard_sha256']==policy.PINNED_PROPOSALS_SHA256
    read=Path.read_bytes
    def changed(path):
        value=read(path)
        return value+b'\n# inert policy drift' if path.resolve()==Path(policy.__file__).resolve() else value
    monkeypatch.setattr(Path,'read_bytes',changed)
    assert policy.identity()['source_sha256']!=identity['source_sha256']
    monkeypatch.setattr(policy,'PINNED_PROPOSALS_SHA256','0'*64)
    with pytest.raises(ValueError,match='pinned legacy'):policy.validate_helper(helper('return None'))


def test_real_rejected_model_body_has_static_future_policy_compatibility_only():
    # Parse captured text only. This does not assemble/evaluate its candidate,
    # alter the immutable no_child result, or prove the requested numerical design.
    path=ROOT/'tests/fixtures/circle_body_list_policy_rejected.json'
    fixture=json.loads(path.read_text());body=fixture['body']
    assert fixture['original_provider_return_sha256']=='62609ea8f8a4af65f9cfc2eb5de806227c6e56164195d80b73b08dd6657f4251'
    assert policy._sha(body.encode())==fixture['body_sha256']
    text=HEADER+'    best_params = incumbent\n    initial_params = initial\n    uniform_noise = noise\n    ordinary_params = default_guess\n'+''.join('    '+line+'\n' for line in body.splitlines())
    with pytest.raises(ValueError,match='attribute'):legacy.validate_helper(text)
    assert policy.validate_helper(text)==text


@pytest.mark.parametrize('kind',['characters','utf8_bytes','normalized_lf','non_text','invalid_utf8'])
def test_rejected_preparse_inputs_never_reenter_ast_parser(kind,monkeypatch):
    if kind=='characters':text='x'*(legacy.MAX_HELPER_BYTES+1)
    elif kind=='utf8_bytes':text=HEADER+'    # '+'é'*9000
    elif kind=='normalized_lf':text=HEADER+'    # '+'x'*(legacy.MAX_HELPER_BYTES-len(HEADER)-6)
    elif kind=='non_text':text=b'not a Python source string'
    else:text=HEADER+'    # \ud800'
    def forbidden(*args,**kwargs):pytest.fail('pre-parser rejection invoked ast.parse')
    monkeypatch.setattr(ast,'parse',forbidden)
    with pytest.raises(ValueError) as before:legacy.validate_helper(text)
    with pytest.raises(ValueError) as after:policy.validate_helper(text)
    assert str(before.value)==str(after.value)
