"""Immutable island tool storage and sandbox handoff; no external dispatches."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from sciel.methods.executor import DockerExecutor, ExecutionRefused, _BOUND_KEYS
from sciel.methods.models import content_hash
from sciel.research_tools.library import ToolLibrary, MAX_OBJECT_BYTES, MAX_CODE_BYTES

IMAGE = 'sha256:' + 'a' * 64
LOCK = {'image_id': IMAGE}
LIMITS = dict(wall_seconds=5, cpu_seconds=2, memory_bytes=128 * 1024 * 1024,
    pids=16, max_output_bytes=65536, max_stderr_bytes=2048)
CODE = 'def run(payload):\n    return payload\n'


class FakeExecutor:
    identity = {'schema':'fixture-executor/1', 'image_id':IMAGE}

    def __init__(self, receipt=None):
        self.calls = []
        self.receipt = receipt or dict(status='ok', dispatch_status='invoked', output={'value':1},
            resources=dict(cpu_seconds=.1, wall_seconds=.2), dependency_use_attested=False)

    def execute(self, bundle, inputs, **kwargs):
        self.calls.append(deepcopy(dict(bundle=bundle, inputs=inputs, **kwargs)))
        return deepcopy(self.receipt)


class ToolLibraryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name).resolve() / 'run'
        self.factory = Mock(return_value=FakeExecutor())
        self.library = ToolLibrary(self.run, LOCK, LIMITS, executor_factory=self.factory)

    def register(self, name='fixture', **kwargs):
        return self.library.register(name, 'Technical fixture; no benefit claim.', CODE,
            kwargs.pop('author_island', 0), **kwargs)

    def execute(self, bundle, inputs=None, **kwargs):
        return self.library.execute(bundle, {'value':1} if inputs is None else inputs,
            invocation_id=kwargs.pop('invocation_id', 'invocation-1'),
            plan_id='engine-decision-1', reservation_id='reservation-1', **kwargs)

    def test_content_addressed_versions_are_global_and_detached_without_laboratories(self):
        original = self.register()
        expected = content_hash(dict(schema='sciel-method/1', manifest=original['manifest'], files=original['files']))
        self.assertEqual(original['id'], expected)
        self.assertEqual(self.register()['id'], expected)
        self.assertEqual(self.library.get(expected)['manifest']['protocol']['author_island'], 0)
        self.assertEqual(self.library.path, self.run / 'research_tools')
        self.assertEqual(len(list(self.library.objects.iterdir())), 1)
        self.assertEqual(json.loads((self.library.path / 'identity.json').read_text())['schema'], 'sciel-research-tool-library/3')
        original['files']['tool.py'] = 'tampered returned copy'
        self.assertEqual(self.library.get(expected)['files']['tool.py'], CODE)
        resumed = ToolLibrary(self.run, LOCK, LIMITS, executor_factory=self.factory)
        foreign_child = resumed.register('repair', 'A separately identified revision.', CODE, 'island-7', parents=[expected])
        self.assertEqual(foreign_child['manifest']['parents'], [expected])
        self.assertEqual(self.library.get(foreign_child['id'])['manifest']['protocol']['author_island'], 'island-7')
        self.assertFalse((self.library.path / 'revisions').exists())
        self.assertFalse((self.library.path / 'journal.jsonl').exists())
        self.assertFalse(hasattr(self.library, 'list'))  # visibility is controller-owned
        self.factory.assert_not_called()
        unrelated = ToolLibrary(self.run.parent / 'other', LOCK, LIMITS, executor_factory=self.factory)
        with self.assertRaises(FileNotFoundError): unrelated.get(expected)

    def test_configuration_and_executor_identity_are_pinned_and_lazy(self):
        self.register()
        self.factory.assert_not_called()
        identity = self.library.identity
        self.factory.assert_called_once_with(IMAGE)
        self.assertEqual(identity['executor']['image_id'], IMAGE)
        identity['executor']['image_id'] = 'changed'
        self.library.limits['cpu_seconds'] = 999
        self.library.runtime_lock['image_id'] = 'changed'
        self.assertEqual(self.library.identity['limits'], LIMITS)
        self.assertEqual(self.library.identity['runtime_lock'], LOCK)
        self.assertEqual(self.library.identity['executor']['image_id'], IMAGE)
        before = (self.library.path / 'identity.json').read_bytes()
        for runtime, limits in [(dict(image_id='sha256:'+'b'*64), LIMITS), (LOCK, dict(LIMITS,cpu_seconds=3))]:
            with self.assertRaises(ValueError): ToolLibrary(self.run, runtime, limits)
        self.assertEqual((self.library.path / 'identity.json').read_bytes(), before)

    def test_invalid_contracts_fail_without_loading_executor_or_running_source(self):
        for code in ['def run(:\n', 'x = 1\n', 'async def run(payload): return payload\n', CODE + '#' * MAX_CODE_BYTES]:
            with self.subTest(code=code[:40]), self.assertRaises(ValueError):
                self.library.register('bad', 'Invalid fixture.', code, 0)
        for field, value in [('cpu_seconds', float('inf')), ('wall_seconds', True), ('pids',1),
                             ('memory_bytes',1024), ('max_output_bytes',17*1024*1024)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                ToolLibrary(self.run.parent / field, LOCK, dict(LIMITS, **{field:value}))
        with self.assertRaises(ValueError): ToolLibrary(self.run.parent / 'mutable', {'image_id':'python:latest'}, LIMITS)
        with self.assertRaises(ValueError): self.register(parents=['../outside'])
        with self.assertRaises(FileNotFoundError): self.register(parents=['b'*64])
        with self.assertRaises(ValueError): self.register(author_island=True)
        self.factory.assert_not_called()

    def test_dependencies_are_exact_transitive_versions_while_parents_are_provenance(self):
        a = self.register('a')
        b = self.register('b', parents=[a['id']], dependencies=[a['id']])
        c = self.register('c', dependencies=[b['id'], a['id']])
        self.assertEqual([entry['id'] for entry in self.library.dependency_closure(c['id'])], [a['id'],b['id']])
        repaired = self.register('revision', parents=[c['id']])
        self.assertEqual(self.library.dependency_closure(repaired['id']), [])
        self.execute(c)
        invocation = self.factory.return_value.calls[0]
        self.assertEqual([entry['id'] for entry in invocation['dependency_bundles']], [a['id'],b['id']])
        with self.assertRaises(ValueError): self.register('duplicate', dependencies=[a['id'],a['id']])
        before = len(list(self.library.objects.iterdir()))
        with patch('sciel.research_tools.library.MAX_DEPENDENCIES', 2):
            with self.assertRaises(ValueError): self.register('too-deep', dependencies=[c['id']])
        self.assertEqual(len(list(self.library.objects.iterdir())), before)

    def test_dependency_cycle_is_refused_before_execution(self):
        a, b = 'a'*64, 'b'*64
        values = {a:dict(id=a,files={'tool.py':CODE},manifest={'dependencies':[b]}),
                  b:dict(id=b,files={'tool.py':CODE},manifest={'dependencies':[a]})}
        with patch.object(self.library, 'get', side_effect=lambda ident:deepcopy(values[ident])):
            with self.assertRaisesRegex(ValueError, 'cyclic'): self.library.dependency_closure(a)
        self.factory.assert_not_called()

    def test_tampered_duplicate_oversized_and_linked_artifacts_are_refused(self):
        bundle = self.register()
        path = self.library.objects / (bundle['id'] + '.json')
        original = path.read_bytes()
        changed = json.loads(original); changed['files']['tool.py'] += '# tamper\n'
        path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, 'hash mismatch'): self.library.get(bundle['id'])
        path.write_bytes(b'{"schema":"one","schema":"two"}')
        with self.assertRaisesRegex(ValueError, 'duplicate'): self.library.get(bundle['id'])
        path.write_bytes(b' ' * (MAX_OBJECT_BYTES + 1))
        with self.assertRaisesRegex(ValueError, 'oversized'): self.library.get(bundle['id'])
        path.unlink()
        outside = self.run.parent / 'outside.json'; outside.write_bytes(original); path.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'symbolic'): self.library.get(bundle['id'])
        with self.assertRaises(ValueError): self.library.get('../outside')
        self.factory.assert_not_called()

    def test_source_limit_precedes_ast_parse_and_nonregular_storage_cannot_block(self):
        with patch('ast.parse') as parse:
            with self.assertRaises(ValueError):
                self.library.register('huge', 'Source limit fixture.', CODE + '#' * MAX_CODE_BYTES, 0)
            parse.assert_not_called()
        ident = 'f'*64
        os.mkfifo(self.library.objects / (ident + '.json'))
        with self.assertRaisesRegex(ValueError, 'not a regular file'):
            self.library.get(ident)
        self.factory.assert_not_called()

    def test_rollback_leaves_immutable_objects_but_does_not_publish_any_visibility(self):
        first = self.register('first')
        committed_visible = [first['id']]
        abandoned = self.register('uncommitted')
        resumed = ToolLibrary(self.run, LOCK, LIMITS, executor_factory=self.factory)
        self.assertEqual(committed_visible, [first['id']])
        self.assertNotIn(abandoned['id'], committed_visible)
        self.assertEqual(resumed.get(abandoned['id'])['id'], abandoned['id'])
        self.factory.assert_not_called()

    def test_execution_arguments_are_bound_and_bad_input_fails_before_backend(self):
        bundle = self.register()
        changed = deepcopy(bundle); changed['files']['tool.py'] += '# changed'
        with self.assertRaises(ExecutionRefused): self.execute(changed)
        for inputs in [[], {'x':float('nan')}, {'x':float('inf')}, {'x':object()}]:
            with self.subTest(inputs=repr(inputs)), self.assertRaises(ExecutionRefused): self.execute(bundle, inputs)
        with self.assertRaises(ExecutionRefused): self.execute(bundle, invocation_id='bad\nidentifier')
        self.factory.assert_not_called()
        result = self.execute(bundle, {'value':0})
        call = self.factory.return_value.calls[0]
        self.assertEqual(call['invocation_id'], 'invocation-1')
        self.assertEqual(call['plan_id'], 'engine-decision-1')
        self.assertEqual(call['reservation_id'], 'reservation-1')
        self.assertEqual(call['inputs'], {'value':0})
        self.assertEqual(call['limits'], LIMITS)
        self.assertFalse(result['dependency_use_attested'])

    def test_unknown_resources_and_post_dispatch_exceptions_are_not_fabricated_or_retried(self):
        backend = FakeExecutor(dict(status='unknown', dispatch_status='unknown', output=None,
            resources=dict(cpu_seconds=None, wall_seconds=.25), reason='missing_receipt'))
        library = ToolLibrary(self.run, LOCK, LIMITS, executor=backend)
        bundle = library.register('unknown', 'Outcome fixture.', CODE, 0)
        result = library.execute(bundle, {}, invocation_id='i', plan_id='p', reservation_id='r')
        self.assertEqual(result['status'], 'unknown')
        self.assertIsNone(result['resources']['cpu_seconds'])
        self.assertEqual(result['resources']['wall_seconds'], .25)
        self.assertEqual(len(backend.calls), 1)
        with patch.object(backend, 'execute', side_effect=ValueError('after possible dispatch')) as execute:
            with self.assertRaises(ValueError) as caught:
                library.execute(bundle, {}, invocation_id='j', plan_id='p', reservation_id='s')
            self.assertNotIsInstance(caught.exception, ExecutionRefused)
            execute.assert_called_once()

    def test_real_executor_stages_code_without_importing_it_on_host(self):
        with patch('sciel.methods.executor._cli', side_effect=[
            subprocess.CompletedProcess([],0,(IMAGE+' linux arm64\n').encode(),b''),
            subprocess.CompletedProcess([],0,b'27.5.1\n',b'')]):
            executor = DockerExecutor(IMAGE)
        library = ToolLibrary(self.run, LOCK, LIMITS, executor=executor)
        code = "raise RuntimeError('this source must not run on the host')\n" + CODE
        bundle = library.register('untrusted', 'Host isolation fixture.', code, 0)
        def container(directory, request):
            self.assertEqual((directory/'bundle/tool.py').read_text(), code)
            raw = dict(schema='sciel-method-container-receipt/1',
                **{key:request[key] for key in _BOUND_KEYS}, status='ok',reason=None,invoked=True,
                exit_code=0,output={'answer':0},output_sha256=content_hash({'answer':0}),limits=request['limits'],
                resources=dict(cpu_seconds=.1,wall_seconds=.2,memory_peak_bytes=20000,output_bytes=12,stderr_bytes=0),
                stderr='',stderr_truncated=False)
            return dict(stdout=json.dumps(raw).encode(),stderr='',exit_code=0,timed_out=False,
                overflow=False,dispatched=True,removed=True,name='fixture')
        with patch.object(executor, '_container', side_effect=container) as dispatch:
            receipt = library.execute(bundle, {}, invocation_id='i',plan_id='p',reservation_id='r')
        dispatch.assert_called_once()
        self.assertEqual(receipt['output'], {'answer':0})
        self.assertEqual(receipt['method_id'], bundle['id'])
        self.assertEqual(receipt['source_sha256'], content_hash(bundle['files']))
        self.assertEqual(receipt['receipt_sha256'], content_hash({k:v for k,v in receipt.items() if k!='receipt_sha256'}))


if __name__ == '__main__':
    unittest.main()
