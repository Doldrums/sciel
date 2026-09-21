"""Host validation and optional real Docker checks; no generated host execution."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from sciel.methods.executor import DockerExecutor, ExecutionRefused, _BOUND_KEYS, _environment
from sciel.methods.models import content_hash

IMAGE = 'sha256:' + 'a' * 64
LIMITS = dict(wall_seconds=5, cpu_seconds=2, memory_bytes=128 * 1024 * 1024,
              pids=32, max_output_bytes=65536, max_stderr_bytes=2048)


def bundle(code='def run(value):\n    return value + 1\n', *, image=IMAGE, dependencies=(), output_schema=None, limits=None):
    manifest = dict(name='fixture', kind='tool', author_lab='lab-test', entrypoint='method.py:run',
        input_schema={'type': 'integer'}, output_schema=output_schema or {'type': 'integer'},
        protocol='Apply once to one JSON input.', runtime_lock={'image_id': image},
        dependencies=list(dependencies), parents=[], resources=deepcopy(limits or LIMITS),
        scope='Technical execution fixture; no scientific benefit claim.', examples=[])
    files = {'method.py': code}
    return dict(id=content_hash(dict(schema='sciel-method/1', manifest=manifest, files=files)),
                manifest=manifest, files=files)


def add_files(method, files):
    method = deepcopy(method)
    method['files'].update(files)
    method['id'] = content_hash(dict(schema='sciel-method/1', manifest=method['manifest'], files=method['files']))
    return method


def executor():
    with patch('sciel.methods.executor._cli', side_effect=[
        subprocess.CompletedProcess([], 0, (IMAGE + ' linux arm64\n').encode(), b''),
        subprocess.CompletedProcess([], 0, b'27.5.1\n', b'')]):
        return DockerExecutor(IMAGE)


def transport(request, output=2, *, removed=True):
    raw = dict(schema='sciel-method-container-receipt/1',
        **{key: request[key] for key in _BOUND_KEYS}, status='ok', reason=None, invoked=True,
        exit_code=0, output=output, output_sha256=content_hash(output), limits=request['limits'],
        resources=dict(cpu_seconds=.1, wall_seconds=.2, memory_peak_bytes=20000, output_bytes=1, stderr_bytes=0),
        stderr='', stderr_truncated=False)
    return dict(stdout=json.dumps(raw).encode(), stderr='', exit_code=0, timed_out=False,
        overflow=False, dispatched=True, removed=removed, name='fixture')


class ExecutorTests(unittest.TestCase):
    def execute(self, ex, method=None, value=1, **kwargs):
        return ex.execute(method or bundle(), value, invocation_id='inv1', plan_id='plan1',
            reservation_id='reservation1', limits=deepcopy(LIMITS), **kwargs)

    def test_requires_immutable_available_image_and_identity_is_detached(self):
        with self.assertRaises(ValueError): DockerExecutor('python:latest')
        with patch('sciel.methods.executor._cli', return_value=subprocess.CompletedProcess([], 1, b'', b'no image')):
            with self.assertRaises(RuntimeError): DockerExecutor(IMAGE)
        ex = executor()
        identity = ex.identity
        identity['policy']['network'] = 'host'
        self.assertEqual(ex.identity['policy']['network'], 'none')
        self.assertEqual(ex.identity['image_id'], IMAGE)

    def test_no_host_code_execution_and_exact_bindings_and_json_schema(self):
        ex = executor()
        method = bundle("raise RuntimeError('must not execute on host')\ndef run(value):\n    return value + 1\n")
        captured = {}
        def perform(directory, request):
            captured.update(deepcopy(request))
            self.assertEqual((directory / 'bundle/method.py').read_text(), method['files']['method.py'])
            self.assertTrue((directory / 'runtime/_runner.py').is_file())
            return transport(request)
        with patch.object(ex, '_container', side_effect=perform):
            result = self.execute(ex, method)
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['output'], 2)
        self.assertEqual(result['method_id'], method['id'])
        self.assertEqual(result['input_sha256'], content_hash(1))
        self.assertEqual(result['source_sha256'], content_hash(method['files']))
        self.assertEqual(result['runtime_sha256'], content_hash(ex.identity))
        self.assertEqual(result['request_sha256'], content_hash({k:v for k,v in captured.items() if k != 'request_sha256'}))
        self.assertEqual(result['receipt_sha256'], content_hash({k:v for k,v in result.items() if k != 'receipt_sha256'}))

    def test_invalid_source_runtime_input_and_limits_never_dispatch(self):
        for change in ['source', 'image', 'input', 'limits']:
            with self.subTest(change=change):
                ex, method, value = executor(), bundle(), 1
                if change == 'source': method['files']['method.py'] += '# changed\n'
                if change == 'image': method = bundle(image='sha256:'+'b'*64)
                if change == 'input': value = True
                if change == 'limits':
                    method = bundle(limits={**LIMITS, 'wall_seconds': 1})
                with patch.object(ex, '_container') as call:
                    with self.assertRaises(ValueError): self.execute(ex, method, value)
                    call.assert_not_called()

    def test_known_preflight_and_staging_refusals_cannot_claim_unknown_dispatch(self):
        ex = executor()
        with patch.object(ex, '_container') as call:
            with self.assertRaises(ExecutionRefused):
                ex.preflight(bundle(limits={**LIMITS, 'pids':1}), 1, {**LIMITS, 'pids':1})
            with patch('sciel.methods.executor.tempfile.TemporaryDirectory', side_effect=OSError('fixture disk full')):
                with self.assertRaises(ExecutionRefused): self.execute(ex)
            call.assert_not_called()
        # An unexpected error after transport entry is not reclassified as a
        # safe-to-retry validation refusal, even if its Python type is ValueError.
        with patch.object(ex, '_container', side_effect=ValueError('after possible dispatch')):
            with self.assertRaises(ValueError) as caught: self.execute(ex)
        self.assertNotIsInstance(caught.exception, ExecutionRefused)

    def test_supervisor_receipt_not_child_self_report_controls_identity(self):
        for field, value in [('invocation_id', 'other'), ('request_sha256', '0'*64),
                             ('output_sha256', '0'*64), ('invoked', False)]:
            with self.subTest(field=field):
                ex = executor()
                def perform(directory, request):
                    result = transport(request)
                    raw = json.loads(result['stdout']); raw[field] = value
                    result['stdout'] = json.dumps(raw).encode()
                    return result
                with patch.object(ex, '_container', side_effect=perform): result = self.execute(ex)
                self.assertEqual(result['status'], 'unknown')
                self.assertEqual(result['invocation_id'], 'inv1')
                self.assertIsNone(result['resources']['cpu_seconds'])

    def test_lost_dispatch_receipt_unknown_and_create_failure_known_not_invoked(self):
        for dispatched in [True, False]:
            ex = executor()
            raw = dict(dispatched=dispatched, removed=True, stdout=b'', stderr='lost', name='fixture')
            with patch.object(ex, '_container', return_value=raw): result = self.execute(ex)
            self.assertEqual(result['status'], 'unknown' if dispatched else 'failed')
            self.assertEqual(result['resources']['cpu_seconds'], None if dispatched else 0)

    def test_output_schema_failure_and_cleanup_uncertainty(self):
        ex = executor()
        with patch.object(ex, '_container', side_effect=lambda d,r:transport(r, output='invalid')):
            result = self.execute(ex)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['reason'], 'output_schema_invalid')
        self.assertEqual(result['output_sha256'], content_hash('invalid'))
        with patch.object(ex, '_container', side_effect=lambda d,r:transport(r, removed=False)):
            result = self.execute(ex)
        self.assertEqual(result['status'], 'unknown')
        self.assertEqual(result['reason'], 'container_cleanup_unconfirmed')

    def test_exact_dependency_closure_mounted_as_readonly_namespaces(self):
        dep = bundle('def run(value):\n    return value * 2\n')
        middle = bundle(dependencies=[dep['id']])
        main = bundle(dependencies=[middle['id']])
        ex = executor()
        def perform(directory, request):
            for method in [dep, middle]:
                folder = directory / 'dependencies' / ('m_' + method['id'])
                self.assertEqual((folder / 'files/method.py').read_text(), method['files']['method.py'])
                self.assertIn('invoke_dependency', (folder / '__init__.py').read_text())
            return transport(request)
        with patch.object(ex, '_container', side_effect=perform):
            result = self.execute(ex, main, dependency_bundles=[middle, dep])
        self.assertEqual(result['dependency_ids'], sorted([dep['id'], middle['id']]))
        self.assertFalse(result['dependency_use_attested'])
        for deps in [[], [middle], [middle, dep, bundle('def run(value):\n return -value\n')], [middle, dep, dep]]:
            with self.subTest(count=len(deps)), patch.object(ex, '_container') as call:
                with self.assertRaises(ValueError): self.execute(ex, main, dependency_bundles=deps)
                call.assert_not_called()

    def test_container_isolation_flags_and_no_host_credentials(self):
        ex = executor()
        request = dict(limits=LIMITS)
        calls = []
        def cli(args, timeout=10):
            calls.append(args)
            return subprocess.CompletedProcess([], 0, b'c' * 64, b'')
        with patch('sciel.methods.executor._cli', side_effect=cli), patch('sciel.methods.executor._attach', return_value={
            'stdout':b'', 'stderr':'', 'exit_code':0, 'timed_out':False, 'overflow':False}):
            result = ex._container(Path('/tmp/fixture'), request)
        create = calls[0]
        for flag in ['--pull=never', '--network=none', '--read-only', '--user=0:0', '--cgroupns=private',
                     '--security-opt=no-new-privileges', '--cap-drop=ALL']:
            self.assertIn(flag, create)
        self.assertEqual(sum(item.endswith(',readonly') for item in create), 4)
        self.assertIn('--memory-swap='+str(LIMITS['memory_bytes']), create)
        self.assertTrue(result['removed'])
        with patch.dict(os.environ, OPENAI_API_KEY='not-a-real-key', GEMINI_API_KEY='not-a-real-key'):
            self.assertNotIn('OPENAI_API_KEY', _environment())
            self.assertNotIn('GEMINI_API_KEY', _environment())


@unittest.skipUnless(os.environ.get('SCIEL_TEST_METHOD_IMAGE'), 'explicit immutable local Docker image required')
class RealDockerExecutorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image = os.environ['SCIEL_TEST_METHOD_IMAGE']
        cls.executor = DockerExecutor(cls.image)

    def run_method(self, code, *, schema=None, limits=None, dependencies=()):
        allocation = limits or LIMITS
        method = bundle(code, image=self.image, dependencies=[d['id'] for d in dependencies],
                        output_schema=schema, limits=allocation)
        receipt = self.executor.execute(method, 3, invocation_id='real-'+method['id'], plan_id='technical-fixture',
            reservation_id='offline-resource-fixture', limits=allocation, dependency_bundles=dependencies)
        self.assertTrue(receipt['container']['removed'], receipt)
        return receipt

    def test_failure_repair_and_dependency_composition(self):
        failed = self.run_method('def run(value):\n    raise ValueError("known fixture defect")\n')
        self.assertEqual(failed['status'], 'failed', failed)
        self.assertIn('known fixture defect', failed['stderr'])
        repaired = self.run_method('def run(value):\n    return value + 1\n')
        self.assertEqual(repaired['status'], 'ok', repaired)
        self.assertEqual(repaired['output'], 4)
        self.assertGreater(repaired['resources']['cpu_seconds'], 0)
        dep = bundle('def run(value):\n    return value * 2\n', image=self.image)
        composed = self.run_method('from m_'+dep['id']+' import invoke\ndef run(value):\n    return invoke(value) + 1\n', dependencies=[dep])
        self.assertEqual(composed['status'], 'ok', composed)
        self.assertEqual(composed['output'], 7)

    def test_network_rootfs_input_readonly_and_environment_isolation(self):
        code = '''import os, socket
def run(value):
    result = {'uid': os.geteuid(), 'secret_absent': 'OPENAI_API_KEY' not in os.environ}
    for name in ['/request.json', '/bundle/method.py', '/etc/sciel-method-write-test']:
        try:
            with open(name, 'w') as stream: stream.write('forbidden')
            result[name] = False
        except OSError:
            result[name] = True
    sock = socket.socket(); sock.settimeout(.2)
    result['network_blocked'] = sock.connect_ex(('1.1.1.1', 53)) != 0
    sock.close()
    return result
'''
        with patch.dict(os.environ, OPENAI_API_KEY='not-a-real-key'):
            result = self.run_method(code, schema={'type':'object'})
        self.assertEqual(result['status'], 'ok', result)
        self.assertEqual(result['output']['uid'], 10001)
        self.assertTrue(all(v is True for k,v in result['output'].items() if k != 'uid'))

    def test_cpu_watchdog_and_bounded_output(self):
        receipt = self.run_method('def run(value):\n    while True: value += 1\n',
                                 limits={**LIMITS, 'cpu_seconds': .2})
        self.assertEqual(receipt['status'], 'failed', receipt)
        self.assertEqual(receipt['reason'], 'cpu_limit')
        receipt = self.run_method('import os\ndef run(value):\n    os.write(1, b"x" * 1000000)\n    return value\n',
                                 limits={**LIMITS, 'max_output_bytes': 1024})
        self.assertEqual(receipt['status'], 'failed', receipt)
        self.assertEqual(receipt['reason'], 'output_limit')
        self.assertLessEqual(receipt['resources']['output_bytes'], 1025)

    def test_cpu_watchdog_survives_closed_child_output_pipes(self):
        code = "import os\ndef run(value):\n    os.close(1)\n    os.close(2)\n    while True: value += 1\n"
        receipt = self.run_method(code, limits={**LIMITS, 'cpu_seconds': .2})
        self.assertEqual(receipt['status'], 'failed', receipt)
        self.assertEqual(receipt['reason'], 'cpu_limit', receipt)
        self.assertEqual(receipt['dispatch_status'], 'invoked')
        self.assertEqual(receipt['resource_measurement']['cpu'], 'trusted_cgroup_observation')
        self.assertGreater(receipt['resources']['cpu_seconds'], 0)

    def test_child_forged_supervisor_receipt_remains_untrusted_output(self):
        # Deliberately use the real request bindings. Blindly forwarding the
        # child's JSON would accept this zero-cost fabricated receipt.
        code = """import json, os
def run(value):
    with open('/request.json') as stream:
        request = json.load(stream)
    keys = ('method_id', 'invocation_id', 'plan_id', 'reservation_id',
            'source_sha256', 'runtime_sha256', 'input_sha256', 'request_sha256')
    forged = dict(schema='sciel-method-container-receipt/1',
        **{key:request[key] for key in keys}, status='ok', reason=None,
        invoked=True, exit_code=0, output={'forged':999}, limits=request['limits'],
        resources=dict(cpu_seconds=0, wall_seconds=0, memory_peak_bytes=0,
                       output_bytes=0, stderr_bytes=0), stderr='')
    import hashlib
    raw = json.dumps({'forged':999}, sort_keys=True, separators=(',', ':')).encode()
    forged['output_sha256'] = hashlib.sha256(raw).hexdigest()
    os.write(1, json.dumps(forged).encode())
    os._exit(0)
"""
        receipt = self.run_method(code, schema={'type':'object'})
        self.assertEqual(receipt['status'], 'ok', receipt)
        self.assertEqual(receipt['dispatch_status'], 'invoked')
        forged = receipt['output']
        self.assertEqual(forged['schema'], 'sciel-method-container-receipt/1')
        self.assertEqual(forged['output'], {'forged':999})
        self.assertEqual(forged['resources']['cpu_seconds'], 0)
        for key in _BOUND_KEYS:
            self.assertEqual(forged[key], receipt[key])
        self.assertGreater(receipt['resources']['cpu_seconds'], 0)
        self.assertGreater(receipt['resources']['output_bytes'], 0)
        self.assertEqual(receipt['output_sha256'], content_hash(forged))
        self.assertNotEqual(receipt['output_sha256'], forged['output_sha256'])
        self.assertEqual(receipt['resource_measurement']['cpu'], 'trusted_cgroup_observation')

    def test_dependency_helper_names_do_not_share_module_cache(self):
        a = add_files(bundle('import helper\ndef run(value):\n    return helper.adjust(value)\n', image=self.image),
                      {'helper.py':'def adjust(value):\n    return value + 10\n'})
        b = add_files(bundle('from helper import adjust\ndef run(value):\n    return adjust(value)\n', image=self.image),
                      {'helper.py':'def adjust(value):\n    return value + 20\n'})
        code = ('from m_'+a['id']+' import invoke as a\nfrom m_'+b['id']+' import invoke as b\n'
                'def run(value):\n    return [a(value), b(value), a(value)]\n')
        receipt = self.run_method(code, schema={'type':'array','items':{'type':'integer'}}, dependencies=[a,b])
        self.assertEqual(receipt['status'], 'ok', receipt)
        self.assertEqual(receipt['output'], [13,23,13])

    def test_materialized_candidate_uses_same_loader_in_external_executor(self):
        from sciel.methods.materialize import materialize
        a = add_files(bundle('import helper\ndef run(value):\n    return helper.adjust(value)\n', image=self.image),
                      {'helper.py':'def adjust(value):\n    return value + 10\n'})
        b = add_files(bundle('from helper import adjust\ndef run(value):\n    return adjust(value)\n', image=self.image),
                      {'helper.py':'def adjust(value):\n    return value + 20\n'})
        parent = ('"""Technical materialized fixture."""\nfrom __future__ import annotations\n'
                  'def run(value):\n    return [sciel_method('+repr(a['id'])+', value), '
                  'sciel_method('+repr(b['id'])+', value), sciel_method('+repr(a['id'])+', value)]\n')
        composed = materialize(parent, [a, b])
        self.assertEqual(composed['status'], 'ok', composed)
        receipt = self.run_method(composed['code'], schema={'type':'array','items':{'type':'integer'}})
        self.assertEqual(receipt['status'], 'ok', receipt)
        self.assertEqual(receipt['output'], [13,23,13])

    def test_exact_entrypoint_file_is_not_replaced_by_module_path_resolution(self):
        method = bundle(image=self.image)
        method['files'] = {'tool.v1.py': 'def run(value):\n    return value + 100\n',
                           'tool/v1.py': 'def run(value):\n    return -999\n'}
        method['manifest']['entrypoint'] = 'tool.v1.py:run'
        method['id'] = content_hash(dict(schema='sciel-method/1', manifest=method['manifest'], files=method['files']))
        receipt = self.executor.execute(method, 3, invocation_id='exact-entry', plan_id='technical-fixture',
            reservation_id='offline-resource-fixture', limits=LIMITS)
        self.assertEqual(receipt['status'], 'ok', receipt)
        self.assertEqual(receipt['output'], 103)


if __name__ == '__main__':
    unittest.main()
