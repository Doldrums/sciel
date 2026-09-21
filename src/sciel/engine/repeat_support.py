"""H15 integration boundary: canonical task, search safe points and explicit export.

Imported only when repeat promotion is enabled. Legacy search observations,
population fitness and research outcomes remain independent of repeat receipts.
"""
from __future__ import annotations

from functools import partial
import hashlib
import json
import os
from pathlib import Path

CANONICAL_TASK_HASH = '86fe9a3ea046a1714b9e67f67b8ca916addaaadcf998e1a5260578f6c4e2489a'
INITIAL_SHA = 'd432ec2da7c307180a4dafff1be1ca47cf37be07c40b12c5fedae69417fd639e'
EVALUATOR_SHA = 'e996baad4acf38ebe8faab8fb2e5d91bb691c7255e26257bdaa4d8efc9494f85'
BODY_TASK_HASH = '6aea22dd864e8788a5e58a4aac257bbd4451ba9cfa4a6f4f97d487f39665420a'
BODY_INITIAL_SHA = '3089f4e36bf787275ade5c3c24d4836ad8dc26e409185c0679c3377f4ce9c5f9'
TRACED_BODY_TASK_HASH = '0726f12ed2a03a754a0a25049268a37e2ad7ff9773db66d7113305383c122095'
TRACED_BODY_INITIAL_SHA = '2f4bd6cc96eb780a73f547f1e24f28961a3b3a70478a04e4c6fc4a864e76e34d'
TRACED_EXECUTION_PROFILE = 'paired_restart_trace/1'
OPTIMIZER_TRACED_EXECUTION_PROFILE = 'paired_restart_trace/2'
OPTIMIZER_TRACED_TASK_HASH = 'd1b5422d2a943f4757b7db09381a8e49d6d138f6ec9994fdc834630fba4c3e0e'
OPTIMIZER_TRACED_INITIAL_SHA = '93bc6ed5658a2a299c660de06d5dc4bf0e623f1d8a258aa2504b2a8a5ad98cae'

# Old observations remain readable under their original metric identity. These
# hashes do not authorize executing an old task with the current repeat mode.
LEGACY_GEOMETRY_EVALUATOR_SHA = '6ca0714c55f7101c3145b030c3a5e754ec94401fabedd4952fe319572edd3ead'
HISTORICAL_NATIVE_GEOMETRY_TASK_EVALUATORS = dict.fromkeys((
    '34f919d0c1937a79ce039bcdf75b863443f1f63edb77734bc7b506ae17f91aeb',
    '149a3887108c5ff96ac783036ea62b318433ec8a0104f8bcf6db9e31253297ef',
    '5f65a12e159ced391a1635833cb2d24f60813763e5a6e237de228eea7d68987e',
    'a936b3365f80fcff7da567c5985e5d327295c66b75ce3144253405f1ebde6481',
), LEGACY_GEOMETRY_EVALUATOR_SHA)
# Before the shared importable-candidate loader, this evaluator already used
# binary64 constraint checks. Its recorded observations retain that exact
# evaluator identity; these entries never authorize a new repeat execution.
HISTORICAL_NATIVE_GEOMETRY_TASK_EVALUATORS.update(dict.fromkeys((
    'a597f167ece11256458997773abbf009ce569b06518f88b17325eb585289d614',
    'e3549f4cb473e8ae8bc319680f9f12a9a761f6520ee31c182682bb17b7acbc0e',
    '976d9ae145dd3cd7c897bfde304c913e6c16b4e4fc0391a4137032191f823926',
    '3fc57bfd26a6164671df0bf4fe3013387a1f9dda5b603b806f82c1b598fbf944',
), '44366fd28850b02da1303a7f1616b0e724f28b837d6ea506b26d4a91096523f4'))
# Public task instructions changed the task hash, not its evaluator or seed.
# Keep every previously recorded metric identity readable without authorizing
# execution under a stale task descriptor.
HISTORICAL_NATIVE_GEOMETRY_TASK_EVALUATORS.update(dict.fromkeys((
    '4a2bfdd6e0eb925e88560bbde9e3f6fab9b63f7e9058c9966393476ee7adff60',
    'fe5a424943169db9b29798713ae0465c4ed02c4e7bc215eff2e3edf0457a17c8',
    '8dc9f2f9e1fa842e22fc68c5642691aba41c8fadec7128c4430b63cf32ba3f67',
    '85b24214a299c3ef6e01c127a4e7875774d2902741d167be839ff445f462856d',
), EVALUATOR_SHA))
HISTORICAL_CIRCLE_TASK_HASHES = (
    '899687d940b3dc83c43db7be6b9a78b3e3c2119fcbfff069f7126e1bdd77f228',
    'd85ebdd86dc5dfccf3663624e74c4a063c7e2b03b1e0595291d261b64fc6c2c9',
    *HISTORICAL_NATIVE_GEOMETRY_TASK_EVALUATORS,
)
KNOWN_CIRCLE_TASK_HASHES = (CANONICAL_TASK_HASH, BODY_TASK_HASH, TRACED_BODY_TASK_HASH, OPTIMIZER_TRACED_TASK_HASH,
                          *HISTORICAL_CIRCLE_TASK_HASHES)
NATIVE_GEOMETRY_TASK_EVALUATORS = {**HISTORICAL_NATIVE_GEOMETRY_TASK_EVALUATORS,
    CANONICAL_TASK_HASH: EVALUATOR_SHA,
    BODY_TASK_HASH: EVALUATOR_SHA, TRACED_BODY_TASK_HASH: EVALUATOR_SHA,
    OPTIMIZER_TRACED_TASK_HASH: EVALUATOR_SHA}


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def validate_registered_native_task(task):
    """Preflight known native evaluators using file hashes, without execution.

    Unregistered/third-party evaluators are outside this registry. A registered
    evaluator must retain an explicitly reviewed complete task identity.
    """
    evaluator_sha = _sha(Path(task.evaluator_path).read_bytes())
    if evaluator_sha not in set(NATIVE_GEOMETRY_TASK_EVALUATORS.values()):
        return None
    if NATIVE_GEOMETRY_TASK_EVALUATORS.get(task.task_hash) != evaluator_sha:
        raise ValueError('unregistered native evaluator/task tuple')
    return {'task_hash': task.task_hash, 'evaluator_sha256': evaluator_sha}


def circle_initial_source_policy(cfg):
    """Resolve the explicit execution profile without running task code."""
    circle = cfg.get('circle_generation', {})
    if 'execution_profile' in circle:
        if (circle.get('enabled') is not True or
                type(circle['execution_profile']) is not str or
                circle['execution_profile'] not in (TRACED_EXECUTION_PROFILE, OPTIMIZER_TRACED_EXECUTION_PROFILE)):
            raise ValueError('unknown or disabled Circle execution profile')
        from ..hypothesis.circle_source_family import builder_for_profile
        return builder_for_profile(circle['execution_profile']).OPERATION
    return 'circle_restart_body/1' if circle.get('enabled') else None


def traced_execution_policy(initial_source_policy):
    """Identity extension for same-source, repeat-only diagnostic capture.

    Absent on legacy paths, including their draw keyword arguments. This policy
    does not assert diagnostics succeeded or change numerical admission.
    """
    if initial_source_policy in (None, 'circle_restart_body/1'):
        return {}
    if initial_source_policy not in ('circle_restart_body_traced/1', 'circle_restart_body_traced/2'):
        raise ValueError('unknown Circle initial source policy')
    from ..hypothesis.circle_source_family import builder_for_operation
    builder = builder_for_operation(initial_source_policy)
    identity = builder.identity()
    return {'execution_profile': builder.EXECUTION_PROFILE,
        'restart_trace': {'enabled': True, 'capture': 'paired_draws_only',
            'ordinary_evaluation': 'disabled',
            'scaffold_sha256': identity['common_scaffold_sha256'],
            'collector_sha256': identity['dependencies']['collector_sha256']}}


def validate_circle_task(task, *, initial_source_policy=None):
    """Accept one pinned task/seed pair, never an arbitrary named Circle task.

    The opt-in body mode changes only the initial source. Its task hash still
    binds the complete canonical YAML, evaluator and assets. A generated warm
    start is not admitted here; checkpoint restore owns continuation sources.
    """
    from ..eval.registry import load_task
    expected_task, expected_source = CANONICAL_TASK_HASH, INITIAL_SHA
    if initial_source_policy == 'circle_restart_body/1':
        from ..hypothesis import circle_restart_body as builder
        expected_task, expected_source = BODY_TASK_HASH, BODY_INITIAL_SHA
    elif initial_source_policy is not None:
        if initial_source_policy not in ('circle_restart_body_traced/1', 'circle_restart_body_traced/2'):
            raise ValueError('unknown Circle initial source policy')
        from ..hypothesis.circle_source_family import builder_for_operation
        builder = builder_for_operation(initial_source_policy)
        expected_task, expected_source = ((TRACED_BODY_TASK_HASH, TRACED_BODY_INITIAL_SHA)
            if initial_source_policy == 'circle_restart_body_traced/1' else
            (OPTIMIZER_TRACED_TASK_HASH, OPTIMIZER_TRACED_INITIAL_SHA))
    if initial_source_policy is not None:
        if _sha(builder.source_for().encode()) != expected_source:
            raise ValueError('Circle body control differs from pinned source')
    current = load_task(task.root)
    if initial_source_policy in ('circle_restart_body_traced/1', 'circle_restart_body_traced/2') and task != current:
        raise ValueError('traced Circle task metadata differs from pinned task files')
    initial = Path(task.initial_program_path).read_bytes()
    evaluator = Path(task.evaluator_path).read_bytes()
    if (task.qualified_name != 'oe/circle_packing' or current.task_hash != expected_task
            or task.task_hash != current.task_hash or _sha(initial) != expected_source
            or _sha(evaluator) != EVALUATOR_SHA):
        raise ValueError('repeat_promotion requires the pinned canonical Circle task and initial source')
    return initial, evaluator


def create(config, task, run_dir, python, *, initial_source_policy=None):
    from ..eval.repeat_draw import adapter_identity, draw
    from .repeat_promotion import CONTRACT, RepeatPromotion

    runtime = config.get('runtime', {'backend': 'native'})
    initial, evaluator = validate_circle_task(task, initial_source_policy=initial_source_policy)
    trace_policy = traced_execution_policy(initial_source_policy)
    # Runtime/source hashes belong to this enabled mode's identity, not to the
    # default Engine contract or any historical experimental manifest.
    here = Path(__file__).resolve().parent
    identity = {'contract': CONTRACT, 'task': task.qualified_name, 'task_hash': task.task_hash,
        'canonical_initial_source_sha256': _sha(initial), 'task_evaluator_sha256': _sha(evaluator),
        'adapter': adapter_identity(python=python, runtime=runtime),
        'integration_sha256': {name: _sha((here / name).read_bytes()) for name in
                              ('repeat_support.py', 'repeat_promotion.py', 'repeat_journal.py',
                               'loop.py', 'checkpoint.py', '../config.py', '../run_log.py')},
        **trace_policy}
    return RepeatPromotion(run_dir, config, identity,
                           partial(draw, python=python, runtime=runtime,
                                   expected_identity=identity['adapter'],
                                   **({'restart_trace': True} if trace_policy else {})))


def assert_search_anchor(log_path, anchor):
    """Refuse to replay a post-checkpoint search step when restoring H15 state."""
    if not isinstance(anchor, dict):
        raise ValueError('repeat promotion requires a bound search journal anchor')
    offset = anchor.get('byte_offset')
    if type(offset) is not int or offset < 1:
        raise ValueError('invalid repeat promotion search journal anchor')
    data = Path(log_path).read_bytes()
    if len(data) < offset or _sha(data[:offset]) != anchor.get('prefix_sha256'):
        raise ValueError('repeat promotion search journal prefix differs')
    if not data.endswith(b'\n'):
        raise ValueError('incomplete search journal tail')
    allowed = {'checkpoint', 'run_end', 'run_start', 'checkpoint_restore'}
    for line in data[offset:].splitlines():
        item = json.loads(line)
        if not isinstance(item, dict) or item.get('event') not in allowed:
            raise ValueError('uncheckpointed search activity; refusing repeat/search replay')


def durable_write(path, data):
    """Replace an export only after all bytes are durable; fsync its directory."""
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def export(engine):
    observed = None
    if engine.db.pop_best_id is not None:
        p = engine.db.get(engine.db.pop_best_id)
        observed = p
    result = engine.repeat_promotion.export(observed)
    confirmed = result.get('confirmed_output_incumbent')
    path = Path(engine.run_dir) / 'confirmed_program.py'
    if confirmed is not None:
        code = confirmed['code']
        if _sha(code.encode()) != confirmed['source_sha256']:
            raise ValueError('confirmed output source digest differs')
        durable_write(path, code.encode())
        # The retained source lives in the repeat ledger; the user-facing summary
        # links to its explicit artifact rather than silently replacing pop_best.
        result = {**result, 'confirmed_output_incumbent':
                  {k: v for k, v in confirmed.items() if k != 'code'}}
        result['confirmed_output_incumbent']['path'] = str(path.resolve())
    elif path.exists():
        # A stale export from another state must not look like a current result.
        raise ValueError('unconfirmed state has an existing confirmed_program.py')
    durable_write(Path(engine.run_dir) / 'repeat_promotion.json',
                  (json.dumps(result, indent=2, allow_nan=False) + '\n').encode())
    return result
