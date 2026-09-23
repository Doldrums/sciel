"""Closed, typed provenance for native calls and their delivered derivatives.

Hashes commit to evidence; only the native resolvers authenticate that evidence.
A provenance root does not claim the producer's entire response was displayed.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import os
import stat

from .types import InferenceSpec, content_hash, require_hash, require_identifier, sha256_bytes

SCHEMA = 'sciel-adaptation-ancestry/2'
FIELDS = {'node_id', 'kind', 'run_id', 'owner', 'owner_scope', 'identity', 'proof',
          'episode_inputs', 'auxiliary_inputs'}


def need(condition, message):
    if not condition:
        # Delayed import keeps the old ancestry/dataset import boundary intact.
        from .ancestry import ResolutionUnknown
        raise ResolutionUnknown(message)


def _ids(values, label):
    need(type(values) is list and all(type(v) is str for v in values)
         and values == sorted(set(values)), 'invalid ' + label)
    return values


def _commit(value):
    need(type(value) is dict and set(value) == {'segment_id', 'event_seq', 'event_sha256'},
         'invalid auxiliary commit binding')
    require_identifier(value['segment_id'], 'commit segment')
    require_hash(value['event_sha256'], 'commit event')
    need(type(value['event_seq']) is int and value['event_seq'] > 0, 'invalid commit sequence')


def validate_node(value):
    need(type(value) is dict and set(value) == FIELDS, 'invalid auxiliary binding')
    require_hash(value['node_id'], 'auxiliary node')
    need(value['node_id'] == content_hash({k: v for k, v in value.items() if k != 'node_id'}),
         'auxiliary binding hash differs')
    for name in ('run_id', 'owner'):
        require_identifier(value[name], name)
    need(value['owner_scope'] in ('learner', 'shared'), 'invalid auxiliary owner scope')
    need(value['owner_scope'] != 'shared' or value['owner'] == 'base',
         'shared auxiliary producer is not the actual base owner')
    _ids(value['episode_inputs'], 'auxiliary episode inputs')
    for ident in _ids(value['auxiliary_inputs'], 'auxiliary inputs'):
        require_hash(ident, 'auxiliary input')
    need(type(value['identity']) is dict and type(value['proof']) is dict,
         'missing typed auxiliary identity/proof')
    if value['kind'] == 'llm_call':
        identity, proof = value['identity'], value['proof']
        need(set(identity) == {'call_id', 'role', 'segment_id'}, 'invalid call identity')
        require_identifier(identity['call_id'], 'auxiliary call')
        require_identifier(identity['segment_id'], 'auxiliary segment')
        need(identity['role'] in ('G', 'repair', 'C', 'R', 'research_tool'),
             'unsupported auxiliary call role')
        need(value['owner_scope'] != 'shared' or identity['role'] in ('C', 'R'),
             'only base conclusion/reflection is a shared call')
        hashes = {'request_record_sha256', 'response_record_sha256', 'prompt_sha256',
                  'response_text_sha256', 'inference_spec_sha256'}
        extra = {'inference_spec', 'adapter_revision_id'} if 'inference_spec' in proof else set()
        need(set(proof) == hashes | {'dispatch_event_seq', 'commit_event'} | extra,
             'invalid call proof')
        if extra:
            spec = InferenceSpec.from_dict(proof['inference_spec'])
            need(spec.adapter_id is not None and spec.role in ('G', 'repair')
                 and spec.role == identity['role'] and spec.owner == value['owner']
                 and value['owner_scope'] == 'learner'
                 and content_hash(spec.to_dict()) == proof['inference_spec_sha256'],
                 'invalid adapted call specification')
            require_hash(proof['adapter_revision_id'], 'call adapter revision')
            need(proof['adapter_revision_id'] in value['auxiliary_inputs'], 'adapted call lost its parameter edge')
        for key in hashes:
            require_hash(proof[key], key)
        need(type(proof['dispatch_event_seq']) is int and proof['dispatch_event_seq'] > 0,
             'invalid auxiliary dispatch sequence')
        _commit(proof['commit_event'])
    elif value['kind'] == 'lesson_revision':
        from .kb_ancestry import validate_node as validate_kb_node
        validate_kb_node(value)
    elif value['kind'] in ('tool_version', 'tool_invocation', 'research_card'):
        from .tool_ancestry import validate_node as validate_tool_node
        validate_tool_node(value)
    elif value['kind'] == 'adapter_revision':
        from .adapter_ancestry import validate_node as validate_adapter_node
        validate_adapter_node(value)
    else:
        need(False, 'unsupported auxiliary kind: ' + str(value['kind']))
    return deepcopy(value)


@dataclass
class GraphFragment:
    episodes: dict = field(default_factory=dict)
    auxiliary: dict = field(default_factory=dict)
    roots: set = field(default_factory=set)
    episode_roots: set = field(default_factory=set)

    def merge(self, other):
        for name in ('episodes', 'auxiliary'):
            target = getattr(self, name)
            for ident, row in getattr(other, name).items():
                need(ident not in target or target[ident] == row,
                     'conflicting ' + name + ' identity')
                if ident not in target:
                    target[ident] = deepcopy(row)
        self.roots.update(other.roots)
        self.episode_roots.update(other.episode_roots)
        return self

    def node(self, *, kind, run_id, owner, owner_scope, identity, proof):
        """Wrap this complete input graph in one authenticated producer node."""
        value = dict(kind=kind, run_id=run_id, owner=owner, owner_scope=owner_scope,
                     identity=deepcopy(identity), proof=deepcopy(proof),
                     episode_inputs=sorted(self.episode_roots), auxiliary_inputs=sorted(self.roots))
        value['node_id'] = content_hash(value)
        validate_node(value)
        result = deepcopy(self)
        ident = value['node_id']
        need(ident not in result.auxiliary or result.auxiliary[ident] == value,
             'conflicting auxiliary producer')
        result.auxiliary[ident] = value
        result.roots = {ident}
        result.episode_roots = set()
        return result


def validate_graph(bindings, prompt_roots, parameter_roots, episode_ids):
    need(type(bindings) is list and bindings, 'empty auxiliary graph must use ancestry/1')
    nodes = {}
    identities = set()
    for raw in bindings:
        value = validate_node(raw)
        native = (value['kind'], value['run_id'], content_hash(value['identity']))
        need(native not in identities, 'conflicting native auxiliary identity')
        identities.add(native)
        nodes[value['node_id']] = value
    need([v['node_id'] for v in bindings] == sorted(nodes), 'duplicate or unordered auxiliary nodes')
    roots = set(_ids(prompt_roots, 'prompt auxiliary roots')) | set(
        _ids(parameter_roots, 'parameter auxiliary roots'))
    need(roots <= set(nodes), 'missing auxiliary root')
    complete, active = set(), set()

    def visit(ident):
        need(ident in nodes, 'missing auxiliary dependency')
        need(ident not in active, 'cyclic auxiliary ancestry')
        if ident in complete:
            return
        active.add(ident)
        node = nodes[ident]
        need(set(node['episode_inputs']) <= set(episode_ids), 'missing auxiliary episode dependency')
        for dependency in node['auxiliary_inputs']:
            visit(dependency)
        active.remove(ident)
        complete.add(ident)

    for ident in roots:
        visit(ident)
    need(complete == set(nodes), 'orphaned auxiliary ancestry')
    for node in nodes.values():
        validate_call_revision(node, nodes)
    return deepcopy(bindings)


def validate_call_revision(node, nodes):
    if node['kind'] != 'llm_call' or 'adapter_revision_id' not in node['proof']:
        return
    spec = InferenceSpec.from_dict(node['proof']['inference_spec'])
    revision = nodes.get(node['proof']['adapter_revision_id'])
    need(revision is not None and revision['kind'] == 'adapter_revision'
         and revision['identity'] == dict(owner=spec.owner, adapter_id=spec.adapter_id,
                                         adapter_sha256=spec.adapter_sha256)
         and revision['proof']['dataset_sha256'] == spec.dataset_sha256,
         'adapted call differs from its parameter revision')


def donor_auxiliary_ids(resolution, owner):
    """Producer donors, not a replacement for following the full input graph."""
    return [node['node_id'] for node in resolution.get('auxiliary_bindings', [])
            if node['owner_scope'] == 'learner' and node['owner'] != owner]


def call_records(native, call_id):
    """Read the exact native pair; caller must bind it to a typed host outcome."""
    require_identifier(call_id, 'auxiliary call')
    request = native.transcripts.get((call_id, 'request'))
    response = native.transcripts.get((call_id, 'response'))
    need(type(request) is dict and type(response) is dict, 'missing auxiliary call pair')
    spec = InferenceSpec.from_dict(request.get('inference_spec'))
    need(all(row.get('call_id') == call_id and row.get('run_id') == native.run_id
             and row.get('segment_id') == request.get('segment_id')
             and row.get('inference_spec') == spec.to_dict() for row in (request, response)),
         'auxiliary call run/segment/spec differs')
    for key in ('base_model', 'base_revision', 'tokenizer_revision', 'chat_template_sha256'):
        need(getattr(spec, key) == native.config['llm']['local_inference'][key],
             'auxiliary model identity differs')
    cutoff = native.positions.get((request.get('segment_id'), request.get('journal_event_seq')))
    need(cutoff is not None, 'auxiliary call has no committed dispatch cutoff')
    texts = []
    for suffix, row, hash_key in (('prompt', request, 'prompt_sha256'),
                                  ('response', response, 'response_sha256')):
        path = native.root / 'artifacts/llm_calls' / (call_id + '.' + suffix + '.txt')
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            before = os.fstat(fd)
            need(stat.S_ISREG(before.st_mode) and before.st_size <= 8 * 1024 * 1024,
                 'auxiliary text is oversized or not regular')
            with os.fdopen(fd, 'rb', closefd=False) as stream:
                raw = stream.read(8 * 1024 * 1024 + 1)
            after = os.fstat(fd)
            signature = lambda value: (value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
            need(signature(before) == signature(after) and len(raw) == before.st_size,
                 'auxiliary text changed during read')
        finally:
            os.close(fd)
        need(sha256_bytes(raw) == row.get(hash_key), 'auxiliary ' + suffix + ' hash differs')
        texts.append(raw.decode('utf-8'))
    return request, response, texts[0], texts[1], spec, cutoff


def authenticate_call(native, call_id, commitment, inputs):
    """Seal an input graph after its role handler authenticated full delivery.

    The caller additionally proves the host commitment's role-specific link to
    this response (proposal, auxiliary result, or committed generation attempt).
    """
    request, response, _, _, spec, cutoff = call_records(native, call_id)
    position = native.positions.get((commitment.get('segment_id'), commitment.get('event_seq')))
    need(position is not None and native.events[position] == commitment and position > cutoff,
         'auxiliary acceptance/rejection is not a committed post-dispatch event')
    need(commitment.get('run_id') == native.run_id, 'foreign auxiliary commitment')
    adapted = {}
    if spec.adapter_id is not None:
        need(spec.role in ('G', 'repair'), 'adapted auxiliary roles are unsupported')
        candidates = [ident for ident in inputs.roots if inputs.auxiliary[ident]['kind'] == 'adapter_revision'
                      and inputs.auxiliary[ident]['run_id'] == native.run_id
                      and inputs.auxiliary[ident]['identity'] == dict(owner=spec.owner,
                          adapter_id=spec.adapter_id, adapter_sha256=spec.adapter_sha256)]
        need(len(candidates) == 1, 'adapted auxiliary producer requires a retained typed adapter revision proof')
        adapted = dict(inference_spec=spec.to_dict(), adapter_revision_id=candidates[0])
    return inputs.node(kind='llm_call', run_id=native.run_id, owner=spec.owner,
        owner_scope='shared' if spec.role in ('C', 'R') and spec.owner == 'base' else 'learner',
        identity=dict(call_id=call_id, role=spec.role, segment_id=request['segment_id']),
        proof=dict(request_record_sha256=content_hash(request), response_record_sha256=content_hash(response),
                   prompt_sha256=request['prompt_sha256'], response_text_sha256=response['response_sha256'],
                   inference_spec_sha256=content_hash(spec.to_dict()), dispatch_event_seq=request['journal_event_seq'],
                   commit_event=dict(segment_id=commitment['segment_id'], event_seq=commitment['event_seq'],
                                     event_sha256=content_hash(commitment)), **adapted))
