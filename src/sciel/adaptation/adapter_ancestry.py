"""Read-only retained parameter evidence; paths locate, hashes identify.

An adapter node's run_id is its native observation scope, not a training run.
No active routing, model loading, or new registry/receipt format is involved.
"""
from copy import deepcopy
from pathlib import Path

from .auxiliary_ancestry import GraphFragment, need
from .types import InferenceSpec, content_hash, require_hash, require_identifier, sha256_bytes


def validate_node(node):
    identity, proof = node['identity'], node['proof']
    need(node['owner_scope'] == 'learner' and set(identity) == {'owner', 'adapter_id', 'adapter_sha256'}
         and identity['owner'] == node['owner'], 'invalid adapter revision identity')
    require_identifier(identity['adapter_id'], 'adapter id')
    require_hash(identity['adapter_sha256'], 'adapter bundle')
    from . import corpus_policy as C
    policy = C.selected(proof)
    need(set(proof) == ({'registry_manifest_sha256', 'training_receipt_sha256', 'dataset_sha256',
         'dataset_manifest_sha256', 'training_schema', 'parent_revision_id'} | ({'corpus_policy'} if policy is not None else set())), 'invalid adapter revision proof')
    for key in ('registry_manifest_sha256', 'training_receipt_sha256', 'dataset_sha256', 'dataset_manifest_sha256'):
        require_hash(proof[key], key)
    from .training_evidence import HF_SCHEMA, DELTA_SCHEMA
    need(proof['training_schema'] in (HF_SCHEMA, DELTA_SCHEMA), 'unknown adapter training schema')
    parent = proof['parent_revision_id']
    if parent is not None:
        require_hash(parent, 'parameter parent')
        need(parent in node['auxiliary_inputs'], 'adapter lost its direct parent revision')


def context(value):
    return value if isinstance(value, AdapterAncestry) else AdapterAncestry(value)


class AdapterAncestry:
    def __init__(self, sources=None):
        self.sources = sources or {}
        need(type(self.sources) is dict and set(self.sources) <= {'registry', 'datasets'},
             'unsupported ancestry sources')
        self.active = set()
        # One immutable retained-evidence operation, never a process/global
        # verdict. New independent verification must construct a new context.
        self._inspections = {}
        self._revisions = {}

    def _inspect(self, registry, owner, adapter_id):
        need(not any(key[:3] == ('revision', owner, adapter_id) for key in self.active),
             'cyclic adapter revision ancestry')
        path = registry._path(owner, adapter_id) / 'registry_manifest.json'
        raw = path.read_bytes()
        key = (str(registry.root.absolute()), owner, adapter_id, sha256_bytes(raw), frozenset(self.active))
        if key not in self._inspections:
            revision = registry.inspect(owner, adapter_id)
            need(path.read_bytes() == raw, 'adapter manifest changed during inspection')
            self._inspections[key] = revision
        return deepcopy(self._inspections[key])

    def dataset(self, dataset_hash, manifest_hash):
        from .dataset import verify_dataset
        key = ('dataset', dataset_hash, manifest_hash)
        need(key not in self.active, 'cyclic adapter dataset ancestry')
        self.active.add(key)
        try:
            matches = []
            for paths in self.sources.get('datasets', []):
                need(type(paths) is dict and set(paths) == {'dataset', 'manifest'}, 'invalid ancestry dataset paths')
                raw = Path(paths['manifest']).read_bytes()
                if sha256_bytes(raw) != manifest_hash:
                    continue
                rows, manifest = verify_dataset(paths['dataset'], paths['manifest'], ancestry_sources=self)
                need(Path(paths['manifest']).read_bytes() == raw, 'ancestry dataset manifest changed')
                need(manifest['dataset_sha256'] == dataset_hash, 'training dataset hash differs')
                matches.append((rows, manifest))
            need(matches and all(match == matches[0] for match in matches),
                 'training corpus/manifest is missing or ambiguous')
            return matches[0]
        finally:
            self.active.remove(key)

    def resolve(self, spec, scope, *, typed):
        from .registry import AdapterRegistry
        if typed:
            require_identifier(scope, 'adapter observation run')
        need(self.sources.get('registry'), 'adapter ancestry requires retained typed adapter revision proof sources')
        registry = AdapterRegistry(self.sources['registry'])
        revision = self._inspect(registry, spec.owner, spec.adapter_id)
        for key in ('adapter_sha256', 'dataset_sha256', 'serving_alias', 'base_model',
                    'base_revision', 'tokenizer_revision', 'chat_template_sha256'):
            need(revision[key] == getattr(spec, key), 'inference adapter differs from registered revision')
        return self._revision(registry, revision, scope, typed=typed)

    def _revision(self, registry, revision, scope, *, typed):
        key = ('revision', revision['owner'], revision['adapter_id'], revision['adapter_sha256'])
        need(key not in self.active, 'cyclic adapter revision ancestry')
        memo = (str(registry.root.absolute()), content_hash(revision), scope, typed, frozenset(self.active))
        if memo not in self._revisions:
            result = self._build_revision(registry, revision, scope, typed=typed)
            self._revisions[memo] = deepcopy(result)
            return result
        return deepcopy(self._revisions[memo])

    def _build_revision(self, registry, revision, scope, *, typed):
        from .ancestry import NativeAncestry, _binding, validate_producer_binding, validate_resolution
        from .training_evidence import DELTA_SCHEMA, read_training_evidence, unqualified_parent
        from .visibility import validate_capsule
        key = ('revision', revision['owner'], revision['adapter_id'], revision['adapter_sha256'])
        need(key not in self.active, 'cyclic adapter revision ancestry')
        self.active.add(key)
        try:
            directory = registry._path(revision['owner'], revision['adapter_id'])
            raw = (directory / 'training_receipt.json').read_bytes()
            need(sha256_bytes(raw) == revision['files']['training_receipt.json'], 'training receipt hash differs')
            training = read_training_evidence(directory)
            need((directory / 'training_receipt.json').read_bytes() == raw,
                 'training receipt changed during verification')
            need(training['schema'] != DELTA_SCHEMA or not unqualified_parent(training),
                 'Delta parameter ancestry is unresolved until native/run evidence is qualified')
            rows, manifest = self.dataset(revision['dataset_sha256'], training.get('dataset_manifest_sha256'))
            from . import corpus_policy as C
            C.same(revision, training, manifest)
            need(manifest['owner'] == revision['owner'] and training['status'] == 'succeeded',
                 'invalid training ownership/status')
            updates = training.get('episode_ids_by_update')
            need(type(updates) is list and updates and all(type(group) is list and group
                 and all(type(episode) is str for episode in group) for group in updates)
                 and type(training.get('optimizer_steps')) is int and training['optimizer_steps'] == len(updates)
                 and len(training.get('updates', [])) == len(updates), 'training receipt lacks completed episode updates')
            need(all(type(update) is dict and type(update.get('step')) is int and update['step'] == i + 1
                 and type(update.get('loss_tokens')) is int and update['loss_tokens'] > 0
                 for i, update in enumerate(training['updates'])), 'training completed updates differ from plan')
            selected = {episode for group in updates for episode in group}
            by_id = {row['episode_id']: row for row in rows}
            need(len(by_id) == len(rows) and selected <= set(by_id), 'training update episode identity is absent/ambiguous')
            # Verified oracle rows require their portable corpus on subsequent
            # generations, including a base G with no other auxiliary inputs.
            if not typed and any(by_id[ident]['provenance'].get('oracle_evaluation') is not None
                                 for ident in selected):
                require_identifier(scope, 'adapter observation run')
                typed = True
            graph = GraphFragment()
            for ident in sorted(selected):
                row = by_id[ident]
                capsule = row['provenance'].get('adaptation_visibility')
                need(capsule is not None, 'training episode lacks native visibility capsule')
                validate_capsule(capsule, prompt=row['prompt'], inference_spec=row['provenance']['inference_spec'])
                resolution = validate_resolution(row['provenance'].get('ancestry_resolution'),
                    capsule_sha256=capsule['sha256'], owner=row['owner'])
                graph.merge(GraphFragment(episodes={v['episode_id']: v for v in resolution['episode_bindings']}))
                graph.merge(GraphFragment(episodes={ident: _binding(row)}, episode_roots={ident}))
                if resolution.get('auxiliary_bindings'):
                    producer = validate_producer_binding(row, resolution)
                    inherited = NativeAncestry.resolution_fragment(resolution)
                    inherited.auxiliary[producer['node_id']] = producer
                    inherited.roots, inherited.episode_roots = {producer['node_id']}, set()
                    graph.merge(inherited)
            used = set(selected)
            bindings = [dict(owner=revision['owner'], adapter_id=revision['adapter_id'],
                adapter_sha256=revision['adapter_sha256'], dataset_sha256=revision['dataset_sha256'],
                training_receipt_sha256=sha256_bytes(raw), dataset_manifest_sha256=training['dataset_manifest_sha256'])]
            parent_id = None
            if revision['parent_id'] is not None:
                parent, previous_used, previous_bindings = self._revision(registry,
                    self._inspect(registry, revision['owner'], revision['parent_id']), scope, typed=typed)
                if typed:
                    parent_id, = parent.roots
                graph.merge(parent)
                used.update(previous_used)
                bindings += previous_bindings
            if typed:
                graph = graph.node(kind='adapter_revision', run_id=scope, owner=revision['owner'], owner_scope='learner',
                    identity={key: revision[key] for key in ('owner', 'adapter_id', 'adapter_sha256')},
                    proof=dict(registry_manifest_sha256=content_hash(revision), training_receipt_sha256=sha256_bytes(raw),
                        dataset_sha256=revision['dataset_sha256'], dataset_manifest_sha256=training['dataset_manifest_sha256'],
                        training_schema=training['schema'], parent_revision_id=parent_id, **C.fields(training)))
            need(registry.inspect(revision['owner'], revision['adapter_id']) == revision,
                 'adapter revision changed during ancestry verification')
            return graph, used, bindings
        finally:
            self.active.remove(key)

    def verify(self, resolution, *, producer=None):
        """Reconstruct each retained revision and its graph from external bytes."""
        nodes = {n['node_id']: n for n in resolution.get('auxiliary_bindings', [])}
        episodes = {e['episode_id']: e for e in resolution['episode_bindings']}
        for node in nodes.values():
            if node['kind'] != 'adapter_revision':
                continue
            need(self.sources.get('registry'), 'adapter ancestry requires retained typed adapter revision proof sources')
            from .registry import AdapterRegistry
            registry = AdapterRegistry(self.sources['registry'])
            revision = self._inspect(registry, node['owner'], node['identity']['adapter_id'])
            expected, _, _ = self._revision(registry, revision, node['run_id'], typed=True)
            need(expected.roots == {node['node_id']}, 'adapter revision proof differs from retained evidence')
            need(all(nodes.get(k) == v for k, v in expected.auxiliary.items())
                 and all(episodes.get(k) == v for k, v in expected.episodes.items()),
                 'adapter revision input closure differs from retained evidence')
        # The complete spec in an adapted call must agree with the actual
        # registered base/alias too, not just its weight digest.
        for node in [*nodes.values(), *([producer] if producer is not None else [])]:
            if node['kind'] == 'llm_call' and 'adapter_revision_id' in node['proof']:
                spec = InferenceSpec.from_dict(node['proof']['inference_spec'])
                revision = nodes[node['proof']['adapter_revision_id']]
                expected, _, _ = self.resolve(spec, revision['run_id'], typed=True)
                need(expected.roots == {revision['node_id']}, 'adapted call differs from retained revision')
