"""Authenticate immutable tool producers and dispatch to typed proof assembly."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import stat
from types import SimpleNamespace

from .auxiliary_ancestry import GraphFragment, call_records, need, _commit
from .types import content_hash, require_hash, sha256_bytes
# Tool producers escape Unicode; adaptation graph/event hashes retain their own format.
from ..methods.models import content_hash as native_hash
from .tool_ancestry_protocol import _sha, _event, _order_like, validate_node
from .tool_ancestry_proofs import ToolProofs


class ToolAncestry(ToolProofs):
    def __init__(self, native):
        from ..research_tools.library import ToolLibrary
        from ..research_tools.config import library_limits
        self.native = native
        self.config = native.config.get('research_tools', {})
        need(self.config.get('enabled') is True, 'unresolved delivered projection: research_card')
        self.state = native.checkpoint['research_tools']
        self.identity = native.checkpoint['identity']['research_tools']['identity']
        need(native_hash(self.identity) == self.state['identity_sha256']
             and self.identity['config'] == self.config, 'tool checkpoint identity differs')
        need(self.config['max_scientific_steps'] == 1 and not self.config['jobs']['enabled'],
             'sequential/background tool producer ancestry is not implemented')
        # Initialize only immutable readers: constructors create files and must
        # never be called against the evidence under audit.
        library = ToolLibrary.__new__(ToolLibrary)
        library.run_dir = native.root.resolve()
        library.path = library.run_dir / 'research_tools'
        library.objects = library.path / 'objects'
        library._runtime_lock = {'image_id': self.config['image_id']}
        library._limits = library_limits(self.config)
        self.library = library
        self.lifecycle = self._lifecycle(self.state['lifecycle'])
        self.controller = self._controller(self.lifecycle)
        self.rows = [e for e in native.events if e.get('event') == 'research_tool']
        self.memo, self.plans = {}, {}
        self.chains = {}
        for cell in self.lifecycle.cells.all_cells():
            cursor, chain = cell['head_revision'], {}
            while cursor is not None:
                revision = self.lifecycle.cells._read(self.lifecycle.cells.revisions, cursor)
                chain[cursor] = revision
                cursor = revision['previous_revision']
            self.chains[cell['id']] = chain

    def _lifecycle(self, state):
        from ..research_tools.cells import CellRegistry
        from ..research_tools.lifecycle import CellLifecycle
        cells = CellRegistry.__new__(CellRegistry)
        cells.run_dir = self.native.root.resolve()
        cells.path = cells.run_dir / 'research_tools/cells'
        cells.genesis, cells.revisions = cells.path / 'genesis', cells.path / 'revisions'
        cells.max_active = self.config['max_active_cells']
        cells.load_state(state['cells'])
        lifecycle = CellLifecycle.__new__(CellLifecycle)
        lifecycle.config, lifecycle.run_dir = self.config, self.native.root.resolve()
        lifecycle.cells = cells
        lifecycle._state = deepcopy(state)
        lifecycle._shared_evidence_index = None
        lifecycle.run_id = self.native.run_id
        lifecycle.emit = lambda **_: None
        return lifecycle

    def _controller(self, lifecycle):
        from ..research_tools.controller import ResearchTools
        controller = ResearchTools.__new__(ResearchTools)
        controller.config, controller.task_identity = self.config, self.identity['task']
        controller.library, controller.lifecycle = self.library, lifecycle
        controller.path = self.library.path / 'artifacts'
        controller._state = deepcopy(self.state)
        controller._identity = deepcopy(self.identity)
        controller._authors = {}
        controller.emit = lambda **_: None
        from ..research_tools import controller as source
        current = sha256_bytes(Path(source.__file__).read_bytes())
        recorded = self.identity['implementation']['controller.py']
        from .tool_prompt_legacy import PROFILES, bind
        if self.identity['contract'] == 'sciel-island-research-tools/34':
            from ..engine.source_identity import package_source_hashes
            from ..research_tools.tool_protocol import CONTRACT
            need(CONTRACT == self.state['schema'] == self.identity['contract'],
                 'unsupported current tool protocol')
            for field, package in (('implementation', 'research_tools'), ('integration', 'engine')):
                installed = {name.removeprefix(package + '/'): digest
                    for name, digest in package_source_hashes(packages=(package,)).items()}
                need(self.identity[field] == installed,
                     'tool renderer package source closure differs: ' + field)
        elif recorded in PROFILES:
            bind(controller, recorded)
        else:
            need(recorded == current and self.identity['contract'] == source.CONTRACT,
                 'unsupported historical tool prompt renderer')
        return controller

    def _one(self, phase, *, cutoff=None, **keys):
        rows = [e for e in self.rows if e.get('phase') == phase
                and all(e.get(k) == v for k, v in keys.items())
                and (cutoff is None or self.position(e) <= cutoff)]
        need(len(rows) == 1, 'tool ' + phase + ' lacks one committed binding')
        return rows[0]

    def position(self, row):
        return self.native.positions[(row['segment_id'], row['event_seq'])]

    def owner(self, island):
        owner = self.native.config['llm']['local_inference']['learner_by_island'].get(str(island))
        need(owner is not None, 'tool island has no native learner identity')
        return owner

    def artifact(self, ident, kind):
        return self.lifecycle._artifact(ident, kind, self.library)

    def _finalization(self, record, receipt_id):
        evidence = {'kind': 'tool_invocation', 'id': receipt_id, 'tool_id': record['tool_id']}
        matches = [(ident, revision) for ident, revision in self.chains.get(record['cell_id'], {}).items()
                   if evidence in revision['evidence']]
        need(matches, 'invocation is outside checkpoint-committed cell history')
        ident, revision = min(matches, key=lambda pair: pair[1]['ordinal'])
        need(record['cell_revision_id'] in self.chains[record['cell_id']], 'invocation cell predecessor is unknown')
        # A later no-op role assignment can report this same immutable head.
        # Only the original invocation's stage/island can finalize its evidence.
        event = self._one('cell_revision', cell_id=record['cell_id'], cell_revision_id=ident,
                          stage=record['stage'], island_id=record['island_id'])
        need(event['evidence'] == revision['evidence'] and event['tool_ids'] == revision['tool_ids'],
             'cell finalization differs from immutable revision')
        return event

    def plan(self, action_id):
        if action_id in self.plans:
            return self.plans[action_id]
        from ..engine.parallel_store import _manifest
        from ..research_tools.controller import _basis
        candidates = []
        waves = {e['wave'] for e in self.native.events if e.get('policy') == 'parallel_search'
                 and e.get('action') == 'wave_dispatch'}
        for wave in sorted(waves):
            directory = self.native.root / '.parallel' / ('wave-' + str(wave))
            manifest = _manifest(directory)
            need(manifest['phase'] == 'committed', 'tool producer wave is not committed')
            dispatches = [e for e in self.native.events if e.get('policy') == 'parallel_search'
                          and e.get('action') == 'wave_dispatch' and e.get('wave') == wave]
            need(len(dispatches) == 1, 'tool wave lacks unique committed dispatch')
            baseline_cutoff = self.position(dispatches[0]) - 1
            jobs = self._json_file(directory / 'jobs.json')
            for job in jobs['jobs']:
                raw = job['snapshot']
                plan = raw.get('tool_episode_plan')
                if plan is None or plan.get('action_id') != action_id:
                    continue
                snap = SimpleNamespace(**raw)
                need(action_id == native_hash({k:v for k,v in plan.items() if k != 'action_id'})
                     and plan['snapshot_sha256'] == native_hash(_basis(snap))
                     and plan['identity_sha256'] == self.state['identity_sha256'], 'frozen tool plan/snapshot differs')
                starts = [e for e in self.rows if e.get('action_id') == action_id
                          and e.get('phase') in ('llm_start', 'invocation_started')]
                need(starts and all(e['island_id'] == plan['island_id'] and e['cell_id'] == plan['cell_id']
                     and e['cell_revision_id'] == plan['cell_revision_id'] for e in starts), 'tool plan has no actual dispatch')
                candidates.append((plan, snap, baseline_cutoff))
        need(len(candidates) == 1, 'tool plan is missing or ambiguous')
        self.plans[action_id] = candidates[0]
        return candidates[0]

    def _json_file(self, path):
        from ..research_tools.library import _decode
        self.library._refuse_links(path)
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            before = os.fstat(fd)
            need(stat.S_ISREG(before.st_mode) and before.st_size <= 64 * 1024 * 1024,
                 'tool intent is oversized or not regular')
            with os.fdopen(fd, 'rb', closefd=False) as stream:
                raw = stream.read(64 * 1024 * 1024 + 1)
            after = os.fstat(fd)
            signature = lambda value: (value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
            need(signature(before) == signature(after) and len(raw) == before.st_size,
                 'tool intent changed during read')
        finally:
            os.close(fd)
        return _decode(raw)

    def targets(self, targets, cutoff, visiting):
        from ..engine.research_targets import validate_targets
        validate_targets(targets, task_hash=self.native.task_hash)
        need(targets['run_id'] == self.native.run_id, 'foreign runtime tool targets')
        graph = GraphFragment()
        for target in (targets['ancestry_parent'], targets['selected'], targets['population_best']):
            if target is None:
                continue
            need(target.get('program_id') is not None, 'failed tool target requires failed-call ancestry')
            graph.merge(self.native.program_graph(target['program_id'], target['source_sha256'], cutoff, visiting))
            saved = target.get('saved_evaluation')
            if saved and saved['status'] == 'available':
                from .export import _receipt
                record = saved['record']
                exact = _receipt(self.native.root, record['evaluation_id'], target['source_sha256'], self.native.run_id)
                need(exact == record, 'tool target evaluator receipt differs')
        return graph

    def _search_inputs(self, snapshot, cutoff, visiting):
        graph = GraphFragment()
        evidence = getattr(snapshot, 'research_search_evidence', None)
        if not evidence:
            return graph
        # Global counters and omission counts depend on the full committed
        # inventory, not only the few displayed recent rows. Keep its roots.
        for event in self.native.events[:cutoff + 1]:
            if event.get('event') != 'iteration':
                continue
            if event.get('child_id') is not None:
                graph.merge(self.native.program_graph(event['child_id'], _sha(event['child_code']), cutoff, visiting))
            else:
                for attempt in event.get('meta', {}).get('validation_trace', {}).get('attempts', []):
                    if attempt.get('call_id'):
                        graph.merge(self.native.kb.resolve_call(attempt['call_id'], cutoff, visiting))
        return graph
