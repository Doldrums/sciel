"""Typed tool/cell proofs and exact current-card reconstruction."""
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


class ToolProofs:
    def proposal(self, action_id, attempt, cutoff, visiting):
        key = ('proposal', action_id, attempt)
        need(key not in visiting, 'cyclic tool proposal ancestry')
        result = self._one('llm_result', action_id=action_id, attempt=attempt, cutoff=cutoff)
        artifact = self.artifact(result['receipt_id'], 'proposal')
        start = self._one('llm_start', action_id=action_id, attempt=attempt, cutoff=cutoff)
        calls = [call for (call, kind), row in self.native.transcripts.items() if kind == 'request'
                 and row.get('segment_id') == start['segment_id'] and row.get('journal_event_seq') == start['event_seq']
                 and row.get('prompt_sha256') == start['prompt_sha256']]
        need(len(calls) == 1, 'tool proposal lacks unique actual call')
        call_id = calls[0]
        request, response, prompt, text, spec, dispatch = call_records(self.native, call_id)
        need(spec.role == 'research_tool' and spec.owner == self.owner(start['island_id'])
             and artifact['action_id'] == action_id and artifact['attempt'] == attempt
             and artifact['prompt'] == prompt and artifact['response']['text'] == text,
             'tool proposal differs from traced producer')
        plan, snapshot, baseline = self.plan(action_id)
        need(attempt == 0, 'tool proposal repair input ancestry is not implemented')
        life = self._lifecycle(self._state_at(baseline))
        controller = self._controller(life)
        need(not any(card.get('decision') is not None for card in life._state['cards'].values()),
             'terminal research decision producer ancestry is not implemented')
        # Durable JSON canonicalization preserves values but not insertion order.
        # Recover only map ordering from the actual typed view; no input value
        # may change. The resulting complete prompt must match byte-for-byte.
        snapshot = deepcopy(snapshot)
        plan = deepcopy(plan)
        signal_marker = 'SCHEDULING CONTEXT (planning only; not the research objective or runtime input):\n'
        if signal_marker in prompt:
            try:
                template = json.JSONDecoder().raw_decode(prompt.split(signal_marker, 1)[1])[0]
            except ValueError:
                template = None
            if template is not None:
                before = content_hash(plan['stagnation_signals'])
                plan['stagnation_signals'] = _order_like(plan['stagnation_signals'], template)
                need(content_hash(plan['stagnation_signals']) == before, 'tool scheduling values changed')
        marker = 'OBSERVED SEARCH LIMITS JSON:\n'
        if marker in prompt and getattr(snapshot, 'research_search_evidence', None):
            template = json.JSONDecoder().raw_decode(prompt.split(marker, 1)[1])[0]
            before = content_hash(snapshot.research_search_evidence)
            snapshot.research_search_evidence = _order_like(snapshot.research_search_evidence, template)
            need(content_hash(snapshot.research_search_evidence) == before, 'tool input changed during ordering recovery')
        target_marker = 'FROZEN_INPUT_VIEW_JSON:\n'
        if target_marker in prompt:
            template = json.JSONDecoder().raw_decode(prompt.split(target_marker, 1)[1])[0]
            before = content_hash(snapshot.research_targets)
            snapshot.research_targets = _order_like(snapshot.research_targets, template)
            need(content_hash(snapshot.research_targets) == before, 'tool targets changed during ordering recovery')
        if getattr(snapshot, 'research_cells_context', None):
            for index, card in enumerate(snapshot.research_cells_context['cards']):
                record = self.lifecycle._receipt(card['source']['receipt_id'], card['source']['tool_id'], self.library)
                restored, _, _ = self._card_material(record, card['source']['receipt_id'])
                restored['revision_id'] = card['revision_id']
                need(restored == card, 'frozen prior research card differs')
                snapshot.research_cells_context['cards'][index] = restored
        rendered = controller._prompt(plan, snapshot, plan['visible_tool_ids'], [], attempt)
        need(rendered == prompt, 'tool producer prompt does not match exact host renderer')
        graph = self.targets(snapshot.research_targets, dispatch, visiting | {key})
        graph.merge(self._search_inputs(snapshot, dispatch, visiting | {key}))
        cards = life.delivery_context(getattr(snapshot, 'research_cells_context', None), recipient_cell_id=plan['cell_id'])
        graph.merge(self.context_inputs(cards, dispatch, visiting | {key}))
        shared = plan.get('shared_observations')
        if shared is not None:
            from ..research_tools.shared_evidence import EvidenceIndex
            exact = EvidenceIndex(life, self.library).view(plan['island_id'], self.identity['task'])
            need(shared == exact, 'shared tool input inventory differs from committed baseline')
            need(not shared.get('canonical_evaluations'), 'canonical auxiliary candidate ancestry is not implemented')
            for row in shared['check_index']:
                graph.merge(self.invocation(row['receipt_id'], dispatch, visiting | {key}))
        for ident in plan['visible_tool_ids']:
            graph.merge(self.version(ident, dispatch, visiting | {key}))
        for row in plan.get('author_replay_feedback', []):
            graph.merge(self.invocation(row['reference_receipt_id'], dispatch, visiting | {key}))
            graph.merge(self.invocation(row['replay_receipt_id'], dispatch, visiting | {key}))
        followups = plan.get('method_followups') or {}
        if followups.get('eligible_count'):
            graph.merge(self._followups(plan, dispatch, visiting | {key}))
        if snapshot.kb_lessons or snapshot.artifacts:
            graph.merge(self.native.kb.snapshot_context(snapshot, prompt, dispatch, visiting | {key}))
        need(spec.adapter_id is None, 'adapted tool producer requires a retained typed adapter revision proof')
        return self.native.authenticate_call(call_id, result, graph), artifact

    def _followups(self, plan, cutoff, visiting):
        from ..research_tools.followups import FollowupLedger, build_followup
        from ..engine.research_targets import input_binding
        def exposure(ref):
            source = ref['source']
            saved = self.lifecycle._receipt(source['receipt_id'], source['tool_id'], self.library)
            expected = dict(tool_id=saved['tool_id'], invocation_id=saved['receipt']['invocation_id'],
                            receipt_id=source['receipt_id'], **input_binding(saved['inputs']))
            need(source == expected and ref['cell_id'] == saved['cell_id']
                 and ref['revision_id'] in self.chains[saved['cell_id']]
                 and ref['observation_status'] == saved['result']['status'], 'followup exposure differs')
        ledger = FollowupLedger(self.state['identity_sha256'], authenticate_exposure=exposure)
        eligible = []
        for row in self.state['followups']['records']:
            events = [e for e in self.rows if e.get('phase') == 'method_followup'
                      and e.get('experiment_id') == row['experiment_id'] and self.position(e) <= cutoff]
            if not events:
                continue
            need(len(events) == 1 and events[0]['followup_sha256'] == row['sha256']
                 and events[0]['evidence_artifact_id'] == row['evidence_artifact_id'],
                 'followup differs from committed source')
            saved = self.artifact(row['evidence_artifact_id'], 'method_followup_evidence')
            outcomes = [e for e in self.native.events[:cutoff + 1] if e.get('event') == 'research'
                        and e.get('phase') == 'outcome' and e.get('experiment_id') == row['experiment_id']]
            need(len(outcomes) == 1 and outcomes[0]['evidence'] == saved['research_record'],
                 'followup original research outcome differs')
            expected = build_followup(saved['research_record'], child_program_id=saved['child_program_id'],
                link_kind=saved['link_kind'], observed_record=saved['observed_record'],
                evidence_artifact_id=row['evidence_artifact_id'])
            need(expected == row, 'followup projection differs from exact host renderer')
            ledger.add(row)
            if any(ref['source']['tool_id'] in plan['visible_tool_ids'] for ref in row['exposures']['cards']):
                eligible.append(row)
        need(ledger.planner_view(plan['visible_tool_ids'], max_chars=8000) == plan['method_followups'],
             'followup inventory/omission accounting differs')
        graph = GraphFragment()
        for row in eligible:
            graph.merge(self.native.kb.experiment_graph(row['experiment_id'], cutoff, visiting))
        return graph

    def version(self, ident, cutoff, visiting):
        key = ('tool', ident)
        need(key not in visiting, 'cyclic tool version ancestry')
        registered = self._one('registered', tool_id=ident, cutoff=cutoff)
        need(ident in self.state['registered'], 'tool registration is outside retained checkpoint state')
        if key in self.memo:
            return deepcopy(self.memo[key])
        bundle = self.library.get(ident)
        protocol = bundle['manifest']['protocol']
        need(registered['author_island_id'] == protocol['author_island']
             and registered['source_sha256'] == _sha(bundle['files']['tool.py']), 'tool registration source/author differs')
        result = self._one('llm_result', action_id=registered['action_id'], cutoff=self.position(registered))
        graph, proposal = self.proposal(registered['action_id'], result['attempt'], cutoff, visiting | {key})
        from ..research_tools.controller import parse_action
        plan, _, _ = self.plan(registered['action_id'])
        action = parse_action(proposal['response']['text'], self.config, visible_tool_ids=plan['visible_tool_ids'])
        need(action['action'] == 'create' and action['code'] == bundle['files']['tool.py']
             and action['name'] == bundle['manifest']['name'] and action['description'] == protocol['description']
             and action['parents'] == bundle['manifest']['parents']
             and action['dependencies'] == bundle['manifest']['dependencies']
             and action['replay_policy'] == protocol['replay_policy'], 'tool source does not match actual generated proposal')
        for parent in sorted(set(bundle['manifest']['parents'] + bundle['manifest']['dependencies'])):
            graph.merge(self.version(parent, self.position(registered), visiting | {key}))
        value = graph.node(kind='tool_version', run_id=self.native.run_id,
            owner=self.owner(protocol['author_island']), owner_scope='learner', identity={'tool_id': ident},
            proof=dict(source_sha256=_sha(bundle['files']['tool.py']), manifest_sha256=content_hash(bundle['manifest']),
                       proposal_artifact_id=result['receipt_id'], registration=_event(registered)))
        self.memo[key] = value
        return deepcopy(value)

    def invocation(self, receipt_id, cutoff, visiting):
        key = ('invocation', receipt_id)
        need(key not in visiting, 'cyclic tool invocation ancestry')
        event = self._one('invocation_result', receipt_id=receipt_id, cutoff=cutoff)
        if key in self.memo:
            return deepcopy(self.memo[key])
        record = self.lifecycle._receipt(receipt_id, event['tool_id'], self.library)
        receipt = record['receipt']
        for field in ('action_id', 'cell_id', 'cell_revision_id', 'stage', 'tool_id', 'island_id'):
            need(record[field] == event[field], 'tool receipt differs from committed ' + field)
        for field in ('invocation_id', 'input_sha256', 'output_sha256', 'dispatch_status', 'status', 'resources'):
            need(receipt.get(field) == event.get(field), 'tool receipt differs from committed ' + field)
        need(receipt['runtime_sha256'] == native_hash(self.identity['library']['executor']),
             'tool runtime identity differs')
        if receipt.get('output_sha256') is not None:
            need(receipt['output_sha256'] == native_hash(receipt.get('output')), 'failed/successful tool output hash differs')
        finalization = self._finalization(record, receipt_id)
        start = self._one('invocation_started', action_id=record['action_id'], invocation_id=receipt['invocation_id'])
        dispatch = self.position(start)
        need(dispatch < self.position(event) < self.position(finalization), 'tool execution/finalization ordering differs')
        plan, snapshot, _ = self.plan(record['action_id'])
        need(record['cell_revision_id'] == plan['cell_revision_id']
             and record['action_id'] in self.state['committed_episodes'], 'invocation plan revision/commit differs')
        graph = self.version(record['tool_id'], dispatch, visiting | {key})
        if record['stage'] == 'independent_replay':
            reference = self.artifact(plan['reference_receipt_id'], 'invocation')
            need(record['inputs'] == reference['inputs'], 'independent replay input changed')
            graph.merge(self.invocation(plan['reference_receipt_id'], dispatch, visiting | {key}))
        else:
            graph.merge(self.targets(record['inputs']['research_targets'], dispatch, visiting | {key}))
            graph.merge(self._search_inputs(snapshot, dispatch, visiting | {key}))
            if record['stage'] == 'research':
                result = self._one('llm_result', action_id=record['action_id'], cutoff=dispatch)
                proposed, artifact = self.proposal(record['action_id'], result['attempt'], dispatch, visiting | {key})
                from ..research_tools.controller import parse_action
                from ..engine.research_targets import tool_inputs
                action = parse_action(artifact['response']['text'], self.config, visible_tool_ids=plan['visible_tool_ids'])
                need(record['inputs'] == tool_inputs(snapshot, self.identity['task'], action['arguments']),
                     'runtime inputs differ from actual proposal/frozen snapshot')
                graph.merge(proposed)
                question = action.get('reason') or plan['question']
            elif record['stage'] in ('adopt_trial', 'use_again'):
                original_id = self.state['lifecycle']['methods'][record['tool_id']]['local_receipt_id']
                original = self.artifact(original_id, 'invocation')
                from ..engine.research_targets import tool_inputs
                need(record['inputs'] == tool_inputs(snapshot, self.identity['task'], original['inputs'].get('arguments', {})),
                     'recipient trial inputs differ from frozen snapshot and original arguments')
                graph.merge(self.invocation(original_id, dispatch, visiting | {key}))
                question = ("Does the released tool execute correctly on this recipient's current input?"
                            if record['stage'] == 'adopt_trial' else
                            'Can the adopted method be invoked again in a later research step?')
            else:
                need(False, 'unsupported tool runtime stage')
        if record['stage'] == 'independent_replay':
            question = 'Does the exact saved tool pass its immutable declared replay policy in an independent execution?'
        from ..research_tools.controller import _clip, _clip_json
        need(record['question'] == _clip_json(_clip(question, 600), min(620, self.config['max_context_chars'] // 12)),
             'runtime question differs from host/proposal source')
        value = graph.node(kind='tool_invocation', run_id=self.native.run_id, owner=self.owner(record['island_id']),
            owner_scope='learner', identity=dict(receipt_id=receipt_id, invocation_id=receipt['invocation_id']),
            proof=dict(receipt_sha256=receipt['receipt_sha256'], input_sha256=receipt['input_sha256'],
                       output_sha256=receipt.get('output_sha256'), runtime_sha256=receipt['runtime_sha256'],
                       source_sha256=receipt['source_sha256'], dispatch_status=receipt['dispatch_status'],
                       host_status=receipt['status'], result_status=record['result']['status'],
                       commit_event=_event(event), finalization_event=_event(finalization)))
        self.memo[key] = value
        return deepcopy(value)

    def _card_material(self, record, receipt_id):
        plan, _, _ = self.plan(record['action_id'])
        receipt = record['receipt']
        observation = dict(tool_id=record['tool_id'], invocation_id=receipt['invocation_id'], receipt_id=receipt_id,
            status=receipt['status'], reason=receipt.get('reason'), diagnostic=receipt.get('diagnostic'),
            output=receipt.get('output'), stderr=receipt.get('stderr'), resources=receipt['resources'],
            result=record['result'], replay_policy=record['replay_policy'], target_binding=record['target_binding'],
            question=record['question'])
        if self.identity['contract'] == 'sciel-island-research-tools/34':
            from ..research_tools.comparison_assessment import from_record
            observation['comparison_assessment'] = from_record(record, receipt_id,
                lambda ident: self.artifact(ident, 'proposal'))
        comparison = None
        if record['stage'] == 'independent_replay':
            from ..research_tools.replay_feedback import build_feedback
            original = self.artifact(plan['reference_receipt_id'], 'invocation')
            comparison = build_feedback(original, record, tool_id=record['tool_id'],
                reference_receipt_id=plan['reference_receipt_id'], replay_receipt_id=receipt_id)
        expected = self.controller._card(plan, observation, comparison)
        finalization = self._finalization(record, receipt_id)
        prepared = self._one('context_prepared', action_id=record['action_id'], )
        completed = [e for e in self.native.events if e.get('event') == 'iteration'
                     and e.get('meta', {}).get('research_tools', {}).get('plan', {}).get('action_id') == record['action_id']]
        need(len(completed) == 1, 'tool card lacks unique committed episode')
        episode = completed[0]['meta']['research_tools']
        need(episode['plan'] == plan, 'committed tool episode differs from frozen plan')
        raw = episode['context']
        prefix = self.controller.format_context([])[:-2]
        need(raw.startswith(prefix) and _sha(raw) == prepared['context_sha256'],
             'tool prepared context differs from pre-dispatch commitment')
        # The original context uses insertion-order JSON. Retained raw context
        # proves exact bytes; parsed ordering permits a byte-identical replay.
        displayed = json.loads(raw[len(prefix):])
        need(displayed == [expected] == episode['cards']
             and self.controller.format_context(displayed) == raw,
             'tool prepared card differs from committed rendering')
        return displayed[0], finalization, prepared

    def _state_at(self, cutoff):
        """Projection of committed immutable cell history, never a deleted baseline.

        Parallel cleanup intentionally removes baseline checkpoint copies. The
        retained journal plus reachable revision chains authenticate this view.
        """
        state = deepcopy(self.state['lifecycle'])
        state['cards'], state['methods'], state['research_state'], state['affinity'] = {}, {}, {}, {}
        entries = {}
        for cell_id, full in self.state['lifecycle']['cells']['cells'].items():
            changes = [e for e in self.rows if e.get('cell_id') == cell_id
                       and e.get('phase') in ('cell_created', 'cell_revision') and self.position(e) <= cutoff]
            if not changes:
                continue
            head = changes[-1]['cell_revision_id']
            need(head in self.chains[cell_id], 'historical cell head is not reachable')
            # Affinity grants execution rights elsewhere; this immutable-reader
            # view uses only actual receipt islands and retained source heads.
            entries[cell_id] = dict(head_revision=head, active=False, affinities=deepcopy(full['affinities']))
            invocations = [e for e in self.rows if e.get('phase') == 'invocation_result'
                           and e.get('cell_id') == cell_id and self.position(e) <= cutoff]
            if invocations:
                latest = invocations[-1]
                record = self.lifecycle._receipt(latest['receipt_id'], latest['tool_id'], self.library)
                expected, finalization, _ = self._card_material(record, latest['receipt_id'])
                if self.position(finalization) <= cutoff:
                    expected['revision_id'] = finalization['cell_revision_id']
                    state['cards'][cell_id] = expected
        state['cells']['cells'] = entries
        for ident, method in self.state['lifecycle']['methods'].items():
            registration = self._one('registered', tool_id=ident)
            if self.position(registration) > cutoff:
                continue
            copied = deepcopy(method)
            released = any(e.get('phase') == 'published' and e.get('tool_id') == ident
                           and self.position(e) <= cutoff for e in self.rows)
            copied['stage'] = 'released' if released else 'locally_tested'
            state['methods'][ident] = copied
        return state

    def card(self, card, cutoff, visiting):
        from ..engine.research_cells_context import _card, _json
        need(_card(card) is not None and card.get('decision') is None,
             'unsupported or malformed delivered research card')
        receipt_id = card['source']['receipt_id']
        need(receipt_id is not None, 'research card lacks actual invocation')
        self._one('invocation_result', receipt_id=receipt_id, cutoff=cutoff)
        record = self.lifecycle._receipt(receipt_id, card['source']['tool_id'], self.library)
        expected, finalization, prepared = self._card_material(record, receipt_id)
        need(self.position(prepared) <= cutoff, 'research card was prepared after consumer dispatch')
        availability = prepared
        if card['revision_id'] != expected['revision_id']:
            need(card['revision_id'] == finalization['cell_revision_id'] and self.position(finalization) <= cutoff,
                 'research card borrows a future/unrelated revision')
            expected['revision_id'] = card['revision_id']
            availability = finalization
        need(card == expected, 'research card differs from exact receipt projection')
        graph = self.invocation(receipt_id, cutoff, visiting)
        return graph.node(kind='research_card', run_id=self.native.run_id,
            owner=self.owner(record['island_id']), owner_scope='learner',
            identity=dict(cell_id=card['cell_id'], revision_id=card['revision_id'], card_sha256=_sha(_json(card))),
            proof=dict(card=card, renderer_sha256=self.identity['implementation']['controller.py'],
                       availability_event=_event(availability), finalization_event=_event(finalization)))

    def context_inputs(self, context, cutoff, visiting):
        from ..engine.research_cells_context import compact_context
        need(compact_context(context) == context, 'research card collection is not a canonical whole projection')
        need(context['omitted_count'] == 0 and not context['omission_reasons'],
             'omitted research card inventory is not yet authenticated')
        graph = GraphFragment()
        for card in context['cards']:
            graph.merge(self.card(card, cutoff, visiting))
        return graph

    def resolve_context(self, context, event, prompt, cutoff, visiting):
        from ..engine.research_cells_context import context_metadata, merge_context, compact_context
        rows = [e for e in self.native.events[:cutoff + 1] if e.get('event') == 'research'
                and e.get('phase') == 'context_prepared' and e.get('experiment_id') == event['meta']['research_id']]
        need(len(rows) == 1 and rows[0]['research_cells_context'] == context_metadata(context),
             'delivered cards differ from committed research preparation')
        if context['omitted_count']:
            # Same-wave outer commit finalizes the already dispatched tool
            # receipt; it must not be mistaken for new information delivered
            # after G. The frozen plan binds the complete pre-tool inventory.
            iterations = [row for row in self.native.events if row.get('event') == 'iteration'
                          and row['iteration_no'] == event['iteration_no'] and row['island'] == event['island']]
            need(len(iterations) == 1, 'omitted research inventory lacks unique committed iteration')
            episode = iterations[0].get('meta', {}).get('research_tools')
            need(episode is not None, 'omitted research inventory lacks a frozen tool plan')
            plan, snapshot, _ = self.plan(episode['plan']['action_id'])
            need(episode['plan'] == plan and snapshot.iteration_no + 1 == event['iteration_no']
                 and snapshot.island == event['island'], 'omitted card plan differs from actual consumer')
            prior, fresh = snapshot.research_cells_context, episode['cards']
            graph = self.context_inputs(prior, cutoff, visiting)
            graph.merge(self.context_inputs(compact_context(fresh), cutoff, visiting))
            exact = merge_context(prior, fresh, max_chars=min(self.config['max_context_chars'], 10000))
            reservation = snapshot.research_context.get('research_cells_reservation')
            if reservation is not None:
                exact = compact_context(exact, max_chars=reservation['reserved_chars'] + reservation['framing_chars'])
            need(exact == context, 'research omission inventory differs from exact bounded host merge')
            # Hidden inputs are provenance of the shown omission accounting,
            # not a claim their complete card text was displayed to G.
            return graph
        return self.context_inputs(context, cutoff, visiting)

    def resolve_ref(self, ref, event, prompt, cutoff, visiting):
        from ..hypothesis.rendering import decode_evidence
        context = decode_evidence(json.loads(self.native._json_line(prompt, 'RESEARCH_CONTEXT_JSON')))
        # Visibility capsules use adaptation's UTF-8 canonical hash. Native
        # card proofs below retain the renderer's separate escaped-JSON hash.
        cards = [card for card in context.get('research_cells_context', {}).get('cards', [])
                 if content_hash(card) == ref.get('card_sha256')]
        need(len(cards) == 1, 'unresolved delivered projection: research_card')
        card = cards[0]
        need(ref.get('cell_id') == card['cell_id'] and ref.get('revision_id') == card['revision_id']
             and ref.get('source') == card['source'], 'research card capsule identity differs')
        return self.card(card, cutoff, visiting)
