"""Circle generation checkpoint and paid-request evidence validation."""
from __future__ import annotations

import copy
import hashlib
import json
from ..hypothesis import circle_restart_body as builder
from .prompt.sources import source_digest
from .circle_generation_contracts import (
    CONTRACT,
    _COUNTERS,
    _PAIR_COUNTERS,
    _finite,
    _journal_known_cost,
    _json,
)


class CircleGenerationState:

    def summary(self):
        return {'identity': self.identity, **{k:getattr(self,k) for k in _COUNTERS},
            'known_cost': self.known_cost, 'seen_proof_count': len(self.seen_proofs),
            'stopped_reason': self.stopped_reason,
            **({'proposal_pair_counts': copy.deepcopy(self.pair_counts)} if self.pair_mode else {})}

    def state(self):
        return {'schema': CONTRACT, **self.summary(), 'seen_proofs': copy.deepcopy(self.seen_proofs),
            'pending': copy.deepcopy(self.pending),
            **({'proposal_pair_history': copy.deepcopy(self.pair_history)} if self.pair_mode else {})}

    def _proof(self, proof):
        if type(proof) is not dict:
            raise ValueError('invalid Circle generation proof')
        expected = self.builder.apply(self.builder.source_for(proof['parent_body']), proof['target_body'])['proof']
        if _json(proof) != _json(expected):
            raise ValueError('Circle generation proof differs')
        return expected

    def load_state(self, state):
        if (type(state) is not dict or set(state) != set(self.state()) or state['schema'] != CONTRACT
                or _json(state['identity']) != _json(self.identity) or state['pending'] is not None
                or state['stopped_reason'] is not None):
            raise ValueError('invalid, stopped or unfinished Circle generation checkpoint')
        if any(type(state[k]) is not int or state[k] < 0 for k in (*_COUNTERS, 'seen_proof_count')):
            raise ValueError('invalid Circle generation counters')
        proofs = state['seen_proofs']
        if type(proofs) is not list:
            raise ValueError('invalid Circle generation proof list')
        validated = [self._proof(p) for p in proofs]
        if (len({p['child_source_sha256'] for p in validated}) != len(validated)
                or state['seen_proof_count'] != len(validated) or not _finite(state['known_cost'])
                or not state['attempts'] == state['finished'] == state['calls_started'] == state['calls_returned']
                or (not self.pair_mode and state['invalid_responses'] + state['repeated_proposals']
                    + len(validated) != state['calls_returned'])
                or state['evaluations'] > len(validated)
                or (not state['calls_returned'] and state['known_cost'] != 0)):
            raise ValueError('inconsistent Circle generation checkpoint')
        if self.pair_mode:
            counts, history = state['proposal_pair_counts'], state['proposal_pair_history']
            if (type(counts) is not dict or set(counts) != set(_PAIR_COUNTERS)
                    or any(type(v) is not int or v < 0 for v in counts.values())
                    or type(history) is not list or len(history) != state['calls_returned']
                    or counts['returned_groups'] != state['calls_returned']
                    or not counts['evaluated_entries'] == counts['selected_entries'] == len(validated) == state['evaluations']
                    or counts['no_eligible_requests'] + len(validated) != state['calls_returned']
                    or counts['entries_present'] > 2 * state['calls_returned']
                    or not counts['selected_entries'] <= counts['eligible_entries'] <= min(
                        counts['forecast_valid_entries'], counts['body_valid_entries'])
                    or max(counts['forecast_valid_entries'], counts['body_valid_entries']) > counts['entries_present']
                    or state['repeated_proposals'] != 0 or state['invalid_responses'] > counts['no_eligible_requests']):
                raise ValueError('inconsistent Circle pair accounting')
            for item in history:
                if (type(item) is not dict or set(item) != {'experiment_id', 'group_sha256',
                        'response_sha256', 'selected_index', 'selected_source_sha256'}
                        or type(item['experiment_id']) is not str or not item['experiment_id']
                        or (item['selected_index'] is not None and
                            (type(item['selected_index']) is not int or item['selected_index'] not in (0, 1)))):
                    raise ValueError('invalid Circle pair history')
                for key in ('group_sha256', 'response_sha256', 'selected_source_sha256'):
                    value = item[key]
                    nullable = key == 'selected_source_sha256' and item['selected_index'] is None
                    if not ((nullable and value is None) or (not nullable and type(value) is str
                            and len(value) == 64 and all(c in '0123456789abcdef' for c in value))):
                        raise ValueError('invalid Circle pair history digest')
            if (len({e['experiment_id'] for e in history}) != len(history)
                    or [e['selected_source_sha256'] for e in history if e['selected_index'] is not None]
                       != [p['child_source_sha256'] for p in validated]):
                raise ValueError('Circle pair history/proof mismatch')
            self.pair_counts, self.pair_history = copy.deepcopy(counts), copy.deepcopy(history)
        for k in _COUNTERS:
            setattr(self, k, state[k])
        self.known_cost, self.seen_proofs = float(state['known_cost']), copy.deepcopy(validated)

    def assert_journal(self, log_path, anchor):
        if type(anchor) is not dict or type(anchor.get('byte_offset')) is not int or anchor['byte_offset'] < 1:
            raise ValueError('Circle generation requires journal provenance')
        with open(log_path, 'rb') as stream:
            prefix = stream.read(anchor['byte_offset'])
        if (len(prefix) != anchor['byte_offset'] or not prefix.endswith(b'\n')
                or hashlib.sha256(prefix).hexdigest() != anchor.get('prefix_sha256')):
            raise ValueError('Circle generation journal anchor differs')
        def unique(pairs):
            result = {}
            for key,value in pairs:
                if key in result:
                    raise ValueError('duplicate Circle generation journal key')
                result[key] = value
            return result
        events, journal_rows = [], []
        for line in prefix.splitlines():
            event = json.loads(line, object_pairs_hook=unique)
            if type(event) is not dict:
                raise ValueError('invalid Circle generation journal row')
            journal_rows.append(event)
            if event.get('event') == 'research' and event.get('phase') == 'circle_generation':
                if event.get('generation_contract') != CONTRACT:
                    raise ValueError('Circle generation journal contract differs')
                events.append(event)
        from .external_research import assert_import_journal, generation_records, record_matches
        assert_import_journal(self.research, journal_rows)
        from .generated_paired_import import assert_import_journal as assert_component_journal
        assert_component_journal(self.research, journal_rows)
        from .evaluated_study_import import assert_import_journal as assert_study_journal
        assert_study_journal(self.research, journal_rows)
        from .evaluated_review_import import assert_import_journal as assert_study_review_journal
        assert_study_review_journal(self.research, journal_rows)
        external = generation_records(self.research)
        proofs = [self._proof(e['proof']) for e in events if e.get('action') == 'source_binding']
        results = [e for e in events if e.get('action') == 'request_result']
        finishes = [e for e in events if e.get('action') == 'finish']
        known_costs = [_journal_known_cost(e) for e in results]
        if (_json(proofs) != _json(self.seen_proofs)
                or sum(e.get('action') == 'request_start' for e in events) != self.calls_started
                or len(results) != self.calls_returned or len(finishes) != self.finished
                or sum(known_costs) != self.known_cost
                or sum(e.get('status') == 'generation_invalid_response' for e in finishes) != self.invalid_responses
                or sum(e.get('status') == 'generation_repeated' for e in finishes) != self.repeated_proposals
                or sum(e.get('status') in ('evaluated', 'eval_failed') for e in finishes) != self.evaluations
                or any(e.get('source_origin') != 'deterministic_body_compiler'
                    or e.get('semantic_or_causal_claim') is not False
                    for e in events if e.get('action') == 'source_binding')):
            raise ValueError('Circle generation checkpoint differs from journal')
        if self.pair_mode:
            self._assert_pair_journal(events)
        # The durable raw reply, the selected hypothesis and the committed
        # source must agree even if someone recomputes checkpoint-local hashes.
        by_id = {e['experiment_id']: e for e in results}
        records = self.research._prediction_records()
        for record in records.values():
            binding = record.get('restart_prediction_binding')
            if binding is None:
                if (self.research.config.get('restart_predictions', False)
                        and record.get('source_sha256') is not None
                        and record.get('operator') == 'circle_generate'):
                    raise ValueError('Circle generation lost its pre-evaluation predictions')
                continue
            if record['id'] in external:
                if record['id'] in by_id or not record_matches(self.research, record):
                    raise ValueError('Circle imported/current prediction identity collision')
                result = external[record['id']]['raw_result']
            else:
                result = by_id.get(record['id'])
            if result is None or binding['response_sha256'] != source_digest(result['response']):
                raise ValueError('Circle prediction binding differs from raw request journal')
            if self.pair_mode:
                from ..hypothesis import circle_proposal_pair
                index = record['proposal_group']['selected_index']
                parsed = circle_proposal_pair.decode_response(result['response'])['entries'][index]['hypothesis']
            else:
                parsed = builder.parse_hypothesis_response(result['response'])
            if parsed is None or _json(parsed.to_dict()) != _json(record['hypothesis']):
                raise ValueError('Circle prediction differs from original model response')

    def _assert_pair_journal(self, events):
        from ..hypothesis.circle_proposal_group import build_group, validate_group
        from . import circle_pair_selection as selection
        counts = dict.fromkeys(_PAIR_COUNTERS, 0)
        history, consumed, normalized_history, records, active = [], set(), set(), {}, None
        for event in events:
            action, ident = event.get('action'), event.get('experiment_id')
            if action == 'request_start':
                if active is not None or type(ident) is not str or ident in records:
                    raise ValueError('Circle pair request ordering differs')
                active = {'id': ident, 'start': event, 'result': None, 'group': None, 'proof': None}
            elif action in ('request_result', 'proposal_group', 'source_binding', 'finish'):
                if active is None or ident != active['id']:
                    raise ValueError('Circle pair request provenance differs')
                if action == 'request_result':
                    if active['result'] is not None or type(event.get('response')) is not str:
                        raise ValueError('Circle pair raw receipt differs')
                    if event.get('response_sha256') != source_digest(event['response']):
                        raise ValueError('Circle pair raw response hash differs')
                    active['result'] = event
                elif action == 'proposal_group':
                    receipt = active['result']
                    if receipt is None or active['group'] is not None or active['proof'] is not None:
                        raise ValueError('Circle pair assessment ordering differs')
                    frozen = event['selection_snapshot']
                    if (frozen['consumed_sources'] != sorted(consumed)
                            or not normalized_history.issubset(frozen['code_hashes'])
                            or frozen.get('restart_predictions', False) != self.research.config.get('restart_predictions', False)
                            or frozen['parent_source_sha256'] != active['start']['parent_source_sha256']):
                        raise ValueError('Circle pair saved selection history differs')
                    provider_error = (None if receipt.get('finish_reason') == 'stop'
                        and receipt.get('refusal') in (None, '') else 'structured_response_not_complete_or_refused')
                    assessed = selection.assess(receipt['response'], frozen,
                        provider_error=provider_error, replay_evidence=event['gate_evidence'])
                    selected = assessed['selected']
                    expected = build_group(request_id=ident, raw_response=receipt['response'],
                        parent_source_sha256=frozen['parent_source_sha256'], gates=assessed['gates'],
                        provider_error=provider_error, selected_proof=selected['built']['proof'] if selected else None,
                        **({'execution_profile': self.execution_profile} if self.execution_profile else {}))
                    actual = validate_group(event['group'], raw_response=receipt['response'])
                    if _json(actual) != _json(expected) or event.get('quality_measured') is not False:
                        raise ValueError('Circle pair indexed proposal group differs')
                    active['group'] = actual
                    active['expected_proof'] = selected['built']['proof'] if selected else None
                    history.append(self._group_history(actual))
                    for key, value in self._group_counts(actual).items():
                        counts[key] += value
                elif action == 'source_binding':
                    if (active['group'] is None or active['proof'] is not None
                            or active.get('expected_proof') is None
                            or _json(event['proof']) != _json(active['expected_proof'])):
                        raise ValueError('Circle pair selected source proof differs')
                    active['proof'] = event['proof']
                    exact, normalized = self._source_history([event['proof']])
                    consumed.update(exact)
                    normalized_history.update(normalized)
                else:
                    group = active['group']
                    if group is None:
                        raise ValueError('Circle pair request finished without terminal group')
                    selected = group['selected_index'] is not None
                    invalid = group['envelope_error'] is not None or group['provider_error'] is not None
                    expected_statuses = (('evaluated', 'eval_failed') if selected else
                        ('generation_invalid_response',) if invalid else ('generation_pair_no_eligible',))
                    if (event.get('status') not in expected_statuses
                            or selected != (active['proof'] is not None)
                            or event.get('known_cost') != active['result']['cost']):
                        raise ValueError('Circle pair terminal outcome differs')
                    counts['evaluated_entries'] += int(selected)
                    records[ident] = active
                    active = None
        if (active is not None or _json(history) != _json(self.pair_history) or counts != self.pair_counts
                or not self.research.attempts == self.research.outcomes == len(records)
                or self.research.valid_hypotheses != counts['selected_entries']):
            raise ValueError('Circle pair journal counters/history differ')
        # This operator owns every request in the run. A schema-optional group
        # must not become optional provenance on restore: dropping one record or
        # its group would otherwise silently erase alternatives and feedback.
        expected_retained = list(records)[-self.research.knowledge.max_experiments:]
        if (set(self.research.knowledge._seen) != set(records)
                or [r['id'] for r in self.research.knowledge.experiments] != expected_retained):
            raise ValueError('Circle pair knowledge request history differs from journal')
        for record in self.research.knowledge._evidence().values():
            group = record.get('proposal_group')
            if group is not None or record.get('id') in records:
                item = records.get(record['id'])
                if (item is None or group is None or _json(group) != _json(item['group'])
                        or record.get('generation_cost') != item['result']['cost']):
                    raise ValueError('Circle pair knowledge differs from native request journal')
