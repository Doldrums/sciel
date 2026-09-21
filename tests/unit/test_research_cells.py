"""Independent research lineage, bounded work, and checkpoint artifact integrity."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sciel.methods.models import canonical_bytes, content_hash
from sciel.research_tools.cells import CellRegistry, MAX_ARTIFACT_BYTES


class CellRegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name).resolve() / 'run'
        self.cells = CellRegistry(self.run, max_active=2)

    def cell(self, island=0, question='Which assumption prevents progress?', **kwargs):
        return self.cells.ensure_cell(island, question,
            {'kind': 'stagnation', 'evidence': {'iteration': 12, 'record_steps': 0, 'parent_id': 'p12'}}, **kwargs)

    def revise(self, cell, **kwargs):
        fields = {'phase': 'trial', 'tool_ids': ['a' * 64],
                  'evidence': [{'kind': 'execution_receipt', 'id': 'b' * 64, 'tool_id': 'a' * 64}],
                  'summary': 'Execution succeeded; task benefit remains untested.'}
        return self.cells.revise(cell['id'], **{**fields, **kwargs})

    def test_question_identity_survives_candidate_changes_and_serves_other_islands(self):
        first = self.cell()
        repeated = self.cells.ensure_cell(0, first['question'],
            {'kind': 'stagnation', 'evidence': {'parent_id': 'p999', 'iteration': 99}})
        self.assertEqual(first['id'], repeated['id'])
        self.assertEqual(first['trigger'], repeated['trigger'])
        foreign = self.cell(island=8, cell_id=first['id'])
        self.assertEqual(foreign['origin_island'], 0)
        self.assertEqual(foreign['affinities'], [0, 8])
        self.assertEqual(foreign['id'], first['id'])
        self.assertEqual(len(self.cells.all_cells()), 1)
        with self.assertRaisesRegex(ValueError, 'retain'):
            self.cell(island=8, question='A different research question', cell_id=first['id'])
        self.assertNotEqual(self.cell(island=9)['id'], first['id'])
        # No candidate/population object or disk scan participates in identity.
        self.assertNotIn('parent_id', self.cells.state()['cells'][first['id']])

    def test_active_bound_is_not_a_total_cell_or_lifetime_tool_bound(self):
        created = [self.cell(question=f'Research question {index}') for index in range(40)]
        self.assertEqual(len(self.cells.all_cells()), 40)
        self.assertEqual(sum(cell['active'] for cell in self.cells.all_cells()), 2)
        self.assertFalse(self.cells.activate(created[-1]['id']))
        self.cells.deactivate(created[0]['id'])
        self.assertTrue(self.cells.activate(created[-1]['id']))
        self.assertEqual(len(self.cells.all_cells()), 40)
        revised = created[-1]
        for index in range(300):
            revised = self.cells.revise(revised['id'], phase='trial', tool_ids=[content_hash({'source': index})])
        self.assertEqual(revised['revision']['ordinal'], 300)
        self.assertEqual(len(list(self.cells.revisions.glob('*.json'))), 300)
        restored = CellRegistry(self.run, max_active=2)
        restored.load_state(self.cells.state())
        self.assertEqual(restored.state(), self.cells.state())
        self.assertEqual(restored.get(revised['id'])['revision']['ordinal'], 300)

    def test_immutable_revisions_pin_versions_without_creating_tools(self):
        cell = self.cell()
        original = self.revise(cell)
        replayed = self.cells.revise(cell['id'], phase='independent_replay',
            evidence=[{'kind': 'execution_receipt', 'id': 'c' * 64}], decision={'passed': True})
        self.assertEqual(original['tool_ids'], replayed['tool_ids'])
        self.assertEqual(replayed['revision']['previous_revision'], original['head_revision'])
        self.assertEqual(replayed['revision']['ordinal'], 2)
        self.assertEqual(replayed['revision']['evidence'], [{'kind': 'execution_receipt', 'id': 'c' * 64}])
        for entry in (original, replayed):
            artifact = json.loads((self.cells.revisions / (entry['head_revision'] + '.json')).read_text())
            self.assertEqual(content_hash(artifact), entry['head_revision'])
        self.assertFalse((self.run / 'research_tools' / 'objects').exists())
        # Labels do not silently establish adoption or publish any tools.
        self.assertNotIn('adopted', self.cells.state())
        self.assertNotIn('published', self.cells.state())

    def test_roles_reject_creator_self_replay_and_keep_independent_receipts_separate(self):
        cell = self.revise(self.cell())
        before = self.cells.state()
        for role in ('replayer', 'recipient'):
            for identity in (0, '0'):
                with self.subTest(role=role, identity=identity), self.assertRaisesRegex(ValueError, 'independent'):
                    self.cells.assign_role(cell['id'], role, identity)
        with self.assertRaisesRegex(ValueError, 'creator'):
            self.cells.assign_role(cell['id'], 'creator', 3)
        self.assertEqual(self.cells.state(), before)
        replay = self.cells.assign_role(cell['id'], 'replayer', 2)
        recipient = self.cells.assign_role(cell['id'], 'recipient', 3)
        self.assertEqual(recipient['roles'], {'creator': 0, 'replayer': 2, 'recipient': 3})
        self.assertEqual(recipient['tool_ids'], ['a' * 64])
        self.assertEqual(recipient['revision']['evidence'], [])
        self.assertEqual(replay['revision']['phase'], 'role_assignment')
        # Repeated assignment does not manufacture another research revision.
        self.assertEqual(self.cells.assign_role(cell['id'], 'recipient', 3)['head_revision'], recipient['head_revision'])

    def test_checkpoint_visibility_and_rollback_ignore_orphan_artifacts(self):
        cell = self.revise(self.cell())
        checkpoint = self.cells.state()
        orphan = self.cell(7)
        abandoned = self.revise(cell, phase='repair', tool_ids=['d' * 64])
        self.cells.load_state(checkpoint)
        with self.assertRaises(KeyError):
            self.cells.get(orphan['id'])
        self.assertEqual(self.cells.get(cell['id'])['head_revision'], cell['head_revision'])
        self.assertTrue((self.cells.genesis / (orphan['id'] + '.json')).exists())
        self.assertTrue((self.cells.revisions / (abandoned['head_revision'] + '.json')).exists())
        cold = CellRegistry(self.run, max_active=2)
        self.assertEqual(cold.all_cells(), [])
        cold.load_state(checkpoint)
        self.assertEqual([row['id'] for row in cold.all_cells()], [cell['id']])
        self.assertEqual(cold.get(cell['id'])['tool_ids'], ['a' * 64])

    def test_invalid_state_and_history_fail_atomically_without_erasing_current_state(self):
        cell = self.revise(self.cell())
        cell = self.revise(cell, phase='repair')
        good = self.cells.state()
        cases = []
        bad = deepcopy(good); bad['max_active'] = 10; cases.append(bad)
        bad = deepcopy(good); bad['cells'][cell['id']]['active'] = 1; cases.append(bad)
        bad = deepcopy(good); bad['cells'][cell['id']]['affinities'] = [0, '0']; cases.append(bad)
        bad = deepcopy(good); bad['cells'][cell['id']]['affinities'] = [9]; cases.append(bad)
        for bad in cases:
            with self.subTest(state=bad), self.assertRaises(ValueError): self.cells.load_state(bad)
            self.assertEqual(self.cells.state(), good)
        # Valid hash alone cannot hide a missing predecessor or skipped ordinal.
        modified = deepcopy(cell['revision']); modified['ordinal'] = 99
        ident = content_hash(modified)
        (self.cells.revisions / (ident + '.json')).write_bytes(canonical_bytes(modified))
        bad = deepcopy(good); bad['cells'][cell['id']]['head_revision'] = ident
        with self.assertRaisesRegex(ValueError, 'contiguous'): self.cells.load_state(bad)
        self.assertEqual(self.cells.state(), good)
        modified['previous_revision'] = None
        ident = content_hash(modified)
        (self.cells.revisions / (ident + '.json')).write_bytes(canonical_bytes(modified))
        bad['cells'][cell['id']]['head_revision'] = ident
        with self.assertRaisesRegex(ValueError, 'incomplete'): self.cells.load_state(bad)
        self.assertEqual(self.cells.state(), good)

    def test_history_cannot_be_spliced_from_another_cell(self):
        one, two = self.revise(self.cell(0)), self.revise(self.cell(1))
        state = self.cells.state()
        bad = deepcopy(state)
        bad['cells'][one['id']]['head_revision'] = two['head_revision']
        with self.assertRaisesRegex(ValueError, 'binding'): self.cells.load_state(bad)
        self.assertEqual(self.cells.state(), state)

    def test_tampering_old_revision_is_detected_on_resume_even_when_head_is_intact(self):
        first = self.revise(self.cell())
        self.revise(first, phase='repair')
        checkpoint = self.cells.state()
        target = self.cells.revisions / (first['head_revision'] + '.json')
        artifact = json.loads(target.read_text()); artifact['summary'] = 'altered'
        target.write_text(json.dumps(artifact))
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            CellRegistry(self.run, max_active=2).load_state(checkpoint)

    def test_bounded_safe_reads_refuse_duplicate_keys_links_fifos_and_oversize(self):
        cell = self.cell()
        target = self.cells.genesis / (cell['id'] + '.json')
        raw = target.read_bytes()
        for payload in (b'{"schema":1,"schema":2}', b' ' * (MAX_ARTIFACT_BYTES + 1)):
            target.write_bytes(payload)
            with self.assertRaises(ValueError): self.cells.get(cell['id'])
        target.unlink()
        outside = self.run.parent / 'outside.json'; outside.write_bytes(raw)
        target.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'symbolic'): self.cells.get(cell['id'])
        target.unlink(); os.mkfifo(target)
        with self.assertRaisesRegex(ValueError, 'regular'): self.cells.get(cell['id'])
        with self.assertRaises(ValueError): self.cells.get('../outside')

    def test_failed_artifact_write_does_not_advance_visible_state(self):
        cell = self.cell()
        before = self.cells.state()
        with patch.object(self.cells, '_write', side_effect=OSError('disk unavailable')):
            with self.assertRaises(OSError): self.revise(cell)
            with self.assertRaises(OSError): self.cell(9)
        self.assertEqual(self.cells.state(), before)

    def test_returned_state_and_records_are_detached_and_cards_stay_bounded(self):
        cell = self.revise(self.cell(question='Long question: ' + 'x' * 12000), summary='Latest verified observation')
        state = self.cells.state(); state['cells'].clear()
        all_cells = self.cells.all_cells(); all_cells[0]['roles']['creator'] = 9
        cell['tool_ids'].append('f' * 64); cell['trigger']['evidence']['iteration'] = -1
        retained = self.cells.get(cell['id'])
        self.assertEqual(retained['roles']['creator'], 0)
        self.assertEqual(retained['tool_ids'], ['a' * 64])
        self.assertEqual(retained['trigger']['evidence']['iteration'], 12)
        for limit in (0, 1, 50, 512, 1024, 4096):
            with self.subTest(limit=limit):
                self.assertLessEqual(len(self.cells.compact_context_cards(max_chars=limit)), limit)
        self.assertEqual(len(retained['question']), 12015)
        self.assertIn('Latest verified observation', self.cells.compact_context_cards(max_chars=512))
        self.cells.deactivate(cell['id'])
        self.assertEqual(self.cells.compact_context_cards(), '')
        self.assertIn(cell['id'], self.cells.compact_context_cards([cell['id']]))

    def test_invalid_research_references_and_unknown_values_do_not_enter_checkpoint(self):
        cell = self.cell()
        before = self.cells.state()
        bad_fields = [dict(tool_ids=['a' * 64, 'a' * 64]), dict(tool_ids=['not-a-hash']),
            dict(evidence=[{'kind': 'receipt', 'id': 'not-a-hash'}]),
            dict(evidence=[{'kind': 'receipt', 'id': 'b' * 64, 'success': True}]),
            dict(decision={'score': float('nan')}), dict(roles={'creator': 9}), dict(phase=''),
            dict(summary='x' * 8193)]
        for fields in bad_fields:
            with self.subTest(fields=fields), self.assertRaises(ValueError): self.revise(cell, **fields)
            self.assertEqual(self.cells.state(), before)
        for trigger in ({}, {'kind': 'stagnation'}, {'kind': 'stagnation', 'score': float('inf')}):
            with self.subTest(trigger=trigger), self.assertRaises(ValueError):
                self.cells.ensure_cell(99, 'new question', trigger)
        self.assertEqual(self.cells.state(), before)


if __name__ == '__main__':
    unittest.main()
