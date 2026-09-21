"""Diff controls are not code citations; projection preserves literal provenance."""
from copy import deepcopy
import hashlib
import json

import pytest

from sciel.hypothesis import conclusion_refs as refs
from test_conclusion_repair_schema import matches


DIFF = ('--- evaluated_parent\n+++ candidate\n@@ -1,2 +1,2 @@\n'
        '-num_starts = 5\n+num_starts = 10\n return num_starts\n')


def evidence(text=DIFF):
    return dict(id='r1', hypothesis=dict(text='Increase the start cap.',
        reasoning='More starts might help.', predicted_low=0.0, predicted_high=0.1),
        change_excerpt=text, status='evaluated', parent_score=0.5, score=0.55,
        actual_delta=0.55 - 0.5, source_repaired=False,
        proposed_sha256='a' * 64, source_sha256='a' * 64)


def payload(data, source_id=None, *, kind='source_effect', assessment='directly_supported'):
    claim = refs.catalog(data)['claims'][0]['id']
    return json.dumps(dict(verdict='CONFIRMED', explanation='The delta is inside the forecast.',
        lessons=[], mechanism_review=[dict(claim_ref=None if kind == 'forecast' else claim,
            source_refs=[] if source_id is None else [source_id], assessment=assessment,
            direct_effect='A bounded source interpretation.',
            untested_downstream='Causality has not been established.', evidence_kind=kind)]))


def source_rows(data, *, schema=refs.SCHEMA):
    return [x for x in refs.catalog(data, schema=schema)['sources']
            if x['field'] == 'change_excerpt']


def metadata(data):
    return next(x for x in refs.catalog(data)['fields'] if x['field'] == 'change_excerpt')


def test_real_code_citations_replace_file_and_hunk_controls_without_rewriting_spans():
    data = evidence()
    before = deepcopy(data)
    selected = source_rows(data)
    assert [x['text'] for x in selected] == [
        '-num_starts = 5\n', '+num_starts = 10\n', ' return num_starts\n']
    legacy = {x['text']: x for x in source_rows(data, schema=refs.LEGACY_SCHEMA)}
    for row in selected:
        assert row == legacy[row['text']]
        assert data['change_excerpt'][row['start']:row['end']] == row['text']
        assert refs.decode(payload(data, row['id']), data)['error'] is None
    meta = metadata(data)
    assert meta['sha256'] == hashlib.sha256(DIFF.encode()).hexdigest()
    assert (meta['total_fragments'], meta['eligible_fragments'], meta['shown_fragments'],
            meta['omitted_fragments'], meta['excluded_metadata_lines'],
            meta['excluded_metadata_fragments']) == (6, 3, 3, 0, 3, 3)
    assert data == before


@pytest.mark.parametrize('newline', ['\n', '\r\n', '\r'])
def test_body_lines_resembling_file_headers_survive_hunk_counts_and_multiple_files(newline):
    lines = ['--- a', '+++ b', '@@ -1,2 +1,2 @@', '--- removed_source',
             '+++ added_source', ' context', '--- c', '+++ d',
             '@@ -3 +7 @@ section', '-old', '+new']
    data = evidence(newline.join(lines) + newline)
    rows = source_rows(data)
    assert [x['text'] for x in rows] == [line + newline for line in
        ['--- removed_source', '+++ added_source', ' context', '-old', '+new']]
    assert metadata(data)['excluded_metadata_lines'] == 6
    for row in rows:
        assert row['text'] == data['change_excerpt'][row['start']:row['end']]


def test_no_newline_control_requires_a_preceding_hunk_body_line():
    marker = '\\ No newline at end of file'
    data = evidence(marker + '\n@@ -1 +1 @@\n-old\n' + marker + '\n+new\n' + marker)
    assert [x['text'] for x in source_rows(data)] == [marker + '\n', '-old\n', '+new\n']
    assert metadata(data)['excluded_metadata_lines'] == 3


@pytest.mark.parametrize('text,kept', [
    ('--- a\n+++ b\n@@ -1,20 +1,30 @@\n-old\n+partial', ['-old\n', '+partial']),
    ('@@ -1,2 +1,2 @@\n--- source\n+++ source', ['--- source\n', '+++ source']),
    ('--- unpaired\n+ordinary\n', ['--- unpaired\n', '+ordinary\n']),
    ('@@ -1, +2 @@\n+ordinary\n', ['@@ -1, +2 @@\n', '+ordinary\n']),
    ('--- a\n+++ b\n@@ -1,2 X', ['@@ -1,2 X']),
    ('@@ -1 +1 @@\n-old\n+new\n\\ No newline at end',
     ['-old\n', '+new\n', '\\ No newline at end']),
])
def test_truncated_hunks_preserve_source_and_uncertain_control_shaped_text(text, kept):
    assert [x['text'] for x in source_rows(evidence(text))] == kept


@pytest.mark.parametrize('partial', ['@@ -', '@@ -1', '@@ -1,', '@@ -1,20 ',
    '@@ -1,20 +', '@@ -1,20 +2,', '@@ -1,20 +2,30 ', '@@ -1,20 +2,30 @'])
def test_eof_hunk_prefix_is_control_only_at_an_established_diff_boundary(partial):
    for boundary in ['--- a\n+++ b\n', '@@ -1 +1 @@\n-old\n+new\n',
                     '@@ -1 +1 @@\n-old\n+new\n\\ No newline at end of file\n']:
        data = evidence(boundary + partial)
        assert all(partial != r['text'] for r in source_rows(data))
        assert metadata(data)['excluded_metadata_chars'] >= len(partial)
    for prefix in ['', '@@ -1,2 +1,2 @@\n-old\n+new\n',
                   '@@ -1 +1 @@\n-old\n+new\nuncertain\n']:
        assert any(r['text'] == partial for r in source_rows(evidence(prefix + partial)))
    # A complete line, or the content of a real added source line, is not an
    # excerpt cut inside the next header.
    assert source_rows(evidence('--- a\n+++ b\n' + partial + '\n'))[-1]['text'] == partial + '\n'
    assert source_rows(evidence('@@ -0,0 +1 @@\n+' + partial))[-1]['text'] == '+' + partial


def test_zero_counts_and_large_counts_do_not_confuse_hunk_body_with_file_headers():
    text = '@@ -0,0 +1,2 @@\n+++ first\n+++ second\n'
    assert [r['text'] for r in source_rows(evidence(text))] == ['+++ first\n', '+++ second\n']
    text = '@@ -1,' + '9' * 5000 + ' +0,0 @@\n--- still removed source\n'
    assert [r['text'] for r in source_rows(evidence(text))] == ['--- still removed source\n']


def test_long_control_lines_excluded_before_chunking_and_only_selected_rows_materialized(monkeypatch):
    text = ('--- ' + 'a' * 1200 + '\n+++ ' + 'b' * 1300 + '\n'
            '@@ -1,100 +1,100 @@ ' + 'section' * 200 + '\n'
            + ''.join(' code_' + str(n) + '\n' for n in range(100)))
    data = evidence(text)
    original, made = refs._fragment, []
    def counted(*args):
        made.append(args)
        return original(*args)
    monkeypatch.setattr(refs, '_fragment', counted)
    prepared = refs.catalog(data)
    rows = [r for r in prepared['sources'] if r['field'] == 'change_excerpt']
    meta = next(x for x in prepared['fields'] if x['field'] == 'change_excerpt')
    assert len(rows) == 32 and len(made) == len(prepared['claims']) + len(rows)
    assert rows[0]['text'] == ' code_0\n' and rows[-1]['text'] == ' code_99\n'
    assert meta['excluded_metadata_lines'] == 3
    assert meta['excluded_metadata_fragments'] > 3
    assert meta['eligible_fragments'] == 100 and meta['omitted_fragments'] == 68
    assert meta['total_fragments'] == meta['eligible_fragments'] + meta['excluded_metadata_fragments']


def test_comments_docstrings_and_arbitrary_source_bodies_are_not_semantically_filtered():
    body = '--- text\n+++ text\n@@ -1 +1 @@\n\\ No newline at end of file\n'
    data = evidence('@@ -1,2 +1,2 @@\n-# old comment\n+# new comment\n """docstring"""\n')
    data['controlled_change'] = dict(target_body=body)
    prepared = refs.catalog(data)
    body_rows = [r for r in prepared['sources'] if r['field'] == 'controlled_change.target_body']
    assert ''.join(r['text'] for r in body_rows) == body
    assert [r['text'] for r in prepared['sources'] if r['field'] == 'change_excerpt'] == [
        '-# old comment\n', '+# new comment\n', ' """docstring"""\n']
    body_meta = next(x for x in prepared['fields'] if x['field'] == 'controlled_change.target_body')
    assert body_meta['excluded_metadata_fragments'] == 0


def test_controls_only_catalog_has_no_supported_source_variant_but_unknown_and_forecast_work():
    data = evidence('--- a\n+++ b\n@@ -0,0 +0,0 @@\n')
    assert source_rows(data) == []
    meta = metadata(data)
    assert meta['eligible_fragments'] == meta['omitted_fragments'] == 0
    review = refs.evidence_response_format(data)['json_schema']['schema']['properties']['mechanism_review']
    for kind, assessment in [('source_effect', 'unknown'), ('unavailable', 'unknown'),
                              ('forecast', 'directly_supported')]:
        text = payload(data, kind=kind, assessment=assessment)
        assert matches(review, json.loads(text)['mechanism_review'])
        assert refs.decode(text, data)['error'] is None
    assert not matches(review, json.loads(payload(data))['mechanism_review'])
    assert refs.decode(payload(data), data)['conclusion'] is None


def test_old_header_id_is_rejected_currently_and_only_explicit_v1_reproduces_old_review():
    data = evidence()
    old = refs.catalog(data, schema=refs.LEGACY_SCHEMA)
    header = next(x for x in old['sources'] if x['text'].startswith('@@'))
    text = payload(data, header['id'])
    rejected = refs.decode(text, data)
    assert rejected['conclusion'] is None
    assert header['id'] not in rejected['error']['reference_repair']['allowed_source_ids']
    accepted = refs.decode(text, data, schema=refs.LEGACY_SCHEMA)
    assert accepted['error'] is None
    assert accepted['resolution']['schema'] == refs.LEGACY_SCHEMA
    assert accepted['resolution']['catalog_sha256'] == refs._sha(refs._json(old))
    assert accepted['conclusion'].mechanism_review[0]['source_quotes'][0]['text'] == header['text']
    assert refs._json(old) in refs.instruction(data, schema=refs.LEGACY_SCHEMA)
    assert all('excluded_metadata_fragments' not in f for f in old['fields'])
    assert old['fields'][-1]['shown_fragments'] == old['fields'][-1]['total_fragments'] == 6


@pytest.mark.parametrize('schema,suffix', [(refs.SCHEMA, 'v2'), (refs.LEGACY_SCHEMA, 'v1')])
def test_initial_repair_and_literal_wire_names_pin_the_selected_catalog(schema, suffix):
    data = evidence()
    assert refs.response_format(schema=schema)['json_schema']['name'] == 'sciel_conclusion_refs_' + suffix
    initial = refs.evidence_response_format(data, schema=schema)
    repair = refs.repair_response_format(data, schema=schema)
    assert initial['json_schema']['name'] == 'sciel_conclusion_evidence_refs_' + suffix
    assert repair['json_schema']['name'] == 'sciel_conclusion_repair_refs_' + suffix
    initial['json_schema']['name'] = repair['json_schema']['name']
    assert initial == repair
    header = next(x for x in refs.catalog(data, schema=refs.LEGACY_SCHEMA)['sources']
                  if x['text'].startswith('@@'))
    assert (header['id'] in json.dumps(repair)) == (schema == refs.LEGACY_SCHEMA)


@pytest.mark.parametrize('operation', [
    lambda s: refs.catalog(evidence(), schema=s),
    lambda s: refs.decode(payload(evidence()), evidence(), schema=s),
    lambda s: refs.instruction(evidence(), schema=s),
    lambda s: refs.response_format(schema=s),
    lambda s: refs.evidence_response_format(evidence(), schema=s),
    lambda s: refs.repair_response_format(evidence(), schema=s),
])
def test_unknown_catalog_identity_is_not_implicitly_reconstructed(operation):
    with pytest.raises(ValueError, match='unsupported conclusion reference'):
        operation('sciel-conclusion-source-refs/unknown')
