"""C-only source projections. Exact text binding is not semantic verification.

Ordinary records retain their historical bounded excerpt. Complete source pairs
live only with queued/detached C jobs and retained conclusions, never lessons.
Offsets are Python Unicode character offsets in the exact full unified diff;
windows are individual contiguous substrings, never code spliced across gaps.
"""
from __future__ import annotations

from bisect import bisect_right
from copy import deepcopy
import difflib
import hashlib
import json
from pathlib import Path

SCHEMA = 'sciel-conclusion-source/1'
BINDING_SCHEMA = 'sciel-conclusion-source-binding/1'
MAX_WINDOWS = 12
SOURCE_FIELDS = ('conclusion_source.diff',) + tuple(
    f'conclusion_source.windows.{i}.text' for i in range(MAX_WINDOWS))
RULES = ('conclusion_source is the C-only source-bound diff view. Its offsets/ranges '
    'refer to the full diff; each window is a separate exact substring. Omitted '
    'ranges are unknown, never join windows into code. Full hashes bind text, '
    'not the truth of a mechanism claim. Quote only supplied diff/window fields.\n')


def _json(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(',', ':'), allow_nan=False)


def sha(text):
    return hashlib.sha256(text.encode('utf-8', errors='surrogatepass')).hexdigest()


def identity():
    return dict(schema=SCHEMA, binding=BINDING_SCHEMA,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def full_diff(parent, candidate):
    return ''.join(difflib.unified_diff(parent.splitlines(keepends=True),
        candidate.splitlines(keepends=True), fromfile='evaluated_parent', tofile='candidate', n=2))


def _layout(text):
    from .conclusion_refs import _diff_line_spans
    lines = list(_diff_line_spans(text))
    changed = [(start, end) for start, end, metadata in lines
               if not metadata and text[start:start+1] in ('+', '-')]
    return lines, changed


def _gaps(length, spans):
    result, previous = [], 0
    for start, end in spans:
        if previous < start:
            result.append([previous, start])
        previous = end
    if previous < length:
        result.append([previous, length])
    return result


def project(parent, candidate, *, window_chars=None, window_count=MAX_WINDOWS):
    """Full diff or stratified windows over *all* changed text, including one hunk.

    Quantiles are selected from cumulative changed-character coordinates, not
    merely hunk starts. Endpoints are included when at least two windows fit.
    Allocation is generic; no optimizer names, syntax or scores are inspected.
    """
    if type(parent) is not str or type(candidate) is not str:
        raise ValueError('conclusion sources must be strings')
    if window_chars is not None and (type(window_chars) is not int or window_chars < 1):
        raise ValueError('invalid conclusion source window size')
    if type(window_count) is not int or not 1 <= window_count <= MAX_WINDOWS:
        raise ValueError('invalid conclusion source window count')
    text = full_diff(parent, candidate)
    lines, changed = _layout(text)
    changed_chars = sum(end-start for start, end in changed)
    result = dict(schema=SCHEMA, parent_source_sha256=sha(parent), source_sha256=sha(candidate),
        full_diff_sha256=sha(text), full_diff_chars=len(text), full_diff_lines=len(lines),
        changed_lines=len(changed), changed_chars=changed_chars,
        selection=dict(window_chars=window_chars, window_count=window_count),
        mode='full' if window_chars is None else 'windows')
    if window_chars is None:
        spans = [(0, len(text))] if text else []
        result['diff'] = text
    else:
        cumulative, total = [], 0
        for start, end in changed:
            total += end-start
            cumulative.append(total)
        centers = []
        count = min(window_count, total)
        for i in range(count):
            offset = ((total-1)*i//(count-1)) if count > 1 else (total-1)//2
            index = bisect_right(cumulative, offset)
            previous = cumulative[index-1] if index else 0
            centers.append(changed[index][0] + offset-previous)
        spans = []
        for center in centers:
            start = max(0, min(center-window_chars//2, len(text)-window_chars))
            end = min(len(text), start+window_chars)
            if spans and start <= spans[-1][1]:
                spans[-1] = (spans[-1][0], max(end, spans[-1][1]))
            else:
                spans.append((start, end))
        line_starts = [row[0] for row in lines]
        result['windows'] = [dict(start=start, end=end, text=text[start:end],
            sha256=sha(text[start:end]), diff_line_start=bisect_right(line_starts, start),
            diff_line_end=bisect_right(line_starts, end-1)) for start, end in spans]
    shown_changes = sum(max(0, min(end, right)-max(start, left))
        for start, end in spans for left, right in changed)
    result['coverage'] = dict(shown_diff_chars=sum(end-start for start, end in spans),
        shown_changed_chars=shown_changes, omitted_changed_chars=changed_chars-shown_changes,
        omitted_diff_ranges=_gaps(len(text), spans), complete=not _gaps(len(text), spans))
    return result


def fields(projection):
    """Validate the finite projection shape before exposing literal quote paths.

    Authenticating a partial projection additionally requires its retained pair
    via validate_binding; this function cannot authenticate absent full bytes.
    """
    common = {'schema', 'parent_source_sha256', 'source_sha256', 'full_diff_sha256',
        'full_diff_chars', 'full_diff_lines', 'changed_lines', 'changed_chars',
        'selection', 'mode', 'coverage'}
    if type(projection) is not dict or projection.get('schema') != SCHEMA:
        raise ValueError('invalid conclusion source schema')
    full = projection.get('mode') == 'full'
    if set(projection) != common | ({'diff'} if full else {'windows'}):
        raise ValueError('invalid conclusion source projection fields')
    selection = projection['selection']
    if (type(selection) is not dict or set(selection) != {'window_chars', 'window_count'}
            or type(selection['window_count']) is not int
            or not 1 <= selection['window_count'] <= MAX_WINDOWS
            or (selection['window_chars'] is not None if full else
                type(selection['window_chars']) is not int or selection['window_chars'] < 1)):
        raise ValueError('invalid conclusion source selection')
    if full:
        text = projection['diff']
        if type(text) is not str or len(text) != projection['full_diff_chars'] or sha(text) != projection['full_diff_sha256']:
            raise ValueError('conclusion source diff binding mismatch')
        return {'conclusion_source.diff': text} if text.strip() else {}
    if projection['mode'] != 'windows' or type(projection['windows']) is not list or len(projection['windows']) > MAX_WINDOWS:
        raise ValueError('invalid conclusion source windows')
    result, previous = {}, 0
    for i, window in enumerate(projection['windows']):
        if (type(window) is not dict or set(window) != {'start', 'end', 'text', 'sha256', 'diff_line_start', 'diff_line_end'}
                or type(window['start']) is not int or type(window['end']) is not int
                or not previous <= window['start'] < window['end'] <= projection['full_diff_chars']
                or type(window['text']) is not str or len(window['text']) != window['end']-window['start']
                or sha(window['text']) != window['sha256']):
            raise ValueError('conclusion source window binding mismatch')
        previous = window['end']
        if window['text'].strip():
            result[f'conclusion_source.windows.{i}.text'] = window['text']
    return result


def bind(record, parent, candidate, projection):
    value = dict(schema=BINDING_SCHEMA, parent_source=parent, candidate_source=candidate,
                 projection=deepcopy(projection))
    validate_binding(record, value)
    return value


def validate_binding(record, value):
    if (type(value) is not dict or set(value) != {'schema', 'parent_source', 'candidate_source', 'projection'}
            or value['schema'] != BINDING_SCHEMA or type(value['parent_source']) is not str
            or type(value['candidate_source']) is not str):
        raise ValueError('invalid conclusion source binding')
    parent, candidate = value['parent_source'], value['candidate_source']
    if sha(parent) != record.get('parent_source_sha256') or sha(candidate) != record.get('source_sha256'):
        raise ValueError('conclusion source differs from evaluated source identities')
    projection = value['projection']
    fields(projection)
    expected = project(parent, candidate, **projection['selection'])
    if _json(expected) != _json(projection):
        raise ValueError('conclusion source projection differs from exact source pair')
    return deepcopy(projection)


def validate_job(job):
    """Validate detached data too; a quoted window never authenticates itself."""
    value = job.get('source_evidence')
    if value is None:
        if 'conclusion_source' in job['evidence']:
            raise ValueError('conclusion source missing retained source binding')
        return
    projection = validate_binding(job['record'], value)
    if _json(job['evidence'].get('conclusion_source')) != _json(projection):
        raise ValueError('conclusion source job evidence mismatch')
