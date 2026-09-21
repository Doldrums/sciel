"""Bounded exact source spans and structural unified-diff metadata."""
from __future__ import annotations

import re

_LINE_END = re.compile(r'\r\n|[\n\r\v\f\x1c-\x1e\x85\u2028\u2029]')
_HUNK = re.compile(r'@@ -[0-9]+(?:,([0-9]+))? \+[0-9]+(?:,([0-9]+))? @@(?:[^\r\n]*)')
_RANGE = r'[0-9]+(?:,[0-9]+)?'
# Only proper prefixes, at EOF after an established diff boundary. A malformed
# header is not a prefix, and raw source/control-looking text is not a boundary.
_HUNK_PREFIX = re.compile(r'@@ -(?:[0-9]*|[0-9]+,[0-9]*|'
    + _RANGE + r' |' + _RANGE + r' \+[0-9]*|'
    + _RANGE + r' \+[0-9]+,[0-9]*|'
    + _RANGE + r' \+' + _RANGE + r' @?)')


def _line_spans(text):
    start = 0
    for match in _LINE_END.finditer(text):
        yield start, match.end(), match.start()
        start = match.end()
    if start < len(text):
        yield start, len(text), len(text)


def _diff_line_spans(text):
    """Classify only structural unified-diff controls, with original offsets.

    Hunk counts protect removed/added source lines beginning with -- or ++
    from being mistaken for the next file's preamble. An incomplete hunk is
    allowed: its remaining source lines stay eligible. Malformed/partial
    control-looking lines are preserved rather than guessed away.
    """
    lines = iter(_line_spans(text))
    pending = next(lines, None)
    old = new = 0
    after_body = at_boundary = False

    def count(value):
        if value is None:
            return 1
        # Counts larger than this text cannot be exhausted. Avoid unbounded
        # integer parsing for pathological but syntactically numeric headers.
        digits = value.lstrip('0') or '0'
        limit = len(text) + 1
        return limit if len(digits) > len(str(limit)) else min(int(digits), limit)

    while pending is not None:
        start, end, content_end = pending
        pending = None
        line = text[start:content_end]
        hunk = _HUNK.fullmatch(line)
        if hunk is not None:
            old, new = (count(value) for value in hunk.groups())
            after_body = False
            at_boundary = not old and not new
            yield start, end, True
        elif (at_boundary and content_end == end == len(text)
                and _HUNK_PREFIX.fullmatch(line) is not None):
            # A bounded excerpt can end part-way into the next hunk header.
            # Never infer this inside an unfinished hunk or without a known
            # preceding file/hunk boundary, and never strip a diff body prefix.
            yield start, end, True
        elif after_body and line == '\\ No newline at end of file':
            after_body = False
            yield start, end, True
        elif ((line.startswith(' ') and old and new)
                or (line.startswith('-') and old)
                or (line.startswith('+') and new)):
            old -= int(line[0] in ' -')
            new -= int(line[0] in ' +')
            after_body = True
            at_boundary = not old and not new
            yield start, end, False
        elif not old and not new and line.startswith('--- '):
            following = next(lines, None)
            if following is not None and text[following[0]:following[2]].startswith('+++ '):
                yield start, end, True
                yield following[0], following[1], True
                at_boundary = True
            else:
                yield start, end, False
                pending = following
                at_boundary = False
            after_body = False
        else:
            # No resynchronization from ambiguous metadata-shaped text. A
            # later complete hunk header can establish a new known boundary.
            after_body = False
            at_boundary = False
            yield start, end, False
        if pending is None:
            pending = next(lines, None)


def _line_fragments(text, start, end):
    from . import conclusion_refs as _public
    while start < end:
        stop = min(end, start + _public.MAX_FRAGMENT_CHARS)
        if stop < end:
            boundary = text.rfind(' ', start + _public.MAX_FRAGMENT_CHARS // 2, stop)
            if boundary > start:
                stop = boundary
        if text[start:stop].strip():
            yield start, stop
        start = stop


def _fragment_spans(text):
    """Yield bounded exact spans without allocating a list of every source line."""
    for start, end, _ in _line_spans(text):
        yield from _line_fragments(text, start, end)


def _fragment(field, text, kind, start, stop):
    from . import conclusion_refs as _public
    piece = text[start:stop]
    return dict(id=kind[0] + _public._sha(_public._json([field, start, stop, piece]))[:20],
                field=field, start=start, end=stop, text=piece)


def _fragments(field, text, kind):
    """Exact bounded substrings with offsets; never execute or rewrite source."""
    from . import conclusion_refs as _public
    for start, stop in _fragment_spans(text):
        yield _public._fragment(field, text, kind, start, stop)


def _sample(items, count):
    if count <= 0:
        return []
    if len(items) <= count:
        return items
    if count == 1:
        return items[:1]
    # Even coverage includes the end; a large leading helper cannot hide every
    # later edit. Omitted fragments are counted, never described as a full diff.
    return [items[i * (len(items) - 1) // (count - 1)] for i in range(count)]


def _sample_fragments(field, text, kind, count):
    # Count cheap offsets first; materialize/hash only the bounded selected set.
    # range is lazy, so even a pathological one-character-per-line field does
    # not allocate one dictionary or index per fragment.
    from . import conclusion_refs as _public
    total = sum(1 for _ in _fragment_spans(text))
    chosen = set(_sample(range(total), count))
    picked = [_public._fragment(field, text, kind, start, stop)
              for index, (start, stop) in enumerate(_fragment_spans(text)) if index in chosen]
    return picked, total


def _sample_diff_fragments(field, text, kind, count):
    from . import conclusion_refs as _public
    total = excluded_lines = excluded_chars = excluded_fragments = 0
    for start, end, metadata in _diff_line_spans(text):
        fragments = sum(1 for _ in _line_fragments(text, start, end))
        total += fragments
        if metadata:
            excluded_lines += 1
            excluded_chars += end - start
            excluded_fragments += fragments
    eligible = total - excluded_fragments
    chosen = set(_sample(range(eligible), count))
    picked, index = [], 0
    for start, end, metadata in _diff_line_spans(text):
        if metadata:
            continue
        for begin, stop in _line_fragments(text, start, end):
            if index in chosen:
                picked.append(_public._fragment(field, text, kind, begin, stop))
            index += 1
    return picked, total, dict(eligible_fragments=eligible,
        excluded_metadata_lines=excluded_lines, excluded_metadata_chars=excluded_chars,
        excluded_metadata_fragments=excluded_fragments)


