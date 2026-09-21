"""Bounded text-authoring lineage, independent of observational support.

Hashes bind recorded bytes/structure; without the separately retained response or
journal they do not authenticate a model response or prove the text's semantics.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import re

from .catalogs import canonical_json, canonical_sha

SCHEMA = 'sciel-lesson-authoring/1'
MAX_NODES = 64
MAX_CHARS = 32768
LIMITS = ('Authoring facts identify which recorded response proposed wording and which '
          'retired wording it replaced. They are not observations, mechanism verification, '
          'independent support, or advice exposures. Missing authoring data means unknown. '
          'A response digest does not promise that its complete text remains available; '
          'the auxiliary journal may have retained only a truncated response.')
_FIELDS = {'schema', 'writer_refs', 'revision_refs', 'prior_writers_unknown', 'catalog'}
_WRITER = {'kind', 'phase', 'reference', 'raw_response_sha256', 'proposal_index',
           'raw_rule', 'raw_rule_sha256', 'stored_rule_sha256'}
_REVISION = {'kind', 'lesson_id', 'rule', 'rule_sha256', 'polarity', 'scope', 'branch',
             'evidence_ids', 'writer_refs', 'revision_refs', 'prior_writers_unknown'}


def require(ok, message):
    if not ok:
        raise ValueError('lesson authoring: ' + message)


def text_sha(value):
    return hashlib.sha256(value.encode('utf-8', errors='surrogatepass')).hexdigest()


def _hash(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def validate_context(value):
    require(type(value) is dict and set(value) == {'phase', 'reference', 'raw_response_sha256'}, 'invalid writer context')
    require(value['phase'] in ('conclusion', 'reflection'), 'invalid writer phase')
    ref = value['reference']
    require((type(ref) is str and 0 < len(ref) <= 128 and not any(c.isspace() for c in ref))
            if value['phase'] == 'conclusion' else (type(ref) is int and ref > 0), 'invalid writer reference')
    require(_hash(value['raw_response_sha256']), 'invalid response digest')
    return deepcopy(value)


def _empty(*, unknown):
    return dict(schema=SCHEMA, writer_refs=[], revision_refs=[], prior_writers_unknown=unknown, catalog={})


def _refs(value):
    require(type(value) is list and all(_hash(ref) for ref in value)
            and value == sorted(set(value)), 'invalid or duplicate references')


def validate(value, *, rule, lesson_id=None, evidence=None, next_lesson_id=None, polarity=None):
    require(type(value) is dict and set(value) == _FIELDS and value['schema'] == SCHEMA, 'invalid schema')
    catalog = value['catalog']
    require(type(catalog) is dict and 0 < len(catalog) <= MAX_NODES, 'catalog capacity exceeded')
    require(len(canonical_json(value)) <= MAX_CHARS, 'serialized capacity exceeded')
    require(type(value['prior_writers_unknown']) is bool, 'invalid unknown-writer flag')
    revisions, writers = set(), {}
    for ref, node in catalog.items():
        require(_hash(ref) and type(node) is dict and canonical_sha(node) == ref, 'catalog digest mismatch')
        if node.get('kind') == 'writer':
            require(set(node) == _WRITER, 'invalid writer fields')
            validate_context({k: node[k] for k in ('phase', 'reference', 'raw_response_sha256')})
            if evidence is not None and node['phase'] == 'conclusion':
                require(node['reference'] in evidence, 'writer observation reference unavailable')
            require(type(node['proposal_index']) is int and 0 <= node['proposal_index'] < 2**31, 'invalid proposal index')
            raw = node['raw_rule']
            require(type(raw) is str and bool(raw.strip()) and len(raw) <= 4096, 'invalid literal rule')
            origin = (node['phase'], node['reference'], node['raw_response_sha256'], node['proposal_index'])
            require(origin not in writers or writers[origin] == raw, 'conflicting writer/proposal identity')
            writers[origin] = raw
            require(node['raw_rule_sha256'] == text_sha(raw)
                    and node['stored_rule_sha256'] == text_sha(raw.strip()), 'literal/stored rule digest mismatch')
        else:
            require(node.get('kind') == 'retired_lesson' and set(node) == _REVISION, 'invalid revision fields')
            ident = node['lesson_id']
            require(type(ident) is str and re.fullmatch(r'l\d{6,}', ident)
                    and ident != lesson_id and ident not in revisions, 'invalid retired lesson identity')
            require(next_lesson_id is None or 0 < int(ident[1:]) < next_lesson_id, 'retired lesson counter mismatch')
            revisions.add(ident)
            require(type(node['rule']) is str and bool(node['rule']) and len(node['rule']) <= 4096
                    and node['rule'] == node['rule'].strip()
                    and node['rule_sha256'] == text_sha(node['rule']), 'retired literal rule digest mismatch')
            require(_hash(node['rule_sha256']) and node['polarity'] in ('positive', 'negative')
                    and node['scope'] in ('branch', 'global') and type(node['prior_writers_unknown']) is bool,
                    'invalid revision facts')
            require(polarity is None or node['polarity'] == polarity, 'revision polarity differs from retained support')
            ids = node['evidence_ids']
            require(type(ids) is list and bool(ids) and all(type(i) is str and 0 < len(i) <= 128 for i in ids)
                    and ids == sorted(set(ids)), 'invalid revision evidence IDs')
            require((node['branch'] is None) if node['scope'] == 'global' else
                    (type(node['branch']) is str and bool(node['branch'])), 'invalid revision branch')
            if evidence is not None:
                require(set(ids) <= set(evidence), 'revision evidence missing from retained support')
                branches = {evidence[i]['branch'] for i in ids}
                require((len(branches) >= 2 and node['branch'] is None) if node['scope'] == 'global'
                        else branches == {node['branch']}, 'revision scope differs from evidence')
    visited, active = set(), set()
    def visit(owner, rule_hash):
        _refs(owner['writer_refs']); _refs(owner['revision_refs'])
        require(owner['writer_refs'] or owner['prior_writers_unknown'], 'known author missing')
        for ref in owner['writer_refs']:
            node = catalog.get(ref)
            require(node is not None and node['kind'] == 'writer'
                    and node['stored_rule_sha256'] == rule_hash, 'writer/current rule binding mismatch')
            visited.add(ref)
        for ref in owner['revision_refs']:
            node = catalog.get(ref)
            require(node is not None and node['kind'] == 'retired_lesson', 'missing revision reference')
            require(ref not in active, 'cyclic revision references')
            if ref not in visited:
                active.add(ref); visit(node, node['rule_sha256']); active.remove(ref); visited.add(ref)
    visit(value, text_sha(rule))
    require(visited == set(catalog), 'unreachable authoring nodes')
    return deepcopy(value)


def add_writer(lesson, context, raw_rule, index, *, existed):
    """Mutate only a caller's detached proposed lesson; no KB commits happen here."""
    if context is None and 'authoring' not in lesson:
        return
    data = deepcopy(lesson.get('authoring', _empty(unknown=existed)))
    if context is None:
        # An old direct caller added support without identifying its text writer.
        data['prior_writers_unknown'] = True
    else:
        context = validate_context(context)
        node = dict(kind='writer', **context, proposal_index=index, raw_rule=raw_rule,
                    raw_rule_sha256=text_sha(raw_rule), stored_rule_sha256=text_sha(raw_rule.strip()))
        ref = canonical_sha(node)
        data['catalog'][ref] = node
        data['writer_refs'] = sorted(set(data['writer_refs']) | {ref})
    lesson['authoring'] = validate(data, rule=lesson['rule'], lesson_id=lesson['id'])


def validate_collection(lessons):
    """Cross-check individually validated catalogs before any KB commit."""
    current_ids = {lesson['id'] for lesson in lessons}
    writers, revisions, retired_rules = {}, {}, {}
    for lesson in lessons:
        for ref, node in lesson.get('authoring', {}).get('catalog', {}).items():
            if node['kind'] == 'writer':
                origin = (node['phase'], node['reference'], node['raw_response_sha256'], node['proposal_index'])
                require(origin not in writers or writers[origin] == node['raw_rule'],
                        'conflicting writer/proposal identity across lessons')
                writers[origin] = node['raw_rule']
            else:
                ident = node['lesson_id']
                require(ident not in current_ids, 'retired lesson is still current')
                require(ident not in revisions or revisions[ident] == ref,
                        'conflicting immutable retired lesson across catalogs')
                revisions[ident] = ref
                retired_rules[ident] = node['rule']
    for lesson in lessons:
        for item in lesson['inherited_exposures']:
            ident = item['from_lesson_id']
            require(ident not in retired_rules or retired_rules[ident] == item['from_rule'],
                    'retired literal differs from inherited exposure origin')


def inherit(target, prior, *, enabled):
    if not enabled and 'authoring' not in target and 'authoring' not in prior:
        return  # Preserve the legacy direct-API behavior for legacy-only input.
    data = deepcopy(target.get('authoring', _empty(unknown=True)))
    old = deepcopy(prior.get('authoring', _empty(unknown=True)))
    for ref, node in old['catalog'].items():
        require(ref not in data['catalog'] or data['catalog'][ref] == node, 'conflicting immutable authoring node')
        data['catalog'][ref] = node
    node = dict(kind='retired_lesson', lesson_id=prior['id'], rule=prior['rule'], rule_sha256=text_sha(prior['rule']),
                polarity=prior['polarity'], scope=prior['scope'], branch=prior['branch'],
                evidence_ids=sorted(prior['evidence_ids']), writer_refs=old['writer_refs'],
                revision_refs=old['revision_refs'], prior_writers_unknown=old['prior_writers_unknown'])
    ref = canonical_sha(node); data['catalog'][ref] = node
    data['revision_refs'] = sorted(set(data['revision_refs']) | {ref})
    target['authoring'] = validate(data, rule=target['rule'], lesson_id=target['id'])


def view(lesson):
    if 'authoring' not in lesson:
        return None
    return {'provenance': validate(lesson['authoring'], rule=lesson['rule'], lesson_id=lesson['id']),
            'interpretation_limits': LIMITS}
