"""Native JSON-schema envelopes for opt-in research auxiliary responses.

The wire uses Claude's supported closed structure/types/enums and minItems 0/1.
String lengths and other array constraints are descriptive on the wire and
strictly enforced locally, as are cross-field and total encoded limits. Source
bindings, forecast checks and lesson admission remain authoritative afterward.
See https://platform.claude.com/docs/en/build-with-claude/structured-outputs
(JSON Schema limitations, checked 2026-09-17). No provider probe or repair here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import mechanism_review as mechanism
from . import parsing

VERSION = 'sciel-auxiliary-json-schema/2'
_KINDS = ('conclusion', 'reflection')


def _object(properties, **extra):
    return dict(type='object', properties=properties, required=list(properties),
                additionalProperties=False, **extra)


def _text(maximum):
    return dict(type='string', description=f'Nonblank text of 1 to {maximum} characters. '
                'The exact length and nonblank requirements are enforced locally.')


def _identifier():
    return dict(type='string', description=f'Identifier of 1 to {parsing.MAX_ID_CHARS} '
                'characters, with no whitespace or character below U+0020. '
                'The exact identifier requirements are enforced locally.')


def _ids(maximum, *, minimum=0):
    return dict(type='array', items=_identifier(), minItems=minimum,
                description=f'Between {minimum} and {maximum} distinct identifiers; '
                            'duplicates and excess items are rejected locally.')


def _lessons():
    branch = _identifier()
    branch['type'] = ['string', 'null']
    branch['description'] += (' Use the exact branch ID for branch scope, null for global scope. '
                             'The local parser enforces this scope/branch relationship.')
    lesson = _object(dict(
        rule=_text(parsing.MAX_TEXT_CHARS),
        polarity=dict(type='string', enum=['positive', 'negative']),
        scope=dict(type='string', enum=['branch', 'global']),
        evidence_ids=_ids(parsing.MAX_EVIDENCE_IDS, minimum=1),
        branch=branch))
    return dict(type='array', items=lesson,
        description=f'At most {parsing.MAX_LESSONS} lessons; the count is enforced locally. '
                    'Only supplied evidence IDs may support a lesson. Schema validity '
                    'does not establish evidence membership or lesson admission.')


def _mechanism_review():
    claim = _text(mechanism.MAX_TEXT_CHARS)
    claim['type'] = ['string', 'null']
    claim['description'] += (' For source_effect or an available claim in unavailable, quote an '
                            'exact substring of the original hypothesis or reasoning. For forecast '
                            'use null: its numeric target is bound by forecast_ref, not arbitrary '
                            'claim text. For unavailable with no original claim, use null. '
                            'Local validation and evidence binding enforce these conditions.')
    quote = _object(dict(
        field=dict(type='string', enum=list(mechanism.SOURCE_FIELDS + mechanism.conclusion_source.SOURCE_FIELDS)),
        text=_text(mechanism.MAX_TEXT_CHARS)))
    entry = _object(dict(
        claim_quote=claim,
        source_quotes=dict(type='array', items=quote,
            description=f'At most {mechanism.MAX_SOURCE_QUOTES} exact quotes from supplied source '
                        'fields; this count is enforced locally. A non-unknown source_effect '
                        'assessment requires at least one quote. For forecast and unavailable '
                        'use []; the local validator enforces these conditions. '
                        + mechanism.SOURCE_QUOTE_RULES),
        assessment=dict(type='string', enum=list(mechanism.ASSESSMENTS)),
        direct_effect=_text(mechanism.MAX_TEXT_CHARS),
        untested_downstream=_text(mechanism.MAX_TEXT_CHARS),
        evidence_kind=dict(type='string', enum=list(mechanism.EVIDENCE_KINDS)),
        forecast_ref={**_object(dict(
            experiment_id=_identifier(),
            target=dict(type='string', enum=[mechanism.FORECAST_TARGET]))),
            'type': ['object', 'null'],
            'description': 'Only for forecast: this exact ordinary experiment ID and native '
                'delta interval. Null for source_effect and unavailable. Local binding '
                'rederives the assessment from the authoritative observation.'}))
    return dict(type='array', items=entry, minItems=1,
        description=f'One to {mechanism.MAX_ENTRIES} entries, enforced locally. '
                    f'The entire mechanism_review must fit {mechanism.MAX_REVIEW_CHARS} '
                    'characters after canonical ASCII-escaped JSON encoding, including '
                    'field names. This aggregate limit is enforced locally. ' + mechanism.TYPED_RULES)


def _comparison_reviews():
    from . import comparison_review as comparison

    proposal = _object(dict(
        comparison_id=dict(type='string', description=f'Exact comparison identifier of 1 to '
            f'{comparison.MAX_ID_CHARS} characters, with no whitespace, character below '
            'U+0020, or U+007F. The exact identifier rules are enforced locally.'),
        comparison_sha256=dict(type='string', description='Exact lowercase 64-character '
            'SHA-256 of the supplied comparison record. The hash syntax and evidence '
            'binding are checked locally; schema validity does not establish either.'),
        interpretation=_text(comparison.MAX_INTERPRETATION_CHARS),
        limitations=_text(comparison.MAX_INTERPRETATION_CHARS),
        next_hypothesis={**_object(dict(
            change=_text(comparison.MAX_HYPOTHESIS_CHARS),
            expected_observation=_text(comparison.MAX_HYPOTHESIS_CHARS),
            falsification=_text(comparison.MAX_HYPOTHESIS_CHARS))),
            'type': ['object', 'null'],
            'description': 'Null if no justified next experiment is proposed. Otherwise '
                'specify a change, the expected observation, and an observation that would '
                'falsify it. This is an untested proposal, never an established lesson.'}))
    return dict(type='array', items=proposal,
        description=f'At most {comparison.MAX_PROPOSALS} reviews with distinct comparison IDs. '
            f'The complete array must fit {comparison.MAX_PROPOSAL_CHARS} characters after '
            'canonical ASCII-escaped JSON encoding, including field names. These limits '
            'are enforced locally. Interpretations and proposed hypotheses remain unverified; '
            'a review can describe mixed results without authorizing a lesson or advancement.')


def response_format(kind, *, comparison_reviews: bool = False, comparison_review_version: int = 1):
    """Return a fresh strict provider envelope; accept only the two exact kinds."""
    if type(comparison_reviews) is not bool:
        raise ValueError('comparison_reviews must be a boolean')
    if type(comparison_review_version) is not int or comparison_review_version not in (1, 2, 3):
        raise ValueError('comparison_review_version must be 1, 2 or 3')
    if comparison_review_version in (2, 3) and not comparison_reviews:
        raise ValueError(f'comparison_review_version {comparison_review_version} requires comparison_reviews')
    if type(kind) is not str or kind not in _KINDS:
        raise ValueError('unknown structured auxiliary response kind')
    if kind == 'conclusion':
        fields = dict(
            verdict=dict(type='string', enum=['CONFIRMED', 'REFUTED', 'INCONCLUSIVE']),
            explanation=_text(parsing.MAX_REASONING_CHARS),
            lessons=_lessons(),
            mechanism_review=_mechanism_review())
    else:
        fields = dict(lessons=_lessons(), retire_ids=_ids(parsing.MAX_RETIRE_IDS))
        if comparison_reviews:
            if comparison_review_version == 3:
                from .reasoned_review import json_schema
                fields['comparison_reviews'] = json_schema()
            elif comparison_review_version == 2:
                from .grounded_review import json_schema
                fields['comparison_reviews'] = json_schema()
            else:
                fields['comparison_reviews'] = _comparison_reviews()
    schema = _object(fields,
        description=f'One bare JSON object, no markers, fences or surrounding prose. '
                    f'Local response and JSON-block limits remain {parsing.MAX_RESPONSE_CHARS} '
                    f'and {parsing.MAX_BLOCK_CHARS} characters respectively. '
                    'Shape validity never overrides source/evidence binding or lesson gates.')
    version = (4 if kind == 'reflection' and comparison_review_version == 3 else
               3 if kind == 'reflection' and comparison_review_version == 2 else
               2 if kind == 'conclusion' or comparison_reviews else 1)
    return dict(type='json_schema', json_schema=dict(
        name=f'sciel_{kind}_v{version}',
        strict=True, schema=schema))


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def identity(*, comparison_reviews: bool = False, comparison_review_version: int = 1):
    """Bind module bytes and both exact provider payloads, with no cached hash."""
    if type(comparison_reviews) is not bool:
        raise ValueError('comparison_reviews must be a boolean')
    payloads = {kind: _sha(json.dumps(response_format(kind, comparison_reviews=comparison_reviews,
        comparison_review_version=comparison_review_version), sort_keys=True,
        separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('utf-8'))
        for kind in _KINDS}
    value = dict(version=VERSION, source_sha256=_sha(Path(__file__).read_bytes()),
                 response_format_sha256=payloads)
    if comparison_reviews:
        from .comparison_review import identity as comparison_identity
        value['comparison_reviews'] = True
        value['review_policy'] = comparison_identity()
        if comparison_review_version == 3:
            from .reasoned_review import identity as reasoned_identity
            value['comparison_review_version'] = 3
            value['reasoned_review_policy'] = reasoned_identity()
        elif comparison_review_version == 2:
            from .grounded_review import identity as grounded_identity
            value['comparison_review_version'] = 2
            value['grounded_review_policy'] = grounded_identity()
    return value
