"""Source-bound observations for prompt feedback, never an efficacy lesson.

Ordinary evaluations are unpaired. A matching score or saved geometry on those
evaluations is not semantic equivalence, a speed benefit, or a causal result.
Only compact facts are retained; candidate code is never imported or executed.
"""

from __future__ import annotations
from copy import deepcopy
import difflib
import hashlib
import json
import math
import re
from ..evidence.change_audit import build_change_audit
from .observed_schema import (
    RECORD_SCHEMA,
    SCHEMA,
    SCORE_ABS_TOLERANCE,
    _CASE_FIELDS,
    _GEOMETRY_FIELDS,
    _HASH,
    _MATCH_RULES,
    _OBS_FIELDS,
    _PREFIX,
    _RECORD_FIELDS,
    _SUFFIX,
    _canonical,
    _compare,
    _condition,
    _digest,
    _hash,
    _identifier,
    _number,
    _require,
    _seal,
    _source_sha,
    _validate_observation,
    _validate_record,
)


class ObservedChanges:
    """Full compact history, with a derived source index and bounded prompt view.

    A program ID identifies one immutable final observation. Repeating an
    identical observation is idempotent; conflicting ID reuse is rejected.
    Missing parent evidence never falls back to Program.score.
    """

    def __init__(self, metric=None):
        if metric is not None:
            from ..evidence.measurement import validate_metric_context
            metric = validate_metric_context(metric)
        self._metric = deepcopy(metric)
        self._records = []
        self._keys = {}
        self._programs = {}
        self._by_source = {}
        self._aliases = {}
        self._aliases_by_record = {}
        self._by_output = {}
        self._output_counts = {}

    def _validity_required(self):
        return (self._metric or {}).get('semantics', {}).get('kind') == 'validity_gated_ratio'

    @property
    def identity(self):
        return dict(schema=SCHEMA, metric=deepcopy(self._metric), score_abs_tolerance=SCORE_ABS_TOLERANCE,
                    score_relative_tolerance=0, ordinary_scope='unpaired',
                    validity_policy='declared_metric_or_validated_native_geometry/1',
                    geometry_policy='validated_native_numeric_sha256_same_evaluator_and_dtypes', edited_units_limit=6,
                    source_diff_char_limit=1000, alias_policy='explicit_exact_migrant_copy',
                    global_output_policy='saved_geometry_or_exact_cases_score_same_metric_stage/1')

    def _resolve_program(self, ident):
        seen = set()
        while ident in self._aliases:
            _require(ident not in seen, 'cyclic observed program alias')
            seen.add(ident)
            ident = self._aliases[ident]['reference_program_id']
        index = self._programs.get(ident)
        return self._records[index] if index is not None else None

    def record_for_program(self, ident):
        """Detached canonical observation, including explicit migrant aliases."""
        return deepcopy(self._resolve_program(ident))

    def _insert_alias(self, alias):
        fields = {'program_id', 'reference_program_id', 'source_sha256', 'reference_record_sha256',
                  'score', 'cases_sha256'}
        _require(type(alias) is dict and set(alias) == fields, 'invalid observed alias fields')
        _require(_identifier(alias['program_id']) and _identifier(alias['reference_program_id'])
                 and alias['program_id'] != alias['reference_program_id'], 'invalid alias identity')
        _require(all(_hash(alias[k]) for k in ('source_sha256', 'reference_record_sha256', 'cases_sha256'))
                 and _number(alias['score']), 'invalid alias binding')
        _require(alias['program_id'] not in self._programs, 'alias conflicts with measured program')
        reference = self._resolve_program(alias['reference_program_id'])
        _require(reference is not None and reference['child_source_sha256'] == alias['source_sha256']
                 and reference['record_sha256'] == alias['reference_record_sha256']
                 and reference['child_observation']['score'] == alias['score'], 'alias reference differs')
        case_vector = reference['child_observation'].get('case_vector')
        _require(case_vector is None or case_vector['cases_sha256'] == alias['cases_sha256'],
                 'alias cases differ from observed vector')
        if alias['program_id'] in self._aliases:
            _require(self._aliases[alias['program_id']] == alias, 'conflicting observed alias identity')
        else:
            self._aliases[alias['program_id']] = deepcopy(alias)
            self._aliases_by_record.setdefault(reference['id'], []).append(alias['program_id'])
        return deepcopy(alias)

    def bind_alias(self, program, reference):
        """Bind an explicit migrated copy, without fabricating another measurement."""
        _require(program.origin == 'migrant' and program.parent_id == reference.id,
                 'only explicit migrant copies can bind an alias')
        _require(program.code == reference.code and program.score == reference.score
                 and type(program.cases) is dict and type(reference.cases) is dict
                 and _canonical(program.cases) == _canonical(reference.cases), 'migrant copy differs')
        source_sha = _source_sha(program.code)
        observed = self._resolve_program(reference.id)
        if observed is None:
            return None
        _require(observed['child_source_sha256'] == source_sha, 'migrant reference source differs')
        alias = dict(program_id=program.id, reference_program_id=reference.id, source_sha256=source_sha,
                     reference_record_sha256=observed['record_sha256'], score=program.score,
                     cases_sha256=_digest(program.cases))
        return self._insert_alias(alias)

    def _observation(self, observation, source_sha):
        observation = observation if type(observation) is dict else {}
        cases = observation.get('cases')
        cases = cases if type(cases) is dict else {}
        semantics = (self._metric or {}).get('semantics', {})
        # Diagnostic names cannot declare a task's acceptance contract. A
        # generic evaluator may legitimately report a continuous validity rate.
        actual = cases.get(semantics.get('validity_case')) if self._validity_required() else None
        validity = (actual == semantics['valid_value']) if _number(actual) else None
        result = dict(ok=observation.get('ok') if type(observation.get('ok')) is bool else None,
                      score=observation.get('score') if _number(observation.get('score')) else None,
                      validity=validity,
                      reached=observation.get('reached') if observation.get('reached') in ('smoke', 'full', 'confirm') else None,
                      confirmed=observation.get('confirmed') if type(observation.get('confirmed')) is bool else None,
                      geometry=None, geometry_status='absent', validity_required=self._validity_required())
        artifacts = observation.get('artifacts')
        if type(artifacts) is dict and 'native_geometry' in artifacts:
            try:
                from ..eval.native_geometry import parse_artifact, validate_binding
                geometry = parse_artifact(artifacts['native_geometry'])
                geometry = validate_binding(geometry, observation, source_sha256=source_sha, metric=self._metric)
                result.update(geometry={k: geometry[k] for k in _GEOMETRY_FIELDS},
                              geometry_status='validated', validity=geometry['native_validity'])
            except (ValueError, TypeError, KeyError, OverflowError, RecursionError):
                result.update(geometry_status='invalid', validity=None)
        result['status'], result['reason'] = _condition(result)
        result['case_vector'] = None
        # Preserve JSON numeric representation exactly, including signed zero
        # and int/float distinctions. Empty cases cannot turn score alone into
        # a vector match. Diagnostic names never invent an acceptance gate.
        if (result['status'] in ('valid', 'score_only') and self._metric is not None
                and self._metric['task_hash'] is not None and cases
                and all(type(key) is str and bool(key) and _number(value) for key, value in cases.items())):
            result['case_vector'] = dict(cases_sha256=_digest(cases), case_count=len(cases),
                                         metric_sha256=self._metric['sha256'])
        return result

    def _output_keys(self, record):
        """Derived indices never infer a measurement from population scores."""
        if (record['scope'] != 'unpaired' or self._metric is None
                or self._metric['task_hash'] is None):
            return []
        obs = record['child_observation']
        if obs['status'] not in ('valid', 'score_only'):
            return []
        context = (self._metric['sha256'], obs['reached'], obs['confirmed'])
        geometry = obs['geometry']
        keys = []
        if geometry is not None and obs['status'] == 'valid' and obs['geometry_status'] == 'validated':
            keys.append(('exact_validated_geometry', *context, *(geometry[k] for k in
                ('evaluator_sha256', 'centers_dtype', 'radii_dtype', 'numeric_sha256'))))
        cases = obs.get('case_vector')
        # If geometry exists, never disguise incompatible evaluator/dtypes or
        # different geometries as a generic vector tie. Geometry has stronger
        # available evidence. Generic matching is for observations without it.
        if cases is not None and geometry is None:
            keys.append(('exact_case_vector_and_score', *context, cases['cases_sha256'],
                         cases['case_count'], _canonical(obs['score'])))
        return keys

    def _insert(self, record):
        key = (record['scope'], record['id'])
        if key in self._keys:
            old = self._records[self._keys[key]]
            _require(old == record, 'conflicting observed change identity')
            return deepcopy(old)
        if record['scope'] == 'unpaired':
            _require(record['child_program_id'] not in self._aliases, 'measurement conflicts with migrant alias')
        index = len(self._records)
        self._records.append(deepcopy(record))
        self._keys[key] = index
        if record['scope'] == 'unpaired':
            self._programs[record['child_program_id']] = index
        for sha in {record['parent_source_sha256'], record['child_source_sha256']} - {None}:
            self._by_source.setdefault(sha, []).append(index)
        for output_key in self._output_keys(record):
            sources = self._by_output.setdefault(output_key, {})
            sources.setdefault(record['child_source_sha256'], []).append(index)
            self._output_counts[output_key] = self._output_counts.get(output_key, 0) + 1
        return deepcopy(record)

    def _history_matches(self, record, *, limit=2):
        """Bounded representatives of distinct-source matches across all islands.

        Same-source registrations (including per-island seed copies) cannot
        masquerade as different methods. Aliases are separately labelled and
        reference the original immutable observation, never extra evaluations.
        """
        def binding(item):
            aliases = self._aliases_by_record.get(item['id'], [])
            return dict(program_id=item['child_program_id'], source_sha256=item['child_source_sha256'],
                        record_sha256=item['record_sha256'],
                        observation_sha256=_digest(item['child_observation']),
                        alias_program_ids=aliases[:2], alias_ids_omitted=max(0, len(aliases) - 2))

        source = record['child_source_sha256']
        for key in self._output_keys(record):
            sources = self._by_output[key]
            distinct = len(sources) - 1
            if distinct <= 0:
                continue
            matches = []
            for sha, indices in sources.items():
                if sha == source:
                    continue
                item = self._records[indices[0]]
                matches.append(dict(binding(item), same_source_record_count=len(indices)))
                if len(matches) == limit:
                    break
            obs = record['child_observation']
            return dict(schema='sciel-global-output-matches/1', scope='unpaired_saved_run_history',
                        kind=key[0], metric_sha256=self._metric['sha256'],
                        reached=obs['reached'], confirmed=obs['confirmed'],
                        geometry=deepcopy(obs['geometry']), case_vector=deepcopy(obs.get('case_vector')),
                        score=obs['score'], validity=obs['validity'], query=binding(record), matches=matches,
                        matching_distinct_sources=distinct,
                        matching_record_count=self._output_counts[key] - len(sources[source]),
                        match_sources_omitted=distinct - len(matches))
        return None

    def observe(self, program, observation, *, parent=None):
        _require(_identifier(program.id), 'invalid program ID')
        child_sha = _source_sha(program.code)
        parent_sha = _source_sha(parent.code) if parent is not None else None
        parent_id = parent.id if parent is not None else program.parent_id
        _require(parent_id is None or _identifier(parent_id), 'invalid parent ID')
        _require(parent_id != program.id, 'program cannot be its own parent')
        parent_observation = None
        previous = self._resolve_program(parent_id)
        if previous is not None:
            if previous['child_source_sha256'] == parent_sha:
                parent_observation = deepcopy(previous['child_observation'])
        child = self._observation(observation, child_sha)
        structural = dict(edited_units=[], edited_units_available=False, edited_units_omitted=0,
                          source_equal=None, ast_equal=None, source_diff=None)
        if parent is not None:
            audit = build_change_audit(parent.code, program.code)
            eligible = [x for x in audit['direct_edited_unit_ids'] if _identifier(x)]
            diff = ''.join(difflib.unified_diff(parent.code.splitlines(keepends=True),
                program.code.splitlines(keepends=True), fromfile='parent.py', tofile='candidate.py'))
            structural.update(edited_units=eligible[:6], edited_units_available=audit['comparison_available'],
                edited_units_omitted=len(audit['direct_edited_unit_ids']) - len(eligible[:6]),
                source_equal=audit['source_equal'], ast_equal=audit['ast_equal'],
                source_diff=dict(text=diff[:1000], sha256=_source_sha(diff), chars_total=len(diff),
                                 chars_omitted=max(0, len(diff) - 1000)))
        record = dict(schema=RECORD_SCHEMA, id='program:' + program.id, scope='unpaired',
                      parent_program_id=parent_id, child_program_id=program.id,
                      parent_source_sha256=parent_sha, child_source_sha256=child_sha,
                      parent_observation=parent_observation, child_observation=child,
                      **_compare(parent_observation, child), **structural)
        return self._insert(_validate_record(_seal(record)))

    def observe_paired(self, group):
        from .observed_paired import paired_record, validate_record
        return self._insert(validate_record(paired_record(group)))

    def block(self, code, max_chars=4000):
        _require(type(max_chars) is int and max_chars >= 0, 'max_chars must be nonnegative integer')
        source_sha = _source_sha(code)
        indices = self._by_source.get(source_sha, [])
        # Put the latest matching observation of this actual source first,
        # ahead of parent/child edit history. A cross-island match otherwise
        # disappears when its programs have no direct ancestry relationship.
        match_index = next((i for i in reversed(indices)
            if self._records[i]['scope'] == 'unpaired'
            and self._records[i]['child_source_sha256'] == source_sha
            and self._history_matches(self._records[i]) is not None), None)
        order = ([match_index] if match_index is not None else []) + [i for i in reversed(indices) if i != match_index]
        selected = []
        def render(records):
            payload = _canonical(records).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
            rules = _MATCH_RULES if any('history_matches' in item for item in records) else ''
            return _PREFIX + rules + payload + _SUFFIX

        def projections(record):
            common = {k: deepcopy(record[k]) for k in ('id', 'scope', 'status', 'parent_source_sha256',
                'child_source_sha256', 'record_sha256', 'edited_units', 'edited_units_available', 'edited_units_omitted')}
            if record['scope'] == 'paired':
                common.update({k: record[k] for k in ('parent_role', 'geometry_representation', 'original_array_dtypes')})
            if record['scope'] == 'unpaired':
                full = dict(common, **{k: deepcopy(record[k]) for k in ('source_diff', 'score_comparison',
                    'geometry_comparison', 'parent_program_id', 'child_program_id')})
                for key in ('parent_observation', 'child_observation'):
                    value = record[key]
                    full[key] = {k: value[k] for k in ('score', 'validity', 'status', 'reason')} if value else None
            else:
                full = dict(common, parent_role='comparator', pairs=deepcopy(record['pairs']))
            yield full
            compact = deepcopy(full)
            if compact.get('source_diff') is not None:
                diff = compact['source_diff']
                diff.update(text='', chars_omitted=diff['chars_total'], omitted_from_prompt=True)
            if 'pairs' in compact:
                compact['pairs_omitted_from_prompt'] = len(compact.pop('pairs'))
            yield compact
            minimal = dict(common, details_omitted_from_prompt=True)
            minimal['edited_units_omitted'] += len(minimal['edited_units'])
            minimal['edited_units'] = []
            yield minimal

        for index in order:
            if len(selected) == 3:
                break
            record = self._records[index]
            if record['scope'] == 'unpaired' and record['parent_source_sha256'] is None and index != match_index:
                # Baselines remain measurable parents in state, but contain no
                # prior source edit to show to a generator selecting this code.
                continue
            # Whole factual projections, then explicit omissions; never cut JSON
            # or silently drop an oversized recent record in favour of an old one.
            def candidates():
                for limit in ((2, 1) if index == match_index else (0,)):
                    for item in projections(record):
                        if limit:
                            item['history_matches'] = self._history_matches(record, limit=limit)
                        yield item
                if index == match_index:
                    # A small caller budget must not erase lineage facts that
                    # still fit. Report the missing global witness explicitly;
                    # this marker is not a substitute for its source bindings.
                    for item in projections(record):
                        item['history_matches_omitted'] = {'reason': 'char_budget'}
                        yield item
            chosen = next((item for item in candidates() if len(render(selected + [item])) <= max_chars), None)
            if chosen is None:
                break
            selected.append(chosen)
        if not selected:
            return ''
        return render(selected)

    def state(self):
        value = dict(schema=SCHEMA, identity=self.identity, records=deepcopy(self._records),
                     aliases=deepcopy(list(self._aliases.values())))
        value['state_sha256'] = _digest(value)
        return value

    def load_state(self, state):
        """Validate fully in a detached instance before changing current state."""
        _require(type(state) is dict and set(state) == {'schema', 'identity', 'records', 'aliases', 'state_sha256'}
                 and state['schema'] == SCHEMA, 'invalid observed changes state fields/schema')
        body = {k: v for k, v in state.items() if k != 'state_sha256'}
        _require(_hash(state['state_sha256']) and _digest(body) == state['state_sha256'], 'observed changes state digest differs')
        _require(state['identity'] == self.identity, 'observed changes identity differs')
        _require(type(state['records']) is list and type(state['aliases']) is list, 'invalid observed changes records')
        restored = ObservedChanges(self._metric)
        for item in state['records']:
            if type(item) is dict and item.get('scope') == 'paired':
                from .observed_paired import validate_record
                record = validate_record(item)
            else:
                record = _validate_record(item)
                _require(record['child_observation']['validity_required'] == restored._validity_required(),
                         'observation validity policy differs')
                for obs in (record['child_observation'], record['parent_observation']):
                    if obs is not None and obs.get('case_vector') is not None:
                        _require(self._metric is not None and self._metric['task_hash'] is not None
                                 and obs['case_vector']['metric_sha256'] == self._metric['sha256'],
                                 'case vector metric binding differs')
            _require((record['scope'], record['id']) not in restored._keys, 'duplicate observation identity in state')
            restored._insert(record)
        for alias in state['aliases']:
            _require(type(alias) is dict and _identifier(alias.get('program_id'))
                     and alias['program_id'] not in restored._aliases,
                     'duplicate or malformed observed alias')
            restored._insert_alias(alias)
        for index, record in enumerate(restored._records):
            if record['scope'] != 'unpaired' or record['parent_observation'] is None:
                continue
            previous = restored._resolve_program(record['parent_program_id'])
            _require(previous is not None, 'parent observation missing from history')
            _require(restored._programs[previous['child_program_id']] < index,
                     'parent observation is not earlier than child')
            _require(previous['child_source_sha256'] == record['parent_source_sha256']
                     and previous['child_observation'] == record['parent_observation'], 'parent observation history differs')
        self._records, self._keys = restored._records, restored._keys
        self._programs, self._by_source = restored._programs, restored._by_source
        self._aliases = restored._aliases
        self._aliases_by_record = restored._aliases_by_record
        self._by_output, self._output_counts = restored._by_output, restored._output_counts
