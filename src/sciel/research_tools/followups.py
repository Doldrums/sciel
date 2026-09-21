"""Receipt-bound chronological followups, never method rewards or causal credit.

All functions are pure JSON operations. Callers authenticate immutable invocation
artifacts and canonical research evidence through the ledger callbacks; this
module never reads files, dispatches work, or interprets tool prose as a fact.
"""
from copy import deepcopy
from bisect import insort
import math

from ..methods.models import canonical_bytes, content_hash, require_hash
from ..hypothesis.measurement import validate_evaluation, validate_metric_context
from ..knowledge_base.eligibility import admission_facts, validate_output_observation, REPEAT_STATUSES

EXPOSURE_SCHEMA = 'sciel-method-exposures/1'
FOLLOWUP_SCHEMA = 'sciel-method-followup/1'
STATE_SCHEMA = 'sciel-method-followups/1'
VIEW_SCHEMA = 'sciel-method-followups-view/1'
SCOPE = 'Chronological exposure only; method use and causal benefit are unknown.'
_REPEAT_SCOPE = 'Recorded output admission refusal; not efficacy evidence or program equivalence.'
_SOURCE_FIELDS = {'tool_id', 'invocation_id', 'receipt_id', 'candidate_id', 'target_kind',
    'target_source_sha256', 'ancestry_parent_id', 'research_targets_sha256',
    'population_best_id', 'population_best_source_sha256'}
_CARD_FIELDS = {'card_sha256', 'cell_id', 'revision_id', 'source',
                'observation_status', 'observations_sha256'}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _text(value, name, *, nullable=False):
    _require(nullable and value is None or type(value) is str and
             0 < len(value) <= 256 and all(ord(c) >= 32 for c in value), 'invalid ' + name)


def _number(value, name, *, nullable=False):
    if nullable and value is None:
        return
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    _require(valid, 'invalid ' + name)


def _optional_hash(value, name):
    if value is not None:
        require_hash(value, name)


def _seal(value):
    return {**deepcopy(value), 'sha256': content_hash(value)}


def _sealed(value, schema, fields):
    _require(type(value) is dict and set(value) == set(fields) | {'schema', 'sha256'}
             and value.get('schema') == schema, 'invalid ' + schema + ' shape')
    require_hash(value['sha256'], schema + ' SHA')
    _require(content_hash({k: v for k, v in value.items() if k != 'sha256'}) == value['sha256'],
             schema + ' digest differs')


def _source(source):
    _require(type(source) is dict and set(source) == _SOURCE_FIELDS, 'invalid exposure source')
    for key in ('tool_id', 'invocation_id', 'receipt_id', 'population_best_source_sha256'):
        _optional_hash(source[key], key)
    for key in ('target_source_sha256', 'research_targets_sha256'):
        require_hash(source[key], key)
    for key in ('candidate_id', 'population_best_id'):
        _text(source[key], key, nullable=True)
    _text(source['ancestry_parent_id'], 'ancestry parent')
    _require(source['target_kind'] in ('program', 'failed_candidate') and
             (source['candidate_id'] is not None) == (source['target_kind'] == 'program'),
             'invalid input program identity')
    _require((source['population_best_id'] is None) ==
             (source['population_best_source_sha256'] is None), 'invalid available best identity')
    # A card without any execution identity is possible; a partial identity
    # cannot manufacture an authenticated method exposure.
    identifiers = [source[k] is not None for k in ('tool_id', 'invocation_id', 'receipt_id')]
    _require(all(identifiers) or not any(identifiers), 'partial method execution identity')


def capture_exposures(context, *, prompt_sha256):
    """Seal exactly the FINAL rendered context, not an earlier snapshot copy.

    Invalid input is rejected, never compacted again: silently dropping a card
    here would describe a different prompt. No context means no exposure field.
    """
    if context is None:
        return None
    from ..engine.research_cells_context import SCHEMA, MAX_CARDS, _card
    _require(type(context) is dict and set(context) ==
             {'schema', 'cards', 'omitted_count', 'omission_reasons'} and context['schema'] == SCHEMA,
             'invalid final research cells context')
    _require(type(context['cards']) is list and len(context['cards']) <= MAX_CARDS,
             'invalid final card collection')
    rows = []
    for card in context['cards']:
        _require(_card(card) is not None, 'invalid final research card')
        rows.append({'card_sha256': content_hash(card), 'cell_id': card['cell_id'],
            'revision_id': card['revision_id'], 'source': deepcopy(card['source']),
            'observation_status': card['observations']['status'],
            'observations_sha256': content_hash(card['observations'])})
    return validate_exposures(_seal({'schema': EXPOSURE_SCHEMA, 'prompt_sha256': prompt_sha256,
        'cards': rows, 'omitted_count': context['omitted_count'],
        'omission_reasons': deepcopy(context['omission_reasons'])}))


def validate_exposures(value):
    """Validate saved exposure metadata; artifact readers bind receipt facts."""
    _sealed(value, EXPOSURE_SCHEMA, {'prompt_sha256', 'cards', 'omitted_count', 'omission_reasons'})
    require_hash(value['prompt_sha256'], 'generation prompt SHA')
    _require(type(value['cards']) is list and len(value['cards']) <= 32, 'invalid exposure cards')
    for card in value['cards']:
        _require(type(card) is dict and set(card) == _CARD_FIELDS, 'invalid exposure card fields')
        for key in ('card_sha256', 'observations_sha256'):
            require_hash(card[key], key)
        for key in ('cell_id', 'revision_id', 'observation_status'):
            _text(card[key], key)
        _source(card['source'])
    _require(type(value['omitted_count']) is int and value['omitted_count'] >= 0,
             'invalid omitted exposure count')
    reasons = value['omission_reasons']
    _require(type(reasons) is dict, 'invalid exposure omission reasons')
    for key, count in reasons.items():
        _text(key, 'omission reason')
        _require(type(count) is int and count >= 0, 'invalid exposure omission count')
    _require(sum(reasons.values()) == value['omitted_count'], 'exposure omission accounting differs')
    return deepcopy(value)


def _linked(exposures):
    return [row for row in exposures['cards'] if row['source']['tool_id'] is not None]


def build_followup(record, *, child_program_id=None, link_kind='mutation', observed_record=None,
                   evidence_artifact_id=None):
    """Join finalized research evidence to optional admitted, source-bound child.

    The proposed hypothesis is an unverified claim. Failed/absent evaluations
    remain failed/absent; a missing child is never represented by a fake ID.
    """
    if 'method_exposures' not in record:
        return None
    exposures = validate_exposures(record['method_exposures'])
    if not _linked(exposures):
        return None
    result = {'schema': FOLLOWUP_SCHEMA, 'experiment_id': record['id'],
        'evidence_sha256': content_hash(record), 'evidence_artifact_id': evidence_artifact_id,
        'island_id': record['island'],
        'iteration': record['iteration'], 'operator': record['operator'],
        'parent': {'program_id': record['parent_id'], 'source_sha256': record['parent_source_sha256'],
                   'recorded_score': record.get('parent_score')},
        'exposures': exposures,
        'generation': {'dispatch': record.get('generation_dispatch', 'unknown'),
                       'response_sha256': record.get('generation_response_sha256')},
        'proposal': {'claim': deepcopy(record.get('hypothesis')),
                     'claim_status': 'unverified_model_statement',
                     'proposed_sha256': record.get('proposed_sha256'),
                     'source_sha256': record.get('source_sha256'),
                     'source_repaired': record.get('source_repaired', False)},
        'outcome': {'status': record['status'], 'score': record.get('score'),
                    'actual_delta': record.get('actual_delta'),
                    'evaluation': deepcopy(record.get('evaluation')),
                    'metric': deepcopy(record['metric']), 'failure': record.get('failure'),
                    'child_program_id': child_program_id, 'link_kind': link_kind,
                    'observed_record': deepcopy(observed_record)},
        'relation': 'exposed_before_generation', 'model_use': 'unknown', 'causal_benefit': 'unknown'}
    if 'output_novelty' in record:
        result['outcome']['output_novelty'] = deepcopy(record['output_novelty'])
    return validate_followup(_seal(result))


def validate_followup(value):
    _sealed(value, FOLLOWUP_SCHEMA, {'experiment_id', 'evidence_sha256', 'evidence_artifact_id', 'island_id', 'iteration',
        'operator', 'parent', 'exposures', 'generation', 'proposal', 'outcome',
        'relation', 'model_use', 'causal_benefit'})
    _text(value['experiment_id'], 'experiment ID')
    require_hash(value['evidence_sha256'], 'research evidence SHA')
    _optional_hash(value['evidence_artifact_id'], 'research evidence artifact')
    _text(value['operator'], 'operator')
    for key in ('island_id', 'iteration'):
        _require(type(value[key]) is int and value[key] >= 0, 'invalid ' + key)
    validate_exposures(value['exposures'])
    _require(bool(_linked(value['exposures'])), 'followup has no linked method exposure')
    _require(value['relation'] == 'exposed_before_generation' and value['model_use'] == 'unknown'
             and value['causal_benefit'] == 'unknown', 'followup cannot assert method use or benefit')
    parent = value['parent']
    _require(type(parent) is dict and set(parent) == {'program_id', 'source_sha256', 'recorded_score'},
             'invalid parent evidence')
    _text(parent['program_id'], 'parent program', nullable=True)
    _optional_hash(parent['source_sha256'], 'parent source')
    _number(parent['recorded_score'], 'recorded parent score', nullable=True)
    generation = value['generation']
    _require(type(generation) is dict and set(generation) == {'dispatch', 'response_sha256'} and
             generation['dispatch'] in ('returned', 'unknown', 'not_dispatched'), 'invalid generation state')
    _optional_hash(generation['response_sha256'], 'generation response SHA')
    _require(generation['response_sha256'] is None or generation['dispatch'] == 'returned',
             'response hash without returned response')
    proposal = value['proposal']
    _require(type(proposal) is dict and set(proposal) == {'claim', 'claim_status', 'proposed_sha256',
        'source_sha256', 'source_repaired'} and proposal['claim_status'] == 'unverified_model_statement'
        and (proposal['claim'] is None or type(proposal['claim']) is dict)
        and type(proposal['source_repaired']) is bool, 'invalid proposed claim')
    for key in ('proposed_sha256', 'source_sha256'):
        _optional_hash(proposal[key], key)
    _require(not proposal['source_repaired'] or (proposal['proposed_sha256'] is not None and
        proposal['source_sha256'] is not None and proposal['proposed_sha256'] != proposal['source_sha256']),
        'source repair lacks distinct sources')
    outcome = value['outcome']
    outcome_fields = {'status', 'score', 'actual_delta', 'evaluation', 'metric', 'failure',
                      'child_program_id', 'link_kind', 'observed_record'}
    _require(type(outcome) is dict and set(outcome) in
             (outcome_fields, outcome_fields | {'output_novelty'}), 'invalid followup outcome')
    _text(outcome['status'], 'research outcome status')
    _require(outcome['status'] != 'pending', 'pending research cannot be a finalized followup')
    _require(outcome['link_kind'] in ('mutation', 'remeasurement'), 'invalid followup link kind')
    _require(outcome['failure'] is None or type(outcome['failure']) is str, 'invalid failure evidence')
    for key in ('score', 'actual_delta'):
        _number(outcome[key], key, nullable=True)
    metric = validate_metric_context(outcome['metric'])
    evaluation = validate_evaluation(outcome['evaluation'])
    if outcome['status'] == 'evaluated':
        _require(generation['dispatch'] == 'returned' and proposal['source_sha256'] is not None
                 and evaluation is not None and evaluation['ok'] and outcome['score'] == evaluation['score'],
                 'evaluated followup lacks bound successful canonical evaluation')
        _number(outcome['score'], 'evaluated score')
        _number(parent['recorded_score'], 'evaluated parent score')
        _number(outcome['actual_delta'], 'evaluated delta')
        # Same binary arithmetic convention as research.resolve and the KB
        # validator. This is a recorded unpaired difference, never tool credit.
        _require(outcome['score'] - parent['recorded_score'] == outcome['actual_delta'],
                 'followup delta differs from recorded score difference')
    else:
        _require(outcome['score'] is None and outcome['actual_delta'] is None,
                 'unmeasured followup cannot claim a score or delta')
    if evaluation is not None:
        _require(proposal['source_sha256'] is not None and generation['dispatch'] == 'returned'
                 and ((outcome['status'] == 'evaluated' and evaluation['ok']) or
                      (outcome['status'] == 'eval_failed' and not evaluation['ok'])),
                 'canonical evaluation conflicts with outcome')
        if 'native_geometry' in evaluation:
            from ..eval.native_geometry import validate_binding
            validate_binding(evaluation['native_geometry'], evaluation,
                source_sha256=proposal['source_sha256'], metric=metric)
    if generation['dispatch'] != 'returned':
        _require(proposal['claim'] is None and proposal['source_sha256'] is None
                 and proposal['proposed_sha256'] is None and evaluation is None
                 and outcome['child_program_id'] is None, 'unreturned generation cannot produce a child')
    if 'output_novelty' in outcome:
        verdict = validate_output_observation(outcome['output_novelty'],
            source_sha256=proposal['source_sha256'], metric=metric, island=value['island_id'],
            evaluation=evaluation, status=outcome['status'])
        _require(verdict['status'] not in REPEAT_STATUSES or
                 outcome['child_program_id'] is None or outcome['link_kind'] == 'remeasurement',
                 'recorded output repeat cannot acquire a mutation child')
    _text(outcome['child_program_id'], 'admitted child', nullable=True)
    observed = outcome['observed_record']
    if outcome['child_program_id'] is None:
        _require(observed is None, 'observation without admitted child')
    else:
        from ..engine.observed_changes import _validate_record
        _require(outcome['link_kind'] == 'remeasurement' or outcome['status'] == 'evaluated',
                 'unadmitted generation cannot acquire a mutation child')
        _require(observed is not None, 'admitted child requires its saved observation')
        observed = _validate_record(observed)
        _require(observed['child_program_id'] == outcome['child_program_id'] and
                 observed['child_source_sha256'] == proposal['source_sha256'],
                 'admitted child differs from final source')
        if evaluation is not None:
            child = observed['child_observation']
            _require(child['score'] == evaluation['score'] and child['ok'] == evaluation['ok'] and
                     child['reached'] == evaluation['reached'] and child['confirmed'] == evaluation['confirmed'],
                     'admitted observation differs from canonical evaluation')
            cases = child.get('case_vector')
            if cases is not None:
                _require(cases['cases_sha256'] == content_hash(evaluation['cases']) and
                         cases['case_count'] == len(evaluation['cases']) and
                         cases['metric_sha256'] == metric['sha256'],
                         'admitted case evidence differs from canonical cases/metric')
            if child['geometry'] is not None:
                geometry = evaluation.get('native_geometry')
                _require(geometry is not None and all(geometry[k] == v for k, v in child['geometry'].items())
                         and child['validity'] == geometry['native_validity'],
                         'admitted geometry differs from canonical geometry')
    return deepcopy(value)


def _excerpt(value, limit=240):
    if value is None:
        return None
    text = canonical_bytes(value).decode('utf-8')
    return {'sha256': content_hash(value), 'excerpt': text[:limit],
            'omitted_chars': max(0, len(text) - limit)}


def _project(record, tool_id):
    outcome, proposal = record['outcome'], record['proposal']
    evaluation = outcome['evaluation']
    measured = None if evaluation is None else {
        **{k: evaluation[k] for k in ('ok', 'score', 'raw_first_score', 'reached', 'confirmed')},
        'cases_sha256': content_hash(evaluation['cases']), 'case_count': len(evaluation['cases']),
        'evaluation_sha256': content_hash(evaluation),
        'scope': 'EvalResult fields; failed scores/cases are diagnostics, not measured quality.'}
    result = {'tool_id': tool_id, 'followup_sha256': record['sha256'],
        'evidence_sha256': record['evidence_sha256'], 'evidence_artifact_id': record['evidence_artifact_id'],
        'experiment_id': record['experiment_id'], 'island_id': record['island_id'],
        'iteration': record['iteration'], 'prompt_sha256': record['exposures']['prompt_sha256'],
        'exposures': [deepcopy(row) for row in record['exposures']['cards'] if row['source']['tool_id'] == tool_id],
        'generation': deepcopy(record['generation']), 'proposed_claim': _excerpt(proposal['claim']),
        'proposed_claim_status': proposal['claim_status'], 'source_sha256': proposal['source_sha256'],
        'outcome_status': outcome['status'], 'canonical_evaluation': measured,
        'metric_sha256': outcome['metric']['sha256'], 'task_hash': outcome['metric']['task_hash'],
        'child_program_id': outcome['child_program_id'], 'link_kind': outcome['link_kind'],
        'failure': _excerpt(outcome['failure']),
        'observed_record_sha256': outcome['observed_record']['record_sha256'] if outcome['observed_record'] else None,
        'model_use': 'unknown', 'causal_benefit': 'unknown'}
    if 'output_novelty' in outcome:
        result.update(admission_facts({'status': outcome['status'],
            'source_sha256': proposal['source_sha256'], 'output_novelty': outcome['output_novelty']}))
        if result['output_admission']['status'] in REPEAT_STATUSES:
            result['output_admission_scope'] = _REPEAT_SCOPE
    return result


def compact_view(frozen_view, *, max_chars):
    """Fit a frozen planner view using whole-row omissions, without live reads.

    This is a presentation validator, not an independent receipt authenticator.
    The containing episode plan authenticates the already-frozen view.
    """
    _require(type(max_chars) is int and max_chars >= 0, 'invalid followup view budget')
    _require(type(frozen_view) is dict and set(frozen_view) ==
             {'schema', 'records', 'eligible_count', 'omitted_count', 'scope'}
             and frozen_view['schema'] == VIEW_SCHEMA and frozen_view['scope'] == SCOPE,
             'invalid frozen followup view')
    _require(type(frozen_view['records']) is list and
             type(frozen_view['eligible_count']) is int and frozen_view['eligible_count'] >= 0 and
             type(frozen_view['omitted_count']) is int and frozen_view['omitted_count'] >= 0 and
             len(frozen_view['records']) + frozen_view['omitted_count'] == frozen_view['eligible_count'],
             'invalid followup view omission accounting')
    for row in frozen_view['records']:
        _require(type(row) is dict and row.get('model_use') == 'unknown' and
                 row.get('causal_benefit') == 'unknown', 'invalid chronological followup view row')
        require_hash(row.get('tool_id'), 'projected method')
        require_hash(row.get('followup_sha256'), 'projected followup')
    value = deepcopy(frozen_view)
    while len(canonical_bytes(value)) > max_chars and value['records']:
        value['records'].pop()
        value['omitted_count'] += 1
    _require(len(canonical_bytes(value)) <= max_chars,
             'followup budget cannot fit whole omission metadata')
    return value


class FollowupLedger:
    """Checkpoint-owned index; an identical commit is harmless, a conflict fails.

    authenticate_exposure(card_ref) binds linked receipt identities. Optional
    authenticate_outcome(followup) additionally compares evidence_sha256 against
    the canonical KB record; use it on add AND restore to reject resealed scalar
    evidence changes. Both callbacks must raise on mismatch and do no execution.
    """
    def __init__(self, identity_sha256, *, authenticate_exposure, authenticate_outcome=None):
        require_hash(identity_sha256, 'followup ledger identity')
        _require(callable(authenticate_exposure), 'followups require an artifact authenticator')
        _require(authenticate_outcome is None or callable(authenticate_outcome), 'invalid outcome authenticator')
        self.identity_sha256 = identity_sha256
        self._authenticate_exposure = authenticate_exposure
        self._authenticate_outcome = authenticate_outcome
        self._records = {}
        self._by_tool = {}

    def _checked(self, raw):
        record = validate_followup(raw)
        for row in _linked(record['exposures']):
            self._authenticate_exposure(deepcopy(row))
        if self._authenticate_outcome is not None:
            self._authenticate_outcome(deepcopy(record))
        return record

    def add(self, raw):
        if raw is None:
            return False
        record = self._checked(raw)
        ident = record['experiment_id']
        old = self._records.get(ident)
        if old is not None:
            _require(canonical_bytes(old) == canonical_bytes(record), 'conflicting method followup for experiment')
            return False
        self._records[ident] = record
        for tool_id in dict.fromkeys(row['source']['tool_id'] for row in _linked(record['exposures'])):
            insort(self._by_tool.setdefault(tool_id, []), (record['iteration'], ident))
        return True

    def state(self):
        return _seal({'schema': STATE_SCHEMA, 'identity_sha256': self.identity_sha256,
                      'records': list(self._records.values())})

    def load_state(self, state):
        _sealed(state, STATE_SCHEMA, {'identity_sha256', 'records'})
        _require(state['identity_sha256'] == self.identity_sha256 and type(state['records']) is list,
                 'followup checkpoint identity differs')
        replacement = FollowupLedger(self.identity_sha256,
            authenticate_exposure=self._authenticate_exposure, authenticate_outcome=self._authenticate_outcome)
        for row in state['records']:
            _require(replacement.add(row), 'duplicate experiment in followup checkpoint')
        # Atomic restore: no partial index after any failed authentication.
        self._records, self._by_tool = replacement._records, replacement._by_tool

    def planner_view(self, visible_tool_ids, *, max_chars=4096, per_method=2):
        _require(type(visible_tool_ids) is list and len(visible_tool_ids) <= 32,
                 'invalid visible method collection')
        for ident in visible_tool_ids:
            require_hash(ident, 'visible method')
        _require(len(visible_tool_ids) == len(set(visible_tool_ids)), 'duplicate visible method')
        _require(type(max_chars) is int and max_chars >= 0 and type(per_method) is int and
                 1 <= per_method <= 8, 'invalid followup projection budget')
        choices, total = [], 0
        for tool_id in visible_tool_ids:
            ids = self._by_tool.get(tool_id, [])
            total += len(ids)
            # Maintain this ordering at commit/restore, not by rescanning a
            # growing method history for each planner request.
            choices.append((tool_id, [ident for _, ident in reversed(ids[-per_method:])]))
        view = {'schema': VIEW_SCHEMA, 'records': [], 'eligible_count': total,
                'omitted_count': total, 'scope': SCOPE}
        _require(len(canonical_bytes(view)) <= max_chars, 'followup budget cannot fit whole omission metadata')
        # Round-robin methods, newest first: one long method history must not
        # monopolize the bounded catalog. Rows are never cut into invalid JSON.
        for offset in range(per_method):
            for tool_id, ids in choices:
                if offset >= len(ids):
                    continue
                row = _project(self._records[ids[offset]], tool_id)
                candidate = {**view, 'records': view['records'] + [row],
                             'omitted_count': view['omitted_count'] - 1}
                if len(canonical_bytes(candidate)) <= max_chars:
                    view = candidate
        return deepcopy(view)
