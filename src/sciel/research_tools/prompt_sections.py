"""Bounded source, observation and replay sections for research prompts."""
from __future__ import annotations

from copy import deepcopy
import json
from ..engine.research_cells_context import compact_context
from ..methods.models import canonical_bytes, content_hash
from .result_contract import RESULT_SCHEMA
from .replay_feedback import shorten_feedback
from .followups import compact_view
from .result_projection import MIN_CONTEXT_CHARS, PROJECTION_SCHEMAS, success_context
from .artifact_manifest import artifact_manifest
from .canonical_diagnostics import feedback as canonical_feedback
from .observation_cards import _clip, _failure_context, _clip_json
from .tool_protocol import (
    RESULTS_VIEW_MARKER,
    TARGET_VIEW_MARKER,
    TARGET_VIEW_SCHEMA,
    _ACTION_GUIDANCE,
    _DECISION_FIELDS_GUIDANCE,
    _DEPENDENCY_GUIDANCE,
    _RESEARCH_PERMISSION,
    _YIELD_GUIDANCE,
)

def _research_results_prompt(context, budget):
    """Whole projections of frozen observations, independent of lesson prose.

    Source/receipt identities are never shortened. Only presentation fields and
    the already-saved output excerpt may be reduced; raw artifacts are unchanged.
    A minimum (possibly empty) block always reports omissions explicitly.
    """
    context = compact_context(context if context is not None else [])
    cards = context['cards']
    if not cards and not context['omitted_count']:
        return ''
    prefix = ('PRIOR RESEARCH CHECK RESULTS (planner-only auxiliary evidence; not canonical score/benefit):\n'
              + RESULTS_VIEW_MARKER)
    selected = []

    def render(rows):
        dropped = len(cards) - len(rows)
        reasons = dict(context['omission_reasons'])
        if dropped:
            reasons['planner_prompt_budget'] = dropped
        return prefix + json.dumps({'schema': 'sciel-tool-research-results-view/1',
            'cards': rows, 'omitted_count': context['omitted_count'] + dropped,
            'omission_reasons': reasons}, ensure_ascii=True, separators=(',', ':'), sort_keys=True)

    for card in cards:
        row = {key: deepcopy(value) for key, value in card.items() if key != 'schema'}
        row.update(card_sha256=content_hash(card), omitted_fields=[], output_excerpt_omitted_chars=0)
        if len(render(selected + [row])) > budget:
            # Preserve actual observations ahead of interpretive prose. The
            # complete question/comparison remains bound by the original hash.
            for key in ('interpretation', 'next_check', 'limitations', 'replay_comparison', 'question'):
                del row[key]
                row['omitted_fields'].append(key)
                if len(render(selected + [row])) <= budget:
                    break
        if len(render(selected + [row])) > budget:
            original = row['observations']['output_excerpt']
            try:
                parsed = json.loads(original)
            except (ValueError, TypeError):
                parsed = None
            structured_result = isinstance(parsed, dict) and (
                (type(parsed.get('schema')) is str and parsed['schema'] in PROJECTION_SCHEMAS)
                or (isinstance(parsed.get('output'), dict)
                    and parsed['output'].get('schema') == RESULT_SCHEMA))
            if structured_result:
                # Never turn a whole observation/value projection back into a
                # prefix of JSON. Keep the receipt binding and explicitly omit
                # the complete excerpt when this planner has insufficient room.
                row['observations']['output_excerpt'] = ''
                row['output_excerpt_omitted_chars'] = len(original)
                if len(render(selected + [row])) <= budget:
                    selected.append(row)
                continue
            low, high, fitted = 0, len(original), None
            # A string excerpt can end inside its encoded output JSON; the outer
            # projection remains valid JSON and labels the missing characters.
            while low <= high:
                middle = (low + high) // 2
                row['observations']['output_excerpt'] = original[:middle]
                row['output_excerpt_omitted_chars'] = len(original) - middle
                if len(render(selected + [row])) <= budget:
                    fitted, low = deepcopy(row), middle + 1
                else:
                    high = middle - 1
            row = fitted
        if row is not None and len(render(selected + [row])) <= budget:
            selected.append(row)
    return render(selected)


def _feedback_card(feedback, limit):
    """Prioritize the newest observation; an old traceback cannot evict a repair.

    The full immutable artifacts retain every output. This is an explicitly
    bounded presentation, not another growing lesson or an aggregate score.
    """
    remaining, cards = limit, []
    for index, item in enumerate(reversed(feedback)):
        if remaining < 80:
            break
        budget = remaining if index == len(feedback) - 1 else int(remaining * .8)
        if item.get('status') == 'canonical_evaluation':
            value = item['output']['observations']['canonical_evaluation']
            card = canonical_feedback(value,item['receipt_id'],budget)
            if len(card) > budget:
                # Never reinterpret a canonical refusal/failure as its earlier
                # successful tool execution, even if a projection regresses.
                card = json.dumps({'status':value.get('status'),'details_omitted':True},
                                  ensure_ascii=True,separators=(',',':'), sort_keys=True)
                if len(card) > budget:
                    card = '{"details_omitted":true}'
            cards.append(card)
            remaining -= len(card) + 2
            continue
        result = item.get('result') or {}
        heading = 'Status: ' + str(result.get('status', item.get('status', 'unavailable')))
        if result:
            heading += '\nHost execution: ' + str(result.get('host_status'))
        if item.get('tool_id'):
            heading += '\nTool: ' + item['tool_id']
        footer = '\nReceipt: ' + item['receipt_id'] if item.get('receipt_id') else ''
        if footer and budget < 512:
            # The complete receipt binds tool/runtime identity. Repeating those
            # fields must not evict the error or cut that immutable reference.
            heading = 'Status: ' + str(result.get('status', item.get('status','unavailable')))
            heading += '\nMetadata: see receipt.'
        body_budget = max(0, budget - len(heading) - len(footer) - 1)
        body = (success_context(item, body_budget) if result.get('status') == 'success'
                and body_budget >= MIN_CONTEXT_CHARS else _failure_context(item, body_budget))
        card = heading + '\n' + body + footer
        card = _clip(card, budget)
        cards.append(card)
        remaining -= len(card) + 2
    if len(cards) < len(feedback) and remaining > 40:
        cards.append('Earlier attempts omitted; retained in full receipts.')
    return _clip('\n\n'.join(cards), limit)


def _compact_tool_instructions(sequential, *, yield_after_success=False, public_task=False, minimal=False):
    """Same tool contract, without examples/repeated prose, for tight prompts."""
    text = '''RESEARCH OBJECTIVE: task/candidate uncertainty, not host trigger counters; check with control/next decision.
''' + _RESEARCH_PERMISSION + '''
create: immutable Python, module-level synchronous def run(payload), not a candidate.
STUDY-SUBJECT DATA: rules apply to candidates, not the tool API. No network/host files/credentials.
Payload keys: research_targets,search_evidence,arguments,task,task_prompt,parent_source,
parent_program_id,parent_score,ancestry_parent_id,island_id. payload/research_targets/
search_evidence/arguments are dicts: never json.loads those.
research_targets: /2; view references/observation_summary are not payload fields.
Follow absence_rules and access_paths[role]+access_paths.fields[field]. parent_* aliases selected;
ancestry_parent_id names its ancestor. Null=unavailable, zero=measured, selection_score=ranking.
artifact_manifest: keys/types/syntax, not observations; json_decoded_value is the inner root.
Import json; decode original strings as needed. Inspect omissions, never guess keys/retype arrays.
Use visible TOOL IDs for use/parents/dependencies ([] if none); arguments_json: object.
''' + _DEPENDENCY_GUIDANCE + '''Name source/hash; replay pins inputs.
run(payload) returns exactly five fields: {"schema":"sciel-tool-result/1","status":"success","observations":{},"diagnostics":{},"error":null}.
Data/state/code: observations; debug: diagnostics; failure: tool_error/nonempty error.
Negative findings may succeed. create pins replay_policy: exact_observations=identical observations,
ignore diagnostics; independent_execution=success, not reproducibility. use/skip cannot change it.
Failed/malformed tools cannot be shared. Replay, recipient trial, adoption, reuse differ; no benefit proof.
JSON only.
'''
    if sequential:
        text = """RESEARCH OBJECTIVE: choose an unresolved task/candidate decision using supplied evidence.
""" + _RESEARCH_PERMISSION + """
create: immutable module-level synchronous def run(payload), not candidate code.
Task rules aren't tool APIs. No network/host files/credentials.
payload/research_targets/search_evidence/arguments are dicts, not JSON strings.
research_targets /2: absence_rules; access_paths[role]+access_paths.fields[field].
View refs/observation_summary aren't payload; parent_*=selected, ancestry_parent_id=ancestor.
Null=unavailable; zero=measured; selection_score=ranking; task/task_prompt/parent_source=strings.
artifact_manifest=structure; json_decoded_value=inner root.
Import json for strings; don't guess keys/retype arrays.
use/parents/dependencies=visible TOOL IDs ([] if none); arguments_json=JSON object string.
""" + _DEPENDENCY_GUIDANCE + """Replay pins inputs.
run returns exactly {"schema":"sciel-tool-result/1","status":"success","observations":{},"diagnostics":{},"error":null}.
observations=data/state/code; diagnostics=debug; tool_error/nonempty error=failure; negatives can succeed.
Fixed replay_policy: exact_observations=identical observations, ignoring diagnostics;
independent_execution=success, not reproducibility. Failed tools cannot share.
Replay/trial/adoption/reuse differ; no benefit proof. JSON only.
""" + _DECISION_FIELDS_GUIDANCE + """Control/conditions and interpretation: decisions/scheduling=planner-only.
input_receipt_id adds research_stage; observations.continuation_state becomes payload.continuation_state.
Don't retype state/seed. observations.candidate_source=complete program.
submit_candidate(candidate_receipt_id): canonical evaluation/novelty; one measured candidate,
two pre-eval refusal retries. Branch trials rebuild inputs.
Trial cap:finish/submit; repairs capped; CPU/wall/calls recorded.
""" + _ACTION_GUIDANCE
        if yield_after_success:
            text += _YIELD_GUIDANCE
        if public_task or minimal:
            text = '''RESEARCH: resolve a task/candidate decision with a control.
''' + _RESEARCH_PERMISSION + '''
create: immutable sync run(payload), not candidate code. No network/host files/credentials.
Dicts:payload/search_evidence/research_targets/arguments; json.loads strings only (import json).
Frozen paths/nulls; view refs aren't payload. parent_*=selected; ancestry_parent_id=ancestor.
artifact_manifest=format; json_decoded_value=inner root. Don't guess keys/retype arrays.
use/parents/dependencies=visible TOOL IDs ([] if none); arguments_json=JSON object string.
m_<id>: only METHOD_ID/invoke(payload), no raw helpers/constants.
Return exactly {"schema":"sciel-tool-result/1","status":"success","observations":{},"diagnostics":{},"error":null}.
observations=data/state/code; diagnostics=debug; tool_error/nonempty error=failure; negatives may succeed.
Fixed replay_policy: exact_observations=identical observations (ignore diagnostics); independent_execution=success.
Failures cannot share; replay/trial/adoption/reuse do not prove benefit. JSON only.
''' + _DECISION_FIELDS_GUIDANCE + '''Planner-only:question/scheduling. input_receipt_id adds research_stage;
observations.continuation_state -> payload.continuation_state. Never retype state/seed.
submit_candidate: canonical/novelty checks of candidate_receipt_id's observations.candidate_source;
1 measured/opportunity, 2 pre-eval refusal retries; branch trials rebuild inputs.
Cap:finish/submit; repairs/costs recorded.
''' + _ACTION_GUIDANCE
            if yield_after_success:
                text += _YIELD_GUIDANCE
    else:
        text += 'Choose create/use/skip; repairs count. Canonical candidate evaluation/novelty unchanged.\n'
    return text


def _target_prompt(targets, budget, *, compact=False):
    """Describe the payload without labeling a projection as that payload.

        Field pointers are relative to the role pointer, both rooted in the
        actual run(payload) input. The digest belongs to the full frozen bundle,
        never this view. Even a small source budget retains the whole contract.
        """
    rows = []
    present = [name for name in ('selected', 'population_best') if targets[name] is not None]
    for name in ('selected', 'population_best'):
        target = targets[name]
        if target is None:
            rows.append({'role': name, 'available': False})
            continue
        measure = target['measurement']
        record = measure['record']
        saved = target['saved_evaluation']
        rows.append({'role':name, 'available':True,
            'kind':target['kind'], 'program_id':target['program_id'],
            'source_sha256':target['source_sha256'], 'selection_score':target['selection_score'],
            'measurement_status':measure['status'],
            'measurement_record_id':record['id'] if record else None,
            'measurement_record_sha256':record['record_sha256'] if record else None,
            'observation_summary':{k:record['child_observation'].get(k) for k in
                ('status','score','reached','confirmed')} if record else None,
            'saved_evaluation_status':saved['status'],
            'saved_evaluation_sha256':saved['reference']['sha256'] if saved['reference'] else None,
            'artifact_manifest':artifact_manifest(saved,
                '/research_targets/' + name + '/saved_evaluation/record/result/artifacts',
                200, inline_reference=False),
            'failure':{k:v for k,v in target['failure'].items() if k != 'artifacts'}
                if target['failure'] else None})
    view = {
        'schema':TARGET_VIEW_SCHEMA, 'payload_schema':targets['schema'],
        'research_targets_sha256':targets['sha256'], 'run_id':targets['run_id'],
        'task_hash':targets['task_hash'], 'frozen_at':targets['frozen_at'],
        'ancestry_parent':targets['ancestry_parent'],
        'access_paths':{
            'selected':'/research_targets/selected',
            'population_best':'/research_targets/population_best',
            'fields':{
                'source':'/source', 'measurement_status':'/measurement/status',
                'measurement_record':'/measurement/record',
                'child_observation':'/measurement/record/child_observation',
                'score':'/measurement/record/child_observation/score',
                'saved_evaluation':'/saved_evaluation',
                'saved_artifacts':'/saved_evaluation/record/result/artifacts',
                'saved_timing':'/saved_evaluation/record/result/timing',
                'saved_conditions':'/saved_evaluation/record/conditions'}},
        'absence_rules':{
            'population_best':'May be null; check before access.',
            'measurement':'record is null unless measurement.status == recorded.',
            'saved_evaluation':'record/reference are null unless status == available; reason explains absence.',
            'failed_candidate':'Source/failure retained; null ID/selection_score, unavailable measurement.',
            'score':'Zero is measured. Null is unavailable; selection_score is not a measurement.'},
        'references':rows}
    if compact:
        # These envelope fields remain in the exact payload identified by
        # research_targets_sha256. Retain every role/source/evaluation ID,
        # measurement summary, format manifest and executable access path.
        omitted = ('payload_schema','run_id','task_hash','frozen_at','ancestry_parent')
        for key in omitted:
            view.pop(key)
        view['envelope_fields_omitted'] = list(omitted)
        view['absence_rules'] = {
            'population_best':'nullable', 'measurement':'record iff status=recorded',
            'saved_evaluation':'record/reference iff available; otherwise reason',
            'failed_candidate':'source/failure retained; no measurement',
            'score':'zero measured; null unavailable; selection_score ranks'}
    def render():
        return ('FROZEN INPUT REFERENCES (available, not proof of actual tool use):\n'
            + TARGET_VIEW_MARKER + json.dumps(view, ensure_ascii=True, separators=(',', ':'), sort_keys=True) + '\n')
    heading = render()
    # Format metadata precedes optional source excerpts. Reserve every role
    # and its immutable reference even when no individual field fits.
    extra = max(0, budget - len(heading) - 500) * 3 // 4 // max(1, len(present))
    for row in rows:
        if row['available']:
            name = row['role']
            minimum = len(json.dumps(row['artifact_manifest'], ensure_ascii=True,
                                     sort_keys=True, separators=(',', ':')))
            expanded = artifact_manifest(targets[name]['saved_evaluation'],
                '/research_targets/' + name + '/saved_evaluation/record/result/artifacts',
                minimum + extra,inline_reference=False)
            if len(json.dumps(expanded,ensure_ascii=True,sort_keys=True,separators=(',',':'))) <= minimum + extra:
                row['artifact_manifest'] = expanded
    heading = render()
    remaining = max(0, budget - len(heading) - 500)
    if remaining < 32 * max(1,len(present)):
        # Exact source pointers/hashes already exist in the whole view.
        # Two empty source wrappers used to consume the repair reservation.
        return heading + 'SOURCE EXCERPTS: {"source_excerpt_omitted":true,"access":"access_paths[role]+access_paths.fields.source"}'
    parts = [heading]
    for name in present:
        source = targets[name]['source']
        cap = remaining // len(present)
        excerpt = _clip_json(source, cap) if cap >= 32 else ''
        parts.append(name.upper() + ' SOURCE: ' + json.dumps({
            'source_excerpt':excerpt, 'source_chars_total':len(source),
            'source_excerpt_omitted':excerpt != source,
            'full_source_path':"payload['research_targets']['"+name+"']['source']"},
            ensure_ascii=True, separators=(',', ':'), sort_keys=True))
    return '\n'.join(parts)


def _followups_prompt(plan, budget):
    view = plan.get('method_followups')
    if view is None or not view['eligible_count']:
        return ''
    prefix = 'METHOD_FOLLOWUPS_VIEW_JSON:\n'
    minimum = {**view, 'records': [], 'omitted_count': view['eligible_count']}
    bound = max(len(canonical_bytes(minimum)), budget - len(prefix))
    return prefix + canonical_bytes(compact_view(view, max_chars=bound)).decode('utf-8')


def _known_jobs_prompt(tools, plan, budget):
    if not tools.background_enabled:
        return ''
    # Keep whole actionable job identities ahead of optional prose. At the
    # minimum budget, report omitted rows rather than emitting partial JSON.
    jobs = plan.get('visible_jobs', [])
    if not jobs:
        return ''
    view = {'jobs': [], 'jobs_omitted': len(jobs)}
    marker = 'KNOWN JOBS (compact last committed observations):\n'
    for row in reversed(jobs):
        summary = {key: row.get(key) for key in ('job_id', 'tool_id', 'status',
            'collected', 'continuation_available', 'tool_result_status')}
        candidate = {'jobs': view['jobs'] + [summary],
                     'jobs_omitted': view['jobs_omitted'] - 1}
        if len(marker + json.dumps(candidate, ensure_ascii=True, sort_keys=True)) > budget:
            break
        view = candidate
    return marker + json.dumps(view, ensure_ascii=True, sort_keys=True)


def _replay_feedback_prompt(tools, plan, budget=None):
    rows = plan.get('author_replay_feedback', [])
    if not rows:
        return ''
    # Whole typed records, never a clipped JSON fragment. The block is
    # reserved before variable task/source/context sections and says which
    # role received the evidence rather than pretending the author reran it.
    bound = min(5000, tools.config['max_prompt_chars'] // 4) if budget is None else budget
    selected = []
    for row in rows:
        for count in range(len(row['changed_paths']), -1, -1):
            candidate = shorten_feedback(row, count)
            body = {'role': 'original_author', 'comparisons': selected + [candidate],
                    'comparisons_omitted': len(rows) - len(selected) - 1}
            if len(json.dumps(body, ensure_ascii=True, sort_keys=True)) <= bound:
                selected.append(candidate)
                break
    body = {'role': 'original_author', 'comparisons': selected,
            'comparisons_omitted': len(rows) - len(selected)}
    return ('REPLAY FEEDBACK: Independent tests, not benefit; version policy is immutable.\n'
            'REPLAY_FEEDBACK_JSON:\n' + json.dumps(body, ensure_ascii=True, sort_keys=True))
