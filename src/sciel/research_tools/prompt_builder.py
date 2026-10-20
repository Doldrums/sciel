"""Assemble bounded research prompts from immutable episode inputs."""
from __future__ import annotations

from copy import deepcopy
import json
from ..engine.research_targets import validate_targets
from ..methods.models import canonical_bytes
from .research_steps import state_prompt
from . import experiment_protocol, shared_evidence
from .canonical_diagnostics import add_summary as add_canonical_diagnostics
from .observation_cards import _clip
from .prompt_sections import _compact_tool_instructions, _feedback_card, _research_results_prompt
from .tool_artifacts import _sha
from .tool_protocol import (
    PROMPT_MARKER,
    _ACTION_GUIDANCE,
    _DECISION_FIELDS_GUIDANCE,
    _DEPENDENCY_GUIDANCE,
    _QUEUE_GUIDANCE,
    _RESEARCH_PERMISSION,
    _YIELD_GUIDANCE,
)

# A worked tool: models given only the contract wrote candidate programs instead.
_TOOL_EXAMPLE = '''A tool imports what a candidate imports: sciel.population.population() lists the run's best and the
selected program with the exact output the evaluator scored; sciel.task.asset(name) loads a module the task provides.
A complete measuring tool (the tool itself never defines the task's entry point):
from sciel.population import population
def run(payload):
    programs = population()
    output = programs[0]['output'] if programs else None
    return {"schema": "sciel-tool-result/1", "status": "success", "error": None, "diagnostics": {},
            "observations": {"best_score": programs[0]['score'] if programs else None,
                             "output_keys": sorted(output) if isinstance(output, dict) else None}}
'''

# The second role: models shown only measuring tools described candidates in next_decision instead of building them.
_CANDIDATE_EXAMPLE = '''A tool can also build a candidate: compute a better output, return a complete program as a string,
then submit_candidate with that invocation's receipt_id. Skeleton (improve is this cell's computation for the task):
from sciel.population import population
def run(payload):
    found = improve(population()[0]['output'], payload)
    source = payload['parent_source'] + '\\nFOUND = ' + repr(found) + '\\n'  # then an entry point returning FOUND
    return {"schema": "sciel-tool-result/1", "status": "success", "error": None, "diagnostics": {},
            "observations": {"candidate_source": source}}
'''


def _prompt(tools, plan, snapshot, available, feedback, attempt, *, research_state=None, scientific_step=0):
    header = {key: plan[key] for key in ('action_id', 'island_id', 'parent_program_id', 'iteration')}
    header['attempt'] = attempt
    instructions = '''RESEARCH OBJECTIVE: a reusable check of task/candidate uncertainty, not host trigger counters.
In reason name uncertainty, supplied input/source, check and next candidate/experiment decision.
Use source/results, failures and prior checks. Missing input: choose another supported check or skip.
Choose create, use or skip.
''' + _RESEARCH_PERMISSION + '''
create.code: module-level synchronous def run(payload). Static checks, CPU search and code transforms allowed.
''' + _TOOL_EXAMPLE + '''STUDY-SUBJECT DATA: rules apply to candidates, not the tool API.
Never submit a standalone candidate as the tool.
parents/dependencies: visible lineage/import TOOL IDs, not program IDs. Use [] for a new independent tool.
use: visible tool_id; arguments_json: object. Repairs count; immutable versions.
Payload keys: task,task_prompt,parent_program_id,parent_source,parent_score,
ancestry_parent_id,research_targets,island_id,arguments. research_targets: /2.
View references/observation_summary aren't payload fields. Use absence_rules,
access_paths[role]+access_paths.fields[field]. artifact_manifest: keys/types/syntax, not measurements.
Import json to decode strings; json_decoded_value=inner root. Inspect omissions; never retype arrays.
SCHEDULING CONTEXT is planning-only, not a runtime payload field. Don't invent missing values
or read payload['stagnation'] or payload['stagnation_evidence'].
Guarded access (role is selected or population_best):
target = payload['research_targets'][role]
measurement = target['measurement'] if target is not None else None
observation = measurement['record']['child_observation'] if measurement is not None and measurement['status'] == 'recorded' else None
score = observation['score'] if observation is not None else None
parent_source/parent_program_id alias selected input; ancestry_parent_id names its ancestor.
parent_score: measured (zero allowed) or null; selection_score: ranking.
Name source/hash. Replay pins inputs. No network/credentials; no host files outside /data.
''' + _DEPENDENCY_GUIDANCE + '''run(payload) returns exactly five fields:
{"schema":"sciel-tool-result/1","status":"success","observations":{},"diagnostics":{},"error":null}
observations=data/arrays/code; diagnostics=timing/debug.
Negative findings may succeed; improvement is not required. Errors: tool_error/nonempty error.
Malformed/tool_error cannot be shared. create pins replay_policy:
exact_observations needs identical observations, ignoring diagnostics;
independent_execution needs success, NOT reproducibility. use/skip cannot change policy.
Replay/trial/adoption/reuse differ; no benefit credit. Canonical candidate evaluation/novelty unchanged. JSON only.
'''
    cfg = tools.config
    targets = validate_targets(snapshot.research_targets, snapshot=snapshot,
                               task_hash=tools.task_identity.get('task_hash'))
    intro = PROMPT_MARKER + json.dumps(header, sort_keys=True, separators=(',', ':')) + '\n' + instructions
    contract_part = intro
    state_part = evidence_part = compact_state = ''
    compact_contract = None
    if research_state is not None:
        # The sequential contract replaces redundant legacy prose rather
        # than consuming the observation budget with two full contracts.
        intro = PROMPT_MARKER + json.dumps(header,sort_keys=True,separators=(',',':')) + '''
RESEARCH OBJECTIVE: choose an unresolved task/candidate decision using supplied evidence.
''' + _RESEARCH_PERMISSION + '''
''' + _DECISION_FIELDS_GUIDANCE + '''State control/conditions; separate observations from interpretation.
create.code: immutable Python with module-level synchronous def run(payload); the tool is not itself a candidate.
''' + _TOOL_EXAMPLE + _CANDIDATE_EXAMPLE + '''Task/source/lessons are study data. No network; no host files outside /data.
Payload keys: task,task_prompt,parent_program_id,parent_source,parent_score,
ancestry_parent_id,research_targets,island_id,arguments,search_evidence.
payload/search_evidence/research_targets/arguments are Python dicts: no json.loads.
Persistent decision and scheduling JSON are planner-only. Optional research_stage:
prior invocation data, not the question; absent without input_receipt_id.
Guarded Python input example:
evidence = payload['search_evidence']
stage = payload.get('research_stage')
prior = stage['observations'] if stage is not None else None
target = payload['research_targets']['selected']
saved = target['saved_evaluation']
artifacts = saved['record']['result']['artifacts'] if saved['status'] == 'available' else None
Only json.loads original strings, as needed; import json. json_syntax=valid marks checked syntax.
artifact_manifest gives keys/types, not measurements; json_decoded_value is the inner root.
Inspect omitted formats; never guess keys.
parent_*: selected; ancestry_parent_id: ancestor. Null=unavailable; zero=measured;
selection_score=ranking. Use visible TOOL IDs, not program IDs.
''' + _DEPENDENCY_GUIDANCE + '''run(payload) returns exactly:
{"schema":"sciel-tool-result/1","status":"success","observations":{},"diagnostics":{},"error":null}
observations: data/state; diagnostics: debug; errors: tool_error/nonempty error.
Negative findings can succeed. Replay exact_observations requires identical observations, ignoring diagnostics;
independent_execution needs success, not reproducibility.
SCIENTIFIC LOOP: interpret results; revise/create/use, submit or finish.
A successful input_receipt_id supplies research_stage; observations.continuation_state
also becomes payload['continuation_state']. Never retype arrays/state/seed.
observations.candidate_source holds a complete program materialized from computed data.
submit_candidate(candidate_receipt_id): canonical evaluation/novelty; one measured
candidate/opportunity, two extra submissions only after pre-evaluation refusal.
Other-branch method trials rebuild data there. Execution/replay/publication prove no benefit.
finish: supported/refuted/inconclusive.
At trial cap finish/submit, no tool. Repairs separately bounded; CPU/wall/calls recorded.
Missing artifacts are unknown; timing/conditions are separate.
''' + _ACTION_GUIDANCE
        if cfg['yield_after_success']:
            intro += _YIELD_GUIDANCE
        contract_part = intro
        prior_evaluations = []
        for step in research_state['steps']:
            ident = step.get('canonical_evaluation_id')
            if ident:
                saved = tools._read_artifact(ident,'candidate_evaluation')
                value = saved['evaluation']
                summary = {key:deepcopy(value[key]) for key in (
                    'status','evaluation_id','source_sha256','score','reached','confirmed',
                    'timing_seconds','timing_scope','efficacy') if key in value}
                summary['artifact_id'] = ident
                summary['kind'] = 'candidate_evaluation'
                if 'source' in saved:
                    summary['source_sha256'] = _sha(saved['source'])
                if len(json.dumps(value.get('cases'),ensure_ascii=True, sort_keys=True)) <= 600:
                    summary['cases'] = deepcopy(value.get('cases'))
                else:
                    summary['cases_omitted'] = True
                prior_evaluations.append(add_canonical_diagnostics(value,summary,budget=500))
        state_part = state_prompt(research_state, step=scientific_step,
            maximum=cfg['max_scientific_steps'],budget=min(2200,cfg['max_prompt_chars']//8),
            canonical_evaluations=prior_evaluations[-2:]) + '\n'
        intro += state_part
    if plan.get('suggested_tool_id'):
        intro += ('SUGGESTED METHOD: published tool ' + plan['suggested_tool_id']
                  + ' is offered for this question; use it if it fits before creating a new tool.\n')
    evidence = getattr(snapshot, 'research_search_evidence', None)
    if evidence:
        view = deepcopy(evidence)
        lineage = view.pop('selected_lineage',None)
        if isinstance(lineage,dict):
            view['selected_lineage_summary'] = {'complete':lineage.get('complete'),
                'program_count':len(lineage.get('program_ids',[])),
                'source_count':len(lineage.get('source_sha256s',[]))}
        encoded = json.dumps(view,ensure_ascii=True,separators=(',',':'), sort_keys=True)
        if len(encoded) > cfg['max_prompt_chars']//8:
            history_key = 'recent_evaluations' if 'recent_evaluations' in view else 'recent_admissions'
            rows = view.get(history_key,[])
            omitted = view.get(history_key+'_omitted',0)
            unknown = len(view.get('unverified_admissions',[])) + view.get('unverified_admissions_omitted',0)
            view = {key:deepcopy(view[key]) for key in ('schema','committed_iteration','accepted_evaluations',
                'global_best_score','last_record_iteration','last_record_accepted_evaluation',
                'evaluated_outputs','exact_recorded_output_repeats','non_improving_near_output_repeats','selected_runtime','selected_lineage_summary',
                'admission_stats') if key in view}
            view.update({history_key:[],history_key+'_omitted':len(rows)+omitted})
            view.update(unverified_admissions_omitted=unknown,
                scope='Bounded exact-field projection; full evidence is in payload.search_evidence.')
            for row in reversed(rows):
                candidate = {**view,history_key:[row]+view[history_key],
                    history_key+'_omitted':view[history_key+'_omitted']-1}
                if len(json.dumps(candidate,ensure_ascii=True,separators=(',',':'), sort_keys=True)) <= cfg['max_prompt_chars']//8:
                    view = candidate
            encoded = json.dumps(view,ensure_ascii=True,separators=(',',':'), sort_keys=True)
        evidence_part = 'OBSERVED SEARCH LIMITS JSON:\n' + encoded + '\n'
        intro += evidence_part
    if cfg.get('experimental_protocol'):
        intro += experiment_protocol.guidance()
    from ..eval.task_prompt import split_task_prompt
    public_task, extra_task = split_task_prompt(snapshot.task_prompt)
    if public_task:
        intro += '\n' + public_task + '\n'
    if tools.data_note:
        intro += tools.data_note
    controls = 'LIMITS: ' + json.dumps(cfg['limits'], sort_keys=True, separators=(',', ':'))
    if tools.background_enabled:
        controls = tools._job_instructions(plan)
    compact_targets = bool(public_task) and cfg['max_prompt_chars'] <= 10000
    minimum_view = tools._target_prompt(targets, 0, compact=compact_targets)
    # Reserve whole typed blocks before variable prose/source excerpts. The
    # old final string clip could cut the actual input contract mid-JSON.
    minimum_replay = tools._replay_feedback_prompt(plan, budget=0)
    prior_results = tools.lifecycle.delivery_context(
        getattr(snapshot, 'research_cells_context', None), recipient_cell_id=plan['cell_id'])
    minimum_results = _research_results_prompt(prior_results, 0)
    minimum_jobs = tools._known_jobs_prompt(plan, 0)
    minimum_followups = tools._followups_prompt(plan, 0)
    shared_view = plan.get('shared_observations')
    minimum_shared = shared_evidence.prompt(shared_view, 0)
    spare = (cfg['max_prompt_chars'] - sum(map(len, (intro, controls, minimum_view,
        minimum_replay, minimum_results, minimum_jobs, minimum_followups, minimum_shared))) - 100)
    if research_state is not None and spare < 1200:
        # A long-lived cell must not lose its next measurement to old IDs,
        # accounting metadata or history. Compact whole fields first.
        latest_evaluations = deepcopy(prior_evaluations[-1:])
        if latest_evaluations and 'failure_diagnostics' in latest_evaluations[0]:
            # Failure diagnosis, exact IDs and returned measurements outrank
            # optional stage/timing metadata in a small planner context.
            latest = latest_evaluations[0]
            for key in ('timing_seconds','timing_scope','reached','confirmed'):
                latest.pop(key, None)
            latest['other_fields_omitted'] = True
            raw_latest = tools._read_artifact(latest['artifact_id'],'candidate_evaluation')['evaluation']
            latest = add_canonical_diagnostics(raw_latest,latest,budget=380)
            diagnosis = latest['failure_diagnostics']
            for entry in diagnosis['entries']:
                entry.pop('sha256', None)
            diagnosis['entry_hashes_omitted'] = True
        compact_state = state_prompt(research_state,step=scientific_step,
            maximum=cfg['max_scientific_steps'],
            budget=max(450, 650 + len(canonical_bytes(latest_evaluations[-1])) if latest_evaluations else 450),
            canonical_evaluations=latest_evaluations) + '\n'
        # Names of every omitted accounting/prose field can outweigh the
        # current short question. Keep exact decision text when it fits the
        # same reservation; long text remains a labeled whole omission.
        marker, encoded_state = compact_state.split('\n',1)
        state_view = json.loads(encoded_state)
        omitted = state_view.pop('omitted_fields',[])
        state_view['other_state_fields_omitted'] = bool(omitted)
        state_view['decision_text_omitted'] = []
        if latest_evaluations and 'failure_diagnostics' in latest_evaluations[0]:
            for key in ('pending_method_trial','conclusion'):
                if state_view.get(key) is None:
                    state_view.pop(key, None)
                    state_view['other_state_fields_omitted'] = True
        for field in ('known_llm_cost','tool_resources_total','scientific_steps_total'):
            if field in state_view:
                state_view.pop(field)
                state_view['other_state_fields_omitted'] = True
        if not state_view.get('canonical_evaluations'):
            state_view.pop('canonical_evaluations',None)
        if state_view.get('canonical_evaluations_omitted') == 0 and not prior_evaluations:
            state_view.pop('canonical_evaluations_omitted')
        for field in ('question','next_decision'):
            value = research_state[field]
            candidate = {**state_view,field:value}
            rendered = marker+'\n'+json.dumps(candidate,ensure_ascii=True,separators=(',',':'), sort_keys=True)+'\n'
            if len(rendered) <= len(compact_state)-48:
                state_view = candidate
            else:
                state_view['decision_text_omitted'].append(field)
        compact_state = marker+'\n'+json.dumps(state_view,ensure_ascii=True,separators=(',',':'), sort_keys=True)+'\n'
        intro = intro.replace(state_part,compact_state,1)
        if evidence_part:
            compact_evidence = {key:deepcopy(view[key]) for key in (
                'schema','selected_runtime','exact_recorded_output_repeats','non_improving_near_output_repeats','global_best_score') if key in view}
            history_key = 'recent_evaluations' if ('recent_evaluations' in view or
                'recent_evaluations_omitted' in view) else 'recent_admissions'
            compact_evidence[history_key+'_omitted'] = (len(view.get(history_key,[]))
                + view.get(history_key+'_omitted',0))
            compact_evidence['unverified_admissions_omitted'] = (len(view.get('unverified_admissions',[]))
                + view.get('unverified_admissions_omitted',0))
            compact_evidence['other_fields_omitted'] = True
            intro = intro.replace(evidence_part,'OBSERVED SEARCH LIMITS JSON:\n'+
                json.dumps(compact_evidence,ensure_ascii=True,separators=(',',':'), sort_keys=True)+'\n',1)
        spare = (cfg['max_prompt_chars'] - sum(map(len,(intro,controls,minimum_view,
            minimum_replay,minimum_results,minimum_jobs,minimum_followups,minimum_shared))) - 100)
    if spare < 900 and tools.background_enabled:
        controls = tools._job_instructions(plan, compact=True)
        spare = (cfg['max_prompt_chars'] - sum(map(len,(intro,controls,minimum_view,
            minimum_replay,minimum_results,minimum_jobs,minimum_followups,minimum_shared))) - 100)
    if spare < 900:
        compact_contract = (PROMPT_MARKER + json.dumps(header,sort_keys=True,separators=(',',':'))
                            + '\n' + _compact_tool_instructions(research_state is not None,
                                yield_after_success=cfg['yield_after_success'], public_task=compact_targets))
        intro = intro.replace(contract_part,compact_contract,1)
        if cfg.get('experimental_protocol'):
            intro = intro.replace(experiment_protocol.guidance(), experiment_protocol.guidance(compact=True), 1)
        spare = (cfg['max_prompt_chars'] - sum(map(len,(intro,controls,minimum_view,
            minimum_replay,minimum_results,minimum_jobs,minimum_followups,minimum_shared))) - 100)
    if cfg.get('independent_research'):
        intro = intro.replace(_ACTION_GUIDANCE, _QUEUE_GUIDANCE)
    if spare < 0 and compact_state and state_view.get('canonical_evaluations_omitted') == 0:
        # On the hard boundary omit a redundant zero, preserving positive
        # omission counts and every retained canonical/decision field.
        state_view.pop('canonical_evaluations_omitted')
        shorter = marker+'\n'+json.dumps(state_view,ensure_ascii=True,
            separators=(',',':'),sort_keys=True)+'\n'
        intro = intro.replace(compact_state,shorter,1)
        spare += len(compact_state)-len(shorter)
    if spare < 0 or (feedback and spare < 512):
        # A restored cell has no transient attempt feedback, but still needs
        # room for its saved measurements. Use the shortest complete API
        # contract before refusing either a restored or an active episode.
        # Artifact paths, state and typed evidence stay intact.
        previous = compact_contract or contract_part
        minimal = (PROMPT_MARKER + json.dumps(header, sort_keys=True, separators=(',', ':'))
            + '\n' + _compact_tool_instructions(research_state is not None,
                yield_after_success=cfg['yield_after_success'], minimal=True))
        if cfg.get('independent_research'):
            previous = previous.replace(_ACTION_GUIDANCE, _QUEUE_GUIDANCE)
            minimal = minimal.replace(_ACTION_GUIDANCE, _QUEUE_GUIDANCE)
        if len(minimal) < len(previous):
            intro = intro.replace(previous, minimal, 1)
            spare += len(previous) - len(minimal)
    if spare < 0:
        raise ValueError('research tool prompt cannot fit its frozen input contract')
    # Do not reserve a large replay allowance when there are no comparisons:
    # that used to starve the actual repair cause behind receipt identities.
    desired_replay = tools._replay_feedback_prompt(plan)
    replay_extra = min(max(0, len(desired_replay) - len(minimum_replay)), spare * 2 // 3)
    # Actual repair causes take precedence over optional comparison detail.
    # The comparison block still reports every omitted comparison explicitly.
    feedback_budget = (min(cfg['max_prompt_chars'] // 8, spare,
        max(512, spare - replay_extra)) if feedback else 0)
    feedback_part = 'PREVIOUS ATTEMPT FEEDBACK:\n' + _feedback_card(feedback, feedback_budget)
    replay_part = tools._replay_feedback_prompt(plan,
        budget=min(5000, cfg['max_prompt_chars'] // 4, spare - feedback_budget))
    # Observations own their space before optional target source excerpts or
    # lessons. A small prompt still retains a whole explicit omission block.
    result_budget = min(10000, cfg['max_prompt_chars'] // 4,
        cfg['max_prompt_chars'] - sum(map(len, (intro, controls, minimum_view,
            feedback_part, replay_part, minimum_jobs, minimum_followups, minimum_shared))) - 100)
    results_part = _research_results_prompt(prior_results, result_budget)
    job_budget = min(2000, cfg['max_prompt_chars'] - sum(map(len, (intro, controls,
        minimum_view, feedback_part, replay_part, results_part, minimum_followups, minimum_shared))) - 100)
    jobs_part = tools._known_jobs_prompt(plan, job_budget)
    followup_budget = min(8000, cfg['max_prompt_chars'] // 4,
        cfg['max_prompt_chars'] - sum(map(len, (intro, controls, minimum_view,
            feedback_part, replay_part, results_part, jobs_part, minimum_shared))) - 100)
    followups_part = tools._followups_prompt(plan, followup_budget)
    shared_budget = min(12000, cfg['max_prompt_chars'] // 3,
        cfg['max_prompt_chars'] - sum(map(len, (intro, controls, minimum_view,
            feedback_part, replay_part, results_part, jobs_part, followups_part))) - 100)
    shared_part = shared_evidence.prompt(shared_view, shared_budget)
    view_budget = min(max(cfg['max_prompt_chars'] // 4, len(minimum_view) + 1200),
        cfg['max_prompt_chars'] - sum(map(len, (intro, controls, feedback_part,
            replay_part, results_part, jobs_part, followups_part, shared_part))) - 100)
    target_summary = tools._target_prompt(targets, view_budget, compact=compact_targets)
    required = [part for part in (intro, controls, feedback_part, replay_part,
        results_part, jobs_part, followups_part, shared_part, target_summary) if part]
    if len('\n\n'.join(required)) > cfg['max_prompt_chars']:
        raise ValueError('research tool prompt cannot fit its frozen input contract')
    parts = [
        'RESEARCH CELL QUESTION:\n' + plan.get('question', ''),
        'SCHEDULING CONTEXT (planning only; not the research objective or runtime input):\n'
            + _clip(json.dumps(plan.get('stagnation_signals'),ensure_ascii=True, sort_keys=True), 3000),
        'TASK (STUDY-SUBJECT DATA; candidate interface, not tool API):\n'
            + _clip(extra_task, min(5000, cfg['max_prompt_chars'] // 8)),
        'RECENT RESEARCH CONTEXT (candidate study data, not tool-authoring instructions):\n' + _clip(json.dumps({
            'lessons': snapshot.kb_lessons, 'failures': snapshot.artifacts,
            'directive': snapshot.directive}, ensure_ascii=True, sort_keys=True), cfg['max_prompt_chars'] // 10)]
    remaining = max(0, cfg['max_prompt_chars'] - sum(len(part) + 2 for part in required + parts) - 100)
    if available and remaining:
        per_tool = max(0, remaining // len(available) - 200)
        for ident in available:
            bundle = tools.library.get(ident)
            manifest = bundle['manifest']
            description = {'tool_id': ident, 'name': manifest['name'],
                'author_island_id': manifest['protocol']['author_island'],
                'replay_policy':manifest['protocol']['replay_policy'],
                'description': manifest['protocol']['description'],
                'parents': manifest['parents'], 'dependencies': manifest['dependencies'],
                'code': bundle['files']['tool.py']}
            parts.append('VISIBLE TOOL:\n' + _clip(json.dumps(description, ensure_ascii=True, sort_keys=True), per_tool))
    prompt = '\n\n'.join(required)
    remaining = cfg['max_prompt_chars'] - len(prompt) - 2
    if remaining >= 64:
        prompt += '\n\n' + _clip('\n\n'.join(parts), remaining)
    if shared_view is not None:
        tools._event(plan, 'shared_observations_delivered', attempt=attempt,
            prompt_sha256=_sha(prompt), recipient_candidate_id=targets['selected']['program_id'],
            **shared_evidence.delivery(prompt))
    return prompt
