"""Versioned research-tool contracts and planner instructions."""

CONTRACT = 'sciel-island-research-tools/34'

PROMPT_MARKER = 'SCIEL_RESEARCH_TOOL_EPISODE\n'

ARTIFACT_SCHEMA = 'sciel-research-tool-artifact/1'

TARGET_VIEW_SCHEMA = 'sciel-research-targets-view/1'

TARGET_VIEW_MARKER = 'FROZEN_INPUT_VIEW_JSON:\n'

RESULTS_VIEW_MARKER = 'RESEARCH_CHECK_RESULTS_VIEW_JSON:\n'

_RESEARCH_PERMISSION = '''CPU search, comparisons and candidate transforms are allowed with existing source/results.
Reuse known values; compute for new evidence. Prior model-only closures are interpretations, not tool policy.
'''

_DECISION_FIELDS_GUIDANCE = '''question, experiment, decision_if_supported, decision_if_refuted, next_decision: nonempty text, even finish/skip.
If no experiment ran, say so and why; state conditional branches and closure honestly. Never invent results/execution.
'''

_ACTION_GUIDANCE = '''For another worthwhile feasible experiment, issue create/use (or eligible continue) now.
finish closes without executing next_decision advice; cite evidence and limits:
answered, no worthwhile feasible check, or missing required input.
'''

_QUEUE_GUIDANCE = '''QUEUE: open work resumes independently. Select another experiment with create/use or eligible continue.
finish closes without executing next_decision advice; cite why: answered, no worthwhile feasible check, or missing required input.
'''

_YIELD_GUIDANCE = ('YIELD: successful foreground tools return observations to ordinary search before another '
    'planner call; this question stays open for a later opportunity.\n')

_DEPENDENCY_GUIDANCE = (
    'm_<id> proxy: ONLY METHOD_ID + invoke(payload), no raw functions/constants/helpers '
    'despite visible source. r=__import__(\'m_\'+exact_id).invoke(payload): five-field '
    "sciel-tool-result/1; check r['status']/r['error'], read nested r['observations'].\n")


LIFECYCLE_SCHEMA = 'sciel-cell-method-lifecycle/5'
