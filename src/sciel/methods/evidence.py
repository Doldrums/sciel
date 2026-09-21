"""Preassigned, source-bound evidence of utility on a finite held-out cohort.

This trusted host service never executes methods or judges prose. Registry
ordering establishes preregistration; immutable task measurements and actual
external invocation receipts establish observations. It cannot prove that an
operator never saw inputs outside this campaign, or statistical generalization.
The coordinator must serialize these calls with planning and charge their work.
"""
from __future__ import annotations

import math

from .models import content_hash, detached, require_hash

SCHEMA = 'sciel-method-holdout-protocol/1'


class EvidenceRefused(ValueError):
    """Evidence is missing, contaminated, or does not bind the assigned trial."""


def _need(condition, reason):
    if not condition:
        raise EvidenceRefused(reason)


def _number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


class EvidenceAssessor:
    def __init__(self, registry):
        self.registry = registry

    def _artifacts(self, kind):
        return [(event['seq'], event['data']['id'],
                 self.registry.get_artifact(kind, event['data']['id']))
                for event in self.registry.events()
                if event['kind'] == 'artifact_registered' and event['data']['kind'] == kind]

    def _seq(self, kind, ident):
        values = [e['seq'] for e in self.registry.events()
                  if e['kind'] == 'artifact_registered' and e['data'] == {'kind': kind, 'id': ident}]
        _need(len(values) == 1, 'artifact is not uniquely registered')
        return values[0]

    def _closure(self, ids):
        result = {}
        def visit(ident):
            require_hash(ident)
            if ident not in result:
                bundle = self.registry.get_method(ident)
                result[ident] = bundle
                for dependency in bundle['manifest']['dependencies']:
                    visit(dependency)
        for ident in ids:
            visit(ident)
        return result

    def _clean_reviewer(self, lab_id, revision_id, *, permitted_plan=None):
        revision = self.registry.revision(revision_id)
        _need(revision['lab_id'] == lab_id and not revision.get('previous_revision')
              and not revision.get('parent_revision') and not revision.get('evidence_refs'),
              'reviewer requires a fresh, unforked history without answer evidence')
        for kind in ('experiment_plan', 'proposal_context', 'proposal_receipt'):
            for _, ident, body in self._artifacts(kind):
                _need(body.get('lab_id') != lab_id or (kind == 'experiment_plan' and ident == permitted_plan),
                      'reviewer has another proposal or experiment history')

    def freeze_holdouts(self, method_id, assignments, *, treatment, control,
                        scope, task, runtime, resources, criterion):
        """Freeze all cases/reviewers before the designated author comparison.

        Local development may precede this protocol, but none of its assigned
        inputs may have appeared in a recorded experiment. No hidden answers
        belong in this artifact; reference measurements remain in the task.
        """
        method = self.registry.get_method(method_id)
        author = method['manifest']['author_lab']
        _need(isinstance(assignments, list) and 2 <= len(assignments) <= 16,
              'replication requires a bounded cohort of at least two reviewers')
        _need(isinstance(treatment, list) and isinstance(control, list)
              and 1 <= len(treatment) <= 8 and len(control) <= 8
              and all(isinstance(v, str) for v in treatment + control)
              and len(set(treatment)) == len(treatment) and len(set(control)) == len(control),
              'invalid treatment/control pipelines')
        _need(method_id in treatment, 'selected method needs an actual top-level treatment invocation')
        _need(method_id not in self._closure(control), 'control includes the selected method or its dependency')
        closure = self._closure(treatment + control)
        _need(isinstance(task, dict) and isinstance(runtime, dict) and isinstance(resources, dict)
              and task and runtime and resources, 'task/runtime/resource identities are required')
        _need(isinstance(criterion, dict) and set(criterion) == {'metric', 'direction', 'minimum_effect'}
              and isinstance(criterion['metric'], str) and criterion['metric']
              and criterion['direction'] in ('min', 'max') and _number(criterion['minimum_effect'])
              and criterion['minimum_effect'] >= 0, 'invalid predeclared effect criterion')
        _need(scope is not None and scope != '' and scope != {}, 'a finite declared scope is required')
        for bundle in closure.values():
            _need(bundle['manifest']['runtime_lock'].get('image_id') == runtime.get('image_id'),
                  'method runtime does not match the assignment')
            _need(all(key in resources and _number(resources[key]) and bound <= resources[key]
                      for key, bound in bundle['manifest']['resources'].items()),
                  'method exceeds the assigned resource envelope')
        seen_labs, seen_cases, seen_inputs, frozen = set(), set(), set(), []
        prior_inputs = {content_hash(value) for _, _, plan in self._artifacts('experiment_plan')
                        for value in plan.get('inputs', [])}
        for assignment in assignments:
            _need(isinstance(assignment, dict) and set(assignment) == {'reviewer_lab', 'cases'},
                  'invalid assignment fields')
            lab_id, cases = assignment['reviewer_lab'], assignment['cases']
            _need(isinstance(lab_id, str) and lab_id != author and lab_id not in seen_labs,
                  'reviewers must be distinct from author and each other')
            lab = self.registry.lab(lab_id)
            revision_id = lab['active_revision']
            self._clean_reviewer(lab_id, revision_id)
            _need(isinstance(cases, list) and 1 <= len(cases) <= 64, 'invalid held-out case count')
            normalized = []
            for case in cases:
                _need(isinstance(case, dict) and set(case) == {'id', 'input'}, 'invalid assigned case')
                case_id = case['id']
                _need(isinstance(case_id, str) and 0 < len(case_id) <= 160 and case_id not in seen_cases,
                      'case IDs must be nonempty and unique across the cohort')
                digest = content_hash(case['input'])
                _need(digest not in seen_inputs and digest not in prior_inputs,
                      'held-out input repeats a cohort or previously observed input')
                seen_cases.add(case_id)
                seen_inputs.add(digest)
                normalized.append(dict(id=case_id, input=detached(case['input']), input_sha256=digest))
            body = dict(reviewer_lab=lab_id, lab_revision_id=revision_id, history_id=lab_id, cases=normalized)
            frozen.append(dict(id=content_hash(body), **body))
            seen_labs.add(lab_id)
        body = dict(schema=SCHEMA, method_id=method_id, author_lab=author, assignments=frozen,
            treatment=treatment, control=control, scope=scope, task=task, runtime=runtime,
            resources=resources, criterion=criterion,
            rule='every assigned reviewer: all treatment outputs valid, paired mean > 0 and >= minimum_effect',
            independence='distinct fresh campaign histories and exact inputs; no claim of unseen external history',
            stopping='all assigned cases; no successful-subset selection')
        return self.registry.put_artifact('holdout_protocol', body)

    def _protocol(self, protocol_id):
        body = self.registry.get_artifact('holdout_protocol', protocol_id)
        _need(body.get('schema') == SCHEMA, 'unsupported holdout protocol')
        # Imported/content-addressed JSON is not proof that freeze_holdouts was
        # the writer. Recheck the structural invariants at the evidence boundary.
        method = self.registry.get_method(body.get('method_id'))
        _need(body.get('author_lab') == method['manifest']['author_lab'], 'protocol author differs')
        treatment, control = body.get('treatment'), body.get('control')
        _need(isinstance(treatment, list) and isinstance(control, list)
              and 1 <= len(treatment) <= 8 and len(control) <= 8
              and all(isinstance(v, str) for v in treatment + control)
              and len(set(treatment)) == len(treatment) and len(set(control)) == len(control)
              and body['method_id'] in treatment
              and body['method_id'] not in self._closure(control), 'invalid assigned method/control')
        criterion = body.get('criterion', {})
        _need(isinstance(criterion, dict) and set(criterion) == {'metric', 'direction', 'minimum_effect'}
              and isinstance(criterion['metric'], str) and criterion['metric']
              and criterion['direction'] in ('min', 'max') and _number(criterion['minimum_effect'])
              and criterion['minimum_effect'] >= 0, 'invalid assigned criterion')
        for key in ('task', 'runtime', 'resources'):
            _need(isinstance(body.get(key), dict) and body[key], 'missing assigned ' + key)
        for bundle in self._closure(treatment + control).values():
            _need(bundle['manifest']['runtime_lock'].get('image_id') == body['runtime'].get('image_id')
                  and all(key in body['resources'] and _number(body['resources'][key])
                          and bound <= body['resources'][key]
                          for key, bound in bundle['manifest']['resources'].items()),
                  'invalid assigned runtime/resource envelope')
        assignments = body.get('assignments')
        _need(isinstance(assignments, list) and 2 <= len(assignments) <= 16, 'invalid frozen cohort size')
        labs, ids, inputs = set(), set(), set()
        for assignment in assignments:
            _need(isinstance(assignment, dict) and set(assignment) ==
                  {'id', 'reviewer_lab', 'lab_revision_id', 'history_id', 'cases'}, 'invalid frozen assignment')
            lab = assignment['reviewer_lab']
            _need(isinstance(lab, str) and lab not in labs and lab != body['author_lab']
                  and assignment['history_id'] == lab
                  and assignment['id'] == content_hash({k: v for k, v in assignment.items() if k != 'id'}),
                  'frozen assignment history or identity differs')
            cases = assignment['cases']
            _need(isinstance(cases, list) and 1 <= len(cases) <= 64, 'invalid frozen cases')
            for case in cases:
                _need(isinstance(case, dict) and set(case) == {'id', 'input', 'input_sha256'}
                      and isinstance(case['id'], str) and 0 < len(case['id']) <= 160
                      and case['id'] not in ids, 'invalid or duplicate frozen case')
                digest = content_hash(case['input'])
                _need(case['input_sha256'] == digest and digest not in inputs, 'frozen input binding differs')
                ids.add(case['id'])
                inputs.add(digest)
            labs.add(lab)
        return body

    def validate_heldout_plan(self, protocol_id, assignment_id, plan):
        protocol = self._protocol(protocol_id)
        assignments = [a for a in protocol['assignments'] if a['id'] == assignment_id]
        _need(len(assignments) == 1, 'assignment does not belong to the frozen cohort')
        assignment = assignments[0]
        _need(plan.get('purpose') == 'heldout', 'assigned trial must be heldout, not a replay')
        for key in ('treatment', 'control', 'scope', 'task', 'runtime', 'resources', 'criterion'):
            _need(plan.get(key) == protocol[key], 'held-out plan changed frozen ' + key)
        _need(plan.get('measurement_protocols') == {arm: protocol['task'] for arm in ('control', 'treatment')},
              'held-out plan changed or omitted the task measurement protocol')
        for key, expected in (('lab_id', assignment['reviewer_lab']),
                              ('lab_revision_id', assignment['lab_revision_id']),
                              ('history_id', assignment['history_id'])):
            _need(plan.get(key) == expected, 'held-out plan changed assigned ' + key)
        _need(plan.get('inputs') == [c['input'] for c in assignment['cases']],
              'held-out plan changed assigned inputs or order')
        fields = dict(heldout_protocol_id=protocol_id, heldout_assignment_id=assignment_id,
                      assigned_case_ids=[c['id'] for c in assignment['cases']])
        bound_id = content_hash(dict(schema='sciel-method-artifact/1', kind='experiment_plan',
                                     body=dict(plan, **fields)))
        self._clean_reviewer(plan['lab_id'], plan['lab_revision_id'], permitted_plan=bound_id)
        return fields

    def _observations(self, experiment_id, plan, plan_id, used_receipts, used_invocations, measurements):
        experiment = self.registry.get_artifact('experiment', experiment_id)
        _need(experiment.get('plan_id') == plan_id, 'experiment refers to another plan')
        plan_seq, result_seq = self._seq('experiment_plan', plan_id), self._seq('experiment', experiment_id)
        _need(plan_seq < result_seq,
              'observations precede the frozen plan')
        refs = experiment.get('invocation_refs')
        _need(isinstance(refs, list) and len(refs) == len(set(refs)), 'duplicate invocation receipt references')
        calls = {}
        for ref in refs:
            _need(ref not in used_receipts, 'same call receipt reused as an independent observation')
            _need(plan_seq < self._seq('call_receipt', ref) < result_seq,
                  'invocation receipt is outside the frozen experiment lifetime')
            outer = self.registry.get_artifact('call_receipt', ref)
            receipt = outer.get('result', {})
            ident = receipt.get('invocation_id')
            _need(isinstance(ident, str) and ident not in calls and ident not in used_invocations,
                  'invocation identity is missing or reused')
            _need(outer.get('account') == self.registry.lab(plan['lab_id'])['budget_account']
                  and outer.get('key') == 'invoke:' + ident and outer.get('category') == 'execution'
                  and receipt.get('plan_id') == plan_id
                  and outer.get('reservation_id') == receipt.get('reservation_id'),
                  'invocation is not bound to the assigned lab/plan/account')
            calls[ident] = (ref, outer, receipt)
            used_receipts.add(ref)
            used_invocations.add(ident)
        results = experiment.get('results')
        _need(isinstance(results, dict) and set(results) == {'control', 'treatment'}, 'missing paired observations')
        metrics = {}
        called, contract_ok, valid_treatment = set(), True, True
        for arm in ('control', 'treatment'):
            rows = results[arm]
            _need(isinstance(rows, list) and len(rows) == len(plan['inputs']), 'assigned cases are incomplete')
            metrics[arm] = []
            for index, (value, row) in enumerate(zip(plan['inputs'], rows)):
                _need(row.get('input_sha256') == content_hash(value), 'observation input binding differs')
                digest, status = content_hash(value), 'ok'
                for stage, method_id in enumerate(plan[arm]):
                    invocation = content_hash(dict(plan=plan_id, arm=arm, case=index, stage=stage))
                    _need(invocation in calls, 'actual external invocation receipt is missing')
                    ref, outer, receipt = calls[invocation]
                    bundle = self.registry.get_method(method_id)
                    _need(receipt.get('method_id') == method_id
                          and receipt.get('source_sha256') == content_hash(bundle['files'])
                          and receipt.get('runtime_sha256') == content_hash(plan['runtime'])
                          and receipt.get('input_sha256') == digest,
                          'method source/runtime/dataflow binding differs')
                    called.add(invocation)
                    status = receipt.get('status')
                    _need(status in ('ok', 'failed'), 'unknown method execution cannot establish evidence')
                    _need(outer.get('status') == ('completed' if status == 'ok' else 'failed'),
                          'outer invocation outcome differs from execution')
                    if status != 'ok':
                        contract_ok = False
                        break
                    _need(receipt.get('dispatch_status') == 'invoked', 'composition is not actual invocation')
                    digest = content_hash(receipt.get('output'))
                    _need(receipt.get('output_sha256') == digest, 'method output binding differs')
                _need(row.get('status') == status, 'experiment hides a failed invocation')
                if status != 'ok':
                    _need(row.get('measurement') is None, 'failed invocation cannot have a successful measurement')
                    metrics[arm].append(None)
                    continue
                _need(row.get('output_sha256') == digest, 'final output binding differs')
                measurement = row.get('measurement')
                key = 'measure:' + content_hash(dict(plan=plan_id, arm=arm, case=index))
                candidates = measurements.get(key, [])
                _need(len(candidates) == 1, 'trusted task measurement receipt missing or ambiguous')
                measurement_seq, measured = candidates[0]
                _need(plan_seq < measurement_seq < result_seq,
                      'task measurement is outside the frozen experiment lifetime')
                _need(measured.get('account') == self.registry.lab(plan['lab_id'])['budget_account']
                      and measured.get('status') == 'completed'
                      and measured.get('result', {}).get('status') == 'ok'
                      and measured['result'].get('value') == measurement,
                      'observation differs from trusted task measurement')
                _need(isinstance(measurement, dict) and type(measurement.get('valid')) is bool,
                      'task did not declare measurement validity')
                metric = measurement.get('metrics', {}).get(plan['criterion']['metric'])
                _need(_number(metric), 'task metric is missing or nonfinite')
                metrics[arm].append(metric)
                if arm == 'treatment' and not measurement['valid']:
                    valid_treatment = False
        _need(called == set(calls), 'extra invocation receipt is outside the frozen pipeline')
        sign = 1 if plan['criterion']['direction'] == 'max' else -1
        effects = ([sign * (t - c) for t, c in zip(metrics['treatment'], metrics['control'])]
                   if contract_ok else [])
        _need(all(_number(v) for v in effects), 'paired effect overflows')
        mean = sum(v / len(effects) for v in effects) if effects else None
        _need(mean is None or _number(mean), 'paired mean overflows')
        positive = bool(contract_ok and valid_treatment and mean is not None
                        and mean > 0 and mean >= plan['criterion']['minimum_effect'])
        return dict(experiment_id=experiment_id, plan_id=plan_id,
                    contract_passed=contract_ok, valid_treatment=bool(contract_ok and valid_treatment),
                    paired_effects=effects, mean_effect=mean, benefit_supported=positive,
                    invocation_refs=refs)

    def assess_benefit(self, method_id, original_id, heldout_ids, *, protocol_id):
        """Assess the complete cohort; cache/replay never creates another sample."""
        protocol = self._protocol(protocol_id)
        _need(protocol['method_id'] == method_id, 'protocol belongs to another method version')
        _need(isinstance(heldout_ids, list) and all(isinstance(v, str) for v in heldout_ids)
              and len(heldout_ids) == len(set(heldout_ids))
              and original_id not in heldout_ids, 'duplicate or author experiment submitted as held-out')
        _need(len(heldout_ids) == len(protocol['assignments']), 'all assigned reviewers must be included')
        original = self.registry.get_artifact('experiment', original_id)
        original_plan = self.registry.get_artifact('experiment_plan', original['plan_id'])
        freeze_seq = self._seq('holdout_protocol', protocol_id)
        _need(freeze_seq < self._seq('experiment_plan', original['plan_id']),
              'author comparison was chosen before cohort preregistration')
        _need(original_plan['lab_id'] == protocol['author_lab']
              and original_plan['purpose'] in ('local', 'transfer'), 'original is not the author comparison')
        for key in ('treatment', 'control', 'scope', 'task', 'runtime', 'resources', 'criterion'):
            _need(original_plan.get(key) == protocol[key], 'author comparison changed frozen ' + key)
        _need(original_plan.get('measurement_protocols') ==
              {arm: protocol['task'] for arm in ('control', 'treatment')},
              'author comparison changed or omitted the task measurement protocol')
        author_plans = [(seq, pid) for seq, pid, p in self._artifacts('experiment_plan')
                        if seq > freeze_seq and p.get('lab_id') == protocol['author_lab']
                        and all(p.get(key) == protocol[key] for key in
                                ('treatment', 'control', 'scope', 'task', 'runtime', 'resources', 'criterion'))]
        _need(author_plans and author_plans[0][1] == original['plan_id'],
              'original must be the first matching author comparison after preregistration')
        all_inputs = {c['input_sha256'] for a in protocol['assignments'] for c in a['cases']}
        _need(not all_inputs.intersection(content_hash(v) for v in original_plan['inputs']),
              'held-out input repeats the author comparison')
        plans, assignments_seen = [], set()
        for experiment_id in heldout_ids:
            body = self.registry.get_artifact('experiment', experiment_id)
            plan_id = body['plan_id']
            plan = self.registry.get_artifact('experiment_plan', plan_id)
            assignment_id = plan.get('heldout_assignment_id')
            fields = self.validate_heldout_plan(protocol_id, assignment_id, plan)
            _need(all(plan.get(key) == value for key, value in fields.items()), 'plan lacks assigned case bindings')
            _need(assignment_id not in assignments_seen, 'same assignment replayed instead of independent cases')
            _need(freeze_seq < self._seq('experiment_plan', plan_id), 'held-out plan predates assignment')
            _need(self._seq('experiment_plan', original['plan_id']) < self._seq('experiment_plan', plan_id),
                  'author comparison must be fixed before held-out trials')
            self._clean_reviewer(plan['lab_id'], plan['lab_revision_id'], permitted_plan=plan_id)
            assignments_seen.add(assignment_id)
            plans.append((experiment_id, plan_id, plan))
        permitted = {pid for _, pid, _ in plans}
        # Also detects reuse after preregistration, not just before freezing.
        for _, pid, plan in self._artifacts('experiment_plan'):
            if pid not in permitted:
                _need(not all_inputs.intersection(content_hash(v) for v in plan.get('inputs', [])),
                      'assigned inputs leaked into another recorded experiment')
        measurements = {}
        for seq, _, body in self._artifacts('call_receipt'):
            if body.get('category') == 'measurement':
                measurements.setdefault(body.get('key'), []).append((seq, body))
        used_receipts, used_invocations = set(), set()
        author = self._observations(original_id, original_plan, original['plan_id'],
                                    used_receipts, used_invocations, measurements)
        cohort = [self._observations(eid, plan, pid, used_receipts, used_invocations, measurements)
                  for eid, pid, plan in plans]
        positive = author['benefit_supported'] and all(row['benefit_supported'] for row in cohort)
        evidence = dict(schema='sciel-method-benefit-evidence/1', method_id=method_id,
            level='benefit_replicated' if positive else 'benefit_not_replicated', scope=protocol['scope'],
            protocol_id=protocol_id, author_comparison=author, heldout_comparisons=cohort,
            experiments=[original_id] + heldout_ids, task=protocol['task'], runtime=protocol['runtime'],
            criterion=protocol['criterion'], supporting=[r['experiment_id'] for r in [author]+cohort if r['benefit_supported']],
            contradicting=[r['experiment_id'] for r in [author]+cohort
                           if r['contract_passed'] and not r['benefit_supported']],
            inconclusive=[r['experiment_id'] for r in [author]+cohort if not r['contract_passed']],
            uncertainty='Finite preassigned inputs and separate campaign histories only; '
                        'no statistical confidence, unseen external-history guarantee, or broader transfer claim.',
            invocation_attribution='actual top-level receipts only; internal dependency imports are not separate uses')
        return self.registry.put_artifact('method_evidence', evidence)
