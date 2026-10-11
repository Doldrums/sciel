"""Finite, frozen-context directive diagnostic; never a resumed evolution run.

Three parent snapshots x two repetitions x three arms. No population, policy,
lesson, or reflection updates; no generation retries. All candidates are audited
after the ordinary mechanical repair/graft and before native evaluation.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
import difflib
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
from types import SimpleNamespace
import uuid

from ..campaign.budget import BudgetedLLM, CampaignStopped, CostLedger, RequestNotDispatched
from ..campaign.budget import money, request_cost_bound, validate_llm_config
from ..campaign.identity import file_sha, object_sha, runtime_identity, source_identity, task_identity, verify_files
from ..config import load_config, validate
from ..engine.llm.trace import TracedLLM
from ..engine.novelty import CandidateNoveltyRegistry, _structural_ast
from ..engine.prompt.build import build, compose_directive
from ..engine.prompt.parse import extract_code
from ..engine.research import ResearchController
from ..engine.types import Program, Snapshot
from ..engine.validation import validate_candidate
from ..eval.registry import load_task
from ..hypothesis import hypothesis_instruction, parse_hypothesis
from ..evidence.change_audit import build_change_audit, summarize_change_audit, unit_catalog
from ..hypothesis.intervention_claim import extract, instruction
from ..evidence.measurement import metric_context
from .paired_eval import paired_draw

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = 'sciel-directive-probe/1'
ARMS = ('neutral', 'refine', 'escape')
DEFAULT_PARENTS = ('p225', 'p200', 'p55')
SEEDS = (2026091901, 2026091902, 2026091903)
MODEL = 'google/gemini-2.5-flash'
MAX_CALLS = 18
MAX_PROMPT_BYTES = 120000


def write(path, value, *, exclusive=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x' if exclusive else 'w', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def emit(folder, event, **fields):
    with (Path(folder) / 'events.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(dict(event=event, timestamp=datetime.now(timezone.utc).isoformat(),
            **fields), ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def schedule(parents):
    """Balanced cyclic arm order; repetitions share the exact same arm prompt."""
    rows = []
    for repeat in range(2):
        for index, parent in enumerate(parents):
            offset = (index + repeat) % len(ARMS)
            for arm in ARMS[offset:] + ARMS[:offset]:
                rows.append(dict(id=f'g{len(rows)+1:02d}', parent_id=parent, arm=arm, repeat=repeat))
    return rows


def prepare(history, output, pricing, *, parent_ids=DEFAULT_PARENTS, root=ROOT):
    """Read-only history preparation. No provider, evaluator, key or ledger."""
    root, history, output = Path(root).resolve(), Path(history).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('diagnostic output must be new; no overwrite or resume')
    if len(parent_ids) != 3 or len(set(parent_ids)) != 3:
        raise ValueError('exactly three distinct fixed parent IDs are required')
    historical = json.loads((history / 'checkpoint.json').read_text())
    cfg = validate(load_config(str(root / 'configs/circle_live_gemini.yaml')))
    research_cfg = deepcopy(cfg['research'])
    research_cfg.update(historical['research']['knowledge']['limits'])
    model_cfg = deepcopy(cfg['llm'])
    model_cfg.update(models=[MODEL], weights=[1.0], temperature=.7, max_tokens=32000, max_retries=1)
    validate_llm_config(model_cfg, pricing)
    task = load_task(str(root / 'tasks/oe/circle_packing'))
    research = ResearchController(research_cfg, SimpleNamespace(models=[MODEL], max_retries=1),
        SimpleNamespace(), lambda _: None, metric=metric_context(task))
    research.knowledge.load_state(historical['research']['knowledge'])
    programs = {row['id']: Program(**row) for row in historical['db']['programs']}
    if not set(parent_ids) <= set(programs):
        raise ValueError('selected parent absent from checkpoint source inventory')
    source = source_identity(root)
    history_pins = {name: file_sha(history / name) for name in ('checkpoint.json', 'sciel_events.jsonl')}
    parents = []
    for ident in parent_ids:
        parent = programs[ident]
        context = research.context(SimpleNamespace(_all=programs), parent, parent.island,
                                   historical['iteration_no'] + 1)
        parents.append(dict(id=ident, program=vars(parent), context=context,
            historical_timing=historical['eval_times'].get(ident), unit_catalog=unit_catalog(parent.code)))
    output.mkdir(parents=True)
    write(output / 'frozen/parents.json', parents)
    write(output / 'frozen/pricing.json', pricing)
    write(output / 'frozen/research_config.json', research_cfg)
    for name in {**source['files'], **source['project_files']}:
        target = output / 'frozen/source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, target)
    shutil.copytree(task.root, output / 'frozen/task', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    frozen = {str(p.relative_to(output)): file_sha(p) for p in sorted((output / 'frozen').rglob('*')) if p.is_file()}
    manifest = dict(schema=SCHEMA, prepared_at=datetime.now(timezone.utc).isoformat(),
        run_id='directive_probe_' + uuid.uuid4().hex, project_root=str(root), history=str(history),
        source=source, runtime=runtime_identity(), frozen_files=frozen, history_pins=history_pins,
        task=task_identity(task), model_config=model_cfg, limit_usd='3.00', max_calls=MAX_CALLS,
        operator='diff', parent_ids=list(parent_ids), ambient_seeds=list(SEEDS),
        schedule=schedule(parent_ids), directives='unchanged legacy compose_directive plus identical factual caveat',
        memory='current compact cards from frozen final historical KB; identical within each parent block',
        parent_selection='fixed competitive structural variants; purposive, not independent algorithm families',
        evaluation='one native smoke per draw, no confirmations; at most 63 invocations',
        novelty_scope='three frozen reference parents and evaluator-admitted probe sources; every proposal has a saved source artifact',
        duplicate_outcomes='linked prior matched-seed observations; reused references, never fresh measurements',
        claims='structural target agreement only; neither semantic truth nor causal attribution',
        learning=False, conclusions=False, resamples=0, resume=False)
    verify_files(root, source['files'])
    verify_files(history, history_pins)
    write(output / 'manifest.json', manifest)
    return manifest


def verify(output):
    output = Path(output).resolve()
    manifest = json.loads((output / 'manifest.json').read_text())
    if manifest.get('schema') != SCHEMA or manifest.get('schedule') != schedule(manifest['parent_ids']):
        raise ValueError('invalid fixed diagnostic manifest')
    if (manifest['max_calls'] != MAX_CALLS or len(manifest['schedule']) != MAX_CALLS
            or manifest['operator'] != 'diff' or manifest['learning'] or manifest['conclusions']
            or manifest['resamples'] != 0 or manifest['resume'] or manifest['ambient_seeds'] != list(SEEDS)
            or manifest['limit_usd'] != '3.00'):
        raise ValueError('diagnostic limits/policies differ')
    if Path(manifest['project_root']).resolve() != ROOT:
        raise ValueError('execute using the source checkout bound by this manifest')
    verify_files(output, manifest['frozen_files'])
    actual_files = {str(path.relative_to(output)) for path in (output / 'frozen').rglob('*')
                    if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ('.pyc', '.pyo')}
    if actual_files != set(manifest['frozen_files']):
        raise ValueError('frozen diagnostic file inventory changed')
    verify_files(ROOT, manifest['source']['files'], exact_root='src', suffixes=('.py', '.txt'),
                 exclude_build_metadata=True)
    verify_files(ROOT, manifest['source']['project_files'])
    verify_files(manifest['history'], manifest['history_pins'])
    if runtime_identity() != manifest['runtime']:
        raise ValueError('runtime changed since diagnostic preparation')
    return manifest


def prompt_for(parent, baseline, task, arm):
    if arm not in ARMS:
        raise ValueError('unknown diagnostic arm')
    program, context = parent['program'], parent['context']
    # Quote the actual native smoke cap. compose_directive asks the program for a safety margin.
    directive = compose_directive(task.smoke_timeout, baseline.get('timing', parent['historical_timing']),
        plateau=arm == 'escape', refine=arm == 'refine')
    directive += ('\nMEASUREMENT CAVEAT (identical across arms): wall-time headroom alone does not '
        'increase a fixed restart/step count. A speed-to-quality claim must identify the actual '
        'additional work and the condition enabling it; no restart/solver-step telemetry is available '
        'here. The directive is a search instruction, not an established bottleneck diagnosis.')
    # Only the chosen baseline is the forecast comparator; the other paired draws are analysis data.
    scalar = baseline['score']
    snapshot = Snapshot(0, 'diff', MODEL, 0, program['id'], program['code'], scalar,
        directive=directive, kb_lessons=context['lesson_texts'], research_context={'fragments': []},
        required_entrypoints=['run_packing'],
        task_prompt='Pack 26 non-overlapping circles into the unit square. Maximize sum of radii. '
            'Keep run_packing() returning centers, radii, sum_radii. The native primary score is '
            'validity-gated sum_radii / 2.635, tolerance 1e-6.')
    data = dict(parent_id=program['id'], parent_source_sha256=hashlib.sha256(program['code'].encode()).hexdigest(),
        parent_score=scalar, metric=metric_context(task), baseline_kind='first successful new ambient-seed draw',
        historical_score=program['score'], prior_lessons_scope='observational history, not causal proof',
        memory_sha256=object_sha(context['lesson_texts']), injected_ids=context['injected_ids'])
    text = (hypothesis_instruction() + '\n' + instruction(parent['unit_catalog']) + '\n'
        'The INTERVENTION_JSON and HYPOTHESIS_JSON declarations share this response with code; '
        'they are pre-evaluation claims, not independent or prior-to-coding evidence.\n'
        'FIXED_BASELINE_CONTEXT:\n' + json.dumps(data, ensure_ascii=False, allow_nan=False) + '\n'
        + build('diff', snapshot))
    if len(text.encode()) > MAX_PROMPT_BYTES:
        raise ValueError('frozen prompt exceeds diagnostic byte allowance')
    return text


class ProbeLedger(CostLedger):
    def reserve(self, bound_usd, *, run_id, model):
        # The single executor owns STARTED. Count also survives interrupted processes.
        state = json.loads(self.path.read_text())
        if len(state['calls']) >= MAX_CALLS:
            raise RequestNotDispatched('18-request diagnostic cap reached')
        return super().reserve(bound_usd, run_id=run_id, model=model)


def _draw_result(draw):
    """One adapter-normalized result, never substitute failed draws with zero."""
    return draw


def execute(output, *, api_key=None, provider=None, draw_fn=paired_draw):
    output = Path(output).resolve()
    manifest = verify(output)
    manifest_digest = file_sha(output / 'manifest.json')
    if (output / 'STARTED.json').exists():
        raise ValueError('diagnostic already started: automatic replay/resume forbidden')
    if (output / 'STOP').exists():
        raise ValueError('diagnostic stopped before dispatch')
    if provider is None and not api_key:
        raise ValueError('missing provider credential')
    write(output / 'STARTED.json', dict(manifest_sha256=manifest_digest,
        started_at=datetime.now(timezone.utc).isoformat(), pid=os.getpid()))
    ledger = ProbeLedger(output / 'cost_ledger.json', manifest['limit_usd'])
    pricing = json.loads((output / 'frozen/pricing.json').read_text())
    cfg = manifest['model_config']
    model = provider or BudgetedLLM(cfg, pricing, ledger, manifest['run_id'], api_key)
    active = {}
    llm = TracedLLM(model, str(output), lambda _: deepcopy(active))
    parents = json.loads((output / 'frozen/parents.json').read_text())
    task = load_task(str(output / 'frozen/task'))
    registry = CandidateNoveltyRegistry(neighbors=3)
    baseline_draws, candidates, observation_cache, started_cells = {}, [], {}, []
    completed = False
    try:
        for parent in parents:
            ident = parent['id']
            draws = []
            for seed in manifest['ambient_seeds']:
                if (output / 'STOP').exists(): raise CampaignStopped('local STOP marker')
                emit(output, 'evaluation_start', role='parent', parent_id=ident, ambient_seed=seed)
                draws.append(draw_fn(parent['program']['code'], task, seed,
                    work_dir=output / 'evaluations' / ident, python=sys.executable))
                emit(output, 'evaluation_result', role='parent', parent_id=ident, draw=draws[-1])
            baseline_draws[ident] = draws
            observation_cache[ident] = draws
            if not all(_draw_result(draw)['ok'] and _draw_result(draw)['valid'] for draw in draws):
                raise CampaignStopped('reference parent failed native calibration')
            registry.register(parent['program']['code'], ident, evaluation_ok=True)
        plan = []
        for cell in manifest['schedule']:
            parent = next(p for p in parents if p['id'] == cell['parent_id'])
            prompt = prompt_for(parent, _draw_result(baseline_draws[parent['id']][0]), task, cell['arm'])
            path = output / 'prompts' / (cell['id'] + '.txt')
            path.parent.mkdir(exist_ok=True)
            path.write_text(prompt, encoding='utf-8')
            plan.append({**cell, 'prompt_path': str(path.relative_to(output)), 'prompt_sha256': file_sha(path),
                'parent_source_sha256': hashlib.sha256(parent['program']['code'].encode()).hexdigest(),
                'memory_sha256': object_sha(parent['context']['lesson_texts'])})
        write(output / 'execution_plan.json', dict(manifest_sha256=file_sha(output / 'manifest.json'),
            baseline_draws=baseline_draws, cells=plan))
        rates = validate_llm_config(cfg, pricing)[MODEL]
        total_bound = sum((request_cost_bound((output / cell['prompt_path']).read_text(), cfg['max_tokens'], rates)
                           for cell in plan), Decimal(0))
        if total_bound > money(manifest['limit_usd']):
            raise RequestNotDispatched('all 18 conservative request bounds exceed fixed allowance')
        emit(output, 'plan_ready', cells=len(plan), conservative_bound_usd=str(total_bound))
        for cell in plan:
            if (output / 'STOP').exists(): raise CampaignStopped('local STOP marker')
            verify(output)
            if file_sha(output / 'manifest.json') != manifest_digest:
                raise ValueError('diagnostic manifest changed after start')
            active.clear(); active.update(cell)
            path = output / cell['prompt_path']
            if file_sha(path) != cell['prompt_sha256']:
                raise ValueError('prepared generation prompt changed')
            prompt = path.read_text()
            emit(output, 'generation_start', **cell)
            started_cells.append(cell['id'])
            response = llm.generate(prompt, MODEL, temperature=.7)
            text = response.text
            folder = output / 'candidates' / cell['id']
            folder.mkdir(parents=True)
            (folder / 'response.txt').write_text(text, encoding='utf-8')
            receipt = dict(**cell, cost=response.cost, finish_reason=response.finish_reason,
                refusal=response.refusal, response_sha256=file_sha(folder / 'response.txt'),
                status='response_received', draws=[])
            candidates.append(receipt)
            # Known incomplete output remains a charged failed attempt, never a retry.
            if response.finish_reason != 'stop' or response.refusal not in (None, ''):
                receipt.update(status='incomplete_response', draws=[])
            else:
                parent = next(p for p in parents if p['id'] == cell['parent_id'])
                declaration = extract(text, parent['unit_catalog'])
                hypothesis = parse_hypothesis(declaration['response_without_intervention'])
                proposed = extract_code(declaration['response_without_intervention'])
                (folder / 'proposed.py').write_text(proposed, encoding='utf-8')
                check = validate_candidate(proposed, parent_code=parent['program']['code'],
                    required_entrypoints=['run_packing'], code_hashes=set())
                final_code = check.code
                (folder / 'candidate.py').write_text(final_code, encoding='utf-8')
                novelty = registry.inspect(final_code)
                audit = build_change_audit(parent['program']['code'], final_code,
                    declared_targets=declaration['declared_targets'])
                write(folder / 'change_audit.json', audit)
                source_diff = ''.join(difflib.unified_diff(parent['program']['code'].splitlines(keepends=True),
                    final_code.splitlines(keepends=True), fromfile='parent.py', tofile='candidate.py'))
                (folder / 'source.diff').write_text(source_diff, encoding='utf-8')
                before_ast, _ = _structural_ast(parent['program']['code'])
                after_ast, _ = _structural_ast(final_code)
                if before_ast is not None and after_ast is not None:
                    (folder / 'ast.diff').write_text('\n'.join(difflib.unified_diff(before_ast.splitlines(),
                        after_ast.splitlines(), fromfile='parent.ast', tofile='candidate.ast')), encoding='utf-8')
                constructor = next((unit for unit in audit['units'] if unit['id'] == 'construct_packing'), None)
                constructor_changed = (constructor['status'] in ('modified', 'added', 'removed')
                    if constructor is not None and audit['comparison_available'] else None)
                declaration_operations = []
                if declaration['claim'] is not None and audit['comparison_available']:
                    actual = {unit['id']: unit for unit in audit['units']}
                    for operation in declaration['claim']['changed_units']:
                        target = operation['parent_unit_id'] or operation['proposed_name']
                        observed = actual.get(target, {}).get('status', 'absent')
                        expected = dict(modify='modified', add='added', remove='removed')[operation['change']]
                        declaration_operations.append(dict(unit_id=target, declared=operation['change'],
                            observed=observed, matches=observed == expected))
                receipt.update(declaration={k:v for k,v in declaration.items() if k != 'response_without_intervention'},
                    hypothesis=hypothesis.to_dict() if hypothesis is not None else None,
                    validation=check.decisions, validation_error=(vars(check.violation) if check.violation else None),
                    novelty=novelty, change_audit=summarize_change_audit(audit),
                    constructor_ast_changed=constructor_changed, declaration_operations=declaration_operations,
                    audit_files={name: file_sha(folder / name) for name in ('change_audit.json', 'source.diff', 'ast.diff')
                                 if (folder / name).exists()},
                    candidate_sha256=file_sha(folder / 'candidate.py'), draws=[])
                if check.violation:
                    receipt['status'] = 'invalid_candidate'
                elif not novelty['accept']:
                    receipt['status'] = 'duplicate'
                    matched = novelty.get('matched_program_id') or novelty.get('matched_evaluation_id')
                    receipt['shared_observation_id'] = matched
                    receipt['reused_draws'] = deepcopy(observation_cache.get(matched, []))
                    receipt['reuse_scope'] = ('Historical reference observations for the matched SHA/AST; '
                        'not fresh trials or a measurement of different source bytes.')
                else:
                    registry.register(final_code, cell['id'])
                    receipt['status'] = 'remeasured' if novelty.get('remeasurement') else 'evaluated'
                    # Save the pre-evaluation audit even if numerical work is interrupted.
                    write(folder / 'pre_evaluation.json', receipt)
                    for seed in manifest['ambient_seeds']:
                        if (output / 'STOP').exists(): raise CampaignStopped('local STOP marker')
                        emit(output, 'evaluation_start', role='candidate', candidate_id=cell['id'], ambient_seed=seed)
                        draw = draw_fn(final_code, task, seed, work_dir=output / 'evaluations' / cell['id'],
                                       python=sys.executable)
                        receipt['draws'].append(draw)
                        emit(output, 'evaluation_result', role='candidate', candidate_id=cell['id'], draw=draw)
                    registry.register(final_code, cell['id'],
                        evaluation_ok=all(_draw_result(d)['ok'] for d in receipt['draws']))
                    observation_cache[cell['id']] = deepcopy(receipt['draws'])
            write(folder / 'receipt.json', receipt)
            write(output / 'novelty_registry.json', registry.state(), exclusive=False)
            emit(output, 'candidate_complete', **receipt)
        completed = True
    except BaseException as exc:
        emit(output, 'stopped', error_type=type(exc).__name__,
            reason=str(exc)[:300] if isinstance(exc, (CampaignStopped, ValueError)) else 'local diagnostic failure',
            retry=False)
        raise
    finally:
        summary = summarize(manifest, baseline_draws, candidates)
        summary['status'] = 'complete' if completed else 'stopped'
        summary['started_cells'] = started_cells
        summary['without_returned_response'] = [ident for ident in started_cells
                                               if ident not in {row['id'] for row in candidates}]
        summary['cost_ledger'] = json.loads((output / 'cost_ledger.json').read_text())
        write(output / 'summary.json', summary)
        emit(output, 'end', status=summary['status'], completed_candidates=len(candidates))
    return summary


def summarize(manifest, baselines, candidates):
    arms = {}
    for arm in ARMS:
        rows = [c for c in candidates if c['arm'] == arm]
        arm_rows = []
        for row in rows:
            def paired(draws):
                differences = []
                for child, parent in zip(draws, baselines.get(row['parent_id'], [])):
                    a, b = _draw_result(child), _draw_result(parent)
                    if child['seed'] != parent['seed']:
                        raise ValueError('paired draw seeds differ')
                    differences.append(a['score'] - b['score'] if a['ok'] and b['ok'] else None)
                return differences
            arm_rows.append(dict(id=row['id'], parent_id=row['parent_id'], status=row['status'],
                declaration_error=row.get('declaration', {}).get('error'),
                constructor_ast_changed=row.get('constructor_ast_changed'),
                declaration_operations=row.get('declaration_operations'),
                change_audit=row.get('change_audit'), paired_native_score_deltas=paired(row.get('draws', [])),
                shared_observation_id=row.get('shared_observation_id'),
                paired_reused_reference_deltas=paired(row.get('reused_draws', [])),
                successful_draws=sum(_draw_result(d)['ok'] for d in row.get('draws', [])),
                valid_draws=sum(_draw_result(d)['ok'] and _draw_result(d)['valid']
                                for d in row.get('draws', []))))
        arms[arm] = dict(planned=6, returned=len(rows), evaluated=sum(c['status']=='evaluated' for c in rows),
            remeasurements=sum(c['status']=='remeasured' for c in rows),
            duplicates=sum(c['status']=='duplicate' for c in rows), candidates=arm_rows)
    return dict(schema=SCHEMA, run_id=manifest['run_id'], arms=arms,
        limitations=['18 proposals, three purposively chosen structural variants; not a sustained search comparison.',
            'Ambient RNG seeds can be overridden; wall-clock programs are not deterministic.',
            'Declared target coverage and AST changes do not verify semantic claims or causality.',
            'Rejected/duplicate proposals remain in arm denominators; failed draws are not imputed as zero.',
            'No C/reflection, lesson, population or controller updates.'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--history', type=Path, required=True)
    prep.add_argument('--output', type=Path, required=True)
    prep.add_argument('--pricing', type=Path, required=True)
    run = sub.add_parser('run')
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        manifest = prepare(args.history, args.output, json.loads(args.pricing.read_text()))
        print(json.dumps(dict(prepared=str(args.output), run_id=manifest['run_id'], provider_calls=0)))
    elif not args.execute:
        manifest = verify(args.output)
        print(json.dumps(dict(verified=True, provider_calls=0, max_calls=manifest['max_calls'])))
    else:
        from ..engine.factory import load_env
        load_env()
        key = os.environ.get('OPENROUTER_API_KEY') or getpass.getpass('OpenRouter API key: ')
        summary = execute(args.output, api_key=key)
        print(json.dumps(dict(status=summary['status'], run_id=summary['run_id'])))


if __name__ == '__main__':
    main()
