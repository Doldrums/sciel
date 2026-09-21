#!/usr/bin/env python3
"""Prepare/verify immutable pilot recipes. Never dispatch or execute candidates."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

from sciel.methods.__main__ import _no_credentials, validate_config
from sciel.methods.campaign import arm_config, freeze_campaign, load_campaign, validate_arm
from sciel.methods.examples import circle_state_schema, circle_toolbox
from sciel.methods.models import canonical_bytes, content_hash, validate_value
from sciel.methods.proposals import PROCEDURE_ID
from sciel.methods.registry import _atomic
from sciel.methods.tasks import Circle26Task, SortTask
from sciel.methods.technical import sort_bundle

REPO = Path(__file__).resolve().parents[1]
SCHEMA = 'sciel-method-pilot-recipe/1'


def source_identity():
    paths = [*sorted((REPO/'src/sciel/methods').glob('*.py')), Path(__file__).resolve()]
    return {str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def assigned_cases(task):
    """Hand-authored numeric fixtures, not candidate runs or optimized packings."""
    if task == 'sort':
        development = [('sort/dev-duplicates', [3, 1, 3, -2]), ('sort/dev-signed', [-4, 2, -4, 0])]
        heldout = [('sort/heldout-reversed', [6, 4, 2, 0, -2]), ('sort/heldout-repeated', [9, 9, -3, 2, -3, 9])]
    else:
        def grid(stagger, radius, seed, reverse=False):
            points = [[(column + .5) / 6 + (stagger if row % 2 else 0), (row + .5) / 5]
                      for row in range(5) for column in range(6)][:26]
            if reverse:
                points = [[1-y, x] for x, y in points]
            value = dict(centers=points, radii=[radius]*26, centers_dtype='float64', radii_dtype='float64',
                         budget_seconds=2.0, seed=seed, restart_index=0, maxiter=80)
            validate_value(value, circle_state_schema())
            return value
        development = [('circle/dev-grid', grid(0, .02, 11)), ('circle/dev-stagger', grid(.01, .015, 23))]
        heldout = [('circle/heldout-rotated', grid(0, .018, 37, True)),
                   ('circle/heldout-rotated-stagger', grid(.015, .012, 53, True))]
    def rows(values):
        return [dict(id=ident, input=value, input_sha256=content_hash(value)) for ident, value in values]
    return dict(schema='sciel-method-assigned-cases/1', development=rows(development), heldout=rows(heldout),
                provenance='Hand-authored data fixtures; no solver, evaluator, candidate or provider was run.',
                scope='Development mechanism pilot; same-task variants do not establish task-family transfer.')


def _bundle(body):
    return dict(body, id=content_hash(dict(schema='sciel-method/1', **body)))


def _write(output, relative, value, files):
    path = output / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_bytes(value) + b'\n'
    _atomic(path, raw, immutable=True)
    files[relative] = hashlib.sha256(raw).hexdigest()


def prepare(profile, pricing, model, image, output, *, repeats=1, controls=()):
    """Pure preparation with explicit prices; no credentials or network access."""
    wall, cpu = time.monotonic(), time.process_time()
    sources = source_identity()
    _no_credentials(profile); _no_credentials(pricing)
    allowed = {'schema','task','budget','resources','max_steps','max_cases','calendar_seconds',
               'storage_bytes','max_transfers','llm','upfront_costs','upfront_provenance'}
    if type(profile) is not dict or set(profile)-allowed or profile.get('schema') != 'sciel-method-pilot-profile/1':
        raise ValueError('invalid pilot profile')
    if profile.get('task') not in ('sort','circle26'):
        raise ValueError('unsupported pilot task')
    if type(repeats) is not int or not 1 <= repeats <= 10:
        raise ValueError('preparation supports 1..10 independent campaign assignments')
    if any(value not in ('unreplayed','solution_only') for value in controls) or len(set(controls)) != len(controls):
        raise ValueError('only implemented unreplayed and solution_only controls are available')
    task = SortTask() if profile['task'] == 'sort' else Circle26Task()
    resources = deepcopy(profile['resources'])
    values = ({'sort': sort_bundle(image, resources)} if profile['task'] == 'sort' else
              {name:_bundle(value) for name,value in circle_toolbox({'image_id':image},resources).items()})
    bundles = [values[name] for name in sorted(values)]
    control = [values['sort']['id']] if profile['task']=='sort' else [values['lp']['id'], values['slsqp']['id']]
    cases = assigned_cases(profile['task'])
    llm = dict(profile['llm'], models=[model], weights=[1], api_base='https://openrouter.ai/api/v1', max_retries=1)
    recipient_inputs = [deepcopy(cases['development'][1]['input'])]
    agendas = []
    for role in ('author','reviewer','recipient'):
        assigned = cases['development'][0 if role == 'author' else 1]
        agenda = ('Bounded mechanism pilot, not a scientific benefit claim. Improve valid task solutions using the same supplied tools. '
                  'Correctness and all charges matter; Stop when no bounded informative next test remains. '
                  'If proposing a JSON-state tool experiment, assigned development input is: ' +
                  canonical_bytes(assigned['input']).decode() + '. Fixed recipient control IDs: ' + json.dumps(control) + '. '
                  'A new standalone Circle solution instead uses the provided canonical cascade input schema; '
                  'do not describe cascade repeats as assigned-seed draws.' if profile['task']=='circle26' else
                  'Bounded sorting mechanism pilot. The supplied sorted control is already correct; no quality headroom is asserted. '
                  'Use the assigned development input '+canonical_bytes(assigned['input']).decode()+
                  '; Stop when no bounded informative next test remains.')
        agendas.append(dict(id=role, agenda=agenda, account_id=role, limits=deepcopy(profile['budget'])))
    base = dict(schema='sciel-method-run/1', mode='A', task=profile['task'], image_id=image,
                llm=llm, pricing=deepcopy(pricing), model=model, budget=deepcopy(profile['budget']),
                resources=resources, max_steps=profile['max_steps'], max_cases=profile['max_cases'],
                calendar_seconds=profile['calendar_seconds'], storage_bytes=profile['storage_bytes'], labs=agendas,
                common_methods=bundles, common_toolbox='none', claim='bounded_arm',
                routing=dict(author='author',reviewer='reviewer',recipient='recipient',control=control,
                             transfer_inputs=recipient_inputs,max_transfers=profile['max_transfers']),
                upfront_costs=deepcopy(profile.get('upfront_costs', [])))
    if 'upfront_provenance' in profile:
        base['upfront_provenance']=deepcopy(profile['upfront_provenance'])
    # Preflight every transport/runtime/resource field before writing anything.
    base = validate_config(base)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('preparation output must be empty; never overwrite a frozen pilot')
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    preparation = dict(schema='sciel-method-pilot-preparation/1',
        resources=dict(dollars=0, cpu_seconds=time.process_time()-cpu, gpu_seconds=0,
                       wall_seconds=time.monotonic()-wall,
                       storage_bytes=len(canonical_bytes(bundles))+len(canonical_bytes(cases)), steps=0),
        scope='Source packaging and fixture construction before manifest serialization; logical package/data bytes only.',
        exclusions=['Earlier toolbox design and implementation costs are unknown unless separately attested.',
                    'Remaining manifest/config serialization, Python startup, operating-system and Docker-daemon overhead are not measured here.'],
        source_identity=sources, toolbox_ids={name:value['id'] for name,value in values.items()},
        evidence='Direct local process timers; no model, optimizer, candidate, Docker or evaluator dispatch.')
    preparation_id = content_hash(preparation)
    base['upfront_costs'].append(dict(id=preparation_id, resources=preparation['resources'],
                                    evidence_refs=[preparation_id], description=preparation['scope']))
    base = validate_config(base)
    common = dict(task=dict(id=profile['task'],identity=content_hash(task.identity),metric=task.metric,direction=task.direction),
                  model=dict(selected=model,llm=llm,pricing=pricing),procedure=PROCEDURE_ID,
                  toolbox=[value['id'] for value in bundles],runtime_lock={'image_id':image},
                  splits={key:[item['id'] for item in cases[key]] for key in ('development','heldout')},
                  lab_count=3,concurrency=1,budget=base['budget'],
                  executor_resources=resources,controller_limits={key:base[key] for key in ('max_steps','max_cases','calendar_seconds','storage_bytes')},
                  lab_definitions=agendas,routing=base['routing'],methods_source_identity=sources,
                  assignments_sha256=content_hash(cases),execution_scope='development_only',
                  upfront_costs=[{key:value[key] for key in ('id','resources','evidence_refs')} for value in base['upfront_costs']])
    if 'upfront_provenance' in base:
        common['upfront_provenance']=base['upfront_provenance']
    plan = freeze_campaign(output/'campaign.json',common,repeats=repeats,controls=controls)
    files['campaign.json']=hashlib.sha256((output/'campaign.json').read_bytes()).hexdigest()
    _write(output,'preparation.json',dict(preparation,id=preparation_id),files)
    _write(output,'development.json',dict(schema=cases['schema'],cases=cases['development'],provenance=cases['provenance']),files)
    _write(output,'heldout.json',dict(schema=cases['schema'],cases=cases['heldout'],
           status='frozen_unexecuted_not_in_proposer_context',provenance=cases['provenance']),files)
    _write(output,'toolbox.json',bundles,files)
    units=[]
    for repeat in range(repeats):
        for arm in plan['arms']:
            frozen=arm_config(plan,arm,repeat)
            unit=deepcopy(base)
            unit.update({key:frozen[key] for key in ('mode','library_policy','transfer_policy')})
            unit=validate_config(unit)
            relative=f'units/{arm}-{repeat}/config.json'
            _write(output,relative,unit,files)
            _write(output,f'units/{arm}-{repeat}/campaign-unit.json',frozen,files)
            units.append(dict(arm=arm,repeat=repeat,config=relative,
                              campaign_unit=f'units/{arm}-{repeat}/campaign-unit.json',
                              configuration_sha256=content_hash(unit),
                              output=f'units/{arm}-{repeat}/run'))
    if source_identity()!=sources:
        raise ValueError('method implementation changed during preparation; incomplete recipe must not execute')
    manifest=dict(schema=SCHEMA,campaign_id=plan['id'],created_at=datetime.now(timezone.utc).isoformat(),
                  source_identity=sources,files=files,units=units,prepared_only=True,
                  scientific_claims=False,execution_scope='development_only',heldout_status='unexecuted',
                  constraints=['Run only after successful technical end-to-end acceptance and observed host headroom.',
                    'Launch units sequentially into fresh independent directories; no cross-arm or cross-repeat warm starts.',
                    'Stop at fixed resource/step/calendar ceilings, unresolved usage, broken contracts or absent informative next question.',
                    'No quality headroom or reproducible benefit is established by these fixtures.',
                    'Heldout values are frozen outside prompts; the current CLI does not execute a final heldout phase.',
                    'Historical construction/controller-overhead coverage remains a lower bound, regardless of a fees declaration.'])
    manifest['id']=content_hash(manifest)
    _atomic(output/'manifest.json',canonical_bytes(manifest)+b'\n',immutable=True)
    return manifest


def verify(path):
    path=Path(path)
    value=json.loads(path.read_bytes())
    claimed=value.pop('id')
    if value.get('schema')!=SCHEMA or content_hash(value)!=claimed:
        raise ValueError('pilot recipe identity differs')
    if source_identity()!=value['source_identity']:
        raise ValueError('method implementation drift; prepare a new pilot before dispatch')
    for relative,digest in value['files'].items():
        target=Path(relative)
        if target.is_absolute() or '..' in target.parts:
            raise ValueError('invalid prepared file path')
        if hashlib.sha256((path.parent/target).read_bytes()).hexdigest()!=digest:
            raise ValueError('prepared artifact drift: '+relative)
    plan=load_campaign(path.parent/'campaign.json')
    task = SortTask() if plan['common']['task']['id']=='sort' else Circle26Task()
    if content_hash(task.identity)!=plan['common']['task']['identity']:
        raise ValueError('trusted task/evaluator drift; prepare a new pilot before dispatch')
    common_cli = None
    for unit in value['units']:
        frozen=json.loads((path.parent/unit['campaign_unit']).read_bytes())
        validate_arm(plan,unit['arm'],unit['repeat'],frozen)
        config=validate_config(json.loads((path.parent/unit['config']).read_bytes()))
        if content_hash(config)!=unit['configuration_sha256']:
            raise ValueError('CLI configuration drift')
        policy = {key:config.pop(key) for key in ('mode','library_policy','transfer_policy')}
        if policy!={key:frozen[key] for key in policy}:
            raise ValueError('CLI arm policy differs from frozen comparison')
        if common_cli is None:
            common_cli=config
        elif config!=common_cli:
            raise ValueError('A/B/C CLI settings differ beyond the declared feature policies')
    return dict(status='verified',campaign_id=plan['id'],units=len(value['units']),
                prepared_only=True,paid_dispatches=0,heldout_status='unexecuted')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    create=commands.add_parser('prepare')
    create.add_argument('--profile',required=True,type=Path)
    create.add_argument('--pricing',required=True,type=Path,help='explicit provider price snapshot; never fetched or invented')
    create.add_argument('--model',required=True)
    create.add_argument('--image',required=True)
    create.add_argument('--output',required=True,type=Path)
    create.add_argument('--repeats',type=int,default=1)
    create.add_argument('--control',action='append',default=[],choices=['unreplayed','solution_only'])
    check=commands.add_parser('verify');check.add_argument('manifest',type=Path)
    args=parser.parse_args(argv)
    try:
        if args.command=='verify':
            result=verify(args.manifest)
        else:
            result=prepare(json.loads(args.profile.read_bytes()),json.loads(args.pricing.read_bytes()),args.model,args.image,
                           args.output,repeats=args.repeats,controls=args.control)
            result=dict(status='prepared',manifest=str(args.output/'manifest.json'),
                        campaign_id=result['campaign_id'],units=len(result['units']),paid_dispatches=0)
        print(json.dumps(result));return 0
    except (ValueError,OSError,KeyError,TypeError) as exc:
        print(json.dumps(dict(status='refused',error_type=type(exc).__name__,reason=str(exc))),file=sys.stderr);return 2


if __name__=='__main__':
    raise SystemExit(main())
