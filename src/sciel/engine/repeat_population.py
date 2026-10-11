"""Pure multi-member repeat evidence projection; never execute or mutate Programs.

The caller authenticates terminal packets against the durable batch journal.
This component independently verifies receipts/geometry and recomputes decisions.
Ranks are a selection heuristic, not confidence bounds or a publication decision.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, is_dataclass
from fractions import Fraction
import hashlib
import json
import math

from ..evidence.measurement import validate_metric_context
from ..knowledge_base import paired_comparisons as evidence
from .repeat_promotion import bootstrap_decision, paired_dominance

CONTRACT = 'sciel-repeat-population/1'
SELECTION_SCHEMA = 'sciel-repeat-population-selection/1'
REWARD_SCHEMA = 'sciel-repeat-search-event/1'
FLOOR = .1


def _require(ok, message):
    if not ok:
        raise ValueError('repeat population: ' + message)


def _json(value, depth=0):
    _require(depth < 64, 'JSON too deep')
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        _require(math.isfinite(value), 'nonfinite JSON')
    elif type(value) is list:
        for item in value:
            _json(item, depth + 1)
    elif type(value) is dict:
        _require(all(type(k) is str for k in value), 'JSON keys must be strings')
        for item in value.values():
            _json(item, depth + 1)
    else:
        raise ValueError('repeat population: requires literal JSON')


def _encoded(value):
    _json(value)
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _same(a, b):
    return _encoded(a) == _encoded(b)


def _sha(code):
    _require(type(code) is str and bool(code), 'source must be nonempty text')
    return hashlib.sha256(code.encode('utf-8')).hexdigest()


def _text(value):
    return type(value) is str and 0 < len(value) <= 256 and bool(value.strip())


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _program(program):
    value = asdict(program) if is_dataclass(program) else deepcopy(program)
    _require(type(value) is dict, 'Program must be a dataclass or JSON object')
    _encoded(value)
    _require(_text(value.get('id')) and type(value.get('island')) is int, 'invalid Program identity')
    _require(_finite(value.get('score')) and type(value.get('cases')) is dict, 'invalid original observation')
    _sha(value.get('code'))
    return value


def _aggregate(receipts):
    return bootstrap_decision(receipts)['aggregates']


class RepeatPopulation:
    """No filesystem/RNG ownership; all state is detached finite JSON.

    config={num_islands,pop_cap}; identity binds task_metric, adapter_identity,
    limits and optionally the coordinator's complete task_identity. Supplied RNG
    needs random()->float in [0,1); select consumes exactly one value.
    """
    def __init__(self, config, identity):
        _require(type(config) is dict and set(config) == {'num_islands', 'pop_cap'}, 'configuration fields differ')
        _require(type(config['num_islands']) is int and config['num_islands'] >= 1 and
                 type(config['pop_cap']) is int and config['pop_cap'] >= 2, 'requires static islands and capacity >=2')
        _require(type(identity) is dict and {'task_metric','adapter_identity','limits'} <= set(identity)
                 and not set(identity)-{'task_metric','adapter_identity','limits','task_identity'}, 'identity fields differ')
        metric = validate_metric_context(identity['task_metric'])
        _require(metric is not None and metric['task_name'] == 'oe/circle_packing', 'requires bound Circle metric')
        evidence._adapter(identity['adapter_identity'])
        evidence._limits(identity['limits'])
        _encoded(identity)
        self.config = deepcopy(config)
        self.identity = {'contract': CONTRACT, 'config': deepcopy(config), 'bindings': deepcopy(identity),
                         'selection': {'rank': 'exact_certified_minimum_then_raw_minimum', 'uniform_floor': FLOOR,
                                       'ties': 'average_rank', 'victim_ties': 'source_sha256_then_program_id'},
                         'reward': 'binary_terminal_utility_not_physical_quality',
                         'publication': 'bootstrap_only_no_automatic_promotion'}
        self._events = []
        self._external = {}
        self._data = {'history': {}, 'active': {str(i): [] for i in range(config['num_islands'])},
                      'evidence_heads': {}, 'batches': {}, 'applied': {}, 'transitions': [], 'rewards': [],
                      'attempted_sources': [], 'visits': {}, 'selection_counter': 0, 'selection': None,
                      'pending': None, 'stopped_reason': None, 'bootstrapped': False,
                      'publication_source_sha256': None, 'namespace': None, 'task_identity': None}

    @property
    def initialized(self):
        return bool(self._data['batches'])

    @property
    def ready(self):
        return self._data['bootstrapped'] and not self.stopped_reason

    @property
    def pending(self):
        return deepcopy(self._data['pending'])

    @property
    def stopped_reason(self):
        return self._data['stopped_reason']

    @property
    def publication_source_sha256(self):
        return self._data['publication_source_sha256']

    def _check_external(self):
        for ident, program in self._external.items():
            _require(_same(_program(program), self._data['history'][ident]), 'original Program changed: '+ident)

    def _remember(self, program):
        program=_program(program)
        ident=program['id']
        _require(0 <= program['island'] < self.config['num_islands'], 'Program island outside fixed profile')
        _require(ident not in self._data['history'], 'Program ID already recorded')
        self._data['history'][ident] = deepcopy(program)

    def _transact(self, event):
        before=deepcopy(self._data)
        try:
            result=self._reduce(event)
            _encoded(self._data)
        except BaseException:
            self._data=before
            raise
        self._events.append(deepcopy(event))
        return deepcopy(result)

    def _key(self, source):
        return _digest({'source_sha256': source, 'identity': self.identity['bindings']})

    def _packet(self, packet, kind, source, comparator=None, context=None):
        packet=deepcopy(packet); _encoded(packet)
        _require(type(packet) is dict and set(packet)=={'namespace','batch_id','kind','plan','decision','receipts',
            'source_pair','adapter_identity','limits','task_identity','task_metric','decision_anchor'}, 'terminal packet fields differ')
        binding=self.identity['bindings']; task=packet['task_identity']
        _require(_same(packet['task_metric'],binding['task_metric']), 'terminal metric differs')
        _require(_same(packet['adapter_identity'],binding['adapter_identity']) and
                 _same(packet['limits'],binding['limits']), 'terminal runtime/limits differ')
        _require(type(task) is dict and task.get('task')==binding['task_metric']['task_name'] and
                 task.get('task_hash')==binding['task_metric']['task_hash'], 'terminal task differs')
        if 'task_identity' in binding:
            _require(_same(task,binding['task_identity']), 'terminal complete task identity differs')
        if self._data['task_identity'] is not None:
            _require(_same(task,self._data['task_identity']), 'task identity changed between batches')
        _require(_text(packet['namespace']) and (self._data['namespace'] is None or
                 packet['namespace']==self._data['namespace']), 'terminal namespace differs')
        plan=packet['plan']; decision=packet['decision']; batch=packet['batch_id']
        _require(_text(batch) and packet['kind']==kind and plan['batch_id']==batch and plan['kind']==kind,
                 'terminal batch identity differs')
        _require(type(plan['sequence']) is int and plan['sequence']==len(self._data['batches']), 'batch sequence differs')
        fields={'batch_id','sequence','kind','challenger','incumbent','program_id','screening','draws','reserved_wall_seconds'}
        if kind=='comparison':fields|={'context','research_context'}
        _require(set(plan)==fields and plan['screening'] is None and batch==f"b{plan['sequence']:06d}", 'plan fields differ')
        cap=binding['limits']['candidate_wall_seconds']+binding['limits']['certification_wall_seconds']
        _require(_finite(plan['reserved_wall_seconds']) and plan['reserved_wall_seconds']==(3 if kind=='bootstrap' else 6)*cap, 'phase reservation differs')
        _require(plan['challenger']==source and plan['incumbent']==comparator, 'terminal compared sources differ')
        _require(_sha(packet['source_pair']['challenger'])==source and
                 (packet['source_pair']['comparator'] is None if comparator is None else
                  _sha(packet['source_pair']['comparator'])==comparator), 'terminal source bytes differ')
        _require(set(packet['source_pair'])=={'comparator','challenger'}, 'source pair fields differ')
        if context is not None:
            _require(_same(plan.get('context'),context), 'terminal prospective context differs')
            _require(plan['program_id']==self._data['pending']['program_id'], 'terminal Program differs')
            _require(_same(plan['research_context'], {k:context[k] for k in ('branch','origin_experiment_id','source_sha256')}), 'research context differs')
        else:
            _require(plan['program_id'] is None and source==task['canonical_initial_source_sha256'], 'bootstrap source differs')
        expected_roles=['bootstrap']*3 if kind=='bootstrap' else ['challenger','incumbent','incumbent','challenger','challenger','incumbent']
        # Alternation follows sequence+draw_index, as the shared seed contract.
        if kind!='bootstrap' and plan['sequence']%2==0:
            expected_roles=['incumbent','challenger','challenger','incumbent','incumbent','challenger']
        rows=packet['receipts']; draws=plan['draws']
        _require(type(rows) is list and type(draws) is list and len(rows)==len(draws)==len(expected_roles), 'terminal draw count differs')
        _require([d['role'] for d in draws]==expected_roles, 'terminal role order differs')
        values={'bootstrap':[],'incumbent':[],'challenger':[]}; seeds={}; ids=set()
        for position,(draw,row) in enumerate(zip(draws,rows)):
            _require(set(row)==set(draw)|{'receipt'} and _same({k:v for k,v in row.items() if k!='receipt'},draw),
                     'terminal ordered draw differs')
            _require(set(draw)=={'draw_id','role','draw_index','source_sha256','seed'}, 'draw fields differ')
            index=position if kind=='bootstrap' else position//2
            _require(type(draw['draw_index']) is int and draw['draw_index']==index and type(draw['seed']) is int
                     and 0 <= draw['seed'] < 2**32 and _text(draw['draw_id']) and draw['draw_id'] not in ids,
                     'draw index/seed/ID differs')
            ids.add(draw['draw_id']); seeds.setdefault(index,draw['seed'])
            _require(seeds[index]==draw['seed'], 'unpaired seeds')
            target=comparator if draw['role']=='incumbent' else source
            _require(draw['source_sha256']==target, 'draw role source differs')
            evidence._receipt(row['receipt'],source=target,seed=draw['seed'],adapter=packet['adapter_identity'],limits=packet['limits'])
            values[draw['role']].append(row['receipt'])
        _require(len(set(seeds.values()))==3, 'three distinct seeds required')
        calculated=bootstrap_decision(values['bootstrap']) if kind=='bootstrap' else paired_dominance(values['incumbent'],values['challenger'])
        statuses=[{'draw_id':row['draw_id'],'status':row['receipt']['status'] if row['receipt'] is not None else 'not_dispatched'} for row in rows]
        stop=('unknown_dispatch' if any(x['status']=='unknown' for x in statuses) else
              'infrastructure' if any(x['status']=='infrastructure' for x in statuses) else
              'bootstrap_failed' if kind=='bootstrap' and calculated['status']!='confirmed' else
              'comparator_failed' if kind=='comparison' and _aggregate(values['incumbent']) is None else None)
        expected={'decision_id':'d'+batch[1:], 'batch_id':batch,'result':calculated,'draw_statuses':statuses,'stopped_reason':stop}
        _require(_same(decision,expected), 'claimed terminal decision differs from receipts')
        _require(all(row['receipt'] is not None for row in rows) or stop in ('unknown_dispatch','infrastructure'), 'premature terminal packet')
        anchor=packet['decision_anchor']
        _require(type(anchor) is dict and type(anchor.get('entries')) is int and anchor['entries']>0 and
                 type(anchor.get('sha256')) is str and len(anchor['sha256'])==64 and
                 all(c in '0123456789abcdef' for c in anchor['sha256']), 'invalid decision journal anchor')
        return packet,values,calculated,[seeds[i] for i in range(3)]

    def _update_block(self, packet, source, receipts, seeds):
        metrics=_aggregate(receipts)
        if metrics is None:
            return
        block={'source_sha256':source,'evidence_key':self._key(source),'batch_id':packet['batch_id'],
               'decision_id':packet['decision']['decision_id'],'completion_index':len(self._data['batches']),
               'seeds':seeds,'receipt_sha256':[_digest(r) for r in receipts],'metrics':metrics}
        block['id']=_digest(block)
        self._data['evidence_heads'][source]=block

    def ranked(self, island):
        self._check_external()
        _require(type(island) is int and str(island) in self._data['active'], 'unknown island')
        rows=[]
        for ident in self._data['active'][str(island)]:
            p=self._data['history'][ident]; source=_sha(p['code']); head=self._data['evidence_heads'][source]
            raw=head['metrics']['raw']['minimum']; cert=Fraction(**head['metrics']['certified']['minimum'])
            rows.append((cert,raw,source,ident,head))
        rows.sort(key=lambda x:x[:4]); result=[]; n=len(rows)
        for i,row in enumerate(rows):
            ties=[j+1 for j,other in enumerate(rows) if other[:2]==row[:2]]
            rank=sum(ties)/len(ties)
            result.append({'program_id':row[3],'source_sha256':row[2],'evidence_id':row[4]['id'],
                           'rank':rank,'probability':FLOOR/n+(1-FLOOR)*rank/(n*(n+1)/2)})
        return deepcopy(result)

    def active_ids(self, island):
        self._check_external()
        _require(type(island) is int and str(island) in self._data['active'], 'unknown island')
        return list(self._data['active'][str(island)])

    def inspirations(self, island, parent_id, limit=3):
        _require(type(limit) is int and limit>=0, 'invalid inspiration limit')
        return [r['program_id'] for r in reversed(self.ranked(island)) if r['program_id']!=parent_id][:limit]

    def bootstrap(self, programs, packet):
        self._check_external()
        originals=list(programs.values()) if type(programs) is dict else list(programs)
        records=[_program(p) for p in originals]
        result=self._transact({'type':'bootstrap','programs':records,'packet':deepcopy(packet)})
        self._external.update({p['id']:original for p,original in zip(records,originals)})
        return result

    def select(self, island, rng):
        self._check_external()
        _require(self._data['bootstrapped'] and not self.stopped_reason and self._data['selection'] is None,
                 'selection requires live bootstrap and no unresolved attempt')
        _require(type(island) is int and str(island) in self._data['active'], 'unknown island')
        draw=rng.random()
        _require(_finite(draw) and 0<=draw<1, 'invalid parent RNG draw')
        return self._transact({'type':'select','island':island,'draw':float(draw)})

    def stage(self, program, snapshot, attribution=None):
        self._check_external(); record=_program(program)
        result=self._transact({'type':'stage','program':record,'snapshot':deepcopy(snapshot),
                               'attribution':deepcopy(attribution or {})})
        self._external[record['id']]=program
        return result

    def finish_attempt(self, reason):
        self._check_external()
        return self._transact({'type':'finish','reason':reason})

    def stop(self, reason):
        self._check_external()
        return self._transact({'type':'stop','reason':reason})

    def apply(self, packet):
        self._check_external()
        _encoded(packet)
        decision_id=packet['decision']['decision_id']; prior=self._data['applied'].get(decision_id)
        if prior is not None:
            _require(prior['packet_sha256']==_digest(packet), 'decision ID reused with changed content')
            return {**deepcopy(prior['transition']),'already_applied':True}
        return self._transact({'type':'apply','packet':deepcopy(packet)})

    def _reduce(self, event):
        d=self._data; kind=event['type']
        fields={'bootstrap':{'type','programs','packet'}, 'select':{'type','island','draw'},
                'stage':{'type','program','snapshot','attribution'}, 'finish':{'type','reason'},
                'stop':{'type','reason'}, 'apply':{'type','packet'}}
        _require(kind in fields and set(event)==fields[kind], 'history event fields differ')
        if kind=='bootstrap':
            _require(not d['batches'] and not d['history'], 'bootstrap already recorded')
            programs=event['programs']
            _require(len(programs)==self.config['num_islands'] and {p['island'] for p in programs}==set(range(self.config['num_islands'])),
                     'bootstrap requires one member per island')
            sources={_sha(p['code']) for p in programs}; _require(len(sources)==1,'bootstrap sources differ')
            _require(all(p.get('parent_id') is None and _same([p['score'],p['cases']],
                     [programs[0]['score'],programs[0]['cases']]) for p in programs), 'bootstrap ordinary observation is not shared exactly')
            source=next(iter(sources)); packet,values,result,seeds=self._packet(event['packet'],'bootstrap',source)
            for p in programs:self._remember(p)
            d['namespace']=packet['namespace']; d['task_identity']=packet['task_identity']
            d['attempted_sources'].append(source); d['batches'][packet['batch_id']]=packet
            self._update_block(packet,source,values['bootstrap'],seeds)
            if result['status']=='confirmed':
                d['bootstrapped']=True; d['publication_source_sha256']=source
                for p in programs:d['active'][str(p['island'])]=[p['id']]
            else:d['stopped_reason']='bootstrap_incomplete'
            transition={'schema':CONTRACT,'decision_id':packet['decision']['decision_id'],'batch_id':packet['batch_id'],
                        'status':'bootstrap_confirmed' if d['bootstrapped'] else 'bootstrap_incomplete','reward':None}
            d['applied'][transition['decision_id']]={'packet_sha256':_digest(packet),'transition':transition}
            d['transitions'].append(transition)
            return transition
        if kind=='select':
            _require(d['bootstrapped'] and not d['stopped_reason'] and d['selection'] is None, 'selection state differs')
            _require(type(event['draw']) is float and math.isfinite(event['draw']) and 0<=event['draw']<1, 'invalid selection draw')
            rows=self.ranked(event['island']); _require(bool(rows),'empty active island')
            total=0.; parent=rows[-1]
            for row in rows:
                total+=row['probability']
                if event['draw']<total:
                    parent=row; break
            comparator=rows[0]; active=self.active_ids(event['island']); d['selection_counter']+=1
            snapshot={'schema':SELECTION_SCHEMA,'selection_id':f"s{d['selection_counter']:06d}",'island':event['island'],
                'parent_id':parent['program_id'],'parent_source_sha256':parent['source_sha256'],
                'comparator_program_id':comparator['program_id'],'comparator_source_sha256':comparator['source_sha256'],
                'victim_id':comparator['program_id'] if len(active)==self.config['pop_cap'] else None,
                'active_ids':active,'evidence_heads':{row['program_id']:row['evidence_id'] for row in rows},
                'distribution':rows,'draw':event['draw'],'parent_propensity':parent['probability']}
            d['selection']=snapshot; d['visits'][parent['program_id']]=d['visits'].get(parent['program_id'],0)+1
            return snapshot
        if kind=='finish':
            _require(d['selection'] is not None and d['pending'] is None and _text(event['reason']), 'invalid unmeasured attempt completion')
            d['selection']=None
            return {'status':'engineering_attempt','reason':event['reason'],'reward':None}
        if kind=='stop':
            _require(_text(event['reason']) and d['stopped_reason'] is None, 'invalid population stop')
            d['stopped_reason']=event['reason']; return {'status':'stopped','reason':event['reason'],'reward':None}
        if kind=='stage':
            snapshot=event['snapshot']; p=event['program']; attribution=event['attribution']
            _require(d['selection'] is not None and d['pending'] is None and not d['stopped_reason'] and
                     _same(snapshot,d['selection']), 'prospective snapshot changed')
            _require(p['island']==snapshot['island'] and p.get('parent_id')==snapshot['parent_id'], 'candidate parent/island differs')
            _require(type(attribution) is dict, 'attribution must be a JSON object'); _encoded(attribution)
            self._remember(p); source=_sha(p['code'])
            cases=p['cases']; valid=(type(cases.get('validity')) in (int,float) and cases['validity']==1 and
                                     _finite(cases.get('sum_radii')) and cases['sum_radii']>=0)
            duplicate=source in d['attempted_sources']
            if not duplicate:d['attempted_sources'].append(source)
            if duplicate or not valid:
                d['selection']=None
                return {'eligible':False,'reason':'source_duplicate' if duplicate else 'ordinary_invalid','context':None}
            root=snapshot['parent_id']; seen=set()
            while self._data['history'][root].get('parent_id') is not None:
                _require(root not in seen, 'cyclic Program ancestry'); seen.add(root)
                root=self._data['history'][root]['parent_id']
                _require(root in self._data['history'], 'missing Program ancestry')
            _require(attribution.get('branch',root)==root, 'research branch differs from original ancestry')
            origin=attribution.get('origin_experiment_id')
            _require(origin is None or (_text(origin) and len(origin)<=128), 'invalid ordinary origin ID')
            context={'branch':root,'origin_experiment_id':attribution.get('origin_experiment_id'),
                'parent_id':snapshot['parent_id'],'parent_source_sha256':snapshot['parent_source_sha256'],
                'comparator_program_id':snapshot['comparator_program_id'],
                'comparator_source_sha256':snapshot['comparator_source_sha256'],
                'prospective_snapshot':deepcopy(snapshot),'source_sha256':source}
            d['pending']={'program_id':p['id'],'source_sha256':source,'context':context,'attribution':attribution}
            return {'eligible':True,'reason':None,'context':context}
        if kind=='apply':
            pending=d['pending']; _require(pending is not None and not d['stopped_reason'],'terminal has no live pending candidate')
            snapshot=d['selection']; source=pending['source_sha256']; comparator=snapshot['comparator_source_sha256']
            packet,values,result,seeds=self._packet(event['packet'],'comparison',source,comparator,pending['context'])
            ident=packet['decision']['decision_id']; _require(ident not in d['applied'],'duplicate decision')
            d['batches'][packet['batch_id']]=packet
            for src,role in ((comparator,'incumbent'),(source,'challenger')):self._update_block(packet,src,values[role],seeds)
            statuses=[r['status'] if r is not None else 'not_dispatched' for role in ('incumbent','challenger') for r in values[role]]
            baseline_complete=_aggregate(values['incumbent']) is not None
            halted=next((v for v in ('infrastructure','unknown','not_dispatched') if v in statuses),None)
            if halted:d['stopped_reason']='comparison_'+halted
            elif not baseline_complete:d['stopped_reason']='comparator_incomplete'
            active=d['active'][str(snapshot['island'])]; before=list(active)
            success=result['status']=='promote'
            if success:
                victim=snapshot['victim_id']
                if victim is None:active.append(pending['program_id'])
                else:active[active.index(victim)]=pending['program_id']
            status=('replaced' if snapshot['victim_id'] else 'admitted') if success else ('stopped' if d['stopped_reason'] else 'rejected')
            reward=None if d['stopped_reason'] else int(success)
            transition={'schema':CONTRACT,'decision_id':ident,'batch_id':packet['batch_id'],'packet_sha256':_digest(packet),
                'status':status,'reason':d['stopped_reason'] or result['reason'],'island':snapshot['island'],
                'parent_id':snapshot['parent_id'],'comparator_program_id':snapshot['comparator_program_id'],
                'candidate_id':pending['program_id'],'victim_id':snapshot['victim_id'],'before_active_ids':before,
                'after_active_ids':list(active),'result':result,'reward':reward,'attribution':pending['attribution']}
            if reward is not None:
                d['rewards'].append({'schema':REWARD_SCHEMA,'decision_id':ident,'batch_id':packet['batch_id'],
                    'value':reward,'currency':'binary_admission_utility_not_physical_quality','transition':deepcopy(transition)})
            d['applied'][ident]={'packet_sha256':_digest(packet),'transition':transition}; d['transitions'].append(transition)
            d['pending']=d['selection']=None
            return transition
        raise ValueError('repeat population: unknown history event')

    def state(self):
        self._check_external()
        return {'schema':CONTRACT,'identity':deepcopy(self.identity),'events':deepcopy(self._events),
                'projection':deepcopy(self._data)}

    def load_state(self, saved, programs):
        """Replay only pure transitions, re-derive every head/rank/reward, then bind history."""
        _require(type(saved) is dict and set(saved)=={'schema','identity','events','projection'} and
                 saved['schema']==CONTRACT and _same(saved['identity'],self.identity), 'state identity differs')
        _require(type(saved['events']) is list and type(programs) is dict,'invalid saved history')
        fresh=RepeatPopulation(self.config,self.identity['bindings'])
        for event in saved['events']:
            _require(type(event) is dict and type(event.get('type')) is str,'invalid history event')
            fresh._transact(deepcopy(event))
        _require(_same(saved['projection'],fresh._data),'saved projection differs from event replay')
        _require(set(fresh._data['history'])<=set(programs),'saved original Programs missing')
        for ident,value in fresh._data['history'].items():
            _require(_same(_program(programs[ident]),value),'rebound original Program differs: '+ident)
        fresh._external={ident:programs[ident] for ident in fresh._data['history']}
        self._events,self._data,self._external=fresh._events,fresh._data,fresh._external

    def assert_pending(self, plan):
        """Bind an unfinished projection to the saved durable pre-dispatch plan."""
        self._check_external()
        pending=self._data['pending']
        if pending is None:
            if plan is None:
                return True
            _require(not self.initialized and type(plan) is dict and plan.get('kind')=='bootstrap',
                     'durable plan has no pending population candidate')
            task=self.identity['bindings'].get('task_identity')
            if task is not None:
                _require(plan.get('challenger')==task['canonical_initial_source_sha256'], 'bootstrap plan source differs')
            return True
        if plan is None:
            _require(self.stopped_reason is not None, 'live pending candidate lacks durable plan')
            return True
        _require(type(plan) is dict and plan.get('kind')=='comparison' and
                 plan.get('program_id')==pending['program_id'] and plan.get('challenger')==pending['source_sha256']
                 and plan.get('incumbent')==pending['context']['comparator_source_sha256'] and
                 _same(plan.get('context'),pending['context']), 'pending state differs from durable comparison plan')
        return True

    def assert_batches(self, packets):
        """Bind stored projections to the verified durable journal's exact prefix.

        A later terminal tail is allowed for subsequent idempotent application;
        missing, substituted or reordered already-applied packets are rejected.
        """
        self._check_external()
        _require(type(packets) is list, 'terminal packets must be a list')
        stored=list(self._data['batches'].values())
        _require(len(packets)>=len(stored), 'durable batches missing')
        _require(len({p['decision']['decision_id'] for p in packets})==len(packets), 'duplicate durable decision ID')
        for actual, expected in zip(packets,stored):
            _require(_same(actual,expected), 'saved evidence differs from durable terminal batch')
        return True

    def summary(self):
        self._check_external()
        return {'contract':CONTRACT,'initialized':self.initialized,'ready':self.ready,'active':deepcopy(self._data['active']),
                'history_count':len(self._data['history']),'applied_decisions':len(self._data['applied']),
                'pending':self.pending,'stopped_reason':self.stopped_reason,
                'publication_source_sha256':self.publication_source_sha256,
                'rewards':deepcopy(self._data['rewards'])}
