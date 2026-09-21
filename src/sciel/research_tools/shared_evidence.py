"""Read-only cross-island observations from committed immutable cell history.

Publication/adoption never governs knowledge of an executed check. Uncommitted
artifacts, rejected proposals and pending jobs cannot become shared observations.
"""
from copy import deepcopy
import json

from ..methods.models import content_hash
from .decision_projection import project_decision, proposal_decision, authenticate_projection
from .experiment_protocol import parse_protocol, scope_key, result_assessment
from .result_projection import success_context, MIN_CONTEXT_CHARS
from .jobs import SCHEMA as JOB_SCHEMA

SCHEMA='sciel-shared-research-observations/1'
MARKER='SHARED_RESEARCH_OBSERVATIONS_JSON:\n'


def _json(value):
    return json.dumps(value,ensure_ascii=True,sort_keys=True,separators=(',',':'),allow_nan=False)


class EvidenceIndex:
    """Derived cache; checkpoint heads are authoritative, artifacts stay immutable.

    Compact indexes retain every receipt. At most 32 full payloads are cached.
    No directory scan exposes uncommitted evidence. Changed immutable file
    identities fail closed instead of silently trusting an earlier hash check.
    """
    def __init__(self, lifecycle, library):
        from collections import OrderedDict
        self.lifecycle, self.library = lifecycle, library
        self.revisions, self.records, self.decisions, self.experiments = {}, {}, {}, {}
        self.receipt_revisions = {}
        self.scoped_decisions = {}
        self.canonical_evaluations = {}
        self.pending_citations = []
        self.decision_cells = {}
        self.signatures = {}
        self.payloads = OrderedDict()
        self.heads = None
        self.views = {}

    @staticmethod
    def signature(path):
        value=path.lstat()
        return value.st_dev,value.st_ino,value.st_mode,value.st_size,value.st_mtime_ns,value.st_ctime_ns

    def remember(self,path):
        self.signatures[path]=self.signature(path)

    def unchanged(self):
        return all(self.signature(path)==signature for path,signature in self.signatures.items())

    def receipt(self, ident, tool_id):
        if ident in self.payloads:
            self.payloads.move_to_end(ident)
            return self.payloads[ident]
        record=self.lifecycle._receipt(ident,tool_id,self.library)
        # Authentication uses complete inputs; the cache never retains source,
        # evaluator blobs or nested predecessor state merely to show a result.
        record={**record,'inputs':{'task':deepcopy(record['inputs']['task'])}}
        self.remember(self.library.path/'artifacts'/(ident+'.json'))
        self.payloads[ident]=record
        while len(self.payloads)>32:self.payloads.popitem(last=False)
        return record

    def revision(self, head):
        revision=self.lifecycle.cells._read(self.lifecycle.cells.revisions,head)
        self.remember(self.lifecycle.cells.revisions/(head+'.json'))
        return revision

    def proposal(self, ident):
        value=self.lifecycle._artifact(ident,'proposal',self.library)
        self.remember(self.library.path/'artifacts'/(ident+'.json'))
        return value

    def add_revision(self, cell, head, revision, known_chain):
        invocation_refs=[r for r in revision['evidence'] if r['kind']=='tool_invocation']
        for evidence in invocation_refs:
            ident=evidence['id']
            if ident in self.records:continue
            record=self.receipt(ident,evidence['tool_id'])
            if record['cell_id']!=cell['id'] or record['cell_revision_id'] not in known_chain:
                raise ValueError('shared observation is not bound to a committed cell revision')
            meta={key:deepcopy(value) for key,value in record.items() if key not in ('inputs','receipt')}
            meta['task']=deepcopy(record['inputs']['task'])
            meta['receipt']={key:deepcopy(value) for key,value in record['receipt'].items() if key not in ('output','stderr','diagnostic')}
            self.records[ident]=meta
            self.receipt_revisions[ident] = head
        for evidence in revision['evidence']:
            if evidence['kind'] != 'candidate_evaluation' or evidence['id'] in self.canonical_evaluations:
                continue
            from .canonical_evidence import authenticate, summary
            ident = evidence['id']
            saved, invocation = authenticate(self.lifecycle, self.library, ident,
                cell_id=cell['id'], run_id=self.lifecycle.run_id)
            if saved['provenance']['receipt_id'] not in self.records:
                raise ValueError('canonical evidence lacks committed producing invocation')
            if invocation['cell_revision_id'] not in known_chain:
                raise ValueError('canonical evidence belongs to another cell history')
            self.remember(self.library.path/'artifacts'/(ident+'.json'))
            self.canonical_evaluations[ident] = summary(ident, saved, invocation)
        for evidence in revision['evidence']:
            if evidence['kind'] not in ('research_decision','research_experiment'):continue
            ident=evidence['id'];proposal=self.proposal(ident)
            if proposal.get('cell_id')!=cell['id'] or proposal.get('cell_revision_id') not in known_chain:
                raise ValueError('shared experiment/decision lacks committed cell provenance')
            if evidence['kind']=='research_decision':
                self.decision_cells[ident] = cell['id']
                decision=project_decision(proposal,ident)
                raw_question=proposal_decision(proposal)['question']
                raw=proposal['response']['text'].strip()
                if raw.startswith('```') and raw.endswith('```'):
                    raw=raw.split('\n',1)[-1].rsplit('```',1)[0].strip()
                action=json.loads(raw)
                declared=action.get('experimental_protocol')
                declared=parse_protocol(declared) if declared is not None else None
                # Other cells may be traversed later during a cold rebuild.
                # Retain only bounded citation metadata, not full prompts.
                self.pending_citations.append({'canonical_evaluation_ids':proposal.get('canonical_evaluation_ids', []),
                    'response':{'text':json.dumps({'experimental_protocol':{'prior_receipt_ids':
                        declared['prior_receipt_ids'] if declared else []}})}})
                source=proposal['target_binding']['target_source_sha256']
                decision_keys={scope_key(source,raw_question)}
                if declared:decision_keys.add(scope_key(source,declared['question_key']))
                for receipt in proposal['evidence_receipt_ids']:
                    record=self.records.get(receipt)
                    if record is None or record['cell_id']!=cell['id']:
                        raise ValueError('shared conclusion cites an uncommitted or foreign-cell observation')
                    rank=(revision['ordinal'],ident)
                    if receipt not in self.decisions or rank>self.decisions[receipt][0]:
                        self.decisions[receipt]=(rank,decision)
                    prior_protocol=self.experiments.get(receipt,(None,None))[1]
                    record_source=record['target_binding']['target_source_sha256']
                    record_keys={scope_key(record_source,record['question'])}
                    if prior_protocol:record_keys.add(scope_key(record_source,prior_protocol['question_key']))
                    if (record_keys.intersection(decision_keys) and
                            (receipt not in self.scoped_decisions or rank>self.scoped_decisions[receipt][0])):
                        self.scoped_decisions[receipt]=(rank,decision)
            else:
                raw=proposal['response']['text'].strip()
                if raw.startswith('```') and raw.endswith('```'):
                    raw=raw.split('\n',1)[-1].rsplit('```',1)[0].strip()
                action=json.loads(raw);protocol=parse_protocol(action.get('experimental_protocol'))
                action_id,attempt=proposal['action_id'],proposal['attempt']
                parent_job=action.get('job_id') if action.get('action')=='continue' else None
                matched=False
                # This revision is bounded to at most 256 references. Bind the
                # proposal directly to its invocation; never scan all history.
                for ref in invocation_refs:
                    record=self.records.get(ref['id'])
                    if record is None:continue
                    tool_id=ref['tool_id']
                    foreground=content_hash({'action_id':action_id,'attempt':attempt,'tool_id':tool_id})
                    job_key=content_hash({'action_id':action_id,'attempt':attempt,'tool_id':tool_id,'parent_job_id':parent_job})
                    background=content_hash({'schema':JOB_SCHEMA,'job_key':job_key})
                    if record['action_id']==action_id and record['receipt']['invocation_id'] in (foreground,background):
                        self.experiments[ref['id']]=(ident,protocol);matched=True
                if not matched:
                    raise ValueError('committed experiment proposal lacks its exact saved invocation')
        self.revisions[head]={'previous_revision':revision['previous_revision'],'cell_id':cell['id']}

    def update(self):
        cells=self.lifecycle.cells.all_cells()
        heads=tuple(sorted((cell['id'],cell['head_revision']) for cell in cells))
        if not self.unchanged():
            raise ValueError('immutable shared evidence file changed after authentication')
        if heads==self.heads:return
        for cell in cells:
            head=cell['head_revision'];added=[];known_chain=set()
            cursor=head
            while cursor is not None:
                known_chain.add(cursor)
                if cursor in self.revisions:
                    cursor=self.revisions[cursor]['previous_revision']
                else:
                    revision=self.revision(cursor);added.append((cursor,revision));cursor=revision['previous_revision']
            for ident,revision in reversed(added):
                self.add_revision(cell,ident,revision,known_chain)
        from .canonical_evidence import authenticate_citations
        for proposal in self.pending_citations:
            authenticate_citations(proposal, self.canonical_evaluations)
        self.pending_citations.clear()
        self.heads=heads;self.views.clear()

    def view(self,island_id,task_identity):
        self.update()
        key=content_hash(task_identity)
        if key not in self.views:
            records=[(ident,record) for ident,record in self.records.items()
                if record['task']==task_identity and record['receipt']['dispatch_status']=='invoked']
            records.sort(key=lambda item:(item[1]['iteration'],item[0]),reverse=True)
            groups={};index=[]
            for ident,record in records:
                groups.setdefault(record['island_id'],[]).append((ident,record))
                protocol=self.experiments.get(ident,(None,None))[1]
                question_key=protocol['question_key'] if protocol else record['question']
                keys=sorted({scope_key(record['target_binding']['target_source_sha256'],record['question']),
                    scope_key(record['target_binding']['target_source_sha256'],question_key)})
                decision=self.scoped_decisions.get(ident,(None,None))[1]
                index.append({'receipt_id':ident,'scope_keys':keys,'status':record['result']['status'],
                    'conclusion':decision['conclusion'] if decision else None})
            selected=[]
            while groups and len(selected)<32:
                for island in list(groups):
                    if len(selected)==32:break
                    selected.append(groups[island].pop(0))
                    if not groups[island]:del groups[island]
            entries=[]
            for ident,record in selected:
                full=self.receipt(ident,record['tool_id']);receipt=record['receipt'];binding=record['target_binding']
                proposal_id,protocol=self.experiments.get(ident,(None,None))
                output=full['receipt'].get('output');observations=output.get('observations') if type(output) is dict else None
                method=self.lifecycle._state['methods'].get(record['tool_id'])
                entries.append({'receipt_id':ident,'invocation_id':receipt['invocation_id'],'tool_id':record['tool_id'],
                    'tool_source_sha256':receipt['source_sha256'],'input_sha256':receipt['input_sha256'],
                    'output_sha256':receipt.get('output_sha256'),'task':deepcopy(task_identity),
                    'cell_id':record['cell_id'],'cell_revision_id':record['cell_revision_id'],'island_id':record['island_id'],
                    'candidate_id':binding['candidate_id'],'target_source_sha256':binding['target_source_sha256'],
                    'question':record['question'],'question_key':protocol['question_key'] if protocol else record['question'],
                    'status':record['result']['status'],'iteration':record['iteration'],
                    'resources':deepcopy(receipt['resources']),'requested_limits':deepcopy(receipt['limits']),
                    'publication':method['stage'] if method else 'unpublished',
                    'measurement':{'output':{'observations':deepcopy(observations)},'error':record['result'].get('error')},
                    'decision':deepcopy(self.scoped_decisions.get(ident,self.decisions.get(ident,(None,None)))[1]),
                    'decision_scope_matches_check':ident in self.scoped_decisions,'experiment_proposal_id':proposal_id,
                    'comparison_report':result_assessment(observations,protocol),
                    'rights':'Observation access only; foreign tool execution still requires replay/adoption.'})
            self.views[key]={'schema':SCHEMA,'total_checks':len(records),'source_islands':sorted({r['island_id'] for _,r in records}),
                'entries':entries,'entries_omitted':len(records)-len(entries),'check_index':sorted(index,key=lambda x:x['receipt_id']),
                'scope':'Committed receipt-bound observations; interpretations and tool-reported comparisons are separate. No adoption or benefit is implied.'}
            canonical = [deepcopy(row) for row in self.canonical_evaluations.values()
                         if row['task'] == task_identity]
            canonical.sort(key=lambda row:(row['iteration'],row['artifact_id']))
            self.views[key].update(canonical_evaluation_index=[{'artifact_id':row['artifact_id'],
                'kind':'candidate_evaluation','status':row['status']} for row in canonical],
                canonical_evaluations=canonical[-32:], canonical_evaluations_total=len(canonical))
        return {**deepcopy(self.views[key]),'recipient_island_id':island_id}


def collect(lifecycle, library, island_id, task_identity):
    if lifecycle._shared_evidence_index is None:
        lifecycle._shared_evidence_index=EvidenceIndex(lifecycle,library)
    return lifecycle._shared_evidence_index.view(island_id,task_identity)


def interpretation_evidence(lifecycle, library, card, *, episode=None):
    """Authenticate evidence behind a terminal-only card, without inventing a check.

    A finish can cite earlier measurements even when this episode invoked no
    tool. Use the complete saved citation set, not its bounded card projection.
    An unmeasured interpretation stays in its own cell/history, not shared
    experimental memory. Existing receipt-bound cards use their normal path.
    """
    decision = card.get('decision')
    if decision is None:
        return {'supported': False, 'records': [], 'proposal': None}
    if lifecycle._shared_evidence_index is None:
        lifecycle._shared_evidence_index = EvidenceIndex(lifecycle, library)
    index = lifecycle._shared_evidence_index
    index.update()
    ident, cell_id = decision['proposal_id'], card['cell_id']
    proposal = index.proposal(ident)
    authenticate_projection(decision, proposal, ident)
    revision = index.revisions.get(proposal.get('cell_revision_id'), {})
    fresh = (episode is not None and episode.get('terminal_decision_proposal_id') == ident
             and episode['plan']['cell_id'] == cell_id
             and episode['plan']['action_id'] == proposal.get('action_id')
             and card in episode.get('cards', []))
    if (proposal.get('cell_id') != cell_id or revision.get('cell_id') != cell_id
            or (index.decision_cells.get(ident) != cell_id and not fresh)):
        raise ValueError('terminal memory lacks committed or current-episode provenance')
    if ({key:card['source'].get(key) for key in proposal['target_binding']}
            != proposal['target_binding']):
        raise ValueError('terminal memory source differs from its saved proposal')
    if card['source']['receipt_id'] is None:
        from .decision_projection import NO_MEASUREMENT_QUESTION
        if (any(card['source'][key] is not None for key in ('tool_id', 'invocation_id'))
                or card['question'] != NO_MEASUREMENT_QUESTION
                or card['observations'] != {'status':'not_invoked_this_episode', 'output_excerpt':'', 'resources':{}}):
            raise ValueError('terminal-only memory invented a tool observation')
    evidence = False
    records = {}
    current = {row['receipt_id']:row for row in episode.get('invocations', [])} if fresh else {}
    for receipt_id in proposal['evidence_receipt_ids']:
        record = index.records.get(receipt_id)
        if record is None:
            row = current.get(receipt_id)
            if row is None:
                raise ValueError('terminal memory cites an uncommitted observation')
            record = lifecycle._receipt(receipt_id, row['tool_id'], library)
            if record['action_id'] != episode['plan']['action_id']:
                raise ValueError('terminal memory observation belongs to another episode')
        if record['cell_id'] != cell_id:
            raise ValueError('terminal memory cites another cell as its own invocation')
        evidence |= record['receipt']['dispatch_status'] == 'invoked'
        records[receipt_id] = record
    from .canonical_evidence import authenticate, authenticate_citations
    canonical = dict(index.canonical_evaluations)
    if fresh:
        for step in (episode.get('research_state') or {}).get('steps', []):
            canonical_id = step.get('canonical_evaluation_id')
            if canonical_id is None or canonical_id in canonical:
                continue
            saved, _ = authenticate(lifecycle, library, canonical_id,
                                   cell_id=cell_id, run_id=lifecycle.run_id)
            if saved['provenance']['action_id'] != episode['plan']['action_id']:
                raise ValueError('terminal memory canonical result belongs to another episode')
            canonical[canonical_id] = saved
    citations = authenticate_citations(proposal, canonical)
    for canonical_id in citations:
        saved = canonical[canonical_id]
        receipt_id = saved.get('tool_receipt_id') or (saved.get('provenance') or {}).get('receipt_id')
        if receipt_id is None:
            continue  # Typed canonical evidence is not itself a tool invocation.
        record = index.records.get(receipt_id)
        if record is None:
            row = current.get(receipt_id)
            if row is None:
                raise ValueError('canonical memory lacks its authenticated producing observation')
            record = lifecycle._receipt(receipt_id, row['tool_id'], library)
            if record['action_id'] != episode['plan']['action_id']:
                raise ValueError('canonical memory observation belongs to another episode')
        records[receipt_id] = record
    raw = proposal['response']['text'].strip()
    if raw.startswith('```') and raw.endswith('```'):
        raw = raw.split('\n', 1)[-1].rsplit('```', 1)[0].strip()
    declared = json.loads(raw).get('experimental_protocol')
    if declared is not None:
        protocol = parse_protocol(declared)
        for receipt_id in protocol['prior_receipt_ids']:
            if receipt_id in proposal['evidence_receipt_ids'] or receipt_id in citations:
                continue
            # A closure may interpret another cell's committed measured check;
            # that citation grants neither tool input nor execution rights.
            record = index.records.get(receipt_id)
            if record is None or record['receipt']['dispatch_status'] != 'invoked':
                raise ValueError('terminal memory cites an unavailable shared observation')
            evidence = True
            records[receipt_id] = record
    return {'supported': evidence or bool(citations), 'records': list(records.items()),
            'proposal': proposal, 'index': index}


def interpretation_has_evidence(lifecycle, library, card, *, episode=None):
    return interpretation_evidence(lifecycle, library, card, episode=episode)['supported']


def prompt(view,budget):
    """Whole JSON with exact provenance and whole atomic observation fields."""
    if view is None:return ''
    value={'schema':SCHEMA,'total_checks':view['total_checks'],'source_islands':view['source_islands'][:16],
        'source_islands_omitted':max(0,len(view['source_islands'])-16),
        'entries':[],'omitted_count':view['total_checks'],
        'scope':'Receipt-bound observations; model interpretation is separate; no tool execution rights or benefit implied.'}
    canonical = view.get('canonical_evaluations', [])
    if canonical:
        value.update(canonical_evaluations=[],
                     canonical_evaluations_omitted=view.get('canonical_evaluations_total', len(canonical)))
    # The newest canonical result is an actionable typed reference, not an
    # optional prose excerpt. Include it in the minimum reservation so large
    # tool output cannot silently starve a just-measured candidate result.
    if canonical:
        value['canonical_evaluations'] = [deepcopy(canonical[-1])]
        value['canonical_evaluations_omitted'] -= 1
    def render():return MARKER+_json(value)+'\n'
    if budget<=len(render()):return render()
    for entry in view['entries']:
        value['entries'].append(deepcopy(entry));value['omitted_count']-=1
        if len(render())<=budget:continue
        value['entries'].pop();value['omitted_count']+=1
        base={key:deepcopy(entry[key]) for key in ('receipt_id','tool_id','cell_id','island_id','candidate_id',
            'target_source_sha256','task','question','status','resources','publication','decision','decision_scope_matches_check')}
        base['detail_omitted']=True
        # Keep decision identity/conclusion before optional prose. Never synthesize a summary.
        if base['decision'] is not None:
            base['decision']={key:base['decision'][key] for key in ('proposal_id','conclusion','scope')}
            base['decision']['text_omitted']=True
        base['measurement']={'omitted':True}
        value['entries'].append(base);value['omitted_count']-=1
        remaining=budget-len(render())
        if remaining>=MIN_CONTEXT_CHARS:
            measured=success_context(entry['measurement'],remaining+len(_json(base['measurement'])))
            base['measurement']=json.loads(measured)
        if len(render())>budget:
            value['entries'].pop();value['omitted_count']+=1
    for entry in reversed(canonical[:-1]):
        value['canonical_evaluations'].append(deepcopy(entry))
        value['canonical_evaluations_omitted'] -= 1
        if len(render()) > budget:
            value['canonical_evaluations'].pop()
            value['canonical_evaluations_omitted'] += 1
    return render()


def delivery(prompt_text):
    if MARKER not in prompt_text:return None
    data=json.JSONDecoder().raw_decode(prompt_text.split(MARKER,1)[1])[0]
    return {'evidence_schema':SCHEMA,'total_checks':data['total_checks'],'source_islands':data['source_islands'],
        'delivered':[{'receipt_id':row['receipt_id'],'source_cell_id':row['cell_id'],
            'source_island_id':row['island_id'],'source_candidate_id':row['candidate_id'],
            'decision_proposal_id':(row.get('decision') or {}).get('proposal_id'),
            'detail_omitted':row.get('detail_omitted',False),
            'measurement_omitted':(row.get('measurement',{}).get('omitted',False)
                or row.get('measurement',{}).get('observations_state')=='all_omitted'),
            'measurement_projection':row.get('measurement',{}).get('observations_state')} for row in data['entries']],
        'omitted_count':data['omitted_count'],'scope':'Delivered to this prompt; actual use or adoption is not established.'}
