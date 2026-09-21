"""Funding lineage, trial blocks and laboratory lifecycle transitions."""
from __future__ import annotations

from .budget import FundingPolicy
from .coordinator_contracts import Refusal
from .models import content_hash, detached


def _funding(coordinator):
    rows = [e['data'] for e in coordinator.registry.events() if e['kind'] in ('funding_state','trial_block')]
    return FundingPolicy.from_state(rows[-1]['state']) if rows else FundingPolicy()


def _save_funding(coordinator, policy):
    state = policy.state()
    if not coordinator._events('funding_state') or coordinator._events('funding_state')[-1]['state'] != state:
        coordinator.registry.append('funding_state', dict(state=state))


def lineage_for_version(coordinator, method_id, *, policy=None):
    """The immutable funding binding; a new hash is not a grant."""
    coordinator.registry.get_method(method_id)
    policy = policy or coordinator._funding()
    rows = [row for row in policy.state()['events']
            if row['op']=='bind_version' and row['version_id']==method_id]
    return rows[0]['lineage_id'] if rows else None


def _active_funding_lineage(coordinator, lab_id, revision=None):
    revision = revision or coordinator.registry.revision(coordinator.registry.lab(lab_id)['active_revision'])
    lineage = revision['capabilities'].get('active_funding_lineage', 'lab:'+lab_id)
    if lineage != 'lab:'+lab_id:
        grants = [row for row in coordinator._events('funding_direction_granted')
                  if row['lab_id']==lab_id and row['lineage_id']==lineage]
        if not grants:
            raise Refusal('active funding direction has no grant for this laboratory')
    return lineage


def _plan_funding_lineage(coordinator, lab_id, revision, treatment):
    policy = coordinator._funding()
    lineages = {coordinator.lineage_for_version(ident, policy=policy) for ident in treatment
                if coordinator.registry.get_method(ident)['manifest']['author_lab']==lab_id}
    lineages.discard(None)
    # A composition remains legal. Its whole experiment spends the active
    # explicit/default grant when own components have different bindings;
    # this accounting choice confers no scientific credit on a component.
    return next(iter(lineages)) if len(lineages)==1 else coordinator._active_funding_lineage(lab_id, revision)


def grant_direction(coordinator, lab_id, question, evidence_refs):
    """Trusted host grant of a bounded trial direction, never new money.

    Model proposal actions do not expose this transition. Identical question
    and evidence identify the same grant, even after another direction has
    become active. Existing child versions retain their parent lineage.
    """
    if not isinstance(question,str) or not question.strip() or len(question)>4096:
        raise Refusal('a direction grant requires a bounded nonempty question')
    if (not isinstance(evidence_refs,(list,tuple)) or not 1<=len(evidence_refs)<=64
            or any(not isinstance(ref,str) for ref in evidence_refs)
            or len(set(evidence_refs))!=len(evidence_refs)):
        raise Refusal('a direction grant requires distinct recorded evidence references')
    request = dict(lab_id=lab_id,question=question.strip(),evidence_refs=sorted(evidence_refs))
    grant_key = content_hash(request)
    lineage = 'grant:'+grant_key
    with coordinator._locked():
        def grant():
            if coordinator.lab_state(lab_id)['status']=='archived':
                raise Refusal('archived labs cannot receive new direction grants')
            registered = {e['data']['id'] for e in coordinator.registry.events()
                          if e['kind']=='artifact_registered'}
            if not set(request['evidence_refs']) <= registered:
                raise Refusal('direction grant requires existing evidence')
            old_id, old = coordinator._revision(lab_id)
            policy = coordinator._funding()
            policy.create_lineage(lineage)
            coordinator._save_funding(policy)
            ident = coordinator.registry.put_artifact('funding_grant',dict(
                schema='sciel-method-direction-grant/1',**request,lineage_id=lineage,
                budget_account=coordinator._account(lab_id),initial_quota=policy.initial_quota,
                previous_revision=old_id,resource_grant='none; existing shared account limits apply'))
            changed = coordinator.registry.revise_lab(lab_id,old_id,dict(
                capabilities={**old['capabilities'],'active_funding_lineage':lineage},
                evidence_refs=list(dict.fromkeys(old['evidence_refs']+[ident]))),
                'funding_direction:'+ident)
            coordinator.registry.append('funding_direction_granted',dict(lab_id=lab_id,
                grant_id=ident,lineage_id=lineage,previous_revision=old_id,new_revision=changed))
            return ident
        return coordinator._host('direction_grant:'+grant_key,lab_id,'funding_direction',request,grant)


def _finish_trial_block(coordinator, lab_id, lineage=None):
    """A conservative operational novelty signal; not a scientific utility score."""
    lineage = lineage or coordinator._active_funding_lineage(lab_id)
    consumed={pid for row in coordinator._events('trial_block') if row['lab_id']==lab_id
              and row.get('lineage_id','lab:'+row['lab_id'])==lineage for pid in row['plan_ids']}
    reserved=[r['plan_id'] for r in coordinator._events('trial_reserved') if r['lab_id']==lab_id
              and r.get('lineage_id','lab:'+r['lab_id'])==lineage and r['plan_id'] not in consumed]
    finished={r['plan_id']:r['experiment_id'] for r in coordinator._events('experiment_finished')}
    policy=coordinator._funding()
    policy.create_lineage(lineage)
    quota=policy.snapshot(lineage)['next_quota']
    if not quota or len(reserved)<quota or any(p not in finished for p in reserved[:quota]):
        return
    plans=reserved[:quota]
    def signature(pid):
        exp=coordinator.registry.get_artifact('experiment',finished[pid])
        observations={arm:[dict(input_sha256=r['input_sha256'],status=r['status'],measurement=r['measurement'])
                            for r in rows] for arm,rows in exp['results'].items()}
        return content_hash(dict(results=observations,status=exp['status'],
            failure_reasons=[d['reason'] for d in exp['diagnostics']]))
    old={signature(pid) for pid in consumed if pid in finished}
    informative=any(signature(pid) not in old for pid in plans)
    refs=[finished[pid] for pid in plans]
    block=content_hash(dict(lab_id=lab_id,lineage_id=lineage,plans=plans))
    policy.record_block(lineage,block,informative=informative,evidence_refs=refs)
    # One atomic journal event covers consumption and new funding state.
    coordinator.registry.append('trial_block',dict(lab_id=lab_id,lineage_id=lineage,plan_ids=plans,trials=len(plans),
        block_id=block,informative=informative,evidence_refs=refs,state=policy.state(),
        assessment='new assigned input/output/failure observation; no claim of general utility'))


def lab_state(coordinator, lab_id):
    lab=coordinator.registry.lab(lab_id)
    rows=[r for r in coordinator._events('lab_lifecycle') if r['lab_id']==lab_id]
    if rows:
        lab['status']=rows[-1]['status']
    funding=coordinator._funding().snapshot().get(coordinator._active_funding_lineage(lab_id))
    if funding and lab['status']!='archived':
        lab['status']=funding['status']
    return lab


def archive_lab(coordinator, lab_id, *, reason):
    if not isinstance(reason,str) or not reason.strip():
        raise Refusal('archiving requires a reason')
    with coordinator._locked():
        coordinator.registry.lab(lab_id)
        coordinator.registry.append('lab_lifecycle',dict(lab_id=lab_id,status='archived',reason=reason))


def renew_trial(coordinator, lab_id, quota, *, question, evidence_refs, lineage_id=None):
    with coordinator._locked():
        if coordinator.lab_state(lab_id)['status']=='archived':
            raise Refusal('archived labs cannot resume; create an explicitly funded fork')
        for ident in evidence_refs:
            if not any(e['kind']=='artifact_registered' and e['data']['id']==ident for e in coordinator.registry.events()):
                raise Refusal('renewal requires existing evidence')
        active=coordinator._active_funding_lineage(lab_id)
        lineage=lineage_id or active
        owned=(lineage=='lab:'+lab_id or any(row['lab_id']==lab_id and row['lineage_id']==lineage
            for row in coordinator._events('funding_direction_granted')))
        if not owned:
            raise Refusal('renewal can only fund this laboratory\'s existing lineage')
        if lineage!=active:
            done=[row for row in coordinator._events('episode_step') if row['lab_id']==lab_id]
            refusal_id=done[-1].get('refusal_id') if done else None
            refusal=coordinator.registry.get_artifact('proposal_receipt',refusal_id) if refusal_id else {}
            if refusal.get('funding_lineage')!=lineage or refusal_id not in evidence_refs:
                raise Refusal('nonactive lineage renewal requires the latest recorded quota refusal as evidence')
        policy=coordinator._funding()
        policy.authorize_trial(lineage,quota,reason=question,evidence_refs=evidence_refs)
        coordinator._save_funding(policy)


def reopen_episode(coordinator, lab_id, *, question, evidence_refs):
    """Explicit host authorization of another episode, retaining all history.

    This allocates no money and does not retry an earlier provider call. The
    new direction grant is evidence-bound; global proposal indexes and all
    previously settled receipts remain immutable. It is not a model action.
    """
    with coordinator._locked():
        if coordinator.lab_state(lab_id)['status']=='archived':
            raise Refusal('archived labs require a separately authorized fork')
        done=[r for r in coordinator._events('episode_step') if r['lab_id']==lab_id]
        if not done or done[-1]['status']!='stopped':
            raise Refusal('episode reopening requires its last recorded stopped step')
        stopped=done[-1]
        existing=[r for r in coordinator._events('episode_reopened') if r['lab_id']==lab_id
            and r['stopped_proposal_id']==stopped['proposal_id']]
        if existing:
            if existing[-1]['question']!=question.strip() or existing[-1]['evidence_refs']!=sorted(evidence_refs):
                raise Refusal('stopped episode already has a different explicit reopening')
            return detached(existing[-1])
        grant_id=coordinator.grant_direction(lab_id,question,evidence_refs)
        record=dict(lab_id=lab_id,stopped_index=stopped['index'],stopped_proposal_id=stopped['proposal_id'],
            next_index=len(done),question=question.strip(),evidence_refs=sorted(evidence_refs),
            grant_id=grant_id,new_revision=coordinator.registry.lab(lab_id)['active_revision'],
            budget_account=coordinator._account(lab_id),resource_grant='none; shared account is unchanged')
        coordinator.registry.append('episode_reopened',record)
        return detached(record)
