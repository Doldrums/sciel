"""Lossless evaluator failure evidence and bounded, source-addressed excerpts.

These are reported diagnostics, not a judgement about a scientific hypothesis.
Full strings remain in the canonical artifact; projections are always derived.
"""
from copy import deepcopy
import hashlib
import json

from ..methods.models import canonical_bytes, content_hash, require_hash

SCHEMA = 'sciel-canonical-failure-diagnostics/1'
DIAGNOSTIC_KEYS = frozenset(('stderr', 'error', 'errors', 'exception', 'exceptions',
    'traceback', 'message', 'diagnostic', 'diagnostics', 'failure', 'reason'))


def _sha(text):
    return hashlib.sha256(text.encode('utf-8', errors='surrogatepass')).hexdigest()


def capture(result, *, source_sha256, evaluation_id, record_reference=None):
    """Keep the original strings, including JSON encoded inside an artifact."""
    evidence = {'schema': SCHEMA, 'source_sha256': source_sha256,
        'evaluation_id': evaluation_id, 'stderr': result.stderr,
        'stderr_sha256': _sha(result.stderr), 'artifacts': deepcopy(result.artifacts or {}),
        'artifacts_sha256': content_hash(result.artifacts or {}),
        'record_reference': deepcopy(record_reference)}
    validate(evidence, source_sha256=source_sha256, evaluation_id=evaluation_id,
             artifacts_sha256=evidence['artifacts_sha256'])
    return evidence


def validate(value, *, source_sha256, evaluation_id, artifacts_sha256, library=None,
             run_id=None, task_hash=None):
    fields = {'schema','source_sha256','evaluation_id','stderr','stderr_sha256',
              'artifacts','artifacts_sha256','record_reference'}
    if (type(value) is not dict or set(value) != fields or value['schema'] != SCHEMA
            or value['source_sha256'] != source_sha256 or value['evaluation_id'] != evaluation_id
            or type(value['stderr']) is not str or type(value['artifacts']) is not dict
            or value['stderr_sha256'] != _sha(value['stderr'])
            or value['artifacts_sha256'] != artifacts_sha256
            or content_hash(value['artifacts']) != artifacts_sha256):
        raise ValueError('canonical diagnostics differ from evaluation/source evidence')
    require_hash(source_sha256, 'canonical diagnostic source')
    reference = value['record_reference']
    # An Engine evaluation has a second immutable identity index. A rehashed
    # canonical wrapper must not remove its reference to evade that authority.
    if library is not None and evaluation_id is not None:
        directory = library.run_dir / 'research_evidence' / 'evaluations'
        index_path = directory / ('evaluation-' + _sha(evaluation_id) + '.json')
        if index_path.exists() or index_path.is_symlink():
            from ..engine.research_artifacts import INDEX_SCHEMA
            if (directory.is_symlink() or directory.parent.is_symlink()
                    or index_path.is_symlink() or index_path.stat().st_size > 4096):
                raise ValueError('invalid canonical diagnostic evaluation index')
            index = json.loads(index_path.read_bytes())
            expected = {'schema': INDEX_SCHEMA, 'run_id': run_id, **(reference or {})}
            if reference is None or index != expected:
                raise ValueError('canonical diagnostics saved-evaluation binding differs')
        elif reference is not None:
            raise ValueError('canonical diagnostic evaluator index is unavailable')
    if reference is not None:
        from ..engine.research_artifacts import MAX_RECORD_BYTES
        if (type(reference) is not dict or set(reference) != {
                'sha256','bytes','evaluation_id','source_sha256','task_hash'}
                or reference['source_sha256'] != source_sha256
                or reference['evaluation_id'] != evaluation_id
                or (task_hash is not None and reference['task_hash'] != task_hash)
                or type(reference['bytes']) is not int or not 0 < reference['bytes'] <= MAX_RECORD_BYTES):
            raise ValueError('canonical diagnostics saved-evaluation binding differs')
        require_hash(reference['sha256'], 'canonical diagnostic evaluation record')
        if library is not None:
            path = library.run_dir / 'research_evidence' / 'evaluations' / (reference['sha256']+'.json')
            if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_RECORD_BYTES:
                raise ValueError('canonical diagnostic evaluator record is unavailable')
            record = json.loads(path.read_bytes())
            if (content_hash(record) != reference['sha256']
                    or len(canonical_bytes(record)) != reference['bytes']
                    or any(record.get(key) != reference[key] for key in
                           ('source_sha256','evaluation_id','task_hash'))
                    or (run_id is not None and record.get('run_id') != run_id)
                    or record['result']['stderr'] != value['stderr']
                    or record['result']['artifacts'] != value['artifacts']):
                raise ValueError('canonical diagnostics differ from saved evaluator record')
    return value


def _entries(value):
    """Bounded JSON inspection; ::json marks decoding an original string."""
    entries, uninspected = [], 0
    stack = [(value['artifacts'], '/artifacts', False, 0)]
    if value['stderr']:
        entries.append(('/stderr', value['stderr']))
    nodes = 0
    while stack:
        current, path, diagnostic, depth = stack.pop()
        nodes += 1
        if nodes > 16384 or depth > 32:
            uninspected += 1 + len(stack)
            break
        if isinstance(current, str):
            if diagnostic and current.strip():
                entries.append((path, current))
            if current.lstrip().startswith(('{','[')):
                try:
                    decoded = json.loads(current, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
                except (ValueError, RecursionError):
                    uninspected += 1
                else:
                    stack.append((decoded, path+'::json', diagnostic, depth+1))
        elif type(current) is dict:
            items = sorted(current.items(), key=lambda item: (
                str(item[0]).lower() not in DIAGNOSTIC_KEYS,
                not (isinstance(item[1], str) and item[1].lstrip().startswith(('{','['))),
                str(item[0])))
            room = max(0, 16384 - nodes - len(stack))
            uninspected += max(0, len(items) - room)
            for key, child in reversed(items[:room]):
                escaped = str(key).replace('~','~0').replace('/','~1')
                stack.append((child, path+'/'+escaped,
                    diagnostic or str(key).lower() in DIAGNOSTIC_KEYS, depth+1))
        elif type(current) is list:
            room = max(0, 16384 - nodes - len(stack))
            uninspected += max(0, len(current) - room)
            for index in range(min(len(current), room)-1,-1,-1):
                stack.append((current[index], path+'/'+str(index), diagnostic, depth+1))
    # Specific nested trial diagnostics precede generic outer summaries; all
    # original text remains available even when only one excerpt fits.
    def priority(item):
        path, text = item
        key = path.rsplit('/', 1)[-1].lower()
        # Specific stderr/traceback/exception fields precede short generic
        # banners; nested trials precede their enclosing aggregate stderr.
        return (text.lstrip().startswith(('{','[')),
                key not in ('stderr', 'traceback', 'exception', 'exceptions'),
                -path.count('/'), len(text))
    # Stable ties follow canonical object traversal and original list order.
    entries.sort(key=priority)
    return entries, uninspected


def project(value, budget=700):
    """Whole JSON within escaped-character budget; preserve error tails on overflow."""
    entries, uninspected = _entries(value)
    result = {'evidence_sha256':content_hash(value), 'entries':[],
              'omitted_entries':len(entries), 'uninspected_values':uninspected}
    size = lambda v: len(json.dumps(v,ensure_ascii=True,separators=(',',':'), sort_keys=True))
    if size(result) > budget:
        # The caller must reserve enough room for an authenticated reference;
        # do not silently return an oversized or unauthenticated empty object.
        raise ValueError('canonical diagnostic projection budget cannot fit its evidence reference')
    for path, text in entries:
        entry = {'path':path,'text':text,'sha256':_sha(text),'omitted_chars':0}
        candidate = {**result, 'entries':result['entries']+[entry],
                     'omitted_entries':result['omitted_entries']-1}
        if size(candidate) > budget:
            if result['entries']:
                continue
            # Keep the terminal exception even for very long/non-ASCII tracebacks.
            low, high = 0, len(text)
            while low < high:
                count = (low+high+1)//2
                excerpt = {'path':path,'tail':text[-count:] if count else '',
                           'sha256':_sha(text),'omitted_chars':len(text)-count}
                trial = {**candidate,'entries':[excerpt]}
                if size(trial) <= budget: low = count
                else: high = count-1
            if not low:
                continue
            candidate['entries'] = [{'path':path,'tail':text[-low:],
                'sha256':_sha(text),'omitted_chars':len(text)-low}]
        result = candidate
    return result


def add_summary(value, summary, *, budget=500):
    if 'failure_diagnostics' in value:
        summary['failure_diagnostics'] = project(value['failure_diagnostics'], budget)
    if value.get('reason'):
        reason = value['reason']
        summary['reason'] = reason if len(reason) <= 240 else reason[-200:]
        if len(reason) > 240:
            summary['reason_excerpt'] = {'part':'tail','omitted_chars':len(reason)-200,'sha256':_sha(reason)}
    return summary


def feedback(value, receipt_id, budget):
    """Canonical failure is not labelled a successful scientific/tool result.

    Sort nested mapping keys before rendering: fresh evaluator dictionaries and
    sorted durable receipts must produce the same subsequent request bytes.
    """
    summary = {'status':value.get('status')}
    summary['receipt_id'] = receipt_id
    size = lambda v:len(json.dumps(v,ensure_ascii=True,separators=(',',':'), sort_keys=True))
    if 'failure_diagnostics' not in value:
        whole = {**deepcopy(value), 'receipt_id':receipt_id}
        if size(whole) <= budget:
            return json.dumps(whole,ensure_ascii=True,separators=(',',':'), sort_keys=True)
    if size(summary) + 40 > budget:
        summary = {'status':value.get('status'), 'details_in':'canonical_evaluations'}
        if size(summary) > budget:
            summary = {'status':value.get('status')}
        return json.dumps(summary,ensure_ascii=True,separators=(',',':'), sort_keys=True) if size(summary) <= budget else ''
    if 'failure_diagnostics' in value:
        allowance = max(0,budget-size(summary)-55)
        try:
            projected = project(value['failure_diagnostics'], allowance)
        except ValueError:
            projected = None
        candidate = {**summary,'failure_diagnostics':projected}
        if projected is not None and size(candidate) <= budget:
            summary = candidate
        else:
            summary['failure_diagnostics_omitted'] = True
    else:
        # Preserve the exact refusal identities before optional case/timing
        # metadata. Neighbour lists and full output identities stay in the
        # immutable canonical artifact when this bounded view cannot fit them.
        for key in ('matched_id','novelty','output_novelty'):
            if key not in value:
                continue
            item = deepcopy(value[key])
            if key != 'matched_id' and isinstance(item,dict):
                fields = (('accept','match_kind','matched_program_id','matched_evaluation_id','reason')
                    if key == 'novelty' else ('accept','status','evaluation_id','match'))
                item = {field:item[field] for field in fields if field in item}
                if set(item) != set(value[key]):
                    item['other_fields_omitted'] = True
            candidate = {**summary,key:item}
            if size(candidate) <= budget:summary = candidate
            elif size({**summary,key+'_omitted':True}) <= budget:summary[key+'_omitted'] = True
    if 'failure_diagnostics' not in value and value.get('reason'):
        candidate = {**summary,'reason':value['reason']}
        if size(candidate) <= budget:summary = candidate
        elif size({**summary,'reason_omitted':True}) <= budget:summary['reason_omitted'] = True
    if 'cases' in value:
        candidate = {**summary,'cases':deepcopy(value['cases'])}
        if size(candidate) <= budget:summary = candidate
        elif size({**summary,'cases_omitted':True}) <= budget:summary['cases_omitted'] = True
    return json.dumps(summary,ensure_ascii=True,separators=(',',':'), sort_keys=True)
