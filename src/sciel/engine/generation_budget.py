"""Opt-in whole-input selection for ordinary G; never provider truncation."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json

SCHEMA = 'sciel-generation-token-budget/1'
SELECTION = 'sciel-generation-input-selection/1'
OPS = frozenset({'full_rewrite', 'diff', 'cross', 'error_repair'})
CORE = frozenset({'parent_score', 'parent_source_sha256', 'metric', 'parent_cases'})


def _need(ok, message):
    if not ok:
        raise ValueError(message)


def _sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _hash(value):
    return _sha(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                           allow_nan=False))


def validate_config(config):
    value = config.get('generation_context_tokens')
    if 'generation_context_tokens' in config:
        _need(type(value) is int and value > 0 and config.get('enabled') is True,
              'generation_context_tokens requires enabled research and a positive integer')
    return value


def contract(config, llm):
    context = validate_config(config)
    if context is None:
        return None
    output = getattr(llm, 'max_tokens', None)
    _need(getattr(llm, 'local_inference', False) is True
          and callable(getattr(llm, 'count_prompt_tokens', None)),
          'generation token budget requires an exact pinned local counter')
    _need(type(output) is int and 0 < output < context,
          'generation context must exceed its reserved output tokens')
    limits = getattr(llm, 'limits', None)
    if isinstance(limits, dict) and 'context_tokens' in limits:
        _need(context <= limits['context_tokens'], 'generation budget exceeds provider context limit')
    tokenizer = deepcopy(llm.prompt_tokenizer_identity)
    _need(type(tokenizer) is dict and bool(tokenizer), 'generation tokenizer identity is missing')
    return dict(schema=SCHEMA, context_tokens=context, reserved_output_tokens=output,
                tokenizer=tokenizer)


class GenerationBudgetExceeded(ValueError):
    def __init__(self, receipt):
        super().__init__('generation fixed framing exceeds configured token budget')
        self.receipt = deepcopy(receipt)


def snapshot_value(snapshot):
    result = asdict(snapshot)
    result['code_hashes'] = sorted(result['code_hashes'])
    return result


def snapshot_from(value):
    from .types import Snapshot
    value = deepcopy(value)
    value['code_hashes'] = set(value['code_hashes'])
    return Snapshot(**value)


def render(view):
    from .prompt.build import build
    from .research import render_generation
    snap = snapshot_from(view['snapshot'])
    return render_generation(build(view['op'], snap) + view['suffix'], view['context'],
        snap.research_context, view['instruction'], view['blocks'], view['retry_feedback'])


def _next(view):
    """A finite ordered prefix of whole-input removals; no text slicing."""
    snap, context = view['snapshot'], view['context']
    ctx = snap['research_context']
    for kind, rows in (('inspiration', snap['inspirations']), ('fragment', ctx['fragments'])):
        if rows and not (kind == 'inspiration' and view['op'] == 'cross' and len(rows) == 1):
            index = len(rows) - 1
            removed = rows.pop()
            if kind == 'fragment':
                _need(len(context['fragments']) == index + 1, 'fragment projection count differs')
                context['fragments'].pop()
            return dict(kind=kind, index=index, removed_sha256=_hash(removed))
    if 'research_cells_context' in context:
        removed = context.pop('research_cells_context')
        ctx.pop('research_cells_context', None)
        snap['research_cells_context'] = None
        return dict(kind='research_cards', removed_sha256=_hash(removed))
    optional = {k: v for k, v in context.items() if k not in CORE}
    if (snap['kb_lessons'] or optional != {'failed_hypotheses': [], 'fragments': []}
            or any(view['blocks'].values()) or view['retry_feedback'] is not None):
        removed = dict(context=optional, lessons=snap['kb_lessons'], blocks=view['blocks'],
                       retry_feedback=view['retry_feedback'])
        view['context'] = {k: deepcopy(v) for k, v in context.items() if k in CORE}
        view['context'].update(failed_hypotheses=[], fragments=[])
        for key in set(optional) | {'restart_observations', 'restart_predictions',
                'evaluated_detached_study', 'evaluated_study_review',
                'generated_paired_history', 'generated_paired_reviews'}:
            ctx.pop(key, None)
        ctx.update(failed_hypotheses=[], fragments=[], injected_ids=[], lesson_texts=[])
        snap['kb_lessons'] = []
        view['blocks'] = {k: '' for k in view['blocks']}
        view['retry_feedback'] = None
        return dict(kind='research_memory', removed_sha256=_hash(removed))
    if snap['artifacts'] and view['op'] != 'error_repair':
        removed = snap['artifacts']; snap['artifacts'] = {}
        return dict(kind='optional_artifacts', removed_sha256=_hash(removed))
    # Only the host-bound complete appended history block can be omitted.
    # The directive's compute/interface instructions remain byte-identical.
    observed = (snap.get('adaptation_context') or {}).get('observed_changes')
    if observed:
        size = observed['chars']
        _need(type(size) is int and 0 < size <= len(snap['directive']), 'invalid history size')
        block = snap['directive'][-size:]
        _need(_sha(block) == observed['text_sha256'], 'history is not the exact bound directive suffix')
        snap['directive'] = snap['directive'][:-size]
        snap['adaptation_context']['observed_changes'] = None
        return dict(kind='observed_changes', removed_sha256=_sha(block))
    return None


def _measure(prompt, identity, count):
    tokens = count(prompt)
    _need(type(tokens) is int and tokens > 0, 'invalid exact generation token count')
    return dict(prompt_sha256=_sha(prompt), prompt_tokens=tokens,
                total_reserved_tokens=tokens + identity['reserved_output_tokens'])


def select(snapshot, context, ctx, instruction, blocks, retry_feedback, prompt, op, identity, count):
    from .prompt.build import build
    _need(op in OPS and identity is not None, 'unsupported generation budget boundary')
    base = build(op, snapshot)
    _need(prompt.startswith(base), 'generation budget requires the exact standard prompt prefix')
    _need(snapshot.research_context == ctx, 'generation context differs from the original snapshot')
    source = dict(snapshot=snapshot_value(snapshot), context=deepcopy(context),
                  instruction=instruction, blocks=deepcopy(blocks), retry_feedback=deepcopy(retry_feedback),
                  suffix=prompt[len(base):], op=op)
    view = deepcopy(source)
    rendered = render(view)
    measurements = [_measure(rendered, identity, count)]
    steps = []
    while measurements[-1]['total_reserved_tokens'] > identity['context_tokens']:
        step = _next(view)
        if step is None:
            break
        steps.append(step)
        rendered = render(view)
        measurements.append(_measure(rendered, identity, count))
    receipt = dict(schema=SELECTION, contract=deepcopy(identity), source=source,
                   source_sha256=_hash(source), steps=steps, measurements=measurements,
                   selected_sha256=_hash(view),
                   status='ready' if measurements[-1]['total_reserved_tokens'] <= identity['context_tokens']
                   else 'fixed_core_exceeds_context', generation_dispatch='not_dispatched')
    if receipt['status'] != 'ready':
        raise GenerationBudgetExceeded(receipt)
    return snapshot_from(view['snapshot']), rendered, view['context'], receipt


def verify_selection(receipt, *, prompt, contract, count=None):
    """Replay exact selected source/visibility, never reinterpret omitted memory."""
    _need(type(receipt) is dict and set(receipt) == {'schema', 'contract', 'source',
          'source_sha256', 'steps', 'measurements', 'selected_sha256', 'status', 'generation_dispatch'}
          and receipt['schema'] == SELECTION and receipt['contract'] == contract
          and receipt['status'] == 'ready' and receipt['generation_dispatch'] == 'not_dispatched',
          'invalid generation selection receipt')
    _need(_hash(receipt['source']) == receipt['source_sha256'], 'generation source selection hash differs')
    view = deepcopy(receipt['source'])
    _need(type(receipt['steps']) is list and type(receipt['measurements']) is list
          and len(receipt['measurements']) == len(receipt['steps']) + 1,
          'generation selection measurements differ')
    for index, measured in enumerate(receipt['measurements']):
        rendered = render(view)
        _need(type(measured) is dict and set(measured) == {'prompt_sha256', 'prompt_tokens', 'total_reserved_tokens'}
              and measured['prompt_sha256'] == _sha(rendered)
              and type(measured['prompt_tokens']) is int and measured['prompt_tokens'] > 0
              and measured['total_reserved_tokens'] == measured['prompt_tokens'] + contract['reserved_output_tokens'],
              'generation selection measurement differs')
        if count is not None:
            _need(measured == _measure(rendered, contract, count), 'generation token replay differs')
        if index < len(receipt['steps']):
            _need(measured['total_reserved_tokens'] > contract['context_tokens']
                  and _next(view) == receipt['steps'][index], 'generation selection order differs')
    _need(measured['total_reserved_tokens'] <= contract['context_tokens']
          and _hash(view) == receipt['selected_sha256'] and rendered == prompt,
          'generation selected prompt differs')
    return snapshot_from(view['snapshot']), view['context']


def verify_native_selection(native, event, request, prompt, capsule, cutoff):
    """Bind the projection to the committed original snapshot and shown refs."""
    configured = native.config.get('research', {}).get('generation_context_tokens')
    ident = event.get('meta', {}).get('research_id')
    attempts = [e for e in native.events[:cutoff + 1] if e.get('event') == 'research'
                and e.get('phase') == 'attempt' and e.get('experiment_id') == ident]
    if configured is None:
        _need(not any('generation_token_budget' in e for e in attempts),
              'unconfigured generation selection receipt')
        return
    _need(len(attempts) == 1 and attempts[0].get('prompt_sha256') == _sha(prompt),
          'generation selection lacks its committed attempt')
    identity = native.checkpoint.get('research', {}).get('generation_token_budget')
    _need(type(identity) is dict and identity.get('schema') == SCHEMA
          and identity.get('context_tokens') == configured
          and identity.get('reserved_output_tokens') == native.config['llm']['max_tokens'],
          'generation selection contract differs from checkpoint')
    receipt = attempts[0].get('generation_token_budget')
    selected, context = verify_selection(receipt, prompt=prompt, contract=identity)
    source = receipt['source']['snapshot']
    _need(selected.parent_id == event['parent_id'] and selected.island == event['island']
          and selected.iteration_no + 1 == event['iteration_no']
          and dict(selected.inference_spec, role=request['inference_spec']['role']) == request['inference_spec'],
          'generation selection differs from native request')
    if request.get('parallel_wave') is not None:
        from .parallel_store import _read_json, _manifest, _wave_id
        directory = native.root / '.parallel' / ('wave-' + _wave_id(request['parallel_wave']))
        _need(_manifest(directory)['phase'] == 'committed', 'generation selection wave is not committed')
        jobs = _read_json(directory / 'jobs.json')['jobs']
        matches = [j for j in jobs if j.get('job_id') == request.get('parallel_job')]
        _need(len(matches) == 1, 'generation original snapshot is missing or ambiguous')
        original = matches[0]['snapshot']
        # A tool episode may add freshly authenticated cards after the frozen
        # job intent. Their actual provenance is checked by existing handlers.
        for key in ('parent_id', 'parent_code', 'parent_score', 'task_prompt', 'directive',
                    'kb_lessons', 'artifacts', 'inspirations', 'adaptation_context',
                    'inference_spec', 'inference_route', 'inference_deployment',
                    'required_entrypoints', 'iteration_no', 'island', 'model'):
            _need(source.get(key) == original.get(key), 'generation original snapshot differs: ' + key)
        without_cards = lambda ctx: {k: v for k, v in ctx.items() if k != 'research_cells_context'}
        _need(without_cards(source['research_context']) == without_cards(original['research_context']),
              'generation original research snapshot differs')
    from ..adaptation.visibility import capture_research_delivery, build_capsule
    from ..hypothesis.rendering import serialize_context
    capture_research_delivery(selected, prompt, context, serialize_context(context))
    expected = build_capsule(selected, prompt, request['inference_spec'], op=receipt['source']['op'])
    # Retry response refs are independently authenticated by the existing KB
    # ancestry reader. All selected task/source/memory refs must match exactly.
    shown = [r for r in capsule['delivered_refs'] if r['kind'] != 'llm_response']
    _need(shown == expected['delivered_refs'], 'generation selected visibility differs')
    _need(attempts[0]['evidence']['injected_ids'] == selected.research_context['injected_ids'],
          'generation selected lesson identities differ')
