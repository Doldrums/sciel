"""Resolve exact committed native delivery through the public reader API."""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
from .types import InferenceSpec, content_hash, require_hash, require_identifier, sha256_bytes
from .visibility import validate_capsule
from .ancestry_protocol import (SCHEMA, ResolutionUnknown, _need, _sha, episode_id, _binding,
    _validate_binding, validate_resolution, validate_episode_resolution, validate_producer_binding)

class NativeAncestry:
    def __init__(self, root, checkpoint, events, transcripts, *, sources=None):
        self.root, self.checkpoint = Path(root), checkpoint
        self.events, self.transcripts = events, transcripts
        self.run_id = checkpoint['provenance']['run_id']
        self.config = checkpoint['config']
        self.task_hash = checkpoint['identity']['task_hash']
        self.positions, self.programs, self.memo = {}, {}, {}
        self.sources = sources or {}
        _need(type(self.sources) is dict and set(self.sources) <= {'registry', 'datasets'},
              'unsupported ancestry sources')
        sequences = {}
        for index, event in enumerate(events):
            key = event.get('segment_id'), event.get('event_seq')
            _need(key not in self.positions, 'duplicate committed journal event identity')
            _need(event.get('run_id') == self.run_id, 'foreign run in ancestry prefix')
            require_identifier(key[0], 'journal segment')
            _need(type(key[1]) is int and key[1] > sequences.get(key[0], 0),
                  'journal segment sequence is not strictly increasing')
            sequences[key[0]] = key[1]
            self.positions[key] = index
            ident = (event.get('program_id') if event.get('policy') == 'initialization'
                and event.get('action') == 'seed_program' else event.get('child_id')
                if event.get('event') == 'iteration' else None)
            if ident is not None:
                require_identifier(ident, 'program admission')
                _need(ident not in self.programs, 'reused program admission identity')
                self.programs[ident] = (index, event)

    def _generation(self, event):
        source = event['child_code']
        digest = _sha(source)
        meta = event['meta']
        attempts = [a for a in meta.get('validation_trace', {}).get('attempts', [])
            if a.get('disposition') == 'evaluate' and a.get('candidate_code_sha256') == digest
            and a.get('call_id') and a.get('inference_spec')]
        _need(len(attempts) == 1, 'ancestor has no unique generated source binding')
        attempt = attempts[0]
        spec = InferenceSpec.from_dict(attempt['inference_spec'])
        _need(spec.role in ('G', 'repair'), 'ancestor is not an ordinary generated program')
        local = self.config['llm']['local_inference']
        _need(local['learner_by_island'].get(str(event['island'])) == spec.owner,
              'ancestor producer owner differs from native island')
        for key in ('base_model', 'base_revision', 'tokenizer_revision', 'chat_template_sha256'):
            _need(getattr(spec, key) == local[key], 'ancestor model identity differs')
        call = require_identifier(attempt['call_id'], 'ancestor call')
        request = self.transcripts[(call, 'request')]
        response = self.transcripts[(call, 'response')]
        _need(all(r.get('run_id') == self.run_id and r.get('segment_id') == event['segment_id']
                  and r.get('inference_spec') == spec.to_dict() for r in (request, response)),
              'generation run/segment/inference differs')
        prompt = (self.root / 'artifacts/llm_calls' / (call + '.prompt.txt')).read_bytes().decode('utf-8')
        text = (self.root / 'artifacts/llm_calls' / (call + '.response.txt')).read_bytes().decode('utf-8')
        _need(_sha(prompt) == request.get('prompt_sha256') and prompt == event.get('prompt'),
              'ancestor prompt differs from committed generation')
        _need(_sha(text) == response.get('response_sha256') == attempt.get('response_sha256'),
              'ancestor response differs from committed generation')
        capsule = attempt.get('adaptation_visibility')
        validate_capsule(capsule, prompt=prompt, inference_spec=spec.to_dict())
        _need(request.get('adaptation_visibility') == response.get('adaptation_visibility') == capsule,
              'ancestor visibility differs across receipts')
        cutoff = self.positions.get((request['segment_id'], request.get('journal_event_seq')))
        _need(cutoff is not None and cutoff < self.positions[(event['segment_id'], event['event_seq'])],
              'generation lacks a committed pre-dispatch journal cutoff')
        from ..engine.generation_budget import verify_native_selection
        verify_native_selection(self, event, request, prompt, capsule, cutoff)
        _need(any(e.get('evaluation_id') == meta.get('evaluation_id') and e.get('source_sha256') == digest
                  and e.get('ok') is True for e in meta.get('evaluations', [])),
              'ancestor source has no committed successful evaluation')
        from .export import _receipt
        evaluated = _receipt(self.root, meta['evaluation_id'], digest, self.run_id)
        _need(evaluated['task_hash'] == self.task_hash and evaluated['result']['ok'] is True
              and evaluated['result']['score'] == event['child_score'], 'ancestor evaluated source/score differs')
        return spec, capsule, prompt, cutoff

    def parent_evaluation_id(self, event):
        parent = event['meta']['adaptation_parent']
        _need(parent['program_id'] in self.programs, 'parent has no committed admission')
        origin = self.programs[parent['program_id']][1]
        source = origin.get('child_code', origin.get('source'))
        _need(_sha(source) == parent['source_sha256'], 'measured parent differs from explicit admission')
        binding = self.checkpoint.get('research_evaluation_bindings', {}).get(parent['program_id'])
        _need(binding is not None and binding['source_sha256'] == parent['source_sha256'],
              'parent has no checkpoint-authenticated evaluation identity')
        cutoff = self._generation(event)[3]
        _need(any(e.get('policy') == 'evaluation' and e.get('evaluation_id') == binding['evaluation_id']
                  and e.get('source_sha256') == parent['source_sha256']
                  for e in self.events[:cutoff + 1]),
              'parent evaluation is outside committed journal')
        return binding['evaluation_id']

    def _program(self, ident, source_hash, cutoff, visiting):
        _need(ident in self.programs, 'program has no explicit committed admission: ' + str(ident))
        index, event = self.programs[ident]
        _need(index <= cutoff, 'program was not committed before generation')
        seed = event.get('policy') == 'initialization'
        source = event.get('source') if seed else event.get('child_code')
        _need(type(source) is str and _sha(source) == source_hash, 'program admission source differs')
        if seed:
            _need(not self.config.get('warm_start') and event.get('source_sha256') == source_hash,
                  'seed is an unresolved imported program')
            return {}, None
        resolution = self.resolve(event, visiting=visiting)
        owner = self.config['llm']['local_inference']['learner_by_island'][str(event['island'])]
        binding = dict(episode_id=episode_id(event), owner=owner, run_id=self.run_id,
            segment_id=event['segment_id'], event_seq=event['event_seq'], source_sha256=source_hash)
        return {**{b['episode_id']: b for b in resolution['episode_bindings']}, binding['episode_id']: binding}, binding['episode_id']

    def _observed(self, reference, target, prompt, cutoff, visiting):
        from ..engine.observed_changes import ObservedChanges, _validate_record, _PREFIX, _MATCH_RULES, _SUFFIX
        metric = self.checkpoint['observed_changes']['identity']['metric']
        observed = ObservedChanges(metric=metric)
        for event in self.events[:cutoff + 1]:
            if event.get('policy') == 'observed_change':
                observed._insert(_validate_record(event['evidence']))
        ident = target['program_id']
        _need(ident in self.programs, 'observed target has no admission')
        event = self.programs[ident][1]
        code = event.get('child_code', event.get('source'))
        rendered = observed.block(code)
        _need(rendered and _sha(rendered) == reference.get('text_sha256')
              and len(rendered) == reference.get('chars') and rendered in prompt,
              'observed history projection differs from committed prefix')
        raw = rendered[len(_PREFIX):-len(_SUFFIX)]
        if raw.startswith(_MATCH_RULES):
            raw = raw[len(_MATCH_RULES):]
        rows = json.loads(raw)
        result, direct = {}, set()
        for row in rows:
            _need(row.get('scope') == 'unpaired', 'paired observed history is unresolved')
            records = [r for r in observed._records if r['record_sha256'] == row['record_sha256']
                       and r['id'] == row['id']]
            _need(len(records) == 1, 'observed projection has no unique committed record')
            record = records[0]
            # Even the minimal projection displays facts about both endpoints.
            # Its explicit record digest authenticates the omitted program IDs.
            refs = [(record['child_program_id'], record['child_source_sha256']),
                    (record['parent_program_id'], record['parent_source_sha256'])]
            _need('history_matches_omitted' not in row, 'omitted history contributors are unresolved')
            history = row.get('history_matches')
            if history:
                # The rendered representatives are bounded, but their counts
                # describe every matching record in the committed prefix.
                # Re-derive that inventory; an omitted ID is never inferred
                # from a score, from the final checkpoint, or from future rows.
                limit = len(history['matches'])
                _need(limit in (1, 2) and history == observed._history_matches(record, limit=limit),
                      'observed aggregate projection differs from committed index')
                _need(all(not r['alias_program_ids'] and not r['alias_ids_omitted']
                          for r in [history['query'], *history['matches']]),
                      'observed aliases lack committed program admissions')
                keys = [key for key in observed._output_keys(record)
                        if len(observed._by_output[key]) > 1]
                _need(bool(keys), 'observed aggregate has no committed output group')
                sources = observed._by_output[keys[0]]
                contributors = [observed._records[index] for source, indices in sources.items()
                                if source != record['child_source_sha256'] for index in indices]
                _need(history['matching_distinct_sources'] == len(sources) - 1
                      and history['matching_record_count'] == len(contributors)
                      and history['match_sources_omitted'] == len(sources) - 1 - limit,
                      'observed aggregate counts differ from committed contributors')
                refs.extend((r['child_program_id'], r['child_source_sha256']) for r in contributors)
            for program_id, digest in refs:
                if program_id is not None:
                    bindings, episode = self._program(program_id, digest, cutoff, visiting)
                    self._merge(result, bindings)
                    if episode:
                        direct.add(episode)
        return result, direct

    def _outcome(self, ident, cutoff):
        from ..knowledge_base.base import _experiment
        rows = [e for e in self.events[:cutoff + 1] if e.get('event') == 'research'
                and e.get('phase') == 'outcome' and e.get('experiment_id') == ident]
        _need(len(rows) == 1, 'research evidence has no unique committed outcome')
        record = _experiment(rows[0]['evidence'])
        _need(record['id'] == ident, 'research outcome identity differs')
        iterations = [e for e in self.events[:cutoff + 1] if e.get('event') == 'iteration'
                      and e.get('meta', {}).get('research_id') == ident]
        _need(len(iterations) == 1, 'research outcome has no unique admitted generation episode')
        event = iterations[0]
        _need(record['source_sha256'] == _sha(event['child_code'])
              and record['parent_id'] == event['parent_id']
              and record['island'] == event['island'] and record['iteration'] == event['iteration_no']
              and record['score'] == event['child_score'], 'research outcome differs from admitted generation')
        _need(record.get('child_id') in (None, event['child_id']), 'research child identity differs')
        _need(record['branch'] == self._branch(event['parent_id'], cutoff), 'research branch identity differs')
        return record, event

    def _branch(self, program_id, cutoff):
        seen = set()
        while True:
            _need(program_id not in seen and program_id in self.programs, 'research branch ancestry is unresolved')
            seen.add(program_id)
            index, event = self.programs[program_id]
            _need(index <= cutoff, 'research branch was not committed before generation')
            if event.get('action') == 'seed_program' or event.get('meta', {}).get('remeasurement'):
                return program_id
            program_id = event['parent_id']

    @staticmethod
    def _json_line(prompt, marker):
        parts = prompt.split('\n' + marker + ':\n')
        _need(len(parts) == 2, 'missing or ambiguous host-framed ' + marker)
        return parts[1].split('\n', 1)[0]

    def _research_fragment(self, ref, prompt, cutoff, visiting):
        from ..engine.prompt.sources import SCHEMA as SOURCE_SCHEMA
        record, event = self._outcome(ref['experiment_id'], cutoff)
        _need(record['source_sha256'] == ref['source_sha256'] and record['code_fragment']
              and _sha(record['code_fragment']) == ref['source_sha256'], 'fragment source differs from outcome')
        catalog = json.loads(self._json_line(prompt, 'SOURCE_CATALOG_JSON'))
        _need(set(catalog) == {'schema', 'catalog'} and catalog['schema'] == SOURCE_SCHEMA
              and catalog['catalog'].get(ref['source_sha256']) == record['code_fragment'],
              'research fragment exact source is not delivered')
        return self._program(event['child_id'], ref['source_sha256'], cutoff, visiting)

    def _research(self, ref, event, prompt, cutoff, visiting):
        from ..hypothesis.rendering import decode_evidence, serialize_context
        from ..knowledge_base.base import KnowledgeBase, _snapshot
        raw = self._json_line(prompt, 'RESEARCH_CONTEXT_JSON')
        _need(_sha(raw) == ref['text_sha256'] and len(raw) == ref['chars'],
              'research context differs from final delivered projection')
        context = decode_evidence(json.loads(raw))
        fields = {'parent_score', 'parent_source_sha256', 'failed_hypotheses', 'fragments',
                  'metric', 'parent_cases'}
        _need(type(context) is dict and fields <= set(context)
              and set(context) <= fields | {'research_cells_context', 'research_cells_reservation', 'retry_feedback', 'conclusion_feedback'}
              and serialize_context(context) == raw,
              'research context contains unresolved auxiliary/tool projections')
        cells = context.get('research_cells_context')
        if 'research_cells_reservation' in context:
            from ..engine.research_cells_context import MAX_RESERVED_CHARS, RESERVATION_FRAMING_CHARS
            reservation = context['research_cells_reservation']
            _need(type(reservation) is dict and set(reservation) == {'schema', 'reserved_chars', 'framing_chars', 'policy'}
                  and reservation['schema'] == 'sciel-research-cells-reservation/1'
                  and type(reservation['reserved_chars']) is int and 0 < reservation['reserved_chars'] <= MAX_RESERVED_CHARS
                  and reservation['framing_chars'] == RESERVATION_FRAMING_CHARS
                  and reservation['policy'] == 'reserve_before_ordinary_memory_selection',
                  'research cell reservation is unresolved')
        ident = event['meta'].get('research_id')
        attempts = [e for e in self.events[:cutoff + 1] if e.get('event') == 'research'
                    and e.get('phase') == 'attempt' and e.get('experiment_id') == ident]
        _need(ident and len(attempts) == 1, 'research context lacks unique committed attempt')
        attempt = attempts[0]
        evidence = attempt['evidence']
        _need(attempt['prompt_sha256'] == _sha(prompt) and evidence['id'] == ident
              and evidence['parent_id'] == event['parent_id'] and evidence['island'] == event['island']
              and evidence['iteration'] == event['iteration_no'],
              'research attempt identity/lesson ancestry differs')
        for key in ('parent_score', 'parent_source_sha256', 'metric', 'parent_cases'):
            _need(context[key] == evidence[key], 'research displayed ' + key + ' differs from attempt')
        measured = event['meta']['adaptation_parent']
        _need(evidence['parent_id'] == measured['program_id']
              and evidence['parent_source_sha256'] == measured['source_sha256']
              and evidence['parent_score'] == measured['score']
              and evidence['branch'] == self._branch(measured['program_id'], cutoff), 'research measured parent/branch differs')
        from .auxiliary_ancestry import GraphFragment
        extra = GraphFragment()
        result, direct = {}, set()
        for fragment in context['fragments']:
            record, _ = self._outcome(fragment['experiment_id'], cutoff)
            expected = dict(id='f:' + record['id'], experiment_id=record['id'], branch=record['branch'],
                source_sha256=record['source_sha256'], score=record['score'], actual_delta=record['actual_delta'],
                evidence=_snapshot(record), source_ref=record['source_sha256'])
            _need(fragment == expected, 'research fragment projection differs from committed outcome')
            added, episode = self._research_fragment(fragment, prompt, cutoff, visiting)
            self._merge(result, added)
            direct.add(episode)
        # Authenticate strings by the actual host renderer. Never parse an ID
        # out of model-authored hypothesis/error prose and treat it as lineage.
        for failure in context['failed_hypotheses']:
            candidates = []
            for previous in self.events[:cutoff + 1]:
                if previous.get('event') != 'research' or previous.get('phase') != 'outcome':
                    continue
                from ..knowledge_base.base import _experiment
                record = _experiment(previous['evidence'])
                knowledge = KnowledgeBase()
                knowledge._experiments = [record]
                if knowledge.failed_hypotheses(evidence['branch'], limit=1) == [failure]:
                    candidates.append(record)
            _need(len(candidates) == 1, 'failure projection has no unique committed source')
            outcomes = [row for row in self.events[:cutoff + 1] if row.get('event') == 'iteration'
                        and row.get('meta', {}).get('research_id') == candidates[0]['id']]
            if len(outcomes) != 1 or outcomes[0].get('child_id') is None:
                extra.merge(self.kb.experiment_graph(candidates[0]['id'], cutoff, visiting))
                continue
            record, previous = self._outcome(candidates[0]['id'], cutoff)
            added, episode = self._program(previous['child_id'], record['source_sha256'], cutoff, visiting)
            self._merge(result, added)
            direct.add(episode)
        graph = self._from_episode_refs(result, direct, visiting).merge(extra)
        if cells is not None:
            from ..engine.research_cells_context import compact_context
            if cells != compact_context([]):
                graph.merge(self.tools.resolve_context(cells, event, prompt, cutoff, visiting))
        if evidence['injected_ids'] or context.get('retry_feedback') or context.get('conclusion_feedback'):
            graph.merge(self.kb.lesson_context(context, event, attempt, prompt, cutoff, visiting))
        return graph

    @staticmethod
    def _merge(target, added):
        for ident, binding in added.items():
            _need(ident not in target or target[ident] == binding, 'conflicting ancestor episode identity')
            target[ident] = deepcopy(binding)

    @property
    def tools(self):
        if not hasattr(self, '_tools'):
            from .tool_ancestry import ToolAncestry
            self._tools = ToolAncestry(self)
        return self._tools

    @property
    def kb(self):
        if not hasattr(self, '_kb'):
            try:
                from .kb_ancestry import KBAncestry
            except ImportError as error:
                raise ResolutionUnknown('KB/call ancestry handler is not installed') from error
            self._kb = KBAncestry(self)
        return self._kb

    def authenticate_call(self, call_id, commitment, inputs):
        from .auxiliary_ancestry import authenticate_call
        return authenticate_call(self, call_id, commitment, inputs)

    @staticmethod
    def resolution_fragment(resolution):
        from .auxiliary_ancestry import GraphFragment
        return GraphFragment(
            episodes={v['episode_id']: deepcopy(v) for v in resolution['episode_bindings']},
            auxiliary={v['node_id']: deepcopy(v) for v in resolution.get('auxiliary_bindings', [])},
            roots=set(resolution.get('prompt_auxiliary_ids', [])) | set(resolution.get('parameter_auxiliary_ids', [])),
            episode_roots=set(resolution['prompt_episode_ids']) | set(resolution['parameter_episode_ids']))

    def producer_binding(self, event, resolution):
        # Only this producer node is returned. Its canonical input edges name
        # the already authenticated complete resolution; copying every inherited
        # proof just to discard it makes repeated lesson consumers expensive.
        # Keep parameter-root nodes for authenticate_call's adapted-owner join.
        from .auxiliary_ancestry import GraphFragment
        roots = set(resolution.get('prompt_auxiliary_ids', [])) | set(
            resolution.get('parameter_auxiliary_ids', []))
        graph = GraphFragment(
            auxiliary={v['node_id']: deepcopy(v) for v in resolution.get('auxiliary_bindings', [])
                       if v['node_id'] in roots},
            roots=roots,
            episode_roots=set(resolution['prompt_episode_ids']) | set(resolution['parameter_episode_ids']))
        digest = _sha(event['child_code'])
        calls = [a['call_id'] for a in event['meta']['validation_trace']['attempts']
                 if a.get('disposition') == 'evaluate' and a.get('candidate_code_sha256') == digest
                 and a.get('extracted_code_sha256') == digest and a.get('call_id')]
        _need(len(calls) == 1, 'producer lacks exact source extraction admission')
        # Authenticates raw response separately from extracted source/evaluation.
        self._generation(event)
        wrapped = self.authenticate_call(calls[0], event, graph)
        return wrapped.auxiliary[next(iter(wrapped.roots))]

    def _from_episode_refs(self, bindings, direct, visiting):
        from .auxiliary_ancestry import GraphFragment
        result = GraphFragment(episodes=deepcopy(bindings), episode_roots=set(direct))
        for ident in sorted(direct):
            binding = bindings[ident]
            position = self.positions.get((binding['segment_id'], binding['event_seq']))
            _need(binding['run_id'] == self.run_id and position is not None,
                  'direct program ancestor is not a native admission')
            event = self.events[position]
            resolution = self.resolve(event, visiting=visiting)
            if resolution.get('auxiliary_bindings'):
                inherited = self.resolution_fragment(resolution)
                node = self.producer_binding(event, resolution)
                inherited.auxiliary[node['node_id']] = node
                inherited.roots = {node['node_id']}
                inherited.episode_roots = set()
                result.merge(inherited)
        return result

    def program_graph(self, ident, source_hash, cutoff, visiting=frozenset()):
        bindings, direct = self._program(ident, source_hash, cutoff, visiting)
        return self._from_episode_refs(bindings, {direct} if direct else set(), visiting)

    def resolve_delivery(self, capsule, prompt, spec, cutoff, visiting, admission=None):
        from .auxiliary_ancestry import GraphFragment
        spec = InferenceSpec.from_dict(spec) if isinstance(spec, dict) else spec
        validate_capsule(capsule, prompt=prompt, inference_spec=spec.to_dict())
        refs = capsule['delivered_refs']
        targets = [r for r in refs if r.get('role') == 'target']
        _need(len(targets) == 1, 'ordinary ancestry requires one displayed target')
        if admission is not None:
            parent = admission['meta']['adaptation_parent']
            _need(targets[0].get('program_id') == parent['program_id'] == admission['parent_id']
                  and targets[0].get('source_sha256') == parent['source_sha256'],
                  'displayed target differs from measured native parent')
        allowed = {'program_ancestry_unresolved', 'observed_history_ancestry_unresolved',
                   'adapter_training_ancestry_unresolved', 'research_ancestry_unresolved',
                   'lesson_ancestry_unresolved', 'failure_ancestry_unresolved', 'retry_ancestry_unresolved'}
        _need(set(capsule['ancestry']['reasons']) <= allowed, 'visibility ancestry is unresolved: '
              + ', '.join(sorted(set(capsule['ancestry']['reasons']) - allowed)))
        graph = GraphFragment()
        for ref in refs:
            kind = ref['kind']
            if kind in ('program', 'public_source'):
                added = self.program_graph(ref['program_id'], ref['source_sha256'], cutoff, visiting)
                _need(kind != 'public_source' or not added.episode_roots,
                      'claimed public source is generated experience')
                graph.merge(added)
            elif kind == 'public_task':
                parts = prompt.split('\nTASK:\n')
                _need(len(parts) == 2, 'public task framing is missing or ambiguous')
                task_text, separator, _ = parts[1].partition('\n\nCURRENT PROGRAM (score ')
                _need(separator and ref.get('task_hash') == self.task_hash
                      and ref.get('text_sha256') == _sha(task_text), 'public task identity differs')
            elif kind == 'observed_changes':
                added, episodes = self._observed(ref, targets[0], prompt, cutoff, visiting)
                graph.merge(self._from_episode_refs(added, episodes, visiting))
            elif kind == 'research_context':
                _need(admission is not None, 'research delivery lacks a bound native attempt')
                graph.merge(self._research(ref, admission, prompt, cutoff, visiting))
            elif kind == 'research_fragment':
                added, episode = self._research_fragment(ref, prompt, cutoff, visiting)
                graph.merge(self._from_episode_refs(added, {episode}, visiting))
            elif kind == 'research_card':
                graph.merge(self.tools.resolve_ref(ref, admission, prompt, cutoff, visiting))
            elif kind in ('lesson_text', 'failure_context', 'llm_response', 'failed_source'):
                graph.merge(self.kb.resolve_ref(ref, admission, prompt, cutoff, visiting))
            else:
                raise ResolutionUnknown('unresolved delivered projection: ' + kind)
        return graph

    def resolve(self, event, *, visiting=frozenset()):
        ident = episode_id(event)
        _need(ident not in visiting, 'cyclic episode ancestry')
        if ident in self.memo:
            return deepcopy(self.memo[ident])
        spec, capsule, prompt, cutoff = self._generation(event)
        visiting = visiting | {ident}
        graph = self.resolve_delivery(capsule, prompt, spec, cutoff, visiting, admission=event)
        direct, prompt_aux = set(graph.episode_roots), set(graph.roots)
        parameters, parameter_aux, adapter_bindings = set(), set(), []
        if spec.adapter_id is not None:
            inherited, parameters, adapter_bindings = self._adapter(spec)
            if graph.auxiliary or inherited.auxiliary:
                inherited, parameters, adapter_bindings = self._adapter(spec, typed=True)
            parameter_aux = set(inherited.roots)
            graph.merge(inherited)
        bindings = graph.episodes
        _need(ident not in bindings, 'episode appears in its own ancestry')
        self.check_cutoff(graph, cutoff)
        value = dict(schema=SCHEMA, capsule_sha256=capsule['sha256'],
            prompt_episode_ids=sorted(direct), parameter_episode_ids=sorted(parameters),
            ancestor_episode_ids=sorted(bindings),
            donor_episode_ids=sorted(k for k, b in bindings.items() if b['owner'] != spec.owner),
            episode_bindings=[bindings[k] for k in sorted(bindings)], adapter_bindings=adapter_bindings)
        if graph.auxiliary:
            from .auxiliary_ancestry import SCHEMA as AUX_SCHEMA
            value.update(schema=AUX_SCHEMA, auxiliary_bindings=[graph.auxiliary[k] for k in sorted(graph.auxiliary)],
                         prompt_auxiliary_ids=sorted(prompt_aux), parameter_auxiliary_ids=sorted(parameter_aux))
        value['sha256'] = content_hash(value)
        validate_resolution(value, capsule_sha256=capsule['sha256'], owner=spec.owner)
        self.memo[ident] = value
        return deepcopy(value)

    def check_cutoff(self, graph, cutoff):
        """Also used for parameter inputs of failed/retried G calls."""
        bindings = graph.episodes
        for binding in bindings.values():
            if binding['run_id'] != self.run_id:
                continue
            position = self.positions.get((binding['segment_id'], binding['event_seq']))
            _need(position is not None and position <= cutoff,
                  'same-run parameter/prompt ancestor is outside pre-dispatch commitment')
            origin = self.events[position]
            owner = self.config['llm']['local_inference']['learner_by_island'].get(str(origin.get('island')))
            _need(origin.get('event') == 'iteration' and _sha(origin['child_code']) == binding['source_sha256']
                  and owner == binding['owner'], 'same-run corpus binding differs from native admission')

    @property
    def adapter_context(self):
        if not hasattr(self, '_adapter_context'):
            from .adapter_ancestry import context
            self._adapter_context = context(self.sources)
        return self._adapter_context

    def _adapter(self, spec, *, typed=False):
        return self.adapter_context.resolve(spec, getattr(self, 'run_id', None), typed=typed)
