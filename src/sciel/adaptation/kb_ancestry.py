"""Build authenticated KB/lesson/call ancestry from immutable native replay."""
from __future__ import annotations
from copy import deepcopy
import json
from .types import content_hash, sha256_bytes
from .kb_ancestry_protocol import _need, _sha, _one, validate_node
from .kb_ancestry_replay import KBReplay


class KBAncestry(KBReplay):
    def __init__(self, native):
        self.native = native
        self.config = native.config.get("research", {})
        self._states = {}
        self._lesson_origins = {}
        # One reader belongs to one immutable retained run/export. Cutoff and
        # active recursion path are part of the key: a proof at a later event
        # or through a different dependency path cannot authorize this one.
        self._calls = {}
        self._lessons = {}

    def _experiment_call(self, ident, cutoff):
        attempts = [row for row in self._events(cutoff)
                    if row.get("event") == "research" and row.get("phase") == "attempt"
                    and row.get("experiment_id") == ident]
        attempt = _one(attempts, "research source lacks a unique native attempt")
        requests = [row for (call_id, kind), row in self.native.transcripts.items()
                    if kind == "request" and row.get("run_id") == self.native.run_id
                    and row.get("segment_id") == attempt.get("segment_id")
                    and row.get("experiment_id") == ident and row.get("role") in ("G", "repair")
                    and row.get("prompt_sha256") == attempt.get("prompt_sha256")]
        request = _one(requests, "research source has no unique native generation call")
        return request["call_id"], attempt

    def _experiment_graph(self, ident, cutoff, visiting):
        call_id, _ = self._experiment_call(ident, cutoff)
        admissions = [row for row in self._events(cutoff)
                      if row.get("event") == "iteration" and row.get("child_id")
                      and row.get("meta", {}).get("research_id") == ident]
        if admissions:
            admission = _one(admissions, "research source has ambiguous program admissions")
            return self.native.program_graph(admission["child_id"], _sha(admission["child_code"]),
                cutoff, visiting)
        # There is no episode/program to invent for a rejected or no-child G.
        return self.resolve_call(call_id, cutoff, visiting)

    def experiment_graph(self, ident, cutoff, visiting=frozenset()):
        """Close an exact native research ID, preserving rejected-call type."""
        return self._experiment_graph(ident, cutoff, visiting)

    def _conclusion_graph(self, ident, cutoff, visiting):
        """Authenticate a projected C admission, not just its cached JSON."""
        from ..hypothesis import conclusion_refs
        from ..hypothesis.parsing import decode_conclusion
        from ..hypothesis.mechanism_review import validate_binding

        admitted = _one([row for row in self._events(cutoff) if row.get("event") == "research"
            and row.get("phase") == "conclusion" and row.get("experiment_id") == ident],
            "interpretation lacks unique committed C admission")
        if admitted.get("status") == "skipped":
            _need(admitted.get("llm") is None and not admitted.get("accepted_lesson_ids"),
                  "skipped C contains invented interpretation")
            skipped = _one([row for row in self.native.events[:self._position(admitted)]
                if row.get("phase") == "aux_skipped" and row.get("kind") == "conclusion"
                and row.get("reference") == ident], "skipped C lacks a unique native skip receipt")
            results = [row for row in self.native.events[:self._position(skipped)]
                       if row.get("phase") == "aux_result" and row.get("kind") == "conclusion"
                       and row.get("reference") == ident]
            if not results:
                _need(skipped.get("reason") == "aux_call_limit", "unresolved C skip without a returned call")
                starts = [row for row in self.native.events[:self._position(skipped)]
                          if row.get("phase") == "aux_start"]
                cap = self.config.get("max_aux_calls")
                _need(type(cap) is int and len(starts) >= cap, "C skip contradicts its native auxiliary budget")
                return self._experiment_graph(ident, cutoff, visiting)
            result = _one(results, "skipped C has ambiguous returned attempts")
            call, _, _, _, _, _ = self._aux_call(result)
            terminal = result.get("response_error") is not None or result.get("finish_reason") == "error"
            _need(self.config.get("incomplete_auxiliary", "stop") == "skip"
                  and skipped.get("call_index") == result["call_index"]
                  and skipped.get("reason") == ("response_error" if terminal else "response_length")
                  and (terminal or result.get("finish_reason") == "length")
                  and result.get("refusal") in (None, "")
                  and skipped.get("generation_returned") is True
                  and skipped.get("current_call_cost_unknown") is False,
                  "skipped C differs from its actual incomplete returned response")
            # A paid partial response still has input ancestry; its skipped
            # status never upgrades its text to an accepted interpretation.
            return self.resolve_call(call, cutoff, visiting)
        result = self._result_before(admitted, "conclusion", ident)
        details = self.conclusion_inputs(result)
        self._usable(result)
        refs, typed = self.config.get("conclusion_source_refs", False), self.config.get("structured_auxiliary", False)
        decoded = (conclusion_refs.decode(details["text"], details["evidence"]) if refs else
                   decode_conclusion(details["text"], require_mechanism=True, require_typed=typed))
        parsed = decoded["conclusion"]
        if parsed is not None and not refs:
            try:
                validate_binding(parsed.mechanism_review, details["evidence"], require_typed=typed)
            except ValueError:
                parsed = None
        _need(admitted.get("status") == ("parsed" if parsed is not None else "invalid")
              and admitted.get("llm") == (parsed.to_dict() if parsed is not None else None),
              "projected C interpretation differs from its exact returned response")
        return self.resolve_call(details["call_id"], cutoff, visiting)

    def _generation_commitment(self, call_id, cutoff):
        request = self.native.transcripts[(call_id, "request")]
        ident = request.get("experiment_id")
        if ident is not None:
            actual_call, attempt = self._experiment_call(ident, cutoff)
            _need(actual_call == call_id, "generation attempt refers to another call")
            outcomes = [row for row in self._events(cutoff)
                        if row.get("event") == "research" and row.get("phase") == "outcome"
                        and row.get("experiment_id") == ident]
            outcome = _one(outcomes, "generation call lacks a committed native outcome")
            record = outcome["evidence"]
            for key in ("id", "parent_id", "parent_source_sha256", "parent_score", "island", "iteration"):
                _need(record.get(key) == attempt["evidence"].get(key),
                      "generation outcome differs from its native attempt")
            response = self.native.transcripts.get((call_id, "response"))
            _need(response is not None and record.get("generation_dispatch") != "not_dispatched",
                  "unreturned generation cannot contribute a response")
            if "generation_response_sha256" in record:
                _need(record["generation_response_sha256"] == response.get("response_sha256"),
                      "generation outcome response differs")
            return outcome, attempt
        admissions = []
        for event in self.native.events:
            if event.get("event") != "iteration":
                continue
            attempts = event.get("meta", {}).get("validation_trace", {}).get("attempts", [])
            for attempt in attempts:
                if attempt.get("call_id") == call_id:
                    response = self.native.transcripts.get((call_id, "response"), {})
                    _need(attempt.get("response_sha256") == response.get("response_sha256")
                          and attempt.get("inference_spec") == request.get("inference_spec"),
                          "generation validation attempt differs from actual call")
                    admissions.append(event)
        return _one(admissions, "generation call lacks unique native validation admission"), None

    def resolve_call(self, call_id, cutoff, visiting=frozenset()):
        from .auxiliary_ancestry import call_records

        self._events(cutoff)
        _need(("call", call_id) not in visiting, "cyclic native call ancestry")
        key = (call_id, cutoff, frozenset(visiting))
        if key not in self._calls:
            self._calls[key] = self._resolve_call(call_id, cutoff, visiting)
        else:
            # Keep the direct raw transport/chronology check on each use;
            # only its already authenticated recursive closure is reused.
            _, _, _, _, spec, dispatch = call_records(self.native, call_id)
            _need(dispatch <= cutoff, "auxiliary source was not dispatched before its consumer")
            _need(spec.adapter_id is None or spec.role in ("G", "repair"),
                  "adapted tool/C/R producers remain unsupported")
        return deepcopy(self._calls[key])

    def _resolve_call(self, call_id, cutoff, visiting):
        from .auxiliary_ancestry import GraphFragment, authenticate_call, call_records

        request, response, prompt, text, spec, dispatch = call_records(self.native, call_id)
        _need(dispatch <= cutoff, "auxiliary source was not dispatched before its consumer")
        _need(spec.adapter_id is None or spec.role in ("G", "repair"),
              "adapted tool/C/R producers remain unsupported")
        key = ("call", call_id)
        _need(key not in visiting, "cyclic native call ancestry")
        visiting = visiting | {key}
        if spec.role in ("C", "R"):
            phase = "conclusion" if spec.role == "C" else "reflection"
            result, bound = self._writer(phase, request.get("reference"), response["response_sha256"], cutoff)
            _need(bound[0] == call_id, "auxiliary writer resolved to another request")
            details = self.conclusion_inputs(result) if phase == "conclusion" else self.reflection_inputs(result)
            inputs = GraphFragment()
            for ident in details["experiment_ids"]:
                inputs.merge(self._experiment_graph(ident, details["cutoff"], visiting))
            for snapshot in details["lesson_snapshots"]:
                inputs.merge(self._lesson_graph(snapshot, details["cutoff"], visiting))
            for previous_call in details.get("prior_calls", []):
                inputs.merge(self.resolve_call(previous_call, details["cutoff"], visiting))
            for ident in details["conclusion_ids"]:
                inputs.merge(self._conclusion_graph(ident, details["cutoff"], visiting))
            commitment = result
        elif spec.role in ("G", "repair"):
            commitment, attempt = self._generation_commitment(call_id, cutoff)
            capsule = request.get("adaptation_visibility")
            _need(capsule is not None and response.get("adaptation_visibility") == capsule,
                  "failed/retry generation lost its actual delivery capsule")
            # This is a read-only view of the actual committed attempt, not an
            # iteration or eligible episode. Every field is checked above.
            admission = commitment
            if attempt is not None:
                record = attempt["evidence"]
                admission = dict(parent_id=record["parent_id"], island=record["island"],
                    iteration_no=record["iteration"], meta=dict(research_id=record["id"],
                        adaptation_parent=dict(program_id=record["parent_id"],
                            source_sha256=record["parent_source_sha256"], score=record["parent_score"])))
            inputs = self.native.resolve_delivery(capsule, prompt, spec, dispatch, visiting,
                admission=admission)
            if spec.adapter_id is not None:
                parameters, used, _ = self.native._adapter(spec, typed=True)
                inputs.merge(parameters)
                inputs.episode_roots.update(used)
                self.native.check_cutoff(inputs, dispatch)
            # One call must retain one proof whether first reached through an
            # outcome or later through its admitted program. The final worker
            # validation names every returned attempt, including rejected ones.
            # It may commit after a same-wave consumer; no episode is invented.
            commitment = self._validation(call_id)[0]
        else:
            _need(False, "KB resolver cannot authenticate this auxiliary call role")
        graph = authenticate_call(self.native, call_id, commitment, inputs)
        return graph

    def _validation(self, call_id):
        """Find the exact committed worker attempt, including rejected calls."""
        from .auxiliary_ancestry import call_records

        request, response, _, _, _, _ = call_records(self.native, call_id)
        found = []
        for event in self.native.events:
            if event.get("event") != "iteration":
                continue
            attempts = event.get("meta", {}).get("validation_trace", {}).get("attempts", [])
            for index, trace in enumerate(attempts):
                if trace.get("call_id") == call_id:
                    _need(trace.get("response_sha256") == response["response_sha256"]
                          and trace.get("inference_spec") == request["inference_spec"]
                          and trace.get("adaptation_visibility") == request.get("adaptation_visibility"),
                          "committed validation differs from returned native call")
                    found.append((event, attempts, index, trace))
        return _one(found, "call lacks one exact committed validation attempt")

    @staticmethod
    def _rejection(trace):
        from ..engine.worker import _bounded_failure

        if trace.get("research", {}).get("status") == "hypothesis_invalid":
            category, status = "hypothesis_invalid", "hypothesis_invalid"
            diagnostic = _bounded_failure(category, "missing or malformed pre-evaluation hypothesis")
        else:
            violations = [row["violation"] for row in trace.get("decisions", [])
                          if row.get("action") == "resample" and row.get("violation")]
            if not violations:
                return None
            violation = _one(violations, "ambiguous native candidate refusal")
            category, status = violation["kind"], "no_child"
            diagnostic = _bounded_failure(category, violation.get("detail"))
        return dict(status=status, category=category, diagnostic=diagnostic,
            proposed_sha256=trace["extracted_code_sha256"], source_sha256=trace["candidate_code_sha256"],
            quality_measured=False, score=None, actual_delta=None)

    def _retry_context(self, feedback, attempt, cutoff, visiting):
        from .auxiliary_ancestry import GraphFragment, call_records

        current, _ = self._experiment_call(attempt["experiment_id"], cutoff)
        _, traces, index, _ = self._validation(current)
        expected, calls = [], []
        for trace in traces[:index]:
            item = self._rejection(trace)
            if item is None:
                continue
            call_id = trace.get("call_id")
            request, _, _, _, _, dispatch = call_records(self.native, call_id)
            _need(dispatch <= cutoff, "retry diagnostic refers to a future request")
            ident = request.get("experiment_id")
            outcome, earlier = self._generation_commitment(call_id, cutoff)
            _need(earlier is not None and outcome["evidence"]["status"] == item["status"]
                  and all(earlier["evidence"][key] == attempt["evidence"][key]
                          for key in ("parent_id", "island", "iteration")),
                  "retry diagnostic differs from its actual worker attempt/outcome")
            expected.append(dict(experiment_id=ident, **item))
            calls.append(call_id)
        projected = dict(schema="sciel-research-retry-feedback/1", total_rejections=len(expected),
            omitted_count=max(0, len(expected) - 3), rejections=expected[-3:])
        _need(expected and feedback == projected, "retry evidence differs from committed worker rejections")
        graph = GraphFragment()
        # The total/omission counters also depend on earlier omitted attempts.
        for call_id in calls:
            graph.merge(self.resolve_call(call_id, cutoff, visiting))
        return graph

    def _lesson_graph(self, snapshot, cutoff, visiting):
        self._events(cutoff)
        digest = content_hash(snapshot)
        _need(("lesson", digest) not in visiting, "cyclic lesson authoring ancestry")
        key = (digest, cutoff, frozenset(visiting))
        if key not in self._lessons:
            self._lessons[key] = self._build_lesson_graph(snapshot, cutoff, visiting)
        return deepcopy(self._lessons[key])

    def _build_lesson_graph(self, snapshot, cutoff, visiting):
        from .auxiliary_ancestry import GraphFragment
        from ..hypothesis.parsing import parse_reflection

        digest = content_hash(snapshot)
        retained = [row for row in self.knowledge_at(cutoff).lessons
                    if row["id"] == snapshot.get("id") and content_hash(row) == digest]
        _one(retained, "lesson revision differs from replayed native KnowledgeBase")
        key = ("lesson", digest)
        _need(key not in visiting, "cyclic lesson authoring ancestry")
        visiting = visiting | {key}
        authoring = snapshot.get("authoring")
        _need(type(authoring) is dict and not authoring.get("prior_writers_unknown"),
              "lesson has unknown prior authors")
        inputs = GraphFragment()
        for writer in authoring["catalog"].values():
            if writer["kind"] == "retired_lesson":
                _need(not writer["prior_writers_unknown"], "inherited lesson has unknown prior authors")
                continue
            result, bound = self._writer(writer["phase"], writer["reference"],
                writer["raw_response_sha256"], cutoff)
            call_id, _, _, _, _, text = bound
            if writer["phase"] == "reflection":
                parsed = parse_reflection(text,
                    comparison_reviews=self.config.get("comparison_reviews", False),
                    comparison_review_version=self.config.get("comparison_review_version", 1))
                _need(parsed is not None, "lesson writer reflection is not parseable")
                proposals = parsed.lessons
            else:
                conclusions = [row for row in self._events(cutoff)
                               if row.get("event") == "research" and row.get("phase") == "conclusion"
                               and row.get("experiment_id") == writer["reference"]]
                admitted = _one(conclusions, "lesson writer C admission is missing")
                _need(admitted.get("status") == "parsed" and admitted.get("accepted_lesson_ids")
                      and admitted.get("lesson_error") is None, "lesson writer C was not admitted")
                # Replaying through the admission checks the parsed proposal
                # against its raw writer response before accepting any IDs.
                self.knowledge_at(self._position(admitted))
                proposals = admitted["llm"]["lessons"]
            index = writer["proposal_index"]
            _need(type(index) is int and 0 <= index < len(proposals)
                  and proposals[index]["rule"] == writer["raw_rule"],
                  "lesson authoring differs from exact writer proposal")
            graph = self.resolve_call(call_id, cutoff, visiting)
            _need(all(graph.auxiliary[root]["owner"] == "base" for root in graph.roots),
                  "lesson writer scope is not the actual shared C/R owner")
            inputs.merge(graph)
        # The displayed lesson statistics/representatives are computed from
        # retained source and exposure records, independent of text authorship.
        records = snapshot["evidence_snapshots"] + snapshot["exposure_snapshots"]
        records += [row["experiment"] for row in snapshot["inherited_exposures"]]
        for ident in sorted({row["id"] for row in records}):
            inputs.merge(self._experiment_graph(ident, cutoff, visiting))
        _need(digest in self._lesson_origins, "lesson revision was not observed by native KB replay")
        origin = self._lesson_origins[digest]
        _need(self._position(origin) <= cutoff, "lesson revision was not available at this cutoff")
        return inputs.node(kind="lesson_revision", run_id=self.native.run_id, owner="base", owner_scope="shared",
            identity=dict(lesson_id=snapshot["id"], snapshot_sha256=digest),
            proof=dict(snapshot=deepcopy(snapshot), observed_at=dict(segment_id=origin["segment_id"],
                event_seq=origin["event_seq"], event_sha256=content_hash(origin))))

    def resolve_lesson(self, ref, event, prompt, cutoff, visiting=frozenset()):
        from ..engine.research import ResearchController

        knowledge = self.knowledge_at(cutoff)
        candidates = []
        for snapshot in knowledge.lessons:
            for render in (ResearchController.lesson_text, ResearchController.decision_lesson_text):
                text = render(snapshot)
                if _sha(text) == ref.get("text_sha256") and prompt.count(text) == 1:
                    candidates.append(snapshot)
        snapshot = _one(candidates, "delivered lesson text has no unique native revision")
        return self._lesson_graph(snapshot, cutoff, visiting)

    def lesson_context(self, context, event, attempt, prompt, cutoff, visiting=frozenset()):
        from .auxiliary_ancestry import GraphFragment
        from ..engine.research import ResearchController

        ids = attempt["evidence"]["injected_ids"]
        _need(type(ids) is list and len(ids) == len(set(ids)), "invalid injected lesson identities")
        graph = GraphFragment()
        if context.get("retry_feedback") is not None:
            graph.merge(self._retry_context(context["retry_feedback"], attempt, cutoff, visiting))
        if context.get("conclusion_feedback") is not None:
            from ..hypothesis.conclusion_feedback import validate_conclusion_feedback

            knowledge = self.knowledge_at(cutoff)
            records = {row["id"]: row for row in knowledge.experiments}
            retained = {ident: value for ident, value in self._conclusions_at(cutoff).items() if ident in records}
            branch = attempt["evidence"]["branch"]
            validate_conclusion_feedback(context["conclusion_feedback"], retained, records, branch,
                enabled=self.config.get("inject_lessons", True),
                batch_only_lessons=self.config.get("batch_only_lessons", False))
            # Even an omitted card contributes its hash/size or the rendered
            # retained counter. Authenticate every candidate of this exact view.
            for ident, result in retained.items():
                if result["status"] == "parsed" and records[ident]["branch"] == branch:
                    graph.merge(self._conclusion_graph(ident, cutoff, visiting))
        if not ids:
            return graph
        knowledge = self.knowledge_at(cutoff)
        lessons = {row["id"]: row for row in knowledge.lessons}
        _need(set(ids) <= set(lessons), "injected lesson was unavailable at native dispatch")
        render = (ResearchController.decision_lesson_text if self.config.get("compact_lesson_memory", False)
                  else ResearchController.lesson_text)
        for ident in ids:
            text = render(lessons[ident])
            _need(prompt.count(text) == 1, "injected lesson revision is absent or changed in final prompt")
            graph.merge(self._lesson_graph(lessons[ident], cutoff, visiting))
        return graph

    def snapshot_context(self, snapshot, prompt, cutoff, visiting=frozenset()):
        """Resolve intersected tool context spans after the caller replays _prompt.

        A prefix of a serialized lesson carries that artifact's ancestry; this
        does not assert that its full snapshot or JSON was delivered. Available
        snapshot lessons outside the retained optional block contribute nothing.
        """
        from .auxiliary_ancestry import GraphFragment
        from ..engine.research import ResearchController

        marker = "RECENT RESEARCH CONTEXT (candidate study data, not tool-authoring instructions):\n"
        graph = GraphFragment()
        if marker not in prompt:
            return graph
        _need(prompt.count(marker) == 1, "ambiguous tool KB context framing")
        full = json.dumps(dict(lessons=snapshot.kb_lessons, failures=snapshot.artifacts,
            directive=snapshot.directive), ensure_ascii=True)
        actual = prompt.split(marker, 1)[1]
        retained = 0
        for expected, shown in zip(full, actual):
            if expected != shown:
                break
            retained += 1
        _need(retained > 0, "tool KB context differs from its frozen snapshot")
        position = len('{"lessons": [')
        for index, text in enumerate(snapshot.kb_lessons):
            encoded = json.dumps(text, ensure_ascii=True)
            _need(full[position:position + len(encoded)] == encoded, "invalid tool lesson JSON span")
            markers = [name for name in ("RESEARCH_LESSON_JSON:\n", "RESEARCH_LESSON_DECISION_JSON:\n")
                       if name in text]
            _need(len(markers) == 1, "tool lesson is not a native typed projection")
            end = text.index(markers[0]) + len(markers[0])
            payload_start = position + len(json.dumps(text[:end], ensure_ascii=True)) - 1
            if retained > payload_start:
                candidates = [row for row in self.knowledge_at(cutoff).lessons
                              if text in (ResearchController.lesson_text(row),
                                          ResearchController.decision_lesson_text(row))]
                lesson = _one(candidates, "partial tool lesson lacks a unique native revision")
                graph.merge(self._lesson_graph(lesson, cutoff, visiting))
            position += len(encoded) + (2 if index + 1 < len(snapshot.kb_lessons) else 0)
        # Other nonempty dynamic fields need their own authentic source; merely
        # copying the frozen snapshot is not a complete model-input proof.
        for key, value in (("failures", snapshot.artifacts), ("directive", snapshot.directive)):
            start = full.index(', "' + key + '": ', position) + len(', "' + key + '": ')
            _need(not value or retained <= start, "unresolved delivered tool " + key + " context")
        return graph

    def resolve_ref(self, ref, event, prompt, cutoff, visiting=frozenset()):
        if ref.get("kind") == "lesson_text":
            return self.resolve_lesson(ref, event, prompt, cutoff, visiting)
        _need(ref.get("kind") == "llm_response" and ref.get("role") == "previous_response",
              "unresolved failed-source/diagnostic projection")
        from .auxiliary_ancestry import call_records

        call_id = ref.get("call_id")
        _, _, _, text, _, _ = call_records(self.native, call_id)
        delivered = self._block(prompt, "PREVIOUS_RESPONSE_JSON (exact previous generator response):\n")
        _need(_sha(text) == ref.get("response_sha256") and delivered[1] == text
              and delivered[0] == json.dumps(text, ensure_ascii=True),
              "retry response differs from actually delivered previous text")
        # Append order is the actual transport order, not a fabricated event
        # counter or a wall clock comparison. Eventual iteration commit is later.
        entries = list(self.native.transcripts)
        prior = entries.index((call_id, "response"))
        consumers = [index for index, ((_, kind), row) in enumerate(self.native.transcripts.items())
                     if kind == "request" and row.get("prompt_sha256") == _sha(prompt)]
        _need(len(consumers) == 1 and prior < consumers[0],
              "retry input response was not available before its actual request")
        _, _, _, trace = self._validation(call_id)
        if trace.get("disposition") == "output_repeat_rejected":
            expected = {**deepcopy(trace["evaluation"]), "quality_measured": True,
                        "admission": "output_repeat_rejected"}
            actual = self._block(prompt, "MEASURED_REPEAT_JSON:\n")
        else:
            refusal = self._rejection(trace)
            _need(refusal is not None, "previous response has no committed candidate refusal")
            violations = [row["violation"] for row in trace["decisions"]
                          if row.get("action") == "resample" and row.get("violation")]
            violation = violations[-1] if violations else None
            reason = (refusal["diagnostic"] if refusal["status"] == "hypothesis_invalid"
                      else violation.get("detail") or violation["kind"])
            novelty = [row.get("novelty") for row in trace["decisions"]
                       if row.get("stage") == "code_novelty"]
            expected = dict(reason=reason, quality_measured=False, novelty=novelty[-1] if novelty else None)
            actual = self._block(prompt, "REFUSAL_JSON:\n")
        _need(actual[1] == expected and actual[0] == json.dumps(expected, ensure_ascii=True, allow_nan=False),
              "retry feedback differs from actual committed validation")
        return self.resolve_call(call_id, cutoff, visiting)
