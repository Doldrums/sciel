"""Replay exact committed KB states and auxiliary request inputs."""
from __future__ import annotations
from copy import deepcopy
import json
from .types import content_hash, sha256_bytes
from .kb_ancestry_protocol import _need, _sha, _one, validate_node


class KBReplay:
    def _events(self, cutoff):
        _need(type(cutoff) is int and 0 <= cutoff < len(self.native.events),
              "invalid KB ancestry cutoff")
        return self.native.events[:cutoff + 1]

    def _position(self, event):
        key = event.get("segment_id"), event.get("event_seq")
        position = self.native.positions.get(key)
        _need(position is not None and self.native.events[position] == event,
              "KB event differs from its committed native identity")
        return position

    def _aux_call(self, result):
        """Resolve one committed C/R result to a unique actual transport call."""
        _need(result.get("event") == "research" and result.get("phase") == "aux_result",
              "lesson writer lacks a committed auxiliary result")
        position = self._position(result)
        kind = result.get("kind")
        _need(kind in ("conclusion", "reflection"), "unsupported lesson writer role")
        starts = [event for event in self.native.events[:position]
                  if event.get("event") == "research" and event.get("phase") == "aux_start"
                  and event.get("kind") == kind
                  and event.get("reference") == result.get("reference")
                  and event.get("call_index") == result.get("call_index")]
        start = _one(starts, "auxiliary result has no unique issued native request")
        role = "C" if kind == "conclusion" else "R"
        candidates = []
        for (call_id, event_type), request in self.native.transcripts.items():
            if (event_type == "request" and request.get("run_id") == self.native.run_id
                    and request.get("segment_id") == start.get("segment_id")
                    and request.get("role") == role
                    and request.get("reference") == start.get("reference")
                    and request.get("prompt_sha256") == start.get("prompt_sha256")):
                response = self.native.transcripts.get((call_id, "response"))
                if response and response.get("response_sha256") == result.get("response_sha256"):
                    candidates.append((call_id, request, response))
        call_id, request, response = _one(candidates,
            "auxiliary writer has no unique role/reference/request binding")
        from .types import InferenceSpec

        spec = InferenceSpec.from_dict(request.get("inference_spec"))
        _need(spec.role == role and response.get("inference_spec") == spec.to_dict()
              and response.get("run_id") == self.native.run_id
              and response.get("segment_id") == request.get("segment_id")
              and response.get("reference") == request.get("reference"),
              "auxiliary writer inference/run identity differs")
        from .auxiliary_ancestry import call_records

        actual_request, actual_response, prompt, text, _, _ = call_records(self.native, call_id)
        _need(actual_request == request and actual_response == response, "auxiliary call changed while binding")
        _need(_sha(prompt) == request.get("prompt_sha256")
              and _sha(text) == response.get("response_sha256")
              and result.get("response_sha256") == _sha(text),
              "auxiliary writer text differs from native receipts")
        dispatch = self.native.positions.get((request["segment_id"], request.get("journal_event_seq")))
        _need(dispatch is not None and self._position(start) <= dispatch < position,
              "auxiliary dispatch falls outside its native request/result interval")
        # The journal can intentionally retain a prefix; the full text file is
        # authoritative only when that prefix/truncation receipt also agrees.
        _need(result.get("response") == text[:32768]
              and result.get("response_truncated") == (len(text) > 32768),
              "auxiliary result text projection differs")
        _need(result.get("response_error") == response.get("response_error"),
              "auxiliary invalid-response metadata differs from transport")
        if self.config.get("structured_auxiliary", False):
            _need(result.get("finish_reason") == response.get("finish_reason"),
                  "auxiliary finish metadata differs from transport")
            usable = (result.get("response_error") is None and result.get("finish_reason") == "stop"
                      and result.get("refusal") in (None, ""))
            _need(result.get("structured_response_usable") is usable,
                  "auxiliary usability differs from actual finish/refusal")
        return call_id, start, request, response, prompt, text

    def _writer(self, phase, reference, response_sha256, cutoff):
        results = [event for event in self._events(cutoff)
                   if event.get("event") == "research"
                   and event.get("phase") == "aux_result"
                   and event.get("kind") == phase
                   and event.get("reference") == reference
                   and event.get("response_sha256") == response_sha256]
        result = _one(results, "lesson writer result is missing or ambiguous")
        return result, self._aux_call(result)

    def _usable(self, result):
        _need(result.get("response_error") is None
              and (not self.config.get("structured_auxiliary", False)
                   or result.get("structured_response_usable") is True),
              "accepted auxiliary admission contradicts its unusable returned response")

    def _result_before(self, event, kind, reference):
        values = [row for row in self.native.events[:self._position(event)]
                  if row.get("event") == "research" and row.get("phase") == "aux_result"
                  and row.get("kind") == kind and row.get("reference") == reference]
        # A C correction can share a reference. Admission follows its last
        # returned response; there must be no intervening admission of that ref.
        _need(values, "KB admission lacks its actual auxiliary response")
        result = values[-1]
        self._aux_call(result)
        return result

    def knowledge_at(self, cutoff):
        """Replay real KB mutations; retain exact revisions before each call."""
        from ..hypothesis.parsing import parse_reflection
        from ..knowledge_base.base import KnowledgeBase

        if cutoff in self._states:
            return deepcopy(self._states[cutoff])
        knowledge = KnowledgeBase(**{key: self.config[key] for key in
            ("max_lessons", "max_experiments", "max_fragments")})
        for event in self._events(cutoff):
            if event.get("event") != "research":
                continue
            phase = event.get("phase")
            if phase == "outcome":
                knowledge.record(deepcopy(event["evidence"]))
            elif phase == "conclusion" and event.get("accepted_lesson_ids"):
                result = self._result_before(event, "conclusion", event["experiment_id"])
                self._usable(result)
                parsed = event.get("llm")
                _need(event.get("status") == "parsed" and event.get("lesson_error") is None
                      and type(parsed) is dict, "accepted conclusion lacks successful native admission")
                inputs = self.conclusion_inputs(result)
                if self.config.get("conclusion_source_refs", False):
                    from ..hypothesis.conclusion_refs import decode

                    decoded = decode(inputs["text"], inputs["evidence"])["conclusion"]
                else:
                    from ..hypothesis.parsing import decode_conclusion
                    from ..hypothesis.mechanism_review import validate_binding

                    decoded = decode_conclusion(inputs["text"], require_mechanism=True,
                        require_typed=self.config.get("structured_auxiliary", False))["conclusion"]
                    if decoded is not None:
                        validate_binding(decoded.mechanism_review, inputs["evidence"],
                            require_typed=self.config.get("structured_auxiliary", False))
                _need(decoded is not None and decoded.to_dict() == parsed,
                      "accepted conclusion differs from its actual writer response")
                accepted = knowledge.add_lessons(deepcopy(parsed["lessons"]),
                    allowed_evidence_ids=[event["experiment_id"]], writer_context={
                        "phase": "conclusion", "reference": event["experiment_id"],
                        "raw_response_sha256": result["response_sha256"]})
                _need(accepted == event["accepted_lesson_ids"],
                      "replayed conclusion lesson IDs differ from native admission")
            elif phase == "reflection" and event.get("applied") is not None:
                result = self._result_before(event, "reflection", event["through"])
                _, _, _, _, _, text = self._aux_call(result)
                self._usable(result)
                parsed = parse_reflection(text,
                    comparison_reviews=self.config.get("comparison_reviews", False),
                    comparison_review_version=self.config.get("comparison_review_version", 1))
                _need(parsed is not None and event.get("status") == "parsed"
                      and event.get("error") is None, "accepted reflection lacks successful native admission")
                contexts = [row for row in self.native.events[:self._position(result)]
                            if row.get("event") == "research" and row.get("phase") == "reflection_context"
                            and row.get("through") == event["through"]]
                context = _one(contexts, "reflection admission lacks unique delivered context")
                _need(set(parsed.retire_ids) <= set(context["lesson_ids"]),
                      "reflection retired a lesson absent from its delivered context")
                applied = knowledge.reflect(parsed.lessons, parsed.retire_ids,
                    allowed_evidence_ids=context["experiment_ids"], writer_context={
                        "phase": "reflection", "reference": event["through"],
                        "raw_response_sha256": result["response_sha256"]})
                _need(applied == event["applied"],
                      "replayed reflection mutations differ from native admission")
            for lesson in knowledge.lessons:
                digest = content_hash(lesson)
                self._lesson_origins.setdefault(digest, deepcopy(event))
        self._states[cutoff] = deepcopy(knowledge)
        return knowledge

    @staticmethod
    def _block(prompt, marker, *, optional=False):
        parts = prompt.split(marker)
        if optional and len(parts) == 1:
            return None
        _need(len(parts) == 2, "auxiliary prompt framing is missing or ambiguous: " + marker.strip())
        raw = parts[1].split("\n", 1)[0]
        _need(bool(raw), "auxiliary prompt has an empty evidence block")
        from ..hypothesis.rendering import decode_evidence

        return raw, decode_evidence(json.loads(raw))

    def _conclusions_at(self, cutoff):
        fields = {"status", "llm", "observed_forecast_verdict", "observed_comparison",
                  "verdict_matches_observation", "accepted_lesson_ids", "lesson_error",
                  "lesson_error_detail", "source_evidence", "lesson_admission"}
        result = {}
        for event in self._events(cutoff):
            if event.get("event") == "research" and event.get("phase") == "conclusion":
                result[event["experiment_id"]] = {
                    key: deepcopy(event[key]) for key in fields if key in event}
        return result

    def reflection_inputs(self, result):
        """Authenticate all delivered ordinary R records, lessons and reviews.

        The selection predicate identifies the already recorded delivered set;
        this verifies projection bytes, not a new token-budget measurement.
        Unsupported auxiliary extensions cannot hide in an extra prompt tail:
        the complete production renderer must reproduce the saved prompt.
        """
        from ..hypothesis.conclusion_feedback import reflection_review_context, REFLECTION_REVIEW_PREFIX
        from ..hypothesis.prompts import reflection_prompt
        from ..hypothesis.rendering import serialize_reflection

        call_id, start, request, response, prompt, text = self._aux_call(result)
        _need(start["kind"] == "reflection", "reflection input requested for another role")
        cutoff = self._position(start) - 1
        contexts = [row for row in self._events(cutoff)
                    if row.get("event") == "research" and row.get("phase") == "reflection_context"
                    and row.get("through") == start["reference"]]
        context = _one(contexts, "R lacks its unique native reflection context")
        knowledge = self.knowledge_at(cutoff)
        wanted_records, wanted_lessons = set(context["experiment_ids"]), set(context["lesson_ids"])
        _need(len(wanted_records) == len(context["experiment_ids"])
              and len(wanted_lessons) == len(context["lesson_ids"]), "duplicate R input identity")
        selector = knowledge.reflection_context if "selection_epoch" in context else knowledge.legacy_reflection_context
        options = ({"selection_epoch": context["selection_epoch"]} if "selection_epoch" in context else {})
        records, lessons = selector(self.config["reflection_interval"], **options,
            accepts=lambda rows, cards: {row["id"] for row in rows} <= wanted_records
            and {row["id"] for row in cards} <= wanted_lessons)
        _need([row["id"] for row in records] == context["experiment_ids"]
              and [row["id"] for row in lessons] == context["lesson_ids"],
              "R delivered membership differs from retained native evidence")
        raw, _ = self._block(prompt, "EVIDENCE_JSON:\n")
        _need(raw == serialize_reflection(records, lessons)
              and _sha(raw) == context.get("context_sha256"),
              "R evidence projection differs from native source records")
        shown_review = self._block(prompt, REFLECTION_REVIEW_PREFIX, optional=True)
        review, review_ids = None, []
        if shown_review is not None:
            review_raw, review = shown_review
            _need(type(review) is dict and type(review.get("cards")) is list,
                  "invalid delivered R review projection")
            review_ids = [row["experiment_id"] for row in review["cards"]]
            omitted_ids = {row["experiment_id"] for row in review["omitted"]}
            ordinary = knowledge._evidence()
            ordinary.update((row["id"], row) for row in knowledge.experiments)
            reproduced = reflection_review_context(self._conclusions_at(cutoff), ordinary,
                context["review_selected_experiment_ids"], max_chars=review["max_chars"],
                batch_only_lessons=self.config.get("batch_only_lessons", False),
                accepts=lambda value: {row["experiment_id"] for row in value["cards"]} <= set(review_ids)
                and {row["experiment_id"] for row in value["omitted"]} <= omitted_ids)
            from ..knowledge_base.catalogs import canonical_json

            _need(reproduced == review and canonical_json(review) == review_raw
                  and _sha(review_raw) == context.get("review_context_sha256"),
                  "R conclusion review differs from retained C outputs")
        else:
            _need(context.get("review_context_sha256") is None,
                  "R dropped its recorded delivered reviews")
        _need(prompt == reflection_prompt(records, lessons, review_context=review,
            structured=self.config.get("structured_auxiliary", False)),
            "R contains unresolved or changed auxiliary prompt content")
        return dict(call_id=call_id, commitment=result, request=request, response=response,
            prompt=prompt, text=text, cutoff=cutoff, experiment_ids=context["experiment_ids"],
            lesson_snapshots=[deepcopy(row) for row in knowledge.lessons if row["id"] in wanted_lessons],
            conclusion_ids=sorted({row["experiment_id"] for row in (review or {}).get("cards", [])
                                   if row.get("review_status") != "unavailable"}
                                  | {row["experiment_id"] for row in (review or {}).get("omitted", [])
                                     if row["experiment_id"] in self._conclusions_at(cutoff)}))

    def conclusion_inputs(self, result):
        """Reproduce the full ordinary C prompt from its real outcome/source pair."""
        from ..hypothesis.conclusion_source import validate_binding
        from ..hypothesis.prompts import conclusion_prompt
        from ..hypothesis.rendering import serialize_record

        call_id, start, request, response, prompt, text = self._aux_call(result)
        _need(start["kind"] == "conclusion", "conclusion input requested for another role")
        cutoff = self._position(start) - 1
        outcomes = [row for row in self._events(cutoff)
                    if row.get("event") == "research" and row.get("phase") == "outcome"
                    and row.get("experiment_id") == start["reference"]]
        outcome = _one(outcomes, "C has no unique committed source outcome")
        record = outcome["evidence"]
        evidence = {key: deepcopy(value) for key, value in record.items() if key != "code_fragment"}
        _, delivered = self._block(prompt, "EVIDENCE_JSON:\n")
        if "conclusion_source" in delivered:
            conclusions = [row for row in self.native.events[self._position(result) + 1:]
                           if row.get("event") == "research" and row.get("phase") == "conclusion"
                           and row.get("experiment_id") == start["reference"]]
            admitted = _one(conclusions, "C source projection lacks its retained native source pair")
            source = admitted.get("source_evidence")
            evidence["conclusion_source"] = validate_binding(record, source)
        raw, _ = self._block(prompt, "EVIDENCE_JSON:\n")
        _need(raw == serialize_record(evidence), "C evidence differs from its actual outcome/source pair")
        expected = conclusion_prompt(evidence,
            structured=self.config.get("structured_auxiliary", False),
            source_refs=self.config.get("conclusion_source_refs", False))
        prior_calls = []
        if prompt != expected:
            from ..hypothesis import conclusion_refs
            from ..hypothesis.parsing import decode_conclusion
            from ..hypothesis.mechanism_review import validate_binding as mechanism_binding

            _need(self.config.get("conclusion_repair_attempts", 0) == 1,
                  "C contains unresolved or changed auxiliary prompt content")
            repairs = [row for row in self._events(cutoff) if row.get("event") == "research"
                       and row.get("phase") == "conclusion_repair"
                       and row.get("experiment_id") == record["id"]]
            repair = _one(repairs, "C correction lacks unique native request")
            previous = [row for row in self.native.events[:self._position(repair)]
                        if row.get("event") == "research" and row.get("phase") == "aux_result"
                        and row.get("kind") == "conclusion" and row.get("reference") == record["id"]
                        and row.get("response_sha256") == repair.get("previous_response_sha256")]
            prior = _one(previous, "C correction lacks its exact earlier response")
            prior_call, _, _, _, prior_prompt, prior_text = self._aux_call(prior)
            self._usable(prior)
            _need(prior_prompt == expected and repair.get("attempt") == 1,
                  "C correction changed original evidence or retry count")
            refs = self.config.get("conclusion_source_refs", False)
            typed = self.config.get("structured_auxiliary", False)
            decoded = (conclusion_refs.decode(prior_text, evidence) if refs else
                       decode_conclusion(prior_text, require_mechanism=True, require_typed=typed))
            parsed, diagnostic = decoded["conclusion"], decoded["error"]
            if parsed is not None and not refs:
                try:
                    mechanism_binding(parsed.mechanism_review, evidence, require_typed=typed)
                except ValueError as exc:
                    parsed = None
                    diagnostic = getattr(exc, "diagnostic", dict(code="mechanism_binding_invalid",
                                                                  path="mechanism_review"))
            _need(parsed is None and diagnostic == repair.get("diagnostic")
                  and prompt == conclusion_refs.repair_prompt(expected, prior_text, diagnostic),
                  "C correction differs from actual parser refusal or previous output")
            prior_calls.append(prior_call)
        return dict(call_id=call_id, commitment=result, request=request, response=response,
            prompt=prompt, text=text, cutoff=cutoff, experiment_ids=[record["id"]],
            lesson_snapshots=[], conclusion_ids=[], prior_calls=prior_calls, evidence=evidence)
