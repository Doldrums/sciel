"""Bounded research memory with authenticated evidence and detached public views."""
from __future__ import annotations

import copy
import json
import re
from ..hypothesis.observed_facts import observed_comparison
from .paired_comparisons import validate_comparison
from .eligibility import output_repeat, admission_view
from .catalogs import canonical_json
from .records import (
    INITIALIZER_REGISTRY_SCHEMA,
    SCHEMA,
    _digest,
    _require,
    _text,
)

from . import initializer_memory, reflection_selection, lesson_updates, knowledge_restore
from .records import _experiment, _snapshot, _number


class KnowledgeBase:
    def __init__(self, max_lessons=50, max_experiments=200, max_fragments=20):
        for value in (max_lessons, max_experiments, max_fragments):
            _require(type(value) is int and value > 0, "knowledge limits must be positive integers")
        self.max_lessons, self.max_experiments, self.max_fragments = max_lessons, max_experiments, max_fragments
        self._experiments, self._lessons, self._fragments = [], [], []
        self._seen = {}
        self._comparisons, self._seen_comparisons = [], {}
        self._initializer_evidence = {}
        self._initializer_seen = {}
        self._generated_paired_history = None
        self._generated_paired_review = None
        self._evaluated_detached_study = None
        self._evaluated_study_review = None
        self._next_lesson_id = 1

    @property
    def experiments(self):
        return copy.deepcopy(self._experiments)

    @property
    def lessons(self):
        return copy.deepcopy(self._lessons)

    @property
    def fragments(self):
        return copy.deepcopy(self._fragments)

    @property
    def comparisons(self):
        return copy.deepcopy(self._comparisons)

    @property
    def generated_paired_history(self):
        return copy.deepcopy(self._generated_paired_history)

    @property
    def generated_paired_review(self):
        return copy.deepcopy(self._generated_paired_review)

    @property
    def evaluated_detached_study(self):
        """Portable evidence, dependencies and authority; no native record."""
        return copy.deepcopy(self._evaluated_detached_study)

    @property
    def evaluated_study_review(self):
        return copy.deepcopy(self._evaluated_study_review)

    def import_evaluated_study_review(self, sidecar, reflection_config, *, expected_sha256):
        from ..engine.evaluated_study_reflection import validate_sidecar
        _require(self._evaluated_detached_study is not None, 'review requires its source study')
        checked = validate_sidecar(sidecar, bundle=self._evaluated_detached_study,
            expected_sha256=expected_sha256, config=reflection_config)
        retained = dict(sidecar=checked, reflection_config=copy.deepcopy(reflection_config))
        if self._evaluated_study_review is not None:
            _require(canonical_json(self._evaluated_study_review) == canonical_json(retained),
                     'different evaluated study review already imported')
            return False
        self._evaluated_study_review = retained
        return True

    def import_evaluated_detached_study(self, capsule, dependencies, authority, *,
                                       expected_sha256, metric, root=None):
        from .evaluated_detached_study import validate_capsule
        validated = validate_capsule(capsule, expected_sha256=expected_sha256,
            metric=metric, root=root, author_dependencies=dependencies, author_authority=authority)
        bundle = dict(capsule=validated, author_dependencies=copy.deepcopy(dependencies),
                      author_authority=copy.deepcopy(authority))
        if self._evaluated_detached_study is not None:
            _require(canonical_json(self._evaluated_detached_study) == canonical_json(bundle),
                     "different detached study already imported")
            return False
        self._evaluated_detached_study = bundle
        return True

    def import_generated_paired_review(self, sidecar, *, expected_sha256):
        from ..engine.generated_paired_reflection import validate_sidecar
        _require(self._generated_paired_history is not None, 'review requires its source component')
        validated = validate_sidecar(sidecar, component=self._generated_paired_history,
                                     expected_sha256=expected_sha256)
        if self._generated_paired_review is not None:
            _require(canonical_json(self._generated_paired_review) == canonical_json(validated),
                     'different generated paired review already imported')
            return False
        self._generated_paired_review = validated
        return True

    def import_generated_paired_history(self, capsule, *, expected_sha256, metric, root=None):
        from .generated_paired_history import validate_capsule
        validated = validate_capsule(capsule, expected_sha256=expected_sha256, metric=metric, root=root)
        if self._generated_paired_history is not None:
            _require(canonical_json(self._generated_paired_history) == canonical_json(validated),
                     'different generated paired history already imported')
            return False
        self._generated_paired_history = validated
        return True

    def _initializer_observations(self):
        return initializer_memory._initializer_observations(self)

    def _age_initializer_evidence(self):
        return initializer_memory._age_initializer_evidence(self)

    def _initializer_binding(self, value):
        return initializer_memory._initializer_binding(self, value)

    def register_initializer_evidence(self, observation_id, evidence, prior_observation_id=None):
        return initializer_memory.register_initializer_evidence(self, observation_id, evidence, prior_observation_id)

    def initializer_evidence_for(self, observation_id):
        return initializer_memory.initializer_evidence_for(self, observation_id)

    def initializer_comparisons_for(self, observation_id):
        return initializer_memory.initializer_comparisons_for(self, observation_id)

    def initializer_context(self, observation_ids, *, max_items=3, max_chars=12000):
        return initializer_memory.initializer_context(self, observation_ids, max_items=max_items, max_chars=max_chars)

    def record_comparison(self, group):
        """Import one terminal paired batch, without inventing attempts/exposures.

        Repeat journal IDs have a separate lifetime digest index. Replay is a
        no-op even after the bounded window evicts its full record; a changed
        payload with the same ID fails before any state mutation.
        """
        group = validate_comparison(group)
        ident, digest = group["id"], _digest(group)
        _require(ident not in self._seen, "comparison ID conflicts with an experiment")
        if ident in self._seen_comparisons:
            _require(self._seen_comparisons[ident] == digest, "conflicting comparison ID")
            return False
        self._seen_comparisons[ident] = digest
        self._comparisons = (self._comparisons + [group])[-self.max_experiments:]
        return True

    def comparison_context(self, branch, *, max_items=3, max_chars=16000):
        """Admit complete recent branch cards and explicitly name omissions."""
        from .presentation import comparison_card, omit_source_context, COMPARISON_CARD_SCHEMA
        from .catalogs import canonical_json
        _text(branch, "branch", limit=128)
        self._limit(max_items)
        _require(type(max_chars) is int and max_chars > 0, "invalid comparison context budget")
        eligible = [g for g in self._comparisons if g["branch"] == branch]
        chosen = []

        def block(cards):
            ids = [c["id"] for c in cards]
            omitted = [g["id"] for g in reversed(eligible) if g["id"] not in ids]
            value = {"schema": "sciel-paired-comparison-context/1", "cards": cards,
                     "presented_ids": ids, "omitted_ids": omitted[:16],
                     "omitted_count": len(omitted), "omitted_ids_truncated": len(omitted) > 16,
                     "retained_count": len(eligible), "rendered_chars": 0}
            while value["rendered_chars"] != len(canonical_json(value)):
                value["rendered_chars"] = len(canonical_json(value))
            return value

        for group in reversed(eligible):
            if len(chosen) >= max_items:
                break
            card = comparison_card(group)
            trial = chosen + [card]
            if block(trial)["rendered_chars"] <= max_chars:
                chosen = trial
            elif card["schema"] == COMPARISON_CARD_SCHEMA:
                # Preserve the complete numerical observation when optional
                # source text cannot fit. No partial code changes are shown.
                trial = chosen + [omit_source_context(card)]
                if block(trial)["rendered_chars"] <= max_chars:
                    chosen = trial
        result = block(chosen)
        _require(result["rendered_chars"] <= max_chars, "comparison omission metadata exceeds context budget")
        return result

    def record(self, experiment):
        return lesson_updates.record(self, experiment)

    def _evidence(self):
        return lesson_updates._evidence(self)

    def proposal_group_context(self, branch, *, max_items=3, max_chars=12000):
        return reflection_selection.proposal_group_context(self, branch, max_items=max_items, max_chars=max_chars)

    def reflection_context(self, recent_limit, *, max_experiments=64, max_lessons=8, max_chars=90000, selection_epoch=None):
        return reflection_selection.reflection_context(self, recent_limit, max_experiments=max_experiments, max_lessons=max_lessons, max_chars=max_chars, selection_epoch=selection_epoch)

    def legacy_reflection_context(self, recent_limit, *, max_experiments=64, max_lessons=8, max_chars=90000):
        return reflection_selection.legacy_reflection_context(self, recent_limit, max_experiments=max_experiments, max_lessons=max_lessons, max_chars=max_chars)

    def _proposals(self, proposals, allowed_evidence_ids, writer_context=None):
        return lesson_updates._proposals(self, proposals, allowed_evidence_ids, writer_context)

    def add_lessons(self, proposals, *, allowed_evidence_ids, writer_context=None):
        return lesson_updates.add_lessons(self, proposals, allowed_evidence_ids=allowed_evidence_ids, writer_context=writer_context)

    def reflect(self, proposals, retire_ids, *, allowed_evidence_ids, writer_context=None):
        return lesson_updates.reflect(self, proposals, retire_ids, allowed_evidence_ids=allowed_evidence_ids, writer_context=writer_context)

    @staticmethod
    def _limit(limit):
        _require(type(limit) is int and limit >= 0, "limit must be a nonnegative integer")
        return limit

    def retrieve(self, *, branch, context, limit=3):
        _text(branch, "branch", limit=128)
        _require(isinstance(context, str), "context must be text")
        tokens = set(re.findall(r"\w+", context.casefold()))
        candidates = [l for l in self._lessons if l["scope"] == "global" or l["branch"] == branch]
        candidates.sort(key=lambda l: (-len(tokens & set(re.findall(r"\w+", l["rule"].casefold()))),
                                       -l["confidence"], l["id"]))
        return copy.deepcopy(candidates[:self._limit(limit)])

    def failed_hypotheses(self, branch, limit=3):
        _text(branch, "branch", limit=128)
        limit = self._limit(limit)
        if limit == 0:
            return []
        failures = []
        for e in reversed(self._experiments):
            if e["branch"] != branch:
                continue
            # Missing forecasts do not erase unmeasured engineering feedback.
            # A measured observation without a forecast retains its historical
            # behavior: it is not a failed-hypothesis presentation entry.
            if e["hypothesis"] is None and e["status"] == "evaluated" and not output_repeat(e):
                continue
            if output_repeat(e):
                disposition = admission_view(e)['status']
                label = ("Measured non-improving near-output repeat; not admitted and not efficacy evidence"
                         if disposition == 'non_improving_near_repeat' else
                         "Measured exact recorded-output repeat; not admitted and not efficacy evidence")
            elif e["status"] != "evaluated":
                label = f"Unmeasured ({e['status']}); engineering outcome, quality not measured"
            elif e["actual_delta"] < 0:
                prefix = "Forecast refuted; " if e["forecast_verdict"] == "REFUTED" else ""
                label = f"{prefix}measured regression (forecast {e['forecast_verdict']})"
            elif e["forecast_verdict"] == "REFUTED":
                direction = "Improved despite forecast miss" if e["actual_delta"] > 0 else "No measured improvement; forecast miss"
                label = direction
            else:
                continue
            # Keep identity intact while bounding model-generated hypothesis and
            # task error text independently. A forecast miss is not bad fitness.
            identity = f"id={e['id']}; source_sha256={e['source_sha256']}"
            hypothesis_text = (e["hypothesis"]["text"][:512] if e["hypothesis"] is not None
                               else "No valid forecast.")
            text = f"{label} [{identity}]"
            if e["status"] == "evaluated":
                measured = observed_comparison(e)["measurement"]
                evaluation = e.get("evaluation") or {}
                semantics = (e.get("metric") or {}).get("semantics", {})
                validity_case = semantics.get("validity_case")
                facts = {"status": e["status"], "evaluator_ok": evaluation.get("ok"),
                    "task_validity": {"case": validity_case,
                        "value": evaluation.get("cases", {}).get(validity_case) if validity_case is not None else None,
                        "valid_value": semantics.get("valid_value")},
                    "parent_score": measured["parent_score"], "child_score": measured["score"],
                    "actual_delta": measured["actual_delta"]}
                # Exact finite numbers, not rounded prose. An evaluator success
                # flag is separate from the task's declared validity case. Legacy
                # missing observations stay unknown; no other cases/arrays leak.
                text += "; measured_facts=" + json.dumps(facts, separators=(",", ":"), allow_nan=False)
                text += " (null=unknown; evaluator_ok is not task validity)"
                if output_repeat(e):
                    text += "; output_repeat=" + json.dumps({key: e["output_novelty"][key] for key in
                        ("status", "evaluation_id", "match", "sha256")}, separators=(",", ":"))
            text += f": {hypothesis_text}"
            if e["status"] != "evaluated" and e.get("failure"):
                text += f"; evaluator/validation diagnostic (untrusted): {e['failure'][:700]}"
            failures.append(text)
            if len(failures) >= limit:
                break
        return failures[:limit]

    def fragments_for(self, branch, limit=2):
        _text(branch, "branch", limit=128)
        global_ids = {eid for l in self._lessons if l["scope"] == "global" and l["polarity"] == "positive"
                      for eid in l["evidence_ids"]}
        fragments = [f for f in self._fragments if f["branch"] == branch or f["experiment_id"] in global_ids]
        fragments.sort(key=lambda f: (f["branch"] != branch, -f["actual_delta"], f["id"]))
        return copy.deepcopy(fragments[:self._limit(limit)])

    def state(self):
        state = {"schema": SCHEMA, "limits": {"max_lessons": self.max_lessons,
            "max_experiments": self.max_experiments, "max_fragments": self.max_fragments},
            "next_lesson_id": self._next_lesson_id, "seen_experiments": self._seen,
            "experiments": self._experiments, "lessons": self._lessons, "fragments": self._fragments}
        # Preserve historical empty-state digests. Comparisons are a typed
        # extension, never entries in the original attempt ledger.
        if self._seen_comparisons:
            state["paired_comparisons"] = {"seen": self._seen_comparisons, "records": self._comparisons}
        if self._initializer_seen:
            state["initializer_evidence"] = {"schema": INITIALIZER_REGISTRY_SCHEMA,
                "seen": self._initializer_seen, "records": list(self._initializer_evidence.values())}
        if self._generated_paired_history is not None:
            state['generated_paired_history'] = self._generated_paired_history
        if self._generated_paired_review is not None:
            state['generated_paired_review'] = self._generated_paired_review
        if self._evaluated_detached_study is not None:
            state['evaluated_detached_study'] = self._evaluated_detached_study
        if self._evaluated_study_review is not None:
            state['evaluated_study_review'] = self._evaluated_study_review
        return copy.deepcopy(state)

    def load_state(self, state):
        """Validate a complete snapshot before atomically replacing this instance."""
        try:
            self._load_state(state)
        except (KeyError, TypeError, AttributeError, OverflowError) as exc:
            raise ValueError("malformed knowledge state") from exc

    def _load_state(self, state):
        return knowledge_restore._load_state(self, state)
