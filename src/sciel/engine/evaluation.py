"""Evaluation dispatch, durable measurement and novelty reservations."""
from __future__ import annotations

import hashlib
import os
import tempfile
from ..eval.facade import EvaluationInfrastructureError
from .runtime import observed_stage


class EngineEvaluation:

    def _evaluation_callback(self, island):
        if getattr(self, 'output_novelty', None) is None:
            return self._evaluate
        return lambda code: self._evaluate(code, island_id=island)

    @observed_stage("evaluating")
    def _confirmation_score(self, code: str) -> float | None:
        """One more full measurement of an already evaluated source, outside the novelty registry."""
        from . import loop as _public
        fd, path = tempfile.mkstemp(suffix=".py", prefix="confirm_", dir=self.run_dir)
        with os.fdopen(fd, "w") as f:
            f.write(code)
        try:
            io = getattr(self, '_parallel_io', None)
            backend = self.evaluation_backend
            if backend is not None:
                outcome = io.call('evaluation', {'source': code, 'task_hash': self.task.task_hash,
                    'confirmation': True, 'evaluation_backend': backend.identity},
                    lambda: backend.evaluate(code.encode('utf-8'), evaluation_id='confirmation:' + hashlib.sha256(code.encode()).hexdigest(),
                                             confirm=False))
            elif io is None:
                outcome = _public.cascade_evaluate(self.task, path, python=self.python)
            else:
                outcome = io.call('evaluation', {'source': code, 'task_hash': self.task.task_hash, 'confirmation': True},
                    lambda: _public.cascade_evaluate(self.task, path, python=self.python))
            return outcome.result.score if outcome.result.ok else None
        finally:
            os.unlink(path)

    @observed_stage("evaluating")
    def _evaluate(self, code: str, *, remeasurement: bool = False, island_id=None):
        from . import loop as _public
        registry = getattr(self, 'candidate_novelty', None)
        novelty = registry.inspect(code, remeasurement=remeasurement) if registry else None
        if novelty is not None and not novelty['accept']:
            # Mutation callers must handle this before dispatch and return the
            # rejection to the generator. This guard catches a missed route.
            raise ValueError('candidate novelty was not resolved before evaluator: '
                             + novelty['reason'])
        evaluation_id = None
        if registry is not None:
            io = getattr(self, '_parallel_io', None)
            evaluation_id = (f'evaluation:{self.log.run_id}:wave:{self._parallel_wave}:'
                             f'{io.current}:{io.sequences.get(io.current, 0)}' if io is not None else
                             f'evaluation:{self.log.run_id}:{self.log.segment_id}:{self.log.event_seq + 1}')
            novelty = registry.reserve(code, evaluation_id, remeasurement=remeasurement)
            if not novelty['accept']:
                raise ValueError('candidate novelty reservation refused: ' + novelty['reason'])
            if novelty['remeasurement']:
                self.n_remeasurements += 1
            self.log.decision(policy='candidate_measurement', action='start',
                evaluation_id=evaluation_id, source_sha256=novelty['source_sha256'],
                iteration_no=self.iteration_no, remeasurement=novelty['remeasurement'],
                matched_program_id=novelty.get('matched_program_id'),
                matched_evaluation_id=novelty.get('matched_evaluation_id'))
        if evaluation_id is None:
            io = getattr(self, '_parallel_io', None)
            evaluation_id = (f'evaluation:{self.log.run_id}:wave:{self._parallel_wave}:'
                             f'{io.current}:{io.sequences.get(io.current, 0)}' if io is not None else
                             f'evaluation:{self.log.run_id}:{self.log.segment_id}:{self.log.event_seq + 1}')
        fd, path = tempfile.mkstemp(suffix=".py", prefix="cand_", dir=self.run_dir)
        with os.fdopen(fd, "w") as f:
            f.write(code)
        measurement_returned = False
        try:
            with open(path, "rb") as source:
                source_sha256 = hashlib.sha256(source.read()).hexdigest()
            task, record_candidate, python = self.task, self.cfg.get("confirm_records", False), self.python
            io = getattr(self, '_parallel_io', None)
            backend = self.evaluation_backend
            if backend is not None:
                try:
                    backend.assert_compatible(self.cfg, task)
                except Exception as error:
                    raise EvaluationInfrastructureError('candidate-only evaluation identity changed') from error
                if io is None:
                    raise EvaluationInfrastructureError('candidate_only evaluation requires the durable wave driver')
                request = {'source': code, 'source_sha256': source_sha256, 'task_hash': task.task_hash,
                    'evaluation_id': evaluation_id, 'evaluation_backend': backend.identity,
                    'confirm_records': record_candidate,
                    'output_resolution': self.cfg.get('candidate_novelty', {}).get('output_resolution')}
                outcome = io.call('evaluation', request, lambda: backend.evaluate(code.encode('utf-8'),
                    evaluation_id=evaluation_id, confirm=record_candidate))
            elif io is None:
                outcome = _public.cascade_evaluate(task, path, record_candidate=record_candidate, python=python)
            else:
                outcome = io.call('evaluation', {'source': code, 'task_hash': task.task_hash,
                    'confirm_records': record_candidate,
                    'output_resolution': self.cfg.get('candidate_novelty', {}).get('output_resolution')},
                    lambda: _public.cascade_evaluate(task, path, record_candidate=record_candidate, python=python))
            if registry is not None:
                registry.register(code, evaluation_id, evaluation_ok=bool(outcome.result.ok))
                measurement_returned = True
            outcome.evaluation_id = evaluation_id
            from .research_artifacts import capture_evaluation
            outcome.saved_evaluation_reference = capture_evaluation(self, code, outcome, evaluation_id)
            # Preserve the returned cascade result before admission/seed handling.
            # This is not a trace of every intermediate smoke/full/confirm stage.
            # No truncation: task diagnostics may contain nested JSON and long text.
            self.log.decision(
                policy="evaluation", action="result", schema="sciel-evaluation-result/1",
                source_sha256=source_sha256, iteration_no_before=self.iteration_no,
                evaluation_id=evaluation_id,
                **({'remeasurement': novelty['remeasurement']} if registry is not None else {}),
                B_evals_before=self.B_evals, **outcome.result.to_json(),
                raw_first_score=outcome.raw_first_score, reached=outcome.reached,
                confirmed=outcome.confirmed,
                **({'execution': outcome.execution} if getattr(outcome, 'execution', None) is not None else {}))
            self.n_evaluation_results += 1
            outputs = getattr(self, 'output_novelty', None)
            if outputs is not None:
                from ..eval.output_identity import adapt_output
                binding = {'source_sha256': source_sha256, 'task_hash': task.task_hash,
                           'island_id': island_id}
                reference = outcome.saved_evaluation_reference
                if reference is not None:
                    binding['receipt_sha256'] = reference['sha256']
                identity = adapt_output(task, source_sha256, outcome, binding)
                from ..eval.output_identity import adapt_near_output
                measurement = adapt_near_output(task, source_sha256, outcome, binding,
                    identity=identity, policy=self.cfg['candidate_novelty'].get('output_resolution'))
                verdict = outputs.observe(evaluation_id, identity, binding, measurement=measurement)
                outcome.output_novelty = verdict
                from .output_novelty import is_repeat
                repeated = is_repeat(verdict)
                self.n_output_repeats += int(repeated)
                self.log.decision(policy='output_novelty',
                    action='recorded_repeat' if verdict['status'] == 'exact_recorded_repeat' else verdict['status'],
                    source_sha256=source_sha256,
                    remeasurement=bool(novelty and novelty['remeasurement']),
                    n_evaluation_results=self.n_evaluation_results,
                    n_output_repeats=self.n_output_repeats, **verdict)
            return outcome
        except Exception as exc:
            if registry is not None and not measurement_returned:
                unknown = (isinstance(exc, EvaluationInfrastructureError)
                           and exc.execution_status != 'not_dispatched')
                # Unknown execution cannot release the reservation for another
                # evaluation. This remains pending across checkpoint restore.
                if not unknown:
                    registry.register(code, evaluation_id, evaluation_ok=False)
                self.log.decision(policy='candidate_measurement', action='unknown' if unknown else 'failed',
                    evaluation_id=evaluation_id, source_sha256=novelty['source_sha256'],
                    remeasurement=novelty['remeasurement'])
            raise
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
