"""Offline execution benchmark: equal fake calls, no scientific quality claim.

Run from the checkout with .venv/bin/python scripts/benchmark_parallel_search.py.
Every mode uses the real coordinator, novelty gate and durable persistence, with
controlled transport/evaluator delays and no external requests.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import tempfile
import threading
import time
from unittest.mock import patch

from sciel.engine.factory import build_engine
from sciel.engine.llm.client import LLMResponse
from sciel.engine.llm.mock import MockLLM
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult


class Provider(MockLLM):
    def __init__(self, latency):
        super().__init__()
        self.latency, self.calls, self.active, self.peak = latency, 0, 0, 0
        self.lock = threading.Lock()

    def generate(self, prompt, model, temperature=None, **options):
        with self.lock:
            self.calls += 1
            number = self.calls
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            time.sleep(self.latency)
            return LLMResponse(f'```python\ndef run_search():\n    return ({number}/100, {number}/200)\n```',
                               model, .01, 100, 100, 'stop')
        finally:
            with self.lock:
                self.active -= 1


def benchmark(root, calls=32, latency=.1):
    rows = []
    for mode, workers in [('serial', 1), ('parallel_1', 1), ('parallel_4', 4), ('parallel_8', 8)]:
        with tempfile.TemporaryDirectory(prefix='sciel-throughput-') as directory:
            provider = Provider(latency)
            cfg = dict(task='test/toy_quadratic', budget=calls, max_iterations=calls,
                random_seed=7, num_islands=8, schedule_policy='round_robin', milestone_fracs=[],
                embed_grid=False, embed_niche=False, portfolio_seed=False, dynamic_islands=False,
                idea_island=False, crossover=False, op_fixed='full_rewrite', policy={'enabled':False},
                llm={'mock':True}, candidate_novelty={'enabled':True},
                parallel_search={'enabled':mode!='serial', 'workers':workers,
                    'capacity_dir':str(Path(directory)/'capacity'),
                    'host_llm_slots':8, 'provider_slots':8, 'host_evaluator_slots':2})
            seen, eval_active, eval_peak = [], 0, 0
            lock = threading.Lock()
            def evaluate(task, path, **options):
                nonlocal eval_active, eval_peak
                with lock:
                    seen.append(hashlib.sha256(Path(path).read_bytes()).hexdigest())
                    eval_active += 1
                    eval_peak = max(eval_peak, eval_active)
                try:
                    time.sleep(.02)
                    return CascadeOutcome(EvalResult(ok=True,score=.25,cases={},timing=.02),.25,'full')
                finally:
                    with lock:
                        eval_active -= 1
            engine = build_engine(cfg,str(root/'tasks'),str(Path(directory)/'run'),
                                  python=sys.executable,llm=provider)
            before = resource.getrusage(resource.RUSAGE_SELF)
            started = time.monotonic()
            with patch('sciel.engine.loop.cascade_evaluate',evaluate):
                result = engine.run()
            wall = time.monotonic()-started
            after = resource.getrusage(resource.RUSAGE_SELF)
            run = Path(directory)/'run'
            io = [json.loads(line) for line in (run/'sciel_events.jsonl').read_text().splitlines()
                  if '"io_receipt"' in line]
            row = dict(mode=mode, workers=workers, elapsed_seconds=wall,
                provider_calls=provider.calls, peak_provider_calls=provider.peak,
                unique_evaluated_sources=len(set(seen))-1, evaluator_calls=len(seen)-1,
                peak_evaluators=eval_peak, B_evals=result['B_evals'], known_cost=result['cost_total'],
                unknown_calls=0, evaluations_per_hour=result['B_evals']/wall*3600,
                evaluations_per_fake_dollar=result['B_evals']/result['cost_total'],
                cpu_seconds=(after.ru_utime+after.ru_stime)-(before.ru_utime+before.ru_stime),
                process_lifetime_peak_rss_bytes=after.ru_maxrss if sys.platform=='darwin' else after.ru_maxrss*1024,
                queue_seconds=sum(item['queue_seconds'] for item in io),
                checkpoint_bytes=(run/'checkpoint.json').stat().st_size,
                artifact_bytes=sum(path.stat().st_size for path in run.rglob('*') if path.is_file()),
                duplicate_attempts=0, failures=result.get('n_eval_fail',0))
            assert provider.calls==calls and result['B_evals']==calls and len(set(seen))==calls+1, row
            rows.append(row)
    return {'schema':'sciel-parallel-execution-benchmark/1','scope':'offline synthetic I/O; not LLM or task quality',
        'research_enabled':False,'controlled_transport_seconds':latency,'controlled_evaluator_seconds':.02,
        'calls_per_mode':calls,'rows':rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calls', type=int, default=32)
    parser.add_argument('--output', type=Path,
        help='JSON output path; default: a new external temporary directory')
    args = parser.parse_args(argv)
    if args.calls < 1:
        parser.error('--calls must be positive')
    output = (args.output.expanduser() if args.output else
        Path(tempfile.mkdtemp(prefix='sciel-benchmark-')) / 'parallel_execution.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    result = benchmark(Path(__file__).resolve().parents[1], args.calls)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    print(f'Saved benchmark to {output.resolve()}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
