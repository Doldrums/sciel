"""Scripted acceptance: real Docker CPU tools + canonical evaluator, no provider.

This is deliberately a technical fixture, not an autonomous discovery or a
scientific performance result. The fixed proposal source includes a known bug
to exercise repair, then shares the repaired tool across ordinary Engine islands.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from sciel.engine import checkpoint
from sciel.engine.factory import build_engine
from sciel.engine.llm.client import LLMResponse
from sciel.research_tools.controller import PROMPT_MARKER


TOOL = '''import ast
import hashlib
def run(payload):
    # Authored fixture objective, matching the explicitly identified toy task.
    points = [(-2.2 + i / 100, .2 + j / 100) for i in range(101) for j in range(101)]
    best = min(points, key=lambda p: (p[0]+1.704)**2 + (p[1]-.678)**2)
    return {"marker":"CPU_GRID_FINISHED", "best":list(best), "samples":len(points),
        "parent_ast_nodes":sum(1 for _ in ast.walk(ast.parse(payload["parent_source"]))),
        "parent_hash":hashlib.sha256(payload["parent_source"].encode()).hexdigest()}
'''
BROKEN = 'def run(payload):\n    raise ValueError("SCRIPTED_FIXTURE_REPAIR_REQUIRED")\n'


class ScriptedProvider:
    models = ['scripted-technical-fixture']
    weights = [1.0]

    def __init__(self):
        self.tool_requests = self.program_requests = 0
        self.delivered_findings = 0

    def choose_model(self, rng):
        return self.models[0]

    def generate(self, prompt, model, temperature=None, **options):
        if prompt.startswith(PROMPT_MARKER):
            self.tool_requests += 1
            header = json.loads(prompt.splitlines()[1])
            visible = []
            for block in prompt.split('VISIBLE TOOL:\n')[1:]:
                visible.append(json.loads(block.split('\n\n', 1)[0]))
            repaired = [row for row in visible if 'CPU_GRID_FINISHED' in row.get('code', row.get('source', ''))]
            if repaired:
                action = dict(action='use', tool_id=repaired[0]['tool_id'], arguments_json='{}')
            elif 'SCRIPTED_FIXTURE_REPAIR_REQUIRED' in prompt:
                action = dict(action='create', name='CPU grid probe', description='Scripted repaired CPU diagnostic.',
                    code=TOOL, parents=[visible[-1]['tool_id']], dependencies=[], arguments_json='{}')
            elif visible:
                selected = next((row for row in visible if row['author_island_id'] != header['island_id']), visible[0])
                action = dict(action='use', tool_id=selected['tool_id'], arguments_json='{}')
            else:
                action = dict(action='create', name='CPU grid probe', description='Scripted CPU fixture; no discovery claim.',
                    code=BROKEN,
                    parents=[], dependencies=[], arguments_json='{}')
            text = json.dumps(action)
        else:
            self.program_requests += 1
            marker = 'RESEARCH_CHECK_RESULTS_JSON:\n'
            x, y = 10.0, 0.0
            if marker in prompt:
                cards, _ = json.JSONDecoder().raw_decode(prompt.split(marker, 1)[1])
                for card in cards:
                    excerpt = card['observations']['output_excerpt']
                    if 'CPU_GRID_FINISHED' in excerpt:
                        finding = json.loads(excerpt)['output']
                        self.delivered_findings += 1
                        # Deliberately inferior candidates keep the known plateau
                        # active while this fixture tests method transitions.
                        x, y = finding['best'][0] + 12, finding['best'][1]
                        break
            # A deterministic, semantically distinct fixture candidate per prompt;
            # comments alone must not get through the whole-run AST novelty gate.
            perturbation = int(hashlib.sha256(prompt.encode()).hexdigest()[:6], 16) / 2**24 / 1000
            text = f'```python\ndef run_search():\n    return ({x+perturbation!r}, {y!r})\n```'
        return LLMResponse(text, model, 0.0, len(prompt)//4, len(text)//4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-id', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=1, choices=(1, 2))
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error('output must be a fresh path; historical experiments are never overwritten')
    cfg = dict(task='test/toy_quadratic', random_seed=7, budget=None, max_iterations=12,
        num_islands=2, max_islands=2, dynamic_islands=False, portfolio_seed=False,
        embed_grid=False, embed_niche=False, idea_island=False, crossover=False,
        op_fixed='full_rewrite', schedule_policy='round_robin', schedule_horizon=32,
        milestone_fracs=[], policy={'enabled':False}, refine_stall=1,
        llm={'mock':True, 'max_retries':1, 'system_message':
            'Technical fixture: improve run_search() -> (x,y). Explicit objective: '
            'minimize (x+1.704)^2 + (y-0.678)^2. This is not a discovery experiment.'},
        candidate_novelty={'enabled':True, 'max_resamples':2, 'neighbors':3},
        parallel_search={'enabled':True, 'workers':args.workers,
            'host_llm_slots':12, 'provider_slots':12, 'host_evaluator_slots':2,
            'max_inflight_per_island':1, 'max_inflight_per_parent':1},
        research_tools={'enabled':True, 'interval':1, 'max_requests':2, 'image_id':args.image_id,
            'limits':{'wall_seconds':15, 'cpu_seconds':5}})
    provider = ScriptedProvider()
    engine = build_engine(cfg, str(ROOT/'tasks'), str(output), python=sys.executable, llm=provider)
    try:
        engine._stop_at = 6
        first = engine.run()
        assert first['B_evals'] == 6
        before = engine.research_tools.state()
    finally:
        engine.close()
    resumed = build_engine(cfg, str(ROOT/'tasks'), str(output), python=sys.executable, llm=provider)
    try:
        checkpoint.load(resumed, str(output/'checkpoint.json'))
        assert resumed.research_tools.state() == before
        result = resumed.run()
        final = resumed.research_tools.state()
    finally:
        resumed.close()
    # Authenticate the completed replay/adoption/later-use chain on a cold load,
    # not only the earlier checkpoint between active research stages.
    verified = build_engine(cfg, str(ROOT/'tasks'), str(output), python=sys.executable, llm=provider)
    try:
        checkpoint.load(verified, str(output/'checkpoint.json'))
        assert verified.research_tools.state() == final
    finally:
        verified.close()
    rows = [json.loads(line) for line in (output/'sciel_events.jsonl').read_text().splitlines()]
    tool_rows = [row for row in rows if row.get('event') == 'research_tool']
    invoked = [row for row in tool_rows if row.get('phase') == 'invocation_result' and row.get('dispatch_status') == 'invoked']
    foreign = [row for row in invoked if row['island_id'] != row['author_island_id']]
    assert result['B_evals'] == 12 and result['iteration_no'] == 12
    assert foreign, 'no actual cross-island execution was observed'
    assert any(row['status'] == 'failed' for row in invoked), 'known fixture repair branch not exercised'
    assert len(final['published']) >= 1
    assert any(row['phase']=='adopted' for row in tool_rows)
    assert any(row['phase']=='post_adoption_use' for row in tool_rows)
    assert provider.delivered_findings > 0, 'successful CPU observation never reached ordinary generation'
    assert all(row['resources']['cpu_seconds'] >= 0 for row in invoked)
    report = {'schema':'sciel-island-tools-technical-acceptance/1',
        'classification':'scripted technical fixture; not scientific benefit or autonomous discovery',
        'output':str(output), 'workers':args.workers, 'provider_requests':0,
        'scripted_tool_responses':provider.tool_requests, 'scripted_program_responses':provider.program_requests,
        'generations_with_tool_findings':provider.delivered_findings,
        'canonical_program_evaluations':result['B_evals'], 'tool_invocations':len(invoked),
        'foreign_tool_invocations':len(foreign), 'published_versions':len(final['published']),
        'cells':len(final['lifecycle']['cells']['cells']),
        'adoptions':sum(row['phase']=='adopted' for row in tool_rows),
        'post_adoption_uses':sum(row['phase']=='post_adoption_use' for row in tool_rows),
        'known_failed_tool_calls':sum(row['status']=='failed' for row in invoked),
        'tool_cpu_seconds':sum(row['resources']['cpu_seconds'] for row in invoked),
        'child_links':sum(row.get('phase')=='child_link' for row in tool_rows),
        'checkpoint_resume':True, 'final_lifecycle_cold_restore':True, 'pass':True}
    (output/'technical-acceptance.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
