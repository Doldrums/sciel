"""First-episode cadence and exact replay of committed research waves."""
from copy import deepcopy
import json

import pytest

from sciel.campaign.budget import CampaignStopped
from sciel.engine.loop import Engine
from sciel.research_tools.controller import PROMPT_MARKER
from test_research_attempt_stall import fixture, intents


def test_first_episode_after_healthy_visits_replays_same_plan_and_charges_once(tmp_path, monkeypatch):
    build, provider, evaluations, executor, directory = fixture(tmp_path, monkeypatch,
        workers=2, max_iterations=6, threshold=4, interval=4)
    commit = Engine._commit_step
    crashed = False
    def crash(self, plan, *args, **kwargs):
        nonlocal crashed
        commit(self, plan, *args, **kwargs)
        if plan['snap'].iteration_no == 4 and not crashed:
            crashed = True
            raise RuntimeError('fixture crash after first-cell commit')
    monkeypatch.setattr(Engine, '_commit_step', crash)
    with pytest.raises(CampaignStopped, match='first-cell commit'):
        build().run()
    frozen = deepcopy(intents(directory))
    assert [j['snapshot'].get('tool_episode_plan') is not None for j in frozen] == [False]*4+[True]*2
    plans = [j['snapshot']['tool_episode_plan'] for j in frozen[4:]]
    assert all(p['opportunity']==1 and p['last_episode_opportunity'] is None
               and p['cadence_reason']=='first_episode' for p in plans)
    baseline=json.loads((directory/'checkpoint.json').read_text())
    assert baseline['iteration_no']==4 and baseline['research_tools']['last_episode_opportunity']=={}
    assert baseline['research_tools']['opportunities']=={str(i):1 for i in range(4)}
    before=(len(provider.calls),len(evaluations),len(executor.calls))
    resumed=build(resume=True); result=resumed.run()
    assert intents(directory)==frozen
    assert (len(provider.calls),len(evaluations),len(executor.calls))==before
    assert result['cost_total']==pytest.approx(.22)  # Six G calls + two tool skip calls.
    assert sum(p.startswith(PROMPT_MARKER) for p in provider.calls)==2
    assert executor.calls==[]
    assert resumed.research_tools.state()['last_episode_opportunity']=={'0':1,'1':1}
