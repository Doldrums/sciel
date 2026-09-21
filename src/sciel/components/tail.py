"""run_sciel_tail: C2 bet resolution + branch calibration, C3 postmortem + lift.

Whole tail is gated by the c2_event_bets flag; with the flag off it never runs, so the
journal is byte-identical to the pre-flag GOLDEN (P4).
"""

from __future__ import annotations

import re

from .kb.kb import KnowledgeBase
from ..campaign.budget import CampaignStopped
from ..engine.llm.client import accounted_cost

_LESSON_TEXT = re.compile(r"LESSON:\s*(?P<text>.+)", re.IGNORECASE)
_MECH = re.compile(r"MECHANISM:\s*(?P<mech>.+)", re.IGNORECASE)

_POSTMORTEM_PROMPT = (
    "An improvement event occurred: a child program beat the island best.\n\n"
    "PARENT:\n```python\n{parent}\n```\n\nCHILD:\n```python\n{child}\n```\n\n"
    "Claimed mechanism: {mechanism}\n\n"
    "State one reusable conditional lesson about when this kind of change causes a jump. "
    "Answer on two lines exactly:\nLESSON: <one sentence>\nMECHANISM: <short phrase>\n")


class BranchCalibration:
    def __init__(self):
        self.brier_sum: dict = {}
        self.count: dict = {}

    def note(self, parent_id: str, brier: float) -> None:
        self.brier_sum[parent_id] = self.brier_sum.get(parent_id, 0.0) + brier
        self.count[parent_id] = self.count.get(parent_id, 0) + 1

    def mean(self, parent_id: str) -> float | None:
        n = self.count.get(parent_id, 0)
        return self.brier_sum[parent_id] / n if n else None

    def state(self) -> dict:
        return {"brier_sum": dict(self.brier_sum), "count": dict(self.count)}

    def load_state(self, d: dict) -> None:
        self.brier_sum = dict(d["brier_sum"])
        self.count = dict(d["count"])


class BaseRate:
    def __init__(self):
        self.children: dict = {}
        self.events: dict = {}

    def rate(self, island: int, phase: int) -> float:
        key = (island, phase)
        n = self.children.get(key, 0)
        return self.events.get(key, 0) / n if n else 0.0

    def observe(self, island: int, phase: int, is_event: bool) -> None:
        key = (island, phase)
        self.children[key] = self.children.get(key, 0) + 1
        self.events[key] = self.events.get(key, 0) + (1 if is_event else 0)

    def state(self) -> dict:
        return {"children": {f"{k[0]},{k[1]}": v for k, v in self.children.items()},
                "events": {f"{k[0]},{k[1]}": v for k, v in self.events.items()}}

    def load_state(self, d: dict) -> None:
        self.children = {_key(k): v for k, v in d["children"].items()}
        self.events = {_key(k): v for k, v in d["events"].items()}


def _key(s: str):
    a, b = s.split(",")
    return (int(a), int(b))


def extract_lesson(llm, model, parent_code: str, child_code: str, mechanism: str):
    prompt = _POSTMORTEM_PROMPT.format(parent=parent_code[:2000], child=child_code[:2000],
                                       mechanism=mechanism or "(none stated)")
    try:
        resp = llm.generate(prompt, model, temperature=0.3)
    except CampaignStopped:
        raise
    except Exception as exc:
        from ..engine.worker import unknown_request_stop
        stopped = unknown_request_stop(exc, 0.0)
        stopped.stage = 'postmortem'
        raise stopped from None
    try:
        cost = accounted_cost(resp)
    except ValueError:
        stopped = CampaignStopped("postmortem response cost is unknown or invalid")
        stopped.known_cost = 0.0
        stopped.current_call_cost_unknown = True
        stopped.stage = "postmortem"
        raise stopped from None
    if getattr(resp, 'response_error', None) is not None or type(resp.text) is not str:
        return None, cost
    mt = _LESSON_TEXT.search(resp.text)
    if not mt:
        return None, cost
    text = mt.group("text").splitlines()[0].strip()
    mm = _MECH.search(resp.text)
    mech = mm.group("mech").splitlines()[0].strip() if mm else (mechanism or text)
    return (text, mech), cost
