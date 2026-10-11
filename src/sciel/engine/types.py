"""Shared engine dataclasses passed between controller and worker (P3)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Program:
    id: str
    code: str
    score: float
    cases: dict
    island: int
    origin: str
    parent_id: str | None = None
    gen: int = 0
    sig_hash: str = ""
    card: str = ""            # one-line strategy summary from the LLM's prose


@dataclass
class Snapshot:
    island: int
    op: str
    model: str
    iteration_no: int
    parent_id: str
    parent_code: str
    parent_score: float
    inspirations: list = field(default_factory=list)
    artifacts: dict = field(default_factory=dict)
    directive: str = ""
    kb_lessons: list = field(default_factory=list)
    code_hashes: set = field(default_factory=set)
    escalation_seed: int = 0
    task_prompt: str = ""
    bet_enabled: bool = False
    required_entrypoints: list = field(default_factory=list)
    research_context: dict | None = None
    parent_card: str = ""
    idea_cards: list = field(default_factory=list)
    tool_episode_plan: dict | None = None
    research_cells_context: dict | None = None
    research_targets: dict | None = None
    research_search_evidence: dict | None = None
    operator_decision: dict | None = None


@dataclass
class Bet:
    mechanism: str
    q: float
    delta_low: float | None = None
    delta_high: float | None = None


@dataclass
class Result:
    child_code: str
    score: float
    cases: dict
    ok: bool
    island: int
    parent_id: str
    origin: str
    bet: Bet | None
    meta: dict
    raw_first_score: float
    reached: str
    confirmed: bool
    prompt: str
    response: str
    cost: float
    timing: float
    artifacts: dict = field(default_factory=dict)


@dataclass
class NoChild:
    island: int
    reason: str
    meta: dict
    # Money was spent even though no child came back: an iteration that exhausts
    # dup_cap makes dup_cap LLM calls. Without this field cost_total booked none of
    # them, and score_per_dollar was computed against an understated denominator.
    cost: float = 0.0
