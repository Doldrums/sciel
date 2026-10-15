"""How the engine itself is working: start, randomized decisions, bandit, schedule, controller, RL step policy,
search world model, niches, island lifecycle and the theory's experiments, folded from the journal and checkpoint."""

from __future__ import annotations

from collections import Counter, defaultdict

KEEP = 12


class EngineView:
    def __init__(self):
        self.start = {"warm_start": None, "inherit": None, "portfolio": None, "epsilon": None, "schedule": None,
                      "seed_programs": 0, "portfolio_designs": 0}
        self.choices = defaultdict(Counter)
        self.last_distribution = {}
        self.propensity = defaultdict(list)
        self.mode = {}
        self.mode_history = []
        self.intensify = 0
        self.plateaus = 0
        self.niches = []
        self.archive = None
        self.swm_fits = []
        self.policy_fits = []
        self.stand_downs = 0
        self.spawns = []
        self.experiments = []
        self.procedures = []
        self.strategies = []
        self.trials = Counter()
        self.forecasts = []
        self.confirmations = Counter()
        self.checkpoint_state = {}

    def fold(self, row):
        event, policy, action = row.get("event"), row.get("policy"), row.get("action")
        if event == "run_start":
            cfg = row.get("config") or {}
            warm = cfg.get("warm_start")
            self.start.update(warm_start=(warm or {}).get("source_run") or (warm and "a previous run"),
                              inherit=cfg.get("inherit"), portfolio=cfg.get("portfolio_seed"),
                              epsilon=cfg.get("decision_epsilon"), schedule=cfg.get("schedule_policy"))
        elif event == "iteration":
            context = row.get("decision_context")
            for d in (context.get("decisions", []) if isinstance(context, dict) else []):
                name = d.get("decision")
                self.choices[name][str(d.get("choice"))] += 1
                self.last_distribution[name] = d.get("distribution")
                if isinstance(d.get("propensity"), (int, float)):
                    self.propensity[name].append(d["propensity"])
        elif policy == "initialization" and action == "seed_program":
            self.start["seed_programs"] += 1
        elif policy == "portfolio_seed":
            self.start["portfolio_designs"] += 1
        elif policy == "controller":
            self.mode[row.get("island")] = {"mode": action, "distribution": row.get("mode_dist"),
                                            "plateau": row.get("plateau"), "iteration": row.get("iteration_no")}
            self.mode_history.append([row.get("iteration_no"), row.get("island"), action, bool(row.get("plateau"))])
            del self.mode_history[:-2000]
        elif policy == "refine" and action == "intensify":
            self.intensify += 1
        elif policy == "opportunity_cost" and action == "plateau":
            self.plateaus += 1
        elif policy == "niches":
            self._keep(self.niches, {"iteration": row.get("iteration_no"), "space": row.get("space"),
                                     "niches": row.get("niches")})
        elif policy == "memory" and action == "rebuild":
            self.archive = {"iteration": row.get("iteration_no"), "cells": row.get("cells"),
                            "elites": len(row.get("elites") or [])}
        elif policy == "swm":
            self._keep(self.swm_fits, {"iteration": row.get("iteration_no"), "fitted": action == "fit",
                                       "actions": row.get("actions"),
                                       **{k: row[k] for k in ("rank", "tau", "passed", "reason") if k in row}})
        elif policy == "step_policy":
            if action == "stand_down":
                self.stand_downs += 1
            else:
                self._keep(self.policy_fits, {"iteration": row.get("iteration_no"), "steps": row.get("steps"),
                                              "moved": row.get("moved") or {}, "reason": row.get("reason")})
        elif policy in ("dynamic_islands", "spawn_target") and row.get("island") is not None:
            entry = next((s for s in self.spawns if s["island"] == row["island"]), None)
            if entry is None:
                entry = {"island": row["island"], "iteration": row.get("iteration_no"), "stall": None, "target": None}
                self.spawns.append(entry)
            if policy == "dynamic_islands":
                entry["stall"] = row.get("stall")
            else:
                entry["target"] = action
        elif policy == "experiment":
            if action == "start":
                self._keep(self.experiments, {"hypothesis": row.get("hypothesis"), "outcome": "running",
                                              "iteration": row.get("iteration_no"), "length": row.get("length"),
                                              "required": row.get("required")})
            elif action == "finish":
                for e in reversed(self.experiments):
                    if e["hypothesis"] == row.get("hypothesis") and e["outcome"] == "running":
                        e["outcome"] = row.get("outcome")
                        break
                else:
                    self._keep(self.experiments, {"hypothesis": row.get("hypothesis"), "outcome": row.get("outcome"),
                                                  "iteration": row.get("iteration_no")})
        elif policy == "procedure" and action == "propose":
            self._keep(self.procedures, {"id": row.get("procedure"), "parent": row.get("parent"),
                                         "params": row.get("params")})
        elif policy == "strategy" and action == "propose":
            self._keep(self.strategies, {"id": row.get("strategy"), "text": " ".join(str(row.get("text")).split())[:240]})
        elif policy == "trial":
            self.trials[action] += 1
        elif policy == "forecast" and action == "scored":
            self.forecasts.append((row.get("brier"), row.get("covered"), row.get("iteration_no")))
        elif policy == "confirmation":
            self.confirmations[action] += 1

    @staticmethod
    def _keep(rows, row):
        rows.append(row)
        del rows[:-KEEP]

    def checkpoint(self, state: dict):
        bandit = state.get("op_bandit") or {}
        schedule = state.get("schedule") or {}
        swm = state.get("swm") or {}
        policy = state.get("policy") or {}
        memory = state.get("memory") or {}
        self.checkpoint_state = {
            "bandit": {arm: {"n": len(r), "mean": (sum(r) / len(r)) if r else None}
                       for arm, r in (bandit.get("events") or {}).items()},
            "bandit_settings": {k: bandit.get(k) for k in ("temperature", "floor", "window")},
            "schedule": {"policy": schedule.get("policy"), "survivors": schedule.get("survivors"),
                         "dispatched": schedule.get("dispatched"), "milestones": schedule.get("milestones_passed"),
                         "min_fair": schedule.get("min_fair_evals")},
            "plateau_until": (state.get("controller") or {}).get("plateau_until"),
            "swm": {"trained_at": swm.get("trained_at"), "actions": len(swm.get("vocab") or []),
                    "weights": swm.get("weights") is not None, "uncertain_above": swm.get("uncertain_above"),
                    "cells": len(swm.get("cells") or [])},
            "policy": {"version": policy.get("version"), "disabled": policy.get("disabled"),
                       "moved": policy.get("moved") or {}, "features": policy.get("features")},
            "archive_cells": memory.get("k"),
            "procedures": [{"id": p.get("id"), "status": p.get("status"), "parent": p.get("parent")}
                           for p in (state.get("procedures") or {}).get("pool", [])],
            "experiment": (state.get("experiments") or {}).get("active"),
            "portfolio": state.get("portfolio_bootstrap"),
        }

    def snapshot(self) -> dict:
        briers = [b for b, _, _ in self.forecasts if isinstance(b, (int, float))]
        covered = [c for _, c, _ in self.forecasts if isinstance(c, bool)]
        decisions = []
        for name, counts in sorted(self.choices.items(), key=lambda kv: -sum(kv[1].values())):
            total = sum(counts.values())
            props = self.propensity.get(name) or []
            decisions.append({"decision": name, "total": total, "options": len(counts),
                              "choices": [[c, n / total] for c, n in counts.most_common(6)],
                              "last": self.last_distribution.get(name),
                              "randomized": bool(props) and min(props) < 1.0,
                              "mean_propensity": sum(props) / len(props) if props else None})
        cold = not self.start["warm_start"] and not self.start["inherit"]
        return {"start": {**self.start, "cold": cold}, "decisions": decisions,
                "mode": [{"island": k, **v} for k, v in sorted(self.mode.items(), key=lambda kv: kv[0] or 0)],
                "intensify": self.intensify, "plateaus": self.plateaus, "niches": self.niches,
                "archive": self.archive, "swm_fits": self.swm_fits, "policy_fits": self.policy_fits,
                "stand_downs": self.stand_downs, "spawns": self.spawns, "experiments": self.experiments,
                "procedures_proposed": self.procedures, "strategies": self.strategies, "trials": dict(self.trials),
                "forecasts": {"n": len(self.forecasts), "brier": sum(briers) / len(briers) if briers else None,
                              "coverage": sum(covered) / len(covered) if covered else None,
                              "series": [[i, b, c] for b, c, i in self.forecasts[-400:]]},
                "mode_history": self.mode_history,
                "confirmations": dict(self.confirmations), **self.checkpoint_state}
