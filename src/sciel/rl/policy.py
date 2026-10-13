"""Step policy: one KL-bounded improvement over the logging policy for every covered decision of a step.

Returns come from rises of the path best along lineages (returns.py), the critic Q is cross-fitted by island
(critic.py), and a decision moves only when the data support it: enough effective sample size and improving
children, and an out-of-fold estimate of the new policy's value above the logged policy's by two standard errors. It acts only as the
planner arm 'rl', judged like any arm, and stands down when its steps concentrate on fewer parents than 'base'.
"""

from __future__ import annotations

import math

import numpy as np

from . import critic, returns

COVERED = ("inspiration_source", "operator", "strategy", "finding", "origin")
MIN_STEPS = 200
MAX_STEPS = 4000
DELTA = 0.2
CLIP = 1.0
SWITCH = 0.05
MIN_ESS = 200
MIN_IMPROVING = 30
MIN_ARM_STEPS = 20


def _pair(record):
    return (record["decision"], record["choice"]) if isinstance(record, dict) else (record.decision, record.choice)


class StepPolicy:
    def __init__(self):
        self.vocab: list[str] = []
        self.net = None
        self.features = 0
        self.moved: dict[str, float] = {}
        self.version = None
        self.disabled = False
        self.world = None

    @property
    def ready(self) -> bool:
        return self.net is not None and bool(self.moved) and not self.disabled

    def _x(self, engine, pid: str, context: str, earlier: list) -> np.ndarray:
        records = [{"decision": d, "choice": c} for d, c in map(_pair, earlier)]
        return np.concatenate([critic.state(engine, pid, context, self.world), critic.chosen(engine, records)])

    def _value(self, engine, net, index):
        operators = [i for t, i in index.items() if t.startswith("operator:")]

        def value(pid, context):
            if net is None or engine.db._all.get(pid) is None or not operators:
                return 0.0
            x = np.repeat(self._x(engine, pid, context, [])[None], len(operators), 0)
            return float(critic.predict(net, x, np.array(operators)).mean())
        return value

    def _rows(self, engine, all_steps, index):
        rows = []
        for step in all_steps:
            earlier = []
            for decision in COVERED:
                record = step.records.get(decision)
                if record is None:
                    continue
                tokens = {a: index[critic.token(engine, decision, a)] for a in record.distribution}
                rows.append((decision, self._x(engine, step.state, step.context, earlier), tokens, record, step))
                earlier.append(record)
        return rows

    def fit(self, engine, seed: int) -> dict:
        all_steps = [s for s in returns.steps(engine.ledger, COVERED) if s.records][-MAX_STEPS:]
        if len(all_steps) < MIN_STEPS:
            return {"steps": len(all_steps), "moved": {}}
        self.vocab = sorted({critic.token(engine, d, a) for s in all_steps for d, r in s.records.items()
                             for a in r.distribution})
        index = {t: i for i, t in enumerate(self.vocab)}
        net = None
        for _ in range(2):
            returns.returns(all_steps, self._value(engine, net, index), engine.ledger.horizon)
            rows = self._rows(engine, all_steps, index)
            x = np.stack([r[1] for r in rows])
            a = np.array([r[2][r[3].choice] for r in rows])
            y = np.array([r[4].y for r in rows], dtype=np.float32)
            net = critic.fit(x, a, y, len(self.vocab), seed)
        self.features = x.shape[1]
        odd = np.array([r[4].island % 2 for r in rows], dtype=bool)
        folds = {False: critic.fit(x[odd], a[odd], y[odd], len(self.vocab), seed + 1) if odd.any() else net,
                 True: critic.fit(x[~odd], a[~odd], y[~odd], len(self.vocab), seed + 2) if (~odd).any() else net}
        self.net, self.moved, self.disabled = net, {}, False
        self.version = engine.iteration_no
        report = {}
        for decision in COVERED:
            mine = [(i, r) for i, r in enumerate(rows) if r[0] == decision and len(r[2]) > 1]
            if not mine:
                continue
            advantages, mus, chosen, ys, gains = [], [], [], [], []
            for i, (_, xi, tokens, record, step) in mine:
                actions = list(tokens)
                q = critic.predict(folds[bool(odd[i])], np.repeat(xi[None], len(actions), 0),
                                   np.array([tokens[b] for b in actions]))
                mu = np.array([record.distribution[b] for b in actions])
                advantages.append(q.mean(1) - q.std(1) - (mu * q.mean(1)).sum())
                mus.append(mu)
                chosen.append(actions.index(record.choice))
                ys.append(step.y)
                gains.append(step.gain)
            eta, kl = self._eta(advantages, mus)
            weights = np.array([self._tilt(m, adv, eta)[c] / m[c] for adv, m, c in zip(advantages, mus, chosen)])
            ys = np.array(ys)
            ess = weights.sum() ** 2 / max((weights ** 2).sum(), 1e-12)
            new_value = float((weights * ys).sum() / max(weights.sum(), 1e-12))
            base_value = float(ys.mean())
            spread = float(np.std(weights / weights.mean() * ys - ys) / math.sqrt(len(ys)))
            improving = int((np.array(gains) > 0).sum())
            passed = bool(ess >= MIN_ESS and improving >= MIN_IMPROVING and new_value - base_value > 2 * spread)
            if passed:
                self.moved[decision] = eta
            report[decision] = {"rows": len(mine), "eta": eta, "kl": kl, "ess": float(ess), "improving": improving,
                                "new_value": new_value, "base_value": base_value, "spread": spread, "moved": passed}
        return {"steps": len(all_steps), "moved": dict(self.moved), "decisions": report}

    @staticmethod
    def _tilt(mu: np.ndarray, advantage: np.ndarray, eta: float) -> np.ndarray:
        tilted = mu * np.exp(np.clip(advantage / eta, -CLIP, CLIP))
        return tilted / tilted.sum()

    def _eta(self, advantages: list, mus: list) -> tuple[float, float]:
        """Smallest temperature whose mean KL to the logging policy stays within DELTA / len(COVERED)."""
        budget = DELTA / len(COVERED)

        def kl(eta):
            return float(np.mean([(p * np.log(p / m)).sum() for adv, m in zip(advantages, mus)
                                  for p in [self._tilt(m, adv, eta)]]))
        low, high = math.log(1e-4), math.log(1e4)
        if kl(math.exp(low)) <= budget:
            return math.exp(low), kl(math.exp(low))
        for _ in range(40):
            mid = (low + high) / 2
            if kl(math.exp(mid)) > budget:
                low = mid
            else:
                high = mid
        return math.exp(high), kl(math.exp(high))

    def steer(self, engine, decision: str, pid: str, context: str, distribution: dict, earlier: list) -> dict:
        if not self.ready or decision not in self.moved or len(distribution) < 2:
            return dict(distribution)
        index = {t: i for i, t in enumerate(self.vocab)}
        tokens = [critic.token(engine, decision, b) for b in distribution]
        if any(t not in index for t in tokens):
            return dict(distribution)
        x = self._x(engine, pid, context, [r for r in earlier if _pair(r)[0] in COVERED])
        q = critic.predict(self.net, np.repeat(x[None], len(tokens), 0), np.array([index[t] for t in tokens]))
        mu = np.array(list(distribution.values()), dtype=float)
        advantage = q.mean(1) - q.std(1) - (mu * q.mean(1)).sum()
        return {a: float(p) for a, p in zip(distribution, self._tilt(mu, advantage, self.moved[decision]))}

    def check(self, ledger) -> bool:
        """Stand down when the arm's steps since the last fit rest on fewer distinct parents than base steps."""
        if self.version is None or self.disabled:
            return False
        parents = {"rl": [], "base": []}
        for row in ledger.rows:
            if row.decision == "planner" and row.iteration > self.version and row.choice in parents:
                parents[row.choice].append(row.state)
        if len(parents["rl"]) < MIN_ARM_STEPS or len(parents["base"]) < MIN_ARM_STEPS:
            return False

        def effective(ids):
            counts = np.unique(ids, return_counts=True)[1] / len(ids)
            return float(np.exp(-(counts * np.log(counts)).sum()) / len(ids))
        self.disabled = effective(parents["rl"]) < effective(parents["base"])
        return self.disabled

    def state(self) -> dict:
        return {"vocab": self.vocab, "features": self.features, "moved": self.moved, "version": self.version,
                "disabled": self.disabled, "weights": critic.dump(self.net) if self.net is not None else None}

    def load_state(self, data: dict) -> None:
        self.vocab, self.features = list(data["vocab"]), int(data["features"])
        self.moved, self.version, self.disabled = dict(data["moved"]), data["version"], bool(data["disabled"])
        self.net = critic.load(data["weights"], self.features, len(self.vocab)) if data["weights"] else None
