"""Step policy: one KL-bounded improvement over the logging policy for every covered decision of a step.

Returns come from rises of the path best along lineages (returns.py) and the critic Q is fitted twice, once per island
parity, each model scoring the other parity's rows and children, so no estimate rests on its own data. A decision
moves only when the data support it: enough effective sample size and improving children, and an out-of-fold
estimate of the new policy's value above the logged policy's by two standard errors clustered by parent. The policy
that acts is the one the gate judged: the same two models tilt the same logged distribution. It acts only as the
planner arm 'rl', judged like any arm, and stands down for WINDOW iterations when its steps rest on fewer distinct
parents than 'base' steps of the same number.
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
MIN_ESS = 200
MIN_IMPROVING = 30
MIN_ARM_STEPS = 10
WINDOW = 100


def _pair(record):
    if isinstance(record, dict):
        return record["decision"], record["choice"], record.get("standing")
    return record.decision, record.choice, record.standing


class StepPolicy:
    def __init__(self):
        self.vocab: list[str] = []
        self.folds: dict[bool, object] = {}
        self.features = 0
        self.moved: dict[str, float] = {}
        self.version = None
        self.disabled = False
        self.stood_down = None
        self.world = None

    @property
    def ready(self) -> bool:
        return bool(self.folds) and bool(self.moved) and not self.disabled

    def _x(self, engine, pid: str, context: str, earlier: list) -> np.ndarray:
        records = [{"decision": d, "choice": c, "standing": s} for d, c, s in map(_pair, earlier)]
        return np.concatenate([critic.state(engine, pid, context, self.world), critic.chosen(engine, records)])

    def _q(self, net, x: np.ndarray, actions: list) -> np.ndarray:
        return critic.predict(net, np.repeat(x[None], len(actions), 0), np.array(actions))

    def _value(self, engine, folds, index, all_steps):
        """V(child) = sum over operators of the mean logged probability times Q, with the mean features that earlier
        decisions of the step give an operator row, from the model that never saw the child's island parity."""
        operators = [t for t in index if t.startswith("operator:")]
        rows = [s for s in all_steps if "operator" in s.records]
        if not folds or not operators or not rows:
            return lambda pid, context, island: 0.0
        mu = np.array([np.mean([s.records["operator"].distribution.get(t.split(":", 1)[1], 0.0) for s in rows])
                       for t in operators])
        mu = mu / mu.sum() if mu.sum() > 0 else np.full(len(operators), 1 / len(operators))
        prior = COVERED[:COVERED.index("operator")]
        chosen = np.mean([critic.chosen(engine, [{"decision": d, "choice": s.records[d].choice,
                                                  "standing": s.records[d].standing}
                                                 for d in prior if d in s.records]) for s in rows], 0)
        actions = [index[t] for t in operators]

        def value(pid, context, island):
            if engine.db._all.get(pid) is None:
                return 0.0
            x = np.concatenate([critic.state(engine, pid, context, self.world), chosen]).astype(np.float32)
            return float(mu @ self._q(folds[bool(island % 2)], x, actions).mean(1))
        return value

    def _rows(self, engine, all_steps, index):
        rows = []
        for step in all_steps:
            earlier = []
            for decision in COVERED:
                record = step.records.get(decision)
                if record is None:
                    continue
                tokens = {a: index[critic.token(engine, decision, a, record.standing)] for a in record.distribution}
                rows.append((decision, self._x(engine, step.state, step.context, earlier), tokens, record, step))
                earlier.append(record)
        return rows

    def fit(self, engine, seed: int) -> dict:
        all_steps = [s for s in returns.steps(engine.ledger, COVERED) if s.records][-MAX_STEPS:]
        self.folds, self.moved, self.version = {}, {}, engine.iteration_no
        self.disabled = self.stood_down is not None and engine.iteration_no - self.stood_down < WINDOW
        if len(all_steps) < MIN_STEPS:
            return {"steps": len(all_steps), "moved": {}}
        if len({s.island % 2 for s in all_steps}) < 2:
            return {"steps": len(all_steps), "moved": {}, "reason": "one island parity"}
        self.vocab = sorted({critic.token(engine, d, a, r.standing) for s in all_steps for d, r in s.records.items()
                             for a in r.distribution})
        index = {t: i for i, t in enumerate(self.vocab)}
        folds = {}
        for _ in range(2):
            returns.returns(all_steps, self._value(engine, folds, index, all_steps), engine.ledger.horizon)
            rows = self._rows(engine, all_steps, index)
            x = np.stack([r[1] for r in rows])
            a = np.array([r[2][r[3].choice] for r in rows])
            y = np.array([r[4].y for r in rows], dtype=np.float32)
            odd = np.array([r[4].island % 2 for r in rows], dtype=bool)
            folds = {False: critic.fit(x[odd], a[odd], y[odd], len(self.vocab), seed + 1),
                     True: critic.fit(x[~odd], a[~odd], y[~odd], len(self.vocab), seed + 2)}
        self.features, self.folds = x.shape[1], folds
        report = {}
        for decision in COVERED:
            mine = [(i, r) for i, r in enumerate(rows) if r[0] == decision and len(r[2]) > 1]
            if not mine:
                continue
            advantages, mus, chosen, ys, gains, parents = [], [], [], [], [], []
            for i, (_, xi, tokens, record, step) in mine:
                actions = list(tokens)
                q = self._q(folds[bool(odd[i])], xi, [tokens[b] for b in actions])
                mu = np.array([record.distribution[b] for b in actions])
                advantages.append(q.mean(1) - q.std(1) - (mu * q.mean(1)).sum())
                mus.append(mu)
                chosen.append(actions.index(record.choice))
                ys.append(step.y)
                gains.append(step.gain)
                parents.append(step.state)
            eta, kl = self._eta(advantages, mus)
            weights = np.array([self._tilt(m, adv, eta)[c] / m[c] for adv, m, c in zip(advantages, mus, chosen)])
            ys = np.array(ys)
            ess = weights.sum() ** 2 / max((weights ** 2).sum(), 1e-12)
            new_value = float((weights * ys).sum() / max(weights.sum(), 1e-12))
            base_value = float(ys.mean())
            spread = self._spread(weights / weights.mean() * ys - ys, parents)
            improving = int((np.array(gains) > 0).sum())
            passed = bool(ess >= MIN_ESS and improving >= MIN_IMPROVING and new_value - base_value > 2 * spread)
            if passed:
                self.moved[decision] = eta
            report[decision] = {"rows": len(mine), "eta": eta, "kl": kl, "ess": float(ess), "improving": improving,
                                "new_value": new_value, "base_value": base_value, "spread": spread, "moved": passed}
        return {"steps": len(all_steps), "moved": dict(self.moved), "decisions": report}

    @staticmethod
    def _spread(z: np.ndarray, clusters: list) -> float:
        """Standard error of the mean of z with rows from one parent kept together."""
        centred = z - z.mean()
        sums = {}
        for value, cluster in zip(centred, clusters):
            sums[cluster] = sums.get(cluster, 0.0) + value
        return float(math.sqrt(sum(s * s for s in sums.values())) / len(z))

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
        """Tilt the distribution with both fold models, as the gate judged them; no tilt without the parent's latent."""
        if not self.ready or decision not in self.moved or len(distribution) < 2:
            return dict(distribution)
        world = self.world or engine.swm
        if world.ready and world.latent(pid) is None:
            return dict(distribution)
        index = {t: i for i, t in enumerate(self.vocab)}
        tokens = [critic.token(engine, decision, b) for b in distribution]
        if any(t not in index for t in tokens):
            return dict(distribution)
        x = self._x(engine, pid, context, [r for r in earlier if _pair(r)[0] in COVERED])
        actions = [index[t] for t in tokens]
        q = np.concatenate([self._q(self.folds[False], x, actions), self._q(self.folds[True], x, actions)], 1)
        mu = np.array(list(distribution.values()), dtype=float)
        advantage = q.mean(1) - q.std(1) - (mu * q.mean(1)).sum()
        return {a: float(p) for a, p in zip(distribution, self._tilt(mu, advantage, self.moved[decision]))}

    def check(self, ledger) -> bool:
        """Stand down when the arm's last steps rest on fewer distinct parents than as many recent base steps."""
        if self.version is None or self.disabled:
            return False
        parents = {"rl": [], "base": []}
        for row in ledger.rows:
            if row.decision == "planner" and row.iteration > ledger.now - WINDOW and row.choice in parents:
                parents[row.choice].append(row.state)
        n = min(len(parents["rl"]), len(parents["base"]))
        if n < MIN_ARM_STEPS:
            return False

        def effective(ids):
            counts = np.unique(ids, return_counts=True)[1] / len(ids)
            return float(np.exp(-(counts * np.log(counts)).sum()))
        self.disabled = effective(parents["rl"][-n:]) < effective(parents["base"][-n:])
        if self.disabled:
            self.stood_down = ledger.now
        return self.disabled

    def state(self) -> dict:
        return {"vocab": self.vocab, "features": self.features, "moved": self.moved, "version": self.version,
                "disabled": self.disabled, "stood_down": self.stood_down,
                "folds": {str(k): critic.dump(net) for k, net in self.folds.items()}}

    def load_state(self, data: dict) -> None:
        self.vocab, self.features = list(data["vocab"]), int(data["features"])
        self.moved, self.version, self.disabled = dict(data["moved"]), data["version"], bool(data["disabled"])
        self.stood_down = data["stood_down"]
        self.folds = {k == "True": critic.load(v, self.features, len(self.vocab)) for k, v in data["folds"].items()}
