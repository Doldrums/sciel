"""The drafting model turns a materials digest into problem.py: solve(), score(), metrics() and, if it can, upper()."""

from __future__ import annotations

import re

PROMPT = """You turn research materials into a search task for SCiEL, an evolutionary program search.
Write one Python file, problem.py, that defines:

- DATA: a pathlib.Path to the materials folder, exactly {folder!r}.
- solve(): a simple, honest baseline that returns a result. The search will rewrite this function to raise score().
- score(result) -> float: how good a result is. Higher is better, and every valid result scores above 0, because
  a crashing candidate counts as 0 (for an error E use for example 1 / (1 + E)). Deterministic. Fixed during the search.
- metrics(result) -> dict: two to four numbers describing HOW a result behaves (not how good it is), for maps.
- upper() -> float, only if the materials let you compute a strong reference or an oracle bound for score();
  omit it otherwise. It may use information solve() must never use.

Rules:
- If the data has labels or targets, split it by a fixed rule: solve() may fit on the training part only, and
  score() judges on a held-out part. solve() must never read the held-out targets.
- Keep solve() fast: under {smoke:g} seconds. Use only the Python standard library and numpy{extra}.
- Do not invent data. If something is unclear, choose the reading the materials support best and say so in ABOUT.

Reply with the file in one ```python block, then one line starting with ABOUT: that says what the task is,
then up to two lines starting with QUESTION: for choices the user should confirm.

{note}MATERIALS:
{digest}
"""

REPAIR = """The problem.py you wrote failed when its baseline was scored:

{error}

Return the corrected complete file in one ```python block, then the ABOUT: and QUESTION: lines again.

The file was:
```python
{source}
```
"""


def parse(text: str) -> dict:
    blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
    if not blocks:
        raise ValueError("the drafting model returned no Python block")
    about = re.search(r"^ABOUT:\s*(.+)$", text, re.M)
    return {"source": max(blocks, key=len), "about": about.group(1).strip() if about else "",
            "questions": [q.strip() for q in re.findall(r"^QUESTION:\s*(.+)$", text, re.M)][:2]}


class Drafter:
    def __init__(self, llm, model):
        self.llm, self.model = llm, model
        self.cost = 0.0

    def _ask(self, prompt):
        response = self.llm.generate(prompt, self.model)
        self.cost += response.cost or 0.0
        return parse(response.text)

    def draft(self, folder, digest, *, smoke=60.0, note=""):
        extra = ", scipy and pandas if the materials need them"
        return self._ask(PROMPT.format(folder=str(folder), smoke=smoke, extra=extra, digest=digest,
                                       note=f"NOTE FROM THE USER: {note}\n\n" if note else ""))

    def repair(self, source, error):
        return self._ask(REPAIR.format(error=error[-3000:], source=source))
