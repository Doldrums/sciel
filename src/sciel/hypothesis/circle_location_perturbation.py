"""Opt-in median-gap center exchange with location noise after the swap.

Only source assembly is exposed here. The default, historical sources, solver,
RNG schedule and traced profile are unchanged. Activation is a changed numeric
multiset of center locations, not helper acceptance or packing-quality evidence.
"""
from __future__ import annotations

from . import circle_optimizer_traced_body as builder

CONTRACT = "sciel-circle-location-perturbation/1"
EXECUTION_PROFILE = builder.EXECUTION_PROFILE
PARENT_BODY_SHA256 = "32fc460630af06c191f67c43bcc8fe5133808f22d4d3b4453645fb7b7eea3ee1"
PARENT_BODY = """if restart_index <= 0 or restart_index % 16 != 0:
    return None

i = (restart_index // 16 - 1) % 26
ci_x = best_params[2 * i]
ci_y = best_params[2 * i + 1]
ri = best_params[52 + i]
eligible_gaps = np.zeros(26)
eligible_count = 0

for j in range(26):
    if j == i:
        continue
    cj_x = best_params[2 * j]
    cj_y = best_params[2 * j + 1]
    rj = best_params[52 + j]
    if ri == rj or (ci_x == cj_x and ci_y == cj_y):
        continue
    if cj_x < ri or cj_x > 1.0 - ri or cj_y < ri or cj_y > 1.0 - ri:
        continue
    if ci_x < rj or ci_x > 1.0 - rj or ci_y < rj or ci_y > 1.0 - rj:
        continue
    eligible_gaps[eligible_count] = abs(ri - rj)
    eligible_count += 1

if eligible_count == 0:
    return None
ordered_gaps = sorted(eligible_gaps[:eligible_count])
if eligible_count % 2 == 1:
    median_gap = ordered_gaps[eligible_count // 2]
else:
    median_gap = (ordered_gaps[eligible_count // 2 - 1] + ordered_gaps[eligible_count // 2]) * 0.5

best_penetration = 0.0
best_difference = -1.0
j_best = -1

for j in range(26):
    if j == i:
        continue
    cj_x = best_params[2 * j]
    cj_y = best_params[2 * j + 1]
    rj = best_params[52 + j]
    if ri == rj or (ci_x == cj_x and ci_y == cj_y):
        continue
    if cj_x < ri or cj_x > 1.0 - ri or cj_y < ri or cj_y > 1.0 - ri:
        continue
    if ci_x < rj or ci_x > 1.0 - rj or ci_y < rj or ci_y > 1.0 - rj:
        continue
    if abs(ri - rj) < median_gap:
        continue
    penetration = 0.0
    for k in range(26):
        if k == i or k == j:
            continue
        ck_x = best_params[2 * k]
        ck_y = best_params[2 * k + 1]
        rk = best_params[52 + k]
        dx = cj_x - ck_x
        dy = cj_y - ck_y
        penetration = max(penetration, ri + rk - math.sqrt(dx * dx + dy * dy))
        dx = ci_x - ck_x
        dy = ci_y - ck_y
        penetration = max(penetration, rj + rk - math.sqrt(dx * dx + dy * dy))
    difference = abs(ri - rj)
    if j_best < 0 or penetration < best_penetration or (penetration == best_penetration and difference > best_difference):
        best_penetration = penetration
        best_difference = difference
        j_best = j

if j_best < 0:
    return None

new_params = np.copy(best_params)
new_params[2 * i] = best_params[2 * j_best]
new_params[2 * i + 1] = best_params[2 * j_best + 1]
new_params[2 * j_best] = ci_x
new_params[2 * j_best + 1] = ci_y
return new_params
"""

_SWAP = """new_params[2 * i] = best_params[2 * j_best]
new_params[2 * i + 1] = best_params[2 * j_best + 1]
new_params[2 * j_best] = ci_x
new_params[2 * j_best + 1] = ci_y
"""
_PERTURB = """new_params[2 * i] = np.clip(best_params[2 * j_best] + uniform_noise[2 * i], ri, 1.0 - ri)
new_params[2 * i + 1] = np.clip(best_params[2 * j_best + 1] + uniform_noise[2 * i + 1], ri, 1.0 - ri)
new_params[2 * j_best] = np.clip(ci_x + uniform_noise[2 * j_best], best_params[52 + j_best], 1.0 - best_params[52 + j_best])
new_params[2 * j_best + 1] = np.clip(ci_y + uniform_noise[2 * j_best + 1], best_params[52 + j_best], 1.0 - best_params[52 + j_best])
"""
if PARENT_BODY.count(_SWAP) != 1:
    raise ValueError("location perturbation parent anchor differs")
BODY = PARENT_BODY.replace(_SWAP, _PERTURB, 1)


def source_for():
    """Assemble this opt-in body under the current profile /2; execute nothing."""
    return builder.source_for(BODY)


def apply(parent_source):
    """Exact body edit proof; callers choose the explicit same-profile parent."""
    return builder.apply(parent_source, BODY)
