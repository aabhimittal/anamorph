"""Non-LLM baseline for the agent ablation: random search over the knob schema under
the same wall-clock budget the agent gets. Every candidate goes through load_knobs, so
the baseline and the agent share one admissible action set.

shortcut: no LLM driver here; an LLM agent plugs in by proposing knob dicts (and
EVOLVE-BLOCK edits checked by boundary.check_evolve_edit) to the same `objective`.
"""
from __future__ import annotations

import math
import random
import time

from anamorph.agent.boundary import SCHEMA, load_knobs

# log-uniform for weights spanning decades; others uniform within the schema bounds
_LOG = {"lambda_pfail", "w_mrc", "w_nils", "w_bin", "w_pvb", "lr"}


def sample(space: dict, rng: random.Random) -> dict:
    out = {}
    for k, (lo, hi) in space.items():
        spec = SCHEMA["properties"][k]
        if "enum" in spec:
            out[k] = rng.choice(spec["enum"])
        elif k in _LOG and lo > 0:
            out[k] = math.exp(rng.uniform(math.log(lo), math.log(hi)))
        else:
            out[k] = rng.uniform(lo, hi)
        if spec.get("type") == "integer":
            out[k] = int(round(out[k]))
    return out


def random_search(objective, space: dict, budget_s: float, fixed: dict | None = None, seed: int = 0):
    """objective(Knobs) -> float (lower is better). space: {knob: (lo, hi)} or {knob: (None, None)} for enums."""
    rng = random.Random(seed)
    best, trials, t0 = (math.inf, None), [], time.time()
    while time.time() - t0 < budget_s:
        d = sample(space, rng) | (fixed or {})
        score = objective(load_knobs(d))
        trials.append((score, d))
        best = min(best, (score, d), key=lambda x: x[0])
    return best, trials
