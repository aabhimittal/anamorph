"""EUV/High-NA clip benchmark v0: parametric N2-class clips with stochastic gauges.

target = 1 where the resist must clear (trench / via opening; bright-field-of-feature).
Gauges encode necessary conditions of topological failure (see resist/rice_pfail.py):
  break   (+1): trench centerline must stay open    -> resist bridge across the trench
  bridge  (-1): space centerline must stay closed   -> two trenches/tips merge
  missing (+1): via center must open                -> missing via
Clips are periodic (FFT optics), so sizes are multiples of the pitch.
shortcut: synthetic clips only; clip extraction from routed ASAP7 GDS goes through
io/gds.py but has no automatic gauge placement yet.
"""
from __future__ import annotations

import dataclasses as dc

import torch

from anamorph.resist.rice_pfail import Gauge


@dc.dataclass
class Clip:
    name: str
    target: torch.Tensor
    gauges: list
    meta: dict


def line_space(pitch: int = 24, cd: int = 12, periods: int = 4, length: int | None = None) -> Clip:
    """Vertical trenches (vary along x) of width cd; tile = periods * pitch square."""
    n = pitch * periods
    t = torch.zeros(n, n, dtype=torch.float64)
    gauges = []
    y0, y1 = n * 0.25, n * 0.75 - 1
    for k in range(periods):
        x = k * pitch + (pitch - cd) // 2
        t[:, x:x + cd] = 1
        gauges.append(Gauge([(x + (cd - 1) / 2, y0), (x + (cd - 1) / 2, y1)], +1, "break"))
        xs = x + cd + (pitch - cd - 1) / 2
        gauges.append(Gauge([(xs % n, y0), (xs % n, y1)], -1, "bridge"))
    return Clip(f"ls_p{pitch}_cd{cd}", t, gauges, {"pitch": pitch, "cd": cd})


def tip_to_tip(pitch: int = 24, cd: int = 12, gap: int = 16, periods: int = 4) -> Clip:
    """Vertical trenches cut by a tip-to-tip gap at mid-height."""
    c = line_space(pitch, cd, periods)
    n = c.target.shape[0]
    g0 = n // 2 - gap // 2
    c.target[g0:g0 + gap, :] = 0
    gauges = []
    for k in range(periods):
        x = k * pitch + (pitch - cd) // 2
        xc = x + (cd - 1) / 2
        gauges.append(Gauge([(x, g0 + (gap - 1) / 2), (x + cd - 1, g0 + (gap - 1) / 2)], -1, "bridge"))
        gauges.append(Gauge([(xc, 4), (xc, g0 - 8)], +1, "break"))  # away from the line end
    return Clip(f"t2t_p{pitch}_cd{cd}_g{gap}", c.target, gauges, {"pitch": pitch, "cd": cd, "gap": gap})


def via_array(pitch: int = 32, cd: int = 16, periods: int = 4) -> Clip:
    n = pitch * periods
    t = torch.zeros(n, n, dtype=torch.float64)
    gauges = []
    for i in range(periods):
        for j in range(periods):
            y, x = i * pitch + (pitch - cd) // 2, j * pitch + (pitch - cd) // 2
            t[y:y + cd, x:x + cd] = 1
            gauges.append(Gauge([(x + (cd - 1) / 2, y + (cd - 1) / 2)], +1, "missing"))
            xm = (x + cd + (pitch - cd - 1) / 2) % n
            gauges.append(Gauge([(xm, y), (xm, y + cd - 1)], -1, "bridge"))
    return Clip(f"via_p{pitch}_cd{cd}", t, gauges, {"pitch": pitch, "cd": cd})


def suite() -> list[Clip]:
    """Benchmark v0 at 1 nm pixels."""
    return [line_space(24, 12), line_space(28, 14), tip_to_tip(24, 12, 16), via_array(32, 16), via_array(36, 18)]
