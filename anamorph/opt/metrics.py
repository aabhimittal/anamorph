"""Tier-A/B metrics: L2, PVBand, EPE violations (ICCAD-13 rules), shots, ILS, MRC.

EPE follows the ICCAD-13 / OpenILT convention: sample target edges (center only if the
edge is <= 80 nm, else every 40 nm from 40 nm inside each end) and flag a violation when
the printed image is off by more than 15 nm inward or outward. Tiles are treated as
periodic (no edges at the tile border). The rule is a parameter: 15 nm is meaningless
at a 12 nm CD, so Tier-B uses HIGHNA. Shot count is a greedy
row-run merge, NOT OpenILT's adabox cover, so only compare shots within this engine.
"""
from __future__ import annotations

import numpy as np
import torch

# (threshold, sample step, single-sample max edge length, first-sample inset) in nm
ICCAD13 = (15, 40, 80, 40)
HIGHNA = (3, 10, 20, 5)   # anamorph Tier-B rule: 25% of a 12 nm CD


def _runs(mask1d: np.ndarray):
    d = np.diff(np.concatenate([[0], mask1d.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


def epe_violations(printed: np.ndarray, target: np.ndarray, pixel_nm: float = 1.0, rule=ICCAD13) -> int:
    t = target > 0.5
    p = printed > 0.5
    h, w = t.shape
    th = int(round(rule[0] / pixel_nm))
    step, mn, st = (r / pixel_nm for r in rule[1:])
    pad = np.pad(t, 1, mode="wrap")  # periodic tile, like the optics
    count = 0
    # (edge map, inward direction (dy, dx), edges run along axis)
    edges = [(pad[1:-1, 1:-1] & ~pad[1:-1, :-2], (0, 1), 0), (pad[1:-1, 1:-1] & ~pad[1:-1, 2:], (0, -1), 0),
             (pad[1:-1, 1:-1] & ~pad[:-2, 1:-1], (1, 0), 1), (pad[1:-1, 1:-1] & ~pad[2:, 1:-1], (-1, 0), 1)]
    for emap, (dy, dx), along in edges:
        lines = emap.T if along == 0 else emap  # iterate over columns for vertical edges
        for i, line in enumerate(lines):
            for a, b in _runs(line):
                if b - a + 1 <= mn:
                    samples = [(a + b) // 2]
                else:
                    c = (a + b) / 2
                    samples = sorted(set(np.arange(a + st, c + 1, step).astype(int)) | set(np.arange(b - st, c, -step).astype(int)))
                for s in samples:
                    y, x = (s, i) if along == 0 else (i, s)
                    yi, xi = (y + dy * th) % h, (x + dx * th) % w
                    yo, xo = (y - dy * th) % h, (x - dx * th) % w
                    count += int(not p[yi, xi]) + int(p[yo, xo])
    return count


def rectangles(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Greedy cover: identical x-runs in consecutive rows merge. Returns (y0, x0, y1, x1), end-exclusive."""
    open_rects, out = {}, []
    rows = np.asarray(mask) > 0.5
    for y, row in enumerate(np.vstack([rows, np.zeros_like(rows[:1])])):
        cur = set(_runs(row))
        for (x0, x1), y0 in open_rects.items():
            if (x0, x1) not in cur:
                out.append((y0, x0, y, x1 + 1))
        open_rects = {r: open_rects.get(r, y) for r in cur}
    return out


def shot_count(mask: np.ndarray) -> int:
    return len(rectangles(mask))


def ils_at_edges(aerial: torch.Tensor, target: np.ndarray, pixel_nm: float) -> torch.Tensor:
    """Image log-slope |grad I| / I (1/nm) at target boundary pixels (differentiable)."""
    t = torch.as_tensor(target > 0.5)
    edge = t & ~(torch.roll(t, 1, 0) & torch.roll(t, -1, 0) & torch.roll(t, 1, 1) & torch.roll(t, -1, 1))
    gy = (torch.roll(aerial, -1, 0) - torch.roll(aerial, 1, 0)) / (2 * pixel_nm)
    gx = (torch.roll(aerial, -1, 1) - torch.roll(aerial, 1, 1)) / (2 * pixel_nm)
    return (torch.sqrt(gx**2 + gy**2 + 1e-20) / aerial.clamp(min=1e-6))[edge]
