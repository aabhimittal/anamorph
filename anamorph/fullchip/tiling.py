"""Full-chip ILT by overlapping tiles + a half-tile-shifted healing pass.

Each window = core tile + halo (>= optical/resist ambit) on every side. Only the core
of each optimized window is kept. Seams are then healed by re-optimizing windows whose
cores straddle the previous seams, initialized from the stitched mask.
shortcut: tiles run sequentially on one device; distribute `windows` across GPUs
(torch.distributed/NCCL) when full-chip runtime matters.
"""
from __future__ import annotations

import torch


def windows(shape, tile: int, halo: int, offset: int = 0):
    """Yield (window slice, core slice inside window, core slice in the full image)."""
    h, w = shape
    for y in range(-offset, h, tile):
        for x in range(-offset, w, tile):
            cy0, cx0, cy1, cx1 = max(y, 0), max(x, 0), min(y + tile, h), min(x + tile, w)
            if cy1 <= cy0 or cx1 <= cx0:
                continue
            wy0, wx0 = cy0 - halo, cx0 - halo
            full = (slice(cy0, cy1), slice(cx0, cx1))
            win = (slice(wy0 + halo, wy0 + halo + tile + 2 * halo), slice(wx0 + halo, wx0 + halo + tile + 2 * halo))
            core = (slice(halo, halo + cy1 - cy0), slice(halo, halo + cx1 - cx0))
            yield win, core, full


def run_tiled(target: torch.Tensor, solve, tile: int = 256, halo: int = 64, heal: bool = True) -> torch.Tensor:
    """solve(window_target, window_init) -> window_mask. Windows are zero-padded at the chip edge."""
    pad = lambda a: torch.nn.functional.pad(a, (halo, halo + tile, halo, halo + tile))
    tp = pad(target)
    out = torch.zeros_like(target)
    for win, core, full in windows(target.shape, tile, halo):
        out[full] = solve(tp[win], tp[win])[core]
    if heal:
        prev = pad(out)
        for win, core, full in windows(target.shape, tile, halo, offset=tile // 2):
            out[full] = solve(tp[win], prev[win])[core]
    return out
