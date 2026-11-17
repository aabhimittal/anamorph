"""Ground truth #1: the slow reference imaging path.

Today: full Abbe (every source point, every polarization/field component, no kernel
truncation) checked against the truncated SOCS the optimizer uses. A run whose SOCS
error exceeds the declared tolerance is void.

shortcut: no rigorous M3D (RCWA/FDTD) yet; candidates are TORCWA/S4/Reticolo after a
license and accuracy check. Until then optics/m3d.py stays at the thin-mask limit and
any M3D claim is out of scope.
"""
from __future__ import annotations

import torch

from anamorph.card import Card
from anamorph.optics.abbe import abbe_image
from anamorph.optics.m3d import thin_mask
from anamorph.optics.socs import SOCS


def socs_error(mask: torch.Tensor, card: Card, energy: float = 0.995, focus_nm: float = 0.0) -> float:
    """Max |I_socs - I_abbe| over the tile (clear field = 1)."""
    t = thin_mask(mask.to(torch.float64), card)
    ref = abbe_image(t, card, focus_nm)
    fast = SOCS.build(card, mask.shape, focus_nm, energy, dtype=torch.complex128)(t)
    return float((fast - ref).abs().max())
