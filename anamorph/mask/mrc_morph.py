"""Mask rule check (MRC) via morphology, with reticle-space (anamorphic) structuring elements.

A min-width rule w (reticle nm) is a box of (w/Mx, w/My) wafer nm: opening with it erases
any feature narrower than the box, closing fills any gap narrower than it. The soft
version uses log-sum-exp pooling (differentiable, CurvyILT-style); the hard version uses
scipy binary morphology and is what violation counts report.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_closing, binary_opening

from anamorph.card import Card
from anamorph.optics.anamorphic import reticle_len_to_wafer_px


def se_size(rule_reticle_nm: float, card: Card) -> tuple[int, int]:
    """(ky, kx) odd box size in wafer pixels; rounds down (lenient by < 1 px)."""
    wx, wy = reticle_len_to_wafer_px(rule_reticle_nm, card)
    odd = lambda w: max(1, int(w) if int(w) % 2 else int(w) - 1)
    return odd(wy), odd(wx)


def _soft_dilate(m: torch.Tensor, k: tuple[int, int], tau: float) -> torch.Tensor:
    """Separable soft box-max: tau * log-mean-exp over the (ky, kx) window, periodic."""
    x = (m / tau)[None, None]
    for kk, pad in ((k[1], (k[1] // 2, k[1] // 2, 0, 0)), (k[0], (0, 0, k[0] // 2, k[0] // 2))):
        win = (1, kk) if pad[0] else (kk, 1)
        x = torch.logsumexp(F.unfold(F.pad(x, pad, mode="circular"), win), 1).reshape(x.shape) - np.log(kk)
    return tau * x[0, 0]


def soft_open(m, k, tau=0.05):
    return _soft_dilate(-_soft_dilate(-m, k, tau), k, tau)


def soft_close(m, k, tau=0.05):
    return -_soft_dilate(-_soft_dilate(m, k, tau), k, tau)


def mrc_penalty(m: torch.Tensor, card: Card, tau: float = 0.05) -> torch.Tensor:
    loss = m.new_zeros(())
    if card.mrc_width_reticle_nm > 0:
        loss = loss + (m - soft_open(m, se_size(card.mrc_width_reticle_nm, card), tau)).clamp(min=0).pow(2).mean()
    if card.mrc_space_reticle_nm > 0:
        loss = loss + (soft_close(m, se_size(card.mrc_space_reticle_nm, card), tau) - m).clamp(min=0).pow(2).mean()
    return loss


def mrc_violations(mask_bin: np.ndarray, card: Card) -> int:
    """Pixels removed by opening (too narrow) or filled by closing (gap too small); periodic."""
    p = 32
    b = np.pad(mask_bin > 0.5, p, mode="wrap")
    v = np.zeros_like(b)
    if card.mrc_width_reticle_nm > 0:
        v |= b & ~binary_opening(b, np.ones(se_size(card.mrc_width_reticle_nm, card), bool))
    if card.mrc_space_reticle_nm > 0:
        v |= ~b & binary_closing(b, np.ones(se_size(card.mrc_space_reticle_nm, card), bool))
    return int(v[p:-p, p:-p].sum())
