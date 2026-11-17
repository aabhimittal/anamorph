"""Abbe (source-point summation) imaging: the slow reference path for SOCS.

I(x) = sum_s w_s sum_c | IFFT[ T(f) * P_c(f + f_s) ] |^2, normalized to clear field = 1.
"""
from __future__ import annotations

import torch

from anamorph.card import Card
from anamorph.optics.pupil import check_sampling, freq_grid, pupil_components, source_points


def clear_field(card: Card, focus_nm: float = 0.0) -> float:
    pts, w = source_points(card)
    fs = pts * card.na / card.wavelength_nm
    p = pupil_components(fs[:, 0], fs[:, 1], card, focus_nm)  # (C, S)
    return float((w * p.abs().pow(2).sum(0)).sum())


def abbe_image(t: torch.Tensor, card: Card, focus_nm: float = 0.0, chunk: int = 8) -> torch.Tensor:
    """t: complex mask transmission (..., H, W) in wafer coordinates, pixel = card.pixel_nm."""
    dx = dy = card.pixel_nm
    check_sampling(card, dx, dy)
    rdt = torch.float64 if t.dtype in (torch.complex128, torch.float64) else torch.float32
    fy, fx = freq_grid(t.shape[-2:], dx, dy, dtype=torch.float64)
    pts, w = source_points(card)
    fs = pts * card.na / card.wavelength_nm
    spec = torch.fft.fft2(t.to(torch.complex128 if rdt == torch.float64 else torch.complex64))
    out = torch.zeros(t.shape, dtype=rdt, device=t.device)
    for i in range(0, len(fs), chunk):
        sl = slice(i, i + chunk)
        p = pupil_components(fx[None] + fs[sl, 0, None, None], fy[None] + fs[sl, 1, None, None], card, focus_nm)
        p = p.to(spec.dtype).to(t.device)                       # (C, s, H, W)
        e = torch.fft.ifft2(spec[..., None, None, :, :] * p)    # (..., C, s, H, W)
        out = out + (e.abs().pow(2) * w[sl].to(rdt).to(t.device)[:, None, None]).sum((-4, -3))
    return out / clear_field(card, focus_nm)
