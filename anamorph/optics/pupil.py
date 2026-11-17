"""Source sampling and the (vector, anamorphic) projection pupil.

Coordinates: everything is in WAFER coordinates. A spatial frequency f (cycles/nm)
at the wafer maps to the image-side NA-space coordinate (a, b) = wavelength * f.
The anamorphic reduction only enters the forward model through the mask-side
sines (a/Mx, b/My): in a thin-mask (Kirchhoff) model that is the radiometric
factor below and nothing else. The large anamorphic effects live in the MRC
(mask/mrc_morph.py, reticle-space rules) and in M3D (optics/m3d.py).
"""
from __future__ import annotations

import math

import torch

from anamorph.card import Card


def source_points(card: Card, dtype=torch.float64):
    """Discretize the illumination pupil fill. Returns (sigma_xy (S,2), weights (S,)) with sum(w)=1."""
    s = card.source
    n = int(math.ceil(1.0 / s.step))
    ax = torch.arange(-n, n + 1, dtype=dtype) * s.step
    sy, sx = torch.meshgrid(ax, ax, indexing="ij")
    r = torch.hypot(sx, sy)
    ring = (r <= s.sigma_out + 1e-9) & (r >= s.sigma_in - 1e-9)
    ang = torch.rad2deg(torch.atan2(sy, sx))
    half = s.opening_deg / 2
    if s.shape == "conventional":
        keep = r <= s.sigma_out + 1e-9
    elif s.shape == "annular":
        keep = ring
    elif s.shape == "dipole":
        base = 0.0 if s.axis == "x" else 90.0
        d = torch.remainder(ang - base + 90, 180) - 90  # angular distance to the pole axis (both poles)
        keep = ring & (d.abs() <= half)
    elif s.shape == "quasar":
        d = torch.remainder(ang - 45 + 45, 90) - 45
        keep = ring & (d.abs() <= half)
    else:
        raise ValueError(f"unknown source shape {s.shape!r}")
    pts = torch.stack([sx[keep], sy[keep]], -1)
    if len(pts) == 0:
        raise ValueError("source sampling produced no points; decrease source.step")
    w = torch.full((len(pts),), 1.0 / len(pts), dtype=dtype)
    return pts, w


def pupil_components(kx: torch.Tensor, ky: torch.Tensor, card: Card, focus_nm: float = 0.0) -> torch.Tensor:
    """Pupil transfer for wafer frequencies (kx, ky). Returns complex (C, *kx.shape).

    Scalar: C=1. Vector (unpolarized): C=6 = 2 incident polarizations (x, y) x 3 field
    components (Ex, Ey, Ez), each scaled by 1/sqrt(2) so the incoherent sum over C is
    the unpolarized intensity. Thin-mask assumption: the incident polarization is kept
    by the diffracted order, so each component depends only on the order's direction.
    """
    lam, na, n = card.wavelength_nm, card.na, card.immersion_index
    a, b = lam * kx, lam * ky
    r2 = a * a + b * b
    inside = (r2 <= na**2 + 1e-12) & (r2 >= (card.obscuration * na) ** 2 - 1e-12)
    an, bn = a / n, b / n
    g = torch.sqrt(torch.clamp(1 - r2 / n**2, min=1e-12))  # cos(theta) in the image medium
    mx, my = card.magnification
    rm2 = (a / mx) ** 2 + (b / my) ** 2                     # mask-side sin^2 (anamorphic)
    radiometric = (torch.clamp(1 - rm2, min=1e-12) / g**2) ** 0.25
    phase = torch.exp(1j * (2 * math.pi / lam * n * focus_nm) * (g - 1))
    base = torch.where(inside, radiometric * phase, torch.zeros_like(phase))  # where(): no 0*inf NaNs
    if not card.vector:
        return base.unsqueeze(0)
    c = -1.0 / (1.0 + g)  # (cos-1)/sin^2, singularity-free
    comps = [1 + an * an * c, an * bn * c, -an,            # x-polarized -> Ex, Ey, Ez
             an * bn * c, 1 + bn * bn * c, -bn]            # y-polarized -> Ex, Ey, Ez
    return torch.stack([base * q for q in comps]) / math.sqrt(2.0)


def freq_grid(shape, dx: float, dy: float, dtype=torch.float64):
    """(FY, FX) wafer frequency grids for an (H, W) image with pixel (dx, dy) nm, FFT ordering."""
    h, w = shape
    fy = torch.fft.fftfreq(h, dy, dtype=dtype)
    fx = torch.fft.fftfreq(w, dx, dtype=dtype)
    return torch.meshgrid(fy, fx, indexing="ij")


def check_sampling(card: Card, dx: float, dy: float):
    smax = card.source.sigma_out
    fmax = card.na * (1 + smax) / card.wavelength_nm
    if max(dx, dy) > 1 / (2 * fmax):
        raise ValueError(f"pixel {max(dx, dy)} nm aliases: need <= {1 / (2 * fmax):.2f} nm")
