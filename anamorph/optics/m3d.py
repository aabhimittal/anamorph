"""Mask transmission: thin (Kirchhoff) mask plus a learnable boundary-layer M3D correction.

Convention: mask m in [0, 1], 1 = reflective opening (bright), 0 = absorber.
Thin mask: t = r_abs + (1 - r_abs) * m, with r_abs the absorber/multilayer amplitude ratio.

M3D (mask topography) is modeled as a correction localized at edges, with a separate
complex kernel for each edge orientation (+x, -x, +y, -y). The orientations must be
separate because EUV illumination is oblique at the reticle (shadowing is asymmetric)
and, with anamorphic optics, mask-side angles differ in x and y. The kernels are fitted
against a rigorous near-field (RCWA/FDTD) offline; until such a reference is attached,
the kernels are zero and the model is exactly the thin mask.

shortcut: one correction for all source points (no angle dependence of the near field);
upgrade to per-source-cluster kernels when a rigorous reference is available.
"""
from __future__ import annotations

import cmath
import math

import torch
import torch.nn.functional as F

from anamorph.card import Card


def thin_mask(m: torch.Tensor, card: Card) -> torch.Tensor:
    r = card.absorber_amplitude * cmath.exp(1j * math.radians(card.absorber_phase_deg))
    return r + (1 - r) * m.to(torch.complex64 if m.dtype != torch.float64 else torch.complex128)


def edge_maps(m: torch.Tensor) -> torch.Tensor:
    """(..., 4, H, W): rising/falling edges along x and y (periodic differences)."""
    gx = m - torch.roll(m, 1, -1)
    gy = m - torch.roll(m, 1, -2)
    return torch.stack([F.relu(gx), F.relu(-gx), F.relu(gy), F.relu(-gy)], -3)


class BoundaryLayerM3D(torch.nn.Module):
    def __init__(self, ksize: int = 7):
        super().__init__()
        self.ksize = ksize
        self.kernel = torch.nn.Parameter(torch.zeros(2, 4, ksize, ksize))  # (re/im, edge dir, kh, kw)

    def correction(self, m: torch.Tensor) -> torch.Tensor:
        e = edge_maps(m).reshape(-1, 4, *m.shape[-2:]).to(self.kernel.dtype)
        p = self.ksize // 2
        e = F.pad(e, (p, p, p, p), mode="circular")
        out = F.conv2d(e, self.kernel)  # (N, 2, H, W): real and imaginary parts
        out = out.reshape(*m.shape[:-2], 2, *m.shape[-2:])
        return torch.complex(out[..., 0, :, :], out[..., 1, :, :])

    def forward(self, m: torch.Tensor, card: Card) -> torch.Tensor:
        t = thin_mask(m, card)
        return t + self.correction(m).to(t.dtype)


def fit(model: BoundaryLayerM3D, masks: torch.Tensor, ref_t: torch.Tensor, card: Card,
        iters: int = 300, lr: float = 0.02) -> list[float]:
    """Fit the boundary-layer kernels so model(mask) matches rigorous near-fields ref_t (N, H, W)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = []
    for _ in range(iters):
        opt.zero_grad()
        loss = (model(masks, card) - ref_t).abs().pow(2).mean()
        loss.backward()
        opt.step()
        hist.append(loss.item())
    return hist
