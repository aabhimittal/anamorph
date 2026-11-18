"""Tier-A adapters: ICCAD-13 .glp clips with the contest SOCS kernels (as shipped by OpenILT),
and LithoBench/MaskOpt-style .npy/.png target tiles.

Conventions reproduced from OpenILT (MIT): 2048x2048 canvas at 1 nm, clip centered;
24 kernels; aerial threshold 0.225, sigmoid steepness 50; corners: max = focus kernels at
dose 1.02, min = defocus kernels at dose 0.98, where dose scales the MASK amplitude.
The 35x35 frequency kernels have df = 1/2048 nm^-1, so they apply unchanged to a
(2048/s)^2 grid at s nm pixels: optimize coarse, evaluate at 1 nm (nearest upsample).
Known difference: OpenILT rasterizes with cv2.fillPoly (inclusive of the far edge);
we rasterize pixel centers, so targets differ by up to one pixel row per edge.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from anamorph.card import Card, Resist
from anamorph.optics.socs import SOCS
from anamorph.resist.grf_stochastic import GRFModel

ICCAD13_CARD = Card(name="iccad13", wavelength_nm=193.0, na=1.35, immersion_index=1.44, magnification=(4, 4),
                    vector=False, pixel_nm=1.0, pw_dose=0.02,
                    resist=Resist(threshold=0.225, steepness=50.0, diffusion_nm=0.0))


def read_glp(path) -> list[np.ndarray]:
    polys = []
    for line in Path(path).read_text().splitlines():
        f = line.split()
        if len(f) >= 7 and f[0] == "RECT":
            x, y, w, h = map(int, f[3:7])
            polys.append(np.array([[x, y], [x, y + h], [x + w, y + h], [x + w, y]]))
        elif len(f) >= 7 and f[0] == "PGON":
            polys.append(np.array(list(map(int, f[3:]))).reshape(-1, 2))
    return polys


def glp_target(path, size: int = 2048, pixel_nm: float = 1.0) -> torch.Tensor:
    import gdstk
    polys = read_glp(path)
    lo = np.min([p.min(0) for p in polys], 0)
    hi = np.max([p.max(0) for p in polys], 0)
    shift = (size * pixel_nm - (hi - lo)) // 2 - lo
    c = (np.arange(size) + 0.5) * pixel_nm
    gx, gy = np.meshgrid(c, c)
    inside = gdstk.inside(np.stack([gx.ravel(), gy.ravel()], 1), [p + shift for p in polys])
    return torch.tensor(np.asarray(inside, float).reshape(size, size))


def load_target(path) -> torch.Tensor:
    """LithoBench / MaskOpt tile (.npy or .png), binarized at 0.5 of its max."""
    path = Path(path)
    if path.suffix == ".npy":
        a = np.load(path, allow_pickle=False).astype(float)
    else:
        from PIL import Image
        a = np.asarray(Image.open(path).convert("L"), float)
    return torch.tensor((a > 0.5 * a.max()).astype(float))


def load_openilt_kernels(kernel_dir, defocus: bool = False):
    """(kernels (24, 35, 35) complex, scales (24,)) from OpenILT's kernel/ directory."""
    name = "defocus" if defocus else "focus"
    k = torch.load(Path(kernel_dir) / "kernels" / f"{name}.pt", map_location="cpu", weights_only=True)
    s = torch.load(Path(kernel_dir) / "scales" / f"{name}.pt", map_location="cpu", weights_only=True)
    return k.permute(2, 0, 1), s


class ICCAD13Litho:
    """Drop-in for opt.ilt.Litho with the contest kernels and corner conventions."""

    def __init__(self, kernel_dir, shape, card: Card = ICCAD13_CARD):
        self.card, self.m3d = card, None
        self.nom = SOCS.from_spatial_kernels(*load_openilt_kernels(kernel_dir), shape)
        self.dfc = SOCS.from_spatial_kernels(*load_openilt_kernels(kernel_dir, True), shape)
        self.resist = GRFModel(card, shape)

    def aerials(self, m):
        m = m.to(torch.complex64)
        d = self.card.pw_dose
        return (self.nom(m).to(torch.float64), self.nom(m * (1 + d)).to(torch.float64),
                self.dfc(m * (1 - d)).to(torch.float64))
