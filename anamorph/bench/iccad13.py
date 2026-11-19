"""Tier-A adapters: ICCAD-13 .glp clips with the contest SOCS kernels (as shipped by OpenILT),
and LithoBench/MaskOpt-style .npy/.png target tiles.

Conventions reproduced from OpenILT (MIT): 2048x2048 canvas at 1 nm, clip centered;
24 kernels; aerial threshold 0.225, sigmoid steepness 50; corners: max = focus kernels at
dose 1.02, min = defocus kernels at dose 0.98, where dose scales the MASK amplitude.
The 35x35 frequency kernels have df = 1/2048 nm^-1, so they apply unchanged to a
(2048/s)^2 grid at s nm pixels: optimize coarse, evaluate at 1 nm (nearest upsample).
Rasterization: OpenILT uses cv2.fillPoly, which fills every integer grid point of the
CLOSED polygon (a w-wide RECT covers w+1 pixels). glp_target(inclusive=True) reproduces
that and is what Tier-A parity uses; the default pixel-center rule is the geometric one.
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


def glp_target(path, size: int = 2048, pixel_nm: float = 1.0, inclusive: bool = False) -> torch.Tensor:
    import gdstk
    polys = read_glp(path)
    lo = np.min([p.min(0) for p in polys], 0)
    hi = np.max([p.max(0) for p in polys], 0)
    shift = (size * pixel_nm - (hi - lo)) // 2 - lo
    polys = [gdstk.Polygon(p + shift) for p in polys]
    if inclusive:  # closed set on the integer grid: a 0.25 miter offset (Manhattan input) keeps edge points inside
        polys = gdstk.offset(polys, 0.25 * pixel_nm, join="miter")
    c = (np.arange(size) + (0.0 if inclusive else 0.5)) * pixel_nm
    gx, gy = np.meshgrid(c, c)
    inside = gdstk.inside(np.stack([gx.ravel(), gy.ravel()], 1), polys)
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


# OpenILT README "SimpleILT" table (commit dabb97c, 2048x2048 evaluation): case -> (L2, PVBand, EPE)
PUBLISHED_SIMPLEILT = {1: (43408, 52281, 3), 2: (35326, 41865, 2), 3: (75428, 78805, 43), 4: (13649, 22112, 2),
                       5: (37330, 54977, 2), 6: (35711, 51036, 0), 7: (29566, 44576, 0), 8: (14327, 20727, 0),
                       9: (45347, 64063, 0), 10: (8404, 16685, 0)}


def _combo_kernel(kernel_dir, name: str, shape) -> torch.Tensor:
    k = torch.load(Path(kernel_dir) / "kernels" / f"{name}.pt", map_location="cpu", weights_only=True)
    return SOCS.from_spatial_kernels(k.permute(2, 0, 1), torch.ones(1), shape).kernels[0]


class _OpenILTGrad(torch.autograd.Function):
    """OpenILT pylitho/simple.py: exact 24-kernel forward, but the backward pass is the
    gradient of a single-kernel coherent image built from the 'combo' kernel K and its
    conjugate-transpose K' (no outer dose factor), copied exactly for protocol parity."""

    @staticmethod
    def forward(ctx, m, dose, socs, kc, kct):
        ctx.save_for_backward(m)
        ctx.args = (dose, kc, kct)
        return socs(m.to(torch.complex64) * dose)

    @staticmethod
    def backward(ctx, g):
        (m,), (dose, kc, kct) = ctx.saved_tensors, ctx.args
        f = lambda x, k: torch.fft.ifft2(torch.fft.fft2(x) * k)
        dm, gc = m.to(torch.complex64) * dose, g.to(torch.complex64)
        return (f(f(dm, kct) * gc, kc) + f(f(dm, kc) * gc, kct)).real.to(m.dtype), None, None, None, None


class ICCAD13Litho:
    """Drop-in for opt.ilt.Litho with the contest kernels and corner conventions.

    gradient="exact" differentiates the true SOCS model; "openilt" reproduces OpenILT's
    approximate combo-kernel backward (only for reproducing their published runs).
    """

    def __init__(self, kernel_dir, shape, card: Card = ICCAD13_CARD, gradient: str = "exact"):
        self.card, self.m3d, self.gradient = card, None, gradient
        self.nom = SOCS.from_spatial_kernels(*load_openilt_kernels(kernel_dir), shape)
        self.dfc = SOCS.from_spatial_kernels(*load_openilt_kernels(kernel_dir, True), shape)
        self.resist = GRFModel(card, shape)
        if gradient == "openilt":
            self.combo = {id(self.nom): (_combo_kernel(kernel_dir, "combo_focus", shape), _combo_kernel(kernel_dir, "combo_ct_focus", shape)),
                          id(self.dfc): (_combo_kernel(kernel_dir, "combo_defocus", shape), _combo_kernel(kernel_dir, "combo_ct_defocus", shape))}
        elif gradient != "exact":
            raise ValueError(gradient)

    def _image(self, socs, m, dose):
        if self.gradient == "exact":
            return socs(m.to(torch.complex64) * dose).to(torch.float64)
        return _OpenILTGrad.apply(m, dose, socs, *self.combo[id(socs)]).to(torch.float64)

    def aerials(self, m):
        d = self.card.pw_dose
        return self._image(self.nom, m, 1.0), self._image(self.nom, m, 1 + d), self._image(self.dfc, m, 1 - d)


def simple_ilt(target: torch.Tensor, litho, iters: int = 20, lr: float = 0.5, steepness: float = 4.0,
               offset: int = 512, size: int = 1024) -> torch.Tensor:
    """OpenILT SimpleILT protocol (config/simpleilt2048.txt): SGD on pixel params
    initialized to 2t-1, only the central window optimized, loss = sum of squared
    errors at the nominal/max/min corners, mask of the lowest-loss iterate returned."""
    r = litho.card.resist
    filt = torch.zeros_like(target)
    filt[offset:offset + size, offset:offset + size] = 1
    backup = 2 * target - 1
    params = backup.clone().requires_grad_(True)
    opt = torch.optim.SGD([params], lr=lr)
    best = (float("inf"), None)
    for _ in range(iters):
        mask = torch.sigmoid(steepness * params) * filt + torch.sigmoid(steepness * backup) * (1 - filt)
        z = [torch.sigmoid(r.steepness * (a - r.threshold)) for a in litho.aerials(mask)]
        loss = sum(((zi - target) ** 2).sum() for zi in z)
        if loss.item() < best[0]:
            best = (loss.item(), mask.detach().clone())
        opt.zero_grad()
        loss.backward()
        opt.step()
    return best[1]
