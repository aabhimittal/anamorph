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


def glp_target(path, size: int = 2048, pixel_nm: float = 1.0, inclusive: bool = False, down: int = 1) -> torch.Tensor:
    """down: integer coordinate downsampling before rasterizing (OpenILT glp.Design(down=...))."""
    import gdstk
    polys = [p // down for p in read_glp(path)]
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

# LithoBench pyilt/curvmulti.py result comment (commit 9c74e82), final "High" level: case -> (L2, PVBand, EPE)
PUBLISHED_CURVMULTI = {1: (39112, 48831, 3), 2: (31082, 39102, 0), 3: (63569, 76183, 17), 4: (8844, 23986, 0),
                       5: (28721, 53856, 0), 6: (29981, 49084, 0), 7: (14813, 42364, 0), 8: (10937, 21210, 0),
                       9: (34791, 62161, 0), 10: (7558, 17393, 0)}


def _raw_kernels(kernel_dir, name: str, shape) -> torch.Tensor:
    """(K, H, W) frequency kernels placed on the FFT grid, WITHOUT the sqrt(scale) factor."""
    k = torch.load(Path(kernel_dir) / "kernels" / f"{name}.pt", map_location="cpu", weights_only=True).permute(2, 0, 1)
    return SOCS.from_spatial_kernels(k, torch.ones(k.shape[0]), shape).kernels


class _HandGrad(torch.autograd.Function):
    """OpenILT/LithoBench hand-written backward, copied exactly for protocol parity:
    grad = Re sum_k s_k [K_k (g . (K'_k dm)) + K'_k (g . (K_k dm))], dm = dose * m, with K' the
    stored 'CT' kernels and no outer dose factor. pylitho/simple.py uses one 'combo' kernel
    (s = 1); pylitho/exact.py uses all 24. Neither is the true adjoint (that needs K(-f)
    where K(f) is used), so these modes only exist to reproduce published runs."""

    @staticmethod
    def forward(ctx, m, dose, socs, k, kct, sc):
        ctx.save_for_backward(m)
        ctx.args = (dose, k, kct, sc)
        return socs(m.to(torch.complex64) * dose)

    @staticmethod
    def backward(ctx, g):
        (m,), (dose, k, kct, sc) = ctx.saved_tensors, ctx.args
        f = lambda x, kk: torch.fft.ifft2(torch.fft.fft2(x) * kk)
        dm, gc = (m.to(torch.complex64) * dose).unsqueeze(-3), g.to(torch.complex64).unsqueeze(-3)
        per_k = f(f(dm, kct) * gc, k) + f(f(dm, k) * gc, kct)
        return (sc[:, None, None] * per_k.real).sum(-3).to(m.dtype), None, None, None, None, None


class ICCAD13Litho:
    """Drop-in for opt.ilt.Litho with the contest kernels and corner conventions.

    gradient="exact"          true SOCS autograd (use this for real work);
    "openilt"                 OpenILT/LithoBench pylitho/simple.py combo-kernel backward;
    "openilt-exact"           OpenILT/LithoBench pylitho/exact.py 24-kernel hand backward.
    The two compat modes only reproduce published runs.
    """

    def __init__(self, kernel_dir, shape, card: Card = ICCAD13_CARD, gradient: str = "exact"):
        self.card, self.m3d, self.gradient = card, None, gradient
        self.nom = SOCS.from_spatial_kernels(*load_openilt_kernels(kernel_dir), shape)
        self.dfc = SOCS.from_spatial_kernels(*load_openilt_kernels(kernel_dir, True), shape)
        self.resist = GRFModel(card, shape)
        if gradient == "exact":
            return
        if gradient not in ("openilt", "openilt-exact"):
            raise ValueError(gradient)
        pre = "combo_" if gradient == "openilt" else ""
        self.hand = {}
        for socs, focus in ((self.nom, "focus"), (self.dfc, "defocus")):
            k, kct = _raw_kernels(kernel_dir, f"{pre}{focus}", shape), _raw_kernels(kernel_dir, f"{pre}ct_{focus}", shape)
            sc = torch.ones(1) if pre else load_openilt_kernels(kernel_dir, focus == "defocus")[1].float()
            self.hand[id(socs)] = (k, kct, sc)

    def _image(self, socs, m, dose):
        if self.gradient == "exact":
            return socs(m.to(torch.complex64) * dose).to(torch.float64)
        return _HandGrad.apply(m, dose, socs, *self.hand[id(socs)]).to(torch.float64)

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


_CURV = torch.tensor([[-1 / 16, 5 / 16, -1 / 16], [5 / 16, -1.0, 5 / 16], [-1 / 16, 5 / 16, -1 / 16]])


def curv_ilt(target, params, litho, iters: int, lr: float = 0.5, steepness: float = 4.0, offset: int = 0,
             size: int | None = None, w_curv: float = 2e2):
    """LithoBench pyilt/curvmulti.py single level: mask = sigmoid(s * (avgpool7(params) - 0.5)) inside
    the window (0 outside); loss = |Z_max - t|^2 + |Z_max - Z_min|^2 + w_curv * |curv(Z_nom)|^2 (sums);
    SGD; returns (params, binarized mask) of the lowest-loss iterate."""
    r = litho.card.resist
    size = size or target.shape[0] - 2 * offset
    filt = torch.zeros_like(target)
    filt[offset:offset + size, offset:offset + size] = 1
    params = params.clone().requires_grad_(True)
    opt = torch.optim.SGD([params], lr=lr)
    pool = lambda p: torch.nn.functional.avg_pool2d(p[None, None], 7, stride=1, padding=3)[0, 0]
    best = (float("inf"), None, None)
    for _ in range(iters):
        mask = torch.sigmoid(steepness * (pool(params) - 0.5)) * filt
        zn, zx, zm = (torch.sigmoid(r.steepness * (a - r.threshold)) for a in litho.aerials(mask))
        curv = torch.nn.functional.conv2d(zn[None, None], _CURV.to(zn.dtype)[None, None])[0, 0]
        loss = ((zx - target) ** 2).sum() + ((zx - zm) ** 2).sum() + w_curv * (curv**2).sum()
        if loss.item() < best[0]:
            with torch.no_grad():
                m = (torch.sigmoid(steepness * (pool(params) - 0.5)) * filt > 0.5).to(target.dtype)
            best = (loss.item(), params.detach().clone(), m)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return best[1], best[2]


def curv_multi(glp_path, kernel_dir, gradient: str = "exact", levels=((8, 200, 48), (4, 100, 96), (2, 100, 192))):
    """LithoBench CurvMulti (config/curvilt{256,512,1024}.txt): coarse-to-fine curv_ilt, params
    carried by 2x nearest upsampling. Yields (scale, binary mask on the (2048/scale)^2 grid) per level."""
    params = None
    for scale, iters, offset in levels:
        n = 2048 // scale
        t = glp_target(glp_path, size=n, inclusive=True, down=scale).float()
        init = t if params is None else torch.nn.functional.interpolate(params[None, None], scale_factor=2, mode="nearest")[0, 0]
        params, mask = curv_ilt(t, init, ICCAD13Litho(kernel_dir, (n, n), gradient=gradient), iters, offset=offset, size=n - 2 * offset)
        yield scale, mask
