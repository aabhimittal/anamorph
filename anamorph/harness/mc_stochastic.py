"""Ground truth #2: Monte-Carlo photon/acid sampler for gauge failure probabilities.

Deliberately independent of the engine: NumPy/SciPy only, its own kernel, path
sampling and interpolation. It samples the generative model documented in
anamorph/resist/grf_stochastic.py (compound Poisson photons -> acids, Gaussian blur)
WITHOUT the Gaussian/Rice approximations, and counts the exact gauge event.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.signal import fftconvolve
from scipy.stats import beta


def _path(points, step=0.5):
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(p) == 1:
        return p
    out = [p[:1]]
    for a, b in zip(p[:-1], p[1:]):
        n = max(int(math.ceil(np.linalg.norm(b - a) / step)), 1)
        out.append(a + np.linspace(0, 1, n + 1)[1:, None] * (b - a))
    return np.concatenate(out)


def mc_pfail(aerial: np.ndarray, card, points, sense: int, trials: int = 100_000,
             seed: int = 0, step: float = 0.5, max_elems: int = 20_000_000) -> dict:
    r = card.resist
    px = card.pixel_nm
    mu_ph = r.dose_mj_cm2 * 1e-17 / card.photon_energy_j * r.absorption * px * px
    q = r.quantum_yield
    mu_acid = mu_ph * q
    s = r.diffusion_nm / px
    halo = int(math.ceil(4 * s)) + 1
    ax = np.arange(-halo, halo + 1)
    ker = np.exp(-(ax[:, None] ** 2 + ax[None, :] ** 2) / (2 * s * s))
    ker /= ker.sum()

    xy = _path(points, step)
    x0, y0 = np.floor(xy.min(0)).astype(int)
    x1, y1 = np.floor(xy.max(0)).astype(int) + 1
    h, w = aerial.shape
    rows = np.arange(y0 - halo, y1 + halo + 1) % h      # periodic, like the FFT engine
    cols = np.arange(x0 - halo, x1 + halo + 1) % w
    patch = np.clip(aerial[np.ix_(rows, cols)], 0, None)
    fx, fy = xy[:, 0] - x0, xy[:, 1] - y0               # coords inside the valid-conv output
    ix, iy = np.floor(fx).astype(int), np.floor(fy).astype(int)
    tx, ty = fx - ix, fy - iy
    ix1, iy1 = np.minimum(ix + 1, x1 - x0), np.minimum(iy + 1, y1 - y0)

    rng = np.random.default_rng(seed)
    batch = max(1, min(trials, max_elems // patch.size))
    fails = done = 0
    while done < trials:
        b = min(batch, trials - done)
        acids = rng.poisson(q * rng.poisson(mu_ph * patch, size=(b, *patch.shape)))
        lat = fftconvolve(acids.astype(np.float64), ker[None], mode="valid", axes=(1, 2)) / mu_acid
        val = (lat[:, iy, ix] * (1 - tx) * (1 - ty) + lat[:, iy, ix1] * tx * (1 - ty)
               + lat[:, iy1, ix] * (1 - tx) * ty + lat[:, iy1, ix1] * tx * ty)
        fails += int((sense * (val - r.threshold) <= 0).any(1).sum())
        done += b
    lo = beta.ppf(0.025, fails, trials - fails + 1) if fails else 0.0
    hi = beta.ppf(0.975, fails + 1, trials - fails) if fails < trials else 1.0
    return {"p": fails / trials, "k": fails, "n": trials, "ci_lo": float(lo), "ci_hi": float(hi)}


def verdict(p_model: float, mc: dict, factor: float = 2.0, floor: float = 1e-6, min_events: int = 10) -> str:
    """HARNESS rule: model P_fail must match MC within `factor` where MC can resolve it.

    pass       : >= min_events MC failures and p_model within [p_mc/factor, p_mc*factor]
    model-only : below the resolvable floor, and not contradicted by the MC interval
    fail       : anything else (counts as a failed run, never as a lower score)
    """
    if p_model < floor or mc["k"] < min_events:
        consistent = p_model / factor <= mc["ci_hi"] and p_model * factor >= mc["ci_lo"]
        return "model-only" if consistent else "fail"
    return "pass" if mc["p"] / factor <= p_model <= mc["p"] * factor else "fail"
