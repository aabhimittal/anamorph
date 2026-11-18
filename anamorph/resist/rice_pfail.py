"""Differentiable stochastic failure probability per gauge (Rice upcrossing bound).

A gauge is a polyline (or a single point) in pixel coordinates (x=col, y=row) with a
sense: +1 = must print (latent R > threshold everywhere on it, e.g. trench centerline,
via center), -1 = must not print (R < threshold, e.g. the space between two features).

Failure event (exactly what the Monte-Carlo checker counts): R crosses to the wrong
side of the threshold at ANY sample point of the gauge. With Y = -sense*(R - E[R])
(noise toward failure) and margin m = sense*(E[R] - threshold), the event is
sup_x (Y - m) >= 0, and for a Gaussian Y

    P_fail <= Phi(-m0/s0) + integral rho(x) dl            (tight as P -> 0)
    rho    = phi(m/s)/s * sqrt(v_c) * psi((dm/dl - mu_c)/sqrt(v_c))

where (Y, dY/dl) is jointly Gaussian with Var Y = s^2, Cov = c, Var dY/dl = d^2,
mu_c = c m / s^2, v_c = d^2 - c^2/s^2, psi(a) = E[(Z - a)^+] = phi(a) - a Phi(-a).
This is Rice's formula for a non-stationary process crossing a moving level.

Event semantics: a centerline excursion is a necessary condition for a full
break/bridge across the feature, so P_fail upper-bounds the topological failure rate.
"""
from __future__ import annotations

import dataclasses as dc
import math

import torch
import torch.nn.functional as F

LOG_SQRT_2PI = 0.5 * math.log(2 * math.pi)


@dc.dataclass
class Gauge:
    points: list            # [(x, y), ...] polyline vertices in pixels; one vertex = point gauge
    sense: int              # +1 must print, -1 must stay dark
    kind: str = "gauge"     # break | bridge | missing | ...


def sample_path(points, step: float = 0.5, dtype=torch.float64):
    """Samples (N,2), unit tangents (N,2), arc-length weights (N,) in pixels."""
    p = torch.tensor(points, dtype=dtype).reshape(-1, 2)
    if len(p) == 1:
        return p, torch.tensor([[1.0, 0.0]], dtype=dtype), torch.zeros(1, dtype=dtype)
    pts, tans, wts = [], [], []
    for a, b in zip(p[:-1], p[1:]):
        seg = b - a
        length = float(seg.norm())
        n = max(int(math.ceil(length / step)), 1)
        t = torch.linspace(0, 1, n + 1, dtype=dtype)
        w = torch.full((n + 1,), length / n, dtype=dtype)
        w[0] = w[-1] = length / n / 2
        pts.append(a + t[:, None] * seg)
        tans.append((seg / length).expand(n + 1, 2))
        wts.append(w)
    return torch.cat(pts), torch.cat(tans), torch.cat(wts)


def _bilinear(maps: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    """maps (C, H, W), xy (N, 2) pixel coords -> (C, N)."""
    h, w = maps.shape[-2:]
    grid = torch.stack([2 * xy[:, 0] / (w - 1) - 1, 2 * xy[:, 1] / (h - 1) - 1], -1).to(maps.dtype)
    return F.grid_sample(maps[None], grid[None, None], mode="bilinear", align_corners=True)[0, :, 0]


def log_psi(a: torch.Tensor) -> torch.Tensor:
    """log E[(Z - a)^+], stable for all a."""
    a = a.clamp(-30.0, 1e3)
    ap, an = a.clamp(min=0), a.clamp(max=0)  # each branch only sees its safe domain (no NaN grads)
    mills = math.sqrt(math.pi / 2) * torch.special.erfcx(ap / math.sqrt(2))
    pos = -0.5 * ap * ap - LOG_SQRT_2PI + torch.log1p(-(ap * mills).clamp(max=1 - 1e-16))
    neg = torch.log(torch.exp(-0.5 * an * an - LOG_SQRT_2PI) - an * torch.special.ndtr(-an))
    return torch.where(a >= 0, pos, neg)


def _saddlepoint(aerial, card, xy, sense: float):
    """Exact-CGF tail of R at each sample point (Barndorff-Nielsen r*).

    R at a sub-pixel point is the bilinear interpolation of pixel latents (what the MC
    checker evaluates), so its weights are the separable discrete Gaussian, interpolated.
    Returns (u_eff, log_density): P(failure at the point) = Phi(-u_eff), and the density
    of R at the threshold for the crossing term.
    """
    from anamorph.resist.grf_stochastic import acid_scales
    r = card.resist
    mu_ph, mu_acid = acid_scales(card)
    q, t, s = r.quantum_yield, r.threshold, r.diffusion_nm / card.pixel_nm
    h = int(math.ceil(4 * s)) + 1
    d = torch.arange(-h - 1, h + 2, dtype=torch.float64)
    g = torch.exp(-d * d / (2 * s * s)) * (d.abs() <= h)
    g = g / g.sum()
    off = torch.arange(-h, h + 2)

    def axis(c, n):
        f = torch.floor(c)
        frac = (c - f)[:, None]
        w = (1 - frac) * g[off + h + 1] + frac * g[off + h]   # g(a) and g(a - 1)
        return (f.long()[:, None] + off) % n, w

    rows, wy = axis(xy[:, 1], aerial.shape[-2])
    cols, wx = axis(xy[:, 0], aerial.shape[-1])
    lam = mu_ph * aerial.to(torch.float64).clamp(min=0)[rows[:, :, None], cols[:, None, :]]
    w = wy[:, :, None] * wx[:, None, :] / mu_acid

    def cgf(th, lam):
        e = torch.exp(th[:, None, None] * w)
        hq = torch.exp(q * (e - 1))
        k0 = (lam * (hq - 1)).sum((1, 2))
        k1 = (lam * w * q * e * hq).sum((1, 2))
        k2 = (lam * w * w * hq * (q * e) * (q * e + 1)).sum((1, 2))
        return k0, k1, k2

    with torch.no_grad():
        lam0 = lam.detach()
        th = torch.zeros(len(xy), dtype=torch.float64)
        cap = 1.0 / w.amax((1, 2))
        for _ in range(100):
            _, k1, k2 = cgf(th, lam0)
            step = ((k1 - t) / k2).clamp(-cap, cap)
            th = th - step
            if step.abs().max() < 1e-10 * cap.min():
                break
    k0, k1, k2 = cgf(th, lam)
    th = th - (k1 - t) / k2          # one Newton step with grad: implicit-function derivative
    k0, _, k2 = cgf(th, lam)
    what = torch.sign(th) * torch.sqrt((2 * (th * t - k0)).clamp(min=1e-300))
    uhat = th * k2.sqrt()
    near = what.abs() < 1e-3
    safe_w = torch.where(near, torch.ones_like(what), what)
    safe_u = torch.where(near, torch.ones_like(uhat), uhat)
    rstar = torch.where(near, (t - k1) / k2.sqrt(), safe_w + torch.log(safe_u / safe_w) / safe_w)
    log_dens = -0.5 * what * what - LOG_SQRT_2PI - 0.5 * k2.log()
    return -sense * rstar, log_dens


def log_pfail(model, aerial: torch.Tensor, gauges: list[Gauge], tail: str = "saddlepoint",
              step: float = 0.5) -> torch.Tensor:
    """log P_fail per gauge, shape (G,). model: GRFModel; tail: 'saddlepoint' | 'gaussian'."""
    card = model.card
    threshold, pixel_nm = card.resist.threshold, card.pixel_nm
    maps = model.maps(aerial)
    names = ["mean", "gx", "gy", "var", "cx", "cy", "dxx", "dyy", "dxy"]
    stack = torch.stack([maps[k] for k in names]).to(torch.float64)
    out = []
    for g in gauges:
        xy, tan, wt = sample_path(g.points, step)
        v = dict(zip(names, _bilinear(stack, xy)))
        ex, ey = tan[:, 0], tan[:, 1]
        s = float(g.sense)
        sd = v["var"].clamp(min=1e-30).sqrt()
        m = s * (v["mean"] - threshold)
        dm = s * (ex * v["gx"] + ey * v["gy"])
        c = ex * v["cx"] + ey * v["cy"]
        d2 = ex * ex * v["dxx"] + 2 * ex * ey * v["dxy"] + ey * ey * v["dyy"]
        if tail == "saddlepoint":
            u, log_dens = _saddlepoint(aerial, card, xy, s)
        elif tail == "gaussian":
            u = m / sd
            log_dens = -0.5 * u * u - LOG_SQRT_2PI - sd.log()
        else:
            raise ValueError(tail)
        log_point = torch.special.log_ndtr(-u[0])
        if len(xy) == 1:
            out.append(log_point)
            continue
        vc = (d2 - c * c / sd**2).clamp(min=1e-30)
        a = (dm - c * m / sd**2) / vc.sqrt()
        log_rate = log_dens + 0.5 * vc.log() + log_psi(a)
        log_cross = torch.logsumexp(log_rate + torch.log(wt * pixel_nm), 0)
        out.append(torch.logaddexp(log_point, log_cross))
    return torch.stack(out)
