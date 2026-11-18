"""Stochastic-aware ILT: gradient descent on the mask through optics -> resist -> P_fail.

loss = w_l2 * L2 + w_pvb * PVB + lambda_pfail * log10(sum_g P_fail,g + floor)
       + w_mrc * MRC - w_nils * mean ILS + w_bin * mean 4m(1-m)

The binarization term matters: without it a pixel optimizer lowers the loss with grey
(partially transmitting) pixels that a real binary/two-level mask cannot reproduce.

L2 and PVB are normalized by target area. The process window follows the OpenILT
convention: outer corner = nominal focus at (1 + pw_dose), inner = defocus at (1 - pw_dose).
P_fail is evaluated at the nominal corner. sum_g P_g is the union bound on "this clip
fails", so its log concentrates the gradient on the worst gauges; a sum of per-gauge
logs would instead reward pushing already-negligible gauges lower. The floor keeps the
term bounded: below ~1e-12 the model is not predictive.
"""
from __future__ import annotations

import dataclasses as dc
import math
import time

import numpy as np
import torch

from anamorph.card import Card
from anamorph.mask.levelset import MaskParam
from anamorph.mask.mrc_morph import mrc_penalty, mrc_violations
from anamorph.optics.m3d import thin_mask
from anamorph.optics.socs import SOCS
from anamorph.opt.metrics import ICCAD13, epe_violations, ils_at_edges, shot_count
from anamorph.resist.grf_stochastic import GRFModel
from anamorph.resist.rice_pfail import Gauge, log_pfail


@dc.dataclass
class Knobs:
    """Agent-settable knobs; mirrors agent/knobs.schema.json."""
    iters: int = 100
    lr: float = 0.1
    mask: str = "pixel"            # pixel | levelset
    beta: float = 4.0
    w_l2: float = 1.0
    w_pvb: float = 1.0
    lambda_pfail: float = 0.0
    pfail_floor: float = 1e-12
    w_mrc: float = 0.0
    w_nils: float = 0.0
    w_bin: float = 0.5
    reinit_every: int = 0
    ste_frac: float = 0.3          # final fraction of iterations run on the binarized mask
    tail: str = "saddlepoint"
    socs_energy: float = 0.995
    seed: int = 0


# EVOLVE-BLOCK-START step_policy
def step_policy(it: int, knobs: Knobs) -> tuple[float, bool, bool]:
    """(learning rate, reinitialize level set now?, optimize the binarized mask now?).

    Two cosine cycles with a warm restart when the binarized (straight-through) phase
    begins: pixels can only flip if the step size there is still large.
    """
    start = int(knobs.iters * (1 - knobs.ste_frac))
    hard = it >= start
    pos, length = (it - start, knobs.iters - start) if hard else (it, start)
    lr = knobs.lr * 0.5 * (1 + math.cos(math.pi * pos / max(length, 1)))
    reinit = knobs.reinit_every > 0 and it > 0 and it % knobs.reinit_every == 0
    return lr, reinit, hard
# EVOLVE-BLOCK-END


class Litho:
    """Forward model bundle: SOCS at nominal/defocus + resist."""

    def __init__(self, card: Card, shape, energy: float = 0.995, m3d=None):
        self.card, self.m3d = card, m3d
        self.nom = SOCS.build(card, shape, 0.0, energy)
        self.dfc = SOCS.build(card, shape, card.pw_focus_nm, energy)
        self.resist = GRFModel(card, shape)

    def transmission(self, m):
        return self.m3d(m, self.card) if self.m3d is not None else thin_mask(m, self.card)

    def aerials(self, m):
        t = self.transmission(m)
        d = self.card.pw_dose
        i_nom = self.nom(t).to(torch.float64)
        return i_nom, i_nom * (1 + d), self.dfc(t).to(torch.float64) * (1 - d)


def optimize(target: torch.Tensor, card: Card, knobs: Knobs = Knobs(), gauges: list[Gauge] | None = None,
             init: torch.Tensor | None = None, m3d=None, log_every: int = 0, litho=None) -> dict:
    """litho: optional prebuilt forward model (e.g. bench.iccad13.ICCAD13Litho)."""
    torch.manual_seed(knobs.seed)
    target = target.to(torch.float64)
    litho = litho or Litho(card, target.shape, knobs.socs_energy, m3d)
    param = MaskParam(target if init is None else init, knobs.mask, knobs.beta)
    opt = torch.optim.Adam(param.parameters(), lr=knobs.lr)
    area = float(target.sum().clamp(min=1))
    hist = []
    best = (math.inf, param.binary())
    t0 = time.time()
    for it in range(knobs.iters):
        lr, reinit, hard = step_policy(it, knobs)
        if reinit:
            param.reinit()
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad()
        m_soft = param()
        # straight-through: forward on the binary mask, gradient through the soft one
        m = (m_soft > 0.5).to(m_soft.dtype) + m_soft - m_soft.detach() if hard else m_soft
        i_nom, i_max, i_min = litho.aerials(m)
        z = [litho.resist.soft_print(i) for i in (i_nom, i_max, i_min)]
        terms = {"l2": ((z[0] - target) ** 2).sum() / area,
                 "pvb": (((z[1] - target) ** 2).sum() + ((z[2] - target) ** 2).sum()) / area}
        loss = knobs.w_l2 * terms["l2"] + knobs.w_pvb * terms["pvb"]
        if knobs.lambda_pfail > 0 and gauges:
            lp = log_pfail(litho.resist, i_nom, gauges, knobs.tail)
            lp = torch.cat([lp, lp.new_tensor([math.log(knobs.pfail_floor)])])
            terms["log10_pfail"] = torch.logsumexp(lp, 0) / math.log(10)
            loss = loss + knobs.lambda_pfail * terms["log10_pfail"]
        if knobs.w_mrc > 0:
            terms["mrc"] = mrc_penalty(m, card)
            loss = loss + knobs.w_mrc * terms["mrc"]
        if knobs.w_bin > 0:
            terms["bin"] = (4 * m_soft * (1 - m_soft)).mean()
            loss = loss + knobs.w_bin * terms["bin"]
        if knobs.w_nils > 0:
            terms["ils"] = ils_at_edges(i_nom, target.numpy(), card.pixel_nm).mean()
            loss = loss - knobs.w_nils * terms["ils"]
        if hard and loss.item() - knobs.w_bin * terms.get("bin", loss.new_zeros(())).item() < best[0]:
            best = (loss.item() - knobs.w_bin * terms.get("bin", loss.new_zeros(())).item(), m.detach().clone())
        loss.backward()
        opt.step()
        hist.append({k: v.item() for k, v in terms.items()} | {"loss": loss.item(), "hard": hard})
        if log_every and it % log_every == 0:
            print(it, {k: round(v, 5) for k, v in hist[-1].items()}, flush=True)
    mask = best[1] if math.isfinite(best[0]) else param.binary()
    return {"mask": mask, "mask_soft": param().detach(), "history": hist,
            "seconds": time.time() - t0, "litho": litho}


@torch.no_grad()
def evaluate(mask: torch.Tensor, target: torch.Tensor, card: Card, litho: Litho | None = None,
             gauges: list[Gauge] | None = None, tail: str = "saddlepoint", epe_rule=ICCAD13) -> dict:
    """Binary-mask metrics. PVB in px (= nm^2 at 1 nm pixels), L2 in px."""
    litho = litho or Litho(card, mask.shape)
    i_nom, i_max, i_min = litho.aerials(mask.to(torch.float64))
    zn, zx, zm = (litho.resist.hard_print(i).numpy() for i in (i_nom, i_max, i_min))
    tg = target.numpy() > 0.5
    out = {"l2": int((zn.astype(bool) != tg).sum()), "pvb": int((zx != zm).sum()),
           "epe": epe_violations(zn, tg, card.pixel_nm, epe_rule), "shots": shot_count(mask.numpy()),
           "mrc": mrc_violations(mask.numpy(), card)}
    ils = ils_at_edges(i_nom, tg, card.pixel_nm)
    out["ils_min"], out["ils_mean"] = float(ils.min()), float(ils.mean())
    if gauges:
        lp = (log_pfail(litho.resist, i_nom, gauges, tail) / math.log(10)).numpy()
        for kind in sorted({g.kind for g in gauges}):
            sel = np.array([g.kind == kind for g in gauges])
            out[f"log10_pfail_{kind}_max"] = float(lp[sel].max())
        out["log10_pfail"] = lp.tolist()
    return out
