import math

import pytest
import torch

from anamorph.card import Card, Resist, Source
from anamorph.harness.mc_stochastic import mc_pfail, verdict
from anamorph.optics.m3d import thin_mask
from anamorph.optics.socs import SOCS
from anamorph.resist.grf_stochastic import GRFModel
from anamorph.resist.rice_pfail import Gauge, log_pfail, log_psi

N = 64


def _setup(dose, thr):
    card = Card(source=Source(shape="dipole", axis="x", sigma_in=0.4, sigma_out=0.9, step=0.1),
                resist=Resist(threshold=thr, diffusion_nm=3.0, dose_mj_cm2=dose), pixel_nm=1.0)
    m = torch.zeros(N, N, dtype=torch.float64)
    for x0 in range(0, N, 32):
        m[:, x0:x0 + 16] = 1
    i = SOCS.build(card, (N, N), dtype=torch.complex128)(thin_mask(m, card))
    return card, i, GRFModel(card, (N, N))


def test_log_psi_matches_definition():
    import mpmath
    mpmath.mp.dps = 50
    vals = [-5.0, -1.0, 0.0, 0.5, 3.0, 8.0, 20.0]
    ref = [float(mpmath.log(mpmath.npdf(a) - a * mpmath.ncdf(-a))) for a in vals]
    assert torch.allclose(log_psi(torch.tensor(vals, dtype=torch.float64)), torch.tensor(ref, dtype=torch.float64), rtol=1e-9)


@pytest.mark.parametrize("gauge", [Gauge([(56, 20), (56, 44)], -1, "bridge"), Gauge([(56, 32)], -1, "pt")])
def test_saddlepoint_matches_monte_carlo(gauge):
    """The HARNESS gate: model P_fail within 2x of an independent MC sampler."""
    card, i, model = _setup(dose=10, thr=0.2)
    p = float(log_pfail(model, i, [gauge]).exp())
    mc = mc_pfail(i.numpy(), card, gauge.points, gauge.sense, trials=20000, seed=1)
    assert 1e-3 < mc["p"] < 0.1
    assert verdict(p, mc) == "pass", (p, mc)


def test_gaussian_tail_underestimates_upper_tail():
    """Documented finding: Poisson skew makes the plain GRF anti-conservative for bridges."""
    card, i, model = _setup(dose=10, thr=0.2)
    g = [Gauge([(56, 32)], -1, "pt")]
    assert float(log_pfail(model, i, g, "gaussian")) < float(log_pfail(model, i, g)) - math.log(3)


def test_tails_agree_at_high_dose_near_mean():
    """CLT regime: at high dose and a 2-sigma level the saddlepoint reduces to the Gaussian."""
    card, i, model = _setup(dose=5000, thr=0.35)
    mp = model.maps(i)
    thr = float(mp["mean"][32, 40] - 2 * mp["var"][32, 40].sqrt())
    model = GRFModel(card.replace(threshold=thr), (N, N))
    g = [Gauge([(40, 32)], +1, "pt")]
    a, b = float(log_pfail(model, i, g, "gaussian")), float(log_pfail(model, i, g))
    assert abs(a - math.log(0.02275)) < 0.01 and abs(a - b) < 0.05


def test_pfail_gradient_is_finite():
    card, i, model = _setup(dose=10, thr=0.2)
    i = i.clone().requires_grad_(True)
    lp = log_pfail(model, i, [Gauge([(56, 20), (56, 44)], -1), Gauge([(40, 20), (40, 44)], +1), Gauge([(40, 32)], +1)])
    lp.sum().backward()
    assert torch.isfinite(i.grad).all() and i.grad.abs().sum() > 0


def test_verdict_rules():
    mc = {"p": 1e-2, "k": 100, "n": 10000, "ci_lo": 8e-3, "ci_hi": 1.2e-2}
    assert verdict(1.5e-2, mc) == "pass" and verdict(3e-2, mc) == "fail"
    none = {"p": 0.0, "k": 0, "n": 10000, "ci_lo": 0.0, "ci_hi": 3.7e-4}
    assert verdict(1e-9, none) == "model-only" and verdict(1e-2, none) == "fail"
