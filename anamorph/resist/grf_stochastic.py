"""Gaussian-random-field model of shot noise + acid statistics in the latent image.

Generative model (the Monte-Carlo checker in harness/mc_stochastic.py samples exactly this):
  photons per pixel  N ~ Poisson(mu_ph * I),        mu_ph = photons/nm^2 * absorption * pixel area
  acids per pixel    A | N ~ Poisson(q * N)          (compound Poisson, q = quantum yield)
  latent image       R = K * (A / mu_acid),          mu_acid = q * mu_ph   (so E[R] = K * I)

Moments used by the analytic model (exact for this generative model):
  Var A     = mu_acid * I * (1 + q)
  kappa3(A) = mu_ph * I * (q^3 + 3 q^2 + q)
  Cov(R(x), R(x')) = sum_y K(x-y) K(x'-y) Var A(y) / mu_acid^2
  CGF  log E exp(theta R(x)) = sum_y mu_ph I_y (exp(q (exp(theta K(x-y)/mu_acid) - 1)) - 1)
The GRF treats R as Gaussian with these first two moments; the exact CGF is used by
the saddlepoint tail in rice_pfail.py, because shot-noise tails are strongly skewed.
"""
from __future__ import annotations

import torch

from anamorph.card import Card
from anamorph.resist.compact import CompactResist, GaussKernels


def acid_scales(card: Card) -> tuple[float, float]:
    r = card.resist
    mu_ph = card.photons_per_nm2 * r.absorption * card.pixel_nm**2
    return mu_ph, mu_ph * r.quantum_yield


class GRFModel(CompactResist):
    def maps(self, aerial: torch.Tensor) -> dict:
        """Per-pixel mean, gradient and covariance maps of the latent image (all in nm units)."""
        if self.card.resist.diffusion_nm <= 0:
            raise ValueError("the stochastic model needs resist.diffusion_nm > 0")
        q = self.card.resist.quantum_yield
        _, mu_acid = acid_scales(self.card)
        g, c = self.kern, GaussKernels.conv
        v = aerial.clamp(min=0) * (1 + q) / mu_acid  # Var(A/mu_acid) per pixel
        return {
            "mean": c(aerial, g.k), "gx": c(aerial, g.kx), "gy": c(aerial, g.ky),
            "var": c(v, g.k2), "cx": c(v, g.kkx), "cy": c(v, g.kky),
            "dxx": c(v, g.kxx), "dyy": c(v, g.kyy), "dxy": c(v, g.kxy),
        }
