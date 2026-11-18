"""Compact resist: Gaussian acid blur of the aerial image, then a (soft) threshold.

R(x) = (K * I)(x) with K a normalized Gaussian of std `diffusion_nm`; printed = R > threshold.
All convolutions are periodic FFT products, consistent with the FFT optics.
"""
from __future__ import annotations

import torch

from anamorph.card import Card


def _signed_dist(n: int, d: float, dtype) -> torch.Tensor:
    return torch.fft.fftfreq(n, 1.0 / n, dtype=dtype) * d  # 0, 1, ..., -1 (wrapped) in nm


class GaussKernels:
    """Spectra of the blur kernel K, its derivatives, and the products the GRF needs."""

    def __init__(self, sigma_nm: float, shape, pixel_nm: float, dtype=torch.float64, device="cpu"):
        y = _signed_dist(shape[0], pixel_nm, dtype)[:, None]
        x = _signed_dist(shape[1], pixel_nm, dtype)[None, :]
        if sigma_nm > 0:
            k = torch.exp(-(x * x + y * y) / (2 * sigma_nm**2))
            k = k / k.sum()
            kx, ky = -x / sigma_nm**2 * k, -y / sigma_nm**2 * k  # d/dx K in 1/nm
        else:
            k = torch.zeros(shape, dtype=dtype)
            k[0, 0] = 1.0
            kx = ky = torch.zeros(shape, dtype=dtype)
        sp = lambda a: torch.fft.fft2(a.to(device))
        self.k, self.kx, self.ky = sp(k), sp(kx), sp(ky)
        self.k2, self.kkx, self.kky = sp(k * k), sp(k * kx), sp(k * ky)
        self.kxx, self.kyy, self.kxy = sp(kx * kx), sp(ky * ky), sp(kx * ky)

    @staticmethod
    def conv(img: torch.Tensor, spec: torch.Tensor) -> torch.Tensor:
        return torch.fft.ifft2(torch.fft.fft2(img.to(spec.dtype)) * spec).real.to(img.dtype)


class CompactResist:
    def __init__(self, card: Card, shape, device="cpu", dtype=torch.float64):
        self.card = card
        self.kern = GaussKernels(card.resist.diffusion_nm, shape, card.pixel_nm, dtype, device)

    def latent(self, aerial: torch.Tensor) -> torch.Tensor:
        return GaussKernels.conv(aerial, self.kern.k)

    def soft_print(self, aerial: torch.Tensor) -> torch.Tensor:
        r = self.card.resist
        return torch.sigmoid(r.steepness * (self.latent(aerial) - r.threshold))

    def hard_print(self, aerial: torch.Tensor) -> torch.Tensor:
        return (self.latent(aerial) > self.card.resist.threshold).to(aerial.dtype)
