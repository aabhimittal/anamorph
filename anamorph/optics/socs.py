"""Hopkins/SOCS imaging: the fast differentiable forward model.

The TCC is never formed explicitly. With A[(s,c), f] = sqrt(w_s) P_c(f + f_s) over the
frequency support, TCC = A^H A, so an SVD A = U S Vh gives the SOCS kernels directly:
H_k(f) = S_k * Vh[k, f] and I(x) = sum_k |IFFT[T(f) H_k(f)]|^2. Truncating at K kernels
keeps a fraction `energy` of trace(TCC). Vector imaging is the same algebra with the
polarization/field components folded into the row index.
"""
from __future__ import annotations

import torch

from anamorph.card import Card
from anamorph.optics.pupil import check_sampling, freq_grid, pupil_components, source_points

_CACHE: dict = {}


class SOCS:
    def __init__(self, kernels: torch.Tensor, norm: float, energy_kept: float = 1.0):
        self.kernels = kernels      # (K, H, W) complex, FFT ordering
        self.norm = norm            # clear-field intensity
        self.energy_kept = energy_kept

    @classmethod
    def build(cls, card: Card, shape, focus_nm: float = 0.0, energy: float = 0.9999,
              max_kernels: int | None = None, dtype=torch.complex64, device="cpu") -> "SOCS":
        key = (card.digest(), tuple(shape), focus_nm, energy, max_kernels, dtype, str(device))
        if key in _CACHE:
            return _CACHE[key]
        dx = dy = card.pixel_nm
        check_sampling(card, dx, dy)
        fy, fx = freq_grid(shape, dx, dy)
        fmax = card.na * (1 + card.source.sigma_out) / card.wavelength_nm
        support = (fx**2 + fy**2 <= (fmax * (1 + 1e-9)) ** 2).flatten().nonzero().squeeze(1)
        pts, w = source_points(card)
        fs = pts * card.na / card.wavelength_nm
        kx = fx.flatten()[support][None] + fs[:, 0, None]
        ky = fy.flatten()[support][None] + fs[:, 1, None]
        a = pupil_components(kx, ky, card, focus_nm) * w.sqrt()[None, :, None]  # (C, S, Nf)
        a = a.reshape(-1, a.shape[-1])
        dc = (fx.flatten()[support] == 0) & (fy.flatten()[support] == 0)
        norm = float(a[:, dc].abs().pow(2).sum())
        _, s, vh = torch.linalg.svd(a, full_matrices=False)
        e = s.pow(2).cumsum(0) / s.pow(2).sum()
        k = min(int((e < energy).sum()) + 1, len(s))
        if max_kernels:
            k = min(k, max_kernels)
        ker = torch.zeros(k, shape[0] * shape[1], dtype=torch.complex128)
        ker[:, support] = s[:k, None] * vh[:k]
        out = cls(ker.reshape(k, *shape).to(dtype).to(device), norm, float(e[k - 1]))
        _CACHE[key] = out
        return out

    @classmethod
    def from_spatial_kernels(cls, kernels: torch.Tensor, scales: torch.Tensor, shape, norm: float = 1.0) -> "SOCS":
        """Wrap externally supplied centered frequency-domain kernels (K, kh, kw), e.g. the
        ICCAD-13 contest kernels shipped with OpenILT, onto an (H, W) FFT grid."""
        k, kh, kw = kernels.shape
        iy = (torch.arange(kh) - kh // 2) % shape[0]
        ix = (torch.arange(kw) - kw // 2) % shape[1]
        ker = torch.zeros(k, *shape, dtype=torch.complex64)
        ker[:, iy[:, None], ix[None, :]] = kernels.to(torch.complex64) * scales.to(torch.float32).sqrt()[:, None, None]
        return cls(ker, norm)

    def field(self, t: torch.Tensor) -> torch.Tensor:
        """Coherent kernel fields (..., K, H, W)."""
        spec = torch.fft.fft2(t.to(self.kernels.dtype))
        return torch.fft.ifft2(spec.unsqueeze(-3) * self.kernels)

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """Aerial image (..., H, W), clear field = 1 (for built kernels)."""
        return self.field(t).abs().pow(2).sum(-3) / self.norm
