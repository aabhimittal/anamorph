"""Mask parameterizations: pixel (MOSAIC-style sigmoid) and level set.

pixel   : m = sigmoid(beta * theta)
levelset: m = sigmoid(-phi / eps), phi a signed distance (negative inside, pixels);
          reinit() restores phi to a true signed distance of its zero level set.
shortcut: no B-spline boundary parameterization yet; add it when shot-count/curvature
control needs fewer variables than pixels.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


def signed_distance(binary: np.ndarray) -> np.ndarray:
    inside = binary > 0.5
    return distance_transform_edt(~inside) - distance_transform_edt(inside) + 0.5 * np.where(inside, -1, 1)


class MaskParam(torch.nn.Module):
    def __init__(self, init: torch.Tensor, kind: str = "pixel", beta: float = 4.0, eps: float = 1.0):
        super().__init__()
        self.kind, self.beta, self.eps = kind, beta, eps
        init = init.to(torch.float64)
        if kind == "pixel":
            self.theta = torch.nn.Parameter((2 * init - 1) * 1.0)
        elif kind == "levelset":
            self.theta = torch.nn.Parameter(torch.from_numpy(signed_distance(init.numpy())))
        else:
            raise ValueError(kind)

    def forward(self) -> torch.Tensor:
        if self.kind == "pixel":
            return torch.sigmoid(self.beta * self.theta)
        return torch.sigmoid(-self.theta / self.eps)

    @torch.no_grad()
    def reinit(self):
        if self.kind == "levelset":
            self.theta.copy_(torch.from_numpy(signed_distance((self.theta < 0).numpy().astype(float))))

    def binary(self) -> torch.Tensor:
        return (self.forward() > 0.5).to(torch.float64).detach()
