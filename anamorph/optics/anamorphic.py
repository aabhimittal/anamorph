"""Wafer <-> reticle geometry for anamorphic (Mx != My) projection optics.

The optimizer works on a square wafer grid. The mask writer, MRC deck and M3D all
live at the reticle, where a wafer pixel is Mx*dx wide and My*dy tall. With
High-NA (4x, 8x) a reticle min-width w becomes w/4 in x but only w/8 in y at the
wafer, i.e. the y direction tolerates features half as large.
"""
from __future__ import annotations

import dataclasses as dc

from anamorph.card import Card


def reticle_len_to_wafer_px(length_nm: float, card: Card) -> tuple[float, float]:
    """A reticle length expressed in wafer pixels along (x, y)."""
    mx, my = card.magnification
    return length_nm / mx / card.pixel_nm, length_nm / my / card.pixel_nm


def to_reticle(x_nm, y_nm, card: Card):
    mx, my = card.magnification
    return x_nm * mx, y_nm * my


def isotropic(card: Card) -> Card:
    """Ablation: the same card with isotropic magnification (Mx, Mx)."""
    return dc.replace(card, magnification=(card.magnification[0], card.magnification[0]))
