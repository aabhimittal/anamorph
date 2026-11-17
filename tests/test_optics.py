import dataclasses as dc

import pytest
import torch

from anamorph.card import Card, Source
from anamorph.harness.rigorous_ref import socs_error
from anamorph.optics.abbe import abbe_image
from anamorph.optics.anamorphic import isotropic
from anamorph.optics.m3d import BoundaryLayerM3D, fit, thin_mask
from anamorph.optics.pupil import pupil_components
from anamorph.optics.socs import SOCS

CARD = Card(source=Source(shape="annular", sigma_in=0.3, sigma_out=0.8, step=0.1), pixel_nm=2.0,
            obscuration=0.2, absorber_amplitude=0.15, absorber_phase_deg=30)


def _mask(n=64):
    m = torch.zeros(n, n, dtype=torch.float64)
    m[20:44, 28:36] = 1
    m[10:14, 10:50] = 1
    return m


@pytest.mark.parametrize("vector", [True, False])
def test_socs_full_rank_equals_abbe(vector):
    card = dc.replace(CARD, vector=vector)
    t = thin_mask(_mask(), card)
    ref = abbe_image(t, card, focus_nm=15)
    fast = SOCS.build(card, (64, 64), 15, energy=1.0, dtype=torch.complex128)(t)
    assert (fast - ref).abs().max() < 1e-12


def test_truncated_socs_within_tolerance():
    assert socs_error(_mask(), CARD, energy=0.995) < 5e-3


def test_clear_field_is_one():
    t = thin_mask(torch.ones(32, 32, dtype=torch.float64), CARD)
    assert abs(float(abbe_image(t, CARD).mean()) - 1) < 1e-12


def test_vector_field_unit_norm_and_low_na_scalar_limit():
    k = torch.linspace(-0.04, 0.04, 21, dtype=torch.float64)
    kx, ky = torch.meshgrid(k, k, indexing="xy")
    card = dc.replace(CARD, obscuration=0.0)
    p = pupil_components(kx, ky, card)
    base = pupil_components(kx, ky, dc.replace(card, vector=False))[0].abs().pow(2)  # radiometric^2
    inside = base > 0
    # per incident polarization the field is a unit vector: |E|^2 = 1/2 (unpolarized split) x base
    for pol in (p[:3], p[3:]):
        assert torch.allclose(pol.abs().pow(2).sum(0)[inside] / base[inside], torch.tensor(0.5, dtype=torch.float64))
    lowc = dc.replace(CARD, na=0.05, obscuration=0.0, pixel_nm=20.0)
    a = abbe_image(thin_mask(_mask(), lowc), lowc)
    b = abbe_image(thin_mask(_mask(), lowc), dc.replace(lowc, vector=False))
    assert (a - b).abs().max() < 2e-3


def test_obscuration_blocks_center():
    k = torch.zeros(1, dtype=torch.float64)
    assert pupil_components(k, k, CARD).abs().max() == 0
    assert pupil_components(k, k, dc.replace(CARD, obscuration=0)).abs().max() > 0


def test_thin_mask_anamorphic_effect_is_small():
    """In a thin-mask model anamorphism only enters via the radiometric factor."""
    t = thin_mask(_mask(), CARD)
    d = (abbe_image(t, CARD) - abbe_image(t, isotropic(CARD))).abs().max()
    assert 0 < d < 1e-2


def test_m3d_boundary_layer_learns_a_shadowing_operator():
    torch.manual_seed(0)
    masks = (torch.rand(4, 32, 32) > 0.7).double()
    true = BoundaryLayerM3D(5).double()
    with torch.no_grad():
        true.kernel.normal_(0, 0.05)
        ref = true(masks, CARD)
    model = BoundaryLayerM3D(5).double()
    hist = fit(model, masks, ref, CARD, iters=200, lr=0.02)
    assert hist[-1] < 1e-3 * hist[0]
