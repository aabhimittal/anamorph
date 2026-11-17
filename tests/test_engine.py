import dataclasses as dc
from pathlib import Path

import numpy as np
import pytest
import torch

from anamorph.agent.boundary import SCHEMA, check_evolve_edit, check_paths, load_knobs
from anamorph.bench.euv_clips import line_space, suite
from anamorph.card import load_card
from anamorph.fullchip.tiling import run_tiled
from anamorph.harness.run import record, set_determinism
from anamorph.io.gds import read_raster, write_mask
from anamorph.mask.mrc_morph import mrc_violations, se_size
from anamorph.opt.ilt import Knobs, evaluate, optimize
from anamorph.opt.metrics import HIGHNA, epe_violations, rectangles

ROOT = Path(__file__).resolve().parents[1]
CARD = load_card(ROOT / "cards/highna_euv.yaml")


def test_anamorphic_mrc_is_anisotropic():
    ky, kx = se_size(32, CARD)  # 32 nm at the reticle -> 8 nm (x), 4 nm (y) at the wafer
    assert (ky, kx) == (3, 7)
    m = np.zeros((64, 64))
    m[:, 30:35] = 1             # 5 px wide in x: violates the 7 px x-rule
    assert mrc_violations(m, CARD) > 0
    assert mrc_violations(m.T.copy(), CARD) == 0   # 5 px tall in y: fine under the 3 px y-rule


def test_epe_and_shots():
    t = np.zeros((128, 128))
    t[30:90, 40:52] = 1
    assert epe_violations(t, t, 1.0, HIGHNA) == 0
    shifted = np.roll(t, 5, axis=1)
    assert epe_violations(shifted, t, 1.0, HIGHNA) > 0
    assert len(rectangles(t)) == 1
    t[60:70, 52:60] = 1
    assert len(rectangles(t)) == 3


def test_ilt_improves_and_is_deterministic():
    clip = line_space(24, 12, periods=2)
    k = Knobs(iters=60)
    a = optimize(clip.target, CARD, k, clip.gauges)
    b = optimize(clip.target, CARD, k, clip.gauges)
    assert torch.equal(a["mask"], b["mask"])
    base = evaluate(clip.target, clip.target, CARD, a["litho"])
    assert evaluate(a["mask"], clip.target, CARD, a["litho"])["l2"] < base["l2"]


def test_stochastic_term_moves_worst_gauge():
    clip = line_space(24, 12, periods=2)
    ev = []
    for lam in (0.0, 1.0):
        r = optimize(clip.target, CARD, Knobs(iters=60, lambda_pfail=lam), clip.gauges)
        ev.append(max(evaluate(r["mask"], clip.target, CARD, r["litho"], clip.gauges)["log10_pfail"]))
    assert ev[1] < ev[0] - 0.3


def test_knob_schema_matches_dataclass_and_rejects_junk():
    assert set(SCHEMA["properties"]) == {f.name for f in dc.fields(Knobs)}
    assert load_knobs({"lambda_pfail": 2.0}).lambda_pfail == 2.0
    for bad in ({"lambda_pfail": -1}, {"threshold": 0.3}, {"mask": "spline"}):
        with pytest.raises(Exception):
            load_knobs(bad)


def test_evolve_gate():
    src = (ROOT / "anamorph/opt/ilt.py").read_text()
    inside = src.replace("    hard = it >= start\n", "    hard = it >= start - 1\n")
    outside = src.replace("self.nom = SOCS.build", "self.nom = SOCS.build  # tweak\n        self.nom = SOCS.build")
    assert inside != src and check_evolve_edit(src, inside) == []
    assert check_evolve_edit(src, outside)
    assert check_evolve_edit(src, src.replace("# EVOLVE-BLOCK-END", "")) != []
    assert check_paths(["anamorph/resist/rice_pfail.py"]) and not check_paths(["anamorph/opt/ilt.py"])


def test_tiling_identity_and_healing():
    t = (torch.rand(100, 140, generator=torch.Generator().manual_seed(0)) > 0.5).double()
    assert torch.equal(run_tiled(t, lambda w, init: w.clone(), tile=32, halo=8), t)
    assert torch.equal(run_tiled(t, lambda w, init: init.clone(), tile=32, halo=8, heal=True), t)


def test_gds_roundtrip(tmp_path):
    m = np.zeros((40, 50))
    m[5:20, 10:30] = 1
    m[25:35, 5:45] = 1
    write_mask(m, tmp_path / "m.gds", pixel_nm=2.0)
    back = read_raster(str(tmp_path / "m.gds"), 1, 0, (0, 0, 100, 80), 2.0)
    assert np.array_equal(back, m)


def test_run_record_is_content_addressed(tmp_path):
    set_determinism(0)
    clip = suite()[0]
    d1 = record(tmp_path, 0, {"x": torch.ones(3), "score": 1.0}, target=clip.target, knobs={"a": 1})
    d2 = record(tmp_path, 0, {"x": torch.ones(3), "score": 1.0}, target=clip.target, knobs={"a": 1})
    d3 = record(tmp_path, 1, {"x": torch.ones(3), "score": 1.0}, target=clip.target, knobs={"a": 1})
    assert d1 == d2 != d3 and (d1 / "manifest.json").exists()
