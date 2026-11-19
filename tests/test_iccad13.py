import torch

from anamorph.bench.iccad13 import ICCAD13Litho, glp_target, simple_ilt


def _glp(path, body):
    path.write_text("BEGIN\nEQUIV 1 1000 MICRON +X,+Y\nCNAME T\nLEVEL M1\nCELL T PRIME\n" + body + "ENDMSG\n")
    return path


def test_inclusive_raster_matches_cv2_fillpoly_rule(tmp_path):
    """cv2.fillPoly fills the closed polygon on the integer grid: a 4-wide RECT covers 5 px,
    and the concave corner of an L belongs to it (both checked against cv2 by hand)."""
    f = _glp(tmp_path / "a.glp", "   RECT N M1  2 3 4 5\n   PGON N M1  10 1 18 1 18 4 14 4 14 9 10 9\n")
    t = glp_target(f, size=32, inclusive=True).numpy()
    c = glp_target(f, size=32).numpy()
    assert t.sum() == 5 * 6 + (9 * 4 + 5 * 5) and c.sum() == 4 * 5 + (8 * 3 + 4 * 5)


def _fake_kernels(root, n=24, k=7):
    g = torch.Generator().manual_seed(0)
    for sub in ("kernels", "scales"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    for name, kk in (("focus", n), ("defocus", n), ("combo_focus", 1), ("combo_ct_focus", 1),
                     ("combo_defocus", 1), ("combo_ct_defocus", 1)):
        torch.save(torch.randn(k, k, kk, dtype=torch.complex64, generator=g) * 0.3, root / "kernels" / f"{name}.pt")
    for name in ("focus", "defocus"):
        torch.save(torch.rand(n, generator=g), root / "scales" / f"{name}.pt")


def test_simpleilt_protocol_runs_both_gradients(tmp_path):
    _fake_kernels(tmp_path)
    t = torch.zeros(64, 64)
    t[20:44, 28:36] = 1
    for grad in ("exact", "openilt"):
        litho = ICCAD13Litho(tmp_path, (64, 64), gradient=grad)
        m = simple_ilt(t, litho, iters=5, offset=16, size=32)
        assert m.shape == t.shape and torch.isfinite(m).all()
        assert torch.equal(m[:16], torch.sigmoid(4 * (2 * t[:16] - 1)))  # outside the window stays frozen
