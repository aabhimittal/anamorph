"""Export the engine's forward model and Tier-B results for the static demo page.

  python demo/build_data.py            # writes demo/data.json (~3 min on CPU)

The page recomputes aerial/resist in the browser from the exported SOCS kernels; P_fail,
optimized masks and the Monte-Carlo tail check come from the Python engine.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from anamorph.bench import euv_clips
from anamorph.card import Card, Resist, Source, load_card
from anamorph.harness.mc_stochastic import mc_pfail
from anamorph.harness.run import set_determinism
from anamorph.optics.m3d import thin_mask
from anamorph.optics.socs import SOCS
from anamorph.opt.ilt import Knobs, evaluate, optimize
from anamorph.opt.metrics import HIGHNA
from anamorph.resist.compact import CompactResist
from anamorph.resist.grf_stochastic import GRFModel
from anamorph.resist.rice_pfail import Gauge, log_pfail

ROOT = Path(__file__).resolve().parent.parent
K_BROWSER = 24  # aerial error <= 0.0075 vs the 0.995-energy set on all three clips
CLIPS = {"ls": ("Line/space, pitch 24 nm, CD 12 nm", euv_clips.line_space(24, 12)),
         "t2t": ("Tip-to-tip, pitch 24 nm, gap 16 nm", euv_clips.tip_to_tip(24, 12, 16)),
         "via": ("Via array, pitch 32 nm, CD 16 nm", euv_clips.via_array(32, 16))}
r6 = lambda x: float(f"{x:.6g}")
bits = lambda m: "".join("1" if v else "0" for v in (m.flatten() > 0.5).tolist())


def grid(card, n):
    s = SOCS.build(card, (n, n), 0.0, 0.995)
    k = s.kernels[:K_BROWSER].reshape(K_BROWSER, -1)
    sup = (s.kernels[0].reshape(-1).abs() > 0).nonzero().squeeze(1)
    h = k[:, sup]
    return s, {"n": n, "norm": r6(s.norm), "support": sup.tolist(), "k_total": s.kernels.shape[0],
               "re": [[r6(v) for v in row] for row in h.real.tolist()],
               "im": [[r6(v) for v in row] for row in h.imag.tolist()]}


def probes(card, s, target, n, count=48):
    """Reference blurred latent image at a few pixels, from the same 24 kernels in torch."""
    trunc = SOCS(s.kernels[:K_BROWSER], s.norm)
    lat = CompactResist(card, (n, n)).latent(trunc(thin_mask(target, card)).to(torch.float64))
    idx = torch.randperm(n * n, generator=torch.Generator().manual_seed(0))[:count]
    return [[int(i), r6(float(lat.reshape(-1)[i]))] for i in idx]


def tail_sweep(trials=20000):
    """README finding 1 at the operating point tests/test_stochastic.py validates: 16 nm
    lines/spaces on a 64 nm tile, dipole, dose 10 mJ/cm2, threshold swept."""
    card = Card(source=Source(shape="dipole", axis="x", sigma_in=0.4, sigma_out=0.9, step=0.1),
                resist=Resist(threshold=0.2, diffusion_nm=3.0, dose_mj_cm2=10), pixel_nm=1.0)
    m = torch.zeros(64, 64, dtype=torch.float64)
    m[:, 0:16] = m[:, 32:48] = 1
    i = SOCS.build(card, (64, 64), dtype=torch.complex128)(thin_mask(m, card))
    out = []
    for g, thrs in ((Gauge([(56, 32)], -1, "bridge"), [0.12 + 0.02 * k for k in range(7)]),
                    (Gauge([(40, 32)], +1, "break"), [0.30 + 0.02 * k for k in range(9)])):
        pts = []
        for thr in thrs:
            c = card.replace(threshold=round(thr, 2))
            model = GRFModel(c, (64, 64))
            pg, ps = (float(log_pfail(model, i, [g], t).exp()) for t in ("gaussian", "saddlepoint"))
            mc = mc_pfail(i.numpy(), c, g.points, g.sense, trials, seed=0)
            pts.append({"threshold": round(thr, 2), "p_gauss": r6(pg), "p_saddle": r6(ps), "p_mc": r6(mc["p"]),
                        "ci": [r6(mc["ci_lo"]), r6(mc["ci_hi"])]})
        out.append({"kind": g.kind, "trials": trials, "points": pts})
    return out


def main():
    card = load_card(ROOT / "cards/highna_euv.yaml")
    grids, clips = {}, []
    for key, (label, clip) in CLIPS.items():
        n = clip.target.shape[0]
        if n not in grids:
            grids[n] = grid(card, n)
        s, g = grids[n]
        row = {"key": key, "label": label, "n": n, "target": bits(clip.target),
               "gauges": [{"points": gg.points, "sense": gg.sense, "kind": gg.kind} for gg in clip.gauges],
               "probes": probes(card, s, clip.target, n), "masks": {}}
        for cfg, knobs in (("target", None), ("epe_only", Knobs(iters=100, w_mrc=1.0)),
                           ("stochastic", Knobs(iters=100, w_mrc=1.0, lambda_pfail=1.0))):
            if knobs is None:  # the drawn target used as the mask: the no-OPC baseline
                mask, litho = clip.target, None
            else:
                set_determinism(knobs.seed)
                r = optimize(clip.target, card, knobs, clip.gauges)
                mask, litho = r["mask"], r["litho"]
            ev = evaluate(mask, clip.target, card, litho, clip.gauges, epe_rule=HIGHNA)
            row["masks"][cfg] = {"bits": bits(mask), "l2": ev["l2"], "pvb": ev["pvb"], "epe": ev["epe"],
                                 "mrc": ev["mrc"], "log10_pfail": [r6(v) for v in ev["log10_pfail"]]}
            print(key, cfg, {k: v for k, v in row["masks"][cfg].items() if k not in ("bits", "log10_pfail")},
                  round(max(ev["log10_pfail"]), 2), flush=True)
        clips.append(row)
    tails = tail_sweep()
    r = card.resist
    out = {"card": {"name": card.name, "wavelength_nm": card.wavelength_nm, "na": card.na,
                    "magnification": list(card.magnification), "obscuration": card.obscuration,
                    "absorber_amplitude": card.absorber_amplitude, "absorber_phase_deg": card.absorber_phase_deg,
                    "threshold": r.threshold, "diffusion_nm": r.diffusion_nm, "dose_mj_cm2": r.dose_mj_cm2,
                    "mrc_reticle_nm": card.mrc_width_reticle_nm, "k_browser": K_BROWSER},
           "grids": {str(n): g for n, (_, g) in grids.items()}, "clips": clips, "tails": tails}
    (ROOT / "demo/data.json").write_text(json.dumps(out, separators=(",", ":")))
    assert all(math.isfinite(t["p_saddle"]) for s in tails for t in s["points"])


if __name__ == "__main__":
    main()
