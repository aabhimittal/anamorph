"""Command line: Tier-A (ICCAD-13 parity), Tier-B (EUV clips), and P_fail validation.

  python -m anamorph tierb --clips ls,t2t,via --lambda 1.0 --iters 100
  python -m anamorph tiera --openilt path/to/OpenILT --cases 1,2,3
  python -m anamorph tiera --openilt path/to/OpenILT --protocol simpleilt --gradient openilt
  python -m anamorph tiera --openilt path/to/lithobench --protocol curvmulti --gradient openilt-exact
  python -m anamorph validate --clip ls --dose 10 --threshold 0.2 --trials 20000
"""
from __future__ import annotations

import argparse
import dataclasses as dc
import json

import torch
import torch.nn.functional as F

from anamorph.bench import euv_clips
from anamorph.card import load_card
from anamorph.harness.mc_stochastic import mc_pfail, verdict
from anamorph.harness.run import record, set_determinism
from anamorph.opt.ilt import Knobs, Litho, evaluate, optimize
from anamorph.opt.metrics import HIGHNA

CLIPS = {"ls": lambda: euv_clips.line_space(24, 12), "ls28": lambda: euv_clips.line_space(28, 14),
         "t2t": lambda: euv_clips.tip_to_tip(24, 12, 16), "via": lambda: euv_clips.via_array(32, 16),
         "via36": lambda: euv_clips.via_array(36, 18)}


def _validate(litho, mask, clip, trials, seed=0):
    """MC-check the worst gauge's analytic P_fail; returns (log10 p_model, mc dict, verdict)."""
    i_nom = litho.aerials(mask.to(torch.float64))[0].detach()
    ev = evaluate(mask, clip.target, litho.card, litho, clip.gauges)
    worst = max(range(len(clip.gauges)), key=lambda i: ev["log10_pfail"][i])
    g = clip.gauges[worst]
    mc = mc_pfail(i_nom.numpy(), litho.card, g.points, g.sense, trials, seed)
    return g.kind, ev["log10_pfail"][worst], mc, verdict(10 ** ev["log10_pfail"][worst], mc)


def tierb(a):
    card = load_card(a.card)
    configs = {"epe_only": Knobs(iters=a.iters, w_mrc=a.w_mrc),
               "nils": Knobs(iters=a.iters, w_mrc=a.w_mrc, w_nils=a.w_nils),
               "stochastic": Knobs(iters=a.iters, w_mrc=a.w_mrc, lambda_pfail=a.lam)}
    for name in a.clips.split(","):
        clip = CLIPS[name]()
        for cfg, knobs in configs.items():
            set_determinism(knobs.seed)
            r = optimize(clip.target, card, knobs, clip.gauges)
            ev = evaluate(r["mask"], clip.target, card, r["litho"], clip.gauges, epe_rule=HIGHNA)
            row = {"clip": clip.name, "config": cfg, "seconds": round(r["seconds"], 1)}
            row |= {k: v for k, v in ev.items() if k != "log10_pfail"}
            if a.trials:
                kind, lp, mc, v = _validate(r["litho"], r["mask"], clip, a.trials)
                row |= {"mc_gauge": kind, "mc_p": mc["p"], "mc_ci": [mc["ci_lo"], mc["ci_hi"]], "verdict": v}
            if a.out:
                row["run_dir"] = str(record(a.out, knobs.seed, {"mask": r["mask"], **row}, target=clip.target,
                                            card=dc.asdict(card), knobs=dc.asdict(knobs)))
            print(json.dumps(row), flush=True)


def tiera(a):
    """--protocol simpleilt / curvmulti reproduce OpenILT's SimpleILT and LithoBench's CurvMulti
    (use --gradient openilt / openilt-exact to match their hand-written backward passes) and
    report the deviation from the published per-case numbers."""
    import time
    from anamorph.bench.iccad13 import (ICCAD13_CARD, PUBLISHED_CURVMULTI, PUBLISHED_SIMPLEILT, ICCAD13Litho,
                                        curv_multi, glp_target, simple_ilt)
    kdir, bdir = f"{a.openilt}/kernel", f"{a.openilt}/benchmark/ICCAD2013"
    full = ICCAD13Litho(kdir, (2048, 2048), gradient=a.gradient)
    if a.protocol == "anamorph":
        n = 2048 // a.scale
        coarse = ICCAD13Litho(kdir, (n, n), gradient=a.gradient)
    for case in a.cases.split(","):
        t = glp_target(f"{bdir}/M1_test{case}.glp", inclusive=True)  # cv2.fillPoly-identical target
        t0 = time.time()
        if a.protocol == "simpleilt":
            m = (simple_ilt(t.float(), full) >= 0.5).double()
        elif a.protocol == "curvmulti":
            scale, m = list(curv_multi(f"{bdir}/M1_test{case}.glp", kdir, a.gradient))[-1]
            m = F.interpolate(m[None, None].double(), scale_factor=scale, mode="nearest")[0, 0]
        else:
            tc = F.avg_pool2d(t[None, None], a.scale)[0, 0].round()
            r = optimize(tc, ICCAD13_CARD, Knobs(iters=a.iters, lr=a.lr), litho=coarse)
            m = F.interpolate(r["mask"][None, None], scale_factor=a.scale, mode="nearest")[0, 0]
        ev = evaluate(m, t, ICCAD13_CARD, full)
        row = {"case": int(case), "seconds": round(time.time() - t0, 1)} | ev
        l2, pvb, epe = (PUBLISHED_CURVMULTI if a.protocol == "curvmulti" else PUBLISHED_SIMPLEILT)[int(case)]
        row |= {"l2_vs_pub_pct": round(100 * (ev["l2"] / l2 - 1), 2), "pvb_vs_pub_pct": round(100 * (ev["pvb"] / pvb - 1), 2)}
        print(json.dumps(row), flush=True)


def validate(a):
    card = load_card(a.card)
    card = card.replace(**{k: v for k, v in (("dose_mj_cm2", a.dose), ("threshold", a.threshold)) if v is not None})
    clip = CLIPS[a.clip]()
    litho = Litho(card, clip.target.shape)
    i_nom = litho.aerials(clip.target)[0]
    from anamorph.resist.rice_pfail import log_pfail
    lp = {t: log_pfail(litho.resist, i_nom, clip.gauges, t).exp() for t in ("gaussian", "saddlepoint")}
    for i, g in enumerate(clip.gauges):
        mc = mc_pfail(i_nom.numpy(), card, g.points, g.sense, a.trials, seed=i)
        print(json.dumps({"gauge": i, "kind": g.kind, "p_gauss": float(lp["gaussian"][i]),
                          "p_saddle": float(lp["saddlepoint"][i]), "p_mc": mc["p"],
                          "mc_ci": [mc["ci_lo"], mc["ci_hi"]], "verdict": verdict(float(lp["saddlepoint"][i]), mc)}))


def main(argv=None):
    p = argparse.ArgumentParser("anamorph")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("tierb")
    b.add_argument("--card", default="cards/highna_euv.yaml")
    b.add_argument("--clips", default="ls,t2t,via")
    b.add_argument("--iters", type=int, default=100)
    b.add_argument("--lambda", dest="lam", type=float, default=1.0)
    b.add_argument("--w-nils", type=float, default=0.5)
    b.add_argument("--w-mrc", type=float, default=1.0)
    b.add_argument("--trials", type=int, default=0, help="MC trials for the worst gauge (0 = skip)")
    b.add_argument("--out", default="")
    a_ = sub.add_parser("tiera")
    a_.add_argument("--openilt", required=True, help="path to an OpenILT checkout (kernels + ICCAD2013 glp)")
    a_.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10")
    a_.add_argument("--scale", type=int, default=4)
    a_.add_argument("--iters", type=int, default=80)
    a_.add_argument("--lr", type=float, default=0.1)
    a_.add_argument("--protocol", choices=["anamorph", "simpleilt", "curvmulti"], default="anamorph")
    a_.add_argument("--gradient", choices=["exact", "openilt", "openilt-exact"], default="exact")
    v = sub.add_parser("validate")
    v.add_argument("--card", default="cards/highna_euv.yaml")
    v.add_argument("--clip", default="ls")
    v.add_argument("--dose", type=float)
    v.add_argument("--threshold", type=float)
    v.add_argument("--trials", type=int, default=20000)
    a = p.parse_args(argv)
    {"tierb": tierb, "tiera": tiera, "validate": validate}[a.cmd](a)


if __name__ == "__main__":
    main()
