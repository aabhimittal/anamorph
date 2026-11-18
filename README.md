# anamorph

A differentiable inverse-lithography (ILT) engine for High-NA EUV that is aware of
stochastics and anamorphic optics. It optimizes curvilinear masks to minimize **stochastic
failure probability**, not just EPE, under (4×, 8×) anamorphic, polarization-aware imaging
at NA 0.55. Every probability it reports is checked against an independent Monte-Carlo
photon/acid simulator.

> **Status:** M0–M1 core, CPU-verified. See the milestone table. All physical parameters
> in `cards/` are illustrative, not calibrated, so P_fail is **model-relative**, not a
> yield prediction.

## Pipeline

```
mask θ ──► MaskParam (pixel | level set) ──► m ∈ [0,1]
          └─ MRC: soft open/close with reticle-space (anamorphic) boxes
m ──► thin mask (+ boundary-layer M3D operator, zero until fitted to a rigorous ref)
  ──► SOCS (vector, anamorphic radiometry, obscuration, defocus)   ◄─ checked vs full Abbe
  ──► aerial I (nominal / +dose / defocus−dose)
  ──► resist: Gaussian acid blur → latent R → soft threshold
  ──► GRF moments of R + exact compound-Poisson CGF
  ──► per-gauge log P_fail: Rice upcrossing bound, saddlepoint (r*) tail  ◄─ checked vs MC
loss = L2 + PVB + λ·log10(Σ_g P_fail) + MRC + binarization  (− NILS for the baseline)
```

| Module | What it does |
|---|---|
| `optics/pupil.py` | Source sampling (conventional/annular/dipole/quasar); vector unpolarized pupil (6 components), anamorphic radiometric factor, central obscuration, defocus, immersion index |
| `optics/abbe.py`, `optics/socs.py` | Abbe reference. SOCS kernels come from an SVD of the source×pupil matrix, so the TCC is never formed and full rank reproduces Abbe to 1e-15. Also wraps external (ICCAD-13) kernels |
| `optics/anamorphic.py` | Wafer↔reticle geometry |
| `optics/m3d.py` | Thin mask + learnable edge-direction-resolved boundary-layer M3D operator and its fitting loop |
| `resist/compact.py`, `resist/grf_stochastic.py` | Blur/threshold resist; per-pixel mean, gradient, variance and derivative-covariance maps |
| `resist/rice_pfail.py` | Differentiable P_fail per gauge: Rice formula for a non-stationary process crossing a moving level, with Gaussian or saddlepoint tails |
| `mask/` | Pixel and level-set parameterizations; differentiable and hard MRC |
| `opt/ilt.py` | Optimizer, process window, straight-through binarized phase, `EVOLVE-BLOCK` step policy |
| `opt/metrics.py` | L2, PVBand, EPE (ICCAD-13 rule or High-NA rule), shots, ILS, MRC |
| `fullchip/tiling.py` | Halo tiling with a half-tile-shifted healing pass |
| `harness/` | MC stochastic checker + verdict, Abbe-vs-SOCS check, deterministic content-addressed run records |
| `agent/` | Knob schema, EVOLVE-BLOCK diff gate, read-only path gate, random-search baseline |
| `bench/` | EUV clip benchmark v0 (L/S, tip-to-tip, vias with gauges), ICCAD-13 `.glp` + contest-kernel adapter, LithoBench/MaskOpt `.npy/.png` loader |
| `io/gds.py` | GDS/OASIS window rasterization in, rectangle-cover mask out (wafer or reticle coordinates) |

## Quickstart

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch   # or a CUDA build
pip install -e ".[test]" mpmath
pytest -q                                                            # ~5 min on 4 CPU cores

# Tier B: EPE-only vs NILS-proxy vs stochastic-aware ILT, MC-checking the worst gauge
python -m anamorph tierb --clips ls,t2t,via --lambda 1.0 --iters 100 --trials 20000 --out runs

# Analytic (Gaussian and saddlepoint) vs Monte-Carlo P_fail on every gauge of a clip
python -m anamorph validate --clip ls --dose 10 --threshold 0.2 --trials 20000

# Tier A: ICCAD-13 with the contest kernels from an OpenILT checkout (MIT)
git clone --depth 1 https://github.com/OpenOPC/OpenILT
python -m anamorph tiera --openilt OpenILT --cases 1,2,3
# M0 parity: OpenILT's SimpleILT protocol + its approximate gradient, vs its README table
python -m anamorph tiera --openilt OpenILT --protocol simpleilt --gradient openilt
```

## Findings so far

These come from runs in this repo, on CPU only, with illustrative parameters. Reproduce
them with the commands above.

1. **The Gaussian GRF tail is not good enough.** On a 16 nm-trench L/S at low dose, the
   Gaussian approximation was off by 4–13× against Monte Carlo at P ≈ 1e-3 to 1e-4. Its
   errors have opposite signs for the two failure types:
   - **Breaks** (lower tail of the acid count): it over-predicts, which is conservative.
   - **Bridges** (upper tail): it under-predicts by about 10×, which is anti-conservative.

   This follows from Poisson skew. The saddlepoint tail built on the exact
   compound-Poisson CGF matched MC to within 1.5× in every resolvable case.
2. **The aggregate is `log Σ P`, not `Σ log P`.** A sum of per-gauge logs rewards pushing
   already-negligible gauges lower. The log of the union bound concentrates the
   gradient on the worst gauge.
3. **Grey-level cheating.** A pixel ILT lowers its loss with partially transmitting pixels
   that binarization then erases. On L/S, the soft mask reached log10 P ≈ −9.4, but its
   binarized version only reached −7.9. A straight-through binarized phase, with a
   warm-restarted step size, closes the gap.
4. **Tier-B v0** (`tierb --iters 100 --lambda 1.0`, single seed, CPU). These are fixed-λ
   points, not equal-EPE Pareto fronts.

   | clip | config | L2 | PVB | EPE (3 nm) | MRC viol. | worst log10 P_fail |
   |---|---|---|---|---|---|---|
   | L/S p24 | EPE-only | 0 | 768 | 0 | 0 | −7.66 (break) |
   | L/S p24 | stochastic | 462 | 216 | 0 | 179 | −9.21 (bridge) |
   | T2T p24 g16 | EPE-only | 128 | 352 | 0 | 96 | −5.27 (break) |
   | T2T p24 g16 | stochastic | 1040 | 216 | 8 | 0 | −14.77 (break) |
   | via p32 | EPE-only | 768 | 768 | 0 | 0 | −11.62 (bridge) |
   | via p32 | stochastic | 752 | 748 | 0 | 304 | −12.88 (bridge) |

   At the card dose, every P_fail lands in the MC **model-only** regime (no failures in
   20k trials). That agrees with the model but does not confirm it. The quantitative
   model-vs-MC agreement comes from the low-dose cases in `tests/test_stochastic.py`. The
   NILS baseline (`w_nils=0.5`) barely moved away from EPE-only, so it still needs its
   equal-budget tuning before any comparison is fair.

   Tier-A on all 10 ICCAD-13 cases, with the contest kernels, 4 nm optimization grid,
   80 iterations and evaluation at 1 nm: mean L2 31.8k, mean PVB 44.3k, 1–3 min per case
   on CPU. These used the pixel-center target and are not comparable to OpenILT's table
   (see finding 6).
5. **On thin-mask optics, anamorphism is nearly invisible.** In wafer coordinates, (4×, 8×)
   only changes the radiometric factor, by less than 1e-2 in intensity. The real anamorphic
   effects live in reticle-space MRC (a 32 nm rule becomes 8 nm in x and 4 nm in y) and in
   M3D shadowing. The engine models MRC; M3D still needs a rigorous reference.
6. **M0 parity: anamorph reproduces OpenILT exactly, but OpenILT's README table does not
   reproduce.** Run the SimpleILT protocol on all 10 ICCAD-13 cases at 2048² (20 SGD steps,
   lr 0.5, center 1024² window, OpenILT's approximate combo-kernel gradient). OpenILT's own
   code at commit `dabb97c` and anamorph give the same final mask (≤ 1 pixel different),
   L2/PVB within 3e-5 relative, and identical EPE. Both land at **+5.9% mean L2 and +7.5% mean
   PVB versus the README table**, up to +20% on a single case:

   | case | README L2 / PVB / EPE | OpenILT @dabb97c | anamorph | L2 vs README | PVB vs README |
   |---|---|---|---|---|---|
   | 1 | 43408 / 52281 / 3 | 48898 / 55022 / 8 | 48898 / 55022 / 8 | +12.6% | +5.2% |
   | 2 | 35326 / 41865 / 2 | 37327 / 46019 / 4 | 37327 / 46019 / 4 | +5.7% | +9.9% |
   | 3 | 75428 / 78805 / 43 | 81325 / 86683 / 47 | 81325 / 86684 / 47 | +7.8% | +10.0% |
   | 4 | 13649 / 22112 / 2 | 16409 / 26358 / 2 | 16409 / 26358 / 2 | +20.2% | +19.2% |
   | 5 | 37330 / 54977 / 2 | 37810 / 57472 / 0 | 37810 / 57472 / 0 | +1.3% | +4.5% |
   | 6 | 35711 / 51036 / 0 | 36710 / 52566 / 0 | 36710 / 52566 / 0 | +2.8% | +3.0% |
   | 7 | 29566 / 44576 / 0 | 29520 / 47599 / 2 | 29520 / 47599 / 2 | −0.2% | +6.8% |
   | 8 | 14327 / 20727 / 0 | 14291 / 24268 / 1 | 14291 / 24268 / 1 | −0.3% | +17.1% |
   | 9 | 45347 / 64063 / 0 | 47368 / 64929 / 2 | 47367 / 64927 / 2 | +4.5% | +1.3% |
   | 10 | 8404 / 16685 / 0 | 8951 / 19874 / 0 | 8951 / 19874 / 0 | +6.5% | +19.1% |
   | mean | 33850 / 44713 / 5.2 | 35861 / 48079 / 6.6 | 35861 / 48079 / 6.6 | +5.9% | +7.5% |

   Supporting checks, run in isolation:
   - our target raster equals `cv2.fillPoly` on all 10 clips (0 pixels different);
   - on the same mask our forward model and metrics equal OpenILT's simulator, pixel for pixel;
   - our compat gradient equals theirs to 6e-7 relative.

   OpenILT's code also reproduces the masks it ships in `tmp/MOSAIC_test*.png`, so the
   README table most likely comes from an earlier code version or a GPU run. That cannot
   be checked from the repository. CPU-only; OpenILT's numbers were measured on an RTX 3090.

## Milestones (honest status)

| | Scope | Status |
|---|---|---|
| M0 | OpenILT/LithoBench reproduction, deterministic CI | **OpenILT: done against its code, not against its README table** (finding 6). We match OpenILT @dabb97c within 3e-5 on all 10 cases, but the README table is +5.9% / +7.5% off even for OpenILT's own code, so a 5% gate against that table fails for both. LithoBench is not yet attempted. Shot counts use a greedy cover, not adabox, so they are not comparable |
| M1 | Anamorphic Abbe/SOCS + GRF P_fail + MC validation | **Done** on CPU, including vector, obscuration, the saddlepoint tail and the MC gate |
| M2 | Learned M3D vs rigorous reference; curvilinear MRC; EUV clips v0 | Operator, fitting loop, MRC and clips v0 are in. **No rigorous RCWA/FDTD reference yet**, so the M3D kernels stay at zero |
| M3 | Multi-GPU full chip, agentic layer, paper | Tiling and healing run on a single device. The agent boundary and random-search baseline are in. **No NCCL and no LLM driver** |

Not implemented: B-spline masks, a neural initializer, multigon/spline export, automatic
gauge placement on GDS clips, and a GPU benchmark (no GPU was available during development).

## Caveats

- No calibrated EUV resist or stochastic data is public. The resist model is a compact
  blur/threshold with compound-Poisson shot noise, so it says nothing about absolute yield.
- The failure event is a centerline excursion. That is a necessary condition for a
  topological break or bridge, so the reported P_fail is an upper bound on those events,
  and the MC checker counts the same event.
- Cards contain generic, public-literature-style values and no vendor data. Check any
  post-employment IP obligations before adding calibrated models.

License: Apache-2.0. Shared engine/agent conventions: [HARNESS.md](HARNESS.md).
