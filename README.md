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
   on CPU.
5. **On thin-mask optics, anamorphism is nearly invisible.** In wafer coordinates, (4×, 8×)
   only changes the radiometric factor, by less than 1e-2 in intensity. The real anamorphic
   effects live in reticle-space MRC (a 32 nm rule becomes 8 nm in x and 4 nm in y) and in
   M3D shadowing. The engine models MRC; M3D still needs a rigorous reference.

## Milestones (honest status)

| | Scope | Status |
|---|---|---|
| M0 | OpenILT/LithoBench reproduction, deterministic CI | Tier-A pipeline runs on all 10 ICCAD-13 cases with the contest kernels. The **within-5% parity gate is not yet checked** against a published table. Shot counts use a greedy cover, not adabox, so they are not comparable to OpenILT's |
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
