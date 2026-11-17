# HARNESS.md — shared engine/agent conventions

These rules apply to every result reported from this repo. Code that enforces them is
referenced in brackets.

## Engine/agent boundary

The agent may only:

1. set typed knobs declared in `anamorph/agent/knobs.schema.json` [`agent/boundary.py: load_knobs`];
2. edit code inside `# EVOLVE-BLOCK-START` / `# EVOLVE-BLOCK-END` regions of the files in
   `EVOLVABLE_FILES` (today: the step/binarization/reinit policy in `opt/ilt.py`)
   [`check_evolve_edit`, `check_paths`];
3. pick from declared action sets (the `enum` knobs).

The agent may never edit: imaging kernels (`optics/`), the stochastic model and its
parameters (`resist/`, `cards/`), metric code (`opt/metrics.py`), the references
(`harness/`), or benchmark data (`bench/`). Mount these read-only in the sandbox.

## Sandbox

Each candidate runs in a rootless container with no network, cgroup CPU/GPU/memory
limits, a wall-clock kill, and a content-addressed run directory holding the inputs
hash, code hash, seed and outputs [`harness/run.py: record`].
(The container itself is deployment configuration and is not part of this repo.)

## Determinism

Fixed seeds; `torch.use_deterministic_algorithms(True)` where supported
[`set_determinism`]. CPU runs reproduce bit-for-bit [`tests/test_engine.py`]. GPU FFT
reductions are not bitwise deterministic across devices. Declared tolerance: 1e-5
relative on reported scalar metrics.

## Budgets

Every comparison uses equal budgets on three axes: wall-clock, GPU-hours and LLM tokens,
with tokens counted and their cost reported. Baselines get the same wall-clock for their
own tuning [`agent/search.py: random_search`].

## Ground truth

A result that fails its checker counts as a failure, never as a lower score.

- **Imaging:** full Abbe (every source point and polarization component, no truncation)
  against the truncated SOCS used by the optimizer [`harness/rigorous_ref.py`]. Rigorous
  M3D (RCWA/FDTD) is not attached yet, so no M3D claim is in scope.
- **Stochastics:** an independent NumPy Monte-Carlo photon→acid sampler
  [`harness/mc_stochastic.py`]. A reported P_fail must be within **2×** of the MC estimate
  wherever MC resolves it: at least 10 failures and P ≥ 1e-6. Below that floor the
  result is labeled **model-only**, but it is still rejected if the MC confidence
  interval contradicts it [`verdict`].

## Splits and statistics

Benchmarks are split into evolve, validation and blind sets. Evolved heuristics are
frozen before blind evaluation. Results are reported as the median of 5 seeds with
bootstrap 95% CIs.

## Mandatory ablations

1. No agent, default knobs.
2. Random or BO search instead of the LLM, at equal budget.
3. Without the learned components (M3D correction, neural initializer).
4. Agent without the ground-truth gate, to quantify reward hacking.
