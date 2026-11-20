// node demo/check.js: the browser forward model must match the engine's latent image at the
// probe pixels exported by build_data.py (same 24 kernels, float32 engine precision).
const { makeLitho } = require("./litho.js");
const d = require("./data.json");
let worst = 0;
for (const c of d.clips) {
  const m = makeLitho(d.card, d.grids[c.n]), mask = Float64Array.from(c.target, ch => +(ch === "1"));
  const t0 = Date.now(), lat = m.latent(mask), ms = Date.now() - t0;
  const err = Math.max(...c.probes.map(([i, v]) => Math.abs(lat[i] - v)));
  worst = Math.max(worst, err);
  console.log(`${c.key} n=${c.n} max|err|=${err.toExponential(2)} ${ms} ms`);
}
if (!(worst < 1e-4)) { console.error("browser model drifted from the engine"); process.exit(1); }
