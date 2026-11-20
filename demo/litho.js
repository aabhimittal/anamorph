// anamorph forward model in the browser: thin mask -> truncated SOCS aerial -> Gaussian latent.
// Mirrors optics/socs.py (fft2 unnormalized, ifft2 with 1/N^2) and resist/compact.py (periodic,
// normalized Gaussian). The pupil passes only a small disk of frequencies, so a separable DFT
// over those bins is exact and fast at any grid size (96 and 128 are not powers of two).
(function (root) {
  function makeLitho(card, g) {
    const n = g.n, K = g.re.length, B = g.support.length;
    const fy = g.support.map(i => Math.floor(i / n)), fx = g.support.map(i => i % n);
    const fxs = [...new Set(fx)], col = fx.map(v => fxs.indexOf(v)), F = fxs.length;
    const cs = new Float64Array(n), sn = new Float64Array(n);
    for (let j = 0; j < n; j++) { cs[j] = Math.cos(2 * Math.PI * j / n); sn[j] = Math.sin(2 * Math.PI * j / n); }
    const ar = card.absorber_amplitude * Math.cos(card.absorber_phase_deg * Math.PI / 180);
    const ai = card.absorber_amplitude * Math.sin(card.absorber_phase_deg * Math.PI / 180);
    // periodic Gaussian blur, separable: 1-D weights over wrapped signed distance
    const s = card.diffusion_nm, gk = new Float64Array(n);
    let gs = 0;
    for (let j = 0; j < n; j++) { const d = j <= n / 2 ? j : j - n; gk[j] = s > 0 ? Math.exp(-d * d / (2 * s * s)) : +(j === 0); gs += gk[j]; }
    for (let j = 0; j < n; j++) gk[j] /= gs;

    function aerial(mask) {
      // T at the support bins: rows first (only the fx columns we need), then fy per bin
      const Ar = new Float64Array(n * F), Ai = new Float64Array(n * F);
      for (let y = 0; y < n; y++) for (let c = 0; c < F; c++) {
        let sr = 0, si = 0;
        for (let x = 0; x < n; x++) {
          const m = mask[y * n + x], tr = ar + (1 - ar) * m, ti = ai - ai * m, k = (fxs[c] * x) % n;
          sr += tr * cs[k] + ti * sn[k]; si += ti * cs[k] - tr * sn[k];
        }
        Ar[y * F + c] = sr; Ai[y * F + c] = si;
      }
      const Tr = new Float64Array(B), Ti = new Float64Array(B);
      for (let b = 0; b < B; b++) {
        let sr = 0, si = 0;
        for (let y = 0; y < n; y++) {
          const k = (fy[b] * y) % n, xr = Ar[y * F + col[b]], xi = Ai[y * F + col[b]];
          sr += xr * cs[k] + xi * sn[k]; si += xi * cs[k] - xr * sn[k];
        }
        Tr[b] = sr; Ti[b] = si;
      }
      const I = new Float64Array(n * n), Gr = new Float64Array(F * n), Gi = new Float64Array(F * n);
      const scale = 1 / (n * n);
      for (let kk = 0; kk < K; kk++) {
        const hr = g.re[kk], hi = g.im[kk];
        Gr.fill(0); Gi.fill(0);
        for (let b = 0; b < B; b++) {
          const pr = Tr[b] * hr[b] - Ti[b] * hi[b], pi = Tr[b] * hi[b] + Ti[b] * hr[b], o = col[b] * n;
          for (let y = 0; y < n; y++) {
            const k = (fy[b] * y) % n;
            Gr[o + y] += pr * cs[k] - pi * sn[k]; Gi[o + y] += pr * sn[k] + pi * cs[k];
          }
        }
        for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
          let er = 0, ei = 0;
          for (let c = 0; c < F; c++) {
            const k = (fxs[c] * x) % n, gr = Gr[c * n + y], gi = Gi[c * n + y];
            er += gr * cs[k] - gi * sn[k]; ei += gr * sn[k] + gi * cs[k];
          }
          I[y * n + x] += (er * er + ei * ei) * scale * scale;
        }
      }
      for (let i = 0; i < n * n; i++) I[i] /= g.norm;
      return I;
    }

    function blur(I) {
      const tmp = new Float64Array(n * n), out = new Float64Array(n * n);
      for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
        let acc = 0;
        for (let j = 0; j < n; j++) acc += gk[j] * I[y * n + ((x - j + n) % n)];
        tmp[y * n + x] = acc;
      }
      for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
        let acc = 0;
        for (let j = 0; j < n; j++) acc += gk[j] * tmp[((y - j + n) % n) * n + x];
        out[y * n + x] = acc;
      }
      return out;
    }

    return { n, aerial, latent: mask => blur(aerial(mask)) };
  }
  if (typeof module !== "undefined") module.exports = { makeLitho };
  else root.makeLitho = makeLitho;
})(this);
