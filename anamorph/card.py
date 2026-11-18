"""Process card: every physical parameter of a run, loaded from YAML.

The card is the single source of truth for the forward model. The agent may NOT
edit it (it only sets knobs in agent/knobs.schema.json); see HARNESS.md.
"""
from __future__ import annotations

import dataclasses as dc
import hashlib
import json
from pathlib import Path

import yaml

H_C_EV_NM = 1239.841984  # h*c in eV*nm
EV_J = 1.602176634e-19


@dc.dataclass(frozen=True)
class Source:
    shape: str = "annular"          # conventional | annular | dipole | quasar
    sigma_in: float = 0.0
    sigma_out: float = 0.9
    axis: str = "x"                 # dipole axis
    opening_deg: float = 90.0       # dipole/quasar pole opening angle
    step: float = 0.05              # source sampling step (pupil-fill units)


@dc.dataclass(frozen=True)
class Resist:
    threshold: float = 0.3          # on blurred, clear-field-normalized acid image
    steepness: float = 30.0         # sigmoid slope for the differentiable contour
    diffusion_nm: float = 3.0       # Gaussian acid blur sigma; 0 = threshold on aerial
    dose_mj_cm2: float = 60.0
    absorption: float = 0.2         # fraction of incident photons absorbed in the film
    quantum_yield: float = 2.0      # mean acids per absorbed photon (Poisson)


@dc.dataclass(frozen=True)
class Card:
    name: str = "unnamed"
    wavelength_nm: float = 13.5
    na: float = 0.55
    immersion_index: float = 1.0
    obscuration: float = 0.0        # central pupil obscuration, fraction of NA
    magnification: tuple = (4.0, 8.0)   # (Mx, My) reticle:wafer
    vector: bool = True
    absorber_amplitude: float = 0.0  # |r_abs / r_ml| of the dark mask region
    absorber_phase_deg: float = 0.0
    source: Source = Source()
    resist: Resist = Resist()
    pw_dose: float = 0.03            # +/- relative dose for the process window
    pw_focus_nm: float = 20.0        # +/- defocus for the process window
    mrc_width_reticle_nm: float = 0.0   # min width at the reticle (0 = off)
    mrc_space_reticle_nm: float = 0.0
    pixel_nm: float = 1.0

    @property
    def photon_energy_j(self) -> float:
        return H_C_EV_NM / self.wavelength_nm * EV_J

    @property
    def photons_per_nm2(self) -> float:
        """Incident photons per nm^2 at the card dose (1 mJ/cm^2 = 1e-17 J/nm^2)."""
        return self.resist.dose_mj_cm2 * 1e-17 / self.photon_energy_j

    def replace(self, **kw) -> "Card":
        res = {k: kw.pop(k) for k in list(kw) if k in {f.name for f in dc.fields(Resist)}}
        src = {k[4:]: kw.pop(k) for k in list(kw) if k.startswith("src_")}
        out = dc.replace(self, **kw)
        if res:
            out = dc.replace(out, resist=dc.replace(out.resist, **res))
        if src:
            out = dc.replace(out, source=dc.replace(out.source, **src))
        return out

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(dc.asdict(self), sort_keys=True).encode()).hexdigest()


def load_card(path: str | Path) -> Card:
    d = yaml.safe_load(Path(path).read_text())
    d["source"] = Source(**d.get("source", {}))
    d["resist"] = Resist(**d.get("resist", {}))
    if "magnification" in d:
        d["magnification"] = tuple(float(m) for m in d["magnification"])
    return Card(**d)
