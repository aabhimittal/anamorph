"""The engine/agent boundary (HARNESS.md), enforced in code.

1. knobs  : validated against knobs.schema.json; anything else is rejected.
2. code   : a candidate source file may differ from the original ONLY inside
            `# EVOLVE-BLOCK-START <name>` ... `# EVOLVE-BLOCK-END` regions.
3. paths  : files outside the evolvable set are read-only for the agent.
"""
from __future__ import annotations

import dataclasses as dc
import json
import re
from pathlib import Path

import jsonschema

from anamorph.opt.ilt import Knobs

SCHEMA = json.loads((Path(__file__).with_name("knobs.schema.json")).read_text())
EVOLVABLE_FILES = {"anamorph/opt/ilt.py"}  # only EVOLVE-BLOCK regions within these
_BLOCK = re.compile(r"(# EVOLVE-BLOCK-START[^\n]*\n)(.*?)(# EVOLVE-BLOCK-END)", re.S)


def load_knobs(d: dict) -> Knobs:
    jsonschema.validate(d, SCHEMA)
    return dc.replace(Knobs(), **d)


def check_evolve_edit(original: str, candidate: str) -> list[str]:
    """Violations of rule 2 (empty list = admissible edit)."""
    so, sc = _BLOCK.split(original), _BLOCK.split(candidate)
    if len(so) != len(sc):
        return ["EVOLVE-BLOCK markers added, removed or altered"]
    errs = []
    # split() yields [outside, start, body, end, outside, start, body, end, ..., outside]
    for i, (a, b) in enumerate(zip(so, sc)):
        if i % 4 == 2:
            continue  # block body: free to change
        if a != b:
            errs.append(f"edit outside EVOLVE-BLOCK (segment {i})" if i % 4 == 0 else f"marker changed (segment {i})")
    return errs


def check_paths(changed: list[str]) -> list[str]:
    """Violations of rule 3 for a list of repo-relative changed paths."""
    return [f"read-only path: {p}" for p in changed if p not in EVOLVABLE_FILES]
