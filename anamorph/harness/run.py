"""Determinism and content-addressed run records (HARNESS.md).

A run directory is named by sha256(inputs hash, code hash, seed); it stores a
manifest (hashes, seed, versions) and the outputs. Re-running the same inputs on the
same code must reproduce the outputs bit-for-bit on CPU (see tests/test_harness.py).
"""
from __future__ import annotations

import hashlib
import json
import platform
import random
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def set_determinism(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def code_hash() -> str:
    h = hashlib.sha256()
    for p in sorted(ROOT.rglob("*.py")) + sorted(ROOT.rglob("*.json")):
        h.update(p.relative_to(ROOT).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def inputs_hash(**inputs) -> str:
    h = hashlib.sha256()
    for k in sorted(inputs):
        v = inputs[k]
        h.update(k.encode())
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy()
        h.update(np.ascontiguousarray(v).tobytes() if isinstance(v, np.ndarray)
                 else json.dumps(v, sort_keys=True, default=str).encode())
    return h.hexdigest()


def record(root, seed: int, outputs: dict, **inputs) -> Path:
    """Write outputs under root/<run-id>/ and return the directory."""
    ih, ch = inputs_hash(**inputs), code_hash()
    rid = hashlib.sha256(f"{ih}{ch}{seed}".encode()).hexdigest()[:16]
    d = Path(root) / rid
    d.mkdir(parents=True, exist_ok=True)
    arrays = {k: (v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v)
              for k, v in outputs.items() if isinstance(v, (torch.Tensor, np.ndarray))}
    scalars = {k: v for k, v in outputs.items() if k not in arrays}
    np.savez_compressed(d / "outputs.npz", **arrays)
    (d / "manifest.json").write_text(json.dumps({
        "run_id": rid, "inputs_sha256": ih, "code_sha256": ch, "seed": seed,
        "torch": torch.__version__, "python": platform.python_version(), "results": scalars,
    }, indent=1, default=str))
    return d
