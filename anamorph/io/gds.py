"""GDSII/OASIS in (rasterize a window) and out (Manhattan pixel-polygon mask) via gdstk.

shortcut: output is the pixel-exact rectangle cover of the binary mask, not a
multigon/spline curvilinear format; add that when a mask-writer flow consumes it.
"""
from __future__ import annotations

import gdstk
import numpy as np

from anamorph.opt.metrics import rectangles


def read_raster(path: str, layer: int, datatype: int, bbox_nm, pixel_nm: float, cell: str | None = None) -> np.ndarray:
    """Rasterize polygons on (layer, datatype) inside bbox (x0, y0, x1, y1) nm; row = y."""
    lib = gdstk.read_oas(path) if str(path).endswith((".oas", ".oasis")) else gdstk.read_gds(path)
    top = next(c for c in lib.cells if c.name == cell) if cell else lib.top_level()[0]
    to_nm = lib.unit / 1e-9
    polys = [p.points * to_nm for p in top.get_polygons(layer=layer, datatype=datatype)]
    x0, y0, x1, y1 = bbox_nm
    xs = x0 + (np.arange(int(round((x1 - x0) / pixel_nm))) + 0.5) * pixel_nm
    ys = y0 + (np.arange(int(round((y1 - y0) / pixel_nm))) + 0.5) * pixel_nm
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel()], 1)
    if not polys:
        return np.zeros(gx.shape)
    return np.asarray(gdstk.inside(pts, polys), dtype=float).reshape(gx.shape)


def write_mask(mask: np.ndarray, path: str, pixel_nm: float, origin_nm=(0.0, 0.0), layer: int = 1,
               magnification=(1.0, 1.0), cell: str = "ANAMORPH_MASK"):
    """Write the binary mask as rectangles. magnification=(4, 8) writes reticle coordinates."""
    lib = gdstk.Library(unit=1e-9, precision=1e-12)  # 1 user unit = 1 nm
    c = lib.new_cell(cell)
    mx, my = magnification
    ox, oy = origin_nm
    for y0, x0, y1, x1 in rectangles(mask):
        c.add(gdstk.rectangle(((ox + x0 * pixel_nm) * mx, (oy + y0 * pixel_nm) * my),
                              ((ox + x1 * pixel_nm) * mx, (oy + y1 * pixel_nm) * my), layer=layer))
    lib.write_gds(path)
