"""COCO compressed RLE encoding in pure NumPy (byte-identical to pycocotools).

The builder uses it so freezing a benchmark needs no compiled dependency.
"""
from __future__ import annotations

import numpy as np


def encode(mask: np.ndarray) -> dict:
    """Binary HxW mask -> {"size": [h, w], "counts": str}."""
    h, w = mask.shape
    flat = np.asarray(mask, dtype=bool).ravel(order="F")
    change = np.flatnonzero(np.diff(flat.astype(np.int8))) + 1
    bounds = np.concatenate(([0], change, [flat.size]))
    runs = np.diff(bounds).tolist()
    if flat.size and flat[0]:
        runs = [0] + runs          # counts always start with a run of zeros
    out = []
    for i, x in enumerate(runs):
        if i > 2:
            x -= runs[i - 2]
        more = True
        while more:
            c = x & 0x1F
            x >>= 5
            more = (x != -1) if (c & 0x10) else (x != 0)
            if more:
                c |= 0x20
            out.append(chr(c + 48))
    return {"size": [h, w], "counts": "".join(out)}


def area(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))
