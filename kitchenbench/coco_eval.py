"""2D detection and instance segmentation scores (COCO protocol, pycocotools).

Unverified annotations are stored as ``iscrowd=1`` regions, so a detection on
them is neither rewarded nor penalised (the standard COCO ignore mechanism).
KITchen frames hold up to ~80 objects, so the headline AP uses up to 300
detections per image (as LVIS does) instead of COCO's 100.
"""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

MAX_DETS = (10, 100, 300)
AREA_NAMES = ("all", "small", "medium", "large")


def load_results(path: str | Path) -> List[dict]:
    with open(path) as f:
        res = json.load(f)
    if isinstance(res, dict) and "annotations" in res:
        res = res["annotations"]
    if not isinstance(res, list):
        raise ValueError(f"{path}: expected a JSON list of COCO results")
    return res


def evaluate(gt_json: str | Path, results: List[dict], iou_type: str,
             image_ids: List[int] | None = None) -> Dict[str, object]:
    """Run COCOeval and return headline numbers plus per-class AP."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(gt_json))
        dt = gt.loadRes(results) if results else COCO()
        ev = COCOeval(gt, dt, iou_type)
        ev.params.maxDets = list(MAX_DETS)
        if image_ids is not None:
            ev.params.imgIds = sorted(image_ids)
        ev.evaluate()
        ev.accumulate()

    prec = ev.eval["precision"]          # [T, R, K, A, M]
    rec = ev.eval["recall"]              # [T, K, A, M]
    iou = ev.params.iouThrs
    m_last = len(MAX_DETS) - 1

    def ap(t=None, a=0, k=None):
        p = prec[:, :, :, a, m_last]
        if t is not None:
            p = p[np.where(np.isclose(iou, t))[0]]
        if k is not None:
            p = p[:, :, [k]]
        p = p[p > -1]
        return float(p.mean()) if p.size else float("nan")

    def ar(m, a=0):
        r = rec[:, :, a, m]
        r = r[r > -1]
        return float(r.mean()) if r.size else float("nan")

    out: Dict[str, object] = {
        "AP": ap(), "AP50": ap(0.5), "AP75": ap(0.75),
        "APs": ap(a=1), "APm": ap(a=2), "APl": ap(a=3),
        **{f"AR{m}": ar(i) for i, m in enumerate(MAX_DETS)},
    }
    cats = gt.loadCats(ev.params.catIds)
    out["per_class_AP"] = {c["name"]: ap(k=i) for i, c in enumerate(cats)}
    img_ids = set(ev.params.imgIds)
    out["n_images"] = len(img_ids)
    out["n_gt"] = int(sum(1 for a in gt.dataset["annotations"]
                          if not a.get("iscrowd") and a["image_id"] in img_ids))
    return out
