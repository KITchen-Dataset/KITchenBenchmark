"""Submission checks: catch format mistakes before they silently cost points."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import numpy as np


def _coco_ids(gt_json: Path) -> Tuple[dict, set]:
    gt = json.loads(gt_json.read_text())
    images = {im["id"]: (im["height"], im["width"]) for im in gt["images"]}
    return images, {c["id"] for c in gt["categories"]}


def check_coco(gt_json: str | Path, results: List[dict], task: str, max_errors: int = 20) -> List[str]:
    images, cats = _coco_ids(Path(gt_json))
    errs: List[str] = []
    per_image = defaultdict(int)
    for i, r in enumerate(results):
        e = []
        if r.get("image_id") not in images:
            e.append(f"image_id {r.get('image_id')!r} is not in this split")
        if r.get("category_id") not in cats:
            e.append(f"category_id {r.get('category_id')!r} is not a KITchen class id")
        s = r.get("score")
        if not isinstance(s, (int, float)) or not math.isfinite(s):
            e.append("score must be a finite number")
        if task == "det":
            b = r.get("bbox")
            if not (isinstance(b, list) and len(b) == 4 and all(isinstance(v, (int, float)) for v in b)):
                e.append("bbox must be [x, y, width, height] in pixels")
            elif b[2] <= 0 or b[3] <= 0:
                e.append("bbox width and height must be positive")
            elif max(b[0], b[1]) <= 1.0 and max(b[2], b[3]) <= 1.0:
                e.append("bbox looks normalised (0-1); COCO wants pixels")
        else:
            seg = r.get("segmentation")
            if not (isinstance(seg, dict) and "counts" in seg and "size" in seg):
                e.append("segmentation must be COCO RLE {'size': [h, w], 'counts': ...}")
            elif r.get("image_id") in images and list(seg["size"]) != list(images[r["image_id"]]):
                e.append(f"RLE size {seg['size']} != image size {list(images[r['image_id']])}")
        per_image[r.get("image_id")] += 1
        for msg in e:
            errs.append(f"result {i}: {msg}")
        if len(errs) >= max_errors:
            errs.append("... stopping after the first errors")
            return errs
    many = [k for k, n in per_image.items() if n > 300]
    if many:
        errs.append(f"{len(many)} images have more than 300 results; only the 300 best per image count")
    if not results:
        errs.append("the file holds no results")
    return errs


def check_pose(test, ests: List[dict], max_errors: int = 20) -> List[str]:
    """``test`` is a :class:`pose_eval.TestSet`; ``ests`` from ``load_bop_csv``."""
    from .pose_eval import MAX_ESTS_PER_IMAGE
    ims = {(t["scene_id"], t["im_id"]) for t in test.targets24}
    errs: List[str] = []
    times = defaultdict(set)
    per_image = defaultdict(int)
    for i, e in enumerate(ests):
        m = []
        if (e["scene_id"], e["im_id"]) not in ims:
            m.append(f"(scene_id {e['scene_id']}, im_id {e['im_id']}) is not a test image")
        if e["obj_id"] not in test.models_info:
            m.append(f"obj_id {e['obj_id']} has no model")
        R = e["R"]
        if not np.all(np.isfinite(R)) or abs(np.linalg.det(R) - 1.0) > 1e-3 or not np.allclose(R @ R.T, np.eye(3), atol=1e-3):
            m.append("R is not a rotation matrix (row-major 3x3, det +1)")
        t = e["t"]
        if not np.all(np.isfinite(t)):
            m.append("t is not finite")
        elif t[2] <= 0:
            m.append("t has z <= 0 (object behind the camera)")
        elif np.linalg.norm(t) < 20:
            m.append("t looks like metres; BOP wants millimetres")
        if not e["time"] > 0:
            m.append("time must be the wall time in seconds for the image (> 0); "
                     "KITchen pose results must report speed (5 fps rule)")
        times[(e["scene_id"], e["im_id"])].add(e["time"])
        per_image[(e["scene_id"], e["im_id"])] += 1
        errs += [f"row {i + 1}: {x}" for x in m]
        if len(errs) >= max_errors:
            errs.append("... stopping after the first errors")
            return errs
    if any(len(v) > 1 for v in times.values()):
        errs.append("time must be the same for all estimates of one image (BOP rule)")
    many = sum(1 for n in per_image.values() if n > MAX_ESTS_PER_IMAGE)
    if many:
        errs.append(f"{many} images have more than {MAX_ESTS_PER_IMAGE} estimates; only the best count")
    if not ests:
        errs.append("the file holds no estimates")
    return errs
