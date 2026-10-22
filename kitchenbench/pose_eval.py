"""6D pose evaluation on the KITchen test set (BOP 2019 + BOP 2024 protocols).

Primary score (BOP 2024 "6D detection")::

    AP_6D = mean(AP_MSSD, AP_MSPD)

where each AP is averaged over the 10 BOP thresholds and over objects.
Secondary score (BOP 2019 "6D localization", target counts given)::

    AR_6D = mean(AR_MSSD, AR_MSPD)

AR_VSD needs a renderer and is not part of the KITchen score; because the test
set is in BOP format, the official ``bop_toolkit`` can add it if wanted.

KITchen addition: ``test_ignore.json`` lists (image, object) pairs whose pose
is not verified. They are absent from ``scene_gt.json`` and estimates for them
are dropped, so a method is never punished for predicting an object whose
reference pose we do not trust.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from . import pose_metrics as pm

MAX_ESTS_PER_IMAGE = 200     # KITchen frames are crowded (BOP uses 200 for XYZ-IBD)
BOP_REF_WIDTH = 640.0        # MSPD is normalised to a 640 px wide image


@dataclass
class Model:
    pts: np.ndarray        # (N, 3) mm
    diameter: float        # mm
    sym_pts: np.ndarray    # (S, N, 3) points under every symmetry transformation


def read_ply_points(path: Path) -> np.ndarray:
    """Read vertex x, y, z from an ASCII or binary little-endian PLY."""
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline().decode("ascii", "replace").strip()
            header.append(line)
            if line == "end_header":
                break
        fmt = next(l.split()[1] for l in header if l.startswith("format"))
        n = 0
        props: List[Tuple[str, str]] = []
        in_vertex = False
        for l in header:
            p = l.split()
            if p[:2] == ["element", "vertex"]:
                n, in_vertex = int(p[2]), True
            elif p and p[0] == "element":
                in_vertex = False
            elif in_vertex and p and p[0] == "property":
                props.append((p[-1], p[1]))
        names = [p[0] for p in props]
        if fmt == "ascii":
            rows = [f.readline().split() for _ in range(n)]
            arr = np.asarray(rows, dtype=np.float64)
            return arr[:, [names.index(c) for c in "xyz"]]
        tmap = {"float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
                "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1", "ushort": "<u2",
                "uint16": "<u2", "short": "<i2", "int16": "<i2", "int": "<i4", "int32": "<i4",
                "uint": "<u4", "uint32": "<u4"}
        if fmt != "binary_little_endian":
            raise ValueError(f"{path}: unsupported PLY format {fmt}")
        dt = np.dtype([(nm, tmap[t]) for nm, t in props])
        v = np.frombuffer(f.read(n * dt.itemsize), dtype=dt, count=n)
        return np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)


class TestSet:
    """The BOP-format part of a built benchmark (``<bench>/bop``)."""

    def __init__(self, bop_dir: str | Path, split: str = "test"):
        self.root = Path(bop_dir)
        self.split = split
        self.models_info = {int(k): v for k, v in _load(self.root / "models_info.json").items()}
        self.targets19 = _load(self.root / f"{split}_targets_bop19.json")
        self.targets24 = _load(self.root / f"{split}_targets_bop24.json")
        ign = self.root / f"{split}_ignore.json"
        self.ignore = {(t["scene_id"], t["im_id"], t["obj_id"]) for t in (_load(ign) if ign.exists() else [])}
        self.gt: Dict[int, Dict[int, list]] = {}
        self.gt_info: Dict[int, Dict[int, list]] = {}
        self.K: Dict[Tuple[int, int], np.ndarray] = {}
        self.width: Dict[int, float] = {}
        for sid in sorted({t["scene_id"] for t in self.targets24}):
            d = self.root / split / f"{sid:06d}"
            self.gt[sid] = {int(k): v for k, v in _load(d / "scene_gt.json").items()}
            self.gt_info[sid] = {int(k): v for k, v in _load(d / "scene_gt_info.json").items()}
            for k, v in _load(d / "scene_camera.json").items():
                self.K[(sid, int(k))] = np.asarray(v["cam_K"], dtype=np.float64).reshape(3, 3)
        cam = self.root / "camera.json"
        self.im_width = float(_load(cam)["width"]) if cam.exists() else 1920.0
        self._models: Dict[int, Model] = {}

    def model(self, obj_id: int) -> Model:
        if obj_id not in self._models:
            pts = read_ply_points(self.root / "models_eval" / f"obj_{obj_id:06d}.ply")
            info = self.models_info[obj_id]
            syms = pm.symmetry_transformations(info, pm.MAX_SYM_DISC_STEP)
            Rs = np.stack([s["R"] for s in syms])                  # (S,3,3)
            ts = np.stack([s["t"].reshape(3) for s in syms])        # (S,3)
            sym_pts = np.einsum("sij,nj->sni", Rs, pts) + ts[:, None, :]
            self._models[obj_id] = Model(pts, float(info["diameter"]), sym_pts)
        return self._models[obj_id]

    @property
    def obj_ids(self) -> List[int]:
        return sorted(self.models_info)


def _load(p: Path):
    with open(p) as f:
        return json.load(f)


# ── Predictions ────────────────────────────────────────────────────────────────

def load_bop_csv(path: str | Path) -> List[dict]:
    """Read a BOP19 results CSV: ``scene_id,im_id,obj_id,score,R,t,time``."""
    out = []
    with open(path, newline="") as f:
        for i, row in enumerate(csv.reader(f)):
            if not row or (i == 0 and row[0].strip() == "scene_id"):
                continue
            if len(row) != 7:
                raise ValueError(f"{path}:{i + 1}: expected 7 comma-separated fields, got {len(row)}")
            R = np.array(row[4].split(), dtype=np.float64)
            t = np.array(row[5].split(), dtype=np.float64)
            if R.size != 9 or t.size != 3:
                raise ValueError(f"{path}:{i + 1}: R needs 9 numbers and t needs 3")
            out.append({"scene_id": int(row[0]), "im_id": int(row[1]), "obj_id": int(row[2]),
                        "score": float(row[3]), "R": R.reshape(3, 3), "t": t.reshape(3),
                        "time": float(row[6])})
    return out


def _top_per_image(ests: List[dict], k: int) -> List[dict]:
    by_im: Dict[tuple, list] = defaultdict(list)
    for e in ests:
        by_im[(e["scene_id"], e["im_id"])].append(e)
    out = []
    for lst in by_im.values():
        out.extend(sorted(lst, key=lambda e: e["score"], reverse=True)[:k])
    return out


# ── Errors ─────────────────────────────────────────────────────────────────────

def _errors(est: dict, gts: List[Tuple[int, dict]], model: Model, K: np.ndarray, width: float) -> Dict[str, Dict[int, float]]:
    p_est = model.pts @ est["R"].T + est["t"]
    x = p_est @ K.T
    q_est = x[:, :2] / x[:, 2:3]
    mssd, mspd = {}, {}
    for gt_id, g in gts:
        R = np.asarray(g["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
        t = np.asarray(g["cam_t_m2c"], dtype=np.float64).reshape(3)
        p_gt = model.sym_pts @ R.T + t                                 # (S,N,3)
        mssd[gt_id] = float(np.linalg.norm(p_gt - p_est, axis=2).max(axis=1).min()) / model.diameter
        y = p_gt @ K.T
        q_gt = y[..., :2] / y[..., 2:3]
        mspd[gt_id] = float(np.linalg.norm(q_gt - q_est, axis=2).max(axis=1).min()) * BOP_REF_WIDTH / width
    return {"mssd": mssd, "mspd": mspd}


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(test: TestSet, ests: List[dict]) -> Dict[str, object]:
    """Score pose estimates. Returns headline metrics and per-object breakdowns."""
    target_ims = {(t["scene_id"], t["im_id"]) for t in test.targets24}
    ests = [e for e in ests if (e["scene_id"], e["im_id"]) in target_ims
            and (e["scene_id"], e["im_id"], e["obj_id"]) not in test.ignore
            and e["obj_id"] in test.models_info]
    for i, e in enumerate(ests):
        e["est_id"] = i

    # Errors of every estimate w.r.t. every GT of the same object in its image.
    by_im_obj: Dict[tuple, list] = defaultdict(list)
    for e in _top_per_image(ests, MAX_ESTS_PER_IMAGE):
        by_im_obj[(e["scene_id"], e["im_id"], e["obj_id"])].append(e)
    errs: Dict[int, dict] = {}
    for (sid, iid, oid), lst in by_im_obj.items():
        gts = [(j, g) for j, g in enumerate(test.gt[sid].get(iid, [])) if g["obj_id"] == oid]
        model = test.model(oid)
        for e in lst:
            errs[e["est_id"]] = _errors(e, gts, model, test.K[(sid, iid)], test.im_width) if gts else {"mssd": {}, "mspd": {}}

    ths = {"mssd": pm.MSSD_THS, "mspd": pm.MSPD_THS}
    out: Dict[str, object] = {}
    det = {m: _detection(test, by_im_obj, errs, m, ths[m]) for m in ("mssd", "mspd")}
    loc = {m: _localization(test, ests, errs, m, ths[m]) for m in ("mssd", "mspd")}
    out["AP_MSSD"], out["AP_MSPD"] = det["mssd"]["mAP"], det["mspd"]["mAP"]
    out["AP_6D"] = (out["AP_MSSD"] + out["AP_MSPD"]) / 2.0
    out["AR_MSSD"], out["AR_MSPD"] = loc["mssd"]["AR"], loc["mspd"]["AR"]
    out["AR_6D"] = (out["AR_MSSD"] + out["AR_MSPD"]) / 2.0
    out["per_object_AP_6D"] = {
        str(o): (det["mssd"]["per_obj"][o] + det["mspd"]["per_obj"][o]) / 2.0
        for o in det["mssd"]["per_obj"]}
    out["per_object_AR_6D"] = {
        str(o): (loc["mssd"]["per_obj"][o] + loc["mspd"]["per_obj"][o]) / 2.0
        for o in loc["mssd"]["per_obj"]}
    out["n_images"] = len(target_ims)
    out["n_targets"] = sum(t["inst_count"] for t in test.targets19)
    out["n_estimates"] = len(ests)
    times = {}
    for e in ests:
        times.setdefault((e["scene_id"], e["im_id"]), set()).add(e["time"])
    if times and all(len(v) == 1 for v in times.values()) and min(next(iter(v)) for v in times.values()) >= 0:
        out["time_per_image_s"] = float(np.mean([next(iter(v)) for v in times.values()]))
    else:
        out["time_per_image_s"] = None
    return out


def _detection(test: TestSet, by_im_obj, errs, metric: str, ths: List[float]) -> dict:
    """BOP 2024 6D detection AP for one error type."""
    ap_per_obj: Dict[int, List[float]] = defaultdict(list)
    n_gt_obj: Dict[int, int] = defaultdict(int)
    est_objs: Dict[tuple, set] = defaultdict(set)
    for (sid, iid, oid) in by_im_obj:
        est_objs[(sid, iid)].add(oid)
    for th in ths:
        scores: Dict[int, list] = defaultdict(list)
        matched: Dict[int, list] = defaultdict(list)
        n_gt: Dict[int, int] = defaultdict(int)
        for (sid, iid) in {(t["scene_id"], t["im_id"]) for t in test.targets24}:
            gts = test.gt[sid].get(iid, [])
            info = test.gt_info[sid].get(iid, [])
            low = {j for j, g in enumerate(gts) if info[j]["visib_fract"] < pm.VISIB_GT_MIN}
            for j, g in enumerate(gts):
                if j not in low:
                    n_gt[g["obj_id"]] += 1
            for oid in {g["obj_id"] for g in gts} | est_objs[(sid, iid)]:
                lst = by_im_obj.get((sid, iid, oid), [])
                es = [{"est_id": e["est_id"], "score": e["score"], "errors": errs[e["est_id"]][metric]} for e in lst]
                ms = {m["est_id"]: m["gt_id"] for m in pm.match_poses(es, th)}
                for e in lst:
                    g = ms.get(e["est_id"], -1)
                    if g in low:
                        continue            # matched an almost invisible GT: ignored
                    scores[oid].append(e["score"])
                    matched[oid].append(g >= 0)
        for oid in test.obj_ids:
            if n_gt[oid] > 0:
                ap_per_obj[oid].append(pm.detection_ap(scores[oid], matched[oid], n_gt[oid]))
                n_gt_obj[oid] = n_gt[oid]
    per_obj = {o: float(np.mean(v)) for o, v in ap_per_obj.items()}
    return {"mAP": pm.mean_or_zero(per_obj.values()), "per_obj": per_obj}


def _localization(test: TestSet, ests: List[dict], errs, metric: str, ths: List[float]) -> dict:
    """BOP 2019 6D localization recall (top ``inst_count`` estimates per target)."""
    by_t: Dict[tuple, list] = defaultdict(list)
    for e in ests:
        by_t[(e["scene_id"], e["im_id"], e["obj_id"])].append(e)
    recalls, per_obj_rec = [], defaultdict(list)
    for th in ths:
        tps = tars = 0
        obj_tp, obj_tar = defaultdict(int), defaultdict(int)
        for t in test.targets19:
            sid, iid, oid = t["scene_id"], t["im_id"], t["obj_id"]
            gts = test.gt[sid].get(iid, [])
            info = test.gt_info[sid].get(iid, [])
            valid = [g["obj_id"] == oid and info[j]["visib_fract"] >= pm.VISIB_GT_MIN for j, g in enumerate(gts)]
            n_valid = sum(valid)
            lst = sorted(by_t.get((sid, iid, oid), []), key=lambda e: e["score"], reverse=True)[: t["inst_count"]]
            es = [{"est_id": e["est_id"], "score": e["score"],
                   "errors": errs.get(e["est_id"], {}).get(metric, {})} for e in lst]
            n_tp = len(pm.match_poses(es, th, gt_valid=valid))
            tps += n_tp; tars += n_valid
            obj_tp[oid] += n_tp; obj_tar[oid] += n_valid
        recalls.append(tps / tars if tars else 0.0)
        for o in obj_tar:
            per_obj_rec[o].append(obj_tp[o] / obj_tar[o] if obj_tar[o] else 0.0)
    return {"AR": pm.mean_or_zero(recalls), "per_obj": {o: float(np.mean(v)) for o, v in per_obj_rec.items()}}
