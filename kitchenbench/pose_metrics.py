"""6D pose errors, matching and scores, following the BOP toolkit exactly.

Everything here mirrors ``bop_toolkit_lib`` (``pose_error.mssd/mspd``,
``misc.get_symmetry_transformations``, ``pose_matching.match_poses``,
``score.calc_ap``, ``score.calc_pose_detection_scores``) so that KITchen numbers
are directly comparable to BOP numbers. ``tests/test_bop_parity.py`` checks the
equivalence against the BOP toolkit whenever it is installed.

Units: rotations are 3x3 object-to-camera matrices, translations and model
points are in millimetres, as in BOP.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence

import numpy as np

# BOP 2019 / 2024 correctness thresholds.
MSSD_THS = [round(x, 2) for x in np.arange(0.05, 0.51, 0.05)]   # fraction of diameter
MSPD_THS = [float(x) for x in range(5, 51, 5)]                   # px at 640 px width
MAX_SYM_DISC_STEP = 0.01
VISIB_GT_MIN = 0.1


# ── Symmetries ─────────────────────────────────────────────────────────────────

def _rotation_about(angle: float, axis: Sequence[float]) -> np.ndarray:
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    c, s = math.cos(angle), math.sin(angle)
    R = np.diag([c, c, c]) + np.outer(a, a) * (1.0 - c)
    a = a * s
    R += np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return R


def symmetry_transformations(model_info: dict, max_sym_disc_step: float = MAX_SYM_DISC_STEP) -> List[dict]:
    """Discrete + discretised continuous symmetries of one model (BOP semantics)."""
    trans_disc = [{"R": np.eye(3), "t": np.zeros((3, 1))}]
    for sym in model_info.get("symmetries_discrete", []):
        m = np.reshape(np.asarray(sym, dtype=np.float64), (4, 4))
        trans_disc.append({"R": m[:3, :3], "t": m[:3, 3].reshape(3, 1)})

    trans_cont = []
    for sym in model_info.get("symmetries_continuous", []):
        offset = np.asarray(sym["offset"], dtype=np.float64).reshape(3, 1)
        n = int(np.ceil(np.pi / max_sym_disc_step))
        step = 2.0 * np.pi / n
        for i in range(n):
            R = _rotation_about(i * step, sym["axis"])
            trans_cont.append({"R": R, "t": -R.dot(offset) + offset})

    if not trans_cont:
        return trans_disc
    out = []
    for d in trans_disc:
        for c in trans_cont:
            out.append({"R": c["R"].dot(d["R"]), "t": c["R"].dot(d["t"]) + c["t"]})
    return out


# ── Errors ─────────────────────────────────────────────────────────────────────

def _transform(pts: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return pts @ R.T + np.asarray(t, dtype=np.float64).reshape(1, 3)


def _project(pts: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    x = _transform(pts, R, t) @ K.T
    return x[:, :2] / x[:, 2:3]


def mssd(R_est, t_est, R_gt, t_gt, pts, syms) -> float:
    """Maximum Symmetry-aware Surface Distance (mm)."""
    p_est = _transform(pts, R_est, t_est)
    best = math.inf
    for s in syms:
        p_gt = _transform(pts, R_gt @ s["R"], R_gt @ s["t"] + np.reshape(t_gt, (3, 1)))
        best = min(best, float(np.linalg.norm(p_est - p_gt, axis=1).max()))
    return best


def mspd(R_est, t_est, R_gt, t_gt, K, pts, syms) -> float:
    """Maximum Symmetry-aware Projection Distance (px, not yet width-normalised)."""
    p_est = _project(pts, K, R_est, t_est)
    best = math.inf
    for s in syms:
        p_gt = _project(pts, K, R_gt @ s["R"], R_gt @ s["t"] + np.reshape(t_gt, (3, 1)))
        best = min(best, float(np.linalg.norm(p_est - p_gt, axis=1).max()))
    return best


# ── Matching and scores ────────────────────────────────────────────────────────

def match_poses(errs: List[dict], th: float, max_ests: int = 0, gt_valid: Sequence[bool] | None = None) -> List[dict]:
    """Greedy matching by decreasing score (``bop_toolkit_lib.pose_matching``).

    Each ``errs`` item: ``{"est_id", "score", "errors": {gt_id: err}}``.
    Returns one ``{"est_id", "gt_id", "score", "error"}`` per matched estimate.
    """
    ordered = sorted(errs, key=lambda e: e["score"], reverse=True)
    if max_ests > 0:
        ordered = ordered[:max_ests]
    matched_gt: set = set()
    out = []
    for e in ordered:
        best_gt, best_err = -1, th
        for gt_id, err in e["errors"].items():
            if (gt_valid is None or gt_valid[gt_id]) and gt_id not in matched_gt and err < best_err:
                best_gt, best_err = gt_id, err
        if best_gt >= 0:
            matched_gt.add(best_gt)
            out.append({"est_id": e["est_id"], "gt_id": best_gt, "score": e["score"], "error": best_err})
    return out


def calc_ap(rec: Iterable[float], pre: Iterable[float]) -> float:
    """COCO-style 101-point interpolated AP (``bop_toolkit_lib.score.calc_ap``)."""
    rec = np.asarray(list(rec), dtype=np.float64)
    pre = np.asarray(list(pre), dtype=np.float64)
    i = np.argsort(rec, kind="stable")
    mrec = np.concatenate(([0.0], rec[i], [1.0]))
    mpre = np.concatenate(([0.0], pre[i], [0.0]))
    vals = []
    for rt in np.linspace(0.0, 1.0, 101):
        sel = mrec >= rt
        vals.append(float(mpre[sel].max()) if sel.any() else 0.0)
    return float(np.mean(vals))


def detection_ap(est_scores: List[float], est_matched: List[bool], n_gt: int) -> float:
    """AP of one object class from its (non-ignored) estimates."""
    if n_gt == 0:
        return 0.0
    order = np.argsort(-np.asarray(est_scores, dtype=np.float64), kind="stable")
    tp = np.asarray(est_matched, dtype=bool)[order]
    tpc = np.cumsum(tp)
    fpc = np.cumsum(~tp)
    rec = tpc / float(n_gt)
    with np.errstate(divide="ignore", invalid="ignore"):
        pre = tpc / (tpc + fpc)
    pre[~np.isfinite(pre)] = 0.0
    return calc_ap(rec, pre)


def mean_or_zero(values: Iterable[float]) -> float:
    v = list(values)
    return float(np.mean(v)) if v else 0.0


__all__ = [
    "MSSD_THS", "MSPD_THS", "MAX_SYM_DISC_STEP", "VISIB_GT_MIN",
    "symmetry_transformations", "mssd", "mspd", "match_poses", "calc_ap", "detection_ap",
]
