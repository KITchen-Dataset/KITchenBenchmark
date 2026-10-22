"""Propose object symmetries (BOP ``models_info`` format) for human review.

MSSD/MSPD forgive pose ambiguities only for symmetries listed in
``models_info.json``. Like BOP, a symmetry counts only if shape AND texture look
the same after the transformation, so a can with a printed label is not
symmetric. This tool tests each principal axis of every mesh for continuous
rotational symmetry and for 180 degree flips, using surface samples with their
colours, and writes proposals plus the evidence. A person confirms them before
they are used (``kitchenbench build --symmetries``).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ANGLES = np.deg2rad(np.arange(20, 360, 20))


def _rot(axis, ang):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    c, s = np.cos(ang), np.sin(ang)
    x = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) * c + (1 - c) * np.outer(a, a) + s * x


def _sample(mesh, n, seed):
    """Surface points and their RGB in [0, 1] (texture or vertex colours), or None."""
    import trimesh
    pts, face = trimesh.sample.sample_surface(mesh, n, seed=seed)
    col = None
    try:
        vis = mesh.visual
        if vis.kind == "texture":
            uv = vis.uv
            if uv is not None and getattr(vis.material, "image", None) is not None \
                    or getattr(vis.material, "baseColorTexture", None) is not None:
                bary = trimesh.triangles.points_to_barycentric(mesh.triangles[face], pts)
                uvs = np.einsum("nk,nkj->nj", bary, uv[mesh.faces[face]])
                col = np.asarray(vis.material.to_color(uvs), float)[:, :3] / 255.0
        elif vis.kind == "vertex":
            vc = np.asarray(vis.vertex_colors, float)[:, :3] / 255.0
            bary = trimesh.triangles.points_to_barycentric(mesh.triangles[face], pts)
            col = np.einsum("nk,nkj->nj", bary, vc[mesh.faces[face]])
        elif vis.kind == "face":
            col = np.asarray(vis.face_colors, float)[face, :3] / 255.0
    except Exception:
        col = None
    if col is not None and np.ptp(col, axis=0).max() < 1e-3:
        col = None          # one flat colour carries no information
    return np.asarray(pts, float), col


def _sample_geo(mesh, n, seed):
    import trimesh
    return np.asarray(trimesh.sample.sample_surface(mesh, n, seed=seed)[0], float), None


def _score(pts, col, tree, center, R, diam, ref_col=None):
    q = (pts - center) @ R.T + center
    ref_col = col if ref_col is None else ref_col
    d, j = tree.query(q)
    geo = float(np.percentile(d, 95)) / diam
    tex = float(np.percentile(np.abs(ref_col[j] - col).max(axis=1), 90)) if col is not None else 0.0
    return geo, tex


def propose(mesh, seed=0, n=6000, geo_tol=0.01, tex_tol=0.08) -> dict:
    """Tolerances are added to the noise floor of two independent surface samples."""
    from scipy.spatial import cKDTree
    ref, ref_col = _sample(mesh, n, seed)
    pts, col = _sample(mesh, n, seed + 7919)
    dense, _ = _sample_geo(mesh, 200000, seed + 1)     # dense sample: stable centre and axes
    center = dense.mean(axis=0)
    diam = float(np.linalg.norm(ref.max(0) - ref.min(0)))
    _, _, vt = np.linalg.svd(dense - center, full_matrices=False)
    tree = cKDTree(ref)
    if col is not None:   # blur colours over ~16 neighbours: compare appearance, not texel noise
        ref_col = ref_col[tree.query(ref, k=16)[1]].mean(axis=1)
        col = col[cKDTree(pts).query(pts, k=16)[1]].mean(axis=1)
    g0, t0 = _score(pts, col, tree, center, np.eye(3), diam, ref_col)
    geo_tol, tex_tol = g0 + geo_tol, t0 + tex_tol
    out = {"symmetries_continuous": [], "symmetries_discrete": [], "evidence": [], "textured": col is not None,
           "noise_floor": [round(g0, 4), round(t0, 4)]}
    for axis in vt:
        axis = axis / np.linalg.norm(axis)
        errs = [_score(pts, col, tree, center, _rot(axis, a), diam, ref_col) for a in ANGLES]
        cont = all(g < geo_tol and t < tex_tol for g, t in errs)
        flip = _score(pts, col, tree, center, _rot(axis, np.pi), diam, ref_col)
        out["evidence"].append({"axis": np.round(axis, 4).tolist(),
                                "worst_rotation": [round(max(e[0] for e in errs), 4), round(max(e[1] for e in errs), 4)],
                                "flip_180": [round(flip[0], 4), round(flip[1], 4)]})
        if cont:
            out["symmetries_continuous"].append({"axis": np.round(axis, 6).tolist(), "offset": np.round(center, 4).tolist()})
        elif flip[0] < geo_tol and flip[1] < tex_tol:
            R = _rot(axis, np.pi)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = center - R @ center
            out["symmetries_discrete"].append(np.round(T, 6).ravel().tolist())
    if out["symmetries_continuous"]:
        out["symmetries_discrete"] = [T for T in out["symmetries_discrete"]
                                      if not _about(T, out["symmetries_continuous"][0]["axis"])]
        out["symmetries_continuous"] = out["symmetries_continuous"][:1]   # >1 axis would mean a sphere
    return out


def _about(T, axis) -> bool:
    R = np.reshape(T, (4, 4))[:3, :3]
    return np.allclose(R @ np.asarray(axis), axis, atol=1e-3)


def run(models: Path, out: Path, names: list | None = None) -> dict:
    import trimesh
    res = {}
    for p in sorted(models.glob("obj_*.ply")):
        oid = int(p.stem.split("_")[1])
        m = trimesh.load(str(p), force="mesh")
        r = propose(m, seed=oid)
        r["name"] = names[oid] if names and oid < len(names) else ""
        r["status"] = "proposed"   # set to "confirmed" or edit after looking at the object
        res[str(oid)] = r
        print(oid, r["name"], "continuous" if r["symmetries_continuous"] else "",
              f"{len(r['symmetries_discrete'])} flip(s)" if r["symmetries_discrete"] else "", flush=True)
    out.write_text(json.dumps(res, indent=1))
    return res
