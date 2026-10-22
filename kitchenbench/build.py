"""Freeze a benchmark version from a KITchen release (maintainers only).

    kitchenbench build --release /media/nagi/DATA/kat_release \\
        --qa pose_quality/v2_harmonized/scene_gt_qa_harmonized.jsonl.gz \\
        --models "/media/nagi/DATA/models(1)/models" \\
        --symmetries symmetries.json --out kitchen_bench_v1.0 --version 1.0

What it writes (everything an evaluator needs, nothing else)::

    benchmark.json          version, sources, split config, sha256 of every file
    splits/                 train.txt val.txt test.txt (all frames), val_eval.txt test_eval.txt
                            setups.csv (stem, setup, split; with ``--split-file``)
    coco/train.json         boxes + polygons (from seg_labels) for training
    coco/val.json test.json boxes + RLE masks of the evaluation frames
    bop/                    BOP layout for the evaluation frames: models_info.json,
                            models_eval/*.ply, camera.json, {val,test}/<scene>/scene_*.json,
                            {val,test}_targets_bop19.json, _bop24.json, _ignore.json

Split design. KITchen has no recorded sequence boundaries and consecutive
frames are near-duplicates, so frames are grouped into blocks of consecutive
stems (``--block``), whole blocks are assigned to train/val/test with a fixed
seed inside each of the four release folders, and train frames closer than
``--buffer`` frames to a val/test block are dropped. The evaluation frames are
every ``--stride``-th frame of the val/test blocks. A block is one BOP scene.

With ``--split-file`` (CSV ``stem,setup,split``) the blocks are replaced by
physical setups: the release is short clips in mixed order and one table/object
arrangement recurs across many clips, so whole setups go to one split. Frames
not in the file are left out (``buffer``). ``scene_id`` stays ``frame // block``.

Ground truth rules:
* Boxes/masks: every label row except class 0 ("other", one instance, no model).
  Rows listed in ``ignore.json`` (not confirmed by a second detector) become
  ``iscrowd=1`` ignore regions. A row whose mask PNG is missing or empty (0 bytes;
  23 rows of frame 095950 in the release) becomes a box-only ignore region.
* 6D poses: only rows whose pose is accepted in the QA file. If any instance of
  an object in a frame lacks an accepted pose, that (frame, object) pair goes
  to ``*_ignore.json`` and none of its instances is a target.
* ``visib_fract`` = mask pixels / pixels covered by the posed model in the image.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Dict, List

import numpy as np

from . import __version__, manifest
from .report import sha256_file

K = [918.7635, 0.0, 957.855, 0.0, 918.3555, 552.21, 0.0, 0.0, 1.0]   # Azure Kinect colour @1920x1080
W, H = 1920, 1080
N_EVAL_PTS = 1000


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--release", type=Path, required=True, help="folder holding KITchen_labels_<N>/")
    p.add_argument("--qa", type=Path, required=True, help="per-row pose QA (.jsonl or .jsonl.gz) with the accept flag")
    p.add_argument("--accept-key", default="in_release_new", help="QA field that marks an accepted pose")
    p.add_argument("--pose-version", default="v2_harmonized", help="name recorded in benchmark.json")
    p.add_argument("--models", type=Path, required=True, help="folder with obj_<id>.ply meshes in mm")
    p.add_argument("--symmetries", type=Path, help="reviewed symmetries.json {obj_id: {symmetries_*}}")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--version", default="1.0")
    p.add_argument("--block", type=int, default=250, help="frames per block")
    p.add_argument("--buffer", type=int, default=25, help="train frames dropped next to val/test blocks")
    p.add_argument("--frac", default="0.7,0.1,0.2", help="train,val,test fractions of blocks")
    p.add_argument("--stride", type=int, default=6, help="every n-th val/test frame is evaluated")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--split-file", type=Path, help="CSV stem,setup,split assigning whole setups; replaces the block split")
    p.add_argument("--splits", default="1,2,3,4", help="release folders to use")
    p.add_argument("--skip-train-json", action="store_true", help="skip coco/train.json (quick checks)")
    p.add_argument("--exclude", type=Path, help="text file of stems to leave out entirely")
    p.add_argument("--workers", type=int, default=1, help="processes for models and evaluation frames (output is identical)")
    p.add_argument("--limit", type=int, default=0, help="(testing) evaluate only the first N frames per subset")


# ── Release reading ────────────────────────────────────────────────────────────

def read_labels(path: Path) -> List[list]:
    rows = []
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) >= 5:
            rows.append([int(p[0])] + [float(v) for v in p[1:5]])
    return rows


def frames_of(lab_dir: Path) -> List[str]:
    return sorted(p.stem for p in (lab_dir / "labels").glob("*.txt"))


# ── Splits ─────────────────────────────────────────────────────────────────────

def assign_splits(stems: List[str], block: int, buffer: int, frac, seed: int, tag: str) -> Dict[str, str]:
    """Return {stem: train|val|test|buffer} for the stems of one release folder."""
    width = len(stems[0])
    nums = sorted(int(s) for s in stems)
    blocks = sorted({n // block for n in nums})
    order = blocks[:]
    random.Random(f"{seed}-{tag}").shuffle(order)
    n_test = round(frac[2] * len(order))
    n_val = round(frac[1] * len(order))
    sub_of = {b: ("test" if i < n_test else "val" if i < n_test + n_val else "train") for i, b in enumerate(order)}
    out = {}
    for n in nums:
        b, pos = n // block, n % block
        s = sub_of[b]
        if s == "train" and buffer > 0 and (
                (pos < buffer and sub_of.get(b - 1, "train") != "train")
                or (pos >= block - buffer and sub_of.get(b + 1, "train") != "train")):
            s = "buffer"
        out[str(n).zfill(width)] = s
    return out


def read_split_file(path: Path) -> Dict[str, tuple]:
    """{stem: (setup, split)}; every setup must sit in exactly one of train/val/test."""
    rows, subs = {}, defaultdict(set)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["split"] not in ("train", "val", "test", "buffer"):
                raise SystemExit(f"{path}: unknown split {r['split']!r} for {r['stem']}")
            rows[r["stem"]] = (r["setup"], r["split"])
            if r["split"] != "buffer":
                subs[r["setup"]].add(r["split"])
    mixed = sorted(k for k, v in subs.items() if len(v) > 1)
    if mixed:
        raise SystemExit(f"{path}: setups in more than one split: {mixed[:5]}")
    return rows


def eval_frames(stems: List[str], stride: int, block: int) -> List[str]:
    return [s for s in stems if (int(s) % block) % stride == 0]


# ── Geometry ───────────────────────────────────────────────────────────────────

def diameter(vertices: np.ndarray) -> float:
    try:
        from scipy.spatial import ConvexHull
        v = vertices[ConvexHull(vertices).vertices]
    except Exception:
        v = vertices
    best = 0.0
    for i in range(0, len(v), 512):
        best = max(best, float(np.linalg.norm(v[i:i + 512, None] - v[None], axis=2).max()))
    return best


def write_ply(path: Path, pts: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(pts)}\n"
                 "property float x\nproperty float y\nproperty float z\nend_header\n").encode())
        f.write(np.asarray(pts, dtype="<f4").tobytes())


def sample_surface(mesh, n: int, seed: int) -> np.ndarray:
    """Area-weighted surface sample with NumPy only, so the frozen files do not
    depend on the trimesh version (its sampler changed between releases)."""
    v = np.asarray(mesh.vertices, dtype=np.float64)
    tri = v[np.asarray(mesh.faces)]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(tri), size=n, p=area / area.sum())
    u, w = rng.random((2, n))
    flip = u + w > 1
    u[flip], w[flip] = 1 - u[flip], 1 - w[flip]
    a, b, c = tri[idx, 0], tri[idx, 1], tri[idx, 2]
    return a + u[:, None] * (b - a) + w[:, None] * (c - a)


class Silhouettes:
    """Full (un-occluded) model silhouette area inside the image, from a dense surface sample."""

    N_PTS = 30000

    def __init__(self, pts: Dict[int, np.ndarray], scale: int = 4):
        self.scale = scale
        self.pts = pts

    def area(self, oid: int, R: np.ndarray, t_mm: np.ndarray) -> int:
        from scipy.ndimage import binary_closing
        s = self.scale
        x = self.pts[oid] @ R.T + t_mm
        x = x[x[:, 2] > 1.0]
        u = (K[0] * x[:, 0] / x[:, 2] + K[2]) / s
        v = (K[4] * x[:, 1] / x[:, 2] + K[5]) / s
        w, h = W // s, H // s
        ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        img = np.zeros((h, w), bool)
        img[v[ok].astype(int), u[ok].astype(int)] = True
        return int(binary_closing(img, iterations=2).sum()) * s * s


# ── Inputs ─────────────────────────────────────────────────────────────────────

def load_qa(path: Path, wanted: set, key: str) -> Dict[str, Dict[int, dict]]:
    """{stem: {label row: {"accept", "obj_id", "R", "t_mm"}}} for the wanted stems only."""
    op = gzip.open if path.suffix == ".gz" else open
    out: Dict[str, Dict[int, dict]] = defaultdict(dict)
    with op(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            stem = str(r["stem"]).zfill(6)
            if stem not in wanted or r.get("cam_R_m2c") is None:
                continue
            t = np.asarray(r["cam_t_m2c"], dtype=np.float64)
            out[stem][int(r["idx"])] = {
                "accept": bool(r.get(key)), "obj_id": int(r["obj_id"]),
                "R": np.asarray(r["cam_R_m2c"], dtype=np.float64).reshape(3, 3),
                "t_mm": t * 1000.0 if np.abs(t).max() < 20 else t}
    return out


def rle_of(mask_path: Path):
    from PIL import Image
    from . import rle
    m = np.asarray(Image.open(mask_path))
    if m.ndim == 3:
        m = m[..., 0]
    m = m > 0
    return rle.encode(m), rle.area(m)


# ── Main ───────────────────────────────────────────────────────────────────────

_CTX: dict = {}  # shared with forked workers (read-only)


def _frame(stem: str):
    """Ground truth of one evaluation frame: COCO image + annotations, BOP poses."""
    c_ = _CTX
    n = c_["folder_of"][stem]
    d = c_["lab_dirs"][n]
    sil = c_["sil"]
    im_id, scene = int(stem), int(stem) // c_["block"]
    image = {"id": im_id, "file_name": f"KITchen_rgb_{n}/images/{stem}.png",
             "width": W, "height": H, "scene_id": scene}
    rows = read_labels(d / "labels" / f"{stem}.txt")
    ignore_rows = set(c_["ignores"][n].get(stem, []))
    frame_qa = c_["qa"].get(stem, {})
    anns, posed, bad_objs, areas = [], [], set(), {}
    for i, (c, cx, cy, w, h) in enumerate(rows):
        if c == 0:          # "other": one instance in the whole release, no model
            continue
        crowd = int(i in ignore_rows)
        box = [(cx - w / 2) * W, (cy - h / 2) * H, w * W, h * H]
        mp = d / "masks" / f"{stem}_obj{i:04d}.png"
        if mp.exists() and mp.stat().st_size > 0:
            seg, area = rle_of(mp)
        else:
            x, y, bw, bh = box
            seg, area = [[x, y, x + bw, y, x + bw, y + bh, x, y + bh]], 0
        anns.append({"image_id": im_id, "category_id": c,
                     "bbox": box, "area": area or box[2] * box[3],
                     "iscrowd": crowd if area else 1, "segmentation": seg,
                     "label_row": i})
        q = frame_qa.get(i)
        if q is None or not q["accept"] or crowd or not area or q["obj_id"] != c:
            bad_objs.add(c)
        else:
            posed.append((i, c, q))
            areas[i] = area
    objs, infos = [], []
    for i, c, q in posed:
        if c in bad_objs:
            continue
        full = sil.area(c, q["R"], q["t_mm"]) if c in sil.pts else 0
        objs.append({"obj_id": c, "cam_R_m2c": q["R"].ravel().tolist(), "cam_t_m2c": q["t_mm"].tolist()})
        infos.append({"visib_fract": round(min(1.0, areas[i] / full), 4) if full else 0.0,
                      "px_count_visib": areas[i], "label_row": i})
    return image, anns, objs, infos, bad_objs


def _model(path: Path):
    """Size, diameter and the two surface samples of one mesh."""
    import trimesh
    oid = int(re.match(r"obj_(\d+)\.ply$", path.name).group(1))
    m = trimesh.load(str(path), force="mesh", process=False)
    v = np.asarray(m.vertices, dtype=np.float64)
    lo, hi = v.min(0), v.max(0)
    info = {"diameter": diameter(v), "min_x": lo[0], "min_y": lo[1], "min_z": lo[2],
            "size_x": hi[0] - lo[0], "size_y": hi[1] - lo[1], "size_z": hi[2] - lo[2]}
    return (oid, {k: float(x) for k, x in info.items()},
            sample_surface(m, N_EVAL_PTS, seed=oid), sample_surface(m, Silhouettes.N_PTS, seed=oid))


def _pool_map(fn, items: list, workers: int, chunksize: int = 16):
    """Results in input order; with workers > 1, forked processes share _CTX."""
    if workers <= 1:
        yield from map(fn, items)
        return
    import multiprocessing as mp
    with mp.get_context("fork").Pool(workers) as pool:
        yield from pool.imap(fn, items, chunksize=chunksize)


def run(a) -> int:
    out: Path = a.out
    if (out / manifest.MANIFEST).exists() or (out.exists() and any(out.iterdir())):
        raise SystemExit(f"{out} is not empty; pick a new --out (versions are never overwritten)")
    frac = [float(x) for x in a.frac.split(",")]
    excluded = set(a.exclude.read_text().split()) if a.exclude else set()
    lab_dirs = {int(n): a.release / f"KITchen_labels_{n}" for n in a.splits.split(",")}
    classes = (lab_dirs[min(lab_dirs)] / "classes.txt").read_text().splitlines()
    ignores = {n: json.loads((d / "ignore.json").read_text()) for n, d in lab_dirs.items()}

    # 1. splits
    subset: Dict[str, str] = {}
    folder_of: Dict[str, int] = {}
    setups = read_split_file(a.split_file) if a.split_file else None
    for n, d in lab_dirs.items():
        stems = [s for s in frames_of(d) if s not in excluded]
        if setups is None:
            subset.update(assign_splits(stems, a.block, a.buffer, frac, a.seed, str(n)))
        else:
            subset.update({s: setups[s][1] if s in setups else "buffer" for s in stems})
        folder_of.update({s: n for s in stems})
    by_sub = defaultdict(list)
    for s, sub in sorted(subset.items()):
        by_sub[sub].append(s)
    evals = {sub: eval_frames(by_sub[sub], a.stride, a.block) for sub in ("val", "test")}
    if a.limit:
        evals = {k: v[: a.limit] for k, v in evals.items()}
    (out / "splits").mkdir(parents=True)
    for sub in ("train", "val", "test"):
        (out / "splits" / f"{sub}.txt").write_text("\n".join(by_sub[sub]) + "\n")
    for sub, st in evals.items():
        (out / "splits" / f"{sub}_eval.txt").write_text("\n".join(st) + "\n")
    if setups is not None:
        (out / "splits" / "setups.csv").write_text("stem,setup,split\n" + "".join(
            f"{s},{setups[s][0]},{subset[s]}\n" for s in sorted(subset) if s in setups))
    print({k: len(v) for k, v in by_sub.items()}, {f"{k}_eval": len(v) for k, v in evals.items()}, flush=True)

    # 2. models (one process per mesh; the output does not depend on --workers)
    syms = {int(k): v for k, v in json.loads(a.symmetries.read_text()).items()} if a.symmetries else {}
    paths = [p for p in sorted(a.models.glob("obj_*.ply")) if re.match(r"obj_(\d+)\.ply$", p.name)]
    models_info, sil_pts = {}, {}
    for oid, info, eval_pts, pts in _pool_map(_model, paths, a.workers, chunksize=1):
        for k in ("symmetries_discrete", "symmetries_continuous"):
            if syms.get(oid, {}).get(k):
                info[k] = syms[oid][k]
        models_info[str(oid)] = info
        sil_pts[oid] = pts
        write_ply(out / "bop" / "models_eval" / f"obj_{oid:06d}.ply", eval_pts)
    (out / "bop" / "models_info.json").write_text(json.dumps(models_info, indent=1))
    (out / "bop" / "camera.json").write_text(json.dumps({"cam_K": K, "width": W, "height": H, "depth_scale": 1.0}))
    sil = Silhouettes(sil_pts)
    print(f"{len(models_info)} models, {sum(1 for i in models_info.values() if len(i) > 7)} with symmetries", flush=True)

    # 3. evaluation frames
    qa = load_qa(a.qa, set(evals["val"]) | set(evals["test"]), a.accept_key)
    cats = [{"id": i, "name": classes[i], "supercategory": "object"} for i in range(1, len(classes))]
    counts = {}
    (out / "coco").mkdir(exist_ok=True)
    for sub, stems in evals.items():
        coco = {"info": {"description": f"KITchen benchmark {a.version} {sub}", "version": a.version},
                "images": [], "annotations": [], "categories": cats}
        gt, gt_info, cam = defaultdict(dict), defaultdict(dict), defaultdict(dict)
        ign, t19, t24 = [], [], []
        aid = 1
        _CTX.update(folder_of=folder_of, lab_dirs=lab_dirs, ignores=ignores, qa=qa, sil=sil, block=a.block)
        for k, (stem, rec) in enumerate(zip(stems, _pool_map(_frame, stems, a.workers))):
            im_id, scene = int(stem), int(stem) // a.block
            image, anns, objs, infos, bad_objs = rec
            coco["images"].append(image)
            for ann in anns:
                coco["annotations"].append({"id": aid, **ann})
                aid += 1
            ign += [{"scene_id": scene, "im_id": im_id, "obj_id": c} for c in sorted(bad_objs)]
            gt[scene][str(im_id)] = objs
            gt_info[scene][str(im_id)] = infos
            cam[scene][str(im_id)] = {"cam_K": K, "depth_scale": 1.0}
            t24.append({"scene_id": scene, "im_id": im_id})
            vis_objs = Counter(o["obj_id"] for o, inf in zip(objs, infos) if inf["visib_fract"] >= 0.1)
            t19 += [{"scene_id": scene, "im_id": im_id, "obj_id": c, "inst_count": nn} for c, nn in sorted(vis_objs.items())]
            if k % 200 == 0:
                print(f"{sub}: {k}/{len(stems)} frames", flush=True)
        (out / "coco" / f"{sub}.json").write_text(json.dumps(coco, separators=(",", ":")))
        for scene in gt:
            sd = out / "bop" / sub / f"{scene:06d}"
            sd.mkdir(parents=True, exist_ok=True)
            for name, data in (("scene_gt", gt), ("scene_gt_info", gt_info), ("scene_camera", cam)):
                (sd / f"{name}.json").write_text(json.dumps(data[scene], separators=(",", ":")))
        for name, data in (("targets_bop19", t19), ("targets_bop24", t24), ("ignore", ign)):
            (out / "bop" / f"{sub}_{name}.json").write_text(json.dumps(data, separators=(",", ":")))
        counts[sub] = {"eval_frames": len(stems),
                       "boxes": sum(1 for x in coco["annotations"] if not x["iscrowd"]),
                       "ignore_regions": sum(1 for x in coco["annotations"] if x["iscrowd"]),
                       "classes_present": len({x["category_id"] for x in coco["annotations"] if not x["iscrowd"]}),
                       "pose_targets": sum(t["inst_count"] for t in t19),
                       "pose_objects_present": len({t["obj_id"] for t in t19}),
                       "pose_ignored_pairs": len(ign)}
        print(sub, counts[sub], flush=True)

    # 4. training annotations (boxes + polygons; no pixel reads)
    if not a.skip_train_json:
        counts["train"] = write_train_json(out / "coco" / "train.json", by_sub["train"], folder_of,
                                           lab_dirs, ignores, cats, a.version)

    meta = {
        "name": "KITchen benchmark", "version": a.version, "built": str(date.today()),
        "builder": f"kitchenbench {__version__}",
        "sources": {"release": str(a.release), "pose_version": a.pose_version, "qa_file": a.qa.name,
                    "qa_sha256": sha256_file(a.qa), "accept_key": a.accept_key, "models": str(a.models),
                    "symmetries": a.symmetries.name if a.symmetries else "none (unreviewed)",
                    "excluded_frames": len(excluded)},
        "split_config": {"unit": "setup" if setups is not None else "block",
                         "split_file_sha256": sha256_file(a.split_file) if a.split_file else None,
                         "block": a.block, "buffer": a.buffer, "frac": frac, "stride": a.stride,
                         "seed": a.seed, "folders": sorted(lab_dirs), "limit": a.limit},
        "counts": {**{f"{k}_frames": len(v) for k, v in by_sub.items()}, **counts},
    }
    meta = manifest.write(out, meta)
    print(f"Benchmark {a.version} written to {out}; ground truth {meta['ground_truth_sha256'][:12]}")
    return 0


def write_train_json(path: Path, stems, folder_of, lab_dirs, ignores, cats, version) -> dict:
    """Stream the (large) training json so memory stays flat."""
    n_ann, aid = 0, 1
    with open(path, "w") as f:
        f.write('{"info":' + json.dumps({"description": f"KITchen benchmark {version} train", "version": version}))
        f.write(',"categories":' + json.dumps(cats) + ',"images":[')
        f.write(",".join(json.dumps({"id": int(s), "file_name": f"KITchen_rgb_{folder_of[s]}/images/{s}.png",
                                     "width": W, "height": H}) for s in stems))
        f.write('],"annotations":[')
        for s in stems:
            n = folder_of[s]
            d = lab_dirs[n]
            sp = d / "seg_labels" / f"{s}.txt"
            polys = sp.read_text().splitlines() if sp.exists() else []
            ign = set(ignores[n].get(s, []))
            for i, (c, cx, cy, w, h) in enumerate(read_labels(d / "labels" / f"{s}.txt")):
                if c == 0:
                    continue
                box = [(cx - w / 2) * W, (cy - h / 2) * H, w * W, h * H]
                ann = {"id": aid, "image_id": int(s), "category_id": c, "bbox": box,
                       "area": box[2] * box[3], "iscrowd": int(i in ign)}
                p = polys[i].split() if i < len(polys) else []
                if len(p) >= 7 and int(p[0]) == c:
                    xy = np.asarray(p[1:], dtype=np.float64).reshape(-1, 2) * [W, H]
                    ann["segmentation"] = [np.round(xy.ravel(), 1).tolist()]
                    x, y = xy[:, 0], xy[:, 1]
                    ann["area"] = float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
                f.write(("," if n_ann else "") + json.dumps(ann, separators=(",", ":")))
                aid += 1
                n_ann += 1
        f.write("]}")
    return {"frames": len(stems), "annotations": n_ann}
