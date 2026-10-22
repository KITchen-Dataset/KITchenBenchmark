"""A tiny synthetic benchmark folder (2 scenes, 3 objects) for end-to-end tests."""
import json
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from kitchenbench import manifest

K = [918.7635, 0, 957.855, 0, 918.3555, 552.21, 0, 0, 1]
W, H = 1920, 1080


def write_ply(path: Path, pts: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\nproperty float x\n"
                "property float y\nproperty float z\nend_header\n")
        for p in pts:
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


@pytest.fixture
def bench(tmp_path):
    rng = np.random.default_rng(0)
    b = tmp_path / "bench"
    bop = b / "bop"
    info = {}
    for oid in (1, 2, 3):
        pts = rng.normal(size=(200, 3)) * 30
        write_ply(bop / "models_eval" / f"obj_{oid:06d}.ply", pts)
        d = float(np.max(np.linalg.norm(pts[:, None] - pts[None], axis=2)))
        info[str(oid)] = {"diameter": d}
    info["2"]["symmetries_continuous"] = [{"axis": [0, 0, 1], "offset": [0, 0, 0]}]
    (bop / "models_info.json").write_text(json.dumps(info))
    (bop / "camera.json").write_text(json.dumps({"width": W, "height": H}))

    gt_poses = {}
    t19, t24, ign = [], [], []
    for sid in (1, 2):
        sd = bop / "test" / f"{sid:06d}"
        sd.mkdir(parents=True)
        sg, si, sc = {}, {}, {}
        for iid in (sid * 100, sid * 100 + 1):
            objs = []
            for oid, x in ((1, -150), (2, 0), (3, 150)):
                R = rot_z(rng.uniform(0, 6.28))
                objs.append({"obj_id": oid, "cam_R_m2c": R.ravel().tolist(), "cam_t_m2c": [x, 20.0, 1000.0]})
            # obj 3 in the 2nd image of scene 2 is almost hidden
            vis = [0.9, 0.8, 0.05 if (sid, iid) == (2, 201) else 0.7]
            sg[str(iid)] = objs
            si[str(iid)] = [{"visib_fract": v} for v in vis]
            sc[str(iid)] = {"cam_K": K}
            gt_poses[(sid, iid)] = objs
            t24.append({"scene_id": sid, "im_id": iid})
            for j, o in enumerate(objs):
                if vis[j] >= 0.1:
                    t19.append({"scene_id": sid, "im_id": iid, "obj_id": o["obj_id"], "inst_count": 1})
        ign.append({"scene_id": 1, "im_id": 100, "obj_id": 5})   # unverified object, no GT
        for name, d in (("scene_gt", sg), ("scene_gt_info", si), ("scene_camera", sc)):
            (sd / f"{name}.json").write_text(json.dumps(d))
    (bop / "test_targets_bop19.json").write_text(json.dumps(t19))
    (bop / "test_targets_bop24.json").write_text(json.dumps(t24))
    (bop / "test_ignore.json").write_text(json.dumps(ign))
    info["5"] = {"diameter": 50.0}
    write_ply(bop / "models_eval" / "obj_000005.ply", rng.normal(size=(50, 3)) * 10)
    (bop / "models_info.json").write_text(json.dumps(info))

    # COCO side: 2 images, 2 boxes each + 1 ignore region
    coco = {"images": [], "annotations": [], "categories": [{"id": i, "name": f"c{i}"} for i in (1, 2, 3)]}
    aid = 1
    for im in (100, 101):
        coco["images"].append({"id": im, "width": W, "height": H, "file_name": f"{im}.png"})
        for cat, box, crowd in ((1, [100, 100, 200, 150], 0), (2, [800, 400, 120, 300], 0), (3, [1500, 50, 80, 80], 1)):
            x, y, w, h = box
            coco["annotations"].append({"id": aid, "image_id": im, "category_id": cat, "bbox": box,
                                        "area": w * h, "iscrowd": crowd,
                                        "segmentation": [[x, y, x + w, y, x + w, y + h, x, y + h]]})
            aid += 1
    (b / "coco").mkdir(parents=True)
    (b / "coco" / "test.json").write_text(json.dumps(coco))
    (b / "splits").mkdir()
    (b / "splits" / "test.txt").write_text("100\n101\n")
    manifest.write(b, {"version": "test"})
    return SimpleNamespace(path=b, gt=gt_poses)


def csv_rows(rows):
    out = ["scene_id,im_id,obj_id,score,R,t,time"]
    for sid, iid, oid, s, R, t in rows:
        out.append(f"{sid},{iid},{oid},{s},{' '.join(map(str, np.ravel(R)))},{' '.join(map(str, np.ravel(t)))},0.5")
    return "\n".join(out) + "\n"
