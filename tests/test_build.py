"""Build a benchmark from a miniature fake release, then score its own reference."""
import gzip
import json

import numpy as np
import pytest
from PIL import Image

from kitchenbench import build, cli, manifest, pose_eval
from conftest import csv_rows, rot_z

trimesh = pytest.importorskip("trimesh")


def _fake_release(root, n_frames=40):
    rel = root / "rel"
    models = root / "models"
    models.mkdir(parents=True)
    for oid, ext in ((1, (60, 40, 120)), (2, (80, 80, 50))):
        trimesh.creation.box(extents=ext).export(models / f"obj_{oid:06d}.ply")
    qa = []
    rng = np.random.default_rng(0)
    for n in (1, 2):
        d = rel / f"KITchen_labels_{n}"
        (d / "labels").mkdir(parents=True)
        (d / "masks").mkdir()
        (d / "seg_labels").mkdir()
        (d / "classes.txt").write_text("other\nbox_a\nbox_b\n")
        ignore = {}
        for k in range(n_frames):
            stem = f"{(n - 1) * 25000 + k:06d}"
            rows, segs = [], []
            for i, (oid, x) in enumerate(((1, -150.0), (2, 150.0))):
                R = rot_z(rng.uniform(0, 6))
                t = np.array([x, 0.0, 900.0])
                pts = np.asarray(trimesh.load(models / f"obj_{oid:06d}.ply").vertices) @ R.T + t
                u = 918.7635 * pts[:, 0] / pts[:, 2] + 957.855
                v = 918.3555 * pts[:, 1] / pts[:, 2] + 552.21
                x0, x1, y0, y1 = int(u.min()), int(u.max()), int(v.min()), int(v.max())
                m = np.zeros((1080, 1920), np.uint8)
                m[y0:y1, x0:x1] = 255
                Image.fromarray(m).save(d / "masks" / f"{stem}_obj{i:04d}.png")
                rows.append(f"{oid} {(x0 + x1) / 3840:.6f} {(y0 + y1) / 2160:.6f} {(x1 - x0) / 1920:.6f} {(y1 - y0) / 1080:.6f}")
                segs.append(f"{oid} {x0/1920:.5f} {y0/1080:.5f} {x1/1920:.5f} {y0/1080:.5f} {x1/1920:.5f} {y1/1080:.5f}")
                accept = not (oid == 2 and k % 5 == 0)       # some rejected poses
                qa.append({"stem": stem, "idx": i, "obj_id": oid, "cam_R_m2c": R.ravel().tolist(),
                           "cam_t_m2c": (t / 1000).tolist(), "in_release_new": accept})
            if k % 7 == 0:
                ignore[stem] = [0]
            (d / "labels" / f"{stem}.txt").write_text("\n".join(rows) + "\n")
            (d / "seg_labels" / f"{stem}.txt").write_text("\n".join(segs) + "\n")
        (d / "ignore.json").write_text(json.dumps(ignore))
    qa_path = root / "qa.jsonl.gz"
    with gzip.open(qa_path, "wt") as f:
        f.write("\n".join(json.dumps(r) for r in qa) + "\n")
    syms = root / "sym.json"
    syms.write_text(json.dumps({"2": {"symmetries_discrete": [[-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]]}}))
    return rel, models, qa_path, syms, qa


def test_build_then_reference_scores_one(tmp_path, capsys):
    rel, models, qa_path, syms, qa = _fake_release(tmp_path)
    out = tmp_path / "bench"
    rc = cli.main(["build", "--release", str(rel), "--qa", str(qa_path), "--models", str(models),
                   "--symmetries", str(syms), "--out", str(out), "--block", "10", "--buffer", "2",
                   "--stride", "2", "--frac", "0.5,0.25,0.25", "--splits", "1,2"])
    assert rc == 0
    assert cli.main(["verify", str(out)]) == 0
    meta = manifest.load(out)
    test = set((out / "splits" / "test.txt").read_text().split())
    train = set((out / "splits" / "train.txt").read_text().split())
    assert test and train and not (test & train)
    # no train frame sits within 2 frames of a test frame
    assert all(abs(int(a) - int(b)) > 2 for a in train for b in test)
    assert meta["counts"]["test"]["pose_ignored_pairs"] > 0

    # Score the reference poses themselves: must be perfect.
    ts = pose_eval.TestSet(out / "bop")
    rows = []
    for t in ts.targets24:
        for g in ts.gt[t["scene_id"]][t["im_id"]]:
            rows.append((t["scene_id"], t["im_id"], g["obj_id"], 1.0, g["cam_R_m2c"], g["cam_t_m2c"]))
    pred = tmp_path / "ref.csv"
    pred.write_text(csv_rows(rows))
    r = pose_eval.evaluate(ts, pose_eval.load_bop_csv(pred))
    assert r["AP_6D"] == pytest.approx(1.0) and r["AR_6D"] == pytest.approx(1.0)
    info = json.loads((out / "bop" / "test" / next(iter(sorted(p.name for p in (out / "bop" / "test").iterdir()))) / "scene_gt_info.json").read_text())
    vis = [x["visib_fract"] for v in info.values() for x in v]
    assert vis and min(vis) > 0.5          # unoccluded boxes: mask ≈ silhouette

    coco = json.loads((out / "coco" / "test.json").read_text())
    res = [{"image_id": a["image_id"], "category_id": a["category_id"], "bbox": a["bbox"],
            "score": 1.0, "segmentation": a["segmentation"]} for a in coco["annotations"] if not a["iscrowd"]]
    from kitchenbench import coco_eval
    for t in ("bbox", "segm"):
        assert coco_eval.evaluate(out / "coco" / "test.json", res, t)["AP"] == pytest.approx(1.0)
    train_json = json.loads((out / "coco" / "train.json").read_text())
    assert len(train_json["images"]) == len(train)
    assert all("segmentation" in a for a in train_json["annotations"])


def test_parallel_build_is_identical(tmp_path):
    rel, models, qa_path, syms, qa = _fake_release(tmp_path)
    args = ["--release", str(rel), "--qa", str(qa_path), "--models", str(models), "--symmetries", str(syms),
            "--block", "10", "--buffer", "2", "--stride", "2", "--frac", "0.5,0.25,0.25", "--splits", "1,2"]
    assert cli.main(["build", *args, "--out", str(tmp_path / "s")]) == 0
    assert cli.main(["build", *args, "--out", str(tmp_path / "p"), "--workers", "3"]) == 0
    assert manifest.load(tmp_path / "s")["ground_truth_sha256"] == manifest.load(tmp_path / "p")["ground_truth_sha256"]


def _setup_file(path, stems, split_of):
    # setups recur in non-adjacent clips, as in the release
    rows = ["stem,setup,split"] + [f"{s},s{(int(s) % 25000) // 5 % 4},{split_of[(int(s) % 25000) // 5 % 4]}"
                                   for s in stems]
    path.write_text("\n".join(rows) + "\n")


def test_setup_split_keeps_setups_whole(tmp_path):
    rel, models, qa_path, syms, qa = _fake_release(tmp_path)
    stems = sorted({r["stem"] for r in qa})
    split_file = tmp_path / "setups.csv"
    _setup_file(split_file, stems, {0: "test", 1: "val", 2: "train", 3: "train"})
    out = tmp_path / "bench"
    assert cli.main(["build", "--release", str(rel), "--qa", str(qa_path), "--models", str(models),
                     "--symmetries", str(syms), "--out", str(out), "--stride", "2", "--splits", "1,2",
                     "--split-file", str(split_file)]) == 0
    assert cli.main(["verify", str(out)]) == 0
    rows = (out / "splits" / "setups.csv").read_text().split()[1:]
    split_of = {}
    for r in rows:
        _, setup, sub = r.split(",")
        assert split_of.setdefault(setup, sub) == sub      # a setup never spans two splits
    assert split_of == {"s0": "test", "s1": "val", "s2": "train", "s3": "train"}
    test = set((out / "splits" / "test.txt").read_text().split())
    assert test == {s for s in stems if (int(s) % 25000) // 5 % 4 == 0}
    assert manifest.load(out)["split_config"]["unit"] == "setup"


def test_setup_split_rejects_a_setup_in_two_splits(tmp_path):
    rel, models, qa_path, syms, qa = _fake_release(tmp_path)
    split_file = tmp_path / "setups.csv"
    split_file.write_text("stem,setup,split\n000000,s0,test\n000001,s0,train\n")
    with pytest.raises(SystemExit, match="more than one split"):
        cli.main(["build", "--release", str(rel), "--qa", str(qa_path), "--models", str(models),
                  "--out", str(tmp_path / "bench"), "--splits", "1,2", "--split-file", str(split_file)])
