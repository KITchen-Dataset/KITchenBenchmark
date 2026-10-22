import json

import numpy as np
import pytest
from pycocotools import mask as mask_utils

from kitchenbench import cli, coco_eval, manifest, pose_eval, validate
from conftest import csv_rows, rot_z


def _perfect_rows(gt):
    rows = []
    for (sid, iid), objs in gt.items():
        for o in objs:
            rows.append((sid, iid, o["obj_id"], 0.9, o["cam_R_m2c"], o["cam_t_m2c"]))
    return rows


def _score(bench, rows):
    p = bench.path / "pred.csv"
    p.write_text(csv_rows(rows))
    test = pose_eval.TestSet(bench.path / "bop")
    return pose_eval.evaluate(test, pose_eval.load_bop_csv(p))


def test_perfect_poses_score_one(bench):
    r = _score(bench, _perfect_rows(bench.gt))
    assert r["AP_6D"] == pytest.approx(1.0)
    assert r["AR_6D"] == pytest.approx(1.0)
    assert r["time_per_image_s"] == pytest.approx(0.5)


def test_no_poses_score_zero(bench):
    r = _score(bench, [])
    assert r["AP_6D"] == 0.0 and r["AR_6D"] == 0.0


def test_symmetric_object_rotated_about_axis_is_correct(bench):
    rows = []
    for sid, iid, oid, s, R, t in _perfect_rows(bench.gt):
        if oid == 2:
            R = (np.reshape(R, (3, 3)) @ rot_z(1.0)).ravel()   # obj 2 is symmetric about z
        rows.append((sid, iid, oid, s, R, t))
    assert _score(bench, rows)["AP_6D"] == pytest.approx(1.0)


def test_asymmetric_object_rotated_is_wrong(bench):
    rows = []
    for sid, iid, oid, s, R, t in _perfect_rows(bench.gt):
        if oid == 1:
            R = (np.reshape(R, (3, 3)) @ rot_z(np.pi)).ravel()
        rows.append((sid, iid, oid, s, R, t))
    r = _score(bench, rows)
    assert r["per_object_AP_6D"]["1"] == 0.0
    assert r["per_object_AP_6D"]["2"] == pytest.approx(1.0)


def test_ignored_and_hidden_objects_do_not_count(bench):
    rows = _perfect_rows(bench.gt)
    base = _score(bench, rows)
    # A guess for the unverified object 5 (listed in test_ignore.json) changes nothing.
    rows2 = rows + [(1, 100, 5, 0.99, np.eye(3).ravel(), [0, 0, 900])]
    assert _score(bench, rows2)["AP_6D"] == pytest.approx(base["AP_6D"])
    # Missing the almost hidden object 3 in image (2, 201) is not a miss.
    rows3 = [r for r in rows if not (r[0] == 2 and r[1] == 201 and r[2] == 3)]
    assert _score(bench, rows3)["AP_6D"] == pytest.approx(1.0)


def test_false_positive_ranked_first_costs_points(bench):
    fp = (1, 100, 1, 0.95, np.eye(3).ravel(), [400, 0, 1500])
    r = _score(bench, _perfect_rows(bench.gt) + [fp])
    assert r["AP_6D"] < 1.0 and r["AR_6D"] < 1.0   # top-scored wrong guess takes the target's slot
    low = (1, 100, 1, 0.1, np.eye(3).ravel(), [400, 0, 1500])
    r = _score(bench, _perfect_rows(bench.gt) + [low])
    assert r["AP_6D"] == pytest.approx(1.0) and r["AR_6D"] == pytest.approx(1.0)


def test_pose_validation_catches_common_mistakes(bench):
    test = pose_eval.TestSet(bench.path / "bop")
    bad = [{"scene_id": 9, "im_id": 1, "obj_id": 1, "score": 1, "R": np.eye(3) * 2, "t": np.array([0, 0, 1.0]), "time": 1}]
    msgs = " ".join(validate.check_pose(test, bad))
    assert "not a test image" in msgs and "rotation" in msgs and "metres" in msgs


def _coco_perfect(bench, segm=False):
    gt = json.loads((bench.path / "coco" / "test.json").read_text())
    res = []
    for a in gt["annotations"]:
        if a["iscrowd"]:
            continue
        r = {"image_id": a["image_id"], "category_id": a["category_id"], "bbox": a["bbox"], "score": 0.9}
        if segm:
            rle = mask_utils.merge(mask_utils.frPyObjects(a["segmentation"], 1080, 1920))
            rle["counts"] = rle["counts"].decode()
            r["segmentation"] = rle
        res.append(r)
    return res


@pytest.mark.parametrize("segm", [False, True])
def test_coco_perfect_and_ignore_region(bench, segm):
    res = _coco_perfect(bench, segm)
    t = "segm" if segm else "bbox"
    gtp = bench.path / "coco" / "test.json"
    assert coco_eval.evaluate(gtp, res, t)["AP"] == pytest.approx(1.0)
    # A detection on the unverified (crowd) region is ignored.
    extra = dict(res[0], category_id=3, bbox=[1500, 50, 80, 80], score=0.99)
    if segm:
        rle = mask_utils.encode(np.asfortranarray(np.pad(np.ones((80, 80), np.uint8), ((50, 950), (1500, 340)))))
        rle["counts"] = rle["counts"].decode()
        extra["segmentation"] = rle
    assert coco_eval.evaluate(gtp, res + [extra], t)["AP"] == pytest.approx(1.0)
    assert not validate.check_coco(gtp, res, "segm" if segm else "det")


def test_cli_eval_writes_card_and_verify_detects_edits(bench, tmp_path, capsys):
    pred = tmp_path / "p.csv"
    pred.write_text(csv_rows(_perfect_rows(bench.gt)))
    method = tmp_path / "m.json"
    method.write_text(json.dumps({"name": "Oracle", "input": "rgbd", "training_data": "kitchen-train"}))
    out = tmp_path / "res"
    assert cli.main(["eval", str(bench.path), "pose", str(pred), "--method", str(method), "--out", str(out)]) == 0
    card = json.loads(next(out.glob("*.json")).read_text())
    assert card["metrics"]["AP_6D"] == pytest.approx(1.0)
    assert card["provenance"]["ground_truth_sha256"] == manifest.load(bench.path)["ground_truth_sha256"]
    assert cli.main(["leaderboard", str(next(out.glob("*.json")))]) == 0
    assert cli.main(["verify", str(bench.path)]) == 0
    (bench.path / "bop" / "test_ignore.json").write_text("[]")
    assert cli.main(["verify", str(bench.path)]) == 1


def test_paper_rules_on_card():
    from kitchenbench.report import pose_rules
    assert pose_rules({"time_per_image_s": 0.15}, {"single_model": "yes"})["compliant"]
    assert not pose_rules({"time_per_image_s": 0.5}, {"single_model": "yes"})["meets_5fps"]
    assert not pose_rules({"time_per_image_s": 0.1}, {"single_model": "no"})["compliant"]
    assert not pose_rules({"time_per_image_s": None}, {"single_model": "yes"})["compliant"]


def test_time_not_shown_as_percent():
    from kitchenbench.report import _fmt
    assert _fmt(1.0, "time_per_image_s") == "1.000" and _fmt(0.5, "AP_6D") == "50.0"
