"""Our pose maths must equal the official BOP toolkit's, number for number.

Runs when ``bop_toolkit_lib`` is importable (``pip install
git+https://github.com/thodan/bop_toolkit``); skipped otherwise.
"""
import numpy as np
import pytest

from kitchenbench import pose_metrics as pm

bop = pytest.importorskip("bop_toolkit_lib")
from bop_toolkit_lib import misc, pose_error, pose_matching, score  # noqa: E402


def _rand_pose(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    t = np.array([rng.uniform(-200, 200), rng.uniform(-200, 200), rng.uniform(600, 2000)]).reshape(3, 1)
    return R, t


INFOS = [
    {"diameter": 120.0},
    {"diameter": 90.0, "symmetries_continuous": [{"axis": [0, 0, 1], "offset": [0, 0, 0]}]},
    {"diameter": 150.0, "symmetries_discrete": [[-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]]},
    {"diameter": 70.0, "symmetries_discrete": [[1, 0, 0, 0, 0, -1, 0, 3.0, 0, 0, -1, 0, 0, 0, 0, 1]],
     "symmetries_continuous": [{"axis": [1, 0, 0], "offset": [0, 1.5, 0]}]},
]


@pytest.mark.parametrize("info", INFOS)
def test_symmetries_and_errors_match_bop(info):
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(300, 3)) * 40
    K = np.array([[918.76, 0, 957.86], [0, 918.36, 552.21], [0, 0, 1.0]])
    ours = pm.symmetry_transformations(info, 0.01)
    theirs = misc.get_symmetry_transformations(info, 0.01)
    assert len(ours) == len(theirs)
    for a, b in zip(ours, theirs):
        np.testing.assert_allclose(a["R"], b["R"], atol=1e-12)
        np.testing.assert_allclose(a["t"], b["t"], atol=1e-9)
    for _ in range(5):
        Re, te = _rand_pose(rng)
        Rg, tg = _rand_pose(rng)
        assert pm.mssd(Re, te, Rg, tg, pts, ours) == pytest.approx(
            pose_error.mssd(Re, te, Rg, tg, pts, theirs), rel=1e-12)
        assert pm.mspd(Re, te, Rg, tg, K, pts, ours) == pytest.approx(
            pose_error.mspd(Re, te, Rg, tg, K, pts, theirs), rel=1e-9)


def test_ap_and_matching_match_bop():
    rng = np.random.default_rng(1)
    for _ in range(50):
        n = int(rng.integers(1, 30))
        rec = np.sort(rng.uniform(0, 1, n))
        pre = rng.uniform(0, 1, n)
        assert pm.calc_ap(rec, pre) == pytest.approx(score.calc_ap(rec, pre), abs=1e-12)
    errs = [{"est_id": i, "score": float(rng.uniform()), "gt_visib_fracts": {0: 1.0, 1: 0.5, 2: 1.0},
             "errors": {g: [float(rng.uniform(0, 0.3))] for g in range(3)}} for i in range(6)]
    theirs = pose_matching.match_poses(errs, [0.2], 0, [True, False, True])
    ours = pm.match_poses([{**e, "errors": {g: v[0] for g, v in e["errors"].items()}} for e in errs],
                          0.2, 0, [True, False, True])
    assert [(m["est_id"], m["gt_id"]) for m in ours] == [(m["est_id"], m["gt_id"]) for m in theirs]
