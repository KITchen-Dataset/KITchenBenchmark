import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
from kitchenbench.symmetry import propose  # noqa: E402


def test_cylinder_is_continuous_and_box_has_flips():
    cyl = propose(trimesh.creation.cylinder(radius=40, height=120, sections=128))
    assert len(cyl["symmetries_continuous"]) == 1
    assert abs(abs(cyl["symmetries_continuous"][0]["axis"][2]) - 1) < 1e-3
    box = propose(trimesh.creation.box(extents=(60, 90, 150)))
    assert not box["symmetries_continuous"]
    assert len(box["symmetries_discrete"]) == 3


def test_asymmetric_shape_has_none():
    m = trimesh.creation.box(extents=(60, 90, 150))
    m = trimesh.util.concatenate([m, trimesh.creation.box(extents=(20, 20, 20)).apply_translation((40, 50, 60))])
    r = propose(m)
    assert not r["symmetries_continuous"] and not r["symmetries_discrete"]
