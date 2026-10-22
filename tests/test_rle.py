import numpy as np
from pycocotools import mask as mu

from kitchenbench import rle


def test_matches_pycocotools():
    rng = np.random.default_rng(0)
    cases = [np.zeros((5, 7), bool), np.ones((5, 7), bool)]
    for _ in range(30):
        m = rng.uniform(size=(int(rng.integers(1, 60)), int(rng.integers(1, 60)))) > rng.uniform(0.05, 0.95)
        cases.append(m)
    big = np.zeros((1080, 1920), bool)
    big[100:900, 300:1700] = True
    big[400:500, 1000:1100] = False
    cases.append(big)
    for m in cases:
        ref = mu.encode(np.asfortranarray(m.astype(np.uint8)))
        assert rle.encode(m)["counts"] == ref["counts"].decode()
        assert rle.area(m) == int(mu.area(ref))
