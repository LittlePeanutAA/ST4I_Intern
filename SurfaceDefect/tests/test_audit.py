import cv2
import numpy as np

from partinspect.audit import ranges, specimens


def test_specimens_groups_photos_of_the_same_part(tmp_path):
    rng = np.random.default_rng(0)
    parts = [rng.normal(90, 12, (803, 1163)) for _ in range(2)]  # two parts, each with its own surface texture
    for part in parts:  # same moulded shape (photos are aligned on it), different texture
        part[150:650, 150:1010] = 250
    paths = []
    for i, part in enumerate([0, 0, 1, 0, 1]):
        shift = np.float32([[1, 0, rng.integers(-3, 4)], [0, 1, rng.integers(-3, 4)]])
        img = cv2.warpAffine(parts[part], shift, (1163, 803), borderMode=cv2.BORDER_REFLECT)
        img = np.clip(img + rng.normal(0, 2, img.shape), 0, 255).astype(np.uint8)  # sensor noise
        paths.append(tmp_path / f"{i:03d}.png")
        cv2.imwrite(str(paths[-1]), cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    groups, corr = specimens(paths)
    assert sorted(sorted(p.stem for p in g) for g in groups) == [["000", "001", "003"], ["002", "004"]]
    assert corr[0, 1] > 0.8 and abs(corr[0, 2]) < 0.2


def test_ranges():
    assert ranges(["000", "001", "002", "005", "new"]) == "000-002, 005, new"
