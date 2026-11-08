"""Check an MVTec-style folder for train/test leakage before trusting any metric.

    partinspect-audit data/MyDataset --json reports/audit.json

Three checks:
1. Exact duplicates: files with the same MD5 in different splits (e.g. a test image copied from train).
2. Edited copies: an anomalous image whose pixels *outside its defect mask* match a training image almost
   exactly, i.e. the defect was painted onto (or removed from) a training photo. Two separate photos of the
   same part never match that closely because of sensor noise, so the reference level is the difference
   between two neighbouring good photos.
3. Physical specimens: how many different parts were photographed. Every moulded part has its own fine surface
   texture (flow marks, scratches); two photos of the same part have almost the same texture, two different
   parts do not. Texture = high-pass filtered grey level of a plain plastic patch, after aligning the photos.
   Photos of the same part in train and test make the test measure "is this the part I memorised" instead of
   "is this a good part", and the false-reject rate on new parts cannot be estimated.
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from .synth import PLAIN, REF_SIZE

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}
SMALL = (291, 201)  # compare at 1/4 size: fast, and still far more precise than sensor noise


def images(folder):
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMG_EXT)


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def exact_duplicates(root):
    """Groups of identical files that live in more than one top-level split (train / test / ...)."""
    groups = defaultdict(list)
    for p in images(root):
        if "ground_truth" not in p.parts:
            groups[md5(p)].append(p.relative_to(root).as_posix())
    return [sorted(g) for g in groups.values() if len({x.split("/")[0] for x in g}) > 1]


def load_small(path, gray=False):
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
    return cv2.resize(im, SMALL, interpolation=cv2.INTER_AREA).astype(np.float32)


def edited_copies(train_dir, ng_dir, mask_dir, tol=None):
    """For each anomalous image, the closest training image measured outside the (dilated) defect mask."""
    train = images(train_dir)
    bank = np.stack([load_small(p) for p in train])
    goods = bank[: min(len(bank), 30)]
    noise = float(np.mean([np.abs(goods[i] - goods[i + 1]).mean() for i in range(len(goods) - 1)]))
    tol = tol if tol is not None else noise / 4
    rows = []
    for p in images(ng_dir):
        mask_path = next(Path(mask_dir).glob(p.stem + "*"), None)
        img = load_small(p)
        keep = np.ones(img.shape[:2], bool)
        if mask_path is not None:
            m = load_small(mask_path, gray=True) > 0
            keep = ~cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        err = (np.abs(bank - img) * keep[None, :, :, None]).sum((1, 2, 3)) / (keep.sum() * 3)
        j = int(err.argmin())
        rows.append({"image": p.name, "closest_train": train[j].name, "diff_outside_mask": round(float(err[j]), 3),
                     "edited_copy": bool(err[j] < tol)})
    return {"photo_noise_level": round(noise, 3), "threshold": round(tol, 3), "images": rows}


def _texture(gray, ref, zone):
    (dx, dy), _ = cv2.phaseCorrelate(ref, gray)
    g = cv2.warpAffine(gray, np.float32([[1, 0, -dx], [0, 1, -dy]]), gray.shape[::-1], borderMode=cv2.BORDER_REPLICATE)
    x0, y0, x1, y1 = zone
    p = (g - cv2.GaussianBlur(g, (0, 0), 2))[y0:y1, x0:x1]
    return ((p - p.mean()) / (p.std() + 1e-6)).ravel()


def specimens(paths, zone=None, same=0.5):
    """Group photos by physical part. Returns (groups of paths, correlation matrix).

    zone: x0, y0, x1, y1 of a plain plastic patch in full-resolution pixels (default: synth.PLAIN scaled).
    same: texture correlation above which two photos show the same part. Measured on this dataset: same part
    0.89-0.99, different parts 0.07-0.21, two different patches of one photo 0.02.
    """
    grays = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2GRAY).astype(np.float32) for p in paths]
    if zone is None:
        sx, sy = grays[0].shape[1] / REF_SIZE[0], grays[0].shape[0] / REF_SIZE[1]
        zone = (int(PLAIN[0] * sx), int(PLAIN[1] * sy), int(PLAIN[2] * sx), int(PLAIN[3] * sy))
    t = np.stack([_texture(g, grays[len(grays) // 2], zone) for g in grays])
    corr = t @ t.T / t.shape[1]
    label = -np.ones(len(paths), int)
    for i in range(len(paths)):  # greedy: each unlabelled photo opens a group with every photo similar to it
        if label[i] < 0:
            label[(corr[i] > same) & (label < 0)] = label.max() + 1
    return [[paths[j] for j in np.flatnonzero(label == k)] for k in range(label.max() + 1)], corr


def ranges(names):
    """'000', '001', '002', '005' -> '000-002, 005' (names in capture order)."""
    out, start, prev = [], None, None
    for n in names:
        if start is not None and n.isdigit() and prev.isdigit() and int(n) == int(prev) + 1:
            prev = n
            continue
        if start is not None:
            out.append(start if start == prev else f"{start}-{prev}")
        start = prev = n
    out.append(start if start == prev else f"{start}-{prev}")
    return ", ".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="dataset root with train/good, test/<class>, ground_truth/<class>")
    ap.add_argument("--json", help="write the full report here")
    a = ap.parse_args()
    root = Path(a.root)
    dups = exact_duplicates(root)
    report = {"exact_duplicates_across_splits": dups, "edited_copies": {}}
    print(f"exact duplicates across splits: {len(dups)} groups")
    for g in dups[:5]:
        print("   ", " == ".join(g))
    for cls_dir in sorted(p for p in (root / "test").iterdir() if p.is_dir() and p.name not in ("good", "masks")):
        mask_dir = root / "ground_truth" / cls_dir.name  # MVTec layout; partinspect-split keeps masks in test/masks
        r = edited_copies(root / "train" / "good", cls_dir, mask_dir if mask_dir.exists() else root / "test" / "masks")
        report["edited_copies"][cls_dir.name] = r
        n = sum(x["edited_copy"] for x in r["images"])
        print(f"test/{cls_dir.name}: {n}/{len(r['images'])} images are edited copies of a training image "
              f"(difference outside the mask < {r['threshold']}; two separate photos differ by ~{r['photo_noise_level']})")
    goods = [p for p in images(root) if p.parent.name == "good"]
    groups, _ = specimens(goods)
    report["specimens"] = [{"n_photos": len(g), "photos": [p.relative_to(root).as_posix() for p in g]}
                           for g in groups]
    print(f"good photos show {len(groups)} physical part(s):")
    for i, g in enumerate(groups):
        by_dir = {}
        for p in g:
            by_dir.setdefault(p.parent.relative_to(root).as_posix(), []).append(p.stem)
        print(f"    part {i}: {len(g)} photos: " + "; ".join(f"{d}/{ranges(n)}" for d, n in by_dir.items()))
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
