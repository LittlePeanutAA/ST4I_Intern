"""Build a leak-free train / val / test split of the part dataset, plus reproducible synthetic defects.

    partinspect-split data/MyDataset --out data/prepared

Why the original folders cannot be used as they are (see `partinspect-audit` and reports/audit.json):
- test/good/000-054 are byte copies of train/good/000-054;
- every test/NG/<i> is train/good/<i> with the defect painted in.
So train/good/000-059 never go into training. They become test "good" images, next to the 5 genuinely new
photos test/good/055-059, and the 60 painted NG images stay as a test set of *edited* defects.

The photos are numbered in capture order and neighbouring shots are almost identical, so held-out good images
must be a contiguous block with a gap of GAP images to the training block, never a random sample: a randomly
drawn val image has its two neighbours in training, scores far lower than an unseen part, and the threshold
chosen on it rejects most good test parts (seen at 512x736: val good 13-33 vs test good 20-58).
    000-059  test good (sources of the edited NG)     GAP  train     GAP  val good (last N_VAL_GOOD)

Layout written (symlinks to the original files; generated images are real files):
    prepared/train/good/                  normal images used for fitting
    prepared/val/{good,NG}/ + val/masks/  threshold selection only (synthetic NG on held-out good images)
    prepared/test/{good,NG}/ + test/masks/
    prepared/manifest.csv                 one row per image: split, label, defect type, source, mask
"""
import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np

from . import synth

N_VAL_GOOD = 30
GAP = 5  # images left out between blocks
VAL_SYNTH_PER_TYPE = 5
TEST_SYNTH_PER_TYPE = 20


def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(Path(src).resolve())


def prepare(root, out, seed=0):
    root, out = Path(root), Path(out)
    if out.exists():
        shutil.rmtree(out)
    rng = np.random.default_rng(seed)
    train_all = sorted((root / "train" / "good").glob("*.png"))
    edited_sources = {p.name for p in sorted((root / "test" / "NG").glob("*.png"))}  # NG <i> was made from train <i>
    unused = [p for p in train_all if p.name not in edited_sources]
    val_good = unused[-N_VAL_GOOD:]
    train = unused[GAP:-(N_VAL_GOOD + GAP)]
    test_good = [p for p in train_all if p.name in edited_sources]
    new_good = [p for p in sorted((root / "test" / "good").glob("*.png"))
                if p.read_bytes() not in {q.read_bytes() for q in train_all}]

    rows = []

    def add(split, label, path, kind, source, mask=""):
        rows.append({"split": split, "label": label, "path": Path(path).relative_to(out).as_posix(),
                     "defect_type": kind, "source": source, "mask": Path(mask).relative_to(out).as_posix() if mask else ""})

    for p in train:
        link(p, out / "train/good" / p.name)
        add("train", "good", out / "train/good" / p.name, "", f"train/good/{p.name}")
    for p in val_good:
        link(p, out / "val/good" / p.name)
        add("val", "good", out / "val/good" / p.name, "", f"train/good/{p.name}")
    for p in test_good:
        link(p, out / "test/good" / f"orig_{p.name}")
        add("test", "good", out / "test/good" / f"orig_{p.name}", "", f"train/good/{p.name} (source of an edited NG)")
    for p in new_good:
        link(p, out / "test/good" / f"new_{p.name}")
        add("test", "good", out / "test/good" / f"new_{p.name}", "", f"test/good/{p.name} (separate photo)")
    for p in sorted((root / "test" / "NG").glob("*.png")):
        m = next((root / "ground_truth" / "NG").glob(p.stem + "*"))
        kind = "edited_mark" if int(p.stem) < 30 else "edited_text"
        link(p, out / "test/NG" / f"edited_{p.name}")
        link(m, out / "test/masks" / f"edited_{p.name}")
        add("test", "NG", out / "test/NG" / f"edited_{p.name}", kind, f"test/NG/{p.name} (photoshopped train/good/{p.name})",
            out / "test/masks" / f"edited_{p.name}")

    def synthesize(split, goods, per_type):
        k = 0
        for kind in synth.TYPES:
            for _ in range(per_type):
                src = goods[k % len(goods)]
                k += 1
                img = cv2.imread(str(src))
                bad, mask = synth.apply(img, kind, rng)
                name = f"synth_{kind}_{k:03d}.png"
                (out / split / "NG").mkdir(parents=True, exist_ok=True)
                (out / split / "masks").mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out / split / "NG" / name), bad)
                cv2.imwrite(str(out / split / "masks" / name), mask.astype(np.uint8) * 255)
                add(split, "NG", out / split / "NG" / name, kind, f"synthetic on {src.name}", out / split / "masks" / name)

    synthesize("val", val_good, VAL_SYNTH_PER_TYPE)
    synthesize("test", test_good + new_good, TEST_SYNTH_PER_TYPE)

    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--out", default="data/prepared")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rows = prepare(a.root, a.out, a.seed)
    from collections import Counter
    for (split, label, kind), n in sorted(Counter((r["split"], r["label"], r["defect_type"]) for r in rows).items()):
        print(f"{split:5s} {label:4s} {kind or '-':14s} {n}")


if __name__ == "__main__":
    main()
