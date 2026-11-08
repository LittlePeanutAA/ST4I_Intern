"""Photometric normalisation applied to every image before the model, in training and on the line.

    partinspect-normalize data/prepared --out data/prepared_norm

The photos of this dataset get steadily darker along the capture order (part grey level 84 at the start,
~80 at the end, 78 for the later session): lamp ageing / ambient light, as on a real line. The anomaly models
score that global change like a defect, so before scoring each image is rescaled per channel so that the mean
colour of the part (pixels darker than the white background) equals a fixed reference. Defects are small, so
they barely move the part mean and are kept. Masks need no change (no geometric transform).
"""
import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np

from .synth import foreground

REF_BGR = (138.0, 76.0, 70.5)  # mean part colour (BGR) over the training good images, rounded


def normalize(img_bgr, ref=REF_BGR):
    fg = foreground(img_bgr)
    gain = np.asarray(ref, np.float32) / img_bgr[fg].reshape(-1, 3).mean(0)
    return np.clip(img_bgr.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", help="prepared dataset with manifest.csv")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src, out = Path(a.data), Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    rows = list(csv.DictReader(open(src / "manifest.csv")))
    for r in rows:
        dst = out / r["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), normalize(cv2.imread(str(src / r["path"]))))
        if r["mask"]:
            (out / r["mask"]).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src / r["mask"], out / r["mask"])
    shutil.copyfile(src / "manifest.csv", out / "manifest.csv")
    print(f"{len(rows)} images normalised to part colour {REF_BGR} -> {out}")


if __name__ == "__main__":
    main()
