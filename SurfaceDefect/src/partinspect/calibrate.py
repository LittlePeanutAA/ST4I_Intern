"""Choose the reject threshold from out-of-fold scores of good parts (blocked k-fold), then test the final model.

    partinspect-calibrate padim_r18 --image-size 512 736 --out results_cal

Why: a threshold must sit above the scores of *unseen* good parts. With ~230 good photos, a single held-out block
of 30 is too small and too dependent on which block is picked (score level drifts along the capture order, and
at high resolution a few dusty parts score far above the rest), so the threshold jumped between "rejects most
good parts" and "misses most defects" depending on the block.

Procedure:
1. pool = every good image that is not in test (train + val good of the manifest), in capture order;
2. split the pool into K contiguous blocks; for each block fit the model on the other blocks, leaving out GAP
   images on each side of the held-out block (neighbouring shots are near-duplicates), export it in the deployed
   precision and score the held-out block -> one out-of-fold (OOF) score for every good image;
3. thresholds from the OOF good scores (no NG image is used): p99, p99.5, max;
4. fit the final model on the whole pool, export, score the test split and apply the thresholds unchanged.
The final model has seen ~25% more good images than each fold model, so its scores on unseen parts are, if
anything, slightly lower: the thresholds are slightly conservative (fewer false rejects, a little less recall).
"""
import argparse
import csv
import json
import os
import shutil
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

from . import experiment, metrics

GAP = 5


def fit_export(name, train_paths, work, data_root, image_size, precision):
    """Fit `name` on train_paths and export it; returns the OpenVINO xml path."""
    from anomalib.deploy import CompressionType, ExportType
    from anomalib.engine import Engine
    from lightning.pytorch import seed_everything

    work = Path(work)
    if work.exists():
        shutil.rmtree(work)
    for p in train_paths:
        dst = work / "train/good" / Path(p).name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(Path(p).resolve())
    (work / "val").symlink_to((Path(data_root) / "val").resolve())  # only Lightning's validation loop reads it
    seed_everything(0, workers=True)
    model, engine_kw = experiment.build(name, image_size)
    dm = experiment.datamodule(work, train_batch_size=1 if name.startswith("efficientad") else 32)
    engine = Engine(default_root_dir=str(work / "lightning"), logger=False, enable_progress_bar=False, **engine_kw)
    engine.fit(model=model, datamodule=dm)
    comp = {"fp32": None, "fp16": CompressionType.FP16, "int8": CompressionType.INT8}[precision]
    return engine.export(model=model, export_type=ExportType.OPENVINO, export_root=str(work / "export"),
                         compression_type=comp, onnx_kwargs={"dynamic_axes": {"input": {2: "height", 3: "width"}}})


def blocks(n, k):
    edges = np.linspace(0, n, k + 1).round().astype(int)
    return [range(edges[i], edges[i + 1]) for i in range(k)]


def run(name, data_root, out, folds=5, image_size=experiment.IMAGE_SIZE, precision="fp16"):
    data_root = Path(data_root)
    rows = list(csv.DictReader(open(data_root / "manifest.csv")))
    pool = sorted((r for r in rows if r["label"] == "good" and r["split"] in ("train", "val")),
                  key=lambda r: Path(r["path"]).name)
    test = [r for r in rows if r["split"] == "test"]
    paths = [data_root / r["path"] for r in pool]
    tag = f"{name}_{image_size[0]}x{image_size[1]}_{precision}"
    out = Path(out) / tag
    out.mkdir(parents=True, exist_ok=True)

    oof = np.full(len(pool), np.nan)
    fold_info = []
    for i, blk in enumerate(blocks(len(pool), folds)):
        lo, hi = blk.start, blk.stop
        train = [p for j, p in enumerate(paths) if j < lo - GAP or j >= hi + GAP]
        t = time.time()
        xml = fit_export(name, train, out / "work", data_root, image_size, precision)
        oof[lo:hi] = [s for s, _ in experiment.evaluate(xml, data_root, pool[lo:hi])]
        fold_info.append({"fold": i, "held_out": [paths[lo].name, paths[hi - 1].name], "n_train": len(train),
                          "seconds": round(time.time() - t, 1), "oof_median": float(np.median(oof[lo:hi])),
                          "oof_max": float(oof[lo:hi].max())})
        print(f"{tag} fold {i}: held out {paths[lo].name}-{paths[hi - 1].name}, trained on {len(train)}, "
              f"OOF good median {np.median(oof[lo:hi]):.1f} max {oof[lo:hi].max():.1f}", flush=True)

    th = {"oof_p99": float(np.quantile(oof, 0.99)), "oof_p995": float(np.quantile(oof, 0.995)),
          "oof_max": float(np.nextafter(oof.max(), np.inf))}
    xml = fit_export(name, paths, out / "work", data_root, image_size, precision)
    te = [s for s, _ in experiment.evaluate(xml, data_root, test)]
    res = metrics.image_metrics(te, [r["label"] == "NG" for r in test], [r["defect_type"] for r in test], th)
    good_new = [s for s, r in zip(te, test) if r["label"] == "good" and "new_" in r["path"]]
    good_old = [s for s, r in zip(te, test) if r["label"] == "good" and "new_" not in r["path"]]
    for k, t in th.items():  # the 5 photos of the later session are reported apart: they drive most false rejects
        res[f"FPR@{k}/same_session"] = float(np.mean(np.array(good_old) >= t))
        res[f"FPR@{k}/new_session"] = float(np.mean(np.array(good_new) >= t))
    summary = {"model": name, "image_size": list(image_size), "precision": precision, "folds": fold_info,
               "thresholds": th, "test": {k: float(v) for k, v in res.items()}}
    (out / "calibration.json").write_text(json.dumps(summary, indent=1))
    with open(out / "scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["set", "path", "label", "defect_type", "score"])
        w.writerows(["oof", r["path"], "good", "", f"{s:.6f}"] for r, s in zip(pool, oof))
        w.writerows(["test", r["path"], r["label"], r["defect_type"], f"{s:.6f}"] for r, s in zip(test, te))
    shutil.rmtree(out / "work")
    for k, t in th.items():
        print(f"{tag} {k}={t:.2f}: recall {res[f'recall@{k}']:.3f} | FPR {res[f'FPR@{k}']:.3f} "
              f"(same session {res[f'FPR@{k}/same_session']:.3f}, new session {res[f'FPR@{k}/new_session']:.1f}) | "
              + " ".join(f"{d}={res[f'recall@{k}/{d}']:.2f}" for d in sorted({r['defect_type'] for r in test if r['defect_type']})),
              flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+")
    ap.add_argument("--data", default="data/prepared")
    ap.add_argument("--out", default="results_cal")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--image-size", type=int, nargs=2, default=list(experiment.IMAGE_SIZE), metavar=("H", "W"))
    ap.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "int8"])
    a = ap.parse_args()
    for m in a.models:
        try:
            run(m, a.data, a.out, a.folds, tuple(a.image_size), a.precision)
        except Exception as e:  # one broken model must not stop the others
            print(f"{m} FAILED: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
