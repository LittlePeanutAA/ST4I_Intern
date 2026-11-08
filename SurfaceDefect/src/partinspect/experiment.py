"""Fit one anomaly model, export it for CPU deployment, and evaluate *the exported files*.

    partinspect-run padim_r18 --data data/prepared --out results

For every precision the OpenVINO model that would be shipped to the line computer is evaluated:
    fp32, fp16       full / half precision weights
    int8             INT8 weights, FP32 activations (NNCF weight compression)
    int8_ptq         full INT8 post-training quantisation as anomalib does it (calibrated on val, every op quantised)
    int8_cnn         INT8 post-training quantisation of the CNN only, calibrated on training good images; MatMul
                     ops stay FP32 (PaDiM's Mahalanobis distance and PatchCore's nearest-neighbour search: their
                     inputs span a range that 8 bits cannot hold, and int8_ptq collapses every score to one value)
Accuracy and latency therefore describe the same artefact. Scores are the raw anomaly scores
(anomalib post-processing is disabled); the operating thresholds are chosen on the validation split only
(see metrics.thresholds_from_val) and then applied unchanged to the test split.

Latency is measured on the CPU with OpenVINO limited to `--threads` threads (default 2) to imitate a weak
industrial PC: one decoded RGB frame in, score and anomaly map out (resize and normalisation are inside the model).
"""
import argparse
import csv
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import cv2
import numpy as np

from . import metrics

IMAGE_SIZE = (256, 368)  # (h, w): keeps the 1163x803 aspect ratio


def build(name, image_size=IMAGE_SIZE, imagenet_dir="data/imagenette"):
    """Model zoo used in the report. Returns (model, engine kwargs)."""
    from anomalib.models import EfficientAd, Padim, Patchcore

    common = dict(post_processor=False, evaluator=False, visualizer=False)
    if name.startswith("padim"):
        pre = Padim.configure_pre_processor(image_size=image_size)
        n_features = {"padim_r18": None, "padim_r18_d50": 50}[name]
        return Padim(backbone="resnet18", n_features=n_features, pre_processor=pre, **common), {"max_epochs": 1}
    if name.startswith("patchcore"):
        pre = Patchcore.configure_pre_processor(image_size=image_size)
        cfg = {"patchcore_r18": ("resnet18", 0.1), "patchcore_wr50": ("wide_resnet50_2", 0.1),
               "patchcore_wr50_c01": ("wide_resnet50_2", 0.01)}[name]
        return Patchcore(backbone=cfg[0], coreset_sampling_ratio=cfg[1], pre_processor=pre, **common), {"max_epochs": 1}
    if name == "efficientad_s":
        pre = EfficientAd.configure_pre_processor(image_size=image_size)
        return EfficientAd(imagenet_dir=imagenet_dir, model_size="small", pre_processor=pre, **common), {"max_steps": 20000}
    raise ValueError(name)


def datamodule(root, train_batch_size=32):
    from anomalib.data import Folder

    # val/ is used by Lightning's validation loop only; thresholds are computed by us from val scores below.
    return Folder(name="part", root=root, normal_dir="train/good", normal_test_dir="val/good", abnormal_dir="val/NG",
                  mask_dir="val/masks", val_split_mode="same_as_test", train_batch_size=train_batch_size,
                  eval_batch_size=32, num_workers=4)


def rgb(path):
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def evaluate(xml, root, rows):
    """Run the exported OpenVINO model on every val/test image of the manifest."""
    from anomalib.deploy import OpenVINOInferencer

    inf = OpenVINOInferencer(path=xml, device="CPU")
    out = []
    for r in rows:
        p = inf.predict(rgb(Path(root) / r["path"]))
        out.append((float(np.asarray(p.pred_score).ravel()[0]), np.asarray(p.anomaly_map)[0, 0]))
    return out


def latency(xml, images, threads, warmup=5):
    from anomalib.deploy import OpenVINOInferencer

    inf = OpenVINOInferencer(path=xml, device="CPU", config={"INFERENCE_NUM_THREADS": str(threads),
                                                            "PERFORMANCE_HINT": "LATENCY"})
    for im in images[:warmup]:
        inf.predict(im)
    ts = []
    for im in images:
        t = time.perf_counter()
        inf.predict(im)
        ts.append(1000 * (time.perf_counter() - t))
    return {"p50_ms": float(np.percentile(ts, 50)), "p95_ms": float(np.percentile(ts, 95)), "max_ms": float(max(ts))}


def quantize(xml, calib_paths, keep_matmul_fp32):
    """Replace the FP32 model at `xml` by its INT8 post-training quantised version (NNCF)."""
    import nncf
    import openvino as ov
    from anomalib.deploy import OpenVINOInferencer

    model = ov.Core().read_model(xml)
    data = nncf.Dataset(calib_paths, lambda p: OpenVINOInferencer.pre_process(rgb(p)))
    ignored = nncf.IgnoredScope(types=["MatMul"]) if keep_matmul_fp32 else None
    q = nncf.quantize(model, data, subset_size=len(calib_paths), ignored_scope=ignored)
    # the FP32 weights are memory-mapped from model.bin: writing over it in place kills the process (SIGBUS)
    tmp = Path(xml).with_name("model_int8.xml")
    ov.save_model(q, tmp, compress_to_fp16=False)
    del q, model
    for ext in (".bin", ".xml"):
        os.replace(tmp.with_suffix(ext), Path(xml).with_suffix(ext))


def run(name, root, out, precisions=("fp32", "fp16", "int8", "int8_ptq", "int8_cnn"), threads=2, n_latency=60,
        image_size=IMAGE_SIZE):
    from anomalib.deploy import CompressionType, ExportType
    from anomalib.engine import Engine

    tag = name if tuple(image_size) == IMAGE_SIZE else f"{name}_{image_size[0]}x{image_size[1]}"
    root, out = Path(root), Path(out) / tag
    out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(root / "manifest.csv")))
    val = [r for r in rows if r["split"] == "val"]
    test = [r for r in rows if r["split"] == "test"]

    from lightning.pytorch import seed_everything

    seed_everything(0, workers=True)  # PaDiM picks its feature channels at random, PatchCore its coreset start
    model, engine_kw = build(name, image_size)
    dm = datamodule(root, train_batch_size=1 if name.startswith("efficientad") else 32)
    engine = Engine(default_root_dir=str(out / "lightning"), logger=False, enable_progress_bar=False, **engine_kw)
    t = time.time()
    engine.fit(model=model, datamodule=dm)
    fit_s = time.time() - t

    lat_images = [rgb(root / r["path"]) for r in test[:n_latency]]
    h, w = image_size
    test_masks = np.stack([cv2.resize(cv2.imread(str(root / r["mask"]), 0), (w, h), interpolation=cv2.INTER_NEAREST) > 0
                           if r["mask"] else np.zeros((h, w), bool) for r in test])
    results = {"model": tag, "image_size": list(image_size), "fit_seconds": fit_s, "precisions": {}}
    for prec in precisions:
        comp = {"fp32": None, "fp16": CompressionType.FP16, "int8": CompressionType.INT8,  # int8 = weights only
                "int8_ptq": None, "int8_cnn": None}[prec]  # PTQ done below: anomalib's PTQ needs a static input shape
        # batch fixed to 1 (the line inspects one part at a time): with a dynamic batch, torch.export specialises
        # PatchCore's batch axis to the size-2 example input and the exported model rejects single images
        xml = engine.export(model=model, export_type=ExportType.OPENVINO, export_root=str(out / prec),
                            compression_type=comp,
                            onnx_kwargs={"dynamic_axes": {"input": {2: "height", 3: "width"}}})
        if prec == "int8_ptq":
            quantize(xml, [root / r["path"] for r in val], keep_matmul_fp32=False)
        if prec == "int8_cnn":
            quantize(xml, [root / r["path"] for r in rows if r["split"] == "train"], keep_matmul_fp32=True)
        size_mb = sum(f.stat().st_size for f in Path(xml).parent.glob("model.*")) / 1e6
        v, te = evaluate(xml, root, val), evaluate(xml, root, test)
        th = metrics.thresholds_from_val([s for s, _ in v], [r["label"] == "NG" for r in val])
        labels = [r["label"] == "NG" for r in test]
        res = metrics.image_metrics([s for s, _ in te], labels, [r["defect_type"] for r in test], th)
        maps = np.stack([cv2.resize(m, (w, h)) for _, m in te])
        res.update(metrics.pixel_metrics(maps, test_masks))
        res.update({"thresholds": th, "model_mb": size_mb, f"latency_{threads}threads": latency(xml, lat_images, threads),
                    "latency_all_threads": latency(xml, lat_images, os.cpu_count())})
        results["precisions"][prec] = {k: (float(x) if isinstance(x, (np.floating, float)) else x) for k, x in res.items()}
        with open(out / f"scores_{prec}.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["split", "path", "label", "defect_type", "score"])
            for r, (s, _) in zip(val + test, v + te):
                wr.writerow([r["split"], r["path"], r["label"], r["defect_type"], f"{s:.6f}"])
        print(f"{name} {prec}: AUROC {res['image_AUROC']:.3f} | recall/FPR@f1 {res['recall@f1']:.3f}/{res['FPR@f1']:.3f} "
              f"| pixel AUROC {res['pixel_AUROC']:.3f} AUPRO {res['pixel_AUPRO']:.3f} | {size_mb:.1f} MB "
              f"| p50 {res[f'latency_{threads}threads']['p50_ms']:.1f} ms ({threads} threads)", flush=True)
    (out / "results.json").write_text(json.dumps(results, indent=1))
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+")
    ap.add_argument("--data", default="data/prepared")
    ap.add_argument("--out", default="results")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--image-size", type=int, nargs=2, default=list(IMAGE_SIZE), metavar=("H", "W"))
    ap.add_argument("--precisions", nargs="+", default=["fp32", "fp16", "int8", "int8_ptq", "int8_cnn"])
    a = ap.parse_args()
    for m in a.models:
        try:
            run(m, a.data, a.out, a.precisions, a.threads, image_size=tuple(a.image_size))
        except Exception as e:  # one broken model must not stop the rest of the zoo
            print(f"{m} FAILED: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
