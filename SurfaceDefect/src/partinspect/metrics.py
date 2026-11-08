"""Metrics for an inspection station, computed the same way for every model and every export format.

Image level (is the part NG?):
    AUROC, AP (threshold-free), and at a fixed operating threshold chosen on the *validation* split only:
    recall (share of NG parts caught), FPR (share of good parts rejected), recall per defect type.
Pixel level (where is the defect?):
    pixel AUROC and AUPRO (area under the per-region-overlap curve up to FPR 0.3, normalised to 1),
    the localisation metric of the MVTec AD benchmark.
"""
import cv2
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def thresholds_from_val(scores, labels):
    """Operating points chosen on validation data.

    f1:          threshold with the best F1 on validation (depends on the NG share of val, which is not the NG
                 share of a real line: with few good val images it sits too low and rejects many good parts)
    recall_100:  highest threshold that still flags every validation NG part (a line that must not let NG pass)
    good_max:    highest score of a good validation part (no good val part rejected); needs only good images,
                 so it is the policy that can be set up on a line before any NG part has been collected
    good_3sd:    mean + 3 std of good validation scores
    """
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    good = scores[~labels]
    cand = np.unique(scores)
    best, best_t = -1.0, cand[0]
    for t in cand:
        pred = scores >= t
        tp, fp, fn = (pred & labels).sum(), (pred & ~labels).sum(), (~pred & labels).sum()
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best:
            best, best_t = f1, t
    return {"f1": float(best_t), "recall_100": float(scores[labels].min()),
            "good_max": float(np.nextafter(good.max(), np.inf)), "good_3sd": float(good.mean() + 3 * good.std(ddof=1))}


def image_metrics(scores, labels, defect_types, thresholds):
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    defect_types = np.asarray(defect_types, object)
    out = {"image_AUROC": roc_auc_score(labels, scores), "image_AP": average_precision_score(labels, scores)}
    for name, t in thresholds.items():
        pred = scores >= t
        out[f"recall@{name}"] = pred[labels].mean()
        out[f"FPR@{name}"] = pred[~labels].mean()
        for kind in sorted(set(defect_types[labels])):
            sel = labels & (defect_types == kind)
            out[f"recall@{name}/{kind}"] = pred[sel].mean()
    return out


def aupro(maps, masks, max_fpr=0.3, steps=200):
    """Per-region overlap: for each connected defect region, the share of it above threshold; averaged over
    regions and integrated against the pixel FPR on good pixels, up to max_fpr."""
    maps = np.asarray(maps, np.float32)
    masks = np.asarray(masks, bool)
    regions = []
    for m in masks:
        n, lab = cv2.connectedComponents(m.astype(np.uint8))
        regions.append([(lab == i) for i in range(1, n)])
    good = maps[~masks]
    ts = np.quantile(good, np.linspace(1, 1 - max_fpr, steps)) if good.size else np.linspace(maps.max(), maps.min(), steps)
    fprs, pros = [], []
    for t in ts:
        fprs.append((good >= t).mean())
        ov = [(mp[r] >= t).mean() for mp, rs in zip(maps, regions) for r in rs]
        pros.append(np.mean(ov) if ov else 0.0)
    fprs, pros = np.array(fprs), np.array(pros)
    keep = fprs <= max_fpr
    return float(np.trapezoid(pros[keep], fprs[keep]) / max_fpr) if keep.sum() > 1 else float("nan")


def pixel_metrics(maps, masks, subsample=4):
    """maps, masks: (N, H, W). Pixel AUROC is computed on every `subsample`-th pixel to keep memory small."""
    maps, masks = np.asarray(maps, np.float32), np.asarray(masks, bool)
    m, k = maps[:, ::subsample, ::subsample].ravel(), masks[:, ::subsample, ::subsample].ravel()
    return {"pixel_AUROC": roc_auc_score(k, m), "pixel_AUPRO": aupro(maps, masks)}
