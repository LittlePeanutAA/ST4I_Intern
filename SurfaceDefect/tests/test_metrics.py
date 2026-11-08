import numpy as np

from partinspect import metrics


def test_thresholds_from_val():
    scores = [0.1, 0.2, 0.3, 0.6, 0.8, 0.05]
    labels = [0, 0, 0, 1, 1, 1]
    t = metrics.thresholds_from_val(scores, labels)
    assert t["recall_100"] == 0.05  # lowest NG score: catching it rejects every good part
    assert t["f1"] == 0.6  # F1 0.8: two NG caught, no good part rejected
    assert 0.3 < t["good_max"] < 0.3001  # just above the best good score: no good val part rejected
    assert np.isclose(t["good_3sd"], 0.2 + 3 * 0.1)


def test_image_metrics_per_type():
    scores = [0.1, 0.2, 0.9, 0.4, 0.95]
    labels = [0, 0, 1, 1, 1]
    kinds = ["", "", "scratch", "spot", "spot"]
    m = metrics.image_metrics(scores, labels, kinds, {"t": 0.5})
    assert m["image_AUROC"] == 1.0
    assert m["recall@t"] == 2 / 3 and m["FPR@t"] == 0.0
    assert m["recall@t/scratch"] == 1.0 and m["recall@t/spot"] == 0.5


def test_pixel_metrics_perfect_map():
    masks = np.zeros((2, 32, 32), bool)
    masks[0, 4:10, 4:10] = True
    maps = masks.astype(np.float32) + np.random.default_rng(0).uniform(0, 0.1, masks.shape)
    m = metrics.pixel_metrics(maps, masks, subsample=1)
    assert m["pixel_AUROC"] == 1.0 and m["pixel_AUPRO"] > 0.99
