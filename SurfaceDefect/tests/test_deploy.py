import json

import cv2
import numpy as np

from partinspect.deploy import DriftMonitor, Station, watch


def fake_infer(rgb):
    """Score = mean red level of the image, so tests can choose each part's score."""
    return float(rgb[..., 0].mean()), np.zeros((8, 8), np.float32)


def test_station_logs_signals_and_saves_ng_overlays(tmp_path):
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    for name, red in [("a.png", 40), ("b.png", 90), ("c.png", 45)]:
        img = np.zeros((32, 32, 3), np.uint8)
        img[..., 2] = red  # BGR: channel 2 is red
        cv2.imwrite(str(incoming / name), img)
    cfg = {"threshold": 60, "drift": {"ref_median": 42, "ref_spread": 3, "window": 2}}
    station = Station(cfg, tmp_path / "out", infer=fake_infer)
    watch(station, incoming, once=True)
    rows = (tmp_path / "out/inspections.csv").read_text().splitlines()[1:]
    assert [r.split(",")[3] for r in rows] == ["OK", "NG", "OK"]
    assert (tmp_path / "out/ng/b_ng.jpg").exists() and not (tmp_path / "out/ng/a_ng.jpg").exists()
    assert (tmp_path / "out/signals.log").read_text().count("REJECT") == 1


def test_drift_monitor_alarms():
    m = DriftMonitor(ref_median=50, ref_spread=4, window=5, max_shift=3, max_ng_rate=0.4)
    assert all(m.update(51, True) == [] for _ in range(5))  # in-control line
    alarms = [m.update(70, True) for _ in range(5)][-1]  # lighting changed: good parts now score ~70
    assert any("drift" in a for a in alarms)
    m2 = DriftMonitor(ref_median=50, ref_spread=4, window=5, max_ng_rate=0.4)
    out = [m2.update(s, ok) for s, ok in [(50, True), (90, False), (91, False), (92, False), (51, True)]]
    assert any("NG rate" in a for a in out[-1])


def test_example_config_is_valid():
    cfg = json.load(open("deploy/line_example.json"))
    assert cfg["threshold"] > cfg["drift"]["ref_median"]
