"""Inspection station prototype: watch a folder for camera images, judge each part OK / NG, log, signal the line.

    partinspect-serve deploy/line_example.json --watch incoming/ --out runs/line1
    partinspect-serve deploy/line_example.json --watch data/prepared/test/NG --out runs/demo --once

Runs on a weak PC with only `openvino` and `opencv` installed (no torch / anomalib): the exported OpenVINO model
contains resize, normalisation, the backbone and the anomaly scoring, so one decoded RGB frame goes in and the
anomaly score and anomaly map come out.

Per image:
    score >= threshold -> NG: signal "reject", save an overlay of the anomaly map in <out>/ng/
    otherwise          -> OK: signal "pass"
    one CSV row in <out>/inspections.csv: time, file, score, decision, latency
Drift monitor: the median score of the last `window` parts judged OK is compared with the median of the good
parts the threshold was set on. Lighting, camera position or a new material batch shift that median long before
parts are wrongly rejected; past `max_shift` (in units of the reference spread) the station raises an alarm, as
it does when the NG rate of the window exceeds `max_ng_rate`. The alarm is the cue to collect good parts and
recalibrate (README, "Recalibration").

The camera trigger and the reject actuator are stubs (`folder watch` and `LineSignal`): on a real line they are
the camera SDK callback and a digital output / PLC write (e.g. Modbus coil).
"""
import argparse
import csv
import json
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


class Inspector:
    """OpenVINO model + threshold. `infer` can be replaced (tests): rgb uint8 HxWx3 -> (score, map HxW)."""

    def __init__(self, model_xml=None, threshold=0.0, threads=2, infer=None):
        self.threshold = float(threshold)
        if infer is None:
            import openvino as ov

            core = ov.Core()
            compiled = core.compile_model(core.read_model(model_xml), "CPU",
                                          {"INFERENCE_NUM_THREADS": int(threads), "PERFORMANCE_HINT": "LATENCY"})
            request = compiled.create_infer_request()

            def infer(rgb):
                x = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
                out = request.infer({"input": x})
                return float(out["pred_score"].ravel()[0]), np.asarray(out["anomaly_map"])[0, 0]
        self.infer = infer

    def inspect(self, bgr):
        t = time.perf_counter()
        score, amap = self.infer(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        return {"score": score, "ok": score < self.threshold, "map": amap,
                "latency_ms": 1000 * (time.perf_counter() - t)}


class DriftMonitor:
    def __init__(self, ref_median, ref_spread, window=50, max_shift=3.0, max_ng_rate=0.2):
        self.ref_median, self.ref_spread = float(ref_median), float(ref_spread)
        self.max_shift, self.max_ng_rate = max_shift, max_ng_rate
        self.ok_scores, self.decisions = deque(maxlen=window), deque(maxlen=window)

    def update(self, score, ok):
        """Returns a list of alarm messages (empty while everything is normal)."""
        self.decisions.append(ok)
        if ok:
            self.ok_scores.append(score)
        alarms = []
        if len(self.ok_scores) == self.ok_scores.maxlen:
            shift = (np.median(self.ok_scores) - self.ref_median) / self.ref_spread
            if abs(shift) > self.max_shift:
                alarms.append(f"score drift: median of good parts moved {shift:+.1f} reference spreads")
        if len(self.decisions) == self.decisions.maxlen:
            ng_rate = 1 - np.mean(self.decisions)
            if ng_rate > self.max_ng_rate:
                alarms.append(f"NG rate {ng_rate:.0%} over the last {len(self.decisions)} parts")
        return alarms


class LineSignal:
    """Stub for the reject actuator / PLC. Replace `send` with a digital output or a Modbus write."""

    def __init__(self, log_path):
        self.log_path = Path(log_path)

    def send(self, ok, part_id):
        with open(self.log_path, "a") as f:
            f.write(f"{datetime.now().isoformat(timespec='milliseconds')} {part_id} {'PASS' if ok else 'REJECT'}\n")


def overlay(bgr, amap, threshold):
    """Anomaly map as a heat map over the photo; red = pixel score at or above the image threshold."""
    m = cv2.resize(amap.astype(np.float32), (bgr.shape[1], bgr.shape[0]))
    heat = cv2.applyColorMap(np.clip(255 * m / max(threshold, 1e-6), 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(bgr, 0.6, heat, 0.4, 0)


class Station:
    def __init__(self, config, out, infer=None):
        self.cfg = config
        self.out = Path(out)
        (self.out / "ng").mkdir(parents=True, exist_ok=True)
        self.inspector = Inspector(config.get("model"), config["threshold"], config.get("threads", 2), infer)
        d = config["drift"]
        self.drift = DriftMonitor(d["ref_median"], d["ref_spread"], d.get("window", 50), d.get("max_shift", 3.0),
                                  d.get("max_ng_rate", 0.2))
        self.signal = LineSignal(self.out / "signals.log")
        self.log = self.out / "inspections.csv"
        if not self.log.exists():
            self.log.write_text("time,file,score,decision,latency_ms,alarms\n")

    def process(self, path):
        bgr = cv2.imread(str(path))
        if bgr is None:
            return None
        r = self.inspector.inspect(bgr)
        self.signal.send(r["ok"], Path(path).name)
        alarms = self.drift.update(r["score"], r["ok"])
        if not r["ok"]:
            cv2.imwrite(str(self.out / "ng" / (Path(path).stem + "_ng.jpg")), overlay(bgr, r["map"], self.inspector.threshold))
        with open(self.log, "a", newline="") as f:
            csv.writer(f).writerow([datetime.now().isoformat(timespec="milliseconds"), Path(path).name,
                                    f"{r['score']:.3f}", "OK" if r["ok"] else "NG", f"{r['latency_ms']:.1f}",
                                    " | ".join(alarms)])
        for a in alarms:
            print("ALARM:", a, flush=True)
        return r


def watch(station, folder, once=False, poll_s=0.2):
    """Process images appearing in `folder` (a file is taken once its size has stopped changing)."""
    folder, seen, sizes = Path(folder), set(), {}
    while True:
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() not in IMG_EXT or p in seen:
                continue
            size = p.stat().st_size
            if not once and sizes.get(p) != size:  # still being written by the camera
                sizes[p] = size
                continue
            seen.add(p)
            r = station.process(p)
            if r is not None:
                print(f"{p.name}: {'OK' if r['ok'] else 'NG'} score {r['score']:.1f} ({r['latency_ms']:.0f} ms)", flush=True)
        if once:
            return
        time.sleep(poll_s)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="JSON: model, threshold, threads, drift{ref_median, ref_spread, ...}")
    ap.add_argument("--watch", required=True, help="folder the camera writes images to")
    ap.add_argument("--out", default="runs/line")
    ap.add_argument("--once", action="store_true", help="process the images already there and exit")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    if not Path(cfg["model"]).is_absolute():
        cfg["model"] = str(Path(a.config).parent / cfg["model"])
    watch(Station(cfg, a.out), a.watch, once=a.once)


if __name__ == "__main__":
    main()
