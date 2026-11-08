"""Compare every model and precision in a results folder from the saved scores, without re-running anything.

    partinspect-report results --policy good_max --md reports/zoo.md

Image metrics are recomputed from scores_<precision>.csv with the current threshold policies (metrics.py), so a
new policy can be compared on old runs. Pixel metrics, model size and latency come from results.json.
"""
import argparse
import csv
import json
from pathlib import Path

from . import metrics


def rows(results_dir, policy):
    out = []
    for res_file in sorted(Path(results_dir).glob("*/results.json")):
        res = json.loads(res_file.read_text())
        for prec, r in res["precisions"].items():
            sc = list(csv.DictReader(open(res_file.parent / f"scores_{prec}.csv")))
            val = [x for x in sc if x["split"] == "val"]
            test = [x for x in sc if x["split"] == "test"]
            th = metrics.thresholds_from_val([float(x["score"]) for x in val], [x["label"] == "NG" for x in val])
            m = metrics.image_metrics([float(x["score"]) for x in test], [x["label"] == "NG" for x in test],
                                      [x["defect_type"] for x in test], {policy: th[policy]})
            lat = next(v for k, v in r.items() if k.startswith("latency_") and k.endswith("threads") and "all" not in k)
            out.append({"model": res["model"], "precision": prec, **m, "pixel_AUROC": r["pixel_AUROC"],
                        "pixel_AUPRO": r["pixel_AUPRO"], "model_mb": r["model_mb"], "p50_ms": lat["p50_ms"],
                        "p50_ms_all": r["latency_all_threads"]["p50_ms"], "fit_seconds": res["fit_seconds"]})
    return out


def markdown(rows, policy):
    kinds = sorted({k.split("/", 1)[1] for r in rows for k in r if k.startswith(f"recall@{policy}/")})
    head = ["model", "prec", "AUROC", "AP", "recall", "FPR", *kinds, "pix AUROC", "AUPRO", "MB", "p50 ms"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        cells = [r["model"], r["precision"], f"{r['image_AUROC']:.3f}", f"{r['image_AP']:.3f}",
                 f"{r[f'recall@{policy}']:.2f}", f"{r[f'FPR@{policy}']:.2f}",
                 *(f"{r[f'recall@{policy}/{k}']:.2f}" for k in kinds),
                 f"{r['pixel_AUROC']:.3f}", f"{r['pixel_AUPRO']:.3f}", f"{r['model_mb']:.0f}", f"{r['p50_ms']:.0f}"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="?", default="results")
    ap.add_argument("--policy", default="good_max", choices=["f1", "recall_100", "good_max", "good_3sd"])
    ap.add_argument("--md", help="also write the table here")
    ap.add_argument("--json", help="write all rows (every metric) here")
    a = ap.parse_args()
    rs = rows(a.results, a.policy)
    table = f"Threshold policy: `{a.policy}` (chosen on val, applied unchanged to test). p50 latency with 2 threads.\n\n"
    table += markdown(rs, a.policy)
    print(table)
    for path, text in ((a.md, table), (a.json, json.dumps(rs, indent=1, default=float))):
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(text + "\n")


if __name__ == "__main__":
    main()
