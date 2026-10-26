# -*- coding: utf-8 -*-
"""Evaluate the model against measured activities (DeepSTARR format).

    python evaluate.py --fasta Sequences_Test.fa \
                       --activity Sequences_activity_Test.txt \
                       --name test --output results \
                       --deepstarr-tsv Test_counts_DeepSTARR_predictions.tsv

Reports PCC, Spearman and MSE for Dev and Hk, with and without RC-TTA, plus
a per-class breakdown. Writes <name>_metrics.json, <name>_predictions.tsv
and <name>_report.txt.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deepstarr import (MODEL_DIR, TARGETS, DeepSTARRModel,  # noqa: E402
                       read_fasta, reverse_complement)

COLUMNS = {"Dev": "Dev_log2_enrichment", "Hk": "Hk_log2_enrichment"}


def pcc(a, b):
    a = np.asarray(a, np.float64) - np.mean(a)
    b = np.asarray(b, np.float64) - np.mean(b)
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def ranks(a):
    a = np.asarray(a, np.float64)
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a), np.float64)
    s = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and s[j + 1] == s[i]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j)
        i = j + 1
    return r


def scc(a, b):
    return pcc(ranks(a), ranks(b))


def stats(p, y):
    return {"PCC": pcc(p, y), "SCC": scc(p, y),
            "MSE": float(np.mean((p - y) ** 2)), "n": int(len(y))}


def read_activity(path):
    with open(path, encoding="utf-8") as fh:
        r = csv.DictReader(fh, delimiter="\t")
        missing = [c for c in COLUMNS.values() if c not in (r.fieldnames or [])]
        if missing:
            raise KeyError(f"{path}: missing columns {missing}; found {r.fieldnames}")
        rows = list(r)
    return {t: np.array([float(w[c]) for w in rows]) for t, c in COLUMNS.items()}


def class_of(sid):
    parts = sid.split("_", 4)
    return parts[4] if len(parts) == 5 else "?"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--activity", required=True,
                    help="TSV with Dev_log2_enrichment and Hk_log2_enrichment, "
                         "rows in the same order as the FASTA")
    ap.add_argument("--name", default="eval")
    ap.add_argument("--output", default="results")
    ap.add_argument("--deepstarr-tsv", default=None,
                    help="optional: published DeepSTARR predictions "
                         "(ID, Predictions_dev, Predictions_hk)")
    ap.add_argument("--fit-calibration", action="store_true",
                    help="fit y = a + b*p on THIS set and write model/calibration.json "
                         "(use validation data, never the test set)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--tf32", action="store_true",
                    help="training-time numerics instead of full FP32")
    a = ap.parse_args()

    if a.fit_calibration and "test" in a.name.lower() and not a.force:
        raise SystemExit("Fitting the calibration on the test set would inflate every "
                         "metric computed on it afterwards. Use validation, or --force.")

    out_dir = Path(a.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    ids, seqs = read_fasta(a.fasta)
    y_all = read_activity(a.activity)
    if len(ids) != len(y_all["Dev"]):
        raise SystemExit(f"{len(ids)} sequences but {len(y_all['Dev'])} activity rows.")

    # DeepSTARR files hold both strands: row i+n/2 is the reverse complement of
    # row i. Scoring every row would count each locus twice.
    n = len(seqs)
    h = n // 2
    paired = n % 2 == 0 and h > 0 and all(
        seqs[i + h].upper() == reverse_complement(seqs[i].upper()) for i in range(h))
    if paired:
        dy = max(float(np.abs(y_all[t][:h] - y_all[t][h:]).max()) for t in TARGETS)
        if dy > 1e-5:
            raise SystemExit(f"Second half is the reverse complement of the first, but "
                             f"the labels differ (max {dy:.3e}). Row order is not what "
                             f"this script assumes; the evaluation would be invalid.")
        ids, seqs = ids[:h], seqs[:h]
        y = {t: y_all[t][:h] for t in TARGETS}
        layout = f"both strands in file: {n} rows = {h} loci; scored per locus"
    else:
        y = y_all
        layout = f"one strand per row: {n} sequences"
    print(f"[{a.name}] {layout}")

    model = DeepSTARRModel(device=a.device, batch_size=a.batch_size,
                           full_precision=not a.tf32)
    print(model.info())

    p_raw = model.predict(seqs, rc_tta=False, ids=ids)
    p_tta = model.predict(seqs, rc_tta=True, ids=ids)

    res = {"set": a.name, "layout": layout, "n_loci": len(ids),
           "model": {"name": "HypBaza", "epoch": model.epoch,
                     "val_mean_PCC": model.val_score, "params": model.n_params,
                     "device": str(model.device),
                     "precision": "TF32" if a.tf32 else "FP32"},
           "modes": {}}
    for mode, p in (("no_tta", p_raw), ("rc_tta", p_tta)):
        m = {t: stats(p[t], y[t]) for t in TARGETS}
        m["mean_PCC"] = 0.5 * (m["Dev"]["PCC"] + m["Hk"]["PCC"])
        m["mean_SCC"] = 0.5 * (m["Dev"]["SCC"] + m["Hk"]["SCC"])
        res["modes"][mode] = m

    if a.fit_calibration:
        cal = {"note": "y = a + b * p, least squares on RC-TTA predictions",
               "set": a.name, "n_loci": len(ids), "rc_tta": True}
        for t in TARGETS:
            b, a0 = np.polyfit(p_tta[t], y[t], 1)
            cal[t] = {"a": float(a0), "b": float(b)}
        (MODEL_DIR / "calibration.json").write_text(
            json.dumps(cal, indent=2), encoding="utf-8")
        model.calibration = cal
        print(f"wrote {MODEL_DIR / 'calibration.json'}")

    p_cal = None
    if model.calibration is not None:
        c = model.calibration
        p_cal = {t: c[t]["a"] + c[t]["b"] * p_tta[t] for t in TARGETS}
        res["calibration"] = {
            "fitted_on": c.get("set"), "same_sample": c.get("set") == a.name,
            **{t: {"a": c[t]["a"], "b": c[t]["b"],
                   "MSE_before": res["modes"]["rc_tta"][t]["MSE"],
                   "MSE_after": float(np.mean((p_cal[t] - y[t]) ** 2))}
               for t in TARGETS}}

    classes = np.array([class_of(s) for s in ids])
    if len(set(classes)) > 1:
        res["classes_rc_tta"] = {
            k: {t: stats(p_tta[t][classes == k], y[t][classes == k]) for t in TARGETS}
            for k in sorted(set(classes)) if (classes == k).sum() >= 30}

    if a.deepstarr_tsv:
        with open(a.deepstarr_tsv, encoding="utf-8") as fh:
            table = {w["ID"]: w for w in csv.DictReader(fh, delimiter="\t")}
        if any(s not in table for s in ids):
            print("!! some loci are absent from the DeepSTARR TSV; comparison skipped")
        else:
            yd = {t: np.array([float(table[s][COLUMNS[t]]) for s in ids]) for t in TARGETS}
            dy = max(float(np.abs(yd[t] - y[t]).max()) for t in TARGETS)
            if dy > 1e-4:
                print(f"!! TSV labels differ from evaluation labels (max {dy:.3e});"
                      " comparison skipped")
            else:
                pd_ = {"Dev": np.array([float(table[s]["Predictions_dev"]) for s in ids]),
                       "Hk": np.array([float(table[s]["Predictions_hk"]) for s in ids])}
                m = {t: stats(pd_[t], y[t]) for t in TARGETS}
                m["mean_PCC"] = 0.5 * (m["Dev"]["PCC"] + m["Hk"]["PCC"])
                res["deepstarr_published"] = m

    res["seconds"] = round(time.time() - t0, 1)
    (out_dir / f"{a.name}_metrics.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")

    cols = ["ID", "class", "Dev_obs", "Hk_obs", "Dev_pred", "Hk_pred",
            "Dev_pred_no_tta", "Hk_pred_no_tta"]
    if p_cal is not None:
        cols += ["Dev_pred_calibrated", "Hk_pred_calibrated"]
    with open(out_dir / f"{a.name}_predictions.tsv", "w", encoding="utf-8",
              newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(cols)
        for i, sid in enumerate(ids):
            row = [sid, classes[i], f"{y['Dev'][i]:.6f}", f"{y['Hk'][i]:.6f}",
                   f"{p_tta['Dev'][i]:.6f}", f"{p_tta['Hk'][i]:.6f}",
                   f"{p_raw['Dev'][i]:.6f}", f"{p_raw['Hk'][i]:.6f}"]
            if p_cal is not None:
                row += [f"{p_cal['Dev'][i]:.6f}", f"{p_cal['Hk'][i]:.6f}"]
            w.writerow(row)

    lines = [f"HypBaza evaluation - {a.name}", layout, model.info(), "",
             f"{'mode':<8} {'target':<7} {'PCC':>8} {'SCC':>8} {'MSE':>8}"]
    for mode in ("no_tta", "rc_tta"):
        for t in TARGETS:
            m = res["modes"][mode][t]
            lines.append(f"{mode:<8} {t:<7} {m['PCC']:8.5f} {m['SCC']:8.5f} "
                         f"{m['MSE']:8.4f}")
        lines.append(f"{mode:<8} {'mean':<7} {res['modes'][mode]['mean_PCC']:8.5f} "
                     f"{res['modes'][mode]['mean_SCC']:8.5f}")
    if "calibration" in res:
        c = res["calibration"]
        lines += ["", f"affine calibration (fitted on: {c['fitted_on']}"
                      f"{', SAME SAMPLE - optimistic' if c['same_sample'] else ''})"]
        for t in TARGETS:
            lines.append(f"  {t}: y = {c[t]['a']:+.4f} + {c[t]['b']:.4f}*p | MSE "
                         f"{c[t]['MSE_before']:.4f} -> {c[t]['MSE_after']:.4f}")
    if "classes_rc_tta" in res:
        lines += ["", "RC-TTA by locus class:"]
        for k, m in res["classes_rc_tta"].items():
            lines.append(f"  {k:<22} n={m['Dev']['n']:6d} | Dev PCC {m['Dev']['PCC']:.4f}"
                         f" | Hk PCC {m['Hk']['PCC']:.4f}")
    if "deepstarr_published" in res:
        d = res["deepstarr_published"]
        lines += ["", "Published DeepSTARR on the same loci:",
                  f"  Dev PCC {d['Dev']['PCC']:.5f} | Hk PCC {d['Hk']['PCC']:.5f} "
                  f"| mean {d['mean_PCC']:.5f}"]
    lines += ["", f"{res['seconds']} s"]
    text = "\n".join(lines)
    (out_dir / f"{a.name}_report.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
