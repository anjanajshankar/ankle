
"""Summarize protocol step 2 (synthetic test) and apply its rules.

Reads <out>/summary.csv written by step2_synthetic.py (columns:
trial_seed, weights, label, cutoff, filtered_angles, mean_q, mean_qdd, ankle_q, ankle_qdd)
and averages over trials and seeds.

Rules (analysis protocol, step 2):
  Pipeline A (OpenSim): f_c = cutoff with the lowest mean q-ddot error.
  Pipeline B (spline):  lambda with the lowest mean q-ddot error among those whose mean angle error
                        is not worse than Pipeline A's filtered angles at f_c.
  Weight safeguard:     if both weight sets were run, adopt "learned" only if the mean angle error,
                        averaged over both pipelines at their tuned settings, is not worse than "ones".

usage: python summarize_step2.py <out_dir>/summary.csv
"""
import csv
import sys
from collections import defaultdict


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    acc = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0])
    with open(sys.argv[1]) as f:
        for row in csv.reader(f):
            if len(row) < 9:
                continue
            tag, weights, label, cutoff, filt, q, qdd, aq, aqdd = row[:9]
            if label.startswith("spline"):
                cutoff, filt = "", "0"           # spline rows repeat for every cutoff; same numbers
            key = (weights, label, cutoff, filt, tag)
            if key in seen:
                continue
            seen.add(key)
            a = acc[(weights, label, cutoff, filt)]
            for k, v in enumerate((q, qdd, aq, aqdd)):
                a[k] += float(v)
            a[4] += 1
    if not acc:
        raise SystemExit("no rows")
    avg = {k: [v[0] / v[4], v[1] / v[4], v[2] / v[4], v[3] / v[4], v[4]] for k, v in acc.items()}

    tuned = {}
    for w in sorted({k[0] for k in avg}):
        print(f"\n=== weights: {w}   (mean over trials x seeds; q deg, q-ddot deg/s^2)")
        print(f"{'method':22s} {'n':>3s} {'mean q':>8s} {'mean qdd':>9s} {'ankle q':>8s} {'ankle qdd':>10s}")
        os_rows = sorted(((k, v) for k, v in avg.items() if k[0] == w and k[1] == "opensim"),
                         key=lambda kv: (float(kv[0][2]), kv[0][3]))
        for k, v in os_rows:
            name = f"OpenSim {float(k[2]):g} Hz" + (" (q filt)" if k[3] == "1" else "")
            print(f"{name:22s} {v[4]:3d} {v[0]:8.3f} {v[1]:9.0f} {v[2]:8.3f} {v[3]:10.0f}")
        sp_rows = sorted(((k, v) for k, v in avg.items() if k[0] == w and k[1].startswith("spline")),
                         key=lambda kv: float(kv[0][1].split("_l")[1]))
        for k, v in sp_rows:
            name = f"spline lambda {k[1].split('_l')[1]}"
            print(f"{name:22s} {v[4]:3d} {v[0]:8.3f} {v[1]:9.0f} {v[2]:8.3f} {v[3]:10.0f}")

        filt_rows = [(k, v) for k, v in os_rows if k[3] == "1"]
        if not filt_rows:
            continue
        kA, vA = min(filt_rows, key=lambda kv: kv[1][1])
        fc, qA = float(kA[2]), vA[0]
        ok = [(k, v) for k, v in sp_rows if v[0] <= qA]
        if ok:
            kB, vB = min(ok, key=lambda kv: kv[1][1])
            note = ""
        elif sp_rows:
            kB, vB = min(sp_rows, key=lambda kv: kv[1][0])
            note = "  (no lambda met the angle condition; lowest angle error shown)"
        else:
            continue
        lam = kB[1].split("_l")[1]
        tuned[w] = (fc, lam, qA, vA[1], vB[0], vB[1])
        print(f"-> Pipeline A: f_c = {fc:g} Hz (mean q {qA:.3f} filtered, mean qdd {vA[1]:.0f})")
        print(f"-> Pipeline B: lambda = {lam} (mean q {vB[0]:.3f}, mean qdd {vB[1]:.0f}){note}")

    if "learned" in tuned and "ones" in tuned:
        L, O = tuned["learned"], tuned["ones"]
        ql, qo = (L[2] + L[4]) / 2, (O[2] + O[4]) / 2
        print(f"\nweight safeguard: mean angle error (both pipelines, tuned) learned {ql:.3f} vs ones {qo:.3f}"
              f" -> use {'learned' if ql <= qo else 'ones'} weights")


seen = set()

if __name__ == "__main__":
    main()
