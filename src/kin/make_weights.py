
"""Marker weights from pooled IK residuals (the rule in the analysis protocol).

Reads the per-marker rows written by `ik_cost --marker-csv` for many trials, pools them per marker,
and sets  w_m = (sigma_median / sigma_m)^2, capped to [0.1, 10],
where sigma_m is the marker's RMS residual (mm) over all frames of all trials.
Writes a CSV (marker, body, rms_mm, frames, weight) that prep_trials.py --weights reads.

usage: python make_weights.py <marker_residuals.csv> <weights_out.csv> [--min 0.1] [--max 10]
"""
import csv
import math
import sys
from collections import defaultdict


def main():
    args = sys.argv[1:]
    lo, hi = 0.1, 10.0
    pos = []
    i = 0
    while i < len(args):
        if args[i] == "--min": lo = float(args[i + 1]); i += 2
        elif args[i] == "--max": hi = float(args[i + 1]); i += 2
        else: pos.append(args[i]); i += 1
    if len(pos) < 2:
        print(__doc__); sys.exit(1)

    ss, n, body = defaultdict(float), defaultdict(int), {}
    with open(pos[0]) as f:
        for row in csv.reader(f):
            if len(row) < 6:
                continue
            label, trc, marker, b, s, c = row[:6]
            ss[marker] += float(s)
            n[marker] += int(c)
            body[marker] = b
    sigma = {m: math.sqrt(ss[m] / n[m]) for m in ss if n[m] > 0}
    if not sigma:
        raise SystemExit("no residual rows found")
    med = sorted(sigma.values())[len(sigma) // 2]

    with open(pos[1], "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["marker", "body", "rms_mm", "frames", "weight"])
        print(f"{'marker':10s} {'body':12s} {'RMS mm':>8s} {'weight':>7s}")
        for m in sorted(sigma, key=lambda k: sigma[k]):
            weight = min(hi, max(lo, (med / sigma[m]) ** 2))
            w.writerow([m, body[m], f"{sigma[m]:.3f}", n[m], f"{weight:.4f}"])
            print(f"{m:10s} {body[m]:12s} {sigma[m]:8.2f} {weight:7.3f}")
    print(f"\nmedian RMS {med:.2f} mm; weights capped to [{lo}, {hi}]; wrote {pos[1]}")


if __name__ == "__main__":
    main()
