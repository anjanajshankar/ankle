
"""How well did spline_ik --fit-offsets recover the marker offsets of a synthetic trial?

usage: python compare_offsets.py <prefix>_true_offsets.csv <out>_offsets.csv [--summary FILE --tag TAG]

Prints, per body and overall: RMS of the true offsets, RMS of (estimated - true), and the share of
the offset error that was removed: 1 - err^2 / true^2 (1 = fully recovered, 0 = no better than
leaving the offsets at zero, negative = worse). Only markers present in both files are compared.
"""
import csv
import sys
from collections import defaultdict

import numpy as np


def load(path):
    rows = {}
    for r in csv.DictReader(open(path)):
        rows[r["marker"]] = (r["body"], np.array([float(r["dx_mm"]), float(r["dy_mm"]), float(r["dz_mm"])]))
    return rows


def main():
    args = sys.argv[1:]
    summary = tag = None
    if "--summary" in args:
        i = args.index("--summary"); summary = args[i + 1]; del args[i:i + 2]
    if "--tag" in args:
        i = args.index("--tag"); tag = args[i + 1]; del args[i:i + 2]
    if len(args) != 2:
        print(__doc__); sys.exit(1)
    true, est = load(args[0]), load(args[1])
    common = [m for m in true if m in est]
    if not common:
        print("no markers in common"); sys.exit(1)
    by_body = defaultdict(list)
    for m in common:
        by_body[true[m][0]].append(m)

    def stats(ms):
        t = np.array([true[m][1] for m in ms]); e = np.array([est[m][1] for m in ms]) - t
        rt, re = np.sqrt((t ** 2).sum(1).mean()), np.sqrt((e ** 2).sum(1).mean())
        return rt, re, 1 - re ** 2 / rt ** 2 if rt > 0 else float("nan")

    print(f"{'body':<14}{'n':>3}{'true RMS mm':>13}{'error RMS mm':>14}{'recovered':>11}")
    for body in sorted(by_body):
        rt, re, f = stats(by_body[body])
        print(f"{body:<14}{len(by_body[body]):>3}{rt:>13.2f}{re:>14.2f}{f:>11.2f}")
    rt, re, f = stats(common)
    print(f"{'all':<14}{len(common):>3}{rt:>13.2f}{re:>14.2f}{f:>11.2f}")
    if summary:
        new = not __import__("os").path.exists(summary)
        with open(summary, "a", newline="") as fh:
            w = csv.writer(fh)
            if new: w.writerow(["tag", "markers", "true_rms_mm", "error_rms_mm", "recovered"])
            w.writerow([tag or args[1], len(common), f"{rt:.3f}", f"{re:.3f}", f"{f:.3f}"])


if __name__ == "__main__":
    main()
