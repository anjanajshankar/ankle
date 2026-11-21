
#!/usr/bin/env python3
"""
marker_rigidity.py DATASET_DIR [--min-travel 0.5] [--rigid-mm 0.5]

Finds virtual markers (channels rebuilt from a cluster, so they move rigidly with
real markers) in the .trc files written by prep_addbio.py.

For each dynamic trial (whole-body centroid travels at least --min-travel metres;
static and hold trials are skipped, because there everything looks rigid) and each
marker pair, the SD of their distance is computed. Per pair, the median over trials
is kept. A marker whose closest partner has median SD < --rigid-mm (default 0.5 mm)
is flagged "virtual?"; real skin markers on one segment are typically 1-5 mm.

Writes DATASET_DIR/markers_rigidity.csv and prints a summary. Reads only marker
positions (hypothesis-blind).
"""
import argparse, csv, glob, os, sys
import numpy as np


def read_trc(fn):
    L = open(fn).read().split("\n")
    names = [n for n in L[3].split("\t")[2:] if n]
    rows = [l.split("\t") for l in L[6:] if l.strip()]
    X = np.full((len(rows), len(names), 3), np.nan)
    for i, r in enumerate(rows):
        v = r[2:2 + 3 * len(names)]
        for j in range(len(names)):
            c = v[3 * j:3 * j + 3]
            if len(c) == 3 and all(c):
                X[i, j] = [float(x) for x in c]
    return names, X


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--min-travel", type=float, default=0.5)
    ap.add_argument("--rigid-mm", type=float, default=0.5)
    ap.add_argument("--max-frames", type=int, default=120)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.dir, "*", "*.trc")))
    pair_sd = {}       # (m1, m2) -> list of per-trial SD (mm)
    present = {}       # m -> trials present
    used = skipped = 0
    for fn in files:
        names, X = read_trc(fn)
        if len(names) < 4:
            skipped += 1; continue
        cen = np.nanmean(X, axis=1)
        ok = ~np.isnan(cen).any(axis=1)
        if ok.sum() < 20 or np.linalg.norm(np.nanmax(cen[ok], 0) - np.nanmin(cen[ok], 0)) < a.min_travel:
            skipped += 1; continue
        used += 1
        idx = np.linspace(0, len(X) - 1, min(len(X), a.max_frames)).astype(int)
        P = X[idx]
        for j, m in enumerate(names):
            if np.isnan(P[:, j, 0]).mean() < 0.5:
                present[m] = present.get(m, 0) + 1
        D = np.linalg.norm(P[:, :, None, :] - P[:, None, :, :], axis=3)   # frames x m x m
        with np.errstate(all="ignore"):
            n_ok = (~np.isnan(D)).sum(0)
            sd = 1000 * np.nanstd(D, axis=0)
        for j in range(len(names)):
            for q in range(j + 1, len(names)):
                if n_ok[j, q] >= 10:
                    pair_sd.setdefault((names[j], names[q]), []).append(float(sd[j, q]))

    med = {k: float(np.median(v)) for k, v in pair_sd.items()}
    markers = sorted(present)
    out = []
    for m in markers:
        cand = [(s, k[1] if k[0] == m else k[0]) for k, s in med.items() if m in k]
        cand.sort()
        best = cand[0] if cand else (np.nan, "")
        second = cand[1] if len(cand) > 1 else (np.nan, "")
        out.append({"marker": m, "dynamic_trials": present[m],
                    "closest": best[1], "median_sd_mm": round(best[0], 2),
                    "2nd_closest": second[1], "2nd_sd_mm": round(second[0], 2),
                    "flag": "virtual?" if best[0] < a.rigid_mm else ""})
    fn = os.path.join(a.dir, "markers_rigidity.csv")
    with open(fn, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    print(f"{used} dynamic trials used, {skipped} static/short/junk skipped -> {fn}")
    flagged = [r for r in out if r["flag"]]
    print(f"{len(markers)} markers, {len(flagged)} flagged virtual? (< {a.rigid_mm} mm):")
    for r in flagged:
        print(f"  {r['marker']:24s} with {r['closest']:22s} {r['median_sd_mm']:6.2f} mm")
    sds = [r["median_sd_mm"] for r in out if not r["flag"] and np.isfinite(r["median_sd_mm"])]
    if sds:
        print(f"others: closest-partner SD median {np.median(sds):.2f} mm (range {min(sds):.2f}-{max(sds):.2f})")


if __name__ == "__main__":
    main()
