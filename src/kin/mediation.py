"""Protocol step 6, part 2, and step 7: the pre-specified mediation analysis and the inference rule.
Definitions: claude/hypothesis_variable_definitions.md (final, 23 Nov 2026); data: hypvars.csv from hypvars.py.

Models (linear mixed models, random intercept per participant, REML; fitted separately per pipeline):
  path a      : M ~ X + dataset + cut angle + approach speed + (1 | participant)
  paths b, c' : Y ~ M + X + dataset + cut angle + approach speed + (1 | participant)
  total c     : Y ~ X + dataset + cut angle + approach speed + (1 | participant)
  indirect    : a * b (a and b from the same trials)
  dataset: fixed effect (dummy); cut angle: categorical (90 deg = reference); speed: m/s.
Test: 95% percentile CI from a cluster bootstrap resampling participants with replacement WITHIN each
dataset (5000 resamples, fixed seed). Passes if the CI excludes 0.
Inference rule (a, b, c, a*b; c' is reported only): SUPPORTED if an effect has the predicted sign (positive) and passes under BOTH pipelines;
PIPELINE-DEPENDENT if it passes under one only, or the two pipelines' estimates have opposite signs;
NOT SUPPORTED otherwise (an effect passing in the opposite direction is reported as such).

usage (from the ankle folder):
  main analysis (Nitschke + Soccer COD, marker-version X in the pelvis plane):
    python3 src/kin/mediation.py assets/Nitschke22/work_final/hypvars.csv work/SoccerCOD/hypvars.csv
  options:
    --x COLUMN       X column (default X_marker; secondary: Xthigh_marker)
    --y COLUMN       Y column (default Y; secondary: Y_ankle)
    --path-a-only    fit path a only (used for the exploratory ACL analysis)
    --x-acl COLUMN   X column for ACL rows (default X_model; ACL has no marker version)
    --boot N         bootstrap resamples (default 5000)   --seed S (default 20261124)
    --by-dataset     also print each dataset's point estimates (secondary; no test)
    --out FILE.csv   write the result table
"""
import csv, sys
import numpy as np
from scipy.optimize import minimize_scalar

CUTS = (45, 135, 180)


# ---------------------------------------------------------------- random-intercept LMM (REML)
def lmm(y, X, g):
    """Fixed effects of y ~ X + (1|g) by REML. X includes the intercept. Returns beta (len = X columns)."""
    keep = np.any(X != X[0], axis=0); keep[0] = True          # drop constant columns (except intercept)
    Xk = X[:, keep]
    groups = [np.where(g == u)[0] for u in np.unique(g)]
    n_i = np.array([len(ix) for ix in groups])
    N, p = Xk.shape

    def transform(lam):
        theta = 1.0 - np.sqrt(1.0 / (1.0 + n_i * lam))
        yt, Xt = y.astype(float).copy(), Xk.astype(float).copy()
        for th, ix in zip(theta, groups):
            yt[ix] -= th * y[ix].mean()
            Xt[ix] -= th * Xk[ix].mean(axis=0)
        return yt, Xt

    def negll(loglam):
        lam = np.exp(loglam)
        yt, Xt = transform(lam)
        beta, *_ = np.linalg.lstsq(Xt, yt, rcond=None)
        rss = float(np.sum((yt - Xt @ beta) ** 2))
        s2 = rss / (N - p)
        sign, logdet = np.linalg.slogdet(Xt.T @ Xt)
        return 0.5 * ((N - p) * np.log(s2) + np.sum(np.log1p(n_i * lam)) + logdet)

    r = minimize_scalar(negll, bounds=(-12, 6), method="bounded", options={"xatol": 1e-4})
    loglam = r.x if r.fun < negll(-12) else -12
    yt, Xt = transform(np.exp(loglam))
    beta, *_ = np.linalg.lstsq(Xt, yt, rcond=None)
    full = np.full(X.shape[1], np.nan); full[keep] = beta
    return full


def design(rows, xcol, with_m):
    """Columns: intercept, X, [M], dataset dummies, cut dummies, speed."""
    ds = sorted({r["dataset"] for r in rows})
    cols = ["const", "X"] + (["M"] if with_m else []) + [f"ds_{d}" for d in ds[1:]] + [f"cut_{c}" for c in CUTS] + ["speed"]
    X = []
    for r in rows:
        v = [1.0, float(r[xcol])] + ([float(r["M"])] if with_m else [])
        v += [1.0 if r["dataset"] == d else 0.0 for d in ds[1:]]
        v += [1.0 if int(r["cut_deg"]) == c else 0.0 for c in CUTS]
        v += [float(r["approach_speed"])]
        X.append(v)
    return np.array(X), cols


def effects(rows, xcol, ycol, path_a_only):
    g = np.array([f"{r['dataset']}|{r['participant']}" for r in rows])
    Xa, ca = design(rows, xcol, False)
    a = lmm(np.array([float(r["M"]) for r in rows]), Xa, g)[ca.index("X")]
    if path_a_only:
        return {"a": a}
    y = np.array([float(r[ycol]) for r in rows])
    Xb, cb = design(rows, xcol, True)
    bb = lmm(y, Xb, g)
    b, cprime = bb[cb.index("M")], bb[cb.index("X")]
    c = lmm(y, Xa, g)[ca.index("X")]
    return {"a": a, "b": b, "c_prime": cprime, "c": c, "ab": a * b}


def bootstrap(rows, xcol, ycol, path_a_only, n_boot, seed):
    rng = np.random.default_rng(seed)
    by = {}
    for r in rows:
        by.setdefault(r["dataset"], {}).setdefault(r["participant"], []).append(r)
    out = []
    for k in range(n_boot):
        sample = []
        for d, parts in by.items():
            ids = list(parts)
            for j, pid in enumerate(rng.choice(len(ids), size=len(ids), replace=True)):
                for r in parts[ids[pid]]:
                    rr = dict(r); rr["participant"] = f"{ids[pid]}#{j}"   # duplicates become separate clusters
                    sample.append(rr)
        out.append(effects(sample, xcol, ycol, path_a_only))
        if (k + 1) % 500 == 0:
            print(f"  bootstrap {k + 1}/{n_boot}", flush=True)
    return out


def verdict(res):
    """res[pipe][eff] = (est, lo, hi). Predicted sign positive for every effect."""
    out = {}
    for e in res["A"]:
        if e == "c_prime":   # direct effect: reported, no prediction and no test in the definitions
            out[e] = "(reported only)"; continue
        passA = res["A"][e][1] > 0 or res["A"][e][2] < 0
        passB = res["B"][e][1] > 0 or res["B"][e][2] < 0
        sA, sB = np.sign(res["A"][e][0]), np.sign(res["B"][e][0])
        if passA and passB and sA == sB:
            out[e] = "SUPPORTED" if sA > 0 else "OPPOSITE DIRECTION (both pipelines)"
        elif passA or passB or sA != sB:
            out[e] = "PIPELINE-DEPENDENT"
        else:
            out[e] = "not supported"
    return out


def main():
    a = sys.argv[1:]
    files = [x for x in a if x.endswith(".csv") and (a.index(x) == 0 or not a[a.index(x) - 1].startswith("--"))]
    opt = lambda k, d: a[a.index(k) + 1] if k in a else d
    xcol, ycol, xacl = opt("--x", "X_marker"), opt("--y", "Y"), opt("--x-acl", "X_model")
    n_boot, seed = int(opt("--boot", "5000")), int(opt("--seed", "20261124"))
    path_a_only = "--path-a-only" in a
    rows = []
    for f in files:
        for r in csv.DictReader(open(f)):
            r = dict(r)
            if r["dataset"] == "ACL": r[xcol] = r.get(xacl, "")
            need = [xcol, "M", "approach_speed"] + ([] if path_a_only else [ycol])
            if all(r.get(k, "") not in ("", "nan") for k in need):
                rows.append(r)
    res, n = {}, {}
    for pipe in ("A", "B"):
        rp = [r for r in rows if r["pipeline"] == pipe]
        n[pipe] = (len(rp), len({(r["dataset"], r["participant"]) for r in rp}))
        print(f"pipeline {pipe}: {n[pipe][0]} trials, {n[pipe][1]} participants "
              f"({', '.join(f'{d} {sum(r['dataset'] == d for r in rp)}' for d in sorted({r['dataset'] for r in rp}))})", flush=True)
        est = effects(rp, xcol, ycol, path_a_only)
        boots = bootstrap(rp, xcol, ycol, path_a_only, n_boot, seed)
        res[pipe] = {}
        for e, v in est.items():
            bs = np.array([b[e] for b in boots]); bs = bs[np.isfinite(bs)]
            res[pipe][e] = (v, np.percentile(bs, 2.5), np.percentile(bs, 97.5))
        if "--by-dataset" in a:
            for d in sorted({r["dataset"] for r in rp}):
                sub = [r for r in rp if r["dataset"] == d]
                try:
                    ed = effects(sub, xcol, ycol, path_a_only)
                    print(f"   {d} alone (point estimates, no test): " + "  ".join(f"{k} {v:+.4f}" for k, v in ed.items()))
                except Exception as ex:
                    print(f"   {d} alone: could not fit ({ex})")
    ver = verdict(res)
    names = {"a": "a   (X -> M)", "b": "b   (M -> Y | X)", "c_prime": "c'  (X -> Y | M)", "c": "c   (X -> Y total)", "ab": "a*b (indirect)"}
    print(f"\nX = {xcol}" + (f" (ACL: {xacl})" if any(r['dataset'] == 'ACL' for r in rows) else "") +
          ("" if path_a_only else f", Y = {ycol}") + f"; {n_boot} bootstrap resamples, seed {seed}")
    print(f"{'effect':20s}{'pipeline A: estimate [95% CI]':>36s}{'pipeline B: estimate [95% CI]':>36s}   verdict")
    table = []
    for e in res["A"]:
        A, B = res["A"][e], res["B"][e]
        fa = f"{A[0]:+.4f} [{A[1]:+.4f}, {A[2]:+.4f}]"; fb = f"{B[0]:+.4f} [{B[1]:+.4f}, {B[2]:+.4f}]"
        print(f"{names[e]:20s}{fa:>36s}{fb:>36s}   {ver[e]}")
        table.append([e, *[f"{v:.6f}" for v in A], *[f"{v:.6f}" for v in B], ver[e]])
    print("units: a in deg of lean per deg of valgus; b in %BW*Ht per deg of lean; c, c' in %BW*Ht per deg of valgus")
    if "--out" in a:
        with open(opt("--out", ""), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["effect", "A_est", "A_lo", "A_hi", "B_est", "B_lo", "B_hi", "verdict"]); w.writerows(table)
        print(f"table -> {opt('--out', '')}")


if __name__ == "__main__":
    main()
