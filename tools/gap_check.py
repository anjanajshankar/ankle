"""Marker gaps and jumps during the plant contact vs pelvis residual, all trials (hypothesis-blind).
usage: python3 tools/gap_check.py work/SoccerCOD"""
import csv, os, re, sys
import numpy as np
sys.path.insert(0, "src/kin")
from assign_plates import read_trc
from residual_check import tag_of
w = sys.argv[1]
res = {x["trial"]: float(x["A_force_bw"]) for x in csv.DictReader(open(f"{w}/step3_results.csv"))}
out = []
for r in csv.DictReader(open(f"{w}/trials_grf.csv")):
    tag = tag_of(r)
    if tag not in res: continue
    txt = open(os.path.join(r["dir"], "ik_setup.xml")).read()
    on = [n for n, b in re.findall(r'<IKMarkerTask name="([^"]+)">(.*?)</IKMarkerTask>', txt, re.S)
          if re.search(r"<apply>\s*true", b)]
    names, t, M = read_trc(r["trc"])
    sel = (t >= float(r["ic"])) & (t <= float(r["toe_off"]))
    idx = [names.index(n) for n in on if n in names]
    X = np.stack([M[sel][:, 3 * i:3 * i + 3] for i in idx], axis=1)          # frames x markers x 3
    gap = float(np.isnan(X[..., 0]).mean())
    step = np.linalg.norm(np.diff(X, axis=0), axis=2)                       # mm per frame
    dt = np.median(np.diff(t))
    jump = float(np.nanmax(step)) if np.isfinite(step).any() else np.nan
    worst = on[int(np.nanargmax(np.nanmax(step, axis=0)))] if np.isfinite(step).any() else ""
    out.append((res[tag], gap, jump, jump / dt / 1000, worst, tag))
out.sort(reverse=True)
r_, g_, j_ = (np.array([o[k] for o in out]) for k in range(3))
rk = lambda v: np.argsort(np.argsort(v))
print(f"Spearman residual vs gap fraction {np.corrcoef(rk(r_), rk(g_))[0,1]:.2f}, vs largest marker jump {np.corrcoef(rk(r_), rk(j_))[0,1]:.2f}  (n={len(out)})")
print(f"{'residual':>9s}{'gaps':>7s}{'jump mm':>9s}{'m/s':>7s}  worst marker / trial")
for o in out[:10] + [None] + out[-5:]:
    if o is None: print("   ..."); continue
    print(f"{o[0]:9.3f}{o[1]:7.1%}{o[2]:9.1f}{o[3]:7.1f}  {o[4]}  {o[5]}")
