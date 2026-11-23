"""Pipeline B's equivalent low-pass cutoff, from the step-2 synthetic trials (for matched GRF filtering).
For every synthetic trial in <step2_dir> (synth_truth.mot + spline_<weights>_l<lambda>_udot.mot): filter the
TRUE coordinates with the 4th-order zero-lag Butterworth used by OpenSim's ID tool at each candidate cutoff,
differentiate twice, compare with the spline's q-ddot (RMS over rotational coordinates, ends trimmed 0.05 s).
The equivalent cutoff = the candidate with the smallest mean difference. Synthetic data only.
usage: python3 tools/equiv_cutoff.py work/<Dataset>/step2 --weights ones --lambda 0.03
"""
import argparse, glob, os, numpy as np
from scipy.signal import butter, filtfilt
def read_mot(path):
    L = open(path).read().split("\n")
    k = next(i for i, l in enumerate(L) if l.strip().lower() == "endheader")
    return L[k + 1].split(), np.loadtxt([l for l in L[k + 2:] if l.strip()], ndmin=2)
ap = argparse.ArgumentParser()
ap.add_argument("step2_dir"); ap.add_argument("--weights", default="ones"); ap.add_argument("--lambda", dest="lam", default="0.03")
ap.add_argument("--cutoffs", default="6,8,10,12,15,20,25,30,40,50")
a = ap.parse_args()
keys = ("ankle", "knee", "hip", "lumbar", "pelvis_tilt", "pelvis_list", "pelvis_obliquity", "pelvis_rotation", "subtalar")
cuts = [float(c) for c in a.cutoffs.split(",")]
per_cut, n = {c: [] for c in cuts}, 0
for d in sorted(glob.glob(os.path.join(a.step2_dir, "*"))):
    truth, sp = os.path.join(d, "synth_truth.mot"), os.path.join(d, f"spline_{a.weights}_l{a.lam}_udot.mot")
    if not (os.path.exists(truth) and os.path.exists(sp)): continue
    tc, tq = read_mot(truth); uc, ud = read_mot(sp)
    t = tq[:, 0]; dt = t[1] - t[0]; keep = (t >= t[0] + 0.05) & (t <= t[-1] - 0.05)
    names = [c for c in tc[1:] if any(k in c for k in keys) and not c.endswith(("_tx", "_ty", "_tz")) and c in uc]
    if not names: continue
    n += 1
    for c in cuts:
        b, aa = butter(4, c / (0.5 / dt))
        errs = []
        for nm in names:
            qdd = np.gradient(np.gradient(filtfilt(b, aa, tq[:, tc.index(nm)]), dt), dt)
            errs.append(np.sqrt(np.mean((qdd - np.interp(t, ud[:, 0], ud[:, uc.index(nm)]))[keep] ** 2)))
        per_cut[c].append(np.mean(errs))
print(f"{n} synthetic trials in {a.step2_dir} (spline {a.weights}, lambda {a.lam})")
for c in cuts: print(f"  {c:5.0f} Hz: mean RMS difference {np.mean(per_cut[c]):8.0f} deg/s^2")
best = min(cuts, key=lambda c: np.mean(per_cut[c]))
print(f"-> equivalent cutoff of Pipeline B: {best:g} Hz" + ("  (at the edge of the candidates: extend --cutoffs)" if best in (cuts[0], cuts[-1]) else ""))
