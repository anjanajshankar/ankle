
#!/usr/bin/env python3
"""check_stance.py PREFIX [--plate N]
Which foot is on the loaded plate, and where the C7 (or any) marker gaps fall.
PREFIX is what you gave extract_b3d.py --out (reads PREFIX.trc and PREFIX_grf.mot)."""
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("prefix")
ap.add_argument("--plate", type=int, default=0, help="plate number (default: the one with the highest peak)")
ap.add_argument("--gap-marker", default="C7")
a = ap.parse_args()

lines = open(a.prefix + ".trc").read().split("\n")
hdr = lines[3].split("\t")
col = {n: i for i, n in enumerate(hdr) if n and n not in ("Frame#", "Time")}
rows = [l.split("\t") for l in lines[6:] if l.strip()]
t = np.array([float(r[1]) for r in rows])


def P(r, m):
    v = r[col[m]:col[m] + 3]
    return np.array([float(x) for x in v]) if len(v) == 3 and all(v) else None


mot = np.loadtxt(a.prefix + "_grf.mot", skiprows=7)
nplates = (mot.shape[1] - 1) // 9
F = [np.linalg.norm(mot[:, 1 + 9 * i:4 + 9 * i], axis=1) for i in range(nplates)]
pl = a.plate - 1 if a.plate else int(np.argmax([f.max() for f in F]))
on = np.where(F[pl] > 20)[0]
print(f"plate {pl+1}: loaded {mot[on[0],0]:.3f}-{mot[on[-1],0]:.3f} s, peak {F[pl].max():.0f} N")

for label, k in [("first 20% of stance", on[len(on) // 10]), ("peak", int(np.argmax(F[pl]))),
                 ("last 20% of stance", on[-len(on) // 10 - 1])]:
    cop = mot[k, 4 + 9 * pl:7 + 9 * pl]
    d = []
    for m in ["LCAL", "LMT1", "LMT5", "RCAL", "RMT1", "RMT5"]:
        p = P(rows[k], m)
        d.append(f"{m} {1000*np.linalg.norm(p-cop):.0f}" if p is not None else f"{m} gap")
    print(f"  {label:20s} t={mot[k,0]:.3f}s  CoP distance (mm): " + ", ".join(d))

g = [i for i, r in enumerate(rows) if P(r, a.gap_marker) is None]
if g:
    runs, s = [], g[0]
    for x, y in zip(g, g[1:] + [None]):
        if y != x + 1:
            runs.append((t[s], t[x])); s = y
    st = (mot[on[0], 0], mot[on[-1], 0])
    print(f"{a.gap_marker} gaps ({len(g)} frames): " + ", ".join(f"{u:.3f}-{v:.3f} s" for u, v in runs))
    inside = sum(1 for i in g if st[0] <= t[i] <= st[1])
    print(f"  {inside} of them during the stance on plate {pl+1}")
else:
    print(f"{a.gap_marker}: no gaps")
