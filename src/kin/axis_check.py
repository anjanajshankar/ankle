"""Do the force plates' horizontal axes match the marker frame? Hypothesis-blind check after step 3.

For the same trials as residual_check.py (lowest and highest Pipeline-A residual per participant, or --only),
reruns ID (Pipeline A only) with the horizontal GRF components (vx, vz of both feet) changed:
  as_is, -x, -z, -x-z, swap (x<->z), rot+90 (x,z -> z,-x), rot-90 (x,z -> -z,x)
COP and vertical force unchanged. Prints the pelvis residual RMS / BW over the plant contact for each variant
(and its signed mean x, y, z). A variant far below as_is means the plates' horizontal axes are flipped/rotated.
Reads only pelvis residual columns. Files go to <trial dir>/axistest/.

usage (from the ankle folder):  python3 src/kin/axis_check.py work/SoccerCOD --fc 20 [--only TAG]
"""
import csv, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step3_run import read_table, write_table, ext_loads, id_setup, run, G  # noqa: E402
from residual_check import tag_of, pelvis_force  # noqa: E402

VARIANTS = {"as_is": lambda x, z: (x, z), "-x": lambda x, z: (-x, z), "-z": lambda x, z: (x, -z),
            "-x-z": lambda x, z: (-x, -z), "swap": lambda x, z: (z, x),
            "rot+90": lambda x, z: (z, -x), "rot-90": lambda x, z: (-z, x)}


def main():
    a = sys.argv[1:]
    if len(a) < 3 or a[1] != "--fc":
        print(__doc__); sys.exit(1)
    work, fc = os.path.abspath(a[0]), float(a[2])
    only = a[a.index("--only") + 1] if "--only" in a else None
    rows = {tag_of(r): r for r in csv.DictReader(open(os.path.join(work, "trials_grf.csv")))}
    if only:
        chosen = [only]
    else:
        by_p = {}
        for x in csv.DictReader(open(os.path.join(work, "step3_results.csv"))):
            if x["trial"] in rows:
                by_p.setdefault(rows[x["trial"]]["participant"], []).append(x)
        chosen = []
        for p, xs in sorted(by_p.items()):
            xs = sorted(xs, key=lambda x: float(x["A_force_bw"]))
            chosen += [xs[0]["trial"]] + ([xs[-1]["trial"]] if len(xs) > 1 else [])
    out = []
    print(f"{'trial':45s}" + "".join(f"{v:>8s}" for v in VARIANTS))
    for tag in chosen:
        r = rows[tag]; d = r["dir"]
        bw = float(r["mass_kg"]) * G
        ic, to, t0, t1 = float(r["ic"]), float(r["toe_off"]), float(r["t0"]), float(r["t1"])
        s0, s1 = max(ic, t0 + 0.05), min(to, t1 - 0.05)
        h, c, g = read_table(os.path.join(d, f"grf_feet_filt{fc:g}.mot"))
        ad = os.path.join(d, "axistest"); os.makedirs(ad, exist_ok=True)
        vals = []
        for name, f in VARIANTS.items():
            gv = g.copy()
            for p in ("ground_force_", "l_ground_force_"):
                ix, iz = c.index(p + "vx"), c.index(p + "vz")
                gv[:, ix], gv[:, iz] = f(g[:, ix], g[:, iz])
            safe = name.replace("+", "p").replace("-", "m")
            gf = os.path.join(ad, f"grf_{safe}.mot"); write_table(gf, h, c, gv)
            ext = os.path.join(ad, f"ext_{safe}.xml"); open(ext, "w").write(ext_loads(gf))
            setup = os.path.join(ad, f"id_{safe}.xml")
            open(setup, "w").write(id_setup(r["model"], os.path.join(d, "opensim_ik.mot"), ext,
                                            t0 + 0.05, t1 - 0.05, fc, ad, f"id_{safe}.sto"))
            if not run(["opensim-cmd", "run-tool", setup], os.path.join(ad, f"id_{safe}_log.txt"), False):
                vals.append(np.nan); out.append([tag, name, "nan", "", "", ""]); continue
            t, F = pelvis_force(os.path.join(ad, f"id_{safe}.sto"))
            sel = (t >= s0) & (t <= s1)
            rms = float(np.sqrt(np.mean((F[sel] ** 2).sum(axis=1)))) / bw
            m = F[sel].mean(axis=0) / bw
            vals.append(rms); out.append([tag, name, f"{rms:.3f}", f"{m[0]:.3f}", f"{m[1]:.3f}", f"{m[2]:.3f}"])
        print(f"{tag:45s}" + "".join(f"{v:8.3f}" for v in vals), flush=True)
    res = os.path.join(work, "residual_check", "axis_check.csv")
    os.makedirs(os.path.dirname(res), exist_ok=True)
    with open(res, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["trial", "variant", "A_rms_bw", "mean_x_bw", "mean_y_bw", "mean_z_bw"]); w.writerows(out)
    print(f"\nper-variant results: {res}")


if __name__ == "__main__":
    main()
