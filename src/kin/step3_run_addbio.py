"""Protocol steps 3-4 for the AddBiomechanics datasets with force plates (Soccer COD, Sidestep):
inverse dynamics for both pipelines on every analysed trial, and the pelvis-residual quality measure.
Reads <work>/trials_grf.csv (assign_plates.py) and reuses step3_run.py's functions, so filtering and
ID settings are identical. GRF filtered at --fc for both pipelines (matched filtering: B's equivalent
cutoff = A's f_c, see protocol). Residual window = plant contact (ic -> toe_off).
Only the pelvis residual columns of the ID results are read (blinding). Rule-5 exclusions skipped.
usage (from the ankle folder):  python3 src/kin/step3_run_addbio.py work/SoccerCOD --fc 20 [--only TAG]
"""
import csv, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step3_run import filter_grf, merge_arms, ext_loads, id_setup, pelvis_residual, run  # noqa: E402


def main():
    args = sys.argv[1:]
    if len(args) < 3 or args[1] != "--fc":
        print(__doc__); sys.exit(1)
    work, fc = os.path.abspath(args[0]), float(args[2])
    only = args[args.index("--only") + 1] if "--only" in args else None
    excl = set()
    ex_path = os.path.join(work, "excluded_trials.csv")
    if os.path.exists(ex_path):
        excl = {r["trial"] for r in csv.DictReader(open(ex_path))}
    rows = list(csv.DictReader(open(os.path.join(work, "trials_grf.csv"))))
    out = []
    for r in rows:
        p = r["participant"]
        tag = f"{p if p.startswith('P') else 'P' + p}_{r['trial']}"
        if only and tag != only: continue
        d, model = r["dir"], r["model"]
        keys = {tag, f"{r['participant']}_{r['trial']}", r["trial"], os.path.basename(os.path.normpath(d))}
        if keys & excl:
            print(tag, "excluded (rule 5)"); continue
        os_mot, sp_mot = os.path.join(d, "opensim_ik.mot"), os.path.join(d, "spline_ik.mot")
        if not (os.path.exists(os_mot) and os.path.exists(sp_mot)):
            print(tag, "missing IK result, skipped"); continue
        t0, t1, mass = float(r["t0"]), float(r["t1"]), float(r["mass_kg"])
        if mass <= 0:
            print(tag, "no body mass in trials.csv, skipped"); continue
        s0, s1 = float(r["ic"]), float(r["toe_off"])
        grf_f = os.path.join(d, f"grf_feet_filt{fc:g}.mot")
        filter_grf(r["grf_feet"], grf_f, fc)
        ext = os.path.join(d, "external_loads_filtered.xml")
        open(ext, "w").write(ext_loads(grf_f))
        sp_arms = os.path.join(d, "spline_ik_witharms.mot")
        merge_arms(sp_mot, os_mot, sp_arms, fc)
        scores = {}
        for name, coords, cut in (("A", os_mot, fc), ("B", sp_arms, -1)):
            setup = os.path.join(d, f"id_setup_{name}.xml")
            open(setup, "w").write(id_setup(model, coords, ext, t0 + 0.05, t1 - 0.05, cut, d, f"id_{name}.sto"))
            if not run(["opensim-cmd", "run-tool", setup], os.path.join(d, f"id_{name}_log.txt"), False):
                print(f"  {tag}: ID {name} failed (see {d}/id_{name}_log.txt)"); scores[name] = (np.nan, np.nan); continue
            scores[name] = pelvis_residual(os.path.join(d, f"id_{name}.sto"), max(s0, t0 + 0.05), min(s1, t1 - 0.05), mass)
        out.append([tag, r["plant_foot"], *scores["A"], *scores["B"]])
        print(f"{tag}  plant {r['plant_foot']}  pelvis residual force/BW: A {scores['A'][0]:.3f}  B {scores['B'][0]:.3f}", flush=True)
    if out:
        res = os.path.join(work, "step3_results.csv")
        with open(res, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["trial", "plant_foot", "A_force_bw", "A_moment_bw", "B_force_bw", "B_moment_bw"])
            w.writerows(out)
        A, B = np.array([o[2] for o in out]), np.array([o[4] for o in out])
        print(f"\n{len(out)} trials; median pelvis residual force/BW over the plant contact: A {np.nanmedian(A):.3f}  B {np.nanmedian(B):.3f}")
        print(f"per-trial results: {res}")


if __name__ == "__main__":
    main()
