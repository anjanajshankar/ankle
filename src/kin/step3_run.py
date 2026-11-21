
"""Protocol steps 3-4 for the Nitschke dataset: both pipelines + ID on every real trial, then the
selection criterion (pelvis residual force during right-foot stance).

For each trial in <work>/trials.csv (setups made by prep_trials.py --weights <learned weights>):
  A. OpenSim IK (ik_setup.xml) -> opensim_ik.mot
  B. spline_ik (knots 0.02 s, --smooth LAMBDA, --no-arms, same weights), started from A -> spline_ik.mot
  GRF: forces and centre of pressure low-pass filtered at F_C (zero-lag 4th-order Butterworth), the same
       cutoff as Pipeline A's kinematics; the centre of pressure is held at its nearest stance value while a
       foot is off the ground so the filter does not drag it to the lab origin. Both pipelines use this file.
  Arms: not solved by spline_ik; to keep the two pipelines identical apart from the legs/trunk/pelvis,
       Pipeline B's ID gets the arm coordinates of Pipeline A, filtered at F_C.
  ID A: OpenSim ID on opensim_ik.mot, coordinates filtered at F_C.
  ID B: OpenSim ID on the spline result, no extra filtering.
  Criterion: RMS of the pelvis residual force magnitude sqrt(Fx^2+Fy^2+Fz^2) / body weight, over right-foot
       stance (first right contact from metaInfo to the first time the right vertical GRF drops below 20 N).
Only pelvis residual columns are read from the ID results (blinding: do not open the ID files).

usage (from the ankle folder):
  python src/kin/step3_run.py <work_final> --fc 15 --lambda 0.03 [--only P02_trial0122] [--dry-run]
"""
import csv
import os
import subprocess
import sys

import numpy as np

BUILD = "/workspaces/codespaces-blank/ankle/build"
ARM_COORDS = ("arm_flex", "arm_add", "arm_rot", "elbow_flex", "pro_sup", "wrist_flex", "wrist_dev")
G = 9.81


def read_table(path):
    lines = open(path).readlines()
    start = next(i for i, l in enumerate(lines) if l.strip().lower() == "endheader") + 1
    header = lines[:start]
    cols = lines[start].split()
    data = np.loadtxt([l for l in lines[start + 1:] if l.strip()], ndmin=2)
    return header, cols, data


def write_table(path, header, cols, data):
    with open(path, "w") as f:
        for h in header:
            if h.lower().startswith("nrows"):
                h = f"nRows={len(data)}\n"
            elif h.lower().startswith("ncolumns"):
                h = f"nColumns={len(cols)}\n"
            f.write(h)
        f.write("\t".join(cols) + "\n")
        for row in data:
            f.write("\t".join(f"{v:.8g}" for v in row) + "\n")


def lowpass(y, dt, fc):
    from scipy.signal import butter, filtfilt
    b, a = butter(4, fc / (0.5 / dt))
    return filtfilt(b, a, y, axis=0)


def filter_grf(src, dst, fc, threshold=20.0):
    header, cols, d = read_table(src)
    dt = d[1, 0] - d[0, 0]
    out = d.copy()
    for prefix in ("ground_force_", "l_ground_force_"):
        if prefix + "vy" not in cols:
            continue
        on = d[:, cols.index(prefix + "vy")] > threshold
        for axis in "xyz":
            name = prefix + "p" + axis
            if name not in cols or not on.any():
                continue
            c = d[:, cols.index(name)].copy()
            idx = np.where(on, np.arange(len(c)), -1)
            fwd = np.maximum.accumulate(idx)                       # last stance index so far
            first = np.argmax(on)
            fwd[fwd < 0] = first
            c = c[fwd]                                             # hold last stance value
            out[:, cols.index(name)] = c
    force_cols = [k for k, n in enumerate(cols) if n != "time"]
    out[:, force_cols] = lowpass(out[:, force_cols], dt, fc)
    write_table(dst, header, cols, out)


def merge_arms(spline_mot, opensim_mot, dst, fc):
    h, sc, sd = read_table(spline_mot)
    _, oc, od = read_table(opensim_mot)
    dt = od[1, 0] - od[0, 0]
    arms = [n for n in oc if n.startswith(ARM_COORDS) and n not in sc]
    filt = lowpass(od[:, [oc.index(n) for n in arms]], dt, fc) if arms else np.zeros((len(od), 0))
    extra = np.column_stack([np.interp(sd[:, 0], od[:, 0], filt[:, k]) for k in range(len(arms))]) \
        if arms else np.zeros((len(sd), 0))
    write_table(dst, h, sc + arms, np.column_stack([sd, extra]))


def ext_loads(grf, right="calcn_r", left="calcn_l"):
    def force(name, body, prefix):
        return (f'\t\t\t<ExternalForce name="{name}">\n\t\t\t\t<applied_to_body>{body}</applied_to_body>\n'
                f'\t\t\t\t<force_expressed_in_body>ground</force_expressed_in_body>\n'
                f'\t\t\t\t<point_expressed_in_body>ground</point_expressed_in_body>\n'
                f'\t\t\t\t<force_identifier>{prefix}force_v</force_identifier>\n'
                f'\t\t\t\t<point_identifier>{prefix}force_p</point_identifier>\n\t\t\t</ExternalForce>')
    return (f'<?xml version="1.0" encoding="UTF-8" ?>\n<OpenSimDocument Version="40000">\n'
            f'\t<ExternalLoads name="grf">\n\t\t<objects>\n{force("right_GRF", right, "ground_")}\n'
            f'{force("left_GRF", left, "l_ground_")}\n\t\t</objects>\n\t\t<groups />\n'
            f'\t\t<datafile>{grf}</datafile>\n\t</ExternalLoads>\n</OpenSimDocument>\n')


def id_setup(model, coords, ext, t0, t1, fc, out_dir, out_name):
    return (f'<?xml version="1.0" encoding="UTF-8" ?>\n<OpenSimDocument Version="40000">\n'
            f'\t<InverseDynamicsTool name="id">\n\t\t<results_directory>{out_dir}</results_directory>\n'
            f'\t\t<model_file>{model}</model_file>\n\t\t<time_range> {t0:.5f} {t1:.5f}</time_range>\n'
            f'\t\t<forces_to_exclude> Muscles</forces_to_exclude>\n'
            f'\t\t<external_loads_file>{ext}</external_loads_file>\n'
            f'\t\t<coordinates_file>{coords}</coordinates_file>\n'
            f'\t\t<lowpass_cutoff_frequency_for_coordinates>{fc}</lowpass_cutoff_frequency_for_coordinates>\n'
            f'\t\t<output_gen_force_file>{out_name}</output_gen_force_file>\n'
            f'\t</InverseDynamicsTool>\n</OpenSimDocument>\n')


def right_stance(grf, ic1, threshold=20.0):
    _, cols, d = read_table(grf)
    t, vy = d[:, 0], d[:, cols.index("ground_force_vy")]
    after = (t >= ic1) & (vy < threshold)
    after &= t > ic1 + 0.02
    t_off = t[np.argmax(after)] if after.any() else t[-1]
    return ic1, t_off


def pelvis_residual(id_sto, t0, t1, mass):
    _, cols, d = read_table(id_sto)
    sel = (d[:, 0] >= t0) & (d[:, 0] <= t1)
    f = [cols.index(f"pelvis_t{a}_force") for a in "xyz"]
    mag = np.sqrt((d[sel][:, f] ** 2).sum(axis=1))
    mcols = [k for k, n in enumerate(cols) if n.startswith("pelvis_") and n.endswith("_moment")]
    mom = np.sqrt((d[sel][:, mcols] ** 2).sum(axis=1)) if mcols else np.array([np.nan])
    return np.sqrt(np.mean(mag ** 2)) / (mass * G), np.sqrt(np.mean(mom ** 2)) / (mass * G)


def run(cmd, log, dry):
    if dry:
        print("  $", " ".join(cmd)); return True
    with open(log, "w") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode == 0


def main():
    args = sys.argv[1:]
    fc, lam, only, dry, pos, i = None, None, None, False, [], 0
    while i < len(args):
        if args[i] == "--fc": fc = float(args[i + 1]); i += 2
        elif args[i] == "--lambda": lam = args[i + 1]; i += 2
        elif args[i] == "--only": only = args[i + 1]; i += 2
        elif args[i] == "--dry-run": dry = True; i += 1
        else: pos.append(args[i]); i += 1
    if not pos or fc is None or lam is None:
        print(__doc__); sys.exit(1)
    work = os.path.abspath(pos[0])
    rows = list(csv.DictReader(open(os.path.join(work, "trials.csv"))))
    results = os.path.join(work, "step3_results.csv")
    out = []
    for r in rows:
        tag = f"P{r['participant']}_{r['trial']}"
        if only and tag != only:
            continue
        d, model, t0, t1 = r["dir"], r["model"], float(r["t0"]), float(r["t1"])
        mass, ic1 = float(r["mass_kg"]), float(r["ic1"])
        os_mot, sp_mot = os.path.join(d, "opensim_ik.mot"), os.path.join(d, "spline_ik.mot")
        print(tag, flush=True)
        if not os.path.exists(os_mot) and not run(["opensim-cmd", "run-tool", os.path.join(d, "ik_setup.xml")],
                                                  os.path.join(d, "opensim_ik_log.txt"), dry):
            print("  OpenSim IK failed"); continue
        if not os.path.exists(sp_mot) and not run(
                [f"{BUILD}/spline_ik", model, r["trc"], os_mot, sp_mot, "0.02", os.path.join(d, "ik_setup.xml"),
                 "--no-arms", "--smooth", lam], os.path.join(d, "spline_ik_log.txt"), dry):
            print("  spline_ik failed"); continue
        if dry:
            continue
        grf_f = os.path.join(d, f"grf_filt{fc:g}.mot")
        filter_grf(r["grf"], grf_f, fc)
        ext = os.path.join(d, "external_loads_filtered.xml")
        open(ext, "w").write(ext_loads(grf_f))
        sp_arms = os.path.join(d, "spline_ik_witharms.mot")
        merge_arms(sp_mot, os_mot, sp_arms, fc)
        scores = {}
        for name, coords, cut in (("A", os_mot, fc), ("B", sp_arms, -1)):
            setup = os.path.join(d, f"id_setup_{name}.xml")
            open(setup, "w").write(id_setup(model, coords, ext, t0 + 0.05, t1 - 0.05, cut, d, f"id_{name}.sto"))
            if not run(["opensim-cmd", "run-tool", setup], os.path.join(d, f"id_{name}_log.txt"), dry):
                print(f"  ID {name} failed"); scores[name] = (np.nan, np.nan); continue
            s0, s1 = right_stance(r["grf"], ic1)
            scores[name] = pelvis_residual(os.path.join(d, f"id_{name}.sto"), s0, s1, mass)
        out.append([tag, *scores["A"], *scores["B"]])
        print(f"  pelvis residual force / BW (stance RMS): A {scores['A'][0]:.4f}  B {scores['B'][0]:.4f}"
              f"   moment / BW: A {scores['A'][1]:.4f}  B {scores['B'][1]:.4f}")

    if out:
        with open(results, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["trial", "A_force_bw", "A_moment_bw", "B_force_bw", "B_moment_bw"])
            w.writerows(out)
        A = np.array([o[1] for o in out]); B = np.array([o[3] for o in out])
        mA, mB = np.nanmedian(A), np.nanmedian(B)
        print(f"\nmedian pelvis residual force / BW over {len(out)} trials: A (OpenSim) {mA:.4f}   B (spline) {mB:.4f}")
        print(f"relative difference {abs(mA - mB) / min(mA, mB) * 100:.1f}%  (protocol: within 10% -> secondary criterion)")
        print(f"per-trial results: {results}")


if __name__ == "__main__":
    main()
