
"""Prepare OpenSim setup files for every trial of the Nitschke et al. change-of-direction dataset.

For each trial of the chosen motion (default: vcut) it writes, into <work>/Participant_XX/trialYYYY/:
  ik_setup.xml        OpenSim IK setup (model, markers, time range, marker weights, output paths)
  external_loads.xml  GRF -> feet mapping for OpenSim ID (right: ground_force_*, left: l_ground_force_*)
and a list of all trials (trials.csv) that the run scripts loop over.

Time range: from (first right initial contact - margin) to (second right initial contact + margin),
clipped to the marker data; contact times come from metaInfo.csv.

Marker weights: all 1 by default; --weights weights.csv (from make_weights.py) overrides them.
Arm markers (elbow, wrist) are always switched off (apply=false), matching spline_ik --no-arms.

usage: python prep_trials.py <raw_dir> <work_dir> [--motion vcut] [--margin 0.1] [--weights weights.csv]
"""
import csv
import os
import re
import sys

ARM_MARKERS = re.compile(r"^[RL](ElbM|ElbL|RSP|USP)$")


def trc_info(path):
    """Marker names and first/last time of a .trc file."""
    with open(path) as f:
        lines = f.readlines()
    names = [n for n in lines[3].rstrip("\n").split("\t")[2:] if n.strip()]
    data = [l for l in lines[5:] if l.strip()]
    first = float(data[0].split("\t")[1])
    last = float(data[-1].split("\t")[1])
    return [n.strip() for n in names], first, last


def mot_times(path):
    with open(path) as f:
        lines = f.readlines()
    start = next(i for i, l in enumerate(lines) if l.strip().lower() == "endheader") + 2
    rows = [l for l in lines[start:] if l.strip()]
    return float(rows[0].split()[0]), float(rows[-1].split()[0])


def foot_bodies(model_path):
    text = open(model_path).read()
    bodies = re.findall(r'<Body name="([^"]+)"', text)
    for r, l in [("calcn_r", "calcn_l")]:
        if r in bodies and l in bodies:
            return r, l
    raise SystemExit(f"no calcn_r/calcn_l body in {model_path}; bodies: {bodies}")


def ik_setup(model, trc, t0, t1, markers, weights, out_mot, results_dir):
    tasks = []
    for m in markers:
        apply = not ARM_MARKERS.match(m)
        w = weights.get(m, 1.0)
        tasks.append(f'\t\t\t\t<IKMarkerTask name="{m}">\n\t\t\t\t\t<apply>{"true" if apply else "false"}</apply>\n'
                     f'\t\t\t\t\t<weight>{w:.6g}</weight>\n\t\t\t\t</IKMarkerTask>')
    return f"""<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="40000">
\t<InverseKinematicsTool name="ik">
\t\t<results_directory>{results_dir}</results_directory>
\t\t<model_file>{model}</model_file>
\t\t<constraint_weight>Inf</constraint_weight>
\t\t<accuracy>1e-05</accuracy>
\t\t<IKTaskSet>
\t\t\t<objects>
{chr(10).join(tasks)}
\t\t\t</objects>
\t\t\t<groups />
\t\t</IKTaskSet>
\t\t<marker_file>{trc}</marker_file>
\t\t<coordinate_file>Unassigned</coordinate_file>
\t\t<time_range> {t0:.5f} {t1:.5f}</time_range>
\t\t<report_errors>true</report_errors>
\t\t<output_motion_file>{out_mot}</output_motion_file>
\t\t<report_marker_locations>false</report_marker_locations>
\t</InverseKinematicsTool>
</OpenSimDocument>
"""


def external_loads(grf, right_body, left_body):
    def force(name, body, prefix):
        return f"""\t\t\t<ExternalForce name="{name}">
\t\t\t\t<applied_to_body>{body}</applied_to_body>
\t\t\t\t<force_expressed_in_body>ground</force_expressed_in_body>
\t\t\t\t<point_expressed_in_body>ground</point_expressed_in_body>
\t\t\t\t<force_identifier>{prefix}force_v</force_identifier>
\t\t\t\t<point_identifier>{prefix}force_p</point_identifier>
\t\t\t</ExternalForce>"""
    return f"""<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="40000">
\t<ExternalLoads name="grf">
\t\t<objects>
{force("right_GRF", right_body, "ground_")}
{force("left_GRF", left_body, "l_ground_")}
\t\t</objects>
\t\t<groups />
\t\t<datafile>{grf}</datafile>
\t</ExternalLoads>
</OpenSimDocument>
"""


def main():
    args = sys.argv[1:]
    motion, margin, weights_file = "vcut", 0.1, None
    pos = []
    i = 0
    while i < len(args):
        if args[i] == "--motion": motion = args[i + 1]; i += 2
        elif args[i] == "--margin": margin = float(args[i + 1]); i += 2
        elif args[i] == "--weights": weights_file = args[i + 1]; i += 2
        else: pos.append(args[i]); i += 1
    if len(pos) < 2:
        print(__doc__); sys.exit(1)
    raw, work = (os.path.abspath(p) for p in pos[:2])

    weights = {}
    if weights_file:
        with open(weights_file) as f:
            for row in csv.DictReader(f):
                weights[row["marker"]] = float(row["weight"])
        print(f"marker weights from {weights_file} ({len(weights)} markers)")
    else:
        print("marker weights: all 1 (arms off)")

    rows = list(csv.DictReader(open(os.path.join(raw, "metaInfo.csv"))))
    os.makedirs(work, exist_ok=True)
    out_rows = []
    for r in rows:
        if r["Motion"] != motion:
            continue
        pid, trial = r["Participant_ID"], r["Trial"]
        pdir = os.path.join(raw, f"Participant_{pid}")
        model = os.path.join(pdir, f"Participant_{pid}.osim")
        trc = os.path.join(pdir, motion, f"{trial}_measured_markers.trc")
        grf = os.path.join(pdir, motion, f"{trial}_measured_GRFs.mot")
        for p in (model, trc, grf):
            if not os.path.exists(p):
                raise SystemExit(f"missing {p}")
        markers, t_first, t_last = trc_info(trc)
        g_first, g_last = mot_times(grf)
        ic1, ic2 = float(r["Time_first_right_IC_in_s"]), float(r["Time_second_right_IC_in_s"])
        t0, t1 = max(t_first, ic1 - margin), min(t_last, ic2 + margin)
        note = ""
        if not (t_first <= ic1 <= t_last and t_first <= ic2 <= t_last):
            note = "CONTACT TIMES OUTSIDE MARKER DATA"
        right_body, left_body = foot_bodies(model)

        tdir = os.path.join(work, f"Participant_{pid}", trial)
        os.makedirs(tdir, exist_ok=True)
        with open(os.path.join(tdir, "ik_setup.xml"), "w") as f:
            f.write(ik_setup(model, trc, t0, t1, markers, weights, os.path.join(tdir, "opensim_ik.mot"), tdir))
        with open(os.path.join(tdir, "external_loads.xml"), "w") as f:
            f.write(external_loads(grf, right_body, left_body))
        out_rows.append({"participant": pid, "trial": trial, "dir": tdir, "model": model, "trc": trc, "grf": grf,
                         "t0": f"{t0:.5f}", "t1": f"{t1:.5f}", "ic1": ic1, "ic2": ic2,
                         "mass_kg": r["Bodymass_in_kg"]})
        print(f"P{pid} {trial}: markers {t_first:.3f}-{t_last:.3f} s, GRF {g_first:.3f}-{g_last:.3f} s, "
              f"contacts {ic1:.3f}/{ic2:.3f}, IK {t0:.3f}-{t1:.3f} {note}")

    with open(os.path.join(work, "trials.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"\n{len(out_rows)} {motion} trials prepared in {work} (list: trials.csv)")


if __name__ == "__main__":
    main()
