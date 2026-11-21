
"""Impact-transient test: can event-based knots (+ less smoothing around contacts) follow a fast change
at foot contact better than OpenSim IK + a fixed low-pass filter?

Built from dataset-1 trials (only their models, markers, GRF timing and pass-1 OpenSim IK; no ID results).
For each chosen trial and seed:
  1. synth_markers: truth = smoothed OpenSim IK of that trial, PLUS a fast transient at every foot contact
     in the trial's GRF (knee, ankle, hip flexion of that leg; --impact-dv, --impact-tau), then
     soft-tissue artifact + noise as usual. Contact times -> synth_impacts.csv.
  2. OpenSim IK on the fake markers (same marker weights as the trial's ik_setup.xml, arms off).
  3. spline_ik, started from that OpenSim result, for each lambda:
       U   uniform knots every 0.02 s
       E   + dense knots (0.01 s) from 30 ms before to 100 ms after each foot event (from the GRF)
       EA  E, and the roughness penalty x 0.1 inside those windows
  4. compare_truth: OpenSim at 6/10/15/20 Hz, and every spline result; errors within 60 ms of a contact
     ("near") and elsewhere ("far"), plus the peak knee q-ddot near contacts as % of the true peak.
Rows go to <out>/summary.csv; the table at the end averages them over trials and seeds.
Results that already exist are reused, so the script can be rerun (e.g. with more lambdas).

usage (from the ankle folder):
  python src/kin/impact_test.py <work_dir> <out_dir> [--participants 02,08] [--seeds 1,2]
         [--lambdas 0.03,0.3] [--impact-dv 1.0] [--impact-tau 0.008] [--summary-only]
<work_dir> = the folder with trials.csv whose trial folders contain opensim_ik.mot and ik_setup.xml
(the one you gave step3_run.py).
"""
import csv
import os
import re
import subprocess
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from prep_trials import ik_setup, trc_info  # noqa: E402

BUILD = "/workspaces/codespaces-blank/ankle/build"
CUTOFFS = [6, 10, 15, 20]


def run(cmd, log):
    with open(log, "w") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode == 0


def setup_weights(path):
    """Marker weights from an OpenSim IK setup file."""
    text = open(path).read()
    tasks = re.findall(r'<IKMarkerTask name="([^"]+)">(.*?)</IKMarkerTask>', text, re.S)
    out = {}
    for name, body in tasks:
        m = re.search(r"<weight>\s*([^<\s]+)\s*</weight>", body)
        if m:
            out[name] = float(m.group(1))
    return out


def summarize(summary):
    acc = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    order = []
    for row in csv.reader(open(summary)):
        if len(row) < 8:
            continue
        tag, label = row[0], row[1]
        region = tag.split("|")[-1]
        key = (region, label)
        if key not in acc:
            order.append(key)
        a = acc[key]
        a[0] += float(row[4]); a[1] += float(row[5])
        if len(row) > 8 and row[8]:
            a[2] += float(row[8])
        a[3] += 1
    labels = []
    for _, label in order:
        if label not in labels:
            labels.append(label)
    n_runs = max(a[3] for a in acc.values()) if acc else 0
    print(f"\nmeans over {n_runs} trial-seeds (q in deg, q-ddot RMS error in deg/s^2, peak = knee q-ddot "
          f"peak near contacts as % of true)")
    print(f"{'method':12s}{'near q':>9s}{'near qdd':>10s}{'peak %':>8s}{'far q':>9s}{'far qdd':>10s}")
    for label in labels:
        n, f = acc.get(("near", label)), acc.get(("far", label))
        if not n or not f:
            continue
        print(f"{label:12s}{n[0] / n[3]:9.2f}{n[1] / n[3]:10.0f}{n[2] / n[3]:8.0f}{f[0] / f[3]:9.2f}{f[1] / f[3]:10.0f}")


def main():
    args = sys.argv[1:]
    opts = {"--participants": "02,08", "--seeds": "1,2", "--lambdas": "0.03,0.3",
            "--impact-dv": "1.0", "--impact-tau": "0.008"}
    pos, i, summary_only = [], 0, False
    while i < len(args):
        if args[i] in opts: opts[args[i]] = args[i + 1]; i += 2
        elif args[i] == "--summary-only": summary_only = True; i += 1
        else: pos.append(args[i]); i += 1
    if len(pos) < 2:
        print(__doc__); sys.exit(1)
    work, out = (os.path.abspath(p) for p in pos[:2])
    os.makedirs(out, exist_ok=True)
    summary = os.path.join(out, "summary.csv")
    if summary_only:
        summarize(summary); return
    if os.path.exists(summary):
        os.remove(summary)   # the comparisons are cheap; rebuild the table from scratch

    rows = list(csv.DictReader(open(os.path.join(work, "trials.csv"))))
    chosen = []
    for pid in opts["--participants"].split(","):
        mine = sorted((r for r in rows if r["participant"] == pid), key=lambda r: r["trial"])
        if mine: chosen.append(mine[0])
        else: print(f"participant {pid} not found, skipped")
    lambdas = opts["--lambdas"].split(",")
    print(f"trials {[r['participant'] + '_' + r['trial'] for r in chosen]}, seeds {opts['--seeds']}, "
          f"lambdas {lambdas}, impact dv {opts['--impact-dv']} rad/s, tau {opts['--impact-tau']} s", flush=True)

    for r in chosen:
        source = os.path.join(r["dir"], "opensim_ik.mot")
        stride = float(r["ic2"]) - float(r["ic1"])
        markers, _, _ = trc_info(r["trc"])
        weights = setup_weights(os.path.join(r["dir"], "ik_setup.xml"))
        for s in opts["--seeds"].split(","):
            tag = f"P{r['participant']}_{r['trial']}_s{s}"
            d = os.path.join(out, tag)
            os.makedirs(d, exist_ok=True)
            prefix = os.path.join(d, "synth")
            print(tag, flush=True)
            if not os.path.exists(prefix + "_impacts.csv"):
                if not run([f"{BUILD}/synth_markers", r["model"], source, prefix, "--seed", s,
                            "--stride", f"{stride:.4f}", "--markers-like", r["trc"], "--impacts", r["grf"],
                            "--impact-dv", opts["--impact-dv"], "--impact-tau", opts["--impact-tau"]],
                           prefix + "_log.txt"):
                    print("  synth_markers failed, see", prefix + "_log.txt"); continue
            _, t0, t1 = trc_info(prefix + ".trc")
            setup = os.path.join(d, "ik_setup.xml")
            os_mot = os.path.join(d, "opensim.mot")
            with open(setup, "w") as f:
                f.write(ik_setup(r["model"], prefix + ".trc", t0, t1, markers, weights, os_mot, d))
            if not os.path.exists(os_mot) and not run(["opensim-cmd", "run-tool", setup],
                                                      os.path.join(d, "opensim_log.txt")):
                print("  OpenSim IK failed"); continue
            results = [f"os{c}:{c}={os_mot}" for c in CUTOFFS]
            for lam in lambdas:
                variants = {
                    "U": ["0.02", setup],
                    "E": ["0.02", setup, r["grf"], "0.01", "0.03", "0.10"],
                    "EA": ["0.02", setup, r["grf"], "0.01", "0.03", "0.10", "--event-smooth-factor", "0.1"],
                }
                for name, extra in variants.items():
                    sp = os.path.join(d, f"spline_{name}_l{lam}.mot")
                    if os.path.exists(sp) or run([f"{BUILD}/spline_ik", r["model"], prefix + ".trc", os_mot, sp,
                                                  *extra, "--no-arms", "--smooth", lam], sp[:-4] + "_log.txt"):
                        results.append(f"{name}_l{lam}={sp}")
                    else:
                        print(f"  spline_ik {name} lambda {lam} failed, see {sp[:-4]}_log.txt")
            run([sys.executable, os.path.join(HERE, "compare_truth.py"), prefix, *results, "--around-impacts", "0.06",
                 "--summary", summary, "--tag", tag], os.path.join(d, "compare.txt"))
            print(f"  done: {os.path.join(d, 'compare.txt')}", flush=True)
    summarize(summary)


if __name__ == "__main__":
    main()
