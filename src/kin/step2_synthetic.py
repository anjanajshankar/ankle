
"""Protocol step 2 for the Nitschke dataset: synthetic test built from this dataset.

For each chosen trial and seed:
  1. truth + fake markers: synth_markers on the pass-1 OpenSim IK of that trial (participant's model,
     only the measured markers, soft-tissue artifact repeating once per right-contact-to-contact cycle);
  2. OpenSim IK on the fake markers (setup written with the given weights, arms off);
  3. spline_ik at each lambda (knots 0.02 s, --no-arms), started from that OpenSim result;
  4. compare_truth against the truth at every cutoff, angles filtered and unfiltered;
     rows are appended to <out>/summary.csv for summarize_step2.py.

Trials (fixed in advance): the first v-cut trial of each chosen participant (default 02, 05, 08, 11).

usage (from the ankle folder):
  python src/kin/step2_synthetic.py <work_pass1> <out_dir> --weights <weights.csv|ones>
         [--lambdas 0.1,0.3,1,3,10] [--seeds 1,2,3,4] [--participants 02,05,08,11] [--dry-run]
"""
import csv
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from prep_trials import ik_setup, trc_info  # noqa: E402

BUILD = "/workspaces/codespaces-blank/ankle/build"
CUTOFFS = [6, 8, 10, 12, 15, 20]


def run(cmd, log, dry):
    if dry:
        print("  $", " ".join(cmd)); return True
    with open(log, "w") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode == 0


def main():
    args = sys.argv[1:]
    opts = {"--weights": None, "--lambdas": "0.1,0.3,1,3,10", "--seeds": "1,2,3,4",
            "--participants": "02,05,08,11"}
    dry, pos, i = False, [], 0
    while i < len(args):
        if args[i] == "--dry-run": dry = True; i += 1
        elif args[i] in opts: opts[args[i]] = args[i + 1]; i += 2
        else: pos.append(args[i]); i += 1
    if len(pos) < 2 or not opts["--weights"]:
        print(__doc__); sys.exit(1)
    work, out = (os.path.abspath(p) for p in pos[:2])
    wname = "ones" if opts["--weights"] == "ones" else "learned"
    weights = {}
    if wname == "learned":
        weights = {r["marker"]: float(r["weight"]) for r in csv.DictReader(open(opts["--weights"]))}
    lambdas = opts["--lambdas"].split(",")
    seeds = opts["--seeds"].split(",")
    wanted = opts["--participants"].split(",")
    os.makedirs(out, exist_ok=True)
    summary = os.path.join(out, "summary.csv")

    rows = list(csv.DictReader(open(os.path.join(work, "trials.csv"))))
    chosen = []
    for pid in wanted:
        mine = sorted((r for r in rows if r["participant"] == pid), key=lambda r: r["trial"])
        if mine: chosen.append(mine[0])
        else: print(f"participant {pid} not found, skipped")
    print(f"weights: {wname}; trials: {[r['participant'] + '_' + r['trial'] for r in chosen]}; "
          f"seeds {seeds}; lambdas {lambdas}")

    for r in chosen:
        source = os.path.join(r["dir"], "opensim_ik.mot")
        stride = float(r["ic2"]) - float(r["ic1"])
        markers, _, _ = trc_info(r["trc"])
        for s in seeds:
            tag = f"P{r['participant']}_{r['trial']}_s{s}"
            d = os.path.join(out, tag)
            os.makedirs(d, exist_ok=True)
            prefix = os.path.join(d, "synth")
            print(f"{tag} [{wname}]", flush=True)
            if not os.path.exists(prefix + ".trc"):
                if not run([f"{BUILD}/synth_markers", r["model"], source, prefix, "--seed", s,
                            "--stride", f"{stride:.4f}", "--markers-like", r["trc"]], prefix + "_log.txt", dry):
                    print("  synth_markers failed"); continue
            if dry:
                t0, t1 = float(r["t0"]), float(r["t1"])
            else:
                _, t0, t1 = trc_info(prefix + ".trc")
            setup = os.path.join(d, f"ik_setup_{wname}.xml")
            os_mot = os.path.join(d, f"opensim_{wname}.mot")
            with open(setup, "w") as f:
                f.write(ik_setup(r["model"], prefix + ".trc", t0, t1, markers, weights, os_mot, d))
            if not run(["opensim-cmd", "run-tool", setup], os.path.join(d, f"opensim_{wname}_log.txt"), dry):
                print("  OpenSim IK failed"); continue
            results = [f"opensim={os_mot}"]
            for lam in lambdas:
                sp = os.path.join(d, f"spline_{wname}_l{lam}.mot")
                if run([f"{BUILD}/spline_ik", r["model"], prefix + ".trc", os_mot, sp, "0.02", setup,
                        "--no-arms", "--smooth", lam], sp[:-4] + "_log.txt", dry):
                    results.append(f"spline_l{lam}={sp}")
                else:
                    print(f"  spline_ik lambda {lam} failed")
            for c in CUTOFFS:
                for filt in ([], ["--filter-angles"]):
                    run([sys.executable, os.path.join(HERE, "compare_truth.py"), prefix, *results,
                         "--cutoff", str(c), *filt, "--summary", summary, "--tag", f"{tag},{wname}"],
                        os.devnull, dry)
    print(f"done; summary rows in {summary}")


if __name__ == "__main__":
    main()
