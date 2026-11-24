
"""Protocol step 2 (synthetic tuning) for the AddBiomechanics datasets (Soccer COD, ACL, Sidestep).

Same procedure as step2_synthetic.py (Nitschke), with three differences:
  - subjects are given explicitly (--participants), and the first trial (sorted by name) of each is used;
  - the soft-tissue artifact repeats once per trial (these trials are single stance segments with no
    contact times in trials.csv), unless trials.csv has ic1/ic2;
  - the IK setup copies each marker's on/off state from the trial's real ik_setup.xml (head and arm
    markers stay off), with weights from --weights (learned) or all 1 (--weights ones).
For each chosen trial and seed:
  1. synth_markers on that trial's pass-1 OpenSim IK (truth = smoothed IK; markers = FK on the subject's
     model, only this trial's markers, STA + 1 mm noise);
  2. OpenSim IK on the fake markers;
  3. spline_ik at each lambda (knots 0.02 s, --no-arms), started from that OpenSim result;
  4. compare_truth at every cutoff, angles filtered and unfiltered; rows -> <out>/summary.csv
     (same format as step2_synthetic.py, so summarize_step2.py applies the protocol rules).
Results that already exist are reused, so the script can be rerun with extra lambdas.

usage (from the ankle folder):
  python3 src/kin/step2_synthetic_addbio.py <work_dir> <out_dir> --weights <weights.csv|ones>
          --participants A,B,C [--lambdas 0.1,0.3,1,3,10] [--seeds 1,2,3,4] [--dry-run]
"""
import csv, os, re, subprocess, sys

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


def real_apply(setup_path):
    """marker -> True/False (apply) from an existing OpenSim IK setup."""
    text = open(setup_path).read()
    out = {}
    for name, body in re.findall(r'<IKMarkerTask name="([^"]+)">(.*?)</IKMarkerTask>', text, re.S):
        m = re.search(r"<apply>\s*(\w+)\s*</apply>", body)
        out[name] = (m.group(1).lower() == "true") if m else True
    return out


def setup_text(model, trc, t0, t1, markers, weights, out_mot, d, apply):
    text = ik_setup(model, trc, t0, t1, markers, weights, out_mot, d)
    for m, on in apply.items():
        if not on:
            text = re.sub(rf'(<IKMarkerTask name="{re.escape(m)}">\s*<apply>)true(</apply>)', r"\1false\2", text)
    return text


def main():
    args = sys.argv[1:]
    opts = {"--weights": None, "--lambdas": "0.1,0.3,1,3,10", "--seeds": "1,2,3,4", "--participants": None, "--cutoffs": "6,8,10,12,15,20"}
    dry, pos, i = False, [], 0
    while i < len(args):
        if args[i] == "--dry-run": dry = True; i += 1
        elif args[i] in opts: opts[args[i]] = args[i + 1]; i += 2
        else: pos.append(args[i]); i += 1
    if len(pos) < 2 or not opts["--weights"] or not opts["--participants"]:
        print(__doc__); sys.exit(1)
    work, out = (os.path.abspath(p) for p in pos[:2])
    wname = "ones" if opts["--weights"] == "ones" else "learned"
    weights = {}
    if wname == "learned":
        weights = {r["marker"]: float(r["weight"]) for r in csv.DictReader(open(opts["--weights"]))}
    lambdas, seeds = opts["--lambdas"].split(","), opts["--seeds"].split(",")
    os.makedirs(out, exist_ok=True)
    summary = os.path.join(out, "summary.csv")

    rows = list(csv.DictReader(open(os.path.join(work, "trials.csv"))))
    chosen = []
    for pid in opts["--participants"].split(","):
        mine = sorted((r for r in rows if r["participant"] == pid), key=lambda r: r["trial"])
        if mine: chosen.append(mine[0])
        else: print(f"participant {pid} not found, skipped")
    print(f"weights: {wname}; trials: {[r['participant'] + '_' + r['trial'] for r in chosen]}; "
          f"seeds {seeds}; lambdas {lambdas}", flush=True)

    for r in chosen:
        source = os.path.join(r["dir"], "opensim_ik.mot")
        markers, tr0, tr1 = trc_info(r["trc"])
        try:
            stride = float(r["ic2"]) - float(r["ic1"])
        except (KeyError, ValueError):
            stride = tr1 - tr0                       # one STA period over the whole trial
        apply = real_apply(os.path.join(r["dir"], "ik_setup.xml"))
        for s in seeds:
            tag = f"{r['participant']}_{r['trial']}_s{s}"
            d = os.path.join(out, tag)
            os.makedirs(d, exist_ok=True)
            prefix = os.path.join(d, "synth")
            print(f"{tag} [{wname}]", flush=True)
            if not os.path.exists(prefix + ".trc"):
                if not run([f"{BUILD}/synth_markers", r["model"], source, prefix, "--seed", s,
                            "--stride", f"{stride:.4f}", "--markers-like", r["trc"]], prefix + "_log.txt", dry):
                    print("  synth_markers failed, see", prefix + "_log.txt"); continue
            t0, t1 = (tr0, tr1) if dry else trc_info(prefix + ".trc")[1:]
            setup = os.path.join(d, f"ik_setup_{wname}.xml")
            os_mot = os.path.join(d, f"opensim_{wname}.mot")
            with open(setup, "w") as f:
                f.write(setup_text(r["model"], prefix + ".trc", t0, t1, markers, weights, os_mot, d, apply))
            if not os.path.exists(os_mot) and not run(["opensim-cmd", "run-tool", setup],
                                                      os.path.join(d, f"opensim_{wname}_log.txt"), dry):
                print("  OpenSim IK failed"); continue
            results = [f"opensim={os_mot}"]
            for lam in lambdas:
                sp = os.path.join(d, f"spline_{wname}_l{lam}.mot")
                if os.path.exists(sp) or run([f"{BUILD}/spline_ik", r["model"], prefix + ".trc", os_mot, sp, "0.02",
                                              setup, "--no-arms", "--smooth", lam], sp[:-4] + "_log.txt", dry):
                    results.append(f"spline_l{lam}={sp}")
                else:
                    print(f"  spline_ik lambda {lam} failed, see {sp[:-4]}_log.txt")
            for c in opts["--cutoffs"].split(","):
                for filt in ([], ["--filter-angles"]):
                    run([sys.executable, os.path.join(HERE, "compare_truth.py"), prefix, *results,
                         "--cutoff", str(c), *filt, "--summary", summary, "--tag", f"{tag},{wname}"],
                        os.devnull, dry)
    print(f"done; summary rows in {summary}")


if __name__ == "__main__":
    main()
