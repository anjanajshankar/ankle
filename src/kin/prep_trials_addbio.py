
"""Prepare OpenSim IK setup files for the AddBiomechanics datasets (Soccer COD, ACL, Sidestep).

Counterpart of prep_trials.py (Nitschke). Reads the folders written by prep_addbio.py:
  <data_dir>/<subject>/subject.json, <trial>.trc, addbio_pass<k>.osim
and writes, into <work_dir>:
  <subject>/model_ik.osim             AddBiomechanics' scaled model (pass --model-pass), with the
                                      coordinates in --unlock unlocked (default: both subtalar joints)
  <subject>/<trial>/ik_setup.xml      OpenSim IK setup (whole trial, weights, output paths)
  trials.csv                          same columns as prep_trials.py, so run_pass1.sh and
                                      make_weights.py work unchanged (grf/ic1/ic2 may be empty)

Trials: those whose name matches --include (regex) and not --exclude. Defaults per dataset:
  soccer_cod : cut trials  _(45|90|135|180)_[LR]_   (static "neutral" trials excluded)
  acl        : 90-degree cuts  ^rotated_C90_
  sidestep   : all trials
Trials with fewer than --min-markers model markers are skipped (junk trials).

Time range: the whole trial (these files are already short segments around the force plate;
the analysis window is set later from foot events).

IK markers: the markers present in both the .trc and the model. Markers on arm bodies (humerus,
ulna, radius, hand) are switched off (apply=false), matching spline_ik --no-arms. --off adds
names to switch off. Marker weights: all 1 by default; --weights weights.csv (from make_weights.py)
overrides them.

usage: python prep_trials_addbio.py <data_dir> <work_dir> --dataset soccer_cod|acl|sidestep
         [--include REGEX] [--exclude REGEX] [--weights weights.csv] [--model-pass 0]
         [--unlock subtalar_angle_r,subtalar_angle_l] [--off M1,M2] [--min-markers 8]
"""
import argparse, csv, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep_trials import trc_info, ik_setup  # noqa: E402  (same folder: src/kin)

DEFAULT_INCLUDE = {"soccer_cod": r"_(45|90|135|180)_[LR]_", "acl": r"^rotated_C90_", "sidestep": r"."}
ARM_BODIES = re.compile(r"^(humerus|ulna|radius|hand)_[rl]$")


def model_markers(text):
    """{marker name: parent body} from an .osim file's MarkerSet."""
    out = {}
    for m in re.finditer(r'<Marker name="([^"]+)">(.*?)</Marker>', text, re.S):
        parent = re.search(r"<socket_parent_frame>\s*(?:/bodyset/)?([^<\s]+)\s*</socket_parent_frame>", m.group(2))
        out[m.group(1)] = parent.group(1).split("/")[-1] if parent else ""
    return out


def unlock(text, coords):
    done = []
    for c in coords:
        pat = re.compile(r'(<Coordinate name="%s">.*?<locked>)\s*true\s*(</locked>)' % re.escape(c), re.S)
        text, n = pat.subn(r"\1false\2", text)
        if n:
            done.append(c)
    return text, done


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir")
    ap.add_argument("work_dir")
    ap.add_argument("--dataset", required=True, choices=sorted(DEFAULT_INCLUDE))
    ap.add_argument("--include", default=None)
    ap.add_argument("--exclude", default=r"neutral|static")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--model-pass", type=int, default=0)
    ap.add_argument("--unlock", default="subtalar_angle_r,subtalar_angle_l")
    ap.add_argument("--off", default="")
    ap.add_argument("--min-markers", type=int, default=8)
    a = ap.parse_args()

    data, work = os.path.abspath(a.data_dir), os.path.abspath(a.work_dir)
    inc = re.compile(a.include or DEFAULT_INCLUDE[a.dataset])
    exc = re.compile(a.exclude, re.I) if a.exclude else None
    off = {m.strip() for m in a.off.split(",") if m.strip()}
    coords = [c.strip() for c in a.unlock.split(",") if c.strip()]

    weights = {}
    if a.weights:
        for row in csv.DictReader(open(a.weights)):
            weights[row["marker"]] = float(row["weight"])
        print(f"marker weights from {a.weights} ({len(weights)} markers)")
    else:
        print("marker weights: all 1 (arm markers off)")

    os.makedirs(work, exist_ok=True)
    out_rows, skipped = [], []
    for sid in sorted(os.listdir(data)):
        sdir = os.path.join(data, sid)
        sj = os.path.join(sdir, "subject.json")
        if not os.path.isfile(sj):
            continue
        info = json.load(open(sj))
        src_model = os.path.join(sdir, f"addbio_pass{a.model_pass}.osim")
        if not os.path.isfile(src_model):
            raise SystemExit(f"missing {src_model}")
        text = open(src_model).read()
        text, unlocked = unlock(text, coords)
        wsub = os.path.join(work, sid)
        os.makedirs(wsub, exist_ok=True)
        model = os.path.join(wsub, "model_ik.osim")
        open(model, "w").write(text)
        mk_body = model_markers(text)

        n_sub = 0
        for tr in info["trials"]:
            name = tr["trial"]
            if not inc.search(name) or (exc and exc.search(name)):
                continue
            trc = os.path.join(sdir, tr["file"])
            names, t_first, t_last = trc_info(trc)
            ik_mk = [m for m in names if m in mk_body]
            if len(ik_mk) < a.min_markers:
                skipped.append(f"{sid}/{name} ({len(ik_mk)} model markers)")
                continue
            # tasks only for markers in the model; arm-body and --off markers switched off below
            tdir = os.path.join(wsub, re.sub(r"[^\w.-]", "_", name))
            os.makedirs(tdir, exist_ok=True)
            xml = ik_setup(model, trc, t_first, t_last, ik_mk, weights, os.path.join(tdir, "opensim_ik.mot"), tdir)
            for m in ik_mk:   # prep_trials.ik_setup switches off Nitschke arm names; do it by body here
                if ARM_BODIES.match(mk_body[m]) or m in off:
                    xml = xml.replace(f'<IKMarkerTask name="{m}">\n\t\t\t\t\t<apply>true</apply>',
                                      f'<IKMarkerTask name="{m}">\n\t\t\t\t\t<apply>false</apply>')
            open(os.path.join(tdir, "ik_setup.xml"), "w").write(xml)
            grf = os.path.join(sdir, tr["file"].replace(".trc", "_grf.mot"))
            out_rows.append({"participant": sid, "trial": name, "dir": tdir, "model": model, "trc": trc,
                             "grf": grf if os.path.isfile(grf) else "", "t0": f"{t_first:.5f}", "t1": f"{t_last:.5f}",
                             "ic1": "", "ic2": "", "mass_kg": info.get("mass_kg", "")})
            n_sub += 1
        print(f"{sid}: {n_sub} trials, {len(mk_body)} model markers, unlocked {unlocked or 'none (already free)'}")

    if not out_rows:
        raise SystemExit("no trials matched")
    with open(os.path.join(work, "trials.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"\n{len(out_rows)} trials from {len({r['participant'] for r in out_rows})} subjects prepared in {work} (trials.csv)")
    if skipped:
        print(f"skipped {len(skipped)} junk trials: " + "; ".join(skipped[:10]))


if __name__ == "__main__":
    main()
