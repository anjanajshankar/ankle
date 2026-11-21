
#!/usr/bin/env python3
"""
prep_addbio.py — protocol step 1 (prepare) for AddBiomechanics .b3d datasets.

For every .b3d file found under SRC:
  * subject id from the folder path (e.g. P04, S1, Test1); duplicate uploads of
    the same subject (identical trial-name set) are skipped and reported
  * every trial -> OUT/<subject>/<trial>.trc  (all marker channels except obvious
    computed ones, see --keep / --drop), plus <trial>_grf.mot if force plates exist
  * OUT/<subject>/subject.json  (mass, height, sex, rate, trials, gaps, plates)
  * OUT/<subject>/addbio_pass<k>.osim  (AddBiomechanics' scaled model text, if
    the .b3d stores it) — only for the model decision, not used automatically
Across the dataset:
  * OUT/inventory.csv       one row per trial
  * OUT/markers_report.csv  one row per marker channel: how often present,
    name-based flag, and the marker it moves rigidly with (min SD of distance).
    A channel rigid (<0.5 mm) to a marker on the same segment may be virtual
    (rebuilt from a cluster). Review this before fixing the IK marker set.

Hypothesis-blind: only marker geometry, timing and force-plate loading are read.

Usage (from ankle/tools):
  python3 prep_addbio.py kept_cuts/soccer_cod  --dataset soccer_cod --out ../assets/SoccerCOD
  python3 prep_addbio.py kept_cuts/acl         --dataset acl        --out ../assets/ACL
  python3 prep_addbio.py kept_cuts/unplanned_sidestep --dataset sidestep --out ../assets/Sidestep
  options: --keep C7,T10,...  (exact marker list; overrides the name filter)
           --drop REGEX       (extra channels to drop)
           --no-osim          (skip model export)
"""
import argparse, csv, hashlib, json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_b3d import write_trc, write_mot  # same folder

# Channels that are model outputs, not markers (Vicon Plug-in Gait / UWA model,
# Visual3D exports). Case-insensitive search in the name.
COMPUTED = re.compile(
    r"(Vel|Acc|COM|Angle|Force|Moment|Power|JC|Position|Origin|Offset|Rel[A-Z]|"
    r"^O[A-Z]|^[XYZ][A-Z]|^[XYZ]global|^av|^dr[XYZ]|CalRig|Midline|FootSide|"
    r"Progression|^new|^Point\d|^x\d\d$|^Mid(HJC|Pel))")


def subject_id(path):
    parts = [p for p in path.split(os.sep) if p]
    # skip hash folders and the X.b3d wrapper folder
    for p in reversed(parts[:-1]):
        if re.fullmatch(r"[0-9a-f]{40,}(\.b3d)?", p):
            continue
        return re.sub(r"[^\w.-]", "_", p)
    return "subject"


def find_b3d(src):
    out = []
    for root, _, files in os.walk(src):
        for f in files:
            if f.lower().endswith(".b3d"):
                out.append(os.path.join(root, f))
    return sorted(out)


def call(obj, name, *a, default=None):
    fn = getattr(obj, name, None)
    if fn is None:
        return default
    try:
        return fn(*a)
    except Exception:
        return default


def safe(name):
    return re.sub(r"[^\w.-]", "_", name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep", default="")
    ap.add_argument("--drop", default="")
    ap.add_argument("--no-osim", action="store_true")
    ap.add_argument("--rigid-mm", type=float, default=0.5)
    a = ap.parse_args()
    try:
        import nimblephysics as nimble
    except ImportError:
        sys.exit("pip install nimblephysics")

    keep = [m.strip() for m in a.keep.split(",") if m.strip()]
    drop = re.compile(a.drop) if a.drop else None
    os.makedirs(a.out, exist_ok=True)
    files = find_b3d(a.src)
    print(f"{len(files)} .b3d files under {a.src}")

    inv, seen_sig = [], {}
    mk_present = {}      # marker -> [frames present, frames total]
    mk_rigid = {}        # marker -> (best sd mm, partner)
    for path in files:
        sid = subject_id(path)
        s = nimble.biomechanics.SubjectOnDisk(path)
        nt = s.getNumTrials()
        names = [call(s, "getTrialName", t, default=f"trial{t}") for t in range(nt)]
        sig = hashlib.md5("|".join(sorted(names)).encode()).hexdigest()
        if sig in seen_sig:
            print(f"  {sid}: DUPLICATE of {seen_sig[sig]} ({os.path.basename(path)[:12]}…), skipped")
            continue
        base_sid, k = sid, 2
        while sid in seen_sig.values():          # same folder name, different data
            sid = f"{base_sid}_{k}"; k += 1
        seen_sig[sig] = sid
        sdir = os.path.join(a.out, sid)
        os.makedirs(sdir, exist_ok=True)
        mass = call(s, "getMassKg", default=float("nan"))
        info = {"dataset": a.dataset, "subject": sid, "source": path,
                "mass_kg": mass, "height_m": call(s, "getHeightM"),
                "sex": call(s, "getBiologicalSex"), "trials": []}

        if not a.no_osim:
            for p in range(call(s, "getNumProcessingPasses", default=0) or 0):
                txt = call(s, "getOpensimFileText", p)
                if txt:
                    open(os.path.join(sdir, f"addbio_pass{p}.osim"), "w").write(txt)

        for t in range(nt):
            dt = s.getTrialTimestep(t)
            F = s.readFrames(t, 0, s.getTrialLength(t), True, True)
            nf = len(F)
            obs = [dict(fr.markerObservations) for fr in F]
            present = sorted(set().union(*[o.keys() for o in obs])) if obs else []
            if keep:
                mk = [m for m in keep if m in present]
            else:
                mk = [m for m in present if m and not COMPUTED.search(m) and not (drop and drop.search(m))]
            X = np.full((nf, len(mk), 3), np.nan)
            for i, o in enumerate(obs):
                for j, m in enumerate(mk):
                    if m in o:
                        X[i, j] = np.asarray(o[m], float)
            time = np.arange(nf) * dt
            stem = safe(names[t])
            write_trc(os.path.join(sdir, stem + ".trc"), mk, time, X, 1 / dt)

            # marker stats (all channels, for the report)
            for m in present:
                c = mk_present.setdefault(m, [0, 0])
                c[0] += sum(1 for o in obs if m in o); c[1] += nf
            # rigidity: SD of distance to every other kept marker (sampled frames)
            idx = np.linspace(0, nf - 1, min(nf, 60)).astype(int)
            P = X[idx]
            for j, m in enumerate(mk):
                best = mk_rigid.get(m, (np.inf, ""))
                for q, m2 in enumerate(mk):
                    if q == j:
                        continue
                    d = np.linalg.norm(P[:, j] - P[:, q], axis=1)
                    d = d[~np.isnan(d)]
                    if len(d) >= 10:
                        sd = 1000 * float(np.std(d))
                        if sd < best[0]:
                            best = (sd, m2)
                mk_rigid[m] = best

            # force plates
            plates = []
            np_ = len(F[0].rawForcePlateForces) if nf and len(F[0].rawForcePlateForces) else 0
            if np_:
                cols, rows = [], []
                for i in range(np_):
                    cols += [f"plate{i+1}_force_v{c}" for c in "xyz"]
                    cols += [f"plate{i+1}_force_p{c}" for c in "xyz"]
                    cols += [f"plate{i+1}_torque_{c}" for c in "xyz"]
                for fr in F:
                    r = []
                    for i in range(np_):
                        r += list(np.asarray(fr.rawForcePlateForces[i], float))
                        r += list(np.asarray(fr.rawForcePlateCenterOfPressures[i], float))
                        r += list(np.asarray(fr.rawForcePlateTorques[i], float))
                    rows.append(r)
                rows = np.array(rows)
                if np.nanmax(np.abs(rows[:, [9 * i + c for i in range(np_) for c in range(3)]])) > 20:
                    write_mot(os.path.join(sdir, stem + "_grf.mot"), time, cols, rows)
                for i in range(np_):
                    fz = np.linalg.norm(rows[:, 9 * i:9 * i + 3], axis=1)
                    on = np.where(fz > 20)[0]
                    if len(on):
                        plates.append({"plate": i + 1, "on_s": round(float(time[on[0]]), 4),
                                       "off_s": round(float(time[on[-1]]), 4),
                                       "peak_BW": round(float(fz.max() / (mass * 9.81)), 2)})
            gaps = {m: int(np.isnan(X[:, j, 0]).sum()) for j, m in enumerate(mk) if np.isnan(X[:, j, 0]).any()}
            tr = {"trial": names[t], "file": stem + ".trc", "frames": nf, "rate_hz": round(1 / dt, 3),
                  "duration_s": round(nf * dt, 3), "n_markers": len(mk), "gap_frames": gaps,
                  "loaded_plates": plates}
            info["trials"].append(tr)
            inv.append({"dataset": a.dataset, "subject": sid, "trial": names[t], "frames": nf,
                        "rate_hz": tr["rate_hz"], "duration_s": tr["duration_s"], "n_markers": len(mk),
                        "markers_with_gaps": len(gaps), "max_gap_frames": max(gaps.values()) if gaps else 0,
                        "loaded_plates": ";".join(f"p{p['plate']}:{p['on_s']}-{p['off_s']}s:{p['peak_BW']}BW" for p in plates),
                        "mass_kg": mass, "height_m": info["height_m"]})
        json.dump(info, open(os.path.join(sdir, "subject.json"), "w"), indent=1)
        print(f"  {sid}: {nt} trials, {len(mk)} markers, mass {mass} kg")

    with open(os.path.join(a.out, "inventory.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(inv[0].keys())); w.writeheader(); w.writerows(inv)
    with open(os.path.join(a.out, "markers_report.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["marker", "present_frac", "name_flag_computed", "in_trc", "rigid_partner", "min_sd_mm", "check"])
        for m in sorted(mk_present):
            p = mk_present[m][0] / max(mk_present[m][1], 1)
            sd, partner = mk_rigid.get(m, (np.nan, ""))
            flag = bool(COMPUTED.search(m)) if m else True
            in_trc = m in mk_rigid
            chk = "possibly virtual" if in_trc and sd < a.rigid_mm else ""
            w.writerow([m, round(p, 3), int(flag), int(in_trc), partner,
                        "" if not np.isfinite(sd) else round(sd, 2), chk])
    subs = sorted({r["subject"] for r in inv})
    print(f"\n{a.dataset}: {len(subs)} subjects, {len(inv)} trials -> {a.out}/inventory.csv")
    print(f"markers report -> {a.out}/markers_report.csv")


if __name__ == "__main__":
    main()
