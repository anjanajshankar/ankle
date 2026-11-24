"""Protocol step 6, part 1: the hypothesis variables X, M, Y per trial and pipeline, exactly as defined in
claude/hypothesis_variable_definitions.md (final, 23 Nov 2026; processing locked at tag lock-2026-11-24).

Per trial (analysed trials only: rule-5 exclusions skipped) and pipeline (A = opensim_ik.mot filtered at --fc,
B = spline result, no extra filter):
  1. OpenSim AnalyzeTool + BodyKinematics (opensim-cmd) -> every body's position and orientation in ground;
     body origins = COM position - R * mass_center (from the model file).
  2. X at IC, plant side (valgus positive), 180 deg - angle HKA in a plane:
       H = femur origin; K, A = knee and ankle centres:
         marker version: midpoints of the medial/lateral epicondyle and malleolus markers (Nitschke, Soccer COD)
         model version : tibia and talus origins (all datasets)
       planes: pelvis frontal plane (primary; normal = pelvis X axis) and thigh plane (secondary; contains
       H->K and the knee's medial-lateral axis). Sign + when K lies medial of the line H->A.
  3. M = mean over W (IC .. IC+100 ms) of the lean of the torso Y axis toward the plant side, measured from
     vertical in the plane perpendicular to the approach heading h (pelvis-origin velocity, IC-100 ms .. IC).
  4. Y = mean over W of the plant-side subtalar_angle ID moment / (body weight x height) x 100 (%BW*Ht),
     sign factor +1 (positive = internal inversion; pre_lock_check.py). Secondary: ankle_angle moment, same
     scaling. Not computed for datasets without GRF (ACL) or without id_<pipeline>.sto.
  Covariates: approach speed (|h| before normalising), cut angle (from the trial name; 90 for Nitschke/ACL).

usage (from the ankle folder):
  python3 src/kin/hypvars.py <work> --dataset NAME --fc HZ [--only TAG] [--heights FILE.csv] [--check]
     --check   : print participants' mass and height (and where they came from) and stop
     --only TAG: one trial, with every intermediate quantity printed (for checking)
  -> <work>/hypvars.csv (one row per trial and pipeline)
Height: from --heights (csv: participant,height_m) if given; else trials.csv column height_m; else the
dataset's metaInfo.csv (Nitschke) or subject.json (AddBiomechanics). Missing height -> Y left empty.
"""
import csv, glob, json, os, re, subprocess, sys
import xml.etree.ElementTree as ET
import numpy as np

G = 9.81
W, APPROACH = 0.10, 0.10
KNEE = {"r": [("RKneM", "RKneL"), ("epi_med_right", "epi_lat_right")],
        "l": [("LKneM", "LKneL"), ("epi_med_left", "epi_lat_left")]}
ANKLE = {"r": [("RAnkM", "RAnkL"), ("mal_med_right", "mal_lat_right")],
         "l": [("LAnkM", "LAnkL"), ("mal_med_left", "mal_lat_left")]}


# ---------------------------------------------------------------- input helpers
def read_table(path):
    lines = open(path).readlines()
    k = next(i for i, l in enumerate(lines) if l.strip().lower() == "endheader") + 1
    cols = lines[k].split()
    return cols, np.loadtxt([l for l in lines[k + 1:] if l.strip()], ndmin=2)


def read_trc(path):
    L = open(path).read().split("\n")
    names = [n for n in L[3].split("\t")[2:] if n.strip()]
    k = next(i for i, l in enumerate(L) if "X1\t" in l)
    rows = [r.split("\t") for r in L[k + 1:] if r.strip()]
    def val(x):
        try: return float(x)
        except ValueError: return np.nan
    t = np.array([float(r[1]) for r in rows])
    D = np.array([[val(x) for x in r[2:2 + 3 * len(names)]] for r in rows])
    units = L[2].split("\t")
    if not any(u.strip() == "m" for u in units):
        D = D / 1000.0
    return names, t, D


def rot_xyz(a):
    x, y, z = a
    Rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


def mass_centers(model):
    out = {}
    for b in ET.parse(model).getroot().iter("Body"):
        mc = b.findtext("mass_center")
        if mc: out[b.get("name")] = np.array([float(v) for v in mc.split()])
    return out


def unit(v): return v / np.linalg.norm(v)


def interp(t, y, tq):
    return np.array([np.interp(tq, t, y[:, k]) for k in range(y.shape[1])]).T


# ---------------------------------------------------------------- body kinematics via opensim-cmd
def body_kinematics(model, mot, fc, t0, t1, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    sto = os.path.join(out_dir, f"{name}_BodyKinematics_pos_global.sto")
    if not os.path.exists(sto):
        setup = os.path.join(out_dir, f"{name}_setup.xml")
        open(setup, "w").write(
            f'<?xml version="1.0" encoding="UTF-8" ?>\n<OpenSimDocument Version="40000">\n'
            f'<AnalyzeTool name="{name}"><model_file>{os.path.abspath(model)}</model_file>'
            f'<results_directory>{os.path.abspath(out_dir)}</results_directory>'
            f'<initial_time>{t0:.5f}</initial_time><final_time>{t1:.5f}</final_time>'
            f'<coordinates_file>{os.path.abspath(mot)}</coordinates_file>'
            f'<lowpass_cutoff_frequency_for_coordinates>{fc}</lowpass_cutoff_frequency_for_coordinates>'
            f'<AnalysisSet><objects><BodyKinematics name="BodyKinematics"><on>true</on><step_interval>1</step_interval>'
            f'<in_degrees>true</in_degrees><bodies> all</bodies>'
            f'<express_results_in_body_local_frame>false</express_results_in_body_local_frame>'
            f'</BodyKinematics></objects></AnalysisSet></AnalyzeTool>\n</OpenSimDocument>\n')
        with open(os.path.join(out_dir, f"{name}_log.txt"), "w") as log:
            if subprocess.run(["opensim-cmd", "run-tool", setup], stdout=log, stderr=subprocess.STDOUT).returncode:
                raise RuntimeError(f"AnalyzeTool failed, see {out_dir}/{name}_log.txt")
    cols, d = read_table(sto)
    mc = mass_centers(model)
    t = d[:, 0]
    def frame(body):
        """origin positions (n x 3) and rotation matrices (n x 3 x 3) of a body in ground"""
        p = d[:, [cols.index(f"{body}_{a}") for a in "XYZ"]]
        ang = np.radians(d[:, [cols.index(f"{body}_O{a}") for a in "xyz"]])
        R = np.array([rot_xyz(a) for a in ang])
        return p - np.einsum("nij,j->ni", R, mc.get(body, np.zeros(3))), R
    return t, frame


# ---------------------------------------------------------------- the variables
def frontal_angle(H, K, A, n, medial):
    """180 - angle HKA after projecting into the plane with normal n; + when K is medial of line H->A."""
    P = lambda v: v - np.dot(v, n) * n
    h, k, a = P(H), P(K), P(A)
    u, w = h - k, a - k
    ang = np.degrees(np.arccos(np.clip(np.dot(u, w) / (np.linalg.norm(u) * np.linalg.norm(w)), -1, 1)))
    ha = a - h
    foot = h + np.dot(k - h, ha) / np.dot(ha, ha) * ha          # closest point to K on the line H->A
    off = k - foot
    m = P(medial); m = m / np.linalg.norm(m)
    return (180.0 - ang) * (1.0 if np.dot(off, m) >= 0 else -1.0)


def marker_points(names, tm, M, side, ic):
    def mid(pairs):
        for a, b in pairs:
            if a in names and b in names:
                pa = interp(tm, M[:, 3 * names.index(a):3 * names.index(a) + 3], ic)
                pb = interp(tm, M[:, 3 * names.index(b):3 * names.index(b) + 3], ic)
                return (pa + pb) / 2, pb - pa   # midpoint, medial->lateral vector
        return None, None
    K, e = mid(KNEE[side]); A, _ = mid(ANKLE[side])
    return K, A, e


def heights_for(rows, work, override):
    """participant -> (height_m, source)"""
    out = {}
    if override:
        for r in csv.DictReader(open(override)):
            out[r["participant"]] = (float(r["height_m"]), override)
    for r in rows:
        p = r["participant"]
        if p in out: continue
        if r.get("height_m"):
            out[p] = (float(r["height_m"]), "trials.csv"); continue
        found = None
        # Nitschke: metaInfo.csv near the model; AddBiomechanics: subject.json in assets/<dataset>
        base = os.path.dirname(os.path.abspath(r["model"]))
        for f in glob.glob(os.path.join(base, "*eta*nfo*.csv")) + glob.glob(os.path.join(os.path.dirname(base), "*eta*nfo*.csv")):
            txt = open(f).read()
            dl = ";" if txt.count(";") > txt.count(",") else ","
            for row in csv.DictReader(txt.splitlines(), delimiter=dl):
                ik = [k for k in row if k and ("participant" in k.lower() or "subject" in k.lower())]
                if ik and row[ik[0]].strip() != p:
                    continue
                hk = [k for k in row if k and "height" in k.lower()]
                if hk and row[hk[0]]:
                    v = float(row[hk[0]].replace(",", ".")); found = (v / 100 if v > 3 else v, f); break
            if found: break
        if not found:
            folder = os.path.basename(base)
            for f in glob.glob("assets/**/subject.json", recursive=True) + glob.glob(os.path.join(work, "**/subject.json"), recursive=True):
                if folder in f or p in f.split(os.sep):
                    j = json.load(open(f))
                    hk = [k for k in j if "height" in k.lower()]
                    if hk:
                        v = float(j[hk[0]]); found = (v / 100 if v > 3 else v, f); break
        if found: out[p] = found
    return out


def events(work, dataset):
    rows = list(csv.DictReader(open(os.path.join(work, "trials.csv"))))
    if os.path.exists(os.path.join(work, "trials_grf.csv")):
        g = {(r["participant"], r["trial"]): r for r in csv.DictReader(open(os.path.join(work, "trials_grf.csv")))}
        return [(r, g[(r["participant"], r["trial"])]["plant_foot"], float(g[(r["participant"], r["trial"])]["ic"]))
                for r in rows if (r["participant"], r["trial"]) in g]
    if os.path.exists(os.path.join(work, "trials_events.csv")):
        e = {(r["participant"], r["trial"]): r for r in csv.DictReader(open(os.path.join(work, "trials_events.csv")))}
        return [(r, e[k]["plant_foot"], float(e[k]["ic"])) for r in rows
                for k in [(r["participant"], r["trial"])] if k in e and e[k]["ic"]]
    return [(r, "r", float(r["ic1"])) for r in rows]


def tag_of(r):
    p = r["participant"]
    return f"{p if p.startswith('P') else 'P' + p}_{r['trial']}"


def cut_angle(trial):
    m = re.search(r"_(45|90|135|180)_", trial)
    return int(m.group(1)) if m else 90


def main():
    a = sys.argv[1:]
    if len(a) < 1 or "--dataset" not in a or "--fc" not in a:
        print(__doc__); sys.exit(1)
    work = a[0]
    dataset = a[a.index("--dataset") + 1]
    fc = float(a[a.index("--fc") + 1])
    only = a[a.index("--only") + 1] if "--only" in a else None
    hfile = a[a.index("--heights") + 1] if "--heights" in a else None
    ev = events(work, dataset)
    heights = heights_for([r for r, _, _ in ev], work, hfile)
    if "--check" in a:
        seen = {}
        for r, _, _ in ev: seen.setdefault(r["participant"], r["mass_kg"])
        for p, m in sorted(seen.items()):
            h = heights.get(p)
            print(f"{p:12s} mass {float(m):6.1f} kg   height {('%.3f m' % h[0]) if h else 'MISSING':10s} {h[1] if h else ''}")
        return
    excl = set()
    if os.path.exists(os.path.join(work, "excluded_trials.csv")):
        excl = {x["trial"] for x in csv.DictReader(open(os.path.join(work, "excluded_trials.csv")))}
    out = []
    for r, side, ic in ev:
        tag = tag_of(r)
        if only and tag != only: continue
        keys = {tag, f"{r['participant']}_{r['trial']}", r["trial"], os.path.basename(os.path.normpath(r["dir"]))}
        if keys & excl: continue
        d, model = r["dir"], r["model"]
        t0, t1, mass = float(r["t0"]), float(r["t1"]), float(r["mass_kg"])
        h = heights.get(r["participant"], (np.nan, ""))[0]
        names, tm, M = read_trc(r["trc"])
        Km, Am, e_mk = marker_points(names, tm, M, side, ic)
        B_mot = os.path.join(d, "spline_ik_witharms.mot")
        if not os.path.exists(B_mot): B_mot = os.path.join(d, "spline_ik.mot")
        for pipe, mot, cut in (("A", os.path.join(d, "opensim_ik.mot"), fc), ("B", B_mot, -1)):
            if not os.path.exists(mot): print(f"{tag} {pipe}: no IK result"); continue
            t, frame = body_kinematics(model, mot, cut, t0, t1, os.path.join(d, "hypvars"), f"bk_{pipe}")
            pel_p, pel_R = frame("pelvis"); fem_p, fem_R = frame(f"femur_{side}")
            tib_p, _ = frame(f"tibia_{side}"); tal_p, _ = frame(f"talus_{side}"); _, tor_R = frame("torso")
            at = lambda y: interp(t, y.reshape(len(t), -1), ic).reshape(y.shape[1:])
            H, Kmod, Amod = at(fem_p), at(tib_p), at(tal_p)
            Rp, Rf = at(pel_R), at(fem_R)
            n_pel = unit(Rp[:, 0])
            med_pel = -Rp[:, 2] if side == "r" else Rp[:, 2]
            med_fem = -Rf[:, 2] if side == "r" else Rf[:, 2]
            res = {"X_model": frontal_angle(H, Kmod, Amod, n_pel, med_pel),
                   "Xthigh_model": frontal_angle(H, Kmod, Amod, unit(np.cross(Kmod - H, Rf[:, 2])), med_fem)}
            if Km is not None and Am is not None:
                res["X_marker"] = frontal_angle(H, Km, Am, n_pel, med_pel)
                res["Xthigh_marker"] = frontal_angle(H, Km, Am, unit(np.cross(Km - H, e_mk)), -e_mk)
            # approach heading and speed
            p_ic, p_0 = interp(t, pel_p, ic), interp(t, pel_p, ic - APPROACH)
            v = (p_ic - p_0) / APPROACH; v[1] = 0.0
            speed = float(np.linalg.norm(v)); hd = v / speed
            Yup = np.array([0.0, 1.0, 0.0])
            r_side = np.cross(hd, Yup) if side == "r" else np.cross(Yup, hd)
            sel = (t >= ic) & (t <= ic + W)
            tY = tor_R[sel][:, :, 1]
            lean = np.degrees(np.arctan2(tY @ r_side, tY @ Yup))
            res["M"] = float(np.mean(lean))
            # Y from ID
            sto = os.path.join(d, f"id_{pipe}.sto")
            if os.path.exists(sto) and np.isfinite(h):
                cols, dd = read_table(sto)
                s2 = (dd[:, 0] >= ic) & (dd[:, 0] <= ic + W)
                scale = 100.0 / (mass * G * h)
                res["Y"] = float(np.mean(dd[s2, cols.index(f"subtalar_angle_{side}_moment")])) * scale * (+1)
                res["Y_ankle"] = float(np.mean(dd[s2, cols.index(f"ankle_angle_{side}_moment")])) * scale
            row = {"dataset": dataset, "participant": r["participant"], "trial": tag, "pipeline": pipe, "plant_foot": side,
                   "ic": f"{ic:.4f}", "cut_deg": cut_angle(r["trial"]), "approach_speed": f"{speed:.3f}"}
            for k in ("X_marker", "X_model", "Xthigh_marker", "Xthigh_model", "M", "Y", "Y_ankle"):
                row[k] = f"{res[k]:.4f}" if k in res else ""
            out.append(row)
            if only:
                print(f"\n{tag} pipeline {pipe} (plant {side}, IC {ic:.3f} s)")
                print(f"  hip H {np.round(H, 3)}  knee K model {np.round(Kmod, 3)} marker {None if Km is None else np.round(Km, 3)}")
                print(f"  ankle A model {np.round(Amod, 3)} marker {None if Am is None else np.round(Am, 3)}")
                print(f"  pelvis X axis {np.round(n_pel, 3)}  heading {np.round(hd, 3)}  approach speed {speed:.2f} m/s")
                print(f"  thigh length {np.linalg.norm(Kmod - H):.3f} m, shank {np.linalg.norm(Amod - Kmod):.3f} m")
                if Km is not None:
                    print(f"  check: model knee centre to marker knee centre {1000 * np.linalg.norm(Kmod - Km):.0f} mm, "
                          f"model ankle centre to marker ankle centre {1000 * np.linalg.norm(Amod - Am):.0f} mm "
                          f"(expect roughly 0-30 mm; much larger = body orientations misread)")
                print(f"  torso Y axis at IC {np.round(at(tor_R)[:, 1], 3)} (should be close to vertical, 0 1 0)")
                print(f"  lean over W: start {lean[0]:.1f}, end {lean[-1]:.1f} deg ({sel.sum()} frames)")
                print("  " + "  ".join(f"{k} {row[k]}" for k in ("X_marker", "X_model", "Xthigh_marker", "Xthigh_model", "M", "Y", "Y_ankle")))
        print(f"{tag} done", flush=True)
    if out and not only:
        path = os.path.join(work, "hypvars.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
        print(f"\n{len(out)} rows -> {path}")


if __name__ == "__main__":
    main()
