"""Checks before the lock (hypothesis-blind: model files and event times only).

1. Subtalar sign, per model and side, from the model file: the subtalar_angle_<s> rotation axis, expressed
   in the parent (talus) body frame (joint-frame orientation applied). With X anterior, Y up, Z right, a
   positive rotation is INVERSION when the axis points forward (X > 0) for the right foot and backward
   (X < 0) for the left. ID reports the internal generalized force in the coordinate's positive direction,
   so Y's sign factor = +1 if positive = inversion, -1 otherwise.
2. Windows, per trial: W = IC .. IC + 100 ms must lie inside the ID range (IK range shortened by 0.05 s at
   each end) and the approach window IC - 100 ms .. IC inside the IK range. Trials excluded by rule 5 are
   left out. Counts per dataset; failing trials listed.

usage (from the ankle folder):
  python3 src/kin/pre_lock_check.py assets/Nitschke22/work_final work/SoccerCOD work/Sidestep
Writes <work>/pre_lock_check.csv per dataset (trial, plant_foot, ic, window_ok, reason, y_sign).
"""
import csv, os, sys
import xml.etree.ElementTree as ET
import numpy as np

W, APPROACH, MARGIN = 0.10, 0.10, 0.05


def rot_xyz(a):
    """Body-fixed X-Y-Z rotation (OpenSim orientation)."""
    x, y, z = a
    Rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


def subtalar_sign(model, side):
    """CustomJoint: the TransformAxis driven by subtalar_angle_<side>. PinJoint (Rajagopal 2015): the Z axis of
    the joint's parent frame. Either way the axis is returned in the parent (talus) body frame."""
    root = ET.parse(model).getroot()
    coord = f"subtalar_angle_{side}"
    for joint in root.iter():
        if not joint.tag.endswith("Joint"):
            continue
        axes = [ta for ta in joint.iter("TransformAxis") if (ta.findtext("coordinates") or "").strip() == coord]
        pin = any(c.get("name") == coord for c in joint.iter("Coordinate")) and joint.tag == "PinJoint"
        if not axes and not pin:
            continue
        R = np.eye(3)
        pf = (joint.findtext("socket_parent_frame") or "").strip()
        for f in joint.iter("PhysicalOffsetFrame"):     # OpenSim 4 format
            if f.get("name") == pf and f.findtext("orientation"):
                R = rot_xyz([float(v) for v in f.findtext("orientation").split()])
        if joint.findtext("orientation_in_parent"):     # OpenSim 3 format
            R = rot_xyz([float(v) for v in joint.findtext("orientation_in_parent").split()])
        ax = np.array([float(v) for v in axes[0].findtext("axis").split()]) if axes else np.array([0.0, 0.0, 1.0])
        a = R @ ax
        inv = a[0] > 0 if side == "r" else a[0] < 0
        return (1 if inv else -1), a
    return 0, None


def events(work):
    """(row, tag, plant_foot, ic) for every trial with a reference IC."""
    if os.path.exists(os.path.join(work, "trials_grf.csv")):
        rows = list(csv.DictReader(open(os.path.join(work, "trials_grf.csv"))))
        out = [(r, r["plant_foot"], r["ic"]) for r in rows]
    else:
        rows = list(csv.DictReader(open(os.path.join(work, "trials.csv"))))
        out = [(r, "r", r["ic1"]) for r in rows]
    res = []
    for r, f, ic in out:
        p = r["participant"]
        tag = f"{p if p.startswith('P') else 'P' + p}_{r['trial']}"
        res.append((r, tag, f, float(ic) if ic not in ("", None) else np.nan))
    return res


def main(works):
    signs = {}
    for w in works:
        excl = set()
        ex = os.path.join(w, "excluded_trials.csv")
        if os.path.exists(ex):
            excl = {x["trial"] for x in csv.DictReader(open(ex))}
        else:
            print(f"{w}: no excluded_trials.csv (rule 5 not applied yet?)")
        rows, n_ok, n_all, fails = [], 0, 0, []
        for r, tag, foot, ic in events(w):
            keys = {tag, f"{r['participant']}_{r['trial']}", r["trial"], os.path.basename(os.path.normpath(r["dir"]))}
            if keys & excl:
                continue
            n_all += 1
            key = (r["model"], foot)
            if key not in signs:
                signs[key] = subtalar_sign(r["model"], foot)
            t0, t1 = float(r["t0"]), float(r["t1"])
            reason = []
            if not np.isfinite(ic): reason.append("no IC")
            else:
                if ic - APPROACH < t0 - 0.001: reason.append(f"approach starts {1000 * (t0 - (ic - APPROACH)):.0f} ms before data")
                if ic < t0 + MARGIN - 0.001: reason.append("IC before ID range")
                if ic + W > t1 - MARGIN + 0.001: reason.append(f"W ends {1000 * (ic + W - (t1 - MARGIN)):.0f} ms after ID range")
            ok = not reason
            n_ok += ok
            if not ok: fails.append((tag, "; ".join(reason)))
            rows.append([tag, foot, f"{ic:.4f}", int(ok), "; ".join(reason), signs[key][0]])
        with open(os.path.join(w, "pre_lock_check.csv"), "w", newline="") as f:
            cw = csv.writer(f); cw.writerow(["trial", "plant_foot", "ic", "window_ok", "reason", "y_sign"]); cw.writerows(rows)
        print(f"\n{w}: {n_ok} of {n_all} analysed trials have both windows inside the data")
        for t, why in fails[:15]:
            print(f"   {t}: {why}")
        if len(fails) > 15: print(f"   ... {len(fails) - 15} more in {w}/pre_lock_check.csv")
    print("\nsubtalar axis per model and side (in the talus frame; X anterior, Y up, Z right):")
    for (m, s), (sg, a) in sorted(signs.items()):
        what = "NOT FOUND" if sg == 0 else ("positive = inversion" if sg > 0 else "positive = EVERSION -> Y sign flipped")
        ax = "" if a is None else f"axis ({a[0]:+.3f}, {a[1]:+.3f}, {a[2]:+.3f})"
        print(f"   {s}  {ax}  {what}   {m}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    main(sys.argv[1:])
