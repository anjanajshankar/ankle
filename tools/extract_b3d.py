
#!/usr/bin/env python3
"""
extract_b3d.py — write one AddBiomechanics .b3d trial as OpenSim files:
  <out>.trc        marker trajectories (metres), real markers only by default
  <out>_grf.mot    raw force-plate data, one block of columns per plate
                   (plate<i>_force_vx/vy/vz, plate<i>_force_px/py/pz, plate<i>_torque_x/y/z)
  <out>_info.txt   subject mass/height, trial name, rate, plates, markers, gaps

Usage:
  python3 extract_b3d.py FILE.b3d --out kept_cuts/SS_R_03            # trial 0, UWA set
  python3 extract_b3d.py FILE.b3d --trial 2 --markers C7,T10,RASI ...
  python3 extract_b3d.py FILE.b3d --list                              # show trials and channels
  python3 extract_b3d.py FILE.b3d --all-markers --out ...             # every channel

Notes
- Coordinates are as stored in the .b3d (AddBiomechanics lab frame, Y up), metres.
- Missing markers are written as empty fields (OpenSim treats them as gaps).
- GRF is per plate, not per foot: check in _info.txt which plate is loaded during
  the cut stance, and map it to the foot in your ExternalLoads file.
"""
import argparse, os, sys
import numpy as np

# UWA marker set, checked on "Unplanned Sidestep/Test1": condyles (MFC/LFC),
# KHel1/2, FTML (15 Nov 2026) and malleoli MMAL/LMAL (20 Nov 2026, rigid to each
# other at 0.00 mm with marker_rigidity.py) are virtual and are excluded.
UWA_REAL = ("C7 T10 CLAV STRN LASI RASI LPSI RPSI SACR "
            "LTH1 LTH2 LTH3 LTH4 RTH1 RTH2 RTH3 RTH4 "
            "LTB1 LTB2 LTB3 LTB4 RTB1 RTB2 RTB3 RTB4 "
            "LCAL LMT1 LMT5 RCAL RMT1 RMT5").split()


def load(path):
    try:
        import nimblephysics as nimble
    except ImportError:
        sys.exit("pip install nimblephysics")
    if os.path.isdir(path):  # Drive layout: X.b3d/ folder holding X.b3d
        inner = os.path.join(path, os.path.basename(path))
        path = inner if os.path.isfile(inner) else path
    return nimble.biomechanics.SubjectOnDisk(path)


def frames(s, t):
    n = s.getTrialLength(t)
    return s.readFrames(t, 0, n, True, True)


def write_trc(fn, names, t, X, rate):
    """X: (frames, markers, 3) in metres, NaN for gaps."""
    nf = len(t)
    with open(fn, "w") as f:
        f.write(f"PathFileType\t4\t(X/Y/Z)\t{os.path.basename(fn)}\n")
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write(f"{rate:.4f}\t{rate:.4f}\t{nf}\t{len(names)}\tm\t{rate:.4f}\t1\t{nf}\n")
        f.write("Frame#\tTime\t" + "\t\t\t".join(names) + "\t\t\n")
        f.write("\t\t" + "\t".join(f"X{i+1}\tY{i+1}\tZ{i+1}" for i in range(len(names))) + "\n\n")
        for k in range(nf):
            vals = []
            for m in range(len(names)):
                p = X[k, m]
                vals += ["" if np.isnan(v) else f"{v:.6f}" for v in p]
            f.write(f"{k+1}\t{t[k]:.6f}\t" + "\t".join(vals) + "\n")


def write_mot(fn, t, cols, data):
    with open(fn, "w") as f:
        f.write(f"{os.path.basename(fn)}\nversion=1\nnRows={len(t)}\nnColumns={len(cols)+1}\n"
                "inDegrees=yes\nendheader\n")
        f.write("time\t" + "\t".join(cols) + "\n")
        for k in range(len(t)):
            f.write(f"{t[k]:.6f}\t" + "\t".join(f"{v:.6f}" for v in data[k]) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("b3d")
    ap.add_argument("--trial", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--markers", default="", help="comma list (default: UWA real-marker set)")
    ap.add_argument("--all-markers", action="store_true")
    ap.add_argument("--list", action="store_true", help="list trials and marker channels, then exit")
    a = ap.parse_args()

    s = load(a.b3d)
    if a.list:
        for t in range(s.getNumTrials()):
            print(t, s.getTrialName(t), s.getTrialLength(t), "frames,", s.getTrialTimestep(t), "s")
        fr = s.readFrames(a.trial, s.getTrialLength(a.trial) // 2, 1, True, True)[0]
        print(" ".join(sorted(o[0] for o in fr.markerObservations)))
        return
    if not a.out:
        sys.exit("--out PREFIX is required")

    t_ = a.trial
    dt = s.getTrialTimestep(t_)
    F = frames(s, t_)
    nf = len(F)
    time = np.arange(nf) * dt
    obs = [dict(fr.markerObservations) for fr in F]
    present = sorted(set().union(*[o.keys() for o in obs]))
    if a.all_markers:
        names = present
    else:
        want = [m.strip() for m in a.markers.split(",") if m.strip()] or UWA_REAL
        missing = [m for m in want if m not in present]
        if missing:
            print("WARNING: not in this trial:", " ".join(missing))
        names = [m for m in want if m in present]

    X = np.full((nf, len(names), 3), np.nan)
    for k, o in enumerate(obs):
        for m, nm in enumerate(names):
            if nm in o:
                X[k, m] = np.asarray(o[nm], float)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    write_trc(a.out + ".trc", names, time, X, 1.0 / dt)

    # force plates (raw)
    nplates = len(F[0].rawForcePlateForces) if len(F[0].rawForcePlateForces) else 0
    cols, rows, peaks = [], [], []
    for i in range(nplates):
        cols += [f"plate{i+1}_force_v{c}" for c in "xyz"]
        cols += [f"plate{i+1}_force_p{c}" for c in "xyz"]
        cols += [f"plate{i+1}_torque_{c}" for c in "xyz"]
    for fr in F:
        r = []
        for i in range(nplates):
            r += list(np.asarray(fr.rawForcePlateForces[i], float))
            r += list(np.asarray(fr.rawForcePlateCenterOfPressures[i], float))
            r += list(np.asarray(fr.rawForcePlateTorques[i], float))
        rows.append(r)
    rows = np.array(rows) if nplates else np.zeros((nf, 0))
    if nplates:
        write_mot(a.out + "_grf.mot", time, cols, rows)

    # info
    mass = s.getMassKg()
    with open(a.out + "_info.txt", "w") as f:
        f.write(f"source\t{a.b3d}\ntrial\t{t_}\t{s.getTrialName(t_)}\n")
        f.write(f"mass_kg\t{mass}\nheight_m\t{s.getHeightM()}\nsex\t{s.getBiologicalSex()}\n")
        f.write(f"rate_hz\t{1/dt:.3f}\nframes\t{nf}\nduration_s\t{nf*dt:.3f}\n")
        f.write(f"markers\t{len(names)}\t{' '.join(names)}\n")
        gaps = {nm: int(np.isnan(X[:, m, 0]).sum()) for m, nm in enumerate(names)}
        f.write("gap_frames\t" + " ".join(f"{k}:{v}" for k, v in gaps.items() if v) + "\n")
        for i in range(nplates):
            fz = np.linalg.norm(rows[:, 9 * i:9 * i + 3], axis=1)
            on = np.where(fz > 20)[0]
            span = f"{time[on[0]]:.3f}-{time[on[-1]]:.3f} s" if len(on) else "unloaded"
            f.write(f"plate{i+1}\tpeak {fz.max():.0f} N ({fz.max()/(mass*9.81):.2f} BW)\tloaded {span}\n")
    print(open(a.out + "_info.txt").read())


if __name__ == "__main__":
    main()
