
"""Compare IK results against the known truth from synth_markers.

usage: python compare_truth.py <truth_prefix> <label>=<result.mot> [<label>=<result.mot> ...]
                               [--cutoff 6] [--coords ankle,knee,hip,lumbar,pelvis]

For each result:
  - angles q are compared directly with the truth;
  - accelerations q-ddot: if <result>_udot.mot exists (spline_ik writes it) it is used as is;
    otherwise q is low-pass filtered (4th-order zero-lag Butterworth, --cutoff Hz, like
    OpenSim's ID tool) and differentiated twice, which is what ID would do with that result.
Errors are RMS over the trial, in degrees and deg/s^2 (translations skipped).
"""
import os
import sys

import numpy as np


def read_mot(path):
    lines = open(path).readlines()
    start = next(i for i, l in enumerate(lines) if l.strip().lower() == "endheader") + 1
    cols = lines[start].split()
    data = np.loadtxt(lines[start + 1:])
    return cols, data


def column(cols, data, name, t_out):
    return np.interp(t_out, data[:, 0], data[:, cols.index(name)])


def lowpass(y, dt, cutoff):
    from scipy.signal import butter, filtfilt
    b, a = butter(4, cutoff / (0.5 / dt))
    return filtfilt(b, a, y)


def main():
    args = sys.argv[1:]
    cutoff, keys = 6.0, ["ankle", "knee", "hip", "lumbar", "pelvis_tilt", "pelvis_list", "pelvis_rotation"]
    rest = []
    i = 0
    while i < len(args):
        if args[i] == "--cutoff":
            cutoff = float(args[i + 1]); i += 2
        elif args[i] == "--coords":
            keys = args[i + 1].split(","); i += 2
        else:
            rest.append(args[i]); i += 1
    if len(rest) < 2:
        print(__doc__); sys.exit(1)

    prefix, results = rest[0], [r.split("=", 1) for r in rest[1:]]
    tc, tq = read_mot(prefix + "_truth.mot")
    _, tqdd = read_mot(prefix + "_truth_udot.mot")
    t = tq[:, 0]
    dt = t[1] - t[0]
    names = [n for n in tc[1:] if any(k in n for k in keys) and not n.endswith(("_tx", "_ty", "_tz"))]

    loaded = []
    for label, path in results:
        rc, rd = read_mot(path)
        udot_path = path[:-4] + "_udot.mot" if path.endswith(".mot") else path + "_udot.mot"
        if os.path.exists(udot_path):
            uc, ud = read_mot(udot_path)
            how = "exact spline derivative"
        else:
            uc, ud = None, None
            how = f"filtered {cutoff:g} Hz + finite differences"
        loaded.append((label, rc, rd, uc, ud))
        print(f"{label}: {path}  (q-ddot: {how})")

    # trim the ends, where any filter or spline is least reliable
    keep = (t >= t[0] + 0.05) & (t <= t[-1] - 0.05)

    head = f"{'coordinate':22s}" + "".join(f"{label + ' q':>14s}" for label, *_ in loaded) \
        + "".join(f"{label + ' qdd':>16s}" for label, *_ in loaded)
    print("\nRMS error vs truth: q in deg, q-ddot in deg/s^2")
    print(head)
    totals = np.zeros(2 * len(loaded))
    for n in names:
        true_q = tq[:, tc.index(n)]
        true_qdd = tqdd[:, tc.index(n)]
        eq, eqdd = [], []
        for label, rc, rd, uc, ud in loaded:
            if n not in rc:
                eq.append(np.nan); eqdd.append(np.nan); continue
            q = column(rc, rd, n, t)
            if uc is not None and n in uc:
                qdd = column(uc, ud, n, t)
            else:
                qdd = np.gradient(np.gradient(lowpass(q, dt, cutoff), dt), dt)
            eq.append(np.sqrt(np.mean((q - true_q)[keep] ** 2)))
            eqdd.append(np.sqrt(np.mean((qdd - true_qdd)[keep] ** 2)))
        totals += np.nan_to_num(np.array(eq + eqdd))
        print(f"{n:22s}" + "".join(f"{e:14.2f}" for e in eq) + "".join(f"{e:16.0f}" for e in eqdd))
    print(f"{'mean':22s}" + "".join(f"{e / len(names):14.2f}" for e in totals[:len(loaded)])
          + "".join(f"{e / len(names):16.0f}" for e in totals[len(loaded):]))


if __name__ == "__main__":
    main()
