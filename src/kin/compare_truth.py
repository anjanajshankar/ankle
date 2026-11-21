
"""Compare IK results against the known truth from synth_markers.

usage: python compare_truth.py <truth_prefix> <label>=<result.mot> [<label>=<result.mot> ...]
                               [--cutoff 6] [--coords ankle,knee,hip,lumbar,pelvis]
                               [--filter-angles] [--summary FILE --tag TAG] [--around-impacts 0.06]

--filter-angles  also low-pass the angles of results without an exact derivative file (what ID receives)
--summary FILE   append one CSV row per result: tag,label,cutoff,filtered_angles,mean_q,mean_qdd,ankle_q,ankle_qdd
--around-impacts W  (truth made with synth_markers --impacts) two tables: errors within W s of a foot contact
                 in <truth_prefix>_impacts.csv, and everywhere else; plus the peak |q-ddot| near contacts as
                 % of the true peak. Summary rows get "|near" / "|far" appended to the tag and one more
                 column, peak_knee_pct (near rows only).
A result can have its own cutoff:  label:15=result.mot  (filtered at 15 Hz whatever --cutoff says).

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
    cutoff, keys = 6.0, ["ankle", "knee", "hip", "lumbar", "pelvis_tilt", "pelvis_list", "pelvis_obliquity",
                         "pelvis_rotation"]
    filter_angles, summary, tag, around = False, None, "", None
    rest = []
    i = 0
    while i < len(args):
        if args[i] == "--cutoff":
            cutoff = float(args[i + 1]); i += 2
        elif args[i] == "--coords":
            keys = args[i + 1].split(","); i += 2
        elif args[i] == "--filter-angles":
            filter_angles = True; i += 1
        elif args[i] == "--summary":
            summary = args[i + 1]; i += 2
        elif args[i] == "--tag":
            tag = args[i + 1]; i += 2
        elif args[i] == "--around-impacts":
            around = float(args[i + 1]); i += 2
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
        fc = cutoff
        if ":" in label:
            label, c = label.split(":", 1); fc = float(c)
        rc, rd = read_mot(path)
        udot_path = path[:-4] + "_udot.mot" if path.endswith(".mot") else path + "_udot.mot"
        if os.path.exists(udot_path):
            uc, ud = read_mot(udot_path)
            how = "exact spline derivative"
        else:
            uc, ud = None, None
            how = f"filtered {fc:g} Hz + finite differences"
        loaded.append((label, rc, rd, uc, ud, fc))
        print(f"{label}: {path}  (q-ddot: {how})")

    # each result's q and q-ddot on the truth's time grid, per coordinate
    series = []
    for label, rc, rd, uc, ud, fc in loaded:
        per = {}
        for n in names:
            if n not in rc:
                continue
            q = column(rc, rd, n, t)
            if uc is not None and n in uc:
                qdd = column(uc, ud, n, t)
            else:
                qf = lowpass(q, dt, fc)
                qdd = np.gradient(np.gradient(qf, dt), dt)
                if filter_angles:
                    q = qf
            per[n] = (q, qdd)
        series.append(per)

    # trim the ends, where any filter or spline is least reliable
    keep = (t >= t[0] + 0.05) & (t <= t[-1] - 0.05)
    regions = [("", keep)]
    impacts = []
    if around is not None:
        lines = open(prefix + "_impacts.csv").read().split()[1:]
        impacts = [(float(l.split(",")[0]), l.split(",")[1].strip()) for l in lines if l.strip()]
        near = np.zeros_like(keep)
        for e, _ in impacts:
            near |= np.abs(t - e) <= around
        regions = [("near", keep & near), ("far", keep & ~near)]
        print(f"\n{len(impacts)} foot contacts; 'near' = within {around * 1000:.0f} ms of one "
              f"({(keep & near).sum()} frames), 'far' = the rest ({(keep & ~near).sum()} frames)")

    L = len(loaded)
    peak_knee = [np.nan] * L
    for region, mask in regions:
        if not mask.any():
            print(f"\n(no frames in region {region})"); continue
        head = f"{'coordinate':22s}" + "".join(f"{label + ' q':>14s}" for label, *_ in loaded) \
            + "".join(f"{label + ' qdd':>16s}" for label, *_ in loaded)
        title = f" -- {region} contacts" if region == "near" else (" -- away from contacts" if region == "far" else "")
        print(f"\nRMS error vs truth: q in deg, q-ddot in deg/s^2{title}")
        print(head)
        totals = np.zeros(2 * L)
        ankle = np.zeros(2 * L); n_ankle = 0
        for n in names:
            true_q = tq[:, tc.index(n)]
            true_qdd = tqdd[:, tc.index(n)]
            eq, eqdd = [], []
            for per in series:
                if n not in per:
                    eq.append(np.nan); eqdd.append(np.nan); continue
                q, qdd = per[n]
                eq.append(np.sqrt(np.mean((q - true_q)[mask] ** 2)))
                eqdd.append(np.sqrt(np.mean((qdd - true_qdd)[mask] ** 2)))
            totals += np.nan_to_num(np.array(eq + eqdd))
            if n.startswith("ankle_angle"):
                ankle += np.nan_to_num(np.array(eq + eqdd)); n_ankle += 1
            print(f"{n:22s}" + "".join(f"{e:14.2f}" for e in eq) + "".join(f"{e:16.0f}" for e in eqdd))
        print(f"{'mean':22s}" + "".join(f"{e / len(names):14.2f}" for e in totals[:L])
              + "".join(f"{e / len(names):16.0f}" for e in totals[L:]))
        if region == "near":
            # peak |q-ddot| within each contact window, as % of the true peak, averaged over contacts
            for stem in ("knee_angle", "ankle_angle", "hip_flexion"):
                pct = [[] for _ in range(L)]
                for e, sd in impacts:
                    n = f"{stem}_{sd}"   # the contacting leg's coordinate
                    w = keep & (np.abs(t - e) <= around)
                    if n not in names or not w.any():
                        continue
                    pk = np.max(np.abs(tqdd[w, tc.index(n)]))
                    for k, per in enumerate(series):
                        if n in per:
                            pct[k].append(100 * np.max(np.abs(per[n][1][w])) / pk)
                if not any(pct):
                    continue
                if stem == "knee_angle":
                    peak_knee = [np.mean(p) if p else np.nan for p in pct]
                print(f"{'peak ' + stem + ' %':22s}" + "".join(f"{np.mean(p) if p else np.nan:14.0f}" for p in pct))
        if summary:
            rtag = tag + (f"|{region}" if region else "")
            with open(summary, "a") as f:
                for k, (label, rc, rd, uc, ud, fc) in enumerate(loaded):
                    exact = uc is not None
                    f.write(f"{rtag},{label},{'' if exact else fc},{int(filter_angles and not exact)},"
                            f"{totals[k] / len(names):.4f},{totals[L + k] / len(names):.2f},"
                            f"{ankle[k] / max(1, n_ankle):.4f},{ankle[L + k] / max(1, n_ankle):.2f}"
                            + ((f",{peak_knee[k]:.1f}" if region == "near" else ",") if around is not None else "")
                            + "\n")

if __name__ == "__main__":
    main()
