"""Why are the pelvis residuals large? Hypothesis-blind diagnostics for a dataset after step 3.

Reads ONLY: the pelvis residual force columns of id_A.sto / id_B.sto (pelvis_tx/ty/tz_force), the filtered
per-foot GRF (grf_feet_filt<fc>.mot), and the foot markers of the .trc. No joint angle or moment is read.

For a few trials per participant (the lowest and highest Pipeline-A residual in step3_results.csv, or --only):
  1. plot  <work>/residual_check/<trial>.png, three panels over time:
       residual force components / BW (A solid, B dashed) | vertical GRF per foot / BW | height of each
       foot's lowest marker above the floor (mm); plant contact (IC, toe-off) marked
  2. numbers per trial over the plant contact -> <work>/residual_check/residual_check.csv:
       mean_x/y/z      signed mean of each residual component / BW   (a steady offset -> mass or COM problem;
                       y = vertical)
       spike_ratio     peak |residual| within 30 ms of IC / RMS over the contact (>> 2 -> sharp impact spike,
                       typical of a GRF-marker time offset or under-filtered impact)
       t_peak_ms       time of the largest |residual| after IC
       offplate_frac   fraction of the plant contact where the OTHER foot's lowest marker is < 30 mm above
                       the floor while its plate force is < 20 N (other foot on the floor off the plates ->
                       missing GRF)
  3. with --lags: time-shift test (Pipeline A only). The GRF file is shifted by each lag (ms; positive = GRF
     later) and ID is rerun; residual RMS over the plant contact vs lag -> lag_test.csv. A clear minimum away
     from 0 means the force plates and markers are out of sync by about that much.

usage (from the ankle folder):
  python3 src/kin/residual_check.py work/SoccerCOD --fc 20 [--lags -30,-20,-10,0,10,20,30] [--only TAG]
"""
import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step3_run import read_table, write_table, ext_loads, id_setup, run, G  # noqa: E402
from assign_plates import read_trc, foot_markers  # noqa: E402


def tag_of(r):
    p = r["participant"]
    return f"{p if p.startswith('P') else 'P' + p}_{r['trial']}"


def pelvis_force(sto):
    _, cols, d = read_table(sto)
    return d[:, 0], d[:, [cols.index(f"pelvis_t{a}_force") for a in "xyz"]]


def residual_rms(sto, s0, s1, bw):
    t, F = pelvis_force(sto)
    sel = (t >= s0) & (t <= s1)
    return float(np.sqrt(np.mean((F[sel] ** 2).sum(axis=1)))) / bw


def foot_heights(r):
    """Lowest foot-marker height above the floor (mm) per foot, and the marker times."""
    names, tm, M = read_trc(r["trc"])
    units = open(r["trc"]).read().split("\n")[2].split("\t")
    scale = 1.0 if any(u.strip() == "m" for u in units) else 0.001   # -> metres
    fm = foot_markers(r["model"])
    h = {}
    for s in "rl":
        idx = [names.index(n) for n in fm[s] if n in names]
        h[s] = np.nanmin(np.stack([M[:, 3 * i + 1] for i in idx]), axis=0) * scale
    floor = np.nanpercentile(np.concatenate([h["r"], h["l"]]), 1)
    return tm, {s: (h[s] - floor) * 1000 for s in "rl"}


def main():
    a = sys.argv[1:]
    if len(a) < 3 or a[1] != "--fc":
        print(__doc__); sys.exit(1)
    work, fc = os.path.abspath(a[0]), float(a[2])
    only = a[a.index("--only") + 1] if "--only" in a else None
    lags = [float(x) for x in a[a.index("--lags") + 1].split(",")] if "--lags" in a else []
    out_dir = os.path.join(work, "residual_check"); os.makedirs(out_dir, exist_ok=True)

    rows = {tag_of(r): r for r in csv.DictReader(open(os.path.join(work, "trials_grf.csv")))}
    res = list(csv.DictReader(open(os.path.join(work, "step3_results.csv"))))
    if only:
        chosen = [only]
    else:   # per participant: lowest and highest Pipeline-A residual
        by_p = {}
        for x in res:
            if x["trial"] in rows:
                by_p.setdefault(rows[x["trial"]]["participant"], []).append(x)
        chosen = []
        for p, xs in sorted(by_p.items()):
            xs = sorted(xs, key=lambda x: float(x["A_force_bw"]))
            chosen += [xs[0]["trial"]] + ([xs[-1]["trial"]] if len(xs) > 1 else [])

    summary, lag_rows = [], []
    for tag in chosen:
        r = rows[tag]; d = r["dir"]
        bw = float(r["mass_kg"]) * G
        ic, to = float(r["ic"]), float(r["toe_off"])
        plant = r["plant_foot"]; other = "l" if plant == "r" else "r"
        _, gc, gd = read_table(os.path.join(d, f"grf_feet_filt{fc:g}.mot"))
        gt = gd[:, 0]
        fy = {"r": gd[:, gc.index("ground_force_vy")], "l": gd[:, gc.index("l_ground_force_vy")]}
        tm, hgt = foot_heights(r)

        fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
        line = {}
        for name, style in (("A", "-"), ("B", "--")):
            sto = os.path.join(d, f"id_{name}.sto")
            if not os.path.exists(sto): continue
            t, F = pelvis_force(sto)
            for k, (lab, col) in enumerate(zip("xyz", ("tab:red", "tab:green", "tab:blue"))):
                ax[0].plot(t, F[:, k] / bw, style, color=col, lw=1.2, label=f"{name} {lab}")
            line[name] = (t, F)
        ax[0].set_ylabel("pelvis residual / BW"); ax[0].legend(ncol=6, fontsize=7); ax[0].axhline(0, color="k", lw=0.5)
        for s, col in (("r", "tab:orange"), ("l", "tab:purple")):
            ax[1].plot(gt, fy[s] / bw, color=col, label=f"{s} foot{' (plant)' if s == plant else ''}")
            ax[2].plot(tm, hgt[s], color=col, label=s)
        ax[1].set_ylabel("vertical GRF / BW"); ax[1].legend(fontsize=7)
        ax[2].set_ylabel("lowest foot marker (mm)"); ax[2].set_ylim(-10, 150); ax[2].axhline(30, color="k", lw=0.5, ls=":")
        ax[2].set_xlabel("time (s)")
        for axx in ax:
            axx.axvline(ic, color="k", lw=0.8); axx.axvline(to, color="k", lw=0.8, ls="--")
        ax[0].set_title(f"{tag}  (plant {plant}; solid = A OpenSim, dashed = B spline)", fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"{tag}.png"), dpi=110); plt.close(fig)

        # numbers over the plant contact
        h_other = np.interp(gt, tm, hgt[other])
        win = (gt >= ic) & (gt <= to)
        offplate = float(np.mean((h_other[win] < 30) & (fy[other][win] < 20))) if win.any() else np.nan
        for name, (t, F) in line.items():
            sel = (t >= ic) & (t <= to)
            mag = np.sqrt((F[sel] ** 2).sum(axis=1)) / bw
            rms = float(np.sqrt(np.mean(mag ** 2)))
            near = np.abs(t[sel] - ic) <= 0.03
            spike = float(mag[near].max() / rms) if near.any() else np.nan
            t_pk = float((t[sel][np.argmax(mag)] - ic) * 1000)
            mx, my, mz = (F[sel].mean(axis=0) / bw)
            summary.append([tag, name, plant, f"{rms:.3f}", f"{mx:.3f}", f"{my:.3f}", f"{mz:.3f}",
                            f"{spike:.2f}", f"{t_pk:.0f}", f"{offplate:.2f}"])
        print(f"{tag}: plot written; other foot on floor off the plates {offplate:.0%} of the plant contact", flush=True)

        if lags:   # time-shift test, Pipeline A
            ld = os.path.join(d, "lagtest"); os.makedirs(ld, exist_ok=True)
            t0, t1 = float(r["t0"]), float(r["t1"])
            m = max(abs(x) for x in lags) / 1000
            s0, s1 = max(ic, t0 + 0.05 + m), min(to, t1 - 0.05 - m)
            h, c, g = read_table(os.path.join(d, f"grf_feet_filt{fc:g}.mot"))
            for lag in lags:
                gs = g.copy(); gs[:, 0] += lag / 1000
                gf = os.path.join(ld, f"grf_lag{lag:+g}.mot"); write_table(gf, h, c, gs)
                ext = os.path.join(ld, f"ext_lag{lag:+g}.xml"); open(ext, "w").write(ext_loads(gf))
                setup = os.path.join(ld, f"id_lag{lag:+g}.xml")
                open(setup, "w").write(id_setup(r["model"], os.path.join(d, "opensim_ik.mot"), ext,
                                                t0 + 0.05 + m, t1 - 0.05 - m, fc, ld, f"id_lag{lag:+g}.sto"))
                ok = run(["opensim-cmd", "run-tool", setup], os.path.join(ld, f"id_lag{lag:+g}_log.txt"), False)
                v = residual_rms(os.path.join(ld, f"id_lag{lag:+g}.sto"), s0, s1, bw) if ok else np.nan
                lag_rows.append([tag, f"{lag:g}", f"{v:.3f}"])
            vals = [(float(x[1]), float(x[2])) for x in lag_rows if x[0] == tag]
            best = min(vals, key=lambda x: x[1] if not np.isnan(x[1]) else 1e9)
            print("   lag test (ms: residual/BW): " + "  ".join(f"{l:+g}: {v:.3f}" for l, v in vals)
                  + f"   -> lowest at {best[0]:+g} ms", flush=True)

    with open(os.path.join(out_dir, "residual_check.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trial", "pipeline", "plant_foot", "rms_bw", "mean_x_bw", "mean_y_bw", "mean_z_bw",
                    "spike_ratio", "t_peak_ms", "offplate_frac"])
        w.writerows(summary)
    if lag_rows:
        with open(os.path.join(out_dir, "lag_test.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["trial", "lag_ms", "A_rms_bw"]); w.writerows(lag_rows)
    print(f"\n{'trial':45s}{'pipe':>5s}{'rms':>7s}{'mean x':>8s}{'mean y':>8s}{'mean z':>8s}{'spike':>7s}{'t_pk':>6s}{'offpl':>7s}")
    for s in summary:
        print(f"{s[0]:45s}{s[1]:>5s}{s[3]:>7s}{s[4]:>8s}{s[5]:>8s}{s[6]:>8s}{s[7]:>7s}{s[8]:>6s}{s[9]:>7s}")
    print(f"\nplots and tables: {out_dir}")


if __name__ == "__main__":
    main()
