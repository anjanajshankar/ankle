"""Assign every force-plate contact to a foot and write the GRF per foot, for every trial of a dataset.
For each trial in <work>/trials.csv (GRF file = the trial's .trc with _grf.mot):
  1. contacts = periods where a plate's vertical force (plateN_force_vy) exceeds 20 N (protocol);
  2. foot = the side whose foot markers (model markers on calcn_* / toes_* bodies) are closest, horizontally,
     to the plate's centre of pressure, averaged over the contact; FLAGGED if the other foot is within 5 cm of
     being as close (ambiguous), if the COP is > 15 cm from the chosen foot (far), or if the other foot is
     also near that plate's COP for > 30% of the contact (other_foot_near);
  3. writes <trial dir>/grf_feet.mot: ground_force_v*/p* (right) and l_ground_force_v*/p* (left), summing the
     plates under each foot (COP = force-weighted mean); COP held at its nearest loaded value off the plates;
  4. plant foot = the foot with the largest vertical impulse; ic / toe_off = its first / last contact time.
Writes <work>/contacts.csv and <work>/trials_grf.csv. No hypothesis variable is computed.
usage (from the ankle folder):  python3 src/kin/assign_plates.py work/SoccerCOD
"""
import csv, os, re, sys, xml.etree.ElementTree as ET
import numpy as np
THRESH = 20.0

def read_mot(path):
    L = open(path).read().split("\n")
    k = next(i for i, l in enumerate(L) if l.strip().lower() == "endheader")
    return L[k + 1].split(), np.loadtxt([l for l in L[k + 2:] if l.strip()], ndmin=2)

def read_trc(path):
    L = open(path).read().split("\n")
    names = [n for n in L[3].split("\t")[2:] if n.strip()]
    k = next(i for i, l in enumerate(L) if "X1\t" in l)
    rows = [r.split("\t") for r in L[k + 1:] if r.strip()]
    def val(x):
        try: return float(x)
        except ValueError: return np.nan
    return names, np.array([float(r[1]) for r in rows]), np.array([[val(x) for x in r[2:2 + 3 * len(names)]] for r in rows])

def foot_markers(model):
    out = {"r": [], "l": []}
    for m in ET.parse(model).getroot().iter("Marker"):
        body = (m.findtext("body") or m.findtext("socket_parent_frame") or "").strip().split("/")[-1]
        mm = re.match(r"(calcn|toes)_([rl])$", body)
        if mm: out[mm.group(2)].append(m.get("name"))
    return out

def contacts_of(vy):
    on, out, i = vy > THRESH, [], 0
    while i < len(on):
        if on[i]:
            j = i
            while j + 1 < len(on) and on[j + 1]: j += 1
            if j > i: out.append((i, j))
            i = j + 1
        else: i += 1
    return out

def write_feet_mot(path, t, feet):
    cols = ["time"] + [f"{p}force_{a}" for p in ("ground_", "l_ground_") for a in ("vx", "vy", "vz", "px", "py", "pz")]
    with open(path, "w") as f:
        f.write(f"grf_feet\nversion=1\nnRows={len(t)}\nnColumns={len(cols)}\ninDegrees=yes\nendheader\n" + "\t".join(cols) + "\n")
        for i in range(len(t)):
            f.write("\t".join(f"{v:.6f}" for v in [t[i]] + list(feet["r"][i]) + list(feet["l"][i])) + "\n")

def main():
    w = sys.argv[1].rstrip("/")
    rows = list(csv.DictReader(open(f"{w}/trials.csv")))
    all_c, out_rows = [], []
    for r in rows:
        trc = r["trc"]; grf = trc[:-4] + "_grf.mot"; tag = f"{r['participant']}_{r['trial']}"
        if not os.path.exists(grf):
            print(f"{tag}: no GRF file, skipped"); continue
        cols, G = read_mot(grf); t = G[:, 0]
        plates = sorted({int(m.group(1)) for c in cols for m in [re.match(r"plate(\d+)_force_vy$", c)] if m})
        fm = foot_markers(r["model"]); names, tm, M = read_trc(trc)
        def foot_xyz(side):
            idx = [names.index(n) for n in fm[side] if n in names]
            P = np.nanmean(np.stack([M[:, 3 * i:3 * i + 3] for i in idx]), axis=0)
            return np.column_stack([np.interp(t, tm, P[:, k]) for k in range(3)])
        F = {"r": foot_xyz("r"), "l": foot_xyz("l")}
        feet = {s: np.zeros((len(t), 6)) for s in "rl"}; fy = {s: np.zeros(len(t)) for s in "rl"}
        cop = {s: np.zeros((len(t), 3)) for s in "rl"}; imp_s = {"r": 0.0, "l": 0.0}
        first_on, last_off, flagged = {}, {}, False; cl = []
        for p in plates:
            c = {a: G[:, cols.index(f"plate{p}_force_{a}")] for a in ("vx", "vy", "vz", "px", "py", "pz")}
            for i0, i1 in contacts_of(c["vy"]):
                sl = slice(i0, i1 + 1); wgt = c["vy"][sl]
                if float(np.sum(0.5 * (wgt[1:] + wgt[:-1]) * np.diff(t[sl]))) < 5.0: continue   # < 5 N*s: plate noise, not a contact
                d = {s: np.average(np.hypot(F[s][sl, 0] - c["px"][sl], F[s][sl, 2] - c["pz"][sl]), weights=wgt) for s in "rl"}
                side = min(d, key=d.get); other = "l" if side == "r" else "r"; flag = []
                if d[other] - d[side] < 0.05: flag.append("ambiguous")
                if d[side] > 0.15: flag.append("far")
                if (np.hypot(F[other][sl, 0] - c["px"][sl], F[other][sl, 2] - c["pz"][sl]) < 0.15).mean() > 0.3: flag.append("other_foot_near")
                flagged = flagged or bool(flag)
                for k, a in enumerate(("vx", "vy", "vz")): feet[side][sl, k] += c[a][sl]
                fy[side][sl] += c["vy"][sl]
                for k, a in enumerate(("px", "py", "pz")): cop[side][sl, k] += c["vy"][sl] * c[a][sl]
                imp = float(np.sum(0.5 * (wgt[1:] + wgt[:-1]) * np.diff(t[sl])))
                imp_s[side] += imp; cl.append((imp, side, t[i0], t[i1]))
                first_on[side] = min(first_on.get(side, 1e9), t[i0]); last_off[side] = max(last_off.get(side, -1e9), t[i1])
                all_c.append([tag, p, side, f"{t[i0]:.4f}", f"{t[i1]:.4f}", f"{wgt.max():.0f}", f"{imp:.1f}",
                              f"{d['r'] * 1000:.0f}", f"{d['l'] * 1000:.0f}", ";".join(flag)])
        for s in "rl":
            on = fy[s] > 0
            feet[s][on, 3:6] = cop[s][on] / fy[s][on, None]
            if on.any():
                idx = np.where(on, np.arange(len(t)), -1); fwd = np.maximum.accumulate(idx)
                fwd[fwd < 0] = np.argmax(on); feet[s][:, 3:6] = feet[s][fwd, 3:6]
        out = os.path.join(r["dir"], "grf_feet.mot"); write_feet_mot(out, t, feet)
        best = max(cl) if cl else None; plant = best[1] if best else ""   # plant = the single largest-impulse contact
        out_rows.append({**r, "grf_feet": out, "plant_foot": plant,
                         "ic": f"{best[2]:.4f}" if best else "", "toe_off": f"{best[3]:.4f}" if best else "",
                         "flagged": int(flagged)})
    with open(f"{w}/contacts.csv", "w", newline="") as f:
        cw = csv.writer(f); cw.writerow(["trial", "plate", "foot", "t_on", "t_off", "peak_vy_N", "impulse_Ns", "dist_r_mm", "dist_l_mm", "flags"]); cw.writerows(all_c)
    if out_rows:
        with open(f"{w}/trials_grf.csv", "w", newline="") as f:
            dw = csv.DictWriter(f, fieldnames=list(out_rows[0].keys())); dw.writeheader(); dw.writerows(out_rows)
    n_fl = sum(1 for c in all_c if c[-1])
    print(f"{len(out_rows)} trials, {len(all_c)} contacts (right {sum(c[2] == 'r' for c in all_c)}, left {sum(c[2] == 'l' for c in all_c)}), "
          f"{n_fl} flagged; plant foot right {sum(o['plant_foot'] == 'r' for o in out_rows)}, left {sum(o['plant_foot'] == 'l' for o in out_rows)}")
    print(f"wrote {w}/contacts.csv, {w}/trials_grf.csv and grf_feet.mot per trial")

if __name__ == "__main__":
    main()
