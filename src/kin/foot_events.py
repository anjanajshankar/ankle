"""Marker-based foot events (plant foot and initial contact, IC) for datasets without force plates (ACL),
tuned and validated against the force plates of datasets that have them. Hypothesis-blind: uses foot and
pelvis MARKERS only (no IK, no joint angle, no moment).

Rule v2 (23 Nov 2026; v1 = height threshold alone failed its acceptance criterion). Segmentation height
H_SEG = 20 mm (fixed); one tuned parameter v, plus a constant bias correction b:
  floor     for each foot marker, its on-floor height = 1st percentile of its height over all of that
            participant's trials (the lab floor is the same for a participant's trials)
  on(t)     a foot is on the ground when its lowest marker, relative to that marker's on-floor height,
            is below H_SEG; contacts = runs of on(t) lasting >= 50 ms
  plant     the contact during which the walking/running direction changes most (horizontal velocity
            of the pelvis-marker centroid just before vs just after the contact)
  IC        refined: within 60 ms either side of the plant contact's start, the first frame at which any
            foot marker of the plant foot is below H_SEG AND moving down slower than v (vertical marker
            velocity > -v; markers low-pass filtered at 15 Hz), minus b. IC at the first frame = "censored".

tune:   python3 src/kin/foot_events.py tune work/SoccerCOD work/<Nitschke folder>
        Reference = GRF: Soccer COD trials_grf.csv (plant_foot, ic); Nitschke trials.csv (right foot, ic1).
        For each v in the grid: plant foot correct (%), and the IC error (marker - GRF, ms) for trials with
        the right foot: bias b = median error, MAE after removing b, 90th percentile |error|.
        Leave-one-dataset-out (tune v and b on one dataset, test on the other) shows how well it transfers.
        Chosen v = lowest mean of the two datasets' MAE (equal weight); b = median error at that v, pooled.
apply:  python3 src/kin/foot_events.py apply work/ACL --v 0.3 --bias 4
        -> work/ACL/trials_events.csv (plant_foot, ic, contact_end, heading_change_deg, censored, name_side)
"""
import csv, os, re, sys
import xml.etree.ElementTree as ET
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from assign_plates import read_trc  # noqa: E402

GRID_V = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]    # m/s
H_SEG = 0.020                               # m
MIN_DUR = 0.05
SEARCH = 0.06                               # s


def lowpass_nan(y, dt, fc=15.0):
    from scipy.signal import butter, filtfilt
    ok = np.isfinite(y)
    if ok.sum() < 20: return np.full_like(y, np.nan)
    idx = np.arange(len(y))
    z = np.interp(idx, idx[ok], y[ok])
    b, a = butter(4, fc / (0.5 / dt))
    z = filtfilt(b, a, z)
    z[~ok] = np.nan
    return z


def body_markers(model):
    out = {}
    for m in ET.parse(model).getroot().iter("Marker"):
        body = (m.findtext("body") or m.findtext("socket_parent_frame") or "").strip().split("/")[-1]
        out.setdefault(body, []).append(m.get("name"))
    return out


def load(r):
    names, t, M = read_trc(r["trc"])
    units = open(r["trc"]).read().split("\n")[2].split("\t")
    if not any(u.strip() == "m" for u in units):
        M = M / 1000.0
    bm = body_markers(r["model"])
    feet = {s: [n for b in (f"calcn_{s}", f"toes_{s}") for n in bm.get(b, []) if n in names] for s in "rl"}
    pelvis = [n for n in bm.get("pelvis", []) if n in names]
    col = lambda n: M[:, 3 * names.index(n):3 * names.index(n) + 3]
    return {"t": t, "feet": {s: {n: col(n)[:, 1] for n in feet[s]} for s in "rl"},
            "pelvis": np.nanmean(np.stack([col(n) for n in pelvis]), axis=0) if pelvis else None}


def floors(trials):
    """Per participant, per foot marker: 1st percentile of its height over all trials."""
    acc = {}
    for r, d in trials:
        for s in "rl":
            for n, h in d["feet"][s].items():
                acc.setdefault((r["participant"], n), []).append(h)
    return {k: np.nanpercentile(np.concatenate(v), 1) for k, v in acc.items()}


def contacts(r, d, fl, h_m):
    t = d["t"]; out = []
    for s in "rl":
        rel = [hh - fl[(r["participant"], n)] for n, hh in d["feet"][s].items()]
        if not rel: continue
        with np.errstate(all="ignore"):
            low = np.nanmin(np.stack(rel), axis=0)
        on = np.nan_to_num(low, nan=1e9) < h_m
        i = 0
        while i < len(on):
            if on[i]:
                j = i
                while j + 1 < len(on) and on[j + 1]: j += 1
                if t[j] - t[i] >= MIN_DUR:
                    out.append((s, i, j))
                i = j + 1
            else:
                i += 1
    return out


def heading_change(d, i, j):
    p, t = d["pelvis"], d["t"]
    if p is None: return np.nan
    def vel(a, b):
        a, b = max(a, 0), min(b, len(t) - 1)
        if b <= a: return None
        return (p[b, [0, 2]] - p[a, [0, 2]]) / (t[b] - t[a])
    k = max(1, int(round(0.05 / np.median(np.diff(t)))))
    v0, v1 = vel(i - k, i), vel(j, j + k)
    if v0 is None or v1 is None or not (np.all(np.isfinite(v0)) and np.all(np.isfinite(v1))): return np.nan
    c = np.dot(v0, v1) / (np.linalg.norm(v0) * np.linalg.norm(v1) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def plant_contact(r, d, fl):
    if "_plant" not in d:
        cs = contacts(r, d, fl, H_SEG)
        if not cs: d["_plant"] = None
        else:
            scored = [(np.nan_to_num(heading_change(d, i, j), nan=-1), d["t"][j] - d["t"][i], s, i, j) for s, i, j in cs]
            hc, _, s, i, j = max(scored)
            d["_plant"] = (s, i, j, hc, len(cs))
    return d["_plant"]


def refine_ic(r, d, fl, s, i, v):
    t = d["t"]; dt = float(np.median(np.diff(t)))
    if "_vel" not in d: d["_vel"] = {}
    if s not in d["_vel"]:
        sig = []
        for n, hh in d["feet"][s].items():
            hf = lowpass_nan(hh, dt)
            sig.append((hh - fl[(r["participant"], n)], np.gradient(hf, dt)))
        d["_vel"][s] = sig
    k = int(round(SEARCH / dt))
    for q in range(max(0, i - k), min(len(t), i + k + 1)):
        for rel, vel in d["_vel"][s]:
            if np.isfinite(rel[q]) and np.isfinite(vel[q]) and rel[q] < H_SEG and vel[q] > -v:
                return q
    return i


def plant_event(r, d, fl, v):
    pc = plant_contact(r, d, fl)
    if pc is None: return None
    s, i, j, hc, n = pc
    q = refine_ic(r, d, fl, s, i, v)
    return {"plant_foot": s, "ic": float(d["t"][q]), "end": float(d["t"][j]), "heading": hc,
            "censored": q == 0, "n_contacts": n}


def reference(work):
    """GRF reference rows: (row, plant_foot, ic)."""
    if os.path.exists(os.path.join(work, "trials_grf.csv")):
        rows = list(csv.DictReader(open(os.path.join(work, "trials_grf.csv"))))
        return [(r, r["plant_foot"], float(r["ic"])) for r in rows if r.get("plant_foot")]
    rows = list(csv.DictReader(open(os.path.join(work, "trials.csv"))))
    return [(r, "r", float(r["ic1"])) for r in rows]


def errors(ref, data, fl, h):
    ok, err, n_cens = 0, [], 0
    for (r, foot, ic), d in zip(ref, data):
        ev = plant_event(r, d, fl, h)
        if ev is None or ev["plant_foot"] != foot: continue
        ok += 1
        if ev["censored"]: n_cens += 1; continue
        err.append((ev["ic"] - ic) * 1000)
    return ok / len(ref), np.array(err), n_cens


def stats(e, b):
    a = np.abs(e - b)
    return float(np.mean(a)), float(np.percentile(a, 90))


def tune(works):
    sets = []
    for w in works:
        ref = reference(w)
        data = [load(r) for r, _, _ in ref]
        sets.append((os.path.basename(w.rstrip("/")), ref, data, floors([(r, d) for (r, _, _), d in zip(ref, data)])))
        print(f"{sets[-1][0]}: {len(ref)} trials", flush=True)
    res = {}
    print(f"\n{'v m/s':>5s} {'dataset':12s}{'plant ok':>9s}{'n':>5s}{'cens':>5s}{'bias ms':>9s}{'MAE ms':>8s}{'p90 ms':>8s}")
    for h in GRID_V:
        for name, ref, data, fl in sets:
            ok, e, nc = errors(ref, data, fl, h)
            b = float(np.median(e)) if len(e) else np.nan
            mae, p90 = stats(e, b) if len(e) else (np.nan, np.nan)
            res[(h, name)] = (ok, e, b, mae)
            print(f"{h:5.2f} {name:12s}{ok:9.0%}{len(e):5d}{nc:5d}{b:9.1f}{mae:8.1f}{p90:8.1f}")
    names = [s[0] for s in sets]
    if len(names) == 2:
        print("\nleave-one-dataset-out (v and bias from the other dataset):")
        for test in names:
            train = [n for n in names if n != test][0]
            h = min(GRID_V, key=lambda x: res[(x, train)][3])
            b = res[(h, train)][2]
            mae, p90 = stats(res[(h, test)][1], b)
            print(f"  test {test:12s} v {h} m/s, bias {b:.1f} ms (from {train}): plant ok {res[(h, test)][0]:.0%}, "
                  f"MAE {mae:.1f} ms, p90 {p90:.1f} ms")
    h = min(GRID_V, key=lambda x: np.nanmean([res[(x, n)][3] for n in names]))
    e = np.concatenate([res[(h, n)][1] for n in names])
    b = float(np.median(e))
    print(f"\nchosen: v = {h} m/s, bias = {b:.1f} ms  (pooled MAE {stats(e, b)[0]:.1f} ms, p90 {stats(e, b)[1]:.1f} ms, "
          f"plant ok " + ", ".join(f"{n} {res[(h, n)][0]:.0%}" for n in names) + ")")


def apply(work, h, b):
    rows = list(csv.DictReader(open(os.path.join(work, "trials.csv"))))
    data = [load(r) for r in rows]
    fl = floors(list(zip(rows, data)))
    out = []
    for r, d in zip(rows, data):
        ev = plant_event(r, d, fl, h)
        m = re.search(r"C\d+_([LR])", r["trial"])
        if ev is None:
            out.append({**{k: r[k] for k in ("participant", "trial", "dir")}, "plant_foot": "", "ic": "", "contact_end": "",
                        "heading_change_deg": "", "censored": "", "n_contacts": 0, "name_side": m.group(1) if m else ""})
            continue
        out.append({**{k: r[k] for k in ("participant", "trial", "dir")}, "plant_foot": ev["plant_foot"],
                    "ic": "" if ev["censored"] else f"{ev['ic'] - b / 1000:.4f}", "contact_end": f"{ev['end']:.4f}",
                    "heading_change_deg": f"{ev['heading']:.0f}", "censored": int(ev["censored"]),
                    "n_contacts": ev["n_contacts"], "name_side": m.group(1) if m else ""})
    path = os.path.join(work, "trials_events.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    ok = [o for o in out if o["plant_foot"]]
    print(f"{len(out)} trials: plant foot found in {len(ok)} (right {sum(o['plant_foot'] == 'r' for o in ok)}, "
          f"left {sum(o['plant_foot'] == 'l' for o in ok)}), censored {sum(o['censored'] == 1 for o in ok)}")
    pairs = [(o["plant_foot"], o["name_side"].lower()) for o in ok if o["name_side"]]
    if pairs:
        same = sum(p == n for p, n in pairs)
        print(f"plant foot vs the letter in the trial name: same {same}, opposite {len(pairs) - same}")
    hc = [float(o["heading_change_deg"]) for o in ok if o["heading_change_deg"]]
    if hc: print(f"heading change over the plant contact: median {np.median(hc):.0f} deg (range {min(hc):.0f}-{max(hc):.0f})")
    print(f"wrote {path}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) >= 2 and a[0] == "tune":
        tune(a[1:])
    elif len(a) >= 2 and a[0] == "apply" and "--v" in a and "--bias" in a:
        apply(a[1], float(a[a.index("--v") + 1]), float(a[a.index("--bias") + 1]))
    else:
        print(__doc__)
