
#!/usr/bin/env python3
"""
screen_addbio.py — find AddBiomechanics .b3d files that contain marker
trajectories and ground reaction forces, especially change-of-direction trials,
without keeping the whole dataset on disk.

Stages (each resumable; results are appended to CSVs in --out):
  list    rclone lists the Drive folder (no download) -> files.csv
  header  for every file, downloads only the first --header-mb MB, reads the
          .b3d header (trial names, lengths, force plates) -> headers.csv
          (files whose header can't be read from a partial download are marked
          'needs_full')
  inspect for candidate files (default: path or a trial name looks like a
          cut / change of direction; --all for every file), downloads the
          whole file, samples frames to check for markers and GRF -> trials.csv,
          then deletes it (unless --keep-dir and it passes).

Requirements (in the codespace):
  pip install nimblephysics   (wheels exist for CPython up to ~3.11/3.12; if it
                               fails, make a venv with python3.11)
  rclone with a Google Drive remote, e.g. "gd" (read-only scope is enough):
      curl https://rclone.org/install.sh | sudo bash
      rclone config   # new remote "gd", type drive, scope 2 (read-only),
                      # answer "n" to auto config, follow the printed steps

Usage examples:
  python3 screen_addbio.py list    --folder core
  python3 screen_addbio.py header  --folder core
  python3 screen_addbio.py inspect --folder core                # COD candidates only
  python3 screen_addbio.py inspect --folder core --path-regex Camargo --max-files 1
  python3 screen_addbio.py inspect --folder core --all --max-gb 20
  python3 screen_addbio.py report  --folder core
"""
import argparse, csv, json, math, os, re, subprocess, sys, tempfile, time

FOLDERS = {
    # AddBiomechanics 1.0 core dataset (manually reviewed)
    "core": "1X3OEJXF5gzvjR5vRG7VO7h58bRwYTV0a",
    # User-upload snapshot 15 May 2024, Rajagopal model without arms
    "may24_noarms": "12CStQmsNVx_UEr9mkBkUApKZgjKOfGUq",
    # Add the "with arms" snapshot id here once you have it from the
    # addbiomechanics.org download page (folder id = part after /folders/).
}

# Change-of-direction keywords. Matched case-insensitively against the file
# path and each trial name, with separators (_ - . / space digits) as
# boundaries so e.g. "cod" does not match "code" and "turn" not "return".
COD_WORDS = [
    "cut", "cuts", "cutting", "sidestep", "sidestepping", "side_step", "sidecut",
    "side_cut", "vcut", "v_cut", "crossover", "cross_over", "cod", "changeofdirection",
    "change_of_direction", "pivot", "pivoting", "agility", "turn", "turning",
    "zigzag", "zig_zag", "505", "shuffle", "lateral", "juke", "decel", "deceleration",
    "planted", "plant", "ss", "unplanned", "unanticipated", "anticipated",
]
_SEP = r"(?:^|[\s_\-./\\0-9])"
_SEP_END = r"(?:$|[\s_\-./\\0-9])"
COD_RE = re.compile(_SEP + "(" + "|".join(sorted(map(re.escape, COD_WORDS), key=len, reverse=True)) + ")" + _SEP_END, re.I)


_NUM_RE = re.compile(r"(?:^|[\s_\-./\\])(505)(?:$|[\s_\-./\\])")  # 505 test: not inside a number/hash


def cod_match(text):
    text = text or ""
    for m in COD_RE.finditer(text):
        w = m.group(1).lower()
        if w != "505":
            return w
    m = _NUM_RE.search(text)
    return m.group(1) if m else ""


# ---------------------------------------------------------------- rclone ----
def rclone(args, remote, folder_id, **kw):
    import shutil
    if shutil.which("rclone") is None:
        sys.exit("rclone not found. Install it:  curl https://rclone.org/install.sh | sudo bash\n"
                 "(or: conda install -y -c conda-forge rclone), then run: rclone config")
    cmd = ["rclone"] + args + ["--drive-root-folder-id", folder_id]
    return subprocess.run(cmd, check=True, **kw)


def rpath(remote, path):
    return f"{remote}:{path}"


# ------------------------------------------------------------- csv utils ----
def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def append_rows(path, rows, fields):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------- nimble -----
def load_nimble():
    try:
        import nimblephysics as nimble
        return nimble
    except ImportError:
        sys.exit("nimblephysics is not installed: pip install nimblephysics "
                 "(use a python3.11 venv if no wheel exists for your Python)")


def call(obj, name, *a, default=None):
    """Call obj.name(*a) if it exists (API differs a bit between nimble versions)."""
    fn = getattr(obj, name, None)
    if fn is None:
        return default
    try:
        return fn(*a)
    except Exception:
        return default


def header_info(nimble, path):
    s = nimble.biomechanics.SubjectOnDisk(path)
    trials = []
    for t in range(s.getNumTrials()):
        trials.append({
            "trial": t,
            "trial_name": call(s, "getTrialName", t, default=""),
            "original_name": call(s, "getTrialOriginalName", t, default=""),
            "frames": call(s, "getTrialLength", t, default=""),
            "dt": call(s, "getTrialTimestep", t, default=""),
            "n_force_plates": call(s, "getNumForcePlates", t, default=""),
        })
    meta = {
        "n_trials": s.getNumTrials(),
        "mass_kg": call(s, "getMassKg", default=""),
        "height_m": call(s, "getHeightM", default=""),
        "sex": call(s, "getBiologicalSex", default=""),
        "href": call(s, "getHref", default=""),
        "notes": (call(s, "getNotes", default="") or "")[:200].replace("\n", " "),
    }
    return s, meta, trials


def inspect_trial(s, t, n_samples, grf_threshold_n):
    """Sample frames across the trial; return marker / GRF evidence."""
    n = s.getTrialLength(t)
    res = {"n_markers_max": 0, "marker_names": "", "grf_peak_N": 0.0,
           "grf_frames_over_thr": 0, "frames_sampled": 0, "grf_not_missing_frac": ""}
    miss = call(s, "getMissingGRF", t)
    if miss is not None and len(miss):
        ok = sum(1 for r in miss if "notmissing" in str(r).lower())
        res["grf_not_missing_frac"] = round(ok / len(miss), 3)
    if n <= 0:
        return res
    idx = sorted(set(int(round(i * (n - 1) / max(n_samples - 1, 1))) for i in range(n_samples)))
    names = set()
    for i in idx:
        try:
            fr = s.readFrames(t, i, 1, True, True)[0]
        except TypeError:
            fr = s.readFrames(t, i, 1)[0]
        res["frames_sampled"] += 1
        obs = getattr(fr, "markerObservations", None) or []
        res["n_markers_max"] = max(res["n_markers_max"], len(obs))
        for o in obs:
            names.add(o[0] if isinstance(o, (tuple, list)) else str(o))
        # raw force plate forces (one 3-vector per plate)
        peak = 0.0
        for f in getattr(fr, "rawForcePlateForces", None) or []:
            try:
                peak = max(peak, math.sqrt(sum(float(x) ** 2 for x in f)))
            except Exception:
                pass
        # fall back to processed GRF in the last processing pass
        if peak == 0.0:
            for p in getattr(fr, "processingPasses", None) or []:
                g = getattr(p, "groundContactForce", None)
                if g is not None and len(g):
                    g = [float(x) for x in g]
                    for k in range(0, len(g) - 2, 3):
                        peak = max(peak, math.sqrt(g[k] ** 2 + g[k + 1] ** 2 + g[k + 2] ** 2))
        res["grf_peak_N"] = max(res["grf_peak_N"], round(peak, 1))
        if peak > grf_threshold_n:
            res["grf_frames_over_thr"] += 1
    res["marker_names"] = ";".join(sorted(names)[:12])
    return res


# ---------------------------------------------------------------- stages ----
FILE_FIELDS = ["path", "size_mb", "cod_in_path"]
HDR_FIELDS = ["path", "status", "n_trials", "cod_trials", "cod_words", "trial_names",
              "mass_kg", "height_m", "sex", "href", "notes", "error"]
TRIAL_FIELDS = ["path", "trial", "trial_name", "original_name", "cod_match", "frames", "dt",
                "duration_s", "n_force_plates", "frames_sampled", "n_markers_max",
                "marker_names", "grf_peak_N", "grf_frames_over_thr",
                "grf_not_missing_frac", "has_markers", "has_grf", "error"]


def _lsjson(a, fid, path, recursive):
    args = ["lsjson", "--files-only", "--include", "*.b3d"]
    if recursive:
        args += ["-R", "--fast-list"]
    p = rclone(args + [rpath(a.remote, path)], a.remote, fid,
               capture_output=True, text=True)
    return [x for x in json.loads(p.stdout or "[]") if x["Path"].lower().endswith(".b3d")]


def stage_list(a, fid, out):
    """List .b3d files one top-level folder at a time (resumable, shows progress)."""
    f = os.path.join(out, "files.csv")
    donef = os.path.join(out, "listed_dirs.txt")
    if a.restart:
        for p in (f, donef):
            if os.path.exists(p):
                os.remove(p)
    done = set(open(donef).read().splitlines()) if os.path.exists(donef) else set()
    print(f"Listing {a.folder} ({fid}) via rclone, no download ...", flush=True)
    def subdirs(p):
        out = rclone(["lsf", "--dirs-only", rpath(a.remote, p)], a.remote, fid,
                     capture_output=True, text=True).stdout.splitlines()
        return [(p + "/" if p else "") + d.rstrip("/") for d in out if d.strip()]
    # descend through single-folder levels so the work splits where the tree branches
    tops = subdirs("")
    for _ in range(a.split_depth):
        if len(tops) >= a.min_pieces:
            break
        nxt = []
        for d in tops:
            sub = subdirs(d)
            # a folder whose name ends in .b3d/.osim is a file wrapper: keep it whole
            if sub and not d.lower().endswith((".b3d", ".osim")):
                nxt += sub
            else:
                nxt.append(d)
        if nxt == tops:
            break
        print(f"  {len(tops)} folders -> {len(nxt)} one level down", flush=True)
        tops = nxt
    print(f"{len(tops)} top-level folders ({len(done)} already listed)", flush=True)
    if "<root>" not in done:   # .b3d files sitting directly in the root
        rows = [{"path": x["Path"], "size_mb": round(x["Size"] / 1e6, 1),
                 "cod_in_path": cod_match(x["Path"])} for x in _lsjson(a, fid, "", False)]
        append_rows(f, rows, FILE_FIELDS)
        open(donef, "a").write("<root>\n")
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    t0 = time.time()
    todo = [t for t in tops if t not in done]
    lock = threading.Lock()

    def one(top):
        items = _lsjson(a, fid, top, True)
        return top, [{"path": f"{top}/{x['Path']}", "size_mb": round(x["Size"] / 1e6, 1),
                      "cod_in_path": cod_match(f"{top}/{x['Path']}")} for x in items]

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(one, t): t for t in todo}
        for k, fu in enumerate(as_completed(futs), 1):
            top = futs[fu]
            try:
                top, rows = fu.result()
            except subprocess.CalledProcessError as e:
                print(f"[{k}/{len(todo)}] {top}: ERROR {str(e)[:150]} (will retry next run)", flush=True)
                continue
            with lock:
                append_rows(f, rows, FILE_FIELDS)
                open(donef, "a").write(top + "\n")
            eta = (time.time() - t0) / k * (len(todo) - k) / 60
            print(f"[{k}/{len(todo)}, ~{eta:.0f} min left] {top}: {len(rows)} files, "
                  f"{sum(r['size_mb'] for r in rows)/1e3:.2f} GB", flush=True)
    summarize_files(f)


def summarize_files(f):
    rows = read_csv(f)
    tot = sum(float(r["size_mb"]) for r in rows) / 1e3
    tops = {}
    for r in rows:
        top = r["path"].split("/")[0]
        tops.setdefault(top, [0, 0.0])
        tops[top][0] += 1
        tops[top][1] += float(r["size_mb"]) / 1e3
    print(f"\n{len(rows)} .b3d files, {tot:.1f} GB total -> {f}")
    for k, (n, gb) in sorted(tops.items(), key=lambda kv: -kv[1][1])[:40]:
        print(f"  {k:50s} {n:5d} files {gb:8.2f} GB")
    if len(tops) > 40:
        print(f"  ... {len(tops) - 40} more folders (see {f})")
    print(f"{sum(1 for r in rows if r['cod_in_path'])} files have a change-of-direction word in their path")


def selected_files(a, out):
    rows = read_csv(os.path.join(out, "files.csv"))
    if not rows:
        sys.exit("Run the 'list' stage first.")
    if a.path_regex:
        rx = re.compile(a.path_regex, re.I)
        rows = [r for r in rows if rx.search(r["path"])]
    return rows


def probe_one(a, fid, nimble, r, nbytes, lock):
    """Download just the header of one file and read it. Thread-safe."""
    row = {"path": r["path"]}
    with tempfile.TemporaryDirectory(dir=a.tmp) as td:
        loc = os.path.join(td, "probe.b3d")
        try:
            fsize = int(float(r["size_mb"]) * 1e6) + 10**6
            # 1) first 64 bytes: the .b3d starts with the header length
            #    (8-byte integer), so we know how much to fetch.
            head = rclone(["cat", rpath(a.remote, r["path"]), "--count", "64"],
                          a.remote, fid, capture_output=True).stdout
            need = nbytes
            if len(head) >= 8:
                hlen = int.from_bytes(head[:8], "little")
                if 0 < hlen < fsize:
                    need = hlen + 8 + 4096      # length field + small margin
                else:
                    need = max(nbytes, 25 * 10**6)  # unknown layout: generous probe
            # 2) download just the header (or the whole file if smaller)
            with open(loc, "wb") as fh:
                args = ["cat", rpath(a.remote, r["path"])]
                if need < fsize:
                    args += ["--count", str(need)]
                rclone(args, a.remote, fid, stdout=fh, stderr=subprocess.DEVNULL)
            with lock:   # parse one at a time (nimble thread safety unknown)
                _, meta, trials = header_info(nimble, loc)
            names = [t["trial_name"] or t["original_name"] for t in trials]
            words = [cod_match(t["trial_name"]) or cod_match(t["original_name"]) for t in trials]
            row.update(meta)
            row.update({"status": "ok", "cod_trials": sum(1 for w in words if w),
                        "cod_words": ";".join(sorted({w for w in words if w})),
                        "trial_names": ";".join(names)[:2000]})
        except subprocess.CalledProcessError as e:
            row.update({"status": "download_error", "error": str(e)[:200]})
        except Exception as e:  # header did not fit in the partial download
            row.update({"status": "needs_full", "error": str(e)[:200]})
    return row


def stage_header(a, fid, out, nimble):
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    rows = selected_files(a, out)
    hpath = os.path.join(out, "headers.csv")
    done = {r["path"] for r in read_csv(hpath)}
    todo = [r for r in rows if r["path"] not in done][: a.max_files or None]
    print(f"Header probe: {len(todo)} files to do ({len(done)} already done), "
          f"{a.workers} in parallel", flush=True)
    nbytes = int(a.header_mb * 1e6)
    lock, wlock = threading.Lock(), threading.Lock()
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(probe_one, a, fid, nimble, r, nbytes, lock) for r in todo]
        for k, f in enumerate(as_completed(futs), 1):
            row = f.result()
            with wlock:
                append_rows(hpath, [row], HDR_FIELDS)
            eta = (time.time() - t0) / k * (len(todo) - k) / 60
            print(f"[{k}/{len(todo)}, ~{eta:.0f} min left] {row['path']}: {row['status']}"
                  + (f", {row.get('n_trials')} trials, {row.get('cod_trials')} COD-like"
                     if row["status"] == "ok" else ""), flush=True)


def stage_inspect(a, fid, out, nimble):
    rows = selected_files(a, out)
    hdr = {r["path"]: r for r in read_csv(os.path.join(out, "headers.csv"))}
    if not a.all:
        def is_cand(r):
            h = hdr.get(r["path"])
            return bool(r["cod_in_path"]) or (h and (h["status"] != "ok" or int(h["cod_trials"] or 0) > 0))
        if not hdr and not a.path_regex:
            print("No headers.csv yet: candidates are judged by path only. "
                  "Run the 'header' stage first to also catch COD trial names.")
        rows = [r for r in rows if a.path_regex or is_cand(r)]
    tpath = os.path.join(out, "trials.csv")
    done = {r["path"] for r in read_csv(tpath)}
    todo = [r for r in rows if r["path"] not in done]
    todo.sort(key=lambda r: float(r["size_mb"]))   # small first
    if a.max_files:
        todo = todo[: a.max_files]
    print(f"Inspect: {len(todo)} files, {sum(float(r['size_mb']) for r in todo)/1e3:.1f} GB "
          f"({len(done)} already done)")
    used_gb = 0.0
    for k, r in enumerate(todo, 1):
        gb = float(r["size_mb"]) / 1e3
        if a.max_gb and used_gb + gb > a.max_gb:
            print(f"Stopping: --max-gb {a.max_gb} reached")
            break
        used_gb += gb
        td = tempfile.mkdtemp(dir=a.tmp)
        loc = os.path.join(td, os.path.basename(r["path"]))
        trow = []
        try:
            t0 = time.time()
            rclone(["copyto", rpath(a.remote, r["path"]), loc], a.remote, fid)
            s, meta, trials = header_info(nimble, loc)
            for t in trials:
                ev = inspect_trial(s, t["trial"], a.samples, a.grf_threshold)
                row = {"path": r["path"], **t, **ev}
                row["cod_match"] = cod_match(t["trial_name"]) or cod_match(t["original_name"]) or r["cod_in_path"]
                try:
                    row["duration_s"] = round(float(t["frames"]) * float(t["dt"]), 2)
                except Exception:
                    row["duration_s"] = ""
                row["has_markers"] = int(ev["n_markers_max"] > 0)
                row["has_grf"] = int(ev["grf_frames_over_thr"] > 0)
                trow.append(row)
            nm = sum(x["has_markers"] for x in trow)
            ng = sum(x["has_grf"] for x in trow)
            nc = sum(1 for x in trow if x["cod_match"] and x["has_markers"] and x["has_grf"])
            print(f"[{k}/{len(todo)}] {r['path']} ({gb:.2f} GB, {time.time()-t0:.0f} s): "
                  f"{len(trow)} trials, markers {nm}, GRF {ng}, COD+markers+GRF {nc}")
            if a.keep_dir and nc > 0:
                dst = os.path.join(a.keep_dir, r["path"])
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                os.replace(loc, dst)
                print(f"    kept -> {dst}")
        except Exception as e:
            trow = [{"path": r["path"], "trial": "", "error": str(e)[:300]}]
            print(f"[{k}/{len(todo)}] {r['path']}: ERROR {e}")
        finally:
            if os.path.exists(loc):
                os.remove(loc)
            os.rmdir(td)
        append_rows(tpath, trow, TRIAL_FIELDS)


def stage_report(a, out):
    rows = read_csv(os.path.join(out, "trials.csv"))
    if not rows:
        sys.exit("No trials.csv yet.")
    by = {}
    for r in rows:
        if not r.get("trial"):
            continue
        top = r["path"].split("/")[0]
        d = by.setdefault(top, {"files": set(), "trials": 0, "markers": 0, "grf": 0, "both": 0, "cod_both": 0})
        d["files"].add(r["path"]); d["trials"] += 1
        m, g = r["has_markers"] == "1", r["has_grf"] == "1"
        d["markers"] += m; d["grf"] += g; d["both"] += m and g
        d["cod_both"] += bool(r["cod_match"]) and m and g
    print(f"{'folder':40s} {'files':>5s} {'trials':>6s} {'markers':>7s} {'GRF':>5s} {'both':>5s} {'COD+both':>8s}")
    for k, d in sorted(by.items()):
        print(f"{k:40s} {len(d['files']):5d} {d['trials']:6d} {d['markers']:7d} {d['grf']:5d} {d['both']:5d} {d['cod_both']:8d}")
    cod = [r for r in rows if r.get("cod_match") and r.get("has_markers") == "1" and r.get("has_grf") == "1"]
    print(f"\n{len(cod)} change-of-direction-like trials with markers and GRF:")
    for r in cod[:200]:
        print(f"  {r['path']} | {r['trial_name']} ({r['original_name']}) | {r['duration_s']} s | "
              f"{r['n_markers_max']} markers | word '{r['cod_match']}'")
    errs = [r for r in rows if r.get("error")]
    if errs:
        print(f"\n{len(errs)} rows with errors (see trials.csv)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["list", "header", "inspect", "report"])
    ap.add_argument("--folder", default="core", help=f"one of {list(FOLDERS)} or a raw Drive folder id")
    ap.add_argument("--remote", default="gd", help="rclone remote name (default gd)")
    ap.add_argument("--out", default="addbio_screen", help="output directory for CSVs")
    ap.add_argument("--tmp", default=None, help="where downloads go temporarily (default system tmp)")
    ap.add_argument("--path-regex", default="", help="only files whose path matches (e.g. Camargo)")
    ap.add_argument("--all", action="store_true", help="inspect every file, not only COD candidates")
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--max-gb", type=float, default=0, help="stop inspecting after this much downloaded")
    ap.add_argument("--header-mb", type=float, default=4, help="bytes downloaded for header probe")
    ap.add_argument("--samples", type=int, default=25, help="frames sampled per trial")
    ap.add_argument("--grf-threshold", type=float, default=20.0, help="N; frames above count as GRF")
    ap.add_argument("--min-pieces", type=int, default=30, help="list stage: expand folders until at least this many")
    ap.add_argument("--split-depth", type=int, default=5, help="list stage: levels to descend through single folders")
    ap.add_argument("--restart", action="store_true", help="list stage: start the listing from scratch")
    ap.add_argument("--workers", type=int, default=8, help="parallel rclone calls (list and header stages)")
    ap.add_argument("--keep-dir", default="", help="keep files with COD+markers+GRF trials here")
    a = ap.parse_args()

    fid = FOLDERS.get(a.folder, a.folder)
    out = os.path.join(a.out, a.folder)
    os.makedirs(out, exist_ok=True)
    if a.tmp:
        os.makedirs(a.tmp, exist_ok=True)

    if a.stage == "list":
        stage_list(a, fid, out)
    elif a.stage == "report":
        stage_report(a, out)
    else:
        nimble = load_nimble()
        (stage_header if a.stage == "header" else stage_inspect)(a, fid, out, nimble)


if __name__ == "__main__":
    main()
