"""Write missing marker values in .trc files as NaN instead of empty cells.
OpenSim's .trc reader drops an empty cell at the very end of a line, so a frame in which the LAST
marker is missing looks one column short and the whole file is refused. NaN is read as "missing"
by OpenSim IK and by our own reader (parseTRC: empty or NaN -> missing), so nothing else changes.
Only data cells are touched. Originals kept as .orig.
usage: python3 fill_trc_nan.py file1.trc [file2.trc ...]      (files without gaps are skipped)
"""
import os, sys
n_fixed = 0
for path in sys.argv[1:]:
    lines = open(path).read().split("\n")
    names = [n for n in lines[3].split("\t")[2:] if n.strip()]
    expected = 2 + 3 * len(names)
    k = next(i for i, l in enumerate(lines) if "X1\t" in l)
    out, filled, problem = [], 0, None
    for i, l in enumerate(lines):
        if i <= k or not l.strip():
            out.append(l); continue
        f = l.rstrip("\r").split("\t")
        if len(f) != expected or not f[0].strip() or not f[1].strip():
            problem = f"row {i + 1}: {len(f)} cells (expected {expected}) or no frame/time"; break
        for j in range(2, expected):
            if not f[j].strip():
                f[j] = "NaN"; filled += 1
        out.append("\t".join(f))
    if problem:
        print(f"{path}: NOT changed, {problem}"); continue
    if filled:
        if not os.path.exists(path + ".orig"):
            os.rename(path, path + ".orig")
        open(path, "w").write("\n".join(out))
        n_fixed += 1
print(f"{n_fixed} files rewritten with NaN for missing values, {len(sys.argv) - 1 - n_fixed} unchanged")
