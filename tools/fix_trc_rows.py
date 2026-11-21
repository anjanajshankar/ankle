"""Pad .trc data rows that are shorter than the header says (a missing marker at the END of a row
written with too few empty cells). OpenSim refuses such files ("Unexpected number of columns").
A short row is padded with empty cells only if every cell of its last, incomplete marker is empty,
so a padded cell can only ever be a missing value; anything else is reported and the file is left
unchanged. The original is kept as <file>.orig.
usage: python3 fix_trc_rows.py file1.trc [file2.trc ...]      (files that are fine are skipped)
"""
import os, sys
n_fixed = n_bad = 0
for path in sys.argv[1:]:
    lines = open(path).read().split("\n")
    names = [n for n in lines[3].split("\t")[2:] if n.strip()]
    expected = 2 + 3 * len(names)
    k = next(i for i, l in enumerate(lines) if "X1\t" in l)
    out, padded, problem = [], 0, None
    for i, l in enumerate(lines):
        if i <= k or not l.strip():
            out.append(l); continue
        f = l.rstrip("\r").split("\t")
        if len(f) < expected:
            start = 2 + 3 * ((len(f) - 2) // 3)
            if any(c.strip() for c in f[start:]):
                problem = f"row {i + 1}: {len(f)} cells, last marker not empty"; break
            f += [""] * (expected - len(f)); padded += 1
        elif len(f) > expected and any(c.strip() for c in f[expected:]):
            problem = f"row {i + 1}: {len(f)} cells with data beyond the header"; break
        out.append("\t".join(f[:expected]))
    if problem:
        print(f"{path}: NOT changed, {problem}"); n_bad += 1; continue
    if padded:
        if not os.path.exists(path + ".orig"):
            os.rename(path, path + ".orig")
        open(path, "w").write("\n".join(out))
        print(f"{path}: padded {padded} short rows (original kept as .orig)"); n_fixed += 1
print(f"{n_fixed} files fixed, {n_bad} files with problems, {len(sys.argv) - 1 - n_fixed - n_bad} already fine")
