
"""Repair .trc files whose marker-name row is broken over two lines (a marker name containing a
line break in the export). Joins the broken row, keeps the original as <file>.orig, and checks that
the number of names matches the number of X/Y/Z columns.

usage: python3 fix_trc_header.py file1.trc [file2.trc ...]
"""
import os, sys
for path in sys.argv[1:]:
    text = open(path).read()
    lines = text.split("\n")
    # standard layout: 0-2 header, 3 = Frame#/Time/names, then the X1/Y1/Z1 row
    k = next(i for i, l in enumerate(lines) if "X1\t" in l)
    if k == 4:
        print(f"{path}: header already fine"); continue
    if not lines[3].startswith("Frame#"):
        print(f"{path}: unexpected header, not changed"); continue
    joined = "".join(lines[3:k]).replace("\r", "")   # drop the stray line breaks inside the name row
    print(f"{path}: name row was split over {k - 3} lines; joined at "
          f"{[lines[i].split(chr(9))[-1] or lines[i].split(chr(9))[-2:] for i in range(3, k - 1)]}")
    new = lines[:3] + [joined] + lines[k:]
    names = [n for n in joined.split("\t")[2:] if n.strip()]
    ncoord = len([c for c in lines[k].split("\t") if c.strip()])
    if 3 * len(names) != ncoord:
        print(f"   WARNING: {len(names)} names but {ncoord} X/Y/Z columns, not changed"); continue
    if not os.path.exists(path + ".orig"):
        os.rename(path, path + ".orig")
    open(path, "w").write("\n".join(new))
    print(f"   fixed: {len(names)} markers (original kept as {os.path.basename(path)}.orig)")
