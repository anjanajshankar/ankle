"""Protocol rule 5: exclude a trial (in both pipelines) if its marker RMS exceeds 2 x the dataset's median
trial RMS in either pipeline. A = OpenSim IK (from marker_residuals_pass1.csv), B = spline_ik final RMS.
Writes <work>/excluded_trials.csv. usage: python3 tools/apply_exclusion.py work/<Dataset>
"""
import csv, math, os, re, sys, statistics, collections
w = sys.argv[1].rstrip('/')
trials = {f"P{r['participant']}_{r['trial']}": r for r in csv.DictReader(open(f'{w}/trials.csv'))}
ss, n = collections.defaultdict(float), collections.defaultdict(int)
for r in csv.reader(open(f'{w}/marker_residuals_pass1.csv')):
    if len(r) >= 6 and r[0] in trials:
        try: ss[r[0]] += float(r[4]); n[r[0]] += int(r[5])
        except ValueError: pass
A = {t: math.sqrt(ss[t] / n[t]) for t in trials if n[t]}
B = {}
for t, r in trials.items():
    f = os.path.join(r['dir'], 'spline_ik_log.txt')
    if os.path.exists(f):
        m = re.findall(r'RMS marker error.*-> end ([0-9.]+)', open(f).read())
        if m: B[t] = float(m[-1])
mA, mB = statistics.median(A.values()), statistics.median(B.values())
print(f"{len(A)} trials with A, {len(B)} with B; median A {mA:.1f} mm (limit {2*mA:.1f}), median B {mB:.1f} mm (limit {2*mB:.1f})")
missing = [t for t in trials if t not in A or t not in B]
if missing: print("missing a pipeline result:", missing)
out = [(t, A.get(t, float('nan')), B.get(t, float('nan'))) for t in trials
       if A.get(t, 0) > 2 * mA or B.get(t, 0) > 2 * mB]
with open(f'{w}/excluded_trials.csv', 'w', newline='') as fh:
    cw = csv.writer(fh); cw.writerow(['trial', 'rms_A_mm', 'rms_B_mm', 'limit_A_mm', 'limit_B_mm'])
    for t, a, b in out: cw.writerow([t, f'{a:.1f}', f'{b:.1f}', f'{2*mA:.1f}', f'{2*mB:.1f}'])
print(f"excluded {len(out)} of {len(trials)}:", ", ".join(f"{t} (A {a:.1f}, B {b:.1f})" for t, a, b in out) or "none")
