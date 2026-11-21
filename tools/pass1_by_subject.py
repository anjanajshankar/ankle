"""Pass-1 marker residuals per subject: median marker RMS, mean pelvis-marker RMS, 3 worst markers.
usage: python3 tools/pass1_by_subject.py work/<Dataset>
"""
import csv, math, sys, collections
w = sys.argv[1].rstrip('/')
subj_of = {f"P{r['participant']}_{r['trial']}": r['participant'] for r in csv.DictReader(open(f'{w}/trials.csv'))}
ss, n = collections.defaultdict(float), collections.defaultdict(int)
for r in csv.reader(open(f'{w}/marker_residuals_pass1.csv')):
    if len(r) < 6 or r[0] not in subj_of: continue
    try: k = (subj_of[r[0]], r[2]); ss[k] += float(r[4]); n[k] += int(r[5])
    except ValueError: continue
pelvis_names = ('SIAS', 'SIPS', 'ASI', 'PSI', 'ASIS', 'PSIS', 'SACR')
done = sorted({s for s, _ in ss})
print(f"{len(done)} of {len(set(subj_of.values()))} subjects have results")
for subj in done:
    rms = sorted((math.sqrt(ss[k] / n[k]), k[1]) for k in ss if k[0] == subj and n[k])
    pel = [r for r, m in rms if any(p in m.upper() for p in pelvis_names)]
    print(f"{subj:8s} median {rms[len(rms)//2][0]:5.1f} mm  pelvis {sum(pel)/max(1,len(pel)):5.1f} mm  worst: "
          + ", ".join(f"{m} {r:.0f}" for r, m in rms[-3:][::-1]))
