"""Re-place each subject's markers on its model from the subject's static "neutral" trial
(OpenSim ScaleTool with segment scaling OFF and marker placement ON).
For every <work>/<subject>/model_ik.osim: keeps the original as model_ik_addbio.osim (first time only),
writes scale_setup.xml (neutral .trc averaged over its whole length, weights 1, --off markers off),
runs opensim-cmd -> model_placed.osim, copies it over model_ik.osim (so ik_setup.xml, trials.csv and
run_pass1.sh use it unchanged) and reports how far the markers moved. Reruns always start from
model_ik_addbio.osim. Running prep_trials_addbio.py again rewrites model_ik.osim, so rerun this after it.
usage: python3 src/kin/place_markers_static.py assets/SoccerCOD work/SoccerCOD --off m1,m2 [--static-pattern neutral_segment_0]
"""
import argparse, glob, json, math, os, shutil, subprocess, sys, xml.etree.ElementTree as ET
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep_trials import trc_info  # noqa: E402

def marker_locations(osim):
    out = {}
    for m in ET.parse(osim).getroot().iter('Marker'):
        loc = (m.findtext('location') or '').split()
        if len(loc) == 3:
            out[m.get('name')] = [float(v) for v in loc]
    return out

def setup_xml(name, mass, model, trc, t0, t1, markers, off, out_model, out_motion):
    tasks = "\n".join(f'<IKMarkerTask name="{m}"><apply>{"false" if m in off else "true"}</apply><weight>1</weight></IKMarkerTask>'
                      for m in markers)
    return f"""<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="40000">
<ScaleTool name="{name}">
<mass>{mass}</mass><height>-1</height><age>-1</age>
<GenericModelMaker><model_file>{model}</model_file></GenericModelMaker>
<ModelScaler><apply>false</apply></ModelScaler>
<MarkerPlacer>
<apply>true</apply>
<IKTaskSet><objects>
{tasks}
</objects><groups /></IKTaskSet>
<marker_file>{trc}</marker_file>
<coordinate_file>Unassigned</coordinate_file>
<time_range> {t0:.6f} {t1:.6f}</time_range>
<output_motion_file>{out_motion}</output_motion_file>
<output_model_file>{out_model}</output_model_file>
<output_marker_file>Unassigned</output_marker_file>
<max_marker_movement>-1</max_marker_movement>
</MarkerPlacer>
</ScaleTool>
</OpenSimDocument>
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data_dir'); ap.add_argument('work_dir')
    ap.add_argument('--off', default='')
    ap.add_argument('--static-pattern', default='neutral')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    data, work = os.path.abspath(a.data_dir), os.path.abspath(a.work_dir)
    off = {m for m in a.off.split(',') if m}
    for model_ik in sorted(glob.glob(os.path.join(work, '*', 'model_ik.osim'))):
        sdir = os.path.dirname(model_ik); subj = os.path.basename(sdir)
        statics = sorted(glob.glob(os.path.join(data, subj, f'*{a.static_pattern}*.trc')))
        if not statics:
            print(f'{subj}: no static trial matching *{a.static_pattern}*.trc, skipped'); continue
        trc = max(statics, key=lambda f: trc_info(f)[2] - trc_info(f)[1])   # longest static segment
        orig = os.path.join(sdir, 'model_ik_addbio.osim')
        if not os.path.exists(orig):
            shutil.copy(model_ik, orig)
        try:
            mass = json.load(open(os.path.join(data, subj, 'subject.json'))).get('mass_kg', -1)
        except (OSError, ValueError):
            mass = -1
        names, t0, t1 = trc_info(trc)
        before = marker_locations(orig)
        markers = [m for m in names if m in before]
        out_model, out_motion = os.path.join(sdir, 'model_placed.osim'), os.path.join(sdir, 'static_pose.mot')
        setup = os.path.join(sdir, 'scale_setup.xml')
        open(setup, 'w').write(setup_xml(subj, mass, orig, trc, t0, t1, markers, off, out_model, out_motion))
        print(f'{subj}: static {os.path.basename(trc)} ({t1 - t0:.2f} s), {len(markers)} markers '
              f'({len([m for m in markers if m not in off])} used for the static pose)')
        if a.dry_run:
            continue
        log = os.path.join(sdir, 'scale_log.txt')
        with open(log, 'w') as f:
            ok = subprocess.run(['opensim-cmd', 'run-tool', setup], stdout=f, stderr=subprocess.STDOUT).returncode == 0
        if not ok or not os.path.exists(out_model):
            print(f'   marker placement FAILED, see {log}'); continue
        after = marker_locations(out_model)
        moved = sorted(((math.dist(before[m], after[m]) * 1000, m) for m in markers if m in after), reverse=True)
        print(f'   markers moved: median {moved[len(moved)//2][0]:.1f} mm; largest: '
              + ', '.join(f'{m} {d:.0f}' for d, m in moved[:5]))
        shutil.copy(out_model, model_ik)
        print('   model_ik.osim now has the re-placed markers (original: model_ik_addbio.osim)')

if __name__ == '__main__':
    main()
