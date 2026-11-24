"""Scale the generic Rajagopal model to each subject from its static "neutral" trial (OpenSim ScaleTool:
segment scaling from marker pairs, then marker placement). Writes the result over
<work>/<subject>/model_ik.osim, so ik_setup.xml, trials.csv and run_pass1.sh use it unchanged.
Per subject: generic_markers.osim = generic model with the dataset's markers (mapped markers take the
generic, anatomical location; others are added to their body at the AddBiomechanics location as a first
guess; unmapped generic markers removed) -> scale_setup_full.xml (measurements below, longest static
segment, subject mass, mass distribution kept; static pose from the mapped markers minus --off; then
every marker moved to its static position) -> opensim-cmd -> model_ik.osim. Reruns start from the
generic model; rerun after prep_trials_addbio.py (which rewrites model_ik.osim).
usage: python3 src/kin/scale_static.py assets/SoccerCOD work/SoccerCOD --generic <generic.osim> --off m1,m2 [--dry-run]
"""
import argparse, glob, json, os, shutil, subprocess, sys, xml.etree.ElementTree as ET
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep_trials import trc_info  # noqa: E402

MAP = {  # Soccer COD: SIAS_* / SIPS_* labels are swapped front-to-back (checked from static-trial geometry)
       'SIPS_left': 'LASI', 'SIPS_right': 'RASI', 'SIAS_left': 'LPSI', 'SIAS_right': 'RPSI',
       'epi_lat_left': 'LLFC', 'epi_med_left': 'LMFC', 'epi_lat_right': 'RLFC', 'epi_med_right': 'RMFC',
       'mal_lat_left': 'LLMAL', 'mal_med_left': 'LMMAL', 'mal_lat_right': 'RLMAL', 'mal_med_right': 'RMMAL',
       'cal_back_left': 'LCAL', 'cal_back_right': 'RCAL', 'toe_left': 'LTOE', 'toe_right': 'RTOE',
       'forefood_lat_left': 'LMT5', 'forefood_lat_right': 'RMT5',
       'acrom_left': 'LACR', 'acrom_right': 'RACR', 'C7': 'C7', 'CLAV': 'CLAV'}

def measurements():
    arms = {f'{b}_{s}': 'X Y Z' for b in ('humerus', 'ulna', 'radius', 'hand') for s in 'rl'}
    m = {'pelvis': ([('SIPS_left', 'SIPS_right')], {'pelvis': 'X Y Z'}),            # ASIS-ASIS width, uniform
         'torso_depth': ([('C7', 'CLAV')], {'torso': 'X'}),
         'torso_height': ([('acrom_left', 'SIPS_left'), ('acrom_right', 'SIPS_right')], {'torso': 'Y Z', **arms})}
    for side, s in (('right', 'r'), ('left', 'l')):
        m[f'thigh_{s}'] = ([(f'SIPS_{side}', f'epi_lat_{side}'), (f'SIPS_{side}', f'epi_med_{side}')],
                           {f'femur_{s}': 'X Y Z', f'patella_{s}': 'X Y Z'})
        m[f'shank_{s}'] = ([(f'epi_lat_{side}', f'mal_lat_{side}'), (f'epi_med_{side}', f'mal_med_{side}')],
                           {f'tibia_{s}': 'X Y Z', f'talus_{s}': 'X Y Z', f'calcn_{s}': 'X Y Z', f'toes_{s}': 'X Y Z'})  # feet scaled with the shank
    return m

def markers_of(osim):
    out = {}
    for mk in ET.parse(osim).getroot().iter('Marker'):
        body = (mk.findtext('body') or mk.findtext('socket_parent_frame') or '').strip().split('/')[-1]
        out[mk.get('name')] = (body, (mk.findtext('location') or '0 0 0').strip())
    return out

def build_generic(generic, addbio, wanted, out_path):
    tree = ET.parse(generic); root = tree.getroot()
    objs = next(root.iter('MarkerSet')).find('objects')
    gen = {mk.get('name'): mk for mk in objs.findall('Marker')}
    v3 = any(mk.find('body') is not None for mk in gen.values())
    ab = markers_of(addbio)
    keep = {MAP[n]: n for n in wanted if n in MAP and MAP[n] in gen}
    for gname, el in gen.items():
        if gname in keep: el.set('name', keep[gname])
        else: objs.remove(el)
    added = []
    for n in wanted:
        if n in keep.values() or n not in ab: continue
        body, loc = ab[n]
        el = ET.SubElement(objs, 'Marker', name=n)
        if v3: ET.SubElement(el, 'body').text = body
        else: ET.SubElement(el, 'socket_parent_frame').text = f'/bodyset/{body}'
        ET.SubElement(el, 'location').text = f' {loc}'
        ET.SubElement(el, 'fixed').text = 'false'
        added.append(n)
    tree.write(out_path, xml_declaration=True, encoding='UTF-8')
    return sorted(keep.values()), added

def setup_xml(name, mass, model, trc, t0, t1, meas, pose, allm, scaled, final, motion, factors):
    mx = []
    for mname, (pairs, bodies) in meas.items():
        p = ''.join(f'<MarkerPair><markers> {a} {b}</markers></MarkerPair>' for a, b in pairs)
        bs = ''.join(f'<BodyScale name="{b}"><axes> {ax}</axes></BodyScale>' for b, ax in bodies.items())
        mx.append(f'<Measurement name="{mname}"><apply>true</apply><MarkerPairSet><objects>{p}</objects></MarkerPairSet>'
                  f'<BodyScaleSet><objects>{bs}</objects></BodyScaleSet></Measurement>')
    tasks = '\n'.join(f'<IKMarkerTask name="{m}"><apply>{"true" if m in pose else "false"}</apply><weight>1</weight></IKMarkerTask>' for m in allm)
    nl = '\n'
    return f"""<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="40000">
<ScaleTool name="{name}">
<mass>{mass}</mass><height>-1</height><age>-1</age>
<GenericModelMaker><model_file>{model}</model_file><marker_set_file>Unassigned</marker_set_file></GenericModelMaker>
<ModelScaler>
<apply>true</apply>
<scaling_order> measurements</scaling_order>
<MeasurementSet><objects>
{nl.join(mx)}
</objects><groups /></MeasurementSet>
<ScaleSet><objects /><groups /></ScaleSet>
<marker_file>{trc}</marker_file>
<time_range> {t0:.6f} {t1:.6f}</time_range>
<preserve_mass_distribution>true</preserve_mass_distribution>
<output_model_file>{scaled}</output_model_file>
<output_scale_file>{factors}</output_scale_file>
</ModelScaler>
<MarkerPlacer>
<apply>true</apply>
<IKTaskSet><objects>
{tasks}
</objects><groups /></IKTaskSet>
<marker_file>{trc}</marker_file>
<coordinate_file>Unassigned</coordinate_file>
<time_range> {t0:.6f} {t1:.6f}</time_range>
<output_motion_file>{motion}</output_motion_file>
<output_model_file>{final}</output_model_file>
<output_marker_file>Unassigned</output_marker_file>
<max_marker_movement>-1</max_marker_movement>
</MarkerPlacer>
</ScaleTool>
</OpenSimDocument>
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data_dir'); ap.add_argument('work_dir')
    ap.add_argument('--generic', required=True); ap.add_argument('--off', default='')
    ap.add_argument('--static-pattern', default='neutral'); ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    data, work, generic = (os.path.abspath(p) for p in (a.data_dir, a.work_dir, a.generic))
    off = {m for m in a.off.split(',') if m}
    for model_ik in sorted(glob.glob(os.path.join(work, '*', 'model_ik.osim'))):
        sdir = os.path.dirname(model_ik); subj = os.path.basename(sdir)
        statics = sorted(glob.glob(os.path.join(data, subj, f'*{a.static_pattern}*.trc')))
        if not statics:
            print(f'{subj}: no static trial, skipped'); continue
        trc = max(statics, key=lambda f: trc_info(f)[2] - trc_info(f)[1])
        addbio = os.path.join(sdir, 'model_ik_addbio.osim')
        if not os.path.exists(addbio): shutil.copy(model_ik, addbio)
        try: mass = json.load(open(os.path.join(data, subj, 'subject.json'))).get('mass_kg', -1)
        except (OSError, ValueError): mass = -1
        names, t0, t1 = trc_info(trc)
        ab = markers_of(addbio)
        wanted = [n for n in names if n in ab]
        gen_path = os.path.join(sdir, 'generic_markers.osim')
        mapped, added = build_generic(generic, addbio, wanted, gen_path)
        meas = {}
        for mname, (pairs, bodies) in measurements().items():
            ok = [(x, y) for x, y in pairs if x in mapped and y in mapped]
            if ok: meas[mname] = (ok, bodies)
            else: print(f'   WARNING {subj}: measurement {mname} has no usable marker pair, bodies keep generic size')
        pose = [m for m in mapped if m not in off]
        allm = mapped + added
        out = {k: os.path.join(sdir, v) for k, v in dict(scaled='scaled_only.osim', final='model_scaled.osim',
               motion='static_pose_scaled.mot', factors='scale_factors.xml', setup='scale_setup_full.xml',
               log='scale_full_log.txt').items()}
        open(out['setup'], 'w').write(setup_xml(subj, mass, gen_path, trc, t0, t1, meas, pose, allm,
                                                out['scaled'], out['final'], out['motion'], out['factors']))
        print(f'{subj}: static {os.path.basename(trc)} ({t1 - t0:.2f} s), mass {mass} kg; {len(mapped)} anatomical '
              f'markers (static pose), {len(added)} others placed; {len(meas)} measurements')
        if a.dry_run: continue
        with open(out['log'], 'w') as f:
            ok = subprocess.run(['opensim-cmd', 'run-tool', out['setup']], stdout=f, stderr=subprocess.STDOUT).returncode == 0
        if not ok or not os.path.exists(out['final']):
            print(f'   scaling FAILED, see {out["log"]}'); continue
        fac = []
        for s in ET.parse(out['factors']).getroot().iter('Scale'):
            seg, f = s.findtext('segment').strip(), s.findtext('scales').split()
            if seg in ('pelvis', 'femur_r', 'femur_l', 'tibia_r', 'tibia_l', 'calcn_r', 'calcn_l', 'torso'):
                fac.append(f'{seg} ' + '/'.join(f'{float(x):.2f}' for x in f))
        print('   scale factors (X/Y/Z): ' + ', '.join(fac))
        shutil.copy(out['final'], model_ik)
        print('   model_ik.osim now = scaled generic model with placed markers')

if __name__ == '__main__':
    main()
