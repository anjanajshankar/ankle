#!/bin/bash
# Offset prior sweep: 2 and 5 mm on the two offset seeds and on the no-offset control.
cd /workspaces/codespaces-blank/ankle
S=assets/SimulationDataAndSetupFiles-4.0/Synth
MODEL=assets/SimulationDataAndSetupFiles-4.0/Scale/subject_scaled_walk.osim
WEIGHTS=assets/SimulationDataAndSetupFiles-4.0/IK/ik_setup_walk_lowtoe.xml
for sig in 2 5; do
  for P in $S/walkoff_s1 $S/walkoff_s2 $S/walk_s1; do
    ./build/spline_ik $MODEL $P.trc ${P}_opensim.mot ${P}_spline_off$sig.mot 0.02 $WEIGHTS --no-arms --smooth 1 \
        --fit-offsets $sig > ${P}_spline_off${sig}_log.txt 2>&1
  done
done
for s in 1 2; do
  P=$S/walkoff_s$s
  echo "=== seed $s (true offsets 10 mm)"
  python src/kin/compare_truth.py $P opensim=${P}_opensim.mot spline=${P}_spline.mot off10=${P}_spline_off.mot \
      off5=${P}_spline_off5.mot off2=${P}_spline_off2.mot | grep -E "coordinate|_l |mean"
  for sig in 5 2; do
    echo "-- offsets, prior $sig mm"
    python src/kin/compare_offsets.py ${P}_true_offsets.csv ${P}_spline_off${sig}_offsets.csv | grep -E "_l |all"
  done
done
P=$S/walk_s1
echo "=== control (no true offsets)"
python src/kin/compare_truth.py $P spline=${P}_spline_s1.mot off10=${P}_spline_off.mot off5=${P}_spline_off5.mot \
    off2=${P}_spline_off2.mot | grep -E "coordinate|_l |mean"
grep -h "marker offsets" ${P}_spline_off5_log.txt ${P}_spline_off2_log.txt
