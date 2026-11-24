
#!/bin/bash
# Shape-only offsets (--fit-offsets 10 --shape-only) on the two offset seeds and the no-offset control,
# compared with plain spline IK and full offsets (off10, from run_offsets.sh / check 3).
#
# usage (from the ankle folder):  nohup bash src/kin/run_shape.sh > shape_log.txt 2>&1 &
cd /workspaces/codespaces-blank/ankle
S=assets/SimulationDataAndSetupFiles-4.0/Synth
MODEL=assets/SimulationDataAndSetupFiles-4.0/Scale/subject_scaled_walk.osim
WEIGHTS=assets/SimulationDataAndSetupFiles-4.0/IK/ik_setup_walk_lowtoe.xml
for P in $S/walkoff_s1 $S/walkoff_s2 $S/walk_s1; do
  ./build/spline_ik $MODEL $P.trc ${P}_opensim.mot ${P}_spline_shape.mot 0.02 $WEIGHTS --no-arms --smooth 1 \
      --fit-offsets 10 --shape-only > ${P}_spline_shape_log.txt 2>&1 || echo "spline_ik failed for $P, see ${P}_spline_shape_log.txt"
done
grep -h "shape-only" $S/walk_s1_spline_shape_log.txt
for s in 1 2; do
  P=$S/walkoff_s$s
  echo "=== seed $s (true offsets 10 mm)"
  grep -hE "Ceres Solver Report|RMS marker error" ${P}_spline_shape_log.txt
  python src/kin/compare_truth.py $P opensim=${P}_opensim.mot spline=${P}_spline.mot off10=${P}_spline_off.mot \
      shape=${P}_spline_shape.mot | grep -E "coordinate|_r |_l |lumbar|mean"
  python src/kin/compare_offsets.py ${P}_true_offsets.csv ${P}_spline_shape_offsets.csv
done
P=$S/walk_s1
echo "=== control (no true offsets)"
grep -hE "Ceres Solver Report|RMS marker error|marker offsets" ${P}_spline_shape_log.txt
python src/kin/compare_truth.py $P spline=${P}_spline_s1.mot off10=${P}_spline_off.mot shape=${P}_spline_shape.mot \
    | grep -E "coordinate|_r |_l |lumbar|mean"
