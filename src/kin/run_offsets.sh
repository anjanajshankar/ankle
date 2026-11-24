#!/bin/bash
# Synthetic test of marker-offset estimation on the Rajagopal walk.
# Markers are generated with misplaced markers (--offset-error 10 mm per axis) plus the usual STA
# and noise. Then: OpenSim IK, spline_ik without offsets, spline_ik with --fit-offsets 10.
# Scores: joint angles vs truth (compare_truth.py) and offsets vs true offsets (compare_offsets.py).
#
# usage (from the ankle folder):  bash src/kin/run_offsets.sh 1 2
cd /workspaces/codespaces-blank/ankle
D=assets/SimulationDataAndSetupFiles-4.0
MODEL=$D/Scale/subject_scaled_walk.osim
WEIGHTS=$D/IK/ik_setup_walk_lowtoe.xml
mkdir -p $D/Synth

for s in "$@"; do
  P=$D/Synth/walkoff_s$s
  echo "=== seed $s: markers with 10 mm offsets, OpenSim IK, spline IK with and without offsets"
  ./build/synth_markers $MODEL $D/IK/my_results_walk/ik_lowtoe.mot $P --seed $s --offset-error 10 \
      > $P.synth_log.txt 2>&1 || { echo "synth_markers failed, see $P.synth_log.txt"; continue; }

  ( cd $D/IK && sed -e "s#<marker_file>[^<]*</marker_file>#<marker_file>../Synth/walkoff_s$s.trc</marker_file>#" \
                    -e "s#<results_directory>[^<]*</results_directory>#<results_directory>../Synth</results_directory>#" \
                    -e "s#<output_motion_file>[^<]*</output_motion_file>#<output_motion_file>../Synth/walkoff_s${s}_opensim.mot</output_motion_file>#" \
                    ik_setup_walk_lowtoe.xml > ik_setup_synthoff_s$s.xml \
      && opensim-cmd run-tool ik_setup_synthoff_s$s.xml > ../Synth/walkoff_s${s}_opensim_log.txt 2>&1 )

  ./build/spline_ik $MODEL $P.trc ${P}_opensim.mot ${P}_spline.mot 0.02 $WEIGHTS --no-arms --smooth 1 \
      > ${P}_spline_log.txt 2>&1
  ./build/spline_ik $MODEL $P.trc ${P}_opensim.mot ${P}_spline_off.mot 0.02 $WEIGHTS --no-arms --smooth 1 \
      --fit-offsets 10 > ${P}_spline_off_log.txt 2>&1
  grep -E "RMS marker error|marker offsets" ${P}_spline_log.txt ${P}_spline_off_log.txt

  python src/kin/compare_truth.py $P opensim=${P}_opensim.mot spline=${P}_spline.mot spline_off=${P}_spline_off.mot \
      | grep -E "coordinate|hip|knee|ankle|lumbar|mean"
  python src/kin/compare_offsets.py ${P}_true_offsets.csv ${P}_spline_off_offsets.csv
done
