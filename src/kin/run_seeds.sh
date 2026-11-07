
#!/bin/bash
# Repeat the synthetic accuracy test with different random seeds (different STA and noise).
# For each seed: make fake markers, run OpenSim IK and spline IK (lambda 10 and 100),
# then score all three against the truth.
#
# usage (from the ankle folder):  bash src/kin/run_seeds.sh 2 3 4 5
cd /workspaces/codespaces-blank/ankle
D=assets/SimulationDataAndSetupFiles-4.0
MODEL=$D/Scale/subject_scaled_walk.osim
WEIGHTS=$D/IK/ik_setup_walk_lowtoe.xml
mkdir -p $D/Synth

for s in "$@"; do
  P=$D/Synth/walk_s$s
  echo "=== seed $s: generating markers, running OpenSim IK and spline IK (a few minutes)"
  ./build/synth_markers $MODEL $D/IK/my_results_walk/ik_lowtoe.mot $P --seed $s > $P.synth_log.txt 2>&1

  ( cd $D/IK && sed -e "s#<marker_file>[^<]*</marker_file>#<marker_file>../Synth/walk_s$s.trc</marker_file>#" \
                    -e "s#<results_directory>[^<]*</results_directory>#<results_directory>../Synth</results_directory>#" \
                    -e "s#<output_motion_file>[^<]*</output_motion_file>#<output_motion_file>../Synth/walk_s${s}_opensim.mot</output_motion_file>#" \
                    ik_setup_walk_lowtoe.xml > ik_setup_synth_s$s.xml \
      && opensim-cmd run-tool ik_setup_synth_s$s.xml > ../Synth/walk_s${s}_opensim_log.txt 2>&1 )

  for L in 0.3 1 3; do
    ./build/spline_ik $MODEL $P.trc ${P}_opensim.mot ${P}_spline_s$L.mot 0.02 $WEIGHTS --no-arms --smooth $L \
        > ${P}_spline_s${L}_log.txt 2>&1
  done

  python src/kin/compare_truth.py $P opensim=${P}_opensim.mot s0.3=${P}_spline_s0.3.mot s1=${P}_spline_s1.mot s3=${P}_spline_s3.mot \
      | grep -E "coordinate|ankle|mean"
done
