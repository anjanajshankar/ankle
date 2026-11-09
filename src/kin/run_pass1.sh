
#!/bin/bash
# Pass 1 of the marker-weight rule: OpenSim IK on every prepared trial (setups made by prep_trials.py
# with all weights 1), then each marker's residuals appended to <work>/marker_residuals_pass1.csv.
#
# usage (from the ankle folder):  bash src/kin/run_pass1.sh <work_dir>
W=${1:?usage: run_pass1.sh <work_dir>}
B=/workspaces/codespaces-blank/ankle/build
OUT=$W/marker_residuals_pass1.csv
rm -f $OUT
tail -n +2 $W/trials.csv | while IFS=, read pid trial dir model trc grf t0 t1 ic1 ic2 mass; do
  if ! opensim-cmd run-tool $dir/ik_setup.xml > $dir/opensim_ik_log.txt 2>&1; then
    echo "P$pid $trial: IK FAILED (see $dir/opensim_ik_log.txt)"; continue
  fi
  worst=$(grep -o "RMS = [0-9.e-]*" $dir/opensim_ik_log.txt | awk '{print $3}' | sort -g | tail -1)
  $B/ik_cost $model $trc $dir/ik_setup.xml --no-arms --marker-csv $OUT P${pid}_$trial=$dir/opensim_ik.mot \
      2>&1 | grep -E "^P" | awk -v w="$worst" '{printf "%s  weighted cost %s  RMS %s mm  (worst frame RMS %.1f mm)\n", $1, $2, $3, w*1000}'
done
echo "residuals written to $OUT"
