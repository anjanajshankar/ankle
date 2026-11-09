
(base) @anjanajshankar ➜ /workspaces/codespaces-blank/ankle (main) $ python src/kin/summarize_step2.py $N/step2/summary.csv | tail -25
spline lambda 1         16    0.758      1284    0.709       2290
spline lambda 3         16    0.889      1464    0.952       2824
spline lambda 10        16    1.174      1620    1.441       3306
-> Pipeline A: f_c = 15 Hz (mean q 0.624 filtered, mean qdd 660)
-> Pipeline B: lambda = 0.03 (mean q 0.619, mean qdd 811)

=== weights: ones   (mean over trials x seeds; q deg, q-ddot deg/s^2)
method                   n   mean q  mean qdd  ankle q  ankle qdd
OpenSim 6 Hz            16    0.720      1537    0.576       3379
OpenSim 6 Hz (q filt)   16    0.868      1537    1.104       3379
OpenSim 8 Hz            16    0.720      1265    0.576       2913
OpenSim 8 Hz (q filt)   16    0.752      1265    0.786       2913
OpenSim 10 Hz           16    0.720       990    0.576       2395
OpenSim 10 Hz (q filt)  16    0.704       990    0.643       2395
OpenSim 12 Hz           16    0.720       768    0.576       1835
OpenSim 12 Hz (q filt)  16    0.684       768    0.560       1835
OpenSim 15 Hz           16    0.720       622    0.576       1221
OpenSim 15 Hz (q filt)  16    0.675       622    0.513       1221
OpenSim 20 Hz           16    0.720       806    0.576       1092
OpenSim 20 Hz (q filt)  16    0.676       806    0.508       1092
spline lambda 0.03      16    0.678       768    0.514       1282
-> Pipeline A: f_c = 15 Hz (mean q 0.675 filtered, mean qdd 622)
-> Pipeline B: lambda = 0.03 (mean q 0.678, mean qdd 768)  (no lambda met the angle condition; lowest angle error shown)

weight safeguard: mean angle error (both pipelines, tuned) learned 0.621 vs ones 0.676 -> use learned weights
(base) @anjanajshankar ➜ /workspaces/codespaces-blank/ankle (main) $ 
