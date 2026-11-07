
import nimblephysics as nimble
import pandas as pd
import numpy as np

# 1. Load the b3d file using the correct class
print("Loading B3D file cleanly...")
subject = nimble.biomechanics.SubjectOnDisk("../assets/Camargo21/AB06_split0.b3d")

# 2. Extract the embedded biological skeleton model
# Processing passes are 0-indexed; the final pass contains the optimized kinematics/dynamics
last_pass_idx = subject.getNumProcessingPasses() - 1
model = subject.readSkel(last_pass_idx, ignoreGeometry=True)

# 3. Read the trajectories and parameters from trial 0
trial_idx = 0
num_frames = subject.getTrialLength(trial_idx)
dof_names = [model.getDofByIndex(i).getName() for i in range(model.getNumDofs())]

print(f"Loaded Subject data with {num_frames} total frames and {len(dof_names)} Degrees of Freedom.")

# 4. Pull down the frame arrays cleanly using sequential positional arguments
print("Extracting trial frames into memory...")
frames = subject.readFrames(
    trial_idx,        # trial (int)
    0,                # startFrame (int)
    num_frames,       # numFramesToRead (int)
    False,            # includeSensorData (bool)
    True              # includeProcessingPasses (bool)
)

# 5. Extract pre-computed Inverse Dynamics torques directly out of the file payload
print("Extracting baked Inverse Dynamics joint moments...")
timestamps = []
id_torques = []

for f_idx, frame in enumerate(frames):
    # Pull timestamps
    t = f_idx * subject.getTrialTimestep(trial_idx)
    timestamps.append(t)

    # Bypassing the manual C++ solver call to completely prevent Segmentation Faults.
    # Pulling `.tau` extracts the optimized torques directly from the AddBiomechanics pipeline.
    tau = frame.processingPasses[last_pass_idx].tau
    id_torques.append(tau)

# 6. Structuring output matrices to CSV
print("Exporting data matrix...")
id_matrix = np.array(id_torques)
df_id = pd.DataFrame(id_matrix, columns=[f"{name}_torque" for name in dof_names])
df_id.insert(0, 'time', timestamps)

df_id.to_csv("inverse_dynamics_output.csv", index=False)
print("Success! Torques safely extracted and stored in 'inverse_dynamics_output.csv'.")

