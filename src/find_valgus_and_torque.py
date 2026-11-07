
import nimblephysics as nimble
import pandas as pd
import numpy as np

# 1. Load the b3d file using the correct class
print("Loading B3D file cleanly...")
subject = nimble.biomechanics.SubjectOnDisk("../assets/Camargo21/AB06_split0.b3d")

# 2. Extract the embedded biological skeleton model
last_pass_idx = subject.getNumProcessingPasses() - 1
model = subject.readSkel(last_pass_idx, ignoreGeometry=True)

# 3. Read the trajectories and parameters from trial 0
trial_idx = 0
num_frames = subject.getTrialLength(trial_idx)
dof_names = [model.getDofByIndex(i).getName() for i in range(model.getNumDofs())]

print(f"Loaded Subject data with {num_frames} total frames and {len(dof_names)} Degrees of Freedom.")

# --- SCANNING FOR KNEE VALGUM/CORONAL PLANE DOFs ---
# OpenSim templates commonly label this 'knee_adduction_l'/'knee_adduction_r'
# or sometimes use 'knee_rotation' / 'knee_valgus' depending on the constraint model.
print("\nScanning Degrees of Freedom for frontal-plane knee motion...")
left_knee_idx = None
right_knee_idx = None

target_keywords = ['knee_adduction', 'knee_valgus', 'knee_rotation']
for idx, name in enumerate(dof_names):
    low_name = name.lower()
    print(f"----> Found DoF: '{name}' at index {idx}")
    if any(kw in low_name for kw in target_keywords):
        if '_l' in low_name or 'left' in low_name:
            left_knee_idx = idx
            print(f"-> Found Left Frontal Knee DoF: '{name}' at index {idx}")
        elif '_r' in low_name or 'right' in low_name:
            right_knee_idx = idx
            print(f"-> Found Right Frontal Knee DoF: '{name}' at index {idx}")

if left_knee_idx is None and right_knee_idx is None:
    print("[!] Warning: Could not automatically detect a frontal plane knee DoF.")
    print("If your model template only has 'knee_flexion', your skeleton is restricted to 1-DoF hinge joints.")
# --------------------------------------------------

# 4. Pull down the frame arrays cleanly using sequential positional arguments
print("\nExtracting trial frames into memory...")
frames = subject.readFrames(
    trial_idx,        # trial (int)
    0,                # startFrame (int)
    num_frames,       # numFramesToRead (int)
    False,            # includeSensorData (bool)
    True              # includeProcessingPasses (bool)
)

# 5. Extract pre-computed Inverse Dynamics torques and Kinematic angles
print("Extracting joint moments and dynamic angles...")
timestamps = []
id_torques = []
left_valgum_angles = []
right_valgum_angles = []

for f_idx, frame in enumerate(frames):
    # Pull timestamps
    t = f_idx * subject.getTrialTimestep(trial_idx)
    timestamps.append(t)

    # Pull the target processing pass data package
    pass_data = frame.processingPasses[last_pass_idx]

    # 1. Torques
    id_torques.append(pass_data.tau)

    # 2. Kinematics (positions are natively in radians in .b3d)
    pos_array = pass_data.pos

    if left_knee_idx is not None:
        left_valgum_angles.append(np.degrees(pos_array[left_knee_idx]))
    else:
        left_valgum_angles.append(np.nan)

    if right_knee_idx is not None:
        right_valgum_angles.append(np.degrees(pos_array[right_knee_idx]))
    else:
        right_valgum_angles.append(np.nan)

# 6. Structuring output matrices to CSV
print("Exporting data matrix...")
id_matrix = np.array(id_torques)

# Create baseline DataFrame for torques
df_out = pd.DataFrame(id_matrix, columns=[f"{name}_torque" for name in dof_names])

# Insert timestamps and the specific extracted dynamic angles
df_out.insert(0, 'time', timestamps)
df_out['knee_valgum_angle_l'] = left_valgum_angles
df_out['knee_valgum_angle_r'] = right_valgum_angles

df_out.to_csv("inverse_dynamics_and_kinematics_output.csv", index=False)
print("Success! Data cleanly stored in 'inverse_dynamics_and_kinematics_output.csv'.")

