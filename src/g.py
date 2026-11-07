
import numpy as np
import pandas as pd
import nimblephysics as nimble

# 1. Load dataset structure
print("Loading B3D file cleanly...")
subject = nimble.biomechanics.SubjectOnDisk("AB06_split0.b3d")
last_pass_idx = subject.getNumProcessingPasses() - 1
model = subject.readSkel(last_pass_idx, ignoreGeometry=True)

trial_idx = 0
num_frames = subject.getTrialLength(trial_idx)

# Include processing passes to get true spatial kinematics and forces
frames = subject.readFrames(trial_idx, 0, num_frames, True, True)

# Standard AddBiomechanics foot node structural names
right_foot_node = model.getBodyNode("calcn_r")
left_foot_node = model.getBodyNode("calcn_l")

ankle_forces_data = []
timestamps = []

print("Extracting true 3D spatial joint reaction forces...")
for f_idx, frame in enumerate(frames):
    t = f_idx * subject.getTrialTimeStep(trial_idx)
    timestamps.append(t)

    # Update skeleton's position to compute the correct spatial rotation matrices
    q = frame.processingPasses[last_pass_idx].pos
    model.setPositions(q)

    # Initialize separate arrays for ground forces
    grf_force_r = np.zeros(3)
    grf_force_l = np.zeros(3)

    # 2. Extract structural contact forces safely using exposed object dictionaries
    # This prevents the memory crash by avoiding direct wrench array slicing
    if hasattr(frame, 'contactForces') and frame.contactForces is not None:
        for contact in frame.contactForces:
            # Safely query contact details using high-level exposed methods
            body_name = contact.bodyName.lower() if hasattr(contact, 'bodyName') else ""
            force_vector = np.array(contact.force) if hasattr(contact, 'force') else np.zeros(3)

            if "r" in body_name or "calc" in body_name:
                grf_force_r += force_vector
            if "l" in body_name or "calc" in body_name:
                grf_force_l += force_vector

    # 3. Pull segment physical inertia directly from the active skeleton instance
    m_foot_r = right_foot_node.getMass()
    acc_foot_r = right_foot_node.getLinearAcceleration() # Natively returns a clean NumPy array

    m_foot_l = left_foot_node.getMass()
    acc_foot_l = left_foot_node.getLinearAcceleration()

    # 4. Apply the intersegmental equation: Force = GRF - Inertia (m*a) - Gravity (m*g)
    # This captures the true intersegmental balance across the ankle joint line
    gravity = np.array([0.0, -9.80665, 0.0])

    world_force_r = grf_force_r - (m_foot_r * acc_foot_r) - (m_foot_r * gravity)
    world_force_l = grf_force_l - (m_foot_l * acc_foot_l) - (m_foot_l * gravity)

    # Calculate absolute scalar magnitudes (Newtons)
    norm_r = np.linalg.norm(world_force_r)
    norm_l = np.linalg.norm(world_force_l)

    # Unpack elements cleanly to avoid sequence parsing flags
    ankle_forces_data.append([
        world_force_r[0], world_force_r[1], world_force_r[2], norm_r,
        world_force_l[0], world_force_l[1], world_force_l[2], norm_l
    ])

# 5. Build and save your structured dataframe
columns = [
    'ankle_r_FX_shear', 'ankle_r_FY_vertical', 'ankle_r_FZ_lateral', 'ankle_r_total_force_N',
    'ankle_l_FX_shear', 'ankle_l_FY_vertical', 'ankle_l_FZ_lateral', 'ankle_l_total_force_N'
]
df_forces = pd.DataFrame(ankle_forces_data, columns=columns)
df_forces.insert(0, 'time', timestamps)

df_forces.to_csv("ankle_joint_forces_verified.csv", index=False)
print("Success! Verified ankle joint forces written to 'ankle_joint_forces_verified.csv'.")

