
import os
import numpy as np
import pandas as pd
import nimblephysics as nimble

def process_b3d_dataset(b3d_file_path):
    print("=" * 60)
    print(f"Loading B3D file cleanly from: {b3d_file_path}")
    print("=" * 60)

    if not os.path.exists(b3d_file_path):
        print(f"Error: File not found at {b3d_file_path}")
        return
    subject = nimble.biomechanics.SubjectOnDisk(b3d_file_path)

    # 1. Extract the embedded biological skeleton model from the final optimized pass
    last_pass_idx = subject.getNumProcessingPasses() - 1
    model = subject.readSkel(last_pass_idx, ignoreGeometry=True)

    trial_idx = 0
    num_frames = subject.getTrialLength(trial_idx)
    dof_names = [model.getDofByIndex(i).getName() for i in range(model.getNumDofs())]

    print(f"-> Loaded Subject data with {num_frames} total frames.")
    print(f"-> Found {len(dof_names)} Degrees of Freedom (DoFs) in the model.")

    # 2. Pull down the frame arrays cleanly using sequential positional arguments
    print("\nExtracting trial frames into memory...")
    frames = subject.readFrames(
        trial_idx,        # trial (int)
        0,                # startFrame (int)
        num_frames,       # numFramesToRead (int)
        False,            # includeSensorData (bool)
        True              # includeProcessingPasses (bool)
    )

    timestamps = []
    id_torques = []

    # We will dynamically collect markers frame-by-frame using a nested dictionary
    # format: { frame_index: { marker_name: [x, y, z] } }
    frame_markers_map = {}
    all_observed_marker_names = set()

    print("Parsing frame payloads for internal dynamics and unconstrained trajectories...")
    for f_idx, frame in enumerate(frames):
        t = f_idx * subject.getTrialTimestep(trial_idx)
        timestamps.append(t)

        # A. Pull baked Inverse Dynamics moments from the target optimized pass
        pass_data = frame.processingPasses[last_pass_idx]
        id_torques.append(pass_data.tau)

        # B. Pull unconstrained marker data via the correct list iteration format
        markers_this_frame = frame.markerObservations
        frame_markers_map[f_idx] = {}

        # Iterating directly through the list pairs (name, 3D vector)
        if markers_this_frame is not None:
            for marker_pair in markers_this_frame:
                m_name = marker_pair[0]  # The string name of the marker
                m_pos = marker_pair[1]   # The 3D position vector numpy array [x,y,z]

                all_observed_marker_names.add(m_name)
                frame_markers_map[f_idx][m_name] = m_pos

    print("\n" + "-" * 50)
    print("Formatting and writing output data...")
    print("-" * 50)

    # 3. Export Torques
    id_matrix = np.array(id_torques)
    df_id = pd.DataFrame(id_matrix, columns=[f"{name}_torque" for name in dof_names])
    df_id.insert(0, 'time', timestamps)

    torque_output_name = "inverse_dynamics_output.csv"
    df_id.to_csv(torque_output_name, index=False)
    print(f"[✓] Torques safely stored in '{torque_output_name}'")

    # 4. Export Markers Dynamically
    sorted_marker_names = sorted(list(all_observed_marker_names))
    print(f"-> Detected {len(sorted_marker_names)} total unique markers across the trial.")

    marker_data_dict = {'time': timestamps}
    for m_name in sorted_marker_names:
        marker_data_dict[f"{m_name}_X"] = []
        marker_data_dict[f"{m_name}_Y"] = []
        marker_data_dict[f"{m_name}_Z"] = []

    for f_idx in range(len(frames)):
        for m_name in sorted_marker_names:
            if m_name in frame_markers_map[f_idx]:
                pos_3d = frame_markers_map[f_idx][m_name]
                marker_data_dict[f"{m_name}_X"].append(pos_3d[0])
                marker_data_dict[f"{m_name}_Y"].append(pos_3d[1])
                marker_data_dict[f"{m_name}_Z"].append(pos_3d[2])
            else:
                marker_data_dict[f"{m_name}_X"].append(np.nan)
                marker_data_dict[f"{m_name}_Y"].append(np.nan)
                marker_data_dict[f"{m_name}_Z"].append(np.nan)

    df_markers = pd.DataFrame(marker_data_dict)
    marker_output_name = "experimental_markers_output.csv"
    df_markers.to_csv(marker_output_name, index=False)
    print(f"[✓] Unconstrained 3D marker trajectories stored in '{marker_output_name}'")
    print("=" * 60)
    print("Unique Marker Names found:")
    print(sorted_marker_names)
    print("=" * 60)

if __name__ == "__main__":
    B3D_TARGET = "../assets/Camargo21/AB06_split0.b:q3d"
    process_b3d_dataset(B3D_TARGET)

