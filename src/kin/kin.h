
#ifndef KIN_H
#define KIN_H

#include <vector>
#include <map>
#include <optional>
#include <string>
#include <compare>

// Segment identifier. References into a standard skeleton model, eg Rajagopal.
using SegID = int;

// Person identifier. References into a table of participant info to get ht,wt etc.
using PersonID = int;

// A Frame is a timepoint.
struct Frame {
   // index of this frame. 0 means starting time point
   int idx;
};

// Euclidean coordinates 
struct Coordinate {
   double x_mm;
   double y_mm;
   double z_mm;
};

// Coordinate of one marker at specific timeframe. Initially we only have observed coords, after we run ik we get the errors.
struct MarkerPt {
   std::optional<Coordinate> observed_coord;
   Coordinate measurement_uncertainty;
   // Error is calculated by IK to get smooth angle trajectories that minimize error.
   Coordinate error_in_coord; 
   Frame frame;
};

// Full trajectory of a marker in ascending time order. 
struct MarkerTrajectory {
   std::vector<MarkerPt> marker_pts;
};

using SegLocalPosition = Coordinate;
using MarkerName = std::string;

// Where the marker is placed (both segment and position relative to segment). 
struct MarkerInfo {
   SegID seg_id;
   SegLocalPosition seg_local_position;
};

// Complete trace info for one activity for one person -- has all marker infos over time. After running ik marker position are also augmented w/ errors so we get smooth marker trajectories.
struct ActivityTrace {
   std::map<MarkerName, MarkerInfo> marker_info_by_name;
   std::map<MarkerName, MarkerTrajectory> trajectories_by_marker;
   double time_step_millis;
   PersonID person_id;
};

// We define a joint as just the meeting point of two segments.
struct JointID {
   auto operator<=> (const JointID&) const = default; 
   SegID seg_proximal;
   SegID seg_distal;
};

// An angle between two segments in 3D (all in radians).
struct Angle {
   double flexion_rad;
   double abduction_rad;
   double rotation_rad;
};

// Maximum and minimum range of angles.
struct Range { double min; double max; };

struct AngleConstraints {
   std::optional<Range> max_flexion_rad;
   std::optional<Range> max_abduction_rad;
   std::optional<Range> max_rotation_rad;
};

// Angle between two segments at a time point.
struct AnglePt {
   Frame frame;
   Angle angle;
};

// Full trajectory over time of the angle at a specific joint.
struct JointAngleTrajectory {
   JointID joint_id;
   std::vector<AnglePt> angle_pts;
};

// Position and orientation of the pelvis (root segment) in the lab frame at one time point.
// position: pelvis origin in mm.
// orientation: tilt, list, and rotation of pelivs relative to lab axis (radians).
struct RootPt { Frame frame; Coordinate position; Angle orientation; };

// Full trajectory over time of a skeleton during an activity.
struct SkeletonTrajectory {
   std::vector<RootPt> root_trajectory;
   std::vector<JointAngleTrajectory> joint_angle_trajectories;
};

// Physical dimensions of a segment, for now only the length.
struct SegDimension {
   double length_mm;
};

// Physical dimensions of an entire skeleton.
struct SkeletonScale {
   // Size of each segment for this skeleton.
   std::map<SegID, SegDimension> segment_dimensions;
   // Constraints on the angle at each segment, for this skeleton
   // (usually common across all skeletons, but we keep it per-skeleton to allow customization).
   std::map<JointID, AngleConstraints> joint_limits;
};


#endif
