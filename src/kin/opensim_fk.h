
#ifndef OPENSIM_FK_H
#define OPENSIM_FK_H

#include <memory>
#include <string>
#include <vector>

#include <OpenSim/OpenSim.h>

#include "marker_fk.h"

// MarkerFK backed by an OpenSim model (the same FK that fk_check verified).
//
// Free coordinates = every coordinate that is not locked and is not the dependent side of a
// CoordinateCouplerConstraint (in Rajagopal: knee_angle_r_beta follows knee_angle_r).
// Dependent coordinates are computed from their coupler function on every call, so the
// optimizer never sees them as separate unknowns.
//
// Bodies can be excluded (e.g. the arms): their markers are not used, and the coordinates of
// the joints that move them are held at the model's default values instead of being solved.
class OpenSimFK : public MarkerFK {
public:
    // wanted_markers: markers to report, in this order; names not in the model are skipped
    // (see marker_names() for the ones actually used). Empty = every marker in the model.
    // excluded_body_prefixes: e.g. {"humerus", "ulna", "radius", "hand"} drops both arms.
    OpenSimFK(const std::string& model_path, const std::vector<std::string>& wanted_markers,
              const std::vector<std::string>& excluded_body_prefixes = {});

    const std::vector<std::string>& coordinate_names() const override { return coord_names_; }
    const std::vector<std::string>& marker_names() const override { return marker_names_; }
    void marker_positions(const std::vector<double>& q, std::vector<double>& out_m) override;

    // Orientation (in the lab) of the frame each marker's location is expressed in, at the q of the
    // last marker_positions() call. A marker offset d (that frame, metres) moves the marker by R * d.
    bool marker_rotations(std::vector<double>& R_out) override;

    // true for rotational coordinates (radians), false for translations (metres).
    const std::vector<bool>& rotational() const { return rotational_; }

    // Range limits from the model, and whether OpenSim enforces them ("clamped").
    // OpenSim's own IK keeps clamped coordinates inside their range; spline_ik does the same.
    const std::vector<bool>& clamped() const { return clamped_; }
    const std::vector<double>& range_min() const { return range_min_; }
    const std::vector<double>& range_max() const { return range_max_; }
    OpenSim::Model& model() { return *model_; }

    // Name of the body each marker is attached to (same order as marker_names()).
    const std::vector<std::string>& marker_body_names() const { return marker_bodies_; }

    // Name of the frame each marker's location is expressed in (usually its body), and that
    // location in metres (3 per marker), as set in the model.
    const std::vector<std::string>& marker_frame_names() const { return marker_frames_; }
    const std::vector<double>& marker_locations() const { return marker_locations_; }

private:
    struct Coupler {
        const OpenSim::Coordinate* dependent;
        std::vector<int> independent_free_index;      // index into q, or -1 if not free
        std::vector<const OpenSim::Coordinate*> independent;
        const OpenSim::Function* function;
    };

    std::unique_ptr<OpenSim::Model> model_;
    SimTK::State* state_ = nullptr;
    std::vector<const OpenSim::Coordinate*> free_;
    std::vector<std::string> coord_names_;
    std::vector<bool> rotational_;
    std::vector<bool> clamped_;
    std::vector<double> range_min_, range_max_;
    std::vector<Coupler> couplers_;
    std::vector<const OpenSim::Marker*> markers_;
    std::vector<std::string> marker_names_;
    std::vector<std::string> marker_bodies_;
    std::vector<std::string> marker_frames_;
    std::vector<double> marker_locations_;
};

#endif
