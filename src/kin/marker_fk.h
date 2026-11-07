
#ifndef MARKER_FK_H
#define MARKER_FK_H

#include <string>
#include <vector>

// Forward kinematics for markers: FK_m(q, s, p_m) for every marker at once.
// Scaling s and marker offsets p_m are fixed inside the implementation;
// only the coordinates q change.
//
// This interface knows nothing about OpenSim or Ceres, so the IK code can be
// tested with a simple toy model and run for real with OpenSim.
class MarkerFK {
public:
    virtual ~MarkerFK() = default;

    // Names of the free coordinates, in the order q is passed.
    virtual const std::vector<std::string>& coordinate_names() const = 0;

    // Names of the markers, in the order positions are returned.
    virtual const std::vector<std::string>& marker_names() const = 0;

    // q: one value per free coordinate (radians for rotations, metres for translations).
    // out_m: resized to 3 * number of markers, filled with x0, y0, z0, x1, ... in metres (lab frame).
    virtual void marker_positions(const std::vector<double>& q, std::vector<double>& out_m) = 0;
};

#endif
