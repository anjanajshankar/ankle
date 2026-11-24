
#ifndef TOY_FK_H
#define TOY_FK_H

#include <cmath>
#include "marker_fk.h"

// The 2D toy chain: pelvis (x0, y0, phi0) -> segment a -> segment b -> segment c.
// Angles are measured from the downward vertical. Two markers on the pelvis and two per
// segment, offset sideways from the long axis, so every rotation is well defined.
// (Without the pelvis markers, pelvis_tilt and theta_a could not be told apart:
// both only rotate segment a.)
// Used to test the IK machinery against a known answer, without OpenSim.
class ToyFK : public MarkerFK {
public:
    ToyFK(double sa = 0.45, double sb = 0.42, double sc = 0.25) : s_{sa, sb, sc} {
        coords_ = {"pelvis_tx", "pelvis_ty", "pelvis_tilt", "theta_a", "theta_b", "theta_c"};
        markers_ = {"P1", "P2", "A1", "A2", "B1", "B2", "C1", "C2"};
    }
    const std::vector<std::string>& coordinate_names() const override { return coords_; }
    const std::vector<std::string>& marker_names() const override { return markers_; }

    // Body frame of each segment: x sideways (n), y pointing back up the segment (-u), z out of plane.
    // So a marker at "along, side" sits at local (side, -along, 0), and R is a plain rotation by the
    // segment's absolute angle.
    bool marker_rotations(std::vector<double>& R) override {
        R.assign(9 * markers_.size(), 0.0);
        for (size_t m = 0; m < markers_.size(); ++m) {
            const double a = angles_[m], c = std::cos(a), s = std::sin(a);
            double* r = &R[9 * m];
            r[0] = c; r[1] = -s; r[3] = s; r[4] = c; r[8] = 1.0;
        }
        return true;
    }

    void marker_positions(const std::vector<double>& q, std::vector<double>& out) override {
        out.assign(3 * markers_.size(), 0.0);
        angles_.assign(markers_.size(), 0.0);
        double x = q[0], y = q[1], ang = q[2];
        // pelvis markers: 10 cm left and right of the pelvis origin, along the pelvis's own x axis
        for (int k = 0; k < 2; ++k) {
            const double side = (k == 0 ? -0.10 : 0.10);
            out[3 * k + 0] = x + side * std::cos(ang);
            out[3 * k + 1] = y + side * std::sin(ang);
            angles_[k] = ang;
        }
        for (int seg = 0; seg < 3; ++seg) {
            ang += q[3 + seg];                       // angles add down the chain
            const double ux = std::sin(ang), uy = -std::cos(ang);   // along the segment
            const double nx = std::cos(ang), ny = std::sin(ang);    // sideways
            // markers at 1/3 and 2/3 of the segment, 4 cm and 6 cm to the side
            for (int k = 0; k < 2; ++k) {
                const double along = s_[seg] * (k + 1) / 3.0, side = 0.04 + 0.02 * k;
                out[3 * (2 + 2 * seg + k) + 0] = x + along * ux + side * nx;
                out[3 * (2 + 2 * seg + k) + 1] = y + along * uy + side * ny;
                angles_[2 + 2 * seg + k] = ang;
            }
            x += s_[seg] * ux;                       // next joint at the end of this segment
            y += s_[seg] * uy;
        }
    }

private:
    double s_[3];
    std::vector<double> angles_;   // absolute angle of each marker's body at the last call
    std::vector<std::string> coords_, markers_;
};

#endif
