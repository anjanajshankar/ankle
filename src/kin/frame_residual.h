
#ifndef FRAME_RESIDUAL_H
#define FRAME_RESIDUAL_H

#include <array>
#include <optional>
#include <vector>

#include "marker_fk.h"
#include "marker_offsets.h"
#include "kin.h"

// One spline coefficient that matters at a given frame: its index k, and its weight B_k(t_i).
struct ActiveCoef {
    int k;       // which coefficient (basis function)
    double B;    // B_k(t_i): its weight in q(t_i), and also dq/dc_k
};

// Everything one frame's residual block needs, apart from the spline coefficients.
struct FrameData {
    // The coefficients with non-zero weight at this frame's time (3 or 4 for a cubic spline).
    // Every coordinate uses the same knots, so these are shared by all coordinates.
    std::vector<ActiveCoef> active;

    // Per marker (same order as MarkerFK::marker_names): observed lab position in mm,
    // or nullopt if the marker was missing in this frame.
    std::vector<std::optional<Coordinate>> observed_marker_positions;

    // Per marker: weight / measurement uncertainty (1/mm). 0 switches a marker off.
    std::vector<double> scale;
};

// Residuals and (optionally) Jacobian for one frame, in the layout Ceres uses.
//
// Parameters: one scalar parameter block per (coordinate j, active coefficient a),
//   ordered p = j * active.size() + a, so params[p][0] is that coefficient's value.
// Residuals: 3 per marker, r = scale * (FK position - observed position), in mm units / mm = unitless.
//   Missing markers give r = 0.
// Jacobian: if jacobians != nullptr, jacobians[p] (if not null) receives dr/dc_p, a column of length 3*M.
//   Computed with the chain rule: dr/dc_{j,a} = dr/dq_j * B_a(t_i), with dr/dq_j from central differences.
void frame_eval(MarkerFK& fk, const FrameData& frame,
                double const* const* params, double* residuals, double** jacobians,
                double step = 1e-6);

// Same as frame_eval, but each marker also has an unknown offset d_m on its body (3 numbers,
// body frame, metres), so its lab position is FK_m(q) + R_m(q) * d_m.
// Parameters: the nq * active.size() coefficient blocks as in frame_eval, followed by one block of
//   size 3 per marker (same order as marker_names), params[nq*A + m] = d_m.
// Jacobian for an offset block: rows of marker m only, scale * 1000 * R_m (mm per metre).
// Requires fk.marker_rotations().
void frame_eval_offsets(MarkerFK& fk, const FrameData& frame,
                        double const* const* params, double* residuals, double** jacobians,
                        double step = 1e-6);

// Same, with shape-only offsets (see ShapeOffsets in marker_offsets.h).
// Parameters: the nq * active.size() coefficient blocks, then one block per group with size > 0
//   (in group order, groups of size 0 skipped), of that group's size k.
// Jacobian for a group block: (3*M x k), row-major = sum over its markers of (d r / d d_m) * N rows.
void frame_eval_shape_offsets(MarkerFK& fk, const FrameData& frame, const ShapeOffsets& shape,
                              double const* const* params, double* residuals, double** jacobians,
                              double step = 1e-6);

// q at this frame from the active coefficients (one value per coordinate).
std::vector<double> frame_q(const FrameData& frame, int num_coords, double const* const* params);

#endif
