
#include "frame_residual.h"

namespace {

// r = scale * (predicted_mm - observed_mm) for every marker; 0 for missing ones.
void residuals_from_positions(const FrameData& frame, const std::vector<double>& pos_m, double* r) {
    const size_t M = frame.observed_marker_positions.size();
    for (size_t m = 0; m < M; ++m) {
        const std::optional<Coordinate>& obs = frame.observed_marker_positions[m];
        if (!obs) { r[3 * m] = r[3 * m + 1] = r[3 * m + 2] = 0.0; continue; }
        const double o[3] = {obs->x_mm, obs->y_mm, obs->z_mm};
        for (int d = 0; d < 3; ++d) r[3 * m + d] = frame.scale[m] * (pos_m[3 * m + d] * 1000.0 - o[d]);
    }
}

}  // namespace

std::vector<double> frame_q(const FrameData& frame, int num_coords, double const* const* params) {
    const size_t A = frame.active.size();
    std::vector<double> q(num_coords, 0.0);
    for (int j = 0; j < num_coords; ++j)
        for (size_t a = 0; a < A; ++a) q[j] += frame.active[a].B * params[j * A + a][0];
    return q;
}

void frame_eval(MarkerFK& fk, const FrameData& frame,
                double const* const* params, double* residuals, double** jacobians, double step) {
    const int nq = static_cast<int>(fk.coordinate_names().size());
    const size_t A = frame.active.size();
    const size_t R = 3 * frame.observed_marker_positions.size();

    // 1. q(t_i) = sum_a B_a * c_{j,a}, then FK, then residuals.
    std::vector<double> q = frame_q(frame, nq, params);
    std::vector<double> pos;
    fk.marker_positions(q, pos);
    residuals_from_positions(frame, pos, residuals);

    if (jacobians == nullptr) return;

    // 2. dr/dq_j by central differences: nudge one coordinate up and down, rerun FK.
    std::vector<double> r_plus(R), r_minus(R), dr_dq(R);
    for (int j = 0; j < nq; ++j) {
        // Skip the FK calls if Ceres doesn't need this coordinate's columns.
        bool needed = false;
        for (size_t a = 0; a < A; ++a) needed = needed || jacobians[j * A + a] != nullptr;
        if (!needed) continue;

        const double q0 = q[j];
        q[j] = q0 + step;
        fk.marker_positions(q, pos);
        residuals_from_positions(frame, pos, r_plus.data());
        q[j] = q0 - step;
        fk.marker_positions(q, pos);
        residuals_from_positions(frame, pos, r_minus.data());
        q[j] = q0;
        for (size_t i = 0; i < R; ++i) dr_dq[i] = (r_plus[i] - r_minus[i]) / (2.0 * step);

        // 3. chain rule: dr/dc_{j,a} = dr/dq_j * B_a.
        for (size_t a = 0; a < A; ++a) {
            double* col = jacobians[j * A + a];
            if (col == nullptr) continue;
            for (size_t i = 0; i < R; ++i) col[i] = dr_dq[i] * frame.active[a].B;
        }
    }
}
