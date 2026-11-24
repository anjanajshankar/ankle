
#include "frame_residual.h"

#include <algorithm>
#include <stdexcept>

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

namespace {

// Lab positions (m) with offsets applied: pos + R * d.
void positions_with_offsets(MarkerFK& fk, const std::vector<double>& q, double const* const* d_blocks,
                            std::vector<double>& pos, std::vector<double>& R) {
    fk.marker_positions(q, pos);
    if (!fk.marker_rotations(R)) throw std::runtime_error("this MarkerFK cannot report marker rotations");
    const size_t M = pos.size() / 3;
    for (size_t m = 0; m < M; ++m) {
        const double* d = d_blocks[m];
        const double* r = &R[9 * m];
        for (int i = 0; i < 3; ++i) pos[3 * m + i] += r[3 * i] * d[0] + r[3 * i + 1] * d[1] + r[3 * i + 2] * d[2];
    }
}

}  // namespace

void frame_eval_offsets(MarkerFK& fk, const FrameData& frame,
                        double const* const* params, double* residuals, double** jacobians, double step) {
    const int nq = static_cast<int>(fk.coordinate_names().size());
    const size_t A = frame.active.size();
    const size_t M = frame.observed_marker_positions.size();
    const size_t R3 = 3 * M;
    double const* const* d_blocks = params + nq * A;

    std::vector<double> q = frame_q(frame, nq, params);
    std::vector<double> pos, R;
    positions_with_offsets(fk, q, d_blocks, pos, R);
    residuals_from_positions(frame, pos, residuals);
    if (jacobians == nullptr) return;

    // offset blocks: d r_m / d d_m = scale_m * 1000 * R_m (rows of marker m only)
    for (size_t m = 0; m < M; ++m) {
        double* J = jacobians[nq * A + m];
        if (J == nullptr) continue;
        std::fill(J, J + R3 * 3, 0.0);
        if (!frame.observed_marker_positions[m]) continue;
        const double* r = &R[9 * m];
        for (int i = 0; i < 3; ++i)
            for (int k = 0; k < 3; ++k) J[(3 * m + i) * 3 + k] = frame.scale[m] * 1000.0 * r[3 * i + k];
    }

    // coefficient blocks: chain rule with dr/dq by central differences (offsets move with the body)
    std::vector<double> r_plus(R3), r_minus(R3), dr_dq(R3), Rtmp;
    for (int j = 0; j < nq; ++j) {
        bool needed = false;
        for (size_t a = 0; a < A; ++a) needed = needed || jacobians[j * A + a] != nullptr;
        if (!needed) continue;
        const double q0 = q[j];
        q[j] = q0 + step;
        positions_with_offsets(fk, q, d_blocks, pos, Rtmp);
        residuals_from_positions(frame, pos, r_plus.data());
        q[j] = q0 - step;
        positions_with_offsets(fk, q, d_blocks, pos, Rtmp);
        residuals_from_positions(frame, pos, r_minus.data());
        q[j] = q0;
        for (size_t i = 0; i < R3; ++i) dr_dq[i] = (r_plus[i] - r_minus[i]) / (2.0 * step);
        for (size_t a = 0; a < A; ++a) {
            double* col = jacobians[j * A + a];
            if (col == nullptr) continue;
            for (size_t i = 0; i < R3; ++i) col[i] = dr_dq[i] * frame.active[a].B;
        }
    }
}

void frame_eval_shape_offsets(MarkerFK& fk, const FrameData& frame, const ShapeOffsets& shape,
                              double const* const* params, double* residuals, double** jacobians,
                              double step) {
    const size_t nq = fk.coordinate_names().size();
    const size_t A = frame.active.size();
    const size_t M = frame.observed_marker_positions.size();
    const size_t R3 = 3 * M, G = shape.members.size();
    const size_t nc = nq * A;

    // group unknowns -> per-marker offsets
    std::vector<const double*> z(G, nullptr);
    std::vector<int> block_of_group(G, -1);
    for (size_t g = 0, b = 0; g < G; ++g)
        if (shape.size[g] > 0) { block_of_group[g] = static_cast<int>(b); z[g] = params[nc + b]; ++b; }
    std::vector<double> d;
    shape.to_offsets(z, M, d);

    std::vector<const double*> P(nc + M);
    for (size_t p = 0; p < nc; ++p) P[p] = params[p];
    for (size_t m = 0; m < M; ++m) P[nc + m] = &d[3 * m];

    if (jacobians == nullptr) { frame_eval_offsets(fk, frame, P.data(), residuals, nullptr, step); return; }

    bool need_z = false;
    for (size_t g = 0; g < G; ++g) need_z = need_z || (block_of_group[g] >= 0 && jacobians[nc + block_of_group[g]]);
    std::vector<double> Jd(need_z ? M * R3 * 3 : 0);
    std::vector<double*> J(nc + M, nullptr);
    for (size_t p = 0; p < nc; ++p) J[p] = jacobians[p];
    if (need_z) for (size_t m = 0; m < M; ++m) J[nc + m] = &Jd[m * R3 * 3];
    frame_eval_offsets(fk, frame, P.data(), residuals, J.data(), step);

    // chain rule: d r / d z_g = sum over markers m in g of (d r / d d_m) * N_g[rows of m]
    for (size_t g = 0; g < G; ++g) {
        if (block_of_group[g] < 0) continue;
        double* Jz = jacobians[nc + block_of_group[g]];
        if (Jz == nullptr) continue;
        const int k = shape.size[g];
        std::fill(Jz, Jz + R3 * k, 0.0);
        for (size_t l = 0; l < shape.members[g].size(); ++l) {
            const size_t m = shape.members[g][l];
            const double* Jm = &Jd[m * R3 * 3];
            for (size_t r = 3 * m; r < 3 * m + 3; ++r)          // d_m only moves marker m's rows
                for (int c = 0; c < k; ++c) {
                    double v = 0;
                    for (int a = 0; a < 3; ++a) v += Jm[r * 3 + a] * shape.basis[g][(3 * l + a) * k + c];
                    Jz[r * k + c] = v;
                }
        }
    }
}
