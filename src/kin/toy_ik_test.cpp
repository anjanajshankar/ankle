
// toy_ik_test: the spline IK machinery on the 2D toy chain, with a known answer.
//
// 1. Make up a smooth "true" motion for the 6 toy coordinates.
// 2. Compute marker positions with FK and add 1 mm of noise -> fake marker data.
// 3. Start the splines at the wrong place (all angles 0) and let Ceres solve.
// 4. Check that the solved angles match the true ones.
// No OpenSim needed, so if this passes, Ceres + splines + frame_eval are working together.

#include <ceres/ceres.h>

#include <cmath>
#include <cstdio>
#include <random>

#include "bspline.h"
#include "frame_residual.h"
#include "toy_fk.h"

namespace {

class FrameCost : public ceres::CostFunction {
public:
    FrameCost(MarkerFK* fk, FrameData frame) : fk_(fk), frame_(std::move(frame)) {
        set_num_residuals(static_cast<int>(3 * frame_.observed_marker_positions.size()));
        const size_t n = fk_->coordinate_names().size() * frame_.active.size();
        for (size_t p = 0; p < n; ++p) mutable_parameter_block_sizes()->push_back(1);
    }
    bool Evaluate(double const* const* params, double* residuals, double** jacobians) const override {
        frame_eval(*fk_, frame_, params, residuals, jacobians);
        return true;
    }
private:
    MarkerFK* fk_;
    FrameData frame_;
};

double truth(int j, double t) {
    const double base[6] = {0.3 * t, 0.95, 0.05, 0.3, -0.4, 0.2};   // m, m, then radians
    const double amp[6] = {0.0, 0.02, 0.05, 0.5, 0.6, 0.3};
    return base[j] + amp[j] * std::sin(2 * M_PI * t + j);
}

}  // namespace

int main() {
    ToyFK fk;
    const int nq = 6, M = static_cast<int>(fk.marker_names().size());
    const int F = 101;
    const double dt = 0.01;
    const BSpline spline = make_uniform(0.0, (F - 1) * dt, 10);
    const int K = spline.num_coeffs();

    std::mt19937 rng(1);
    std::normal_distribution<double> noise_mm(0.0, 1.0);

    // Starting guess: pelvis at roughly the right height, every angle 0.
    std::vector<std::vector<double>> c(nq, std::vector<double>(K, 0.0));
    for (int k = 0; k < K; ++k) c[1][k] = 0.9;

    ceres::Problem problem;
    std::vector<double> q(nq), pos;
    for (int i = 0; i < F; ++i) {
        const double t = i * dt;
        FrameData f;
        for (int k = 0; k < K; ++k) {
            const double b = basis(spline, k, t);
            if (b != 0.0) f.active.push_back({k, b});
        }
        for (int j = 0; j < nq; ++j) q[j] = truth(j, t);
        fk.marker_positions(q, pos);
        f.observed_marker_positions.resize(M);
        f.scale.assign(M, 1.0);   // weight 1, sigma 1 mm
        for (int m = 0; m < M; ++m)
            f.observed_marker_positions[m] = Coordinate{pos[3 * m] * 1000 + noise_mm(rng), pos[3 * m + 1] * 1000 + noise_mm(rng), 0.0};
        if (i == 50) f.observed_marker_positions[2].reset();   // one missing marker sample

        std::vector<double*> blocks;
        for (int j = 0; j < nq; ++j)
            for (const ActiveCoef& ac : f.active) blocks.push_back(&c[j][ac.k]);
        problem.AddResidualBlock(new FrameCost(&fk, std::move(f)), nullptr, blocks);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    if (!ceres::IsSparseLinearAlgebraLibraryTypeAvailable(options.sparse_linear_algebra_library_type))
        options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = 50;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::printf("%s\n", summary.BriefReport().c_str());

    double worst_deg = 0;
    for (int i = 0; i < F; ++i)
        for (int j = 2; j < nq; ++j)
            worst_deg = std::max(worst_deg, std::abs(evaluate(spline, c[j], i * dt) - truth(j, i * dt)) * 180 / M_PI);
    std::printf("max angle error vs truth: %.2f deg  ->  %s\n", worst_deg, worst_deg < 3.0 ? "PASS" : "CHECK");
    return worst_deg < 3.0 ? 0 : 1;
}
