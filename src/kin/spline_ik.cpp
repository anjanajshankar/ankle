
// spline_ik: inverse kinematics where each coordinate is a cubic B-spline over the whole trial.
//
// Unknowns:  spline coefficients (number of knot intervals + 3 per free coordinate).
// Residuals: for every frame and marker, weight/sigma * (FK position - observed position).
// Fixed:     segment scaling and marker offsets (taken from the scaled model).
// Start:     splines fitted to an existing IK result (e.g. OpenSim's), then refined by Ceres.
//
// usage: spline_ik <model.osim> <markers.trc> <init_ik.mot> <out.mot>
//                  [knot_spacing_s = 0.05] [ik_setup.xml (marker weights), or - for none]
//                  [grf.mot] [dense_spacing_s = 0.01] [before_s = 0.03] [after_s = 0.10]
// With a GRF file, knots are placed every knot_spacing_s, plus every dense_spacing_s from
// before_s ahead of each foot contact / toe-off to after_s after it.
//
// Options (anywhere on the command line):
//   --no-arms                   ignore arm markers and hold the arm coordinates fixed
//   --exclude-bodies a,b,...    same for any bodies whose names start with these prefixes
//   --smooth LAMBDA             roughness penalty: adds LAMBDA * integral of q-ddot^2 dt for every
//                               coordinate (rad/s^2; translations scaled so 1 cm counts like 1 deg).
//                               0 (default) = off. Larger = smoother. Sweep it on synthetic data.

#include <ceres/ceres.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include "bspline.h"
#include "event_knots.h"
#include "frame_residual.h"
#include "ik_weights.h"
#include "mot_io.h"
#include "opensim_fk.h"
#include "parseTRC.h"
#include "trc_io.h"

namespace {

// One frame = one Ceres residual block. The real work is in frame_eval().
// Roughness penalty for one coordinate at one evaluation time t:
//   r = sqrt(weight) * q-ddot(t) = sqrt(weight) * sum_a B''_a(t) * c_a
// Summed over frames (weight = lambda * dt), r^2 approximates lambda * integral of q-ddot^2.
// It is linear in the coefficients, so the Jacobian is just the scaled B'' values.
class SmoothCost : public ceres::CostFunction {
public:
    explicit SmoothCost(std::vector<double> scaled_Bdd) : w_(std::move(scaled_Bdd)) {
        set_num_residuals(1);
        for (size_t a = 0; a < w_.size(); ++a) mutable_parameter_block_sizes()->push_back(1);
    }
    bool Evaluate(double const* const* params, double* residuals, double** jacobians) const override {
        residuals[0] = 0.0;
        for (size_t a = 0; a < w_.size(); ++a) residuals[0] += w_[a] * params[a][0];
        if (jacobians)
            for (size_t a = 0; a < w_.size(); ++a)
                if (jacobians[a]) jacobians[a][0] = w_[a];
        return true;
    }
private:
    std::vector<double> w_;
};

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


// Linear interpolation of (t, y) at time x (clamped at the ends).
double interp(const std::vector<double>& t, const std::vector<double>& y, double x) {
    if (x <= t.front()) return y.front();
    if (x >= t.back()) return y.back();
    size_t i = 1;
    while (t[i] < x) ++i;
    const double a = (x - t[i - 1]) / (t[i] - t[i - 1]);
    return y[i - 1] + a * (y[i] - y[i - 1]);
}

}  // namespace

int main(int argc_in, char** argv_in) {
    // Pull out the --options; what is left are the positional arguments.
    std::vector<std::string> excluded;
    std::vector<char*> positional;
    double smooth_lambda = 0.0;
    for (int i = 0; i < argc_in; ++i) {
        const std::string a = argv_in[i];
        if (a == "--no-arms") {
            for (const std::string& p : arm_body_prefixes()) excluded.push_back(p);
        } else if (a == "--exclude-bodies" && i + 1 < argc_in) {
            std::stringstream ss(argv_in[++i]);
            std::string p;
            while (std::getline(ss, p, ',')) if (!p.empty()) excluded.push_back(p);
        } else if (a == "--smooth" && i + 1 < argc_in) {
            smooth_lambda = std::stod(argv_in[++i]);
        } else {
            positional.push_back(argv_in[i]);
        }
    }
    const int argc = static_cast<int>(positional.size());
    char** argv = positional.data();

    if (argc < 5) {
        std::cerr << "usage: spline_ik <model.osim> <markers.trc> <init_ik.mot> <out.mot>"
                     " [knot_spacing_s=0.05] [ik_setup.xml|-] [grf.mot] [dense_s=0.01] [before_s=0.03] [after_s=0.10]"
                     " [--no-arms] [--exclude-bodies a,b,...] [--smooth LAMBDA]\n";
        return 1;
    }
    const double spacing = argc >= 6 ? std::stod(argv[5]) : 0.05;

    try {
        // ---- data ----
        const ActivityTrace trace = parse_trc(argv[2], 0);
        std::map<std::string, double> weights;
        if (argc >= 7 && std::string(argv[6]) != "-") weights = read_ik_weights(argv[6]);   // "-" = no weights file
        std::vector<std::string> wanted;
        for (const auto& [name, traj] : trace.trajectories_by_marker) {
            const double w = weights.count(name) ? weights[name] : 1.0;
            if (w > 0) wanted.push_back(name);
            else std::cout << "note: marker " << name << " has weight 0, not used\n";
        }

        // ---- model ----
        OpenSimFK fk(argv[1], wanted, excluded);
        const int nq = static_cast<int>(fk.coordinate_names().size());
        const int M = static_cast<int>(fk.marker_names().size());
        const int F_all = static_cast<int>(trace.trajectories_by_marker.begin()->second.marker_pts.size());
        const double dt = trace.time_step_millis / 1000.0;
        const double trc_t0 = trc_first_time(argv[2]);   // time of marker frame 0

        // ---- initial IK result: the solve covers only the frames inside its time range ----
        OpenSim::Storage init(argv[3]);
        if (init.isInDegrees()) fk.model().getSimbodyEngine().convertDegreesToRadians(init);
        // (Uses the same Storage calls as fk_check: column labels + state vectors.)
        const OpenSim::Array<std::string>& labels = init.getColumnLabels();
        std::vector<double> t_init(init.getSize());
        for (int r = 0; r < init.getSize(); ++r) t_init[r] = init.getStateVector(r)->getTime();
        const int i0 = std::max(0, static_cast<int>(std::ceil((t_init.front() - trc_t0) / dt - 1e-6)));
        const int i1 = std::min(F_all - 1, static_cast<int>(std::floor((t_init.back() - trc_t0) / dt + 1e-6)));
        if (i1 - i0 < 4) throw std::runtime_error("initial IK and marker data barely overlap in time");
        const int F = i1 - i0 + 1;                    // frames solved
        const double t_abs0 = trc_t0 + i0 * dt;       // absolute time of the first solved frame
        const double T = (F - 1) * dt;
        std::vector<double> t_frames(F);              // spline time, 0 at the first solved frame
        for (int i = 0; i < F; ++i) t_frames[i] = i * dt;
        std::cout << "solving marker frames " << i0 << "-" << i1 << " (" << t_abs0 << "-" << t_abs0 + T << " s)\n";

        // ---- spline shape: uniform, or denser around foot events if a GRF file is given ----
        BSpline spline;
        if (argc >= 8) {
            double dense = argc >= 9 ? std::stod(argv[8]) : 0.01;
            const double before = argc >= 10 ? std::stod(argv[9]) : 0.03;
            const double after = argc >= 11 ? std::stod(argv[10]) : 0.10;
            if (dense < dt) {   // every coefficient needs frames under it
                std::cout << "note: dense spacing raised to the frame interval " << dt << " s\n";
                dense = dt;
            }
            std::vector<double> events = detect_contact_events(argv[7]);
            std::cout << "foot events (s):";
            for (double& e : events) { std::cout << " " << e; e -= t_abs0; }   // to spline time
            std::cout << "\n";
            spline = make_clamped(0.0, T, event_knots(T, spacing, dense, events, before, after));
        } else {
            spline = make_uniform(0.0, T, std::max(1, static_cast<int>(std::lround(T / spacing))));
        }
        const int K = spline.num_coeffs();
        std::cout << F << " frames, " << T << " s, " << K - 3 << " knot intervals -> "
                  << K << " coefficients x " << nq << " coordinates = " << K * nq << " unknowns\n";

        // ---- starting point: fit splines to the initial IK result ----
        std::vector<std::vector<double>> c(nq);
        for (int j = 0; j < nq; ++j) {
            std::vector<double> y(F, 0.0);
            const int col = labels.findIndex(fk.coordinate_names()[j]);   // 0 is "time"
            if (col > 0) {
                std::vector<double> yi(init.getSize());
                for (int r = 0; r < init.getSize(); ++r) yi[r] = init.getStateVector(r)->getData()[col - 1];
                for (int i = 0; i < F; ++i) y[i] = interp(t_init, yi, t_frames[i] + t_abs0);
            } else {
                std::cout << "note: " << fk.coordinate_names()[j] << " not in initial .mot, starting at 0\n";
            }
            c[j] = fit(spline, t_frames, y);
        }

        // Keep clamped coordinates inside their range from the start (Ceres needs a feasible start).
        // A B-spline always lies between its smallest and largest coefficient, so bounding the
        // coefficients bounds the whole curve.
        int n_bounded = 0;
        for (int j = 0; j < nq; ++j) {
            if (!fk.clamped()[j]) continue;
            ++n_bounded;
            for (double& v : c[j]) v = std::min(std::max(v, fk.range_min()[j]), fk.range_max()[j]);
        }
        const std::vector<std::vector<double>> c_start = c;

        // ---- one residual block per frame ----
        ceres::Problem problem;
        std::vector<FrameData> frames_copy;   // kept for the error report after solving
        long observed = 0;
        for (int i = 0; i < F; ++i) {
            FrameData f;
            for (int k = 0; k < K; ++k) {
                const double b = basis(spline, k, t_frames[i]);
                if (b != 0.0) f.active.push_back({k, b});
            }
            f.observed_marker_positions.resize(M);
            f.scale.resize(M);
            for (int m = 0; m < M; ++m) {
                const std::string& name = fk.marker_names()[m];
                const MarkerPt& pt = trace.trajectories_by_marker.at(name).marker_pts[i0 + i];
                if (pt.observed_coord) {
                    f.observed_marker_positions[m] = *pt.observed_coord;
                    ++observed;
                }
                const double w = weights.count(name) ? weights[name] : 1.0;
                f.scale[m] = std::sqrt(w) / pt.measurement_uncertainty.x_mm;   // OpenSim weights the squared error by w
            }
            std::vector<double*> blocks;
            for (int j = 0; j < nq; ++j)
                for (const ActiveCoef& ac : f.active) blocks.push_back(&c[j][ac.k]);
            frames_copy.push_back(f);
            problem.AddResidualBlock(new FrameCost(&fk, std::move(f)), nullptr, blocks);
        }

        // Range limits as bounds on each coefficient (each coefficient is its own parameter block).
        for (int j = 0; j < nq; ++j) {
            if (!fk.clamped()[j]) continue;
            for (int k = 0; k < K; ++k) {
                problem.SetParameterLowerBound(&c[j][k], 0, fk.range_min()[j]);
                problem.SetParameterUpperBound(&c[j][k], 0, fk.range_max()[j]);
            }
        }
        std::cout << n_bounded << " of " << nq << " coordinates have range limits (clamped)\n";

        // Roughness penalty: one small residual per coordinate per frame, on q-ddot.
        if (smooth_lambda > 0) {
            const double rad_per_m = (M_PI / 180.0) / 0.01;   // 1 cm counts like 1 degree
            for (int i = 0; i < F; ++i) {
                std::vector<int> ks;
                std::vector<double> bdd;
                for (int k = 0; k < K; ++k) {
                    const double b = basis(spline, k, t_frames[i], 2);
                    if (b != 0.0 || basis(spline, k, t_frames[i]) != 0.0) { ks.push_back(k); bdd.push_back(b); }
                }
                for (int j = 0; j < nq; ++j) {
                    const double unit = fk.rotational()[j] ? 1.0 : rad_per_m;
                    const double sw = std::sqrt(smooth_lambda * dt) * unit;
                    std::vector<double> w(bdd.size());
                    std::vector<double*> blocks;
                    for (size_t a = 0; a < ks.size(); ++a) { w[a] = sw * bdd[a]; blocks.push_back(&c[j][ks[a]]); }
                    problem.AddResidualBlock(new SmoothCost(std::move(w)), nullptr, blocks);
                }
            }
            std::cout << "roughness penalty lambda = " << smooth_lambda << "\n";
        }

        // ---- solve ----
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        if (!ceres::IsSparseLinearAlgebraLibraryTypeAvailable(options.sparse_linear_algebra_library_type))
            options.linear_solver_type = ceres::DENSE_QR;
        options.num_threads = 1;                 // one OpenSim model/state: not thread-safe
        options.max_num_iterations = 50;
        options.minimizer_progress_to_stdout = true;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        std::cout << summary.BriefReport() << "\n";

        // Ceres' cost includes the marker weights, so it is not a distance. Report real
        // marker errors in mm (unweighted): overall RMS, and the worst markers at the end.
        auto marker_errors = [&](const std::vector<std::vector<double>>& coef,
                                 std::vector<double>& sum_sq, std::vector<int>& count) {
            sum_sq.assign(M, 0.0);
            count.assign(M, 0);
            std::vector<double> pos;
            for (int i = 0; i < F; ++i) {
                std::vector<double> q(nq, 0.0);
                for (int j = 0; j < nq; ++j) q[j] = evaluate(spline, coef[j], t_frames[i]);
                fk.marker_positions(q, pos);
                for (int m = 0; m < M; ++m) {
                    const auto& obs = frames_copy[i].observed_marker_positions[m];
                    if (!obs) continue;
                    const double dx = pos[3 * m] * 1000 - obs->x_mm, dy = pos[3 * m + 1] * 1000 - obs->y_mm,
                                 dz = pos[3 * m + 2] * 1000 - obs->z_mm;
                    sum_sq[m] += dx * dx + dy * dy + dz * dz;
                    ++count[m];
                }
            }
        };
        auto overall_rms = [&](const std::vector<double>& s, const std::vector<int>& n) {
            double a = 0; long b = 0;
            for (int m = 0; m < M; ++m) { a += s[m]; b += n[m]; }
            return std::sqrt(a / std::max(1L, b));
        };
        std::vector<double> s0, s1; std::vector<int> n0, n1;
        marker_errors(c_start, s0, n0);
        marker_errors(c, s1, n1);
        std::cout << "RMS marker error (mm, unweighted): start " << overall_rms(s0, n0)
                  << " -> end " << overall_rms(s1, n1) << "\n";
        std::vector<int> order(M);
        for (int m = 0; m < M; ++m) order[m] = m;
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return s1[a] / std::max(1, n1[a]) > s1[b] / std::max(1, n1[b]); });
        std::cout << "worst markers at the end (RMS mm):";
        for (int i = 0; i < std::min(5, M); ++i)
            std::cout << "  " << fk.marker_names()[order[i]] << " "
                      << std::sqrt(s1[order[i]] / std::max(1, n1[order[i]]));
        std::cout << "\n";

        // ---- write q, and its exact derivatives from the splines: q-dot (_u) and q-ddot (_udot) ----
        std::vector<double> t_out(F);
        for (int i = 0; i < F; ++i) t_out[i] = t_frames[i] + t_abs0;
        for (int d = 0; d <= 2; ++d) {
            std::vector<std::vector<double>> vals(F, std::vector<double>(nq));
            for (int i = 0; i < F; ++i)
                for (int j = 0; j < nq; ++j) vals[i][j] = evaluate(spline, c[j], t_frames[i], d);
            const std::string path = d == 0 ? std::string(argv[4]) : mot_with_suffix(argv[4], d == 1 ? "_u" : "_udot");
            write_mot(path, "spline_ik", fk.coordinate_names(), fk.rotational(), t_out, vals);
            std::cout << "wrote " << path << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
