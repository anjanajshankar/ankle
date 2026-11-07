
// synth_markers: marker data with a KNOWN true motion, for testing how accurate an IK method is.
//
// 1. Truth: take an existing IK result (e.g. OpenSim's) and smooth it with a cubic spline
//    (knots every --truth-spacing s). That spline is the true motion: q, q-dot and q-ddot are
//    known exactly at every frame.
// 2. Markers: run OpenSim FK on the true motion, then add errors:
//      - soft-tissue artifact (STA): each marker drifts smoothly relative to the bone, repeating
//        with the stride (3 harmonics, random amplitudes and phases). Typical size per body:
//        thigh 8 mm, shank 5, torso 6, pelvis 4, foot 2, other 3 (scaled by --sta).
//      - measurement noise: independent Gaussian, --noise mm per axis.
// 3. Write <prefix>.trc (markers) and <prefix>_truth.mot, _truth_u.mot, _truth_udot.mot
//    (true q, q-dot, q-ddot of the free coordinates).
//
// usage: synth_markers <model.osim> <source_ik.mot> <out_prefix>
//                      [--noise 1.0] [--sta 1.0] [--truth-spacing 0.037] [--stride 1.144] [--seed 1]

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "bspline.h"
#include "mot_io.h"
#include "opensim_fk.h"
#include "trc_io.h"

namespace {

// Typical soft-tissue artifact size (mm) for markers on each body.
double sta_size_mm(const std::string& body) {
    const std::vector<std::pair<std::string, double>> table = {
        {"femur", 8.0}, {"patella", 5.0}, {"tibia", 5.0}, {"torso", 6.0}, {"pelvis", 4.0},
        {"calcn", 2.0}, {"talus", 2.0}, {"toes", 2.0}, {"humerus", 8.0}, {"ulna", 5.0},
        {"radius", 5.0}, {"hand", 3.0}};
    for (const auto& [prefix, size] : table)
        if (body.compare(0, prefix.size(), prefix) == 0) return size;
    return 3.0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: synth_markers <model.osim> <source_ik.mot> <out_prefix> [--noise 1.0] [--sta 1.0]"
                     " [--truth-spacing 0.037] [--stride 1.144] [--seed 1]\n";
        return 1;
    }
    double noise_mm = 1.0, sta_scale = 1.0, truth_spacing = 0.037, stride = 1.144;
    unsigned seed = 1;
    for (int i = 4; i + 1 < argc; i += 2) {
        const std::string k = argv[i];
        const double v = std::stod(argv[i + 1]);
        if (k == "--noise") noise_mm = v;
        else if (k == "--sta") sta_scale = v;
        else if (k == "--truth-spacing") truth_spacing = v;
        else if (k == "--stride") stride = v;
        else if (k == "--seed") seed = static_cast<unsigned>(v);
        else { std::cerr << "unknown option " << k << "\n"; return 1; }
    }
    const std::string prefix = argv[3];

    try {
        OpenSimFK fk(argv[1], {});   // all markers
        const int nq = static_cast<int>(fk.coordinate_names().size());
        const int M = static_cast<int>(fk.marker_names().size());

        // ---- source motion ----
        OpenSim::Storage src(argv[2]);
        if (src.isInDegrees()) fk.model().getSimbodyEngine().convertDegreesToRadians(src);
        const int F = src.getSize();
        std::vector<double> t_abs(F), t(F);
        for (int i = 0; i < F; ++i) t_abs[i] = src.getStateVector(i)->getTime();
        for (int i = 0; i < F; ++i) t[i] = t_abs[i] - t_abs[0];
        const double T = t.back();

        // ---- truth: smooth each coordinate with a spline (kept inside its range if clamped) ----
        const BSpline truth = make_uniform(0.0, T, std::max(1, static_cast<int>(std::lround(T / truth_spacing))));
        const OpenSim::Array<std::string>& labels = src.getColumnLabels();
        std::vector<std::vector<double>> c(nq);
        for (int j = 0; j < nq; ++j) {
            std::vector<double> y(F, 0.0);
            const int col = labels.findIndex(fk.coordinate_names()[j]);
            if (col > 0) for (int i = 0; i < F; ++i) y[i] = src.getStateVector(i)->getData()[col - 1];
            else std::cout << "note: " << fk.coordinate_names()[j] << " not in source, truth = 0\n";
            c[j] = fit(truth, t, y);
            if (fk.clamped()[j])
                for (double& v : c[j]) v = std::min(std::max(v, fk.range_min()[j]), fk.range_max()[j]);
        }

        std::vector<std::vector<double>> q(F, std::vector<double>(nq)), u = q, udot = q;
        for (int i = 0; i < F; ++i)
            for (int j = 0; j < nq; ++j) {
                q[i][j] = evaluate(truth, c[j], t[i], 0);
                u[i][j] = evaluate(truth, c[j], t[i], 1);
                udot[i][j] = evaluate(truth, c[j], t[i], 2);
            }

        // ---- soft-tissue artifact: smooth, stride-periodic drift per marker and axis ----
        std::mt19937 rng(seed);
        std::normal_distribution<double> gauss(0.0, 1.0);
        std::uniform_real_distribution<double> phase(0.0, 2 * M_PI);
        struct Wave { double amp, phi; };
        std::vector<std::vector<Wave>> sta(3 * M);   // [3*m + axis] -> 3 harmonics
        for (int m = 0; m < M; ++m) {
            const double size = sta_scale * sta_size_mm(fk.marker_body_names()[m]);
            for (int a = 0; a < 3; ++a)
                for (int h = 1; h <= 3; ++h) sta[3 * m + a].push_back({0.6 * size * gauss(rng) / h, phase(rng)});
        }

        // ---- markers: FK on the truth + STA + noise ----
        std::vector<std::vector<double>> pos_mm(F, std::vector<double>(3 * M));
        std::map<std::string, std::pair<double, int>> sta_rms;   // per body: sum of squares, count
        std::vector<double> pos;
        for (int i = 0; i < F; ++i) {
            fk.marker_positions(q[i], pos);
            for (int m = 0; m < M; ++m) {
                double sq = 0;
                for (int a = 0; a < 3; ++a) {
                    double d = 0;
                    for (int h = 0; h < 3; ++h)
                        d += sta[3 * m + a][h].amp * std::sin(2 * M_PI * (h + 1) * t[i] / stride + sta[3 * m + a][h].phi);
                    sq += d * d;
                    pos_mm[i][3 * m + a] = pos[3 * m + a] * 1000.0 + d + noise_mm * gauss(rng);
                }
                auto& r = sta_rms[fk.marker_body_names()[m]];
                r.first += sq;
                r.second += 1;
            }
        }

        write_trc(prefix + ".trc", fk.marker_names(), t_abs, pos_mm);
        write_mot(prefix + "_truth.mot", "truth", fk.coordinate_names(), fk.rotational(), t_abs, q);
        write_mot(prefix + "_truth_u.mot", "truth", fk.coordinate_names(), fk.rotational(), t_abs, u);
        write_mot(prefix + "_truth_udot.mot", "truth", fk.coordinate_names(), fk.rotational(), t_abs, udot);

        std::cout << "truth: " << nq << " coordinates, spline knots every " << truth_spacing << " s\n";
        std::cout << "markers: " << M << ", " << F << " frames, noise " << noise_mm << " mm per axis\n";
        std::cout << "soft-tissue artifact, RMS distance per body (mm):";
        for (const auto& [body, r] : sta_rms) std::cout << "  " << body << " " << std::sqrt(r.first / r.second);
        std::cout << "\nwrote " << prefix << ".trc and " << prefix << "_truth{,_u,_udot}.mot\n";
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
