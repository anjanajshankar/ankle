
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
//      - marker placement error (--offset-error MM, default 0): each marker sits a fixed random
//        distance from where the model puts it (Gaussian, MM per axis, in its body's frame).
//        The true offsets are written to <prefix>_true_offsets.csv, for checking spline_ik --fit-offsets.
//    Optional impact transients (--impacts grf.mot): at every foot contact found in that file, the truth of
//    that leg's knee_angle, ankle_angle and hip_flexion gets a fast "kick" added on top of the smooth spline:
//    q-ddot = dv * [G(t; tc, tau) - G(t; tc + 0.05, 3 tau)] (G = Gaussian pulse of unit area), i.e. the
//    joint speeds up by dv within ~2 tau of contact and slows back down over the next ~80 ms; the joint ends
//    up dv * 0.05 rad further on (2.9 deg for dv = 1 rad/s). Gains: knee 1, ankle 0.7, hip 0.5.
//    Peak q-ddot about 0.96 dv / (tau sqrt(2 pi)): about 2750 deg/s^2 at the knee for dv = 1 rad/s, tau = 8 ms.
//    q, q-dot and q-ddot of the transient are exact, so the truth files include it. Contact times are
//    written to <prefix>_impacts.csv. Options: --impact-dv 1.0 (rad/s), --impact-tau 0.008 (s).
// 3. Write <prefix>.trc (markers) and <prefix>_truth.mot, _truth_u.mot, _truth_udot.mot
//    (true q, q-dot, q-ddot of the free coordinates).
//
// usage: synth_markers <model.osim> <source_ik.mot> <out_prefix>
//                      [--noise 1.0] [--sta 1.0] [--truth-spacing 0.037] [--stride 1.144] [--seed 1]
//                      [--markers-like measured.trc]   (only write the markers that file contains)
//                      [--offset-error 0] [--impacts grf.mot] [--impact-dv 1.0] [--impact-tau 0.008]

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "bspline.h"
#include "event_knots.h"
#include "marker_offsets.h"
#include "mot_io.h"
#include "opensim_fk.h"
#include "trc_io.h"

namespace {

// One impact transient: q-ddot = dv * (G(t; tc, tau) - G(t; tc + delay, 3 tau)), G a unit-area Gaussian.
// Returns q (d = 0), q-dot (d = 1) or q-ddot (d = 2) at time t.
double impact_term(double t, double tc, double dv, double tau, int d) {
    const double delay = 0.05;
    auto one = [&](double c, double w) {
        const double u = (t - c) / w;
        const double pdf = std::exp(-0.5 * u * u) / std::sqrt(2 * M_PI);
        const double cdf = 0.5 * std::erfc(-u / std::sqrt(2.0));
        if (d == 2) return pdf / w;
        if (d == 1) return cdf;
        return (t - c) * cdf + w * pdf;     // integral of the cdf
    };
    return dv * (one(tc, tau) - one(tc + delay, 3 * tau));
}

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
                     " [--truth-spacing 0.037] [--stride 1.144] [--seed 1] [--markers-like measured.trc] [--offset-error 0]"
                     " [--impacts grf.mot] [--impact-dv 1.0] [--impact-tau 0.008]\n";
        return 1;
    }
    double noise_mm = 1.0, sta_scale = 1.0, truth_spacing = 0.037, stride = 1.144, offset_mm = 0.0;
    double impact_dv = 1.0, impact_tau = 0.008;
    std::string impacts_grf;
    unsigned seed = 1;
    std::vector<std::string> marker_names;   // empty = all model markers
    for (int i = 4; i + 1 < argc; i += 2) {
        const std::string k = argv[i];
        if (k == "--impacts") { impacts_grf = argv[i + 1]; continue; }
        if (k == "--markers-like") {
            std::ifstream in(argv[i + 1]);
            std::string line;
            for (int l = 0; l < 4 && std::getline(in, line); ++l) {}
            std::stringstream ss(line);
            std::string field;
            for (int col = 0; std::getline(ss, field, '\t'); ++col) {
                field.erase(field.find_last_not_of(" \r\n") + 1);
                if (col >= 2 && !field.empty()) marker_names.push_back(field);
            }
            if (marker_names.empty()) { std::cerr << "no marker names in " << argv[i + 1] << "\n"; return 1; }
            continue;
        }
        const double v = std::stod(argv[i + 1]);
        if (k == "--noise") noise_mm = v;
        else if (k == "--sta") sta_scale = v;
        else if (k == "--truth-spacing") truth_spacing = v;
        else if (k == "--stride") stride = v;
        else if (k == "--seed") seed = static_cast<unsigned>(v);
        else if (k == "--offset-error") offset_mm = v;
        else if (k == "--impact-dv") impact_dv = v;
        else if (k == "--impact-tau") impact_tau = v;
        else { std::cerr << "unknown option " << k << "\n"; return 1; }
    }
    const std::string prefix = argv[3];

    try {
        OpenSimFK fk(argv[1], marker_names);   // all model markers, or those of --markers-like
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

        // ---- impact transients on top of the smooth truth ----
        std::vector<ContactOnset> impacts;
        if (!impacts_grf.empty()) {
            for (const ContactOnset& e : detect_contact_onsets(impacts_grf))
                if (e.time > t_abs.front() && e.time < t_abs.back()) impacts.push_back(e);
            const std::vector<std::pair<std::string, double>> gains = {{"knee_angle_", 1.0}, {"ankle_angle_", 0.7},
                                                                         {"hip_flexion_", 0.5}};
            std::ofstream ev(prefix + "_impacts.csv");
            ev << "time,side\n";
            std::cout << "impact transients (dv " << impact_dv << " rad/s, tau " << impact_tau * 1000 << " ms) at:";
            for (const ContactOnset& e : impacts) {
                ev << e.time << "," << e.side << "\n";
                std::cout << " " << e.time << e.side;
                for (const auto& [stem, gain] : gains) {
                    const std::string name = stem + e.side;
                    int j = -1;
                    for (int jj = 0; jj < nq; ++jj) if (fk.coordinate_names()[jj] == name) j = jj;
                    if (j < 0) { std::cout << " (no " << name << ")"; continue; }
                    for (int i = 0; i < F; ++i) {
                        q[i][j] += impact_term(t_abs[i], e.time, gain * impact_dv, impact_tau, 0);
                        u[i][j] += impact_term(t_abs[i], e.time, gain * impact_dv, impact_tau, 1);
                        udot[i][j] += impact_term(t_abs[i], e.time, gain * impact_dv, impact_tau, 2);
                    }
                }
            }
            std::cout << "\n";
            if (impacts.empty()) std::cout << "note: no foot contacts inside the motion's time range\n";
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

        // ---- marker placement error: a fixed offset per marker, in its body's frame (metres) ----
        // Drawn from a separate generator so the STA and noise for a given seed stay the same
        // whether or not offsets are added.
        std::vector<double> d_true(3 * M, 0.0);
        if (offset_mm > 0) {
            std::mt19937 rng_off(seed + 1000003u);
            std::normal_distribution<double> g_off(0.0, offset_mm / 1000.0);
            for (double& v : d_true) v = g_off(rng_off);
        }

        // ---- markers: FK on the truth + offsets + STA + noise ----
        std::vector<std::vector<double>> pos_mm(F, std::vector<double>(3 * M));
        std::map<std::string, std::pair<double, int>> sta_rms;   // per body: sum of squares, count
        std::vector<double> pos;
        for (int i = 0; i < F; ++i) {
            fk.marker_positions(q[i], pos);
            if (offset_mm > 0) add_offsets(fk, d_true, pos);
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

        if (offset_mm > 0) {
            write_offsets_csv(prefix + "_true_offsets.csv", fk.marker_names(), fk.marker_body_names(), d_true, {});
            std::cout << "marker offsets: " << offset_mm << " mm per axis, written to " << prefix << "_true_offsets.csv\n";
        }
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
