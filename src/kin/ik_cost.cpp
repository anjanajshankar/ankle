
// ik_cost: the same weighted marker cost that spline_ik minimizes, evaluated for any motion files.
//
// Lets you ask: which motion fits the markers best under the IK objective - the truth,
// OpenSim's IK, or spline IK? If a method beats the truth, it is fitting the errors
// (soft-tissue artifact, noise) rather than the bones; if it is worse than another method,
// it is not reaching the best fit.
//
// usage: ik_cost <model.osim> <markers.trc> <ik_setup.xml|-> [--no-arms] label=motion.mot [label=motion.mot ...]
//
// For each motion: weighted cost = 1/2 * sum over frames and markers of |weight/sigma * (FK - observed)|^2
// (the data part of spline_ik's cost; no roughness penalty), plus the unweighted RMS marker error
// in mm, and the RMS per body so you can see where the fits differ.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "ik_weights.h"
#include "opensim_fk.h"
#include "parseTRC.h"

int main(int argc, char** argv) {
    std::vector<std::string> excluded;
    std::vector<std::string> pos;
    std::vector<std::pair<std::string, std::string>> motions;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--no-arms") excluded = arm_body_prefixes();
        else if (pos.size() < 3) pos.push_back(a);
        else {
            const auto eq = a.find('=');
            motions.push_back(eq == std::string::npos ? std::make_pair(a, a)
                                                      : std::make_pair(a.substr(0, eq), a.substr(eq + 1)));
        }
    }
    if (pos.size() < 3 || motions.empty()) {
        std::cerr << "usage: ik_cost <model.osim> <markers.trc> <ik_setup.xml|-> [--no-arms] label=motion.mot ...\n";
        return 1;
    }

    try {
        const ActivityTrace trace = parse_trc(pos[1], 0);
        std::map<std::string, double> weights;
        if (pos[2] != "-") weights = read_ik_weights(pos[2]);
        std::vector<std::string> wanted;
        for (const auto& [name, traj] : trace.trajectories_by_marker)
            if (!weights.count(name) || weights[name] > 0) wanted.push_back(name);

        OpenSimFK fk(pos[0], wanted, excluded);
        const int nq = static_cast<int>(fk.coordinate_names().size());
        const int M = static_cast<int>(fk.marker_names().size());
        const int F = static_cast<int>(trace.trajectories_by_marker.begin()->second.marker_pts.size());
        const double dt = trace.time_step_millis / 1000.0;

        std::vector<double> scale(M);
        for (int m = 0; m < M; ++m) {
            const std::string& n = fk.marker_names()[m];
            const double w = weights.count(n) ? weights[n] : 1.0;
            scale[m] = std::sqrt(w) / trace.trajectories_by_marker.at(n).marker_pts[0].measurement_uncertainty.x_mm;
        }

        double t_start = 0.0;   // trc frame 0 is taken to be the first motion file's first time
        std::cout << "\n" << std::string(14, ' ') << "  weighted cost   RMS (mm)   RMS by body (mm)\n";
        for (size_t f = 0; f < motions.size(); ++f) {
            OpenSim::Storage sto(motions[f].second);
            if (sto.isInDegrees()) fk.model().getSimbodyEngine().convertDegreesToRadians(sto);
            const OpenSim::Array<std::string>& labels = sto.getColumnLabels();
            const int R = sto.getSize();
            std::vector<double> tm(R);
            for (int r = 0; r < R; ++r) tm[r] = sto.getStateVector(r)->getTime();
            if (f == 0) t_start = tm.front();

            std::vector<int> col(nq);
            for (int j = 0; j < nq; ++j) {
                col[j] = labels.findIndex(fk.coordinate_names()[j]);
                if (col[j] <= 0) std::cout << "note: " << motions[f].first << " has no " << fk.coordinate_names()[j] << "\n";
            }

            double cost = 0, sum_sq = 0;
            long n_obs = 0;
            std::map<std::string, std::pair<double, long>> by_body;
            std::vector<double> q(nq, 0.0), p;
            for (int i = 0; i < F; ++i) {
                const double t = t_start + i * dt;
                int r = 0;
                while (r + 1 < R && tm[r + 1] <= t) ++r;
                const int r2 = std::min(r + 1, R - 1);
                const double a = (r2 == r || tm[r2] == tm[r]) ? 0.0 : (t - tm[r]) / (tm[r2] - tm[r]);
                for (int j = 0; j < nq; ++j)
                    if (col[j] > 0) {
                        const double v0 = sto.getStateVector(r)->getData()[col[j] - 1];
                        const double v1 = sto.getStateVector(r2)->getData()[col[j] - 1];
                        q[j] = v0 + a * (v1 - v0);
                    }
                fk.marker_positions(q, p);
                for (int m = 0; m < M; ++m) {
                    const auto& obs = trace.trajectories_by_marker.at(fk.marker_names()[m]).marker_pts[i].observed_coord;
                    if (!obs) continue;
                    const double d[3] = {p[3 * m] * 1000 - obs->x_mm, p[3 * m + 1] * 1000 - obs->y_mm,
                                         p[3 * m + 2] * 1000 - obs->z_mm};
                    const double e2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                    cost += 0.5 * scale[m] * scale[m] * e2;
                    sum_sq += e2;
                    ++n_obs;
                    auto& b = by_body[fk.marker_body_names()[m]];
                    b.first += e2;
                    b.second += 1;
                }
            }
            std::printf("%-14s  %13.6e   %8.3f  ", motions[f].first.c_str(), cost, std::sqrt(sum_sq / std::max(1L, n_obs)));
            for (const auto& [body, b] : by_body) std::printf(" %s %.2f", body.c_str(), std::sqrt(b.first / b.second));
            std::printf("\n");
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
