
#ifndef MARKER_OFFSETS_H
#define MARKER_OFFSETS_H

#include <cmath>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "marker_fk.h"

// Marker offsets d_m: how far each marker really sits from where the model puts it, in the frame
// the marker is attached to (metres). Stored flat: d[3*m + axis].

// pos (from the last fk.marker_positions call) += R_m * d_m for every marker.
inline void add_offsets(MarkerFK& fk, const std::vector<double>& d, std::vector<double>& pos) {
    std::vector<double> R;
    if (!fk.marker_rotations(R)) throw std::runtime_error("this MarkerFK cannot report marker rotations");
    for (size_t m = 0; m < pos.size() / 3; ++m)
        for (int i = 0; i < 3; ++i)
            pos[3 * m + i] += R[9 * m + 3 * i] * d[3 * m] + R[9 * m + 3 * i + 1] * d[3 * m + 1] +
                              R[9 * m + 3 * i + 2] * d[3 * m + 2];
}

// "dir/run.mot" -> "dir/run_offsets.csv"
inline std::string offsets_csv_path(const std::string& mot_path) {
    const std::string stem = (mot_path.size() > 4 && mot_path.compare(mot_path.size() - 4, 4, ".mot") == 0)
                                 ? mot_path.substr(0, mot_path.size() - 4) : mot_path;
    return stem + "_offsets.csv";
}

// One row per marker: marker,body,dx_mm,dy_mm,dz_mm,norm_mm,frames  (frames = how often it was seen;
// pass an empty vector if not known).
inline void write_offsets_csv(const std::string& path, const std::vector<std::string>& markers,
                              const std::vector<std::string>& bodies, const std::vector<double>& d,
                              const std::vector<int>& frames) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot open " + path);
    out << "marker,body,dx_mm,dy_mm,dz_mm,norm_mm,frames\n";
    out.precision(6);
    for (size_t m = 0; m < markers.size(); ++m) {
        const double x = d[3 * m] * 1000, y = d[3 * m + 1] * 1000, z = d[3 * m + 2] * 1000;
        out << markers[m] << "," << bodies[m] << "," << x << "," << y << "," << z << ","
            << std::sqrt(x * x + y * y + z * z) << "," << (frames.empty() ? -1 : frames[m]) << "\n";
    }
}

// ---- shape-only offsets ----
// If every marker on a body moves together (same shift, or the same small rotation of the whole
// cluster), that looks exactly like a change in the body's pose, i.e. in the joint angles. Those
// 6 directions per body cannot be told apart from the motion, so shape-only mode removes them:
// a body's offsets may only change the SHAPE of its marker cluster.
//
// Per body (group of markers sharing a frame) with n markers: d_group (3n numbers) = N * z, where
// N (3n x k) has orthonormal columns perpendicular to the cluster's rigid motions, and z (k numbers)
// are the unknowns. k = 3n - 6 for 3+ markers not on a line, 1 for 2 markers, 0 for 1 marker.
struct ShapeOffsets {
    std::vector<std::string> group_names;
    std::vector<std::vector<int>> members;      // marker indices in each group
    std::vector<int> size;                      // k for each group (0 = nothing to estimate)
    std::vector<std::vector<double>> basis;     // N for each group, (3n x k), row-major

    // d (3 per marker) from the group unknowns; z[g] points to group g's k numbers (unused if k = 0).
    void to_offsets(const std::vector<const double*>& z, size_t num_markers, std::vector<double>& d) const {
        d.assign(3 * num_markers, 0.0);
        for (size_t g = 0; g < members.size(); ++g) {
            const int k = size[g];
            if (k == 0) continue;
            for (size_t l = 0; l < members[g].size(); ++l)
                for (int a = 0; a < 3; ++a) {
                    double v = 0;
                    for (int c = 0; c < k; ++c) v += basis[g][(3 * l + a) * k + c] * z[g][c];
                    d[3 * members[g][l] + a] = v;
                }
        }
    }
};

// frame_of_marker: name of the frame each marker's location is expressed in.
// location_m: each marker's position in that frame (metres), 3 per marker.
inline ShapeOffsets make_shape_offsets(const std::vector<std::string>& frame_of_marker,
                                       const std::vector<double>& location_m) {
    ShapeOffsets so;
    std::map<std::string, int> index;
    for (size_t m = 0; m < frame_of_marker.size(); ++m) {
        auto it = index.find(frame_of_marker[m]);
        if (it == index.end()) {
            it = index.emplace(frame_of_marker[m], static_cast<int>(so.members.size())).first;
            so.group_names.push_back(frame_of_marker[m]);
            so.members.emplace_back();
        }
        so.members[it->second].push_back(static_cast<int>(m));
    }
    auto dot = [](const std::vector<double>& a, const std::vector<double>& b) {
        double s = 0; for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i]; return s; };
    auto remove_from = [&](std::vector<double>& v, const std::vector<std::vector<double>>& Q) {
        for (int pass = 0; pass < 2; ++pass)   // twice, for numerical safety
            for (const auto& u : Q) { const double c = dot(v, u); for (size_t i = 0; i < v.size(); ++i) v[i] -= c * u[i]; }
    };
    for (const auto& mem : so.members) {
        const size_t n = mem.size(), L = 3 * n;
        double cen[3] = {0, 0, 0};
        for (int m : mem) for (int a = 0; a < 3; ++a) cen[a] += location_m[3 * m + a] / n;
        // rigid motions of the cluster: 3 shifts, 3 small rotations (omega x p)
        std::vector<std::vector<double>> rigid;
        for (int t = 0; t < 6; ++t) {
            std::vector<double> v(L, 0.0);
            for (size_t l = 0; l < n; ++l) {
                const double* p0 = &location_m[3 * mem[l]];
                const double p[3] = {p0[0] - cen[0], p0[1] - cen[1], p0[2] - cen[2]};
                if (t < 3) v[3 * l + t] = 1.0;
                else {
                    const int a = t - 3;   // omega = unit axis a
                    const double w[3] = {a == 0 ? 1.0 : 0.0, a == 1 ? 1.0 : 0.0, a == 2 ? 1.0 : 0.0};
                    v[3 * l + 0] = w[1] * p[2] - w[2] * p[1];
                    v[3 * l + 1] = w[2] * p[0] - w[0] * p[2];
                    v[3 * l + 2] = w[0] * p[1] - w[1] * p[0];
                }
            }
            const double n0 = std::sqrt(dot(v, v));
            if (n0 == 0) continue;
            remove_from(v, rigid);
            const double n1 = std::sqrt(dot(v, v));
            if (n1 > 1e-6 * n0) { for (double& x : v) x /= n1; rigid.push_back(v); }
        }
        // the rest: pick unit vectors one at a time, the one with the most left after removing what we have
        std::vector<std::vector<double>> shape, all = rigid;
        while (all.size() < L) {
            std::vector<double> best; double best_norm = 0;
            for (size_t i = 0; i < L; ++i) {
                std::vector<double> v(L, 0.0); v[i] = 1.0;
                remove_from(v, all);
                const double nv = std::sqrt(dot(v, v));
                if (nv > best_norm) { best_norm = nv; best = v; }
            }
            if (best_norm < 1e-6) break;
            for (double& x : best) x /= best_norm;
            shape.push_back(best); all.push_back(best);
        }
        const int k = static_cast<int>(shape.size());
        std::vector<double> N(L * k);
        for (size_t r = 0; r < L; ++r) for (int c = 0; c < k; ++c) N[r * k + c] = shape[c][r];
        so.size.push_back(k);
        so.basis.push_back(N);
    }
    return so;
}

#endif
