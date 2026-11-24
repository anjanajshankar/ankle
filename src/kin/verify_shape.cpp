
// Check (no Ceres needed): shape-only marker offsets on the 2D toy.
// 1. The shape basis is orthonormal and perpendicular to each body's rigid motions.
// 2. Jacobian of frame_eval_shape_offsets vs brute force.
// 3. Gauss-Newton: offsets fixed / full offsets / shape-only offsets, on data with misplaced markers.
//    Also a control with NO true offsets: shape-only should then stay near zero and not hurt the angles.
#include <cmath>
#include <cstdio>
#include <random>
#include "bspline.h"
#include "frame_residual.h"
#include "toy_fk.h"

static double truth(int j, double t) {
    const double base[6] = {0.3 * t, 0.95, 0.05, 0.3, -0.4, 0.2}, amp[6] = {0.0, 0.02, 0.05, 0.5, 0.6, 0.3};
    return base[j] + amp[j] * std::sin(2 * M_PI * t + j);
}

int main() {
    ToyFK fk;
    const int nq = 6, M = 8, R3 = 3 * M, F = 101;
    const double dt = 0.01, sigma_prior_mm = 20.0;
    BSpline s = make_uniform(0, 1.0, 10);
    const int K = s.num_coeffs();
    std::mt19937 rng(3); std::normal_distribution<double> g(0, 1);

    // true offsets: in-plane, ~12 mm, random direction
    std::vector<std::array<double,3>> dtrue(M);
    for (auto& d : dtrue) { double a = 2 * M_PI * (g(rng) * 0.3 + 0.5); double r = 0.010 + 0.005 * std::abs(g(rng)); d = {r * std::cos(a), r * std::sin(a), 0}; }

    std::vector<FrameData> frames(F);
    std::vector<double> q(nq), pos, R;
    for (int i = 0; i < F; ++i) {
        const double t = i * dt;
        for (int k = 0; k < K; ++k) { double b = basis(s, k, t); if (b != 0) frames[i].active.push_back({k, b}); }
        for (int j = 0; j < nq; ++j) q[j] = truth(j, t);
        fk.marker_positions(q, pos); fk.marker_rotations(R);
        frames[i].observed_marker_positions.resize(M); frames[i].scale.assign(M, 1.0);
        for (int m = 0; m < M; ++m) {
            double p[3];
            for (int a = 0; a < 3; ++a) p[a] = pos[3*m+a] + R[9*m+3*a]*dtrue[m][0] + R[9*m+3*a+1]*dtrue[m][1] + R[9*m+3*a+2]*dtrue[m][2];
            frames[i].observed_marker_positions[m] = Coordinate{p[0]*1000 + g(rng), p[1]*1000 + g(rng), p[2]*1000};
        }
    }

    // marker positions in their body frames (see toy_fk.h) and which body each is on
    const std::vector<std::string> frame_of = {"pelvis", "pelvis", "a", "a", "b", "b", "c", "c"};
    std::vector<double> loc(3 * M, 0.0);
    loc[0] = -0.10; loc[3] = 0.10;
    const double seg[3] = {0.45, 0.42, 0.25};
    for (int sg = 0; sg < 3; ++sg) for (int k = 0; k < 2; ++k) {
        const int m = 2 + 2 * sg + k;
        loc[3 * m] = 0.04 + 0.02 * k; loc[3 * m + 1] = -seg[sg] * (k + 1) / 3.0;
    }
    const ShapeOffsets shape = make_shape_offsets(frame_of, loc);
    int Z = 0; for (int k : shape.size) Z += k;

    // --- 1. basis check ---
    {
        double worst = 0;
        for (size_t g = 0; g < shape.members.size(); ++g) {
            const int k = shape.size[g]; const size_t L = 3 * shape.members[g].size();
            for (int a = 0; a < k; ++a) for (int b = 0; b < k; ++b) {
                double s2 = 0; for (size_t r = 0; r < L; ++r) s2 += shape.basis[g][r*k+a] * shape.basis[g][r*k+b];
                worst = std::max(worst, std::abs(s2 - (a == b)));
            }
            // perpendicular to shifts and to small rotations about the origin
            for (int c = 0; c < k; ++c) for (int t = 0; t < 6; ++t) {
                double s2 = 0;
                for (size_t l = 0; l < shape.members[g].size(); ++l) {
                    const double* p = &loc[3 * shape.members[g][l]];
                    double v[3] = {0, 0, 0};
                    if (t < 3) v[t] = 1;
                    else { const int a = t - 3; const double w[3] = {double(a==0), double(a==1), double(a==2)};
                           v[0] = w[1]*p[2]-w[2]*p[1]; v[1] = w[2]*p[0]-w[0]*p[2]; v[2] = w[0]*p[1]-w[1]*p[0]; }
                    for (int a = 0; a < 3; ++a) s2 += v[a] * shape.basis[g][(3*l+a)*k+c];
                }
                worst = std::max(worst, std::abs(s2));
            }
        }
        std::printf("shape basis: %d unknowns for %d markers; orthonormal + perpendicular to rigid motions, max error %.1e\n", Z, M, worst);
    }

    // --- 2. Jacobian check ---
    {
        FrameData& f = frames[40]; const size_t A = f.active.size();
        std::vector<double> store(nq*A + Z); std::vector<const double*> P;
        for (int j = 0; j < nq; ++j) for (size_t a = 0; a < A; ++a) store[j*A+a] = truth(j, 0.4) + 0.05*j;
        for (int k = 0; k < Z; ++k) store[nq*A + k] = 0.004 * (k + 1);
        for (size_t p = 0; p < nq*A; ++p) P.push_back(&store[p]);
        for (size_t g = 0, o = 0; g < shape.members.size(); ++g) if (shape.size[g]) { P.push_back(&store[nq*A + o]); o += shape.size[g]; }
        std::vector<double> r(R3), Jall(R3 * store.size()); std::vector<double*> J;
        for (size_t p = 0; p < nq*A; ++p) J.push_back(&Jall[p*R3]);
        for (size_t g = 0, o = 0; g < shape.members.size(); ++g) if (shape.size[g]) { J.push_back(&Jall[(nq*A + o)*R3]); o += shape.size[g]; }
        frame_eval_shape_offsets(fk, f, shape, P.data(), r.data(), J.data());
        double worst = 0; const double h = 1e-7;
        for (size_t v = 0; v < store.size(); ++v) {
            std::vector<double> rp(R3), rm(R3); double s0 = store[v];
            store[v] = s0 + h; frame_eval_shape_offsets(fk, f, shape, P.data(), rp.data(), nullptr);
            store[v] = s0 - h; frame_eval_shape_offsets(fk, f, shape, P.data(), rm.data(), nullptr); store[v] = s0;
            double an_col_worst = 0;
            for (int i = 0; i < R3; ++i) {
                double an;
                if (v < nq*A) an = J[v][i];
                else {   // find the group block and column
                    size_t o = v - nq*A, b = nq*A; int col = 0; size_t g = 0;
                    for (; g < shape.members.size(); ++g) { if (!shape.size[g]) continue; if (o < (size_t)shape.size[g]) { col = (int)o; break; } o -= shape.size[g]; ++b; }
                    an = J[b][i * shape.size[g] + col];
                }
                an_col_worst = std::max(an_col_worst, std::abs((rp[i]-rm[i])/(2*h) - an));
            }
            worst = std::max(worst, an_col_worst);
        }
        std::printf("Jacobian check (coefficients + shape offsets): max |analytic - brute force| = %.2e\n", worst);
    }

    // --- 3. Gauss-Newton (LM-damped): mode 0 = offsets fixed, 1 = full offsets, 2 = shape-only ---
    auto solve = [&](int mode, const std::vector<FrameData>& data, std::vector<std::vector<double>>& c, std::vector<double>& d) {
        const int Nc = nq*K, No = mode == 1 ? 3*M : mode == 2 ? Z : 0, N = Nc + No;
        std::vector<double> z(Z, 0.0);
        for (int it = 0; it < 25; ++it) {
            std::vector<double> H(N*N, 0.0), gvec(N, 0.0);
            for (const auto& f : data) {
                const size_t A = f.active.size();
                std::vector<const double*> P; std::vector<int> cols; std::vector<double> r(R3);
                for (int j = 0; j < nq; ++j) for (size_t a = 0; a < A; ++a) { P.push_back(&c[j][f.active[a].k]); cols.push_back(j*K + f.active[a].k); }
                std::vector<std::vector<double>> colv(cols.size(), std::vector<double>(R3));
                std::vector<double*> J; for (auto& v : colv) J.push_back(v.data());
                if (mode == 0) { std::vector<double> d0(3*M, 0.0); for (int m = 0; m < M; ++m) P.push_back(&d0[3*m]);
                                 for (int m = 0; m < M; ++m) J.push_back(nullptr);
                                 frame_eval_offsets(fk, f, P.data(), r.data(), J.data()); }
                else if (mode == 1) {
                    std::vector<std::vector<double>> Jd(M, std::vector<double>(R3*3));
                    for (int m = 0; m < M; ++m) { P.push_back(&d[3*m]); J.push_back(Jd[m].data()); }
                    frame_eval_offsets(fk, f, P.data(), r.data(), J.data());
                    for (int m = 0; m < M; ++m) for (int k = 0; k < 3; ++k) { cols.push_back(Nc + 3*m + k);
                        std::vector<double> v(R3); for (int i = 0; i < R3; ++i) v[i] = Jd[m][i*3+k]; colv.push_back(v); }
                } else {
                    std::vector<std::vector<double>> Jz;
                    for (size_t g = 0, o = 0; g < shape.members.size(); ++g) if (shape.size[g]) {
                        P.push_back(&z[o]); Jz.emplace_back(R3 * shape.size[g]); o += shape.size[g]; }
                    for (auto& v : Jz) J.push_back(v.data());
                    frame_eval_shape_offsets(fk, f, shape, P.data(), r.data(), J.data());
                    for (size_t g = 0, o = 0, b = 0; g < shape.members.size(); ++g) if (shape.size[g]) {
                        for (int k = 0; k < shape.size[g]; ++k) { cols.push_back(Nc + o + k);
                            std::vector<double> v(R3); for (int i = 0; i < R3; ++i) v[i] = Jz[b][i*shape.size[g]+k]; colv.push_back(v); }
                        o += shape.size[g]; ++b; }
                }
                colv.resize(cols.size());
                for (size_t a = 0; a < cols.size(); ++a) {
                    for (int i = 0; i < R3; ++i) gvec[cols[a]] += colv[a][i] * r[i];
                    for (size_t b = 0; b < cols.size(); ++b) { double sm = 0; for (int i = 0; i < R3; ++i) sm += colv[a][i]*colv[b][i]; H[cols[a]*N + cols[b]] += sm; }
                }
            }
            const double w = 1000.0 / sigma_prior_mm;
            if (mode == 1) for (int k = 0; k < 3*M; ++k) { H[(Nc+k)*N + Nc+k] += w*w; gvec[Nc+k] += w*w*d[k]; }
            if (mode == 2) for (int k = 0; k < Z; ++k) { H[(Nc+k)*N + Nc+k] += w*w; gvec[Nc+k] += w*w*z[k]; }
            for (int n = 0; n < N; ++n) H[n*N+n] *= 1.0 + (it < 5 ? 1.0 : 1e-4);
            std::vector<double> A2 = H, b(N); for (int n = 0; n < N; ++n) b[n] = -gvec[n];
            for (int col = 0; col < N; ++col) { int pv = col; for (int r2 = col+1; r2 < N; ++r2) if (std::abs(A2[r2*N+col]) > std::abs(A2[pv*N+col])) pv = r2;
                for (int k2 = 0; k2 < N; ++k2) std::swap(A2[col*N+k2], A2[pv*N+k2]);
                std::swap(b[col], b[pv]);
                for (int r2 = col+1; r2 < N; ++r2) { double fct = A2[r2*N+col]/A2[col*N+col]; for (int k2 = col; k2 < N; ++k2) A2[r2*N+k2] -= fct*A2[col*N+k2]; b[r2] -= fct*b[col]; } }
            std::vector<double> dx(N); for (int r2 = N-1; r2 >= 0; --r2) { double v = b[r2]; for (int k2 = r2+1; k2 < N; ++k2) v -= A2[r2*N+k2]*dx[k2]; dx[r2] = v/A2[r2*N+r2]; }
            for (int n = 0; n < Nc; ++n) c[n / K][n % K] += dx[n];
            if (mode == 1) for (int k = 0; k < 3*M; ++k) d[k] += dx[Nc+k];
            if (mode == 2) for (int k = 0; k < Z; ++k) z[k] += dx[Nc+k];
        }
        if (mode == 2) { std::vector<const double*> zp; for (size_t g = 0, o = 0; g < shape.members.size(); ++g) { zp.push_back(&z[o]); o += shape.size[g]; }
                         shape.to_offsets(zp, M, d); }
    };
    auto report = [&](const char* label, std::vector<std::vector<double>>& c, std::vector<double>& d, const std::vector<std::array<double,3>>& dt_) {
        double e = 0; int n = 0, worst_j = 0; double wj = 0;
        for (int j = 2; j < nq; ++j) { double ej = 0; for (int i = 5; i < F-5; ++i) { double x = (evaluate(s, c[j], i*dt) - truth(j, i*dt)) * 180 / M_PI; ej += x*x; } ej = std::sqrt(ej/(F-10)); e += ej; ++n; if (ej > wj) { wj = ej; worst_j = j; } }
        double de = 0, dl = 0; for (int m = 0; m < M; ++m) for (int k = 0; k < 2; ++k) { double x = (d[3*m+k] - dt_[m][k]) * 1000; de += x*x; dl += d[3*m+k]*d[3*m+k]*1e6; }
        std::printf("  %-22s mean angle RMS error %.2f deg (worst %-11s %.2f)   offsets RMS %.1f mm, error %.1f mm\n", label, e/n,
                    fk.coordinate_names()[worst_j].c_str(), wj, std::sqrt(dl/(2*M)), std::sqrt(de/(2*M)));
    };

    // control data: same motion and noise, markers exactly where the model puts them
    std::vector<FrameData> control = frames;
    {
        std::mt19937 rng2(3); std::normal_distribution<double> g2(0, 1);
        for (int i = 0; i < F; ++i) {
            for (int j = 0; j < nq; ++j) q[j] = truth(j, i * dt);
            fk.marker_positions(q, pos);
            for (int m = 0; m < M; ++m)
                control[i].observed_marker_positions[m] = Coordinate{pos[3*m]*1000 + g2(rng2), pos[3*m+1]*1000 + g2(rng2), pos[3*m+2]*1000};
        }
    }
    const std::vector<std::array<double,3>> zero(M, {0, 0, 0});
    const char* names[3] = {"offsets fixed at 0", "full offsets", "shape-only offsets"};
    for (int set = 0; set < 2; ++set) {
        std::printf(set == 0 ? "data with misplaced markers (prior %.0f mm):\n" : "control, no true offsets:\n", sigma_prior_mm);
        for (int mode = 0; mode < 3; ++mode) {
            std::vector<std::vector<double>> c(nq, std::vector<double>(K, 0.0)); for (int k = 0; k < K; ++k) c[1][k] = 0.9;
            std::vector<double> d(3*M, 0.0);
            solve(mode, set == 0 ? frames : control, c, d);
            report(names[mode], c, d, set == 0 ? dtrue : zero);
        }
    }
    double tm = 0; for (auto& x : dtrue) tm += x[0]*x[0]+x[1]*x[1]; std::printf("(true offsets RMS %.1f mm per axis)\n", std::sqrt(tm/(2*M))*1000);
}
