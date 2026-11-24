
// Sandbox check (no Ceres): marker offsets on the 2D toy.
// 1. Jacobian of frame_eval_offsets vs brute force.  2. Gauss-Newton with and without offset estimation,
// on data generated with misplaced markers (true offsets 10-15 mm), to see what offset estimation buys.
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

    // --- 1. Jacobian check ---
    {
        FrameData& f = frames[40]; const size_t A = f.active.size();
        std::vector<double> store(nq*A + 3*M); std::vector<const double*> P(nq*A + M);
        for (int j = 0; j < nq; ++j) for (size_t a = 0; a < A; ++a) { store[j*A+a] = truth(j, 0.4) + 0.05*j; P[j*A+a] = &store[j*A+a]; }
        for (int m = 0; m < M; ++m) { for (int k = 0; k < 3; ++k) store[nq*A + 3*m + k] = 0.003*(k+1)*(m%3); P[nq*A+m] = &store[nq*A+3*m]; }
        std::vector<double> r(R3), Jc(R3*nq*A), Jd(R3*3*M); std::vector<double*> J(nq*A + M);
        for (size_t p = 0; p < nq*A; ++p) J[p] = &Jc[p*R3];
        for (int m = 0; m < M; ++m) J[nq*A+m] = &Jd[m*R3*3];
        frame_eval_offsets(fk, f, P.data(), r.data(), J.data());
        double worst = 0; const double h = 1e-7;
        for (size_t v = 0; v < store.size(); ++v) {
            std::vector<double> rp(R3), rm(R3); double s0 = store[v];
            store[v] = s0 + h; frame_eval_offsets(fk, f, P.data(), rp.data(), nullptr);
            store[v] = s0 - h; frame_eval_offsets(fk, f, P.data(), rm.data(), nullptr); store[v] = s0;
            for (int i = 0; i < R3; ++i) {
                double an = v < nq*A ? J[v][i] : J[nq*A + (v-nq*A)/3][i*3 + (v-nq*A)%3];
                worst = std::max(worst, std::abs((rp[i]-rm[i])/(2*h) - an));
            }
        }
        std::printf("Jacobian check (coefficients + offsets): max |analytic - brute force| = %.2e\n", worst);
    }

    // --- 2. Gauss-Newton (LM-damped), with or without offsets ---
    auto solve = [&](bool fit_offsets, std::vector<std::vector<double>>& c, std::vector<double>& d) {
        const int Nc = nq*K, N = Nc + (fit_offsets ? 3*M : 0);
        for (int it = 0; it < 25; ++it) {
            std::vector<double> H(N*N, 0.0), gvec(N, 0.0);
            for (auto& f : frames) {
                const size_t A = f.active.size();
                std::vector<const double*> P(nq*A + M); std::vector<int> idx(nq*A);
                for (int j = 0; j < nq; ++j) for (size_t a = 0; a < A; ++a) { P[j*A+a] = &c[j][f.active[a].k]; idx[j*A+a] = j*K + f.active[a].k; }
                for (int m = 0; m < M; ++m) P[nq*A+m] = &d[3*m];
                std::vector<double> r(R3), Jc(R3*nq*A), Jd(R3*3*M); std::vector<double*> J(nq*A + M);
                for (size_t p = 0; p < nq*A; ++p) J[p] = &Jc[p*R3];
                for (int m = 0; m < M; ++m) J[nq*A+m] = &Jd[m*R3*3];
                frame_eval_offsets(fk, f, P.data(), r.data(), J.data());
                // dense rows: column list
                std::vector<int> cols; std::vector<std::vector<double>> colv;
                for (size_t p = 0; p < nq*A; ++p) { cols.push_back(idx[p]); colv.emplace_back(J[p], J[p]+R3); }
                if (fit_offsets) for (int m = 0; m < M; ++m) for (int k = 0; k < 3; ++k) {
                    cols.push_back(Nc + 3*m + k); std::vector<double> v(R3); for (int i = 0; i < R3; ++i) v[i] = J[nq*A+m][i*3+k]; colv.push_back(v); }
                for (size_t a = 0; a < cols.size(); ++a) {
                    for (int i = 0; i < R3; ++i) gvec[cols[a]] += colv[a][i] * r[i];
                    for (size_t b = 0; b < cols.size(); ++b) { double sm = 0; for (int i = 0; i < R3; ++i) sm += colv[a][i]*colv[b][i]; H[cols[a]*N + cols[b]] += sm; }
                }
            }
            if (fit_offsets) for (int k = 0; k < 3*M; ++k) { const double w = 1000.0 / sigma_prior_mm; H[(Nc+k)*N + Nc+k] += w*w; gvec[Nc+k] += w*w*d[k]; }
            for (int n = 0; n < N; ++n) H[n*N+n] *= 1.0 + (it < 5 ? 1.0 : 1e-4);
            std::vector<double> A2 = H, b(N); for (int n = 0; n < N; ++n) b[n] = -gvec[n];
            for (int col = 0; col < N; ++col) { int pv = col; for (int r2 = col+1; r2 < N; ++r2) if (std::abs(A2[r2*N+col]) > std::abs(A2[pv*N+col])) pv = r2;
                for (int k2 = 0; k2 < N; ++k2) std::swap(A2[col*N+k2], A2[pv*N+k2]);
                std::swap(b[col], b[pv]);
                for (int r2 = col+1; r2 < N; ++r2) { double fct = A2[r2*N+col]/A2[col*N+col]; for (int k2 = col; k2 < N; ++k2) A2[r2*N+k2] -= fct*A2[col*N+k2]; b[r2] -= fct*b[col]; } }
            std::vector<double> dx(N); for (int r2 = N-1; r2 >= 0; --r2) { double v = b[r2]; for (int k2 = r2+1; k2 < N; ++k2) v -= A2[r2*N+k2]*dx[k2]; dx[r2] = v/A2[r2*N+r2]; }
            for (int n = 0; n < Nc; ++n) c[n / K][n % K] += dx[n];
            if (fit_offsets) for (int k = 0; k < 3*M; ++k) d[k] += dx[Nc+k];
        }
    };
    auto report = [&](const char* label, std::vector<std::vector<double>>& c, std::vector<double>& d) {
        double e = 0; int n = 0, worst_j = 0; double wj = 0;
        for (int j = 2; j < nq; ++j) { double ej = 0; for (int i = 5; i < F-5; ++i) { double x = (evaluate(s, c[j], i*dt) - truth(j, i*dt)) * 180 / M_PI; ej += x*x; } ej = std::sqrt(ej/(F-10)); e += ej; ++n; if (ej > wj) { wj = ej; worst_j = j; } }
        double de = 0; for (int m = 0; m < M; ++m) for (int k = 0; k < 2; ++k) { double x = (d[3*m+k] - dtrue[m][k]) * 1000; de += x*x; }
        std::printf("%-26s mean angle RMS error %.2f deg (worst %s %.2f)   offset error RMS %.1f mm\n", label, e/n,
                    fk.coordinate_names()[worst_j].c_str(), wj, std::sqrt(de/(2*M)));
    };
    for (int fit = 0; fit < 2; ++fit) {
        std::vector<std::vector<double>> c(nq, std::vector<double>(K, 0.0)); for (int k = 0; k < K; ++k) c[1][k] = 0.9;
        std::vector<double> d(3*M, 0.0);
        solve(fit == 1, c, d);
        report(fit ? "IK + offsets (prior 20 mm)" : "IK, offsets fixed at 0", c, d);
    }
    double tm = 0; for (auto& x : dtrue) tm += x[0]*x[0]+x[1]*x[1]; std::printf("(true offsets RMS %.1f mm per axis)\n", std::sqrt(tm/(2*M))*1000);
}
