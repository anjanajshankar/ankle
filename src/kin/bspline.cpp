
#include "bspline.h"

#include <cmath>
#include <stdexcept>
#include <utility>

BSpline make_clamped(double t0, double t1, const std::vector<double>& interior, int degree) {
    if (!(t0 < t1)) throw std::invalid_argument("make_clamped: need t0 < t1");
    BSpline s;
    s.degree = degree;
    s.knots.assign(degree + 1, t0);
    double prev = t0;
    for (double k : interior) {
        if (!(k > prev && k < t1)) throw std::invalid_argument("make_clamped: interior knots must increase strictly inside (t0, t1)");
        s.knots.push_back(k);
        prev = k;
    }
    s.knots.insert(s.knots.end(), degree + 1, t1);
    return s;
}

BSpline make_uniform(double t0, double t1, int num_intervals, int degree) {
    std::vector<double> interior;
    for (int i = 1; i < num_intervals; ++i) interior.push_back(t0 + (t1 - t0) * i / num_intervals);
    return make_clamped(t0, t1, interior, degree);
}

namespace {

// a / b, treating 0/0 (and x/0) as 0. Repeated knots make some denominators zero;
// the B-spline definition says those terms are dropped.
double safe_div(double a, double b) { return b == 0.0 ? 0.0 : a / b; }

// Cox-de Boor recursion, with the derivative formula
//   d/dt B_{k,p} = p * ( B_{k,p-1} / (u_{k+p} - u_k)  -  B_{k+1,p-1} / (u_{k+p+1} - u_{k+1}) ).
double basis_rec(const std::vector<double>& u, int k, int p, double t, int d) {
    if (d > p) return 0.0;  // a degree-p polynomial has no derivative above order p
    if (d > 0)
        return p * (safe_div(basis_rec(u, k, p - 1, t, d - 1), u[k + p] - u[k]) -
                    safe_div(basis_rec(u, k + 1, p - 1, t, d - 1), u[k + p + 1] - u[k + 1]));
    if (p == 0) {
        // Degree 0: 1 on [u_k, u_{k+1}), else 0. At the very last time, use the last
        // non-empty interval instead, so the curve is defined at its end point too.
        if (u[k] <= t && t < u[k + 1]) return 1.0;
        return (t == u.back() && u[k] < u[k + 1] && u[k + 1] == u.back()) ? 1.0 : 0.0;
    }
    return safe_div(t - u[k], u[k + p] - u[k]) * basis_rec(u, k, p - 1, t, 0) +
           safe_div(u[k + p + 1] - t, u[k + p + 1] - u[k + 1]) * basis_rec(u, k + 1, p - 1, t, 0);
}

}  // namespace

double basis(const BSpline& s, int k, double t, int d) {
    if (k < 0 || k >= s.num_coeffs()) throw std::out_of_range("basis: k out of range");
    return basis_rec(s.knots, k, s.degree, t, d);
}

double evaluate(const BSpline& s, const std::vector<double>& c, double t, int d) {
    if (static_cast<int>(c.size()) != s.num_coeffs()) throw std::invalid_argument("evaluate: wrong number of coefficients");
    double sum = 0.0;
    for (int k = 0; k < s.num_coeffs(); ++k) sum += c[k] * basis(s, k, t, d);
    return sum;
}

std::vector<double> fit(const BSpline& s, const std::vector<double>& t, const std::vector<double>& y) {
    const int n = s.num_coeffs();
    if (t.size() != y.size()) throw std::invalid_argument("fit: t and y differ in length");
    if (static_cast<int>(t.size()) < n) throw std::invalid_argument("fit: fewer samples than coefficients");

    // Normal equations (A^T A) c = A^T y, where A[i][k] = B_k(t_i).
    std::vector<std::vector<double>> M(n, std::vector<double>(n + 1, 0.0));  // augmented [A^T A | A^T y]
    std::vector<double> row(n);
    for (size_t i = 0; i < t.size(); ++i) {
        for (int k = 0; k < n; ++k) row[k] = basis(s, k, t[i]);
        for (int a = 0; a < n; ++a) {
            if (row[a] == 0.0) continue;
            for (int b = 0; b < n; ++b) M[a][b] += row[a] * row[b];
            M[a][n] += row[a] * y[i];
        }
    }

    // Gaussian elimination with partial pivoting.
    for (int col = 0; col < n; ++col) {
        int piv = col;
        for (int r = col + 1; r < n; ++r)
            if (std::abs(M[r][col]) > std::abs(M[piv][col])) piv = r;
        if (std::abs(M[piv][col]) < 1e-12)
            throw std::runtime_error("fit: singular system (an interval between knots has no samples?)");
        std::swap(M[col], M[piv]);
        for (int r = col + 1; r < n; ++r) {
            const double f = M[r][col] / M[col][col];
            for (int c2 = col; c2 <= n; ++c2) M[r][c2] -= f * M[col][c2];
        }
    }
    std::vector<double> c(n);
    for (int r = n - 1; r >= 0; --r) {
        double v = M[r][n];
        for (int c2 = r + 1; c2 < n; ++c2) v -= M[r][c2] * c[c2];
        c[r] = v / M[r][r];
    }
    return c;
}
