
#ifndef BSPLINE_H
#define BSPLINE_H

#include <vector>

// A B-spline "shape": its degree and knot vector. The coefficients are stored separately,
// because in IK they are the unknowns the optimizer changes.
struct BSpline {
    int degree = 3;               // 3 = cubic
    std::vector<double> knots;    // non-decreasing times
    int num_coeffs() const { return static_cast<int>(knots.size()) - degree - 1; }
};

// Clamped knot vector on [t0, t1]: (degree + 1) copies of t0 and of t1, with the given
// interior knots (strictly between t0 and t1, increasing) in between.
// "Clamped" makes the curve start at the first coefficient and end at the last one.
BSpline make_clamped(double t0, double t1, const std::vector<double>& interior, int degree = 3);

// Same, with num_intervals equal-length intervals between t0 and t1.
BSpline make_uniform(double t0, double t1, int num_intervals, int degree = 3);

// Value (d = 0) or d-th time derivative (d = 1, 2) of basis function k at time t.
double basis(const BSpline& s, int k, double t, int d = 0);

// q(t) = sum_k c[k] * B_k(t), or its d-th derivative.
double evaluate(const BSpline& s, const std::vector<double>& c, double t, int d = 0);

// Least-squares coefficients so the spline passes as close as possible to samples (t[i], y[i]).
std::vector<double> fit(const BSpline& s, const std::vector<double>& t, const std::vector<double>& y);

#endif
