
// Tests for the B-spline module.
#include <cmath>
#include <cstdio>
#include <vector>
#include "bspline.h"

static int failures = 0;
static void check(bool ok, const char* what, double value) {
    std::printf("%-52s %-4s (%.2e)\n", what, ok ? "ok" : "FAIL", value);
    if (!ok) ++failures;
}

// Basis functions must sum to 1 everywhere (value), and their derivatives must sum to 0.
static void test_partition_of_unity(const BSpline& s, const char* label) {
    double worst0 = 0, worst1 = 0;
    const double t0 = s.knots.front(), t1 = s.knots.back();
    for (int i = 0; i <= 1000; ++i) {
        const double t = t0 + (t1 - t0) * i / 1000.0;   // includes both end points
        double sum0 = 0, sum1 = 0;
        for (int k = 0; k < s.num_coeffs(); ++k) { sum0 += basis(s, k, t); sum1 += basis(s, k, t, 1); }
        worst0 = std::max(worst0, std::abs(sum0 - 1.0));
        worst1 = std::max(worst1, std::abs(sum1));
    }
    char msg[128];
    std::snprintf(msg, sizeof msg, "%s: basis sums to 1", label);
    check(worst0 < 1e-12, msg, worst0);
    std::snprintf(msg, sizeof msg, "%s: basis derivatives sum to 0", label);
    check(worst1 < 1e-9, msg, worst1);
}

// Fit sin(t) on [0, 2*pi]; check q, q', q'' against sin, cos, -sin.
static void test_fit_sin(const BSpline& s, const char* label) {
    std::vector<double> ts, ys;
    for (int i = 0; i <= 400; ++i) { const double t = 2 * M_PI * i / 400.0; ts.push_back(t); ys.push_back(std::sin(t)); }
    const std::vector<double> c = fit(s, ts, ys);
    double e0 = 0, e1 = 0, e2 = 0;
    for (double t : ts) {
        e0 = std::max(e0, std::abs(evaluate(s, c, t, 0) - std::sin(t)));
        e1 = std::max(e1, std::abs(evaluate(s, c, t, 1) - std::cos(t)));
        e2 = std::max(e2, std::abs(evaluate(s, c, t, 2) + std::sin(t)));
    }
    char msg[128];
    std::snprintf(msg, sizeof msg, "%s: fits sin       (max error)", label);  check(e0 < 1e-4, msg, e0);
    std::snprintf(msg, sizeof msg, "%s: q'  matches cos (max error)", label); check(e1 < 1e-2, msg, e1);
    std::snprintf(msg, sizeof msg, "%s: q'' matches -sin (max error)", label); check(e2 < 1e-1, msg, e2);
}

int main() {
    const BSpline uniform = make_uniform(0.0, 2 * M_PI, 20);
    std::printf("uniform: %d coefficients\n", uniform.num_coeffs());
    test_partition_of_unity(uniform, "uniform");
    test_fit_sin(uniform, "uniform");

    // Uneven: dense knots around t = pi (like the cluster around foot contact), sparser elsewhere.
    // Knots every 0.3 s, plus a dense cluster every 0.05 s between 2.9 and 3.4.
    std::vector<double> interior;
    for (double k = 0.3; k < 2.9; k += 0.3) interior.push_back(k);
    for (double k = 2.9; k < 3.4; k += 0.05) interior.push_back(k);
    for (double k = 3.6; k < 6.2; k += 0.3) interior.push_back(k);
    const BSpline uneven = make_clamped(0.0, 2 * M_PI, interior);
    std::printf("uneven:  %d coefficients\n", uneven.num_coeffs());
    test_partition_of_unity(uneven, "uneven");
    test_fit_sin(uneven, "uneven");

    std::printf(failures ? "\n%d test(s) FAILED\n" : "\nall tests passed\n", failures);
    return failures ? 1 : 0;
}
