
// Tests for event_knots: event detection on a small fake GRF file, knot placement rules,
// and the reason for it all: a curve with a fast change at an event is followed much better
// by event-based knots than by uniform knots with the same number of coefficients.
#include <cmath>
#include <cstdio>
#include <fstream>
#include <vector>

#include "bspline.h"
#include "event_knots.h"

static int failures = 0;
static void check(bool ok, const char* what) {
    std::printf("%-62s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++failures;
}

int main() {
    // --- 1. event detection: right foot on 0.20-0.80 s, left foot on 0.60-1.30 s ---
    {
        std::ofstream f("fake_grf.mot");
        f << "fake\nversion=1\nnRows=151\nnColumns=3\ninDegrees=no\nendheader\n";
        f << "time\tground_force_r_vy\tground_force_l_vy\n";
        for (int i = 0; i <= 150; ++i) {
            const double t = i * 0.01;
            const double r = (t >= 0.2 - 1e-9 && t < 0.8 - 1e-9) ? 700.0 : 0.0;
            const double l = (t >= 0.6 - 1e-9 && t < 1.3 - 1e-9) ? 700.0 : 0.0;
            f << t << "\t" << r << "\t" << l << "\n";
        }
    }
    const std::vector<double> ev = detect_contact_events("fake_grf.mot");
    std::printf("events found:");
    for (double e : ev) std::printf(" %.2f", e);
    std::printf("\n");
    const double expect[4] = {0.2, 0.6, 0.8, 1.3};
    bool ev_ok = ev.size() == 4;
    for (size_t i = 0; ev_ok && i < 4; ++i) ev_ok = std::abs(ev[i] - expect[i]) < 1e-6;
    check(ev_ok, "detects heel strikes and toe-offs of both feet");

    // --- 2. knot rules ---
    const double T = 1.5;
    const std::vector<double> kn = event_knots(T, 0.10, 0.01, ev, 0.03, 0.10);
    bool inside = true, gaps = true;
    for (size_t i = 0; i < kn.size(); ++i) {
        inside = inside && kn[i] > 0 && kn[i] < T;
        if (i > 0) gaps = gaps && kn[i] - kn[i - 1] >= 0.0099;
    }
    check(inside, "all knots strictly inside (0, T)");
    check(gaps, "no two knots closer than the dense spacing");
    int near_event = 0;
    for (double k : kn) if (k >= 0.17 && k <= 0.30) ++near_event;
    check(near_event >= 10, "dense knots in the window around the first event");
    std::printf("event-based knots: %zu interior knots\n", kn.size());

    // --- 3. why: fit a curve with a fast change right after each event ---
    // A smooth wave plus a quick 0.03 s "kick" after each event, like an ankle at foot contact.
    auto curve = [&](double t) {
        double y = 0.3 * std::sin(2 * M_PI * t);
        for (double e : ev) y += 0.2 * std::exp(-std::pow((t - e - 0.03) / 0.015, 2));
        return y;
    };
    std::vector<double> ts, ys;
    for (int i = 0; i <= 1500; ++i) { ts.push_back(i * 0.001); ys.push_back(curve(i * 0.001)); }

    const BSpline evspline = make_clamped(0, T, kn);
    const BSpline uniform = make_uniform(0, T, evspline.num_coeffs() - 3);   // same number of coefficients
    auto max_err = [&](const BSpline& s) {
        const std::vector<double> c = fit(s, ts, ys);
        double e = 0;
        for (size_t i = 0; i < ts.size(); ++i) e = std::max(e, std::abs(evaluate(s, c, ts[i]) - ys[i]));
        return e;
    };
    const double e_uni = max_err(uniform), e_ev = max_err(evspline);
    std::printf("same %d coefficients: max error uniform %.4f, event-based %.4f\n",
                evspline.num_coeffs(), e_uni, e_ev);
    check(e_ev < 0.5 * e_uni, "event-based knots follow the fast changes much better");

    std::printf(failures ? "\n%d test(s) FAILED\n" : "\nall tests passed\n", failures);
    return failures ? 1 : 0;
}
