
#include "event_knots.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

std::vector<double> detect_contact_events(const std::string& path, double threshold_N) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("detect_contact_events: cannot open " + path);

    // Skip the header up to "endheader"; the next line holds the column names.
    std::string line;
    bool found = false;
    while (std::getline(in, line)) {
        std::string t = line;
        t.erase(std::remove_if(t.begin(), t.end(), ::isspace), t.end());
        std::transform(t.begin(), t.end(), t.begin(), ::tolower);
        if (t == "endheader") { found = true; break; }
    }
    if (!found || !std::getline(in, line)) throw std::runtime_error("detect_contact_events: no header in " + path);

    std::vector<std::string> names;
    {
        std::stringstream ss(line);
        std::string n;
        while (ss >> n) names.push_back(n);
    }
    std::vector<int> cols;   // vertical force columns
    for (size_t i = 0; i < names.size(); ++i) {
        const std::string& n = names[i];
        if (n.find("force") != std::string::npos && n.size() >= 3 && n.compare(n.size() - 3, 3, "_vy") == 0)
            cols.push_back(static_cast<int>(i));
    }
    if (cols.empty()) throw std::runtime_error("detect_contact_events: no *force*_vy columns in " + path);

    std::vector<double> events;
    std::vector<int> prev(cols.size(), -1);   // -1 = unknown, 0 = off, 1 = on
    while (std::getline(in, line)) {
        std::stringstream ss(line);
        std::vector<double> v;
        double x;
        while (ss >> x) v.push_back(x);
        if (v.size() < names.size()) continue;
        for (size_t c = 0; c < cols.size(); ++c) {
            const int on = v[cols[c]] > threshold_N ? 1 : 0;
            if (prev[c] >= 0 && on != prev[c]) events.push_back(v[0]);
            prev[c] = on;
        }
    }
    std::sort(events.begin(), events.end());
    return events;
}

std::vector<double> event_knots(double T, double base_spacing, double dense_spacing,
                                const std::vector<double>& events, double before, double after) {
    if (base_spacing <= 0 || dense_spacing <= 0) throw std::invalid_argument("event_knots: spacings must be > 0");

    std::vector<double> cand;
    for (double t = base_spacing; t < T; t += base_spacing) cand.push_back(t);
    for (double e : events)
        for (double t = e - before; t <= e + after + 1e-12; t += dense_spacing) cand.push_back(t);
    std::sort(cand.begin(), cand.end());

    // Keep knots strictly inside (0, T), at least min_gap from each other and from both ends.
    const double min_gap = 0.999 * dense_spacing;
    std::vector<double> knots;
    double last = 0.0;
    for (double t : cand) {
        if (t <= 0.0 || t >= T) continue;
        if (t - last < min_gap) continue;
        if (T - t < min_gap) continue;
        knots.push_back(t);
        last = t;
    }
    return knots;
}

std::vector<ContactOnset> detect_contact_onsets(const std::string& path, double threshold_N) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("detect_contact_onsets: cannot open " + path);
    std::string line;
    bool found = false;
    while (std::getline(in, line)) {
        std::string t = line;
        t.erase(std::remove_if(t.begin(), t.end(), ::isspace), t.end());
        std::transform(t.begin(), t.end(), t.begin(), ::tolower);
        if (t == "endheader") { found = true; break; }
    }
    if (!found || !std::getline(in, line)) throw std::runtime_error("detect_contact_onsets: no header in " + path);
    std::vector<std::string> names;
    {
        std::stringstream ss(line);
        std::string n;
        while (ss >> n) names.push_back(n);
    }
    std::vector<int> cols;
    std::vector<char> side;
    for (size_t i = 0; i < names.size(); ++i) {
        std::string n = names[i];
        std::transform(n.begin(), n.end(), n.begin(), ::tolower);
        if (n.find("force") == std::string::npos || n.size() < 3 || n.compare(n.size() - 3, 3, "_vy") != 0) continue;
        cols.push_back(static_cast<int>(i));
        const bool left = n.compare(0, 2, "l_") == 0 || n.find("_l_") != std::string::npos || n.find("left") != std::string::npos;
        side.push_back(left ? 'l' : 'r');
    }
    if (cols.empty()) throw std::runtime_error("detect_contact_onsets: no *force*_vy columns in " + path);
    std::vector<ContactOnset> out;
    std::vector<int> prev(cols.size(), -1);
    while (std::getline(in, line)) {
        std::stringstream ss(line);
        std::vector<double> v;
        double x;
        while (ss >> x) v.push_back(x);
        if (v.size() < names.size()) continue;
        for (size_t c = 0; c < cols.size(); ++c) {
            const int on = v[cols[c]] > threshold_N ? 1 : 0;
            if (prev[c] == 0 && on == 1) out.push_back({v[0], side[c]});
            prev[c] = on;
        }
    }
    std::sort(out.begin(), out.end(), [](const ContactOnset& a, const ContactOnset& b) { return a.time < b.time; });
    return out;
}
