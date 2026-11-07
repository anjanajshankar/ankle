
#ifndef IK_WEIGHTS_H
#define IK_WEIGHTS_H

#include <fstream>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Marker weights from an OpenSim IK setup file: <IKMarkerTask name="X"> ... <apply> ... <weight>.
// Markers with apply=false get weight 0. Markers not listed keep weight 1.
inline std::map<std::string, double> read_ik_weights(const std::string& path) {
    std::map<std::string, double> w;
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string text = ss.str();
    const std::regex task(R"re(<IKMarkerTask\s+name="([^"]+)"\s*>([\s\S]*?)</IKMarkerTask>)re");
    const std::regex apply(R"re(<apply>\s*(\w+)\s*</apply>)re"), weight(R"re(<weight>\s*([-+0-9.eE]+)\s*</weight>)re");
    for (auto it = std::sregex_iterator(text.begin(), text.end(), task); it != std::sregex_iterator(); ++it) {
        const std::string name = (*it)[1], body = (*it)[2];
        std::smatch m;
        const bool on = !std::regex_search(body, m, apply) || m[1] == "true";
        const double wt = std::regex_search(body, m, weight) ? std::stod(m[1]) : 1.0;
        w[name] = on ? wt : 0.0;
    }
    return w;
}

// Body-name prefixes of the arm segments in Rajagopal-style models (used by --no-arms).
inline const std::vector<std::string>& arm_body_prefixes() {
    static const std::vector<std::string> p = {"humerus", "ulna", "radius", "hand"};
    return p;
}

#endif
