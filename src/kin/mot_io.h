
#ifndef MOT_IO_H
#define MOT_IO_H

#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

// Writes an OpenSim-style .mot: one column per coordinate, rotations converted to degrees.
// values[i][j] = coordinate j at times[i], in radians (rotations) or metres (translations).
// The same function writes velocities or accelerations (deg/s, deg/s^2 for rotations).
inline void write_mot(const std::string& path, const std::string& title,
                      const std::vector<std::string>& names, const std::vector<bool>& rotational,
                      const std::vector<double>& times, const std::vector<std::vector<double>>& values) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("write_mot: cannot open " + path);
    out << title << "\nversion=1\nnRows=" << times.size() << "\nnColumns=" << names.size() + 1
        << "\ninDegrees=yes\nendheader\ntime";
    for (const auto& n : names) out << "\t" << n;
    out << "\n";
    out.precision(10);
    for (size_t i = 0; i < times.size(); ++i) {
        out << times[i];
        for (size_t j = 0; j < names.size(); ++j)
            out << "\t" << (rotational[j] ? values[i][j] * 180.0 / M_PI : values[i][j]);
        out << "\n";
    }
}

// Path for a derivative file next to a .mot: "dir/run.mot" + "_u" -> "dir/run_u.mot".
inline std::string mot_with_suffix(const std::string& path, const std::string& suffix) {
    const std::string stem = (path.size() > 4 && path.compare(path.size() - 4, 4, ".mot") == 0)
                                 ? path.substr(0, path.size() - 4) : path;
    return stem + suffix + ".mot";
}

#endif
