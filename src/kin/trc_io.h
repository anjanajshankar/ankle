
#ifndef TRC_IO_H
#define TRC_IO_H

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

// Writes an OpenSim .trc marker file (the format parse_trc and OpenSim's IK tool read).
// positions_mm[i] holds x0, y0, z0, x1, ... for frame i, in mm, lab frame.
inline void write_trc(const std::string& path, const std::vector<std::string>& marker_names,
                      const std::vector<double>& times, const std::vector<std::vector<double>>& positions_mm) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("write_trc: cannot open " + path);
    const size_t F = times.size(), M = marker_names.size();
    const double rate = F > 1 ? (F - 1) / (times.back() - times.front()) : 100.0;
    out << "PathFileType\t4\t(X/Y/Z)\t" << path << "\n";
    out << "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n";
    out << rate << "\t" << rate << "\t" << F << "\t" << M << "\tmm\t" << rate << "\t1\t" << F << "\n";
    out << "Frame#\tTime";
    for (const auto& n : marker_names) out << "\t" << n << "\t\t";
    out << "\n\t";
    for (size_t m = 0; m < M; ++m) out << "\tX" << m + 1 << "\tY" << m + 1 << "\tZ" << m + 1;
    out << "\n\n";
    out.precision(10);
    for (size_t i = 0; i < F; ++i) {
        out << i + 1 << "\t" << times[i];
        for (size_t k = 0; k < 3 * M; ++k) out << "\t" << positions_mm[i][k];
        out << "\n";
    }
}

// Time of the first data row of a .trc file (the "Time" column), so frames can be matched to
// motions that start later (e.g. an IK run over only part of the trial).
inline double trc_first_time(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("trc_first_time: cannot open " + path);
    std::string line;
    for (int i = 0; i < 5 && std::getline(in, line); ++i) {}
    while (std::getline(in, line)) {
        if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
        const auto a = line.find('\t');
        const auto b = line.find('\t', a + 1);
        return std::stod(line.substr(a + 1, b - a - 1));
    }
    throw std::runtime_error("trc_first_time: no data rows in " + path);
}

#endif
