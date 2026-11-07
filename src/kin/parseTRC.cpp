#include "parseTRC.h"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {

   std::string trim_ws(const std::string& s) {
      const char* ws = " \t\r\n";
      const auto b = s.find_first_not_of(ws);
      if (b == std::string::npos) return "";
      const auto e = s.find_last_not_of(ws);
      return s.substr(b, e - b + 1);
   }

   std::vector<std::string> split_tabs(const std::string& line) {
      std::vector<std::string> out;
      std::string field;
      std::stringstream ss(line);
      while (std::getline(ss, field, '\t')) out.push_back(trim_ws(field));
      if (!line.empty() && line.back() == '\t') out.push_back("");
      return out;
   }

   std::vector<std::string> split_ws(const std::string& line) {
      std::vector<std::string> out;
      std::stringstream ss(line); 
      std::string tok;
      while (ss >> tok) out.push_back(tok);
      return out;
   }

   std::optional<double> parse_number(const std::string& s) {
      if (s.empty()) return std::nullopt;
      char* end = nullptr;
      const double v = std::strtod(s.c_str(), &end);
      if (end == s.c_str() || std::isnan(v)) return std::nullopt;
      return v;
   }

   double units_to_mm(const std::string& units) {
      if (units == "mm") return 1.0;
      if (units == "cm") return 10.0;
      if (units == "m")  return 1000.0;
      throw std::runtime_error("read_trc: unsupported Units '" + units + "'");
   }

}  // namespace

   ActivityTrace parse_trc(const std::string& path, PersonID person_id, double default_uncertainty_mm) {
      std::ifstream in(path);
      if (!in) throw std::runtime_error("read_trc: cannot open " + path);

      std::string line1, line2, line3, line4, line5;
      if (!std::getline(in, line1) || !std::getline(in, line2) || !std::getline(in, line3) ||
            !std::getline(in, line4) || !std::getline(in, line5))
         throw std::runtime_error("read_trc: header shorter than 5 lines in " + path);

      const auto keys = split_ws(line2);
      const auto vals = split_ws(line3);
      if (keys.size() != vals.size())
         throw std::runtime_error("read_trc: header lines 2 and 3 have different lengths");
      std::map<std::string, std::string> header;
      for (size_t i = 0; i < keys.size(); ++i) header[keys[i]] = vals[i];
      for (const char* k : {"DataRate", "NumFrames", "NumMarkers", "Units"})
         if (!header.count(k)) throw std::runtime_error(std::string("read_trc: header missing ") + k);

      const double data_rate = std::stod(header["DataRate"]);
      const int num_frames = std::stoi(header["NumFrames"]);
      const int num_markers = std::stoi(header["NumMarkers"]);
      const double to_mm = units_to_mm(header["Units"]);

      // Line 4: "Frame#", "Time", then marker names, each followed by two empty fields.
      std::vector<MarkerName> names;
      const auto name_fields = split_tabs(line4);
      for (size_t i = 2; i < name_fields.size(); ++i)
         if (!name_fields[i].empty()) names.push_back(name_fields[i]);
      if (static_cast<int>(names.size()) != num_markers)
         throw std::runtime_error("read_trc: found " + std::to_string(names.size()) +
               " marker names but header says NumMarkers = " +
               std::to_string(num_markers));

      ActivityTrace trace;
      trace.person_id = person_id;
      trace.time_step_millis = 1000.0 / data_rate;
      for (const auto& n : names) trace.trajectories_by_marker[n].marker_pts.reserve(num_frames);

      const Coordinate uncertainty{default_uncertainty_mm, default_uncertainty_mm, default_uncertainty_mm};
      const Coordinate zero{0.0, 0.0, 0.0};

      // Data rows (blank lines, such as line 6, are skipped).
      std::string line;
      int row = 0;
      while (std::getline(in, line)) {
         if (trim_ws(line).empty()) continue;
         const auto f = split_tabs(line);
         if (f.size() < 2) throw std::runtime_error("read_trc: malformed data row " + std::to_string(row + 1));

         for (size_t m = 0; m < names.size(); ++m) {
            MarkerPt pt{std::nullopt, uncertainty, zero, Frame{row}};
            const size_t c = 2 + 3 * m;
            if (c + 2 < f.size()) {
               const auto x = parse_number(f[c]);
               const auto y = parse_number(f[c + 1]);
               const auto z = parse_number(f[c + 2]);
               const bool all_zero = x && y && z && *x == 0.0 && *y == 0.0 && *z == 0.0;
               if (x && y && z && !all_zero)
                  pt.observed_coord = Coordinate{*x * to_mm, *y * to_mm, *z * to_mm};
            }
            trace.trajectories_by_marker[names[m]].marker_pts.push_back(pt);
         }
         ++row;
      }

      if (row != num_frames)
         throw std::runtime_error("read_trc: read " + std::to_string(row) +
               " data rows but header says NumFrames = " + std::to_string(num_frames));
      return trace;
   }

