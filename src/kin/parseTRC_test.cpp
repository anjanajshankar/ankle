
// Quick check of read_trc: prints counts, the first sample of one marker,
// and how many samples are missing for each marker that has gaps.
#include <iostream>
#include "parseTRC.h"

int main(int argc, char** argv) {
   if (argc < 2) {
      std::cerr << "usage: trc_test <file.trc> [marker_to_print]\n";
      return 1;
   }
   const std::string marker = argc >= 3 ? argv[2] : "R.Shoulder";

   try {
      const ActivityTrace trace = parse_trc(argv[1], /*person_id=*/0);
      const auto& trajs = trace.trajectories_by_marker;
      const size_t frames = trajs.empty() ? 0 : trajs.begin()->second.marker_pts.size();

      std::cout << "markers: " << trajs.size() << "\n"
         << "frames:  " << frames << "\n"
         << "time step: " << trace.time_step_millis << " ms\n";

      const auto it = trajs.find(marker);
      if (it == trajs.end()) {
         std::cout << "marker '" << marker << "' not found\n";
      } else {
         const auto& p = it->second.marker_pts.front();
         std::cout << marker << " frame 0: ";
         if (p.observed_coord)
            std::cout << p.observed_coord->x_mm << ", " << p.observed_coord->y_mm << ", "
               << p.observed_coord->z_mm << " mm\n";
         else
            std::cout << "missing\n";
      }

      int with_gaps = 0;
      for (const auto& [name, traj] : trajs) {
         int missing = 0;
         for (const auto& p : traj.marker_pts)
            if (!p.observed_coord) ++missing;
         if (missing > 0) {
            std::cout << "  " << name << ": " << missing << " missing\n";
            ++with_gaps;
         }
      }
      std::cout << with_gaps << " marker(s) have missing samples\n";
   } catch (const std::exception& e) {
      std::cerr << "error: " << e.what() << "\n";
      return 1;
   }
   return 0;
}
