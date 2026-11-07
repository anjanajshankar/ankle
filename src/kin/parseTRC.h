
#ifndef PARSE_TRC_H
#define PARSE_TRC_H


#include <string>
#include "kin.h"



// Reads an OpenSim .trc marker file into an ActivityTrace.
//
// - Coordinates are converted to mm (from the file's Units: mm, cm or m).
// - A missing sample (empty fields, NaN, or exactly 0 0 0) becomes std::nullopt.
// - Every MarkerPt gets measurement_uncertainty = default_uncertainty_mm on each axis
//   and error_in_coord = 0 (IK fills it in later).
// - frame.idx is the 0-based row index of the data (row 0 = first time point).
//
// Throws std::runtime_error if the file can't be opened or doesn't match its own header.
ActivityTrace parse_trc(const std::string& file_path, PersonID person_id,
      double default_uncertainity_mm = 1.0) ;



#endif

