
#ifndef EVENT_KNOTS_H
#define EVENT_KNOTS_H

#include <string>
#include <vector>

// Foot-contact events from a ground reaction force file (.mot, as used by OpenSim's ID tool).
// Uses every column whose name contains "force" and ends in "_vy" (vertical force, e.g.
// ground_force_r_vy). An event is each time a foot's vertical force crosses threshold_N,
// in either direction: heel strike (rising) or toe-off (falling).
// Returns the event times, sorted, in the file's own time.
std::vector<double> detect_contact_events(const std::string& grf_mot_path, double threshold_N = 20.0);

// Foot contacts only (rising crossings), with the side: 'r' or 'l'. The side is guessed from the
// column name: left if it starts with "l_" or contains "_l_" or "left", otherwise right.
// (Check the printout if your plates are named by number rather than by foot.)
struct ContactOnset { double time; char side; };
std::vector<ContactOnset> detect_contact_onsets(const std::string& grf_mot_path, double threshold_N = 20.0);

// Interior knots on [0, T]: a regular grid every base_spacing, plus a dense grid every
// dense_spacing from (event - before) to (event + after) around each event.
// Knots closer together than dense_spacing are merged, so no interval is shorter than that.
// Pass the result to make_clamped(0, T, knots).
std::vector<double> event_knots(double T, double base_spacing, double dense_spacing,
                                const std::vector<double>& events, double before, double after);

#endif
