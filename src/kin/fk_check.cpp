
// fk_check: forward kinematics with OpenSim, checked against OpenSim's own IK output.
//
// For every frame of an IK result (.mot):
//   1. set each joint coordinate q on the model,
//   2. ask OpenSim where each marker is in the lab (ground) frame   <- this is FK_m(q, s, p_m)
//   3. compare with the marker positions IK reported (*_ik_model_marker_locations.sto).
// They should agree to well under 0.1 mm if FK is being used correctly.
//
// usage: fk_check <scaled_model.osim> <ik_output.mot> <ik_model_marker_locations.sto>

#include <OpenSim/OpenSim.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace OpenSim;

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: fk_check <model.osim> <ik.mot> <ik_model_marker_locations.sto>\n";
        return 1;
    }

    try {
        Model model(argv[1]);
        SimTK::State& state = model.initSystem();

        // Joint angles from IK. IK writes rotations in degrees; convert to radians,
        // using OpenSim's own rule for which coordinates are rotational.
        Storage q_sto(argv[2]);
        if (q_sto.isInDegrees()) model.getSimbodyEngine().convertDegreesToRadians(q_sto);

        // Marker positions reported by IK (metres, lab frame).
        Storage m_sto(argv[3]);

        const Array<std::string>& q_labels = q_sto.getColumnLabels();
        const Array<std::string>& m_labels = m_sto.getColumnLabels();

        // Which .mot column holds each coordinate (-1 = not in the file).
        const CoordinateSet& coords = model.getCoordinateSet();
        std::vector<int> q_col(coords.getSize(), -1);
        for (int i = 0; i < coords.getSize(); ++i) {
            q_col[i] = q_labels.findIndex(coords.get(i).getName());
            if (q_col[i] < 0)
                std::cout << "note: coordinate " << coords.get(i).getName()
                          << " not in .mot, left at default\n";
        }

        // Which .sto columns hold each marker's x, y, z.
        const MarkerSet& markers = model.getMarkerSet();
        struct MarkerCols { int x, y, z; };
        std::vector<MarkerCols> m_col(markers.getSize(), MarkerCols{-1, -1, -1});
        for (int k = 0; k < markers.getSize(); ++k) {
            const std::string& n = markers.get(k).getName();
            for (const auto& sfx : std::vector<std::vector<std::string>>{{"_tx", "_ty", "_tz"},
                                                                       {"_x", "_y", "_z"}}) {
                MarkerCols c{m_labels.findIndex(n + sfx[0]), m_labels.findIndex(n + sfx[1]),
                             m_labels.findIndex(n + sfx[2])};
                if (c.x >= 0 && c.y >= 0 && c.z >= 0) { m_col[k] = c; break; }
            }
            if (m_col[k].x < 0) std::cout << "note: marker " << n << " not in .sto, skipped\n";
        }

        const int frames = std::min(q_sto.getSize(), m_sto.getSize());
        std::vector<double> worst_mm(markers.getSize(), 0.0);
        double worst_time_gap = 0.0;

        for (int f = 0; f < frames; ++f) {
            const StateVector* qv = q_sto.getStateVector(f);
            const StateVector* mv = m_sto.getStateVector(f);
            worst_time_gap = std::max(worst_time_gap, std::abs(qv->getTime() - mv->getTime()));

            // 1. set q. Column labels include "time" at index 0; data arrays do not.
            const Array<double>& qd = qv->getData();
            for (int i = 0; i < coords.getSize(); ++i) {
                const Coordinate& c = coords.get(i);
                if (q_col[i] < 0 || c.getLocked(state)) continue;
                c.setValue(state, qd[q_col[i] - 1], /*enforceConstraints=*/false);
            }

            // 2. forward kinematics: compute every body's position and orientation.
            model.realizePosition(state);

            // 3. compare each marker with IK's reported position.
            const Array<double>& md = mv->getData();
            for (int k = 0; k < markers.getSize(); ++k) {
                if (m_col[k].x < 0) continue;
                const SimTK::Vec3 p = markers.get(k).getLocationInGround(state);
                const SimTK::Vec3 ref(md[m_col[k].x - 1], md[m_col[k].y - 1], md[m_col[k].z - 1]);
                worst_mm[k] = std::max(worst_mm[k], (p - ref).norm() * 1000.0);
            }
        }

        // Report: worst markers first.
        std::vector<int> order(markers.getSize());
        for (int k = 0; k < markers.getSize(); ++k) order[k] = k;
        std::sort(order.begin(), order.end(), [&](int a, int b) { return worst_mm[a] > worst_mm[b]; });

        std::cout << "frames compared: " << frames
                  << "   (largest time mismatch between files: " << worst_time_gap << " s)\n";
        std::cout << "worst markers (max distance, mm):\n";
        for (int i = 0; i < std::min<int>(5, order.size()); ++i)
            std::cout << "  " << markers.get(order[i]).getName() << "  " << worst_mm[order[i]] << "\n";

        const double overall = order.empty() ? 0.0 : worst_mm[order[0]];
        std::cout << (overall < 0.1 ? "PASS" : "CHECK") << ": max difference over all markers = "
                  << overall << " mm\n";
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
