
#include "opensim_fk.h"

#include <iostream>
#include <set>

using namespace OpenSim;

namespace {
bool starts_with_any(const std::string& name, const std::vector<std::string>& prefixes) {
    for (const auto& p : prefixes)
        if (name.compare(0, p.size(), p) == 0) return true;
    return false;
}
}  // namespace

OpenSimFK::OpenSimFK(const std::string& model_path, const std::vector<std::string>& wanted_markers,
                     const std::vector<std::string>& excluded_body_prefixes)
    : model_(std::make_unique<Model>(model_path)) {
    state_ = &model_->initSystem();

    // Coupler constraints: remember which coordinate each one drives.
    std::set<std::string> dependent_names;
    std::vector<const CoordinateCouplerConstraint*> coupler_list;
    const ConstraintSet& cset = model_->getConstraintSet();
    for (int i = 0; i < cset.getSize(); ++i) {
        if (const auto* cc = dynamic_cast<const CoordinateCouplerConstraint*>(&cset.get(i))) {
            coupler_list.push_back(cc);
            dependent_names.insert(cc->getDependentCoordinateName());
        }
    }

    // Free coordinates: not locked, not driven by a coupler, not moving an excluded body.
    const CoordinateSet& cs = model_->getCoordinateSet();
    int n_excluded = 0;
    for (int i = 0; i < cs.getSize(); ++i) {
        const Coordinate& c = cs.get(i);
        if (c.getLocked(*state_) || dependent_names.count(c.getName())) continue;
        const std::string child_body = c.getJoint().getChildFrame().findBaseFrame().getName();
        if (starts_with_any(child_body, excluded_body_prefixes)) { ++n_excluded; continue; }
        free_.push_back(&c);
        coord_names_.push_back(c.getName());
        rotational_.push_back(c.getMotionType() == Coordinate::Rotational);
        clamped_.push_back(c.getClamped(*state_));
        range_min_.push_back(c.getRangeMin());
        range_max_.push_back(c.getRangeMax());
    }

    for (const auto* cc : coupler_list) {
        Coupler k;
        k.dependent = &cs.get(cc->getDependentCoordinateName());
        k.function = &cc->getFunction();
        const Array<std::string>& names = cc->getIndependentCoordinateNames();
        for (int i = 0; i < names.getSize(); ++i) {
            k.independent.push_back(&cs.get(names[i]));
            int idx = -1;
            for (size_t f = 0; f < coord_names_.size(); ++f)
                if (coord_names_[f] == names[i]) idx = static_cast<int>(f);
            k.independent_free_index.push_back(idx);
        }
        couplers_.push_back(k);
    }

    // Markers, in the requested order.
    const MarkerSet& ms = model_->getMarkerSet();
    std::vector<std::string> names = wanted_markers;
    if (names.empty())
        for (int i = 0; i < ms.getSize(); ++i) names.push_back(ms.get(i).getName());
    for (const auto& name : names) {
        if (ms.contains(name) &&
            starts_with_any(ms.get(name).getParentFrame().findBaseFrame().getName(), excluded_body_prefixes)) {
            std::cout << "note: marker " << name << " is on an excluded body, skipped\n";
        } else if (ms.contains(name)) {
            markers_.push_back(&ms.get(name));
            marker_names_.push_back(name);
            marker_bodies_.push_back(ms.get(name).getParentFrame().findBaseFrame().getName());
        } else {
            std::cout << "note: marker " << name << " is not in the model, skipped\n";
        }
    }

    std::cout << "OpenSimFK: " << coord_names_.size() << " free coordinates, "
              << couplers_.size() << " coupled, " << n_excluded << " held fixed (excluded bodies), "
              << marker_names_.size() << " markers\n";
}

void OpenSimFK::marker_positions(const std::vector<double>& q, std::vector<double>& out_m) {
    // 1. free coordinates
    for (size_t i = 0; i < free_.size(); ++i) free_[i]->setValue(*state_, q[i], /*enforceConstraints=*/false);

    // 2. coupled coordinates follow their independent ones
    for (const auto& k : couplers_) {
        SimTK::Vector x(static_cast<int>(k.independent.size()));
        for (size_t i = 0; i < k.independent.size(); ++i)
            x[static_cast<int>(i)] = k.independent_free_index[i] >= 0 ? q[k.independent_free_index[i]]
                                                                       : k.independent[i]->getValue(*state_);
        k.dependent->setValue(*state_, k.function->calcValue(x), false);
    }

    // 3. forward kinematics, then each marker's lab position
    model_->realizePosition(*state_);
    out_m.resize(3 * markers_.size());
    for (size_t m = 0; m < markers_.size(); ++m) {
        const SimTK::Vec3 p = markers_[m]->getLocationInGround(*state_);
        out_m[3 * m + 0] = p[0];
        out_m[3 * m + 1] = p[1];
        out_m[3 * m + 2] = p[2];
    }
}
