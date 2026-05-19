#include "HoldDetector.hpp"
#include <algorithm>

HoldDetector::HoldDetector(const HoldDetectorConfig& cfg) : config_(cfg) {}

void HoldDetector::reset() {
    history_.clear();
    prev_coords_.clear();
    in_hold_         = false;
    hold_start_idx_  = 0;
    hold_argmin_idx_ = 0;
    hold_argmin_e_   = std::numeric_limits<float>::infinity();
    hold_emitted_    = false;
}

HoldDetector::Event HoldDetector::process(const cg_handshot& shot) {
    if (shot.is_absent) {
        auto ev = finishCurrentHold();
        reset();
        if (ev) return *ev;
        return {};
    }

    auto curr_coords = MotionGate::normalize(shot);
    float raw_e = 0.0f;
    if (!prev_coords_.empty() && !curr_coords.empty()) {
        raw_e = MotionGate::energy(curr_coords, prev_coords_);
    }
    prev_coords_ = curr_coords;
    Frame fr;
    fr.shot       = shot;
    fr.raw_energy = raw_e;
    history_.push_back(fr);

    // Prune history older than 5 s to bound memory.
    const double cutoff = shot.timestamp - 5.0;
    int prune = 0;
    while (prune < static_cast<int>(history_.size()) &&
           history_[prune].shot.timestamp < cutoff) ++prune;
    if (prune > 0) {
        history_.erase(history_.begin(), history_.begin() + prune);
        if (in_hold_) {
            hold_start_idx_  = std::max(0, hold_start_idx_  - prune);
            hold_argmin_idx_ = std::max(0, hold_argmin_idx_ - prune);
        }
    }

    int cur_idx = static_cast<int>(history_.size()) - 1;
    float smoothed = smoothedEnergy(cur_idx);

    if (smoothed < config_.t_hold) {
        if (!in_hold_) {
            in_hold_         = true;
            hold_emitted_    = false;
            hold_start_idx_  = cur_idx;
            hold_argmin_idx_ = cur_idx;
            hold_argmin_e_   = smoothed;
        } else if (smoothed < hold_argmin_e_) {
            hold_argmin_e_   = smoothed;
            hold_argmin_idx_ = cur_idx;
        }

        if (!hold_emitted_) {
            double start_t    = history_[hold_start_idx_].shot.timestamp;
            double duration_ms = (shot.timestamp - start_t) * 1000.0;
            if (duration_ms >= config_.k_hold_ms) {
                hold_emitted_ = true;
                Event ev;
                ev.hold_detected = true;
                ev.rep_shot      = history_[hold_argmin_idx_].shot;
                ev.start_time    = start_t;
                ev.end_time      = shot.timestamp;
                return ev;
            }
        }
    } else {
        if (in_hold_) {
            auto ev = finishCurrentHold();
            in_hold_       = false;
            hold_argmin_e_ = std::numeric_limits<float>::infinity();
            hold_emitted_  = false;
            if (ev) return *ev;
        }
    }
    return {};
}

float HoldDetector::lastSmoothedEnergy() const {
    if (history_.empty()) return 0.0f;
    return smoothedEnergy(static_cast<int>(history_.size()) - 1);
}

std::optional<HoldDetector::Event> HoldDetector::finishCurrentHold() {
    if (!in_hold_ || hold_emitted_) return std::nullopt;

    double start_t    = history_[hold_start_idx_].shot.timestamp;
    double end_t      = history_.back().shot.timestamp;
    double duration_ms = (end_t - start_t) * 1000.0;
    if (duration_ms < config_.k_hold_ms) return std::nullopt;

    Event ev;
    ev.hold_detected = true;
    ev.rep_shot      = history_[hold_argmin_idx_].shot;
    ev.start_time    = start_t;
    ev.end_time      = end_t;
    return ev;
}

float HoldDetector::smoothedEnergy(int index) const {
    double target_t   = history_[index].shot.timestamp;
    double window_start = target_t - config_.smooth_k_ms / 1000.0;
    float sum = 0.0f;
    int   cnt = 0;
    for (int i = index; i >= 0; --i) {
        if (history_[i].shot.timestamp < window_start) break;
        sum += history_[i].raw_energy;
        ++cnt;
    }
    return cnt > 0 ? sum / static_cast<float>(cnt) : 0.0f;
}
