#include "MotionGate.hpp"
#include <cmath>
#include <algorithm>

MotionGate::MotionGate(const MotionGateConfig& cfg, int buffer_cap)
    : config_(cfg), buffer_cap_(buffer_cap) {}

void MotionGate::reset() {
    state_ = MotionGateState::closed;
    prev_coords_.clear();
    above_threshold_since_.reset();
    below_threshold_since_.reset();
    gate_buffer_.clear();
}

MotionGate::Event MotionGate::process(const cg_handshot& shot) {
    const double now = shot.timestamp;

    // Absent frame → immediate close (cycle end if was open).
    if (shot.is_absent) {
        bool was_open = (state_ == MotionGateState::open);
        std::vector<cg_handshot> captured;
        captured.swap(gate_buffer_);
        state_ = MotionGateState::closed;
        prev_coords_.clear();
        above_threshold_since_.reset();
        below_threshold_since_.reset();
        if (was_open) {
            return {Event::Kind::cycle_ended, 0, std::move(captured)};
        }
        return {Event::Kind::still_closed};
    }

    // Compute energy vs previous frame.
    auto curr_coords = normalize(shot);
    float e = 0.0f;
    if (!prev_coords_.empty() && !curr_coords.empty()) {
        e = energy(curr_coords, prev_coords_);
    }
    prev_coords_ = curr_coords;

    if (state_ == MotionGateState::closed) {
        if (e > config_.t_open) {
            if (!above_threshold_since_) above_threshold_since_ = now;
            double duration_ms = (now - *above_threshold_since_) * 1000.0;
            if (duration_ms >= config_.k_open_ms) {
                state_ = MotionGateState::open;
                above_threshold_since_.reset();
                below_threshold_since_.reset();
                gate_buffer_.clear();
                return {Event::Kind::opened};
            }
        } else {
            above_threshold_since_.reset();
        }
        return {Event::Kind::still_closed};
    }

    // state == open
    if (static_cast<int>(gate_buffer_.size()) < buffer_cap_) {
        gate_buffer_.push_back(shot);
    }

    bool should_close = false;
    if (e < config_.t_close) {
        if (!below_threshold_since_) below_threshold_since_ = now;
        double duration_ms = (now - *below_threshold_since_) * 1000.0;
        if (duration_ms >= config_.k_close_ms) should_close = true;
    } else {
        below_threshold_since_.reset();
    }
    if (static_cast<int>(gate_buffer_.size()) >= buffer_cap_) should_close = true;

    if (should_close) {
        std::vector<cg_handshot> captured;
        captured.swap(gate_buffer_);
        state_ = MotionGateState::closed;
        above_threshold_since_.reset();
        below_threshold_since_.reset();
        return {Event::Kind::cycle_ended, 0, std::move(captured)};
    }
    return {Event::Kind::still_open, static_cast<int>(gate_buffer_.size())};
}

std::vector<float> MotionGate::normalize(const cg_handshot& shot) {
    if (shot.is_absent) return {};
    const auto& lm = shot.landmarks;
    const float wx = lm[0].x, wy = lm[0].y, wz = lm[0].z;
    const float dx = lm[9].x - wx, dy = lm[9].y - wy, dz = lm[9].z - wz;
    const float scale = std::sqrt(dx*dx + dy*dy + dz*dz);
    if (scale < 1e-6f) return {};

    std::vector<float> coords(63);
    for (int i = 0; i < 21; ++i) {
        coords[i*3]   = (lm[i].x - wx) / scale;
        coords[i*3+1] = (lm[i].y - wy) / scale;
        coords[i*3+2] = (lm[i].z - wz) / scale;
    }
    return coords;
}

float MotionGate::energy(const std::vector<float>& cur, const std::vector<float>& prev) {
    float e = 0.0f;
    for (int i = 0; i < 21; ++i) {
        float ddx = cur[i*3]   - prev[i*3];
        float ddy = cur[i*3+1] - prev[i*3+1];
        float ddz = cur[i*3+2] - prev[i*3+2];
        e += std::sqrt(ddx*ddx + ddy*ddy + ddz*ddz);
    }
    return e;
}
