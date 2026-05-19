#pragma once
#include "MotionGate.hpp"
#include <vector>
#include <optional>
#include <limits>

// Mirror of iOS HandGestureRecognizing/HoldDetector.swift.
// Pure Phase-2 hold-detection state machine — no threading, no I/O.
// Feed HandShots (only while the Phase-1 gate is open); inspect the returned Event.

struct cg_handshot;

struct HoldDetectorConfig {
    float  t_hold      = 2.10f;
    double k_hold_ms   = 100.0;
    double smooth_k_ms = 100.0;
};

class HoldDetector {
public:
    struct Event {
        bool      hold_detected = false;
        cg_handshot rep_shot    = {};
        double    start_time    = 0.0;
        double    end_time      = 0.0;
    };

    explicit HoldDetector(const HoldDetectorConfig& cfg = {});

    void  reset();
    Event process(const cg_handshot& shot);

    float lastSmoothedEnergy() const;

private:
    HoldDetectorConfig config_;

    struct Frame {
        cg_handshot shot;
        float       raw_energy;
    };
    std::vector<Frame>  history_;
    std::vector<float>  prev_coords_;

    bool   in_hold_          = false;
    int    hold_start_idx_   = 0;
    int    hold_argmin_idx_  = 0;
    float  hold_argmin_e_    = std::numeric_limits<float>::infinity();
    bool   hold_emitted_     = false;

    float smoothedEnergy(int index) const;
    std::optional<Event> finishCurrentHold();
};
