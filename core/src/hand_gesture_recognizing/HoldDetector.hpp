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

        // Per-frame state after this shot, for telemetry. energy_valid is false
        // only for absent frames, which reset the detector.
        bool      energy_valid    = false;
        float     smoothed_energy = 0.0f;
        int       hold_run_frames = 0;    // frames in the current below-T_hold run; 0 = none
        double    hold_run_ms     = 0.0;  // duration of that run, as compared to k_hold_ms
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
    // The hold state machine for the frame just appended at cur_idx.
    Event decide(const cg_handshot& shot, int cur_idx, float smoothed);
    std::optional<Event> finishCurrentHold();
};
