#pragma once
#include "CameraGestures/Types.h"
#include <vector>
#include <optional>

// Mirror of iOS HandGestureRecognizing/MotionGate.swift.
// Pure state machine — no threading, no callbacks, no I/O.
// Feed HandShots in order, inspect the returned Event.

struct MotionGateConfig {
    float  t_open      = 1.0f;
    double k_open_ms   = 33.0;
    float  t_close     = 0.5f;
    double k_close_ms  = 200.0;
    double cooldown_ms = 1000.0;
};

enum class MotionGateState { closed, open };

class MotionGate {
public:
    struct Event {
        enum class Kind { still_closed, opened, still_open, cycle_ended };
        // Why a cycle ended (only meaningful for cycle_ended). When the energy
        // and the buffer cap both close the gate on the same frame, low_energy wins.
        enum class EndReason { none, absent_frame, low_energy, buffer_cap };
        Kind                   kind;
        int                    buffer_count = 0;
        std::vector<cg_handshot> cycle_buffer; // only set for cycle_ended
        EndReason              end_reason   = EndReason::none;
        // The energy this frame was compared against the thresholds. energy_valid
        // is false when there was no previous frame to diff against (energy 0 was
        // used) or the frame was absent.
        float                  energy       = 0.0f;
        bool                   energy_valid = false;
    };

    explicit MotionGate(const MotionGateConfig& cfg, int buffer_cap = 30);

    void reset();

    // Feed one frame; returns the event this frame produced.
    Event process(const cg_handshot& shot);

    MotionGateState state()        const { return state_; }
    int             bufferCount()  const { return static_cast<int>(gate_buffer_.size()); }

    // Static helpers (public so HoldDetector can reuse them).
    // Returns wrist-relative, scale-normalised 63-dim vector, or empty on degenerate input.
    static std::vector<float> normalize(const cg_handshot& shot);
    // Sum of per-landmark L2 distances between two 63-dim vectors.
    static float energy(const std::vector<float>& current,
                        const std::vector<float>& previous);

private:
    MotionGateConfig          config_;
    int                       buffer_cap_;
    MotionGateState           state_  = MotionGateState::closed;

    std::vector<float>        prev_coords_;
    std::optional<double>     above_threshold_since_;
    std::optional<double>     below_threshold_since_;
    std::vector<cg_handshot>  gate_buffer_;
};
