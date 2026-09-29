#include "CameraGestures/CameraGestures.h"
#include "CameraGestures/HandGestureRecognizing.h"
#include "hand_gesture_recognizing/HandGestureRecognizing.hpp"
#include "gesture_model/GestureModel.hpp"
#include <cstring>
#include <vector>

extern "C" {

const char* cg_version(void) {
    return "0.1.0";
}

// ---------------------------------------------------------------------------
// cg_recognizer_default_config
// ---------------------------------------------------------------------------

cg_recognizer_config cg_recognizer_default_config(void) {
    cg_recognizer_config c{};
    c.gate_enabled        = 0;
    c.gesture_buffer_size = 30;
    c.holds_enabled       = 0;
    c.confidence_threshold = 0.7f;
    c.retain_landmarks_for_review = 0;

    c.motion_gate.t_open      = 1.0f;
    c.motion_gate.k_open_ms   = 33.0;
    c.motion_gate.t_close     = 0.5f;
    c.motion_gate.k_close_ms  = 200.0;
    c.motion_gate.cooldown_ms = 1000.0;

    c.holds.t_hold              = 2.10f;
    c.holds.k_hold_ms           = 100.0;
    c.holds.smooth_k_ms         = 100.0;
    c.holds.t_commit_ms         = 300.0;
    c.holds.t_min_buffer_ms     = 200.0;
    c.holds.tau_pose_confidence  = 0.6f;
    c.holds.tau_phase3_confidence = 0.7f;
    return c;
}

// ---------------------------------------------------------------------------
// Internal wrapper around the C++ class
// ---------------------------------------------------------------------------

struct cg_recognizer_s {
    HandGestureRecognizing   recognizer;
    GestureModel*            model_ptr = nullptr; // non-owning

    // Callback state
    cg_gesture_callback      gesture_cb      = nullptr;
    void*                    gesture_ctx     = nullptr;

    cg_gate_update_callback  gate_cb         = nullptr;
    void*                    gate_ctx        = nullptr;

    cg_holds_telemetry_callback holds_cb     = nullptr;
    void*                       holds_ctx    = nullptr;

    cg_frame_telemetry_callback frame_cb     = nullptr;
    void*                       frame_ctx    = nullptr;

    cg_decision_event_callback  event_cb     = nullptr;
    void*                       event_ctx    = nullptr;

    explicit cg_recognizer_s(const HandGestureRecognizingConfig& cfg)
        : recognizer(cfg) {
        wireCppCallbacks();
    }

    void wireCppCallbacks() {
        recognizer.on_gesture = [this](const DetectedGestureInfo& info) {
            if (gesture_cb) {
                gesture_cb(gesture_ctx, &info.prediction, info.film,
                           info.candidate_set_size);
            }
        };
        recognizer.on_gate_update = [this](MotionGateState state, int count) {
            if (gate_cb) {
                gate_cb(gate_ctx, state == MotionGateState::open ? 1 : 0, count);
            }
        };
        recognizer.on_holds_telemetry = [this](int pose_id, float conf,
                                               const std::vector<int>& seq,
                                               const std::string& matched,
                                               const cg_handshot* rep_shot,
                                               const std::vector<float>& norm_coords) {
            if (holds_cb) {
                holds_cb(holds_ctx, pose_id, conf,
                         seq.data(), static_cast<int>(seq.size()),
                         matched.c_str(),
                         rep_shot,
                         norm_coords.empty() ? nullptr : norm_coords.data(),
                         static_cast<int>(norm_coords.size()));
            }
        };
    }

    // The telemetry hooks are installed only while a C callback is set, so the
    // recognizer skips building rows and events nobody reads.
    void setFrameCallback(cg_frame_telemetry_callback cb, void* ctx) {
        frame_cb  = cb;
        frame_ctx = ctx;
        if (!cb) { recognizer.on_frame_telemetry = nullptr; return; }
        recognizer.on_frame_telemetry = [this](const cg_frame_telemetry& row) {
            frame_cb(frame_ctx, &row);
        };
    }

    void setEventCallback(cg_decision_event_callback cb, void* ctx) {
        event_cb  = cb;
        event_ctx = ctx;
        if (!cb) { recognizer.on_decision_event = nullptr; return; }
        recognizer.on_decision_event = [this](const cg_decision_event& ev) {
            event_cb(event_ctx, &ev);
        };
    }
};

static HandGestureRecognizingConfig toCppConfig(const cg_recognizer_config* c) {
    HandGestureRecognizingConfig cfg;
    cfg.gate_enabled = c->gate_enabled != 0;
    cfg.gesture_buffer_size = c->gesture_buffer_size;
    cfg.confidence_threshold = c->confidence_threshold;

    cfg.motion_gate.t_open     = c->motion_gate.t_open;
    cfg.motion_gate.k_open_ms  = c->motion_gate.k_open_ms;
    cfg.motion_gate.t_close    = c->motion_gate.t_close;
    cfg.motion_gate.k_close_ms = c->motion_gate.k_close_ms;
    cfg.motion_gate.cooldown_ms = c->motion_gate.cooldown_ms;

    if (c->holds_enabled) {
        HoldsConfig h;
        h.t_hold              = c->holds.t_hold;
        h.k_hold_ms           = c->holds.k_hold_ms;
        h.smooth_k_ms         = c->holds.smooth_k_ms;
        h.t_commit_ms         = c->holds.t_commit_ms;
        h.t_min_buffer_ms     = c->holds.t_min_buffer_ms;
        h.tau_pose_confidence  = c->holds.tau_pose_confidence;
        h.tau_phase3_confidence = c->holds.tau_phase3_confidence;
        cfg.holds = h;
    }
    cfg.retain_landmarks_for_review = c->retain_landmarks_for_review != 0;
    return cfg;
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

cg_recognizer_ref cg_recognizer_create(const cg_recognizer_config* config,
                                        cg_gesture_model_ref gesture_model) {
    if (!config) return nullptr;
    auto* obj = new(std::nothrow) cg_recognizer_s(toCppConfig(config));
    if (!obj) return nullptr;

    if (gesture_model) {
        obj->model_ptr = reinterpret_cast<GestureModel*>(gesture_model);
        obj->recognizer.setGestureModel(obj->model_ptr);
    }
    return obj;
}

void cg_recognizer_destroy(cg_recognizer_ref recognizer) {
    delete recognizer;
}

// ---------------------------------------------------------------------------
// Callbacks
// ---------------------------------------------------------------------------

void cg_recognizer_set_gesture_callback(cg_recognizer_ref recognizer,
                                         cg_gesture_callback callback,
                                         void* context) {
    if (!recognizer) return;
    recognizer->gesture_cb  = callback;
    recognizer->gesture_ctx = context;
}

void cg_recognizer_set_gate_update_callback(cg_recognizer_ref recognizer,
                                             cg_gate_update_callback callback,
                                             void* context) {
    if (!recognizer) return;
    recognizer->gate_cb  = callback;
    recognizer->gate_ctx = context;
}

void cg_recognizer_set_holds_telemetry_callback(cg_recognizer_ref recognizer,
                                                 cg_holds_telemetry_callback callback,
                                                 void* context) {
    if (!recognizer) return;
    recognizer->holds_cb  = callback;
    recognizer->holds_ctx = context;
}

void cg_recognizer_set_frame_telemetry_callback(cg_recognizer_ref recognizer,
                                                 cg_frame_telemetry_callback callback,
                                                 void* context) {
    if (!recognizer) return;
    recognizer->setFrameCallback(callback, context);
}

void cg_recognizer_set_decision_event_callback(cg_recognizer_ref recognizer,
                                                cg_decision_event_callback callback,
                                                void* context) {
    if (!recognizer) return;
    recognizer->setEventCallback(callback, context);
}

// ---------------------------------------------------------------------------
// Telemetry names
// ---------------------------------------------------------------------------

const char* cg_decision_event_kind_name(int kind) {
    switch (kind) {
    case CG_EVENT_GATE_OPENED:       return "gate_opened";
    case CG_EVENT_CYCLE_ENDED:       return "cycle_ended";
    case CG_EVENT_CYCLE_SKIPPED:     return "cycle_skipped";
    case CG_EVENT_HOLD_COMPLETED:    return "hold_completed";
    case CG_EVENT_PHASE3_PREDICTION: return "phase3_prediction";
    case CG_EVENT_COMMIT_FIRED:      return "commit_fired";
    }
    return "unknown";
}

const char* cg_decision_event_reason_name(int kind, int reason) {
    switch (kind) {
    case CG_EVENT_CYCLE_ENDED:
        switch (reason) {
        case CG_CYCLE_END_ABSENT_FRAME:   return "absent_frame";
        case CG_CYCLE_END_LOW_ENERGY:     return "low_energy";
        case CG_CYCLE_END_BUFFER_CAP:     return "buffer_cap";
        case CG_CYCLE_END_PHASE2_DISCARD: return "phase2_discard";
        case CG_CYCLE_END_COMMITTED:      return "committed";
        case CG_CYCLE_END_EXTERNAL_RESET: return "external_reset";
        }
        break;
    case CG_EVENT_CYCLE_SKIPPED:
        switch (reason) {
        case CG_CYCLE_SKIP_ALREADY_COMMITTED: return "already_committed";
        case CG_CYCLE_SKIP_EMPTY_BUFFER:      return "empty_buffer";
        case CG_CYCLE_SKIP_NO_MODEL:          return "no_model";
        case CG_CYCLE_SKIP_TOO_FEW_FRAMES:    return "too_few_frames";
        }
        break;
    case CG_EVENT_HOLD_COMPLETED:
        switch (reason) {
        case CG_PREFIX_NOT_OBSERVED:       return "not_observed";
        case CG_PREFIX_NO_PREFIX:          return "no_prefix";
        case CG_PREFIX_LIVE_PREFIX:        return "live_prefix";
        case CG_PREFIX_COMMIT_NOW:         return "commit_now";
        case CG_PREFIX_START_COMMIT_TIMER: return "start_commit_timer";
        case CG_PREFIX_IDLE_RESET:         return "idle_reset";
        case CG_PREFIX_IDLE_DISCARD:       return "idle_discard";
        case CG_PREFIX_IDLE_COMMIT:        return "idle_commit";
        }
        break;
    case CG_EVENT_COMMIT_FIRED:
        switch (reason) {
        case CG_COMMIT_IMMEDIATE:    return "immediate";
        case CG_COMMIT_T_COMMIT:     return "t_commit";
        case CG_COMMIT_T_MIN_BUFFER: return "t_min_buffer";
        }
        break;
    case CG_EVENT_GATE_OPENED:
    case CG_EVENT_PHASE3_PREDICTION:
        return "";
    }
    return "unknown";
}

// ---------------------------------------------------------------------------
// Runtime
// ---------------------------------------------------------------------------

void cg_recognizer_process_shot(cg_recognizer_ref recognizer,
                                 const cg_handshot* shot) {
    if (!recognizer || !shot) return;
    recognizer->recognizer.processShot(*shot);
}

int cg_recognizer_tick_timers(cg_recognizer_ref recognizer, double now) {
    if (!recognizer) return 0;
    return recognizer->recognizer.tickTimers(now) ? 1 : 0;
}

void cg_recognizer_reset_gate(cg_recognizer_ref recognizer) {
    if (!recognizer) return;
    recognizer->recognizer.resetGateState();
}

void cg_recognizer_set_gate_enabled(cg_recognizer_ref recognizer, int enabled) {
    if (!recognizer) return;
    recognizer->recognizer.setGateEnabled(enabled != 0);
}

void cg_recognizer_set_bypass_phase2(cg_recognizer_ref recognizer, int bypass) {
    if (!recognizer) return;
    recognizer->recognizer.bypass_phase2 = (bypass != 0);
}

void cg_recognizer_set_gesture_model(cg_recognizer_ref recognizer,
                                      cg_gesture_model_ref model) {
    if (!recognizer) return;
    auto* m = reinterpret_cast<GestureModel*>(model);
    recognizer->model_ptr = m;
    recognizer->recognizer.setGestureModel(m);
}

} // extern "C"
