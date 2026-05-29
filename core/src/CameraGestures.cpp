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
