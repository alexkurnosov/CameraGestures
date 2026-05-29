#pragma once
#include "MotionGate.hpp"
#include "HoldDetector.hpp"
#include "PrefixMatcher.hpp"
#include "GestureModel.hpp"
#include "CameraGestures/Types.h"
#include <functional>
#include <optional>
#include <memory>
#include <set>
#include <vector>
#include <string>

// Mirror of iOS HandGestureRecognizing/HandGestureRecognizing.swift.
//
// Threading model: this class is NOT thread-safe. The caller (Swift binding)
// must serialise all calls on a dedicated serial queue — exactly as the iOS
// V1 pod did with handshotQueue.

// --------------------------------------------------------------------------
// Configuration
// --------------------------------------------------------------------------

struct HoldsConfig {
    float  t_hold             = 2.10f;
    double k_hold_ms          = 100.0;
    double smooth_k_ms        = 100.0;
    double t_commit_ms        = 300.0;
    double t_min_buffer_ms    = 200.0;
    float  tau_pose_confidence = 0.6f;
    float  tau_phase3_confidence = 0.7f;
};

struct HandGestureRecognizingConfig {
    // Phase 1
    bool             gate_enabled       = false;
    MotionGateConfig motion_gate;
    int              gesture_buffer_size = 30;

    // Phase 2 (optional — when present, holds-mode is active)
    std::optional<HoldsConfig> holds;

    // Phase 3
    float confidence_threshold = 0.7f;

    // Review support: when true, rep_shot + normalized_coords are populated in
    // the holds-telemetry callback (training app only; production apps pay no cost).
    bool retain_landmarks_for_review = false;
};

// --------------------------------------------------------------------------
// Callbacks (all invoked synchronously, i.e. on the caller's thread)
// --------------------------------------------------------------------------

struct DetectedGestureInfo {
    cg_gesture_prediction prediction;
    cg_handfilm_ref       film;          // caller must NOT destroy; valid only during callback
    int                   candidate_set_size = -1; // -1 = unrestricted
};

using GestureCallback      = std::function<void(const DetectedGestureInfo&)>;
using GateUpdateCallback   = std::function<void(MotionGateState, int buffer_count)>;
using HoldsTelemetryCallback = std::function<void(int pose_id, float confidence,
                                                   const std::vector<int>& observed_seq,
                                                   const std::string& matched_gesture,
                                                   const cg_handshot* rep_shot,
                                                   const std::vector<float>& normalized_coords)>;

// --------------------------------------------------------------------------
// Pending commit state (used for T_commit / T_min_buffer timers)
// --------------------------------------------------------------------------
struct PendingCommit {
    std::set<std::string> candidate_set;
    double                commit_deadline;   // absolute timestamp (epoch seconds)
    double                min_buffer_deadline; // 0 if already satisfied
};

// --------------------------------------------------------------------------
// HandGestureRecognizing
// --------------------------------------------------------------------------

class HandGestureRecognizing {
public:
    // Callbacks — set before first processShot() call.
    GestureCallback         on_gesture;
    GateUpdateCallback      on_gate_update;
    HoldsTelemetryCallback  on_holds_telemetry;

    // bypassPhase2Filter: always run Phase 3 unrestricted even in holds mode.
    bool bypass_phase2 = false;

    explicit HandGestureRecognizing(const HandGestureRecognizingConfig& cfg);
    ~HandGestureRecognizing();

    // Attach GestureModel (must outlive this object).
    void setGestureModel(GestureModel* model);

    // Enable / disable the Phase-1 gate at runtime.
    void setGateEnabled(bool enabled);

    // Main per-frame entry point — call from the camera capture callback.
    // Fires on_gesture synchronously if a gesture is detected.
    void processShot(const cg_handshot& shot);

    // Tick T_commit / T_min_buffer timers. Must be called by the platform
    // layer at regular intervals (e.g. every ~10 ms on the serial queue).
    // Returns true if a commit was fired.
    bool tickTimers(double now);

    // Reset all gate / Phase-2 state.
    void resetGateState();

    // Accumulated in-progress handfilm (used by the Swift binding to expose
    // harvestHandfilm / resetHandfilm equivalents).
    cg_handfilm_ref borrowCurrentFilm(); // caller does NOT own; ref valid until next processShot()

    const HandGestureRecognizingConfig& config() const { return config_; }

private:
    HandGestureRecognizingConfig config_;
    GestureModel*                model_      = nullptr;

    std::unique_ptr<MotionGate>    gate_;
    std::unique_ptr<HoldDetector>  hold_detector_;
    std::unique_ptr<PrefixMatcher> prefix_matcher_;

    // Current-cycle state
    std::vector<cg_handshot> cycle_buffer_;
    double                   gate_open_time_       = 0.0;
    bool                     already_committed_    = false;

    // Pending commit timer state (non-null when T_commit / T_min_buffer is pending)
    std::optional<PendingCommit> pending_commit_;

    // Current film handle (rebuilt each cycle from the gate buffer)
    cg_handfilm_ref current_film_ = nullptr;

    void handleShotWithGate(const cg_handshot& shot);
    void handleCycleEnd(std::vector<cg_handshot> buffer);
    void handlePhase2Hold(const cg_handshot& rep_shot, double start_t, double end_t);
    void scheduleCommitOrDefer(std::set<std::string> candidate_set, double now);
    void triggerPhase3Commit(const std::set<std::string>& candidate_set);
    void recognizeRestricted(cg_handfilm_ref film, const std::set<std::string>& candidates);
    void recognizeUnrestricted(cg_handfilm_ref film);
    void emitGesture(cg_handfilm_ref film, const cg_gesture_prediction& pred, int candidate_set_size);
    void reportGateUpdate();
    cg_handfilm_ref makeFilm(const std::vector<cg_handshot>& shots);
    void cancelPendingCommit();
};
