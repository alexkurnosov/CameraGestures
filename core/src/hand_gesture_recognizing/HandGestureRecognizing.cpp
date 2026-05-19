#include "HandGestureRecognizing.hpp"
#include "CameraGestures/Types.h"
#include <algorithm>
#include <cassert>

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

HandGestureRecognizing::HandGestureRecognizing(const HandGestureRecognizingConfig& cfg)
    : config_(cfg) {
    gate_ = std::make_unique<MotionGate>(cfg.motion_gate, cfg.gesture_buffer_size);

    if (cfg.holds.has_value()) {
        const auto& h = *cfg.holds;
        hold_detector_ = std::make_unique<HoldDetector>(
            HoldDetectorConfig{h.t_hold, h.k_hold_ms, h.smooth_k_ms});
        // PrefixMatcher is created once setGestureModel() is called and
        // model.poseManifest() is available.
    }
}

HandGestureRecognizing::~HandGestureRecognizing() {
    if (current_film_) cg_handfilm_destroy(current_film_);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void HandGestureRecognizing::setGestureModel(GestureModel* model) {
    model_ = model;
    // If holds-mode is configured and the model has a pose manifest, create
    // the PrefixMatcher now.
    if (config_.holds.has_value() && model_ && model_->isPoseLoaded()) {
        const CgPoseManifest* manifest = model_->poseManifest();
        if (manifest) {
            prefix_matcher_ = std::make_unique<PrefixMatcher>(*manifest);
        }
    }
}

void HandGestureRecognizing::setGateEnabled(bool enabled) {
    config_.gate_enabled = enabled;
    if (!enabled) resetGateState();
}

void HandGestureRecognizing::processShot(const cg_handshot& shot) {
    if (config_.gate_enabled) {
        handleShotWithGate(shot);
    } else {
        // Legacy per-film mode: no gate, recognition fires from handfilm stream.
        // (Training App v2 always uses gate-enabled; this path exists for parity.)
    }
}

bool HandGestureRecognizing::tickTimers(double now) {
    if (!pending_commit_) return false;
    if (already_committed_) { cancelPendingCommit(); return false; }

    auto& pc = *pending_commit_;

    // Wait for min-buffer first.
    if (pc.min_buffer_deadline > 0.0 && now < pc.min_buffer_deadline) return false;
    // Then wait for T_commit.
    if (now < pc.commit_deadline) return false;

    // Both deadlines satisfied — fire.
    auto candidate_set = pc.candidate_set;
    cancelPendingCommit();
    triggerPhase3Commit(candidate_set);
    return true;
}

void HandGestureRecognizing::resetGateState() {
    gate_->reset();
    if (hold_detector_) hold_detector_->reset();
    if (prefix_matcher_) prefix_matcher_->reset();
    cycle_buffer_.clear();
    already_committed_ = false;
    cancelPendingCommit();
    reportGateUpdate();
}

cg_handfilm_ref HandGestureRecognizing::borrowCurrentFilm() {
    return current_film_;
}

// ---------------------------------------------------------------------------
// Private — gate logic
// ---------------------------------------------------------------------------

void HandGestureRecognizing::handleShotWithGate(const cg_handshot& shot) {
    auto event = gate_->process(shot);

    switch (event.kind) {
    case MotionGate::Event::Kind::still_closed:
        break;

    case MotionGate::Event::Kind::opened:
        cycle_buffer_.clear();
        gate_open_time_    = shot.timestamp;
        already_committed_ = false;
        cancelPendingCommit();
        if (hold_detector_)  hold_detector_->reset();
        if (prefix_matcher_) prefix_matcher_->reset();
        break;

    case MotionGate::Event::Kind::still_open:
        cycle_buffer_.push_back(shot);
        // Phase-2 hold detection.
        if (hold_detector_ && model_ && model_->isPoseLoaded()) {
            auto hold_event = hold_detector_->process(shot);
            if (hold_event.hold_detected) {
                handlePhase2Hold(hold_event.rep_shot,
                                 hold_event.start_time, hold_event.end_time);
            }
        }
        break;

    case MotionGate::Event::Kind::cycle_ended:
        cancelPendingCommit();
        handleCycleEnd(std::move(event.cycle_buffer));
        break;
    }

    reportGateUpdate();
}

void HandGestureRecognizing::handleCycleEnd(std::vector<cg_handshot> buffer) {
    // Holds mode: if Phase 2 already committed this cycle, skip Phase 3.
    if (already_committed_) {
        already_committed_ = false;
        cycle_buffer_.clear();
        if (prefix_matcher_) prefix_matcher_->reset();
        return;
    }

    if (buffer.empty() || !model_ || !model_->isLoaded()) return;

    cg_handfilm_ref film = makeFilm(buffer);

    if (config_.gate_enabled && config_.holds.has_value()) {
        // Gate-close commit path.
        std::optional<std::set<std::string>> candidate_set;
        if (prefix_matcher_) candidate_set = prefix_matcher_->gateCloseCommitSet();
        if (prefix_matcher_) prefix_matcher_->reset();
        cycle_buffer_.clear();

        if (candidate_set && !bypass_phase2) {
            recognizeRestricted(film, *candidate_set);
        } else {
            recognizeUnrestricted(film);
        }
    } else {
        recognizeUnrestricted(film);
    }

    cg_handfilm_destroy(film);
}

// ---------------------------------------------------------------------------
// Private — Phase-2 hold handler
// ---------------------------------------------------------------------------

void HandGestureRecognizing::handlePhase2Hold(
        const cg_handshot& rep_shot, double /*start_t*/, double /*end_t*/) {
    if (!config_.holds.has_value()) return;
    const auto& holds_cfg = *config_.holds;

    if (!model_ || !prefix_matcher_) return;

    // Predict pose cluster.
    int   pose_id    = 0;
    float confidence = 0.0f;
    int ok = model_->predictPose(rep_shot, &pose_id, &confidence);
    if (!ok) return;

    // Reject below τ_pose_confidence.
    if (confidence < holds_cfg.tau_pose_confidence) {
        // Report telemetry but don't advance the sequence.
        if (on_holds_telemetry) {
            on_holds_telemetry(pose_id, confidence,
                               prefix_matcher_->observedSequence(), "");
        }
        return;
    }

    CgClusterKind kind = model_->poseManifest()
        ? model_->poseManifest()->clusterKind(pose_id)
        : CgClusterKind::unconfirmed;

    auto action = prefix_matcher_->observe(pose_id, kind);

    // Telemetry
    if (on_holds_telemetry) {
        std::string matched;
        if (auto cs = prefix_matcher_->gateCloseCommitSet()) {
            if (!cs->empty()) matched = *cs->begin();
        }
        on_holds_telemetry(pose_id, confidence,
                           prefix_matcher_->observedSequence(), matched);
    }

    double now = rep_shot.timestamp;

    switch (action.kind) {
    case PrefixMatcher::Action::Kind::no_prefix:
    case PrefixMatcher::Action::Kind::idle_discard:
        cancelPendingCommit();
        prefix_matcher_->reset();
        cycle_buffer_.clear();
        gate_->reset();
        break;

    case PrefixMatcher::Action::Kind::live_prefix:
        cancelPendingCommit();
        break;

    case PrefixMatcher::Action::Kind::commit_now:
        cancelPendingCommit();
        scheduleCommitOrDefer(action.candidate_set, now);
        break;

    case PrefixMatcher::Action::Kind::start_commit_timer: {
        cancelPendingCommit();
        double commit_dl = now + holds_cfg.t_commit_ms / 1000.0;
        // min-buffer check
        double elapsed_ms = (now - gate_open_time_) * 1000.0;
        double min_buf_dl = 0.0;
        if (elapsed_ms < holds_cfg.t_min_buffer_ms) {
            min_buf_dl = gate_open_time_ + holds_cfg.t_min_buffer_ms / 1000.0;
        }
        pending_commit_ = PendingCommit{action.candidate_set, commit_dl, min_buf_dl};
        break;
    }

    case PrefixMatcher::Action::Kind::idle_reset:
        cancelPendingCommit();
        prefix_matcher_->reset();
        cycle_buffer_.clear();
        gate_->reset();
        break;

    case PrefixMatcher::Action::Kind::idle_commit:
        cancelPendingCommit();
        scheduleCommitOrDefer(action.candidate_set, now);
        break;
    }
}

void HandGestureRecognizing::scheduleCommitOrDefer(
        std::set<std::string> candidate_set, double now) {
    double elapsed_ms = (now - gate_open_time_) * 1000.0;
    double remaining  = 0.0;
    if (config_.holds.has_value()) {
        remaining = config_.holds->t_min_buffer_ms - elapsed_ms;
    }

    if (remaining <= 0.0) {
        triggerPhase3Commit(candidate_set);
    } else {
        // Defer: set pending_commit with min_buffer_deadline only; T_commit deadline = 0.
        double min_buf_dl = gate_open_time_ + (config_.holds->t_min_buffer_ms / 1000.0);
        pending_commit_ = PendingCommit{candidate_set, now /* commit_dl = now = immediate once buffer satisfied */, min_buf_dl};
    }
}

void HandGestureRecognizing::triggerPhase3Commit(const std::set<std::string>& candidate_set) {
    if (already_committed_) return;
    already_committed_ = true;

    auto buffer = cycle_buffer_;
    if (prefix_matcher_) prefix_matcher_->reset();
    cycle_buffer_.clear();
    gate_->reset();

    if (buffer.empty() || !model_ || !model_->isLoaded()) return;

    cg_handfilm_ref film = makeFilm(buffer);
    if (bypass_phase2) {
        recognizeUnrestricted(film);
    } else {
        recognizeRestricted(film, candidate_set);
    }
    cg_handfilm_destroy(film);
}

// ---------------------------------------------------------------------------
// Private — recognition
// ---------------------------------------------------------------------------

void HandGestureRecognizing::recognizeRestricted(
        cg_handfilm_ref film, const std::set<std::string>& candidates) {
    if (!model_) return;

    float tau = config_.confidence_threshold;
    if (config_.holds.has_value()) tau = config_.holds->tau_phase3_confidence;

    std::vector<const char*> c_ids;
    c_ids.reserve(candidates.size());
    for (const auto& id : candidates) c_ids.push_back(id.c_str());

    cg_gesture_prediction pred{};
    int ok = model_->classifyRestricted(film,
                                        c_ids.data(),
                                        static_cast<int>(c_ids.size()),
                                        tau, &pred);
    if (ok) emitGesture(film, pred, static_cast<int>(candidates.size()));
}

void HandGestureRecognizing::recognizeUnrestricted(cg_handfilm_ref film) {
    if (!model_) return;
    cg_gesture_prediction pred{};
    int ok = model_->classify(film, config_.confidence_threshold, &pred);
    if (ok) emitGesture(film, pred, -1);
}

void HandGestureRecognizing::emitGesture(cg_handfilm_ref film,
                                          const cg_gesture_prediction& pred,
                                          int candidate_set_size) {
    if (!on_gesture) return;
    DetectedGestureInfo info{pred, film, candidate_set_size};
    on_gesture(info);
}

// ---------------------------------------------------------------------------
// Private — utilities
// ---------------------------------------------------------------------------

void HandGestureRecognizing::reportGateUpdate() {
    if (on_gate_update) {
        on_gate_update(gate_->state(), gate_->bufferCount());
    }
}

cg_handfilm_ref HandGestureRecognizing::makeFilm(const std::vector<cg_handshot>& shots) {
    double start = shots.empty() ? 0.0 : shots.front().timestamp;
    cg_handfilm_ref film = cg_handfilm_create(start);
    for (const auto& shot : shots) {
        cg_handfilm_add_shot(film, &shot);
    }
    return film;
}

void HandGestureRecognizing::cancelPendingCommit() {
    pending_commit_.reset();
}
