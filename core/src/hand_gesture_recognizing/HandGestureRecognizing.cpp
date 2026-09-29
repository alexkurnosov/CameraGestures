#include "HandGestureRecognizing.hpp"
#include "CameraGestures/Types.h"
#include <algorithm>
#include <cassert>

namespace {

cg_prefix_action toCPrefixAction(PrefixMatcher::Action::Kind k) {
    using K = PrefixMatcher::Action::Kind;
    switch (k) {
    case K::no_prefix:          return CG_PREFIX_NO_PREFIX;
    case K::live_prefix:        return CG_PREFIX_LIVE_PREFIX;
    case K::commit_now:         return CG_PREFIX_COMMIT_NOW;
    case K::start_commit_timer: return CG_PREFIX_START_COMMIT_TIMER;
    case K::idle_reset:         return CG_PREFIX_IDLE_RESET;
    case K::idle_discard:       return CG_PREFIX_IDLE_DISCARD;
    case K::idle_commit:        return CG_PREFIX_IDLE_COMMIT;
    }
    return CG_PREFIX_NOT_OBSERVED;
}

cg_cycle_end_reason toCEndReason(MotionGate::Event::EndReason r) {
    using R = MotionGate::Event::EndReason;
    switch (r) {
    case R::absent_frame: return CG_CYCLE_END_ABSENT_FRAME;
    case R::low_energy:   return CG_CYCLE_END_LOW_ENERGY;
    case R::buffer_cap:   return CG_CYCLE_END_BUFFER_CAP;
    case R::none:         break;
    }
    return CG_CYCLE_END_LOW_ENERGY; // unreachable: cycle_ended always carries a reason
}

std::vector<const char*> cStrings(const std::set<std::string>& ids) {
    std::vector<const char*> out;
    out.reserve(ids.size());
    for (const auto& id : ids) out.push_back(id.c_str());
    return out;
}

} // namespace

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

void HandGestureRecognizing::setPoseOverrideForTesting(PoseOverrideForTesting override_) {
    pose_override_ = std::move(override_);
    if (config_.holds.has_value()) {
        prefix_matcher_ = std::make_unique<PrefixMatcher>(pose_override_->manifest);
    }
}

void HandGestureRecognizing::setGateEnabled(bool enabled) {
    config_.gate_enabled = enabled;
    if (!enabled) resetGateState();
}

void HandGestureRecognizing::processShot(const cg_handshot& shot) {
    frame_          = FrameScratch{};
    last_shot_time_ = shot.timestamp;

    if (config_.gate_enabled) {
        handleShotWithGate(shot);
    } else {
        // Legacy per-film mode: no gate, recognition fires from handfilm stream.
        // (Training App v2 always uses gate-enabled; this path exists for parity.)
    }

    reportFrameTelemetry(shot);
}

bool HandGestureRecognizing::tickTimers(double now) {
    if (!pending_commit_) return false;
    if (already_committed_) { cancelPendingCommit(); return false; }

    auto& pc = *pending_commit_;

    // Wait for min-buffer first.
    if (pc.min_buffer_deadline > 0.0 && now < pc.min_buffer_deadline) return false;
    // Then wait for T_commit.
    if (now < pc.commit_deadline) return false;

    // Both deadlines satisfied — fire. The later of the two is the one that gated it.
    auto   candidate_set = pc.candidate_set;
    double commit_dl     = pc.commit_deadline;
    double min_buf_dl    = pc.min_buffer_deadline;
    cancelPendingCommit();
    cg_commit_gate gate = (min_buf_dl > commit_dl) ? CG_COMMIT_T_MIN_BUFFER : CG_COMMIT_T_COMMIT;
    triggerPhase3Commit(candidate_set, gate, now, commit_dl, min_buf_dl);
    return true;
}

void HandGestureRecognizing::resetGateState() {
    if (gate_->state() == MotionGateState::open) {
        reportCycleEnded(CG_CYCLE_END_EXTERNAL_RESET,
                         static_cast<int>(cycle_buffer_.size()), last_shot_time_);
    }
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
    frame_.raw_energy       = event.energy;
    frame_.raw_energy_valid = event.energy_valid;

    switch (event.kind) {
    case MotionGate::Event::Kind::still_closed:
        break;

    case MotionGate::Event::Kind::opened: {
        cg_decision_event ev{};
        ev.kind      = CG_EVENT_GATE_OPENED;
        ev.timestamp = shot.timestamp;
        reportEvent(ev);
        cycle_buffer_.clear();
        gate_open_time_    = shot.timestamp;
        already_committed_ = false;
        cancelPendingCommit();
        if (hold_detector_)  hold_detector_->reset();
        if (prefix_matcher_) prefix_matcher_->reset();
        break;
    }

    case MotionGate::Event::Kind::still_open:
        cycle_buffer_.push_back(shot);
        // Phase-2 hold detection.
        if (hold_detector_ && poseAvailable()) {
            auto hold_event = hold_detector_->process(shot);
            frame_.smoothed_valid  = hold_event.energy_valid;
            frame_.smoothed_energy = hold_event.smoothed_energy;
            frame_.hold_run_frames = hold_event.hold_run_frames;
            frame_.hold_run_ms     = hold_event.hold_run_ms;
            if (hold_event.hold_detected) {
                handlePhase2Hold(hold_event.rep_shot,
                                 hold_event.start_time, hold_event.end_time);
            }
        }
        break;

    case MotionGate::Event::Kind::cycle_ended:
        reportCycleEnded(toCEndReason(event.end_reason),
                         static_cast<int>(event.cycle_buffer.size()), shot.timestamp);
        cancelPendingCommit();
        handleCycleEnd(std::move(event.cycle_buffer), shot.timestamp);
        break;
    }

    reportGateUpdate();
}

void HandGestureRecognizing::handleCycleEnd(std::vector<cg_handshot> buffer, double now) {
    cg_decision_event skip{};
    skip.kind         = CG_EVENT_CYCLE_SKIPPED;
    skip.timestamp    = now;
    skip.buffer_count = static_cast<int>(buffer.size());

    // Holds mode: if Phase 2 already committed this cycle, skip Phase 3.
    if (already_committed_) {
        already_committed_ = false;
        cycle_buffer_.clear();
        if (prefix_matcher_) prefix_matcher_->reset();
        skip.reason = CG_CYCLE_SKIP_ALREADY_COMMITTED;
        reportEvent(skip);
        return;
    }

    if (buffer.empty() || !model_ || !model_->isLoaded()) {
        skip.reason = buffer.empty() ? CG_CYCLE_SKIP_EMPTY_BUFFER : CG_CYCLE_SKIP_NO_MODEL;
        reportEvent(skip);
        return;
    }

    // Require at least 5 real frames; brief false-positive cycles (1–2 frames)
    // produce feature vectors with all-zero velocity/std that map to garbage
    // gesture predictions even above the confidence threshold.
    static constexpr int kMinFramesForRecognition = 5;
    if (static_cast<int>(buffer.size()) < kMinFramesForRecognition) {
        skip.reason = CG_CYCLE_SKIP_TOO_FEW_FRAMES;
        reportEvent(skip);
        return;
    }

    cg_handfilm_ref film = makeFilm(buffer);

    if (config_.gate_enabled && config_.holds.has_value()) {
        // Gate-close commit path.
        std::optional<std::set<std::string>> candidate_set;
        if (prefix_matcher_) candidate_set = prefix_matcher_->gateCloseCommitSet();
        if (prefix_matcher_) prefix_matcher_->reset();
        cycle_buffer_.clear();

        if (candidate_set && !bypass_phase2) {
            recognizeRestricted(film, *candidate_set, now);
        } else {
            recognizeUnrestricted(film, now);
        }
    } else {
        recognizeUnrestricted(film, now);
    }

    cg_handfilm_destroy(film);
}

// ---------------------------------------------------------------------------
// Private — Phase-2 hold handler
// ---------------------------------------------------------------------------

void HandGestureRecognizing::handlePhase2Hold(
        const cg_handshot& rep_shot, double start_t, double end_t) {
    if (!config_.holds.has_value()) return;
    const auto& holds_cfg = *config_.holds;

    if ((!model_ && !pose_override_) || !prefix_matcher_) return;

    cg_decision_event hold_ev{};
    hold_ev.kind            = CG_EVENT_HOLD_COMPLETED;
    hold_ev.reason          = CG_PREFIX_NOT_OBSERVED;
    hold_ev.timestamp       = end_t;
    hold_ev.hold_start_time = start_t;
    hold_ev.rep_shot_time   = rep_shot.timestamp;
    auto reportHold = [&](int reason) {
        const auto& seq    = prefix_matcher_->observedSequence();
        hold_ev.reason       = reason;
        hold_ev.observed_seq = seq.data();
        hold_ev.n_observed   = static_cast<int>(seq.size());
        reportEvent(hold_ev);
    };

    // Predict pose cluster.
    int   pose_id    = 0;
    float confidence = 0.0f;
    int ok = predictPose(rep_shot, &pose_id, &confidence);
    if (!ok) {
        hold_ev.pose_id = -1;
        reportHold(CG_PREFIX_NOT_OBSERVED);
        return;
    }
    hold_ev.pose_id    = pose_id;
    hold_ev.confidence = confidence;

    // Compute normalised pose vector for review if requested (shared by both
    // telemetry call-sites below; empty when flag is off).
    std::vector<float> norm_coords;
    if (config_.retain_landmarks_for_review) {
        norm_coords = MotionGate::normalize(rep_shot);
    }

    // Reject below τ_pose_confidence.
    if (confidence < holds_cfg.tau_pose_confidence) {
        // Report telemetry but don't advance the sequence.
        if (on_holds_telemetry) {
            on_holds_telemetry(pose_id, confidence,
                               prefix_matcher_->observedSequence(), "",
                               &rep_shot, norm_coords);
        }
        reportHold(CG_PREFIX_NOT_OBSERVED);
        return;
    }

    CgClusterKind kind = poseManifest()
        ? poseManifest()->clusterKind(pose_id)
        : CgClusterKind::unconfirmed;

    auto action = prefix_matcher_->observe(pose_id, kind);

    // Telemetry
    if (on_holds_telemetry) {
        std::string matched;
        if (auto cs = prefix_matcher_->gateCloseCommitSet()) {
            if (!cs->empty()) matched = *cs->begin();
        }
        on_holds_telemetry(pose_id, confidence,
                           prefix_matcher_->observedSequence(), matched,
                           &rep_shot, norm_coords);
    }

    hold_ev.accepted = 1;
    reportHold(toCPrefixAction(action.kind));

    double now = rep_shot.timestamp;

    switch (action.kind) {
    case PrefixMatcher::Action::Kind::no_prefix:
    case PrefixMatcher::Action::Kind::idle_discard:
        cancelPendingCommit();
        prefix_matcher_->reset();
        reportCycleEnded(CG_CYCLE_END_PHASE2_DISCARD,
                         static_cast<int>(cycle_buffer_.size()), end_t);
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
        reportCycleEnded(CG_CYCLE_END_PHASE2_DISCARD,
                         static_cast<int>(cycle_buffer_.size()), end_t);
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
        // `now` is the representative frame's time; the event happens on this shot.
        triggerPhase3Commit(candidate_set, CG_COMMIT_IMMEDIATE, last_shot_time_);
    } else {
        // Defer: set pending_commit with min_buffer_deadline only; T_commit deadline = 0.
        double min_buf_dl = gate_open_time_ + (config_.holds->t_min_buffer_ms / 1000.0);
        pending_commit_ = PendingCommit{candidate_set, now /* commit_dl = now = immediate once buffer satisfied */, min_buf_dl};
    }
}

void HandGestureRecognizing::triggerPhase3Commit(const std::set<std::string>& candidate_set,
                                                 cg_commit_gate gate, double now,
                                                 double commit_deadline,
                                                 double min_buffer_deadline) {
    if (already_committed_) return;
    already_committed_ = true;

    if (on_decision_event) {
        auto ids = cStrings(candidate_set);
        cg_decision_event ev{};
        ev.kind                = CG_EVENT_COMMIT_FIRED;
        ev.reason              = gate;
        ev.timestamp           = now;
        ev.candidate_ids       = ids.data();
        ev.n_candidates        = static_cast<int>(ids.size());
        ev.commit_deadline     = commit_deadline;
        ev.min_buffer_deadline = min_buffer_deadline;
        reportEvent(ev);
    }

    auto buffer = cycle_buffer_;
    if (prefix_matcher_) prefix_matcher_->reset();
    cycle_buffer_.clear();
    if (gate_->state() == MotionGateState::open) {
        reportCycleEnded(CG_CYCLE_END_COMMITTED, static_cast<int>(buffer.size()), now);
    }
    gate_->reset();

    if (buffer.empty() || !model_ || !model_->isLoaded()) {
        cg_decision_event skip{};
        skip.kind         = CG_EVENT_CYCLE_SKIPPED;
        skip.reason       = buffer.empty() ? CG_CYCLE_SKIP_EMPTY_BUFFER : CG_CYCLE_SKIP_NO_MODEL;
        skip.timestamp    = now;
        skip.buffer_count = static_cast<int>(buffer.size());
        reportEvent(skip);
        return;
    }

    cg_handfilm_ref film = makeFilm(buffer);
    if (bypass_phase2) {
        recognizeUnrestricted(film, now);
    } else {
        recognizeRestricted(film, candidate_set, now);
    }
    cg_handfilm_destroy(film);
}

// ---------------------------------------------------------------------------
// Private — recognition
// ---------------------------------------------------------------------------

void HandGestureRecognizing::recognizeRestricted(
        cg_handfilm_ref film, const std::set<std::string>& candidates, double now) {
    if (!model_) return;

    float tau = config_.confidence_threshold;
    if (config_.holds.has_value()) tau = config_.holds->tau_phase3_confidence;

    std::vector<const char*> c_ids = cStrings(candidates);

    cg_gesture_prediction pred{};
    int ok = model_->classifyRestricted(film,
                                        c_ids.data(),
                                        static_cast<int>(c_ids.size()),
                                        tau, &pred);

    cg_decision_event ev{};
    ev.kind          = CG_EVENT_PHASE3_PREDICTION;
    ev.timestamp     = now;
    ev.buffer_count  = static_cast<int>(cg_handfilm_shot_count(film));
    ev.restricted    = 1;
    ev.accepted      = ok ? 1 : 0;
    ev.gesture_id    = ok ? pred.gesture_id : "";
    ev.confidence    = ok ? pred.confidence : 0.0f;
    ev.candidate_ids = c_ids.data();
    ev.n_candidates  = static_cast<int>(c_ids.size());
    reportEvent(ev);

    if (ok) emitGesture(film, pred, static_cast<int>(candidates.size()));
}

void HandGestureRecognizing::recognizeUnrestricted(cg_handfilm_ref film, double now) {
    if (!model_) return;
    cg_gesture_prediction pred{};
    int ok = model_->classify(film, config_.confidence_threshold, &pred);

    cg_decision_event ev{};
    ev.kind         = CG_EVENT_PHASE3_PREDICTION;
    ev.timestamp    = now;
    ev.buffer_count = static_cast<int>(cg_handfilm_shot_count(film));
    ev.restricted   = 0;
    ev.accepted     = ok ? 1 : 0;
    ev.gesture_id   = ok ? pred.gesture_id : "";
    ev.confidence   = ok ? pred.confidence : 0.0f;
    reportEvent(ev);

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

bool HandGestureRecognizing::poseAvailable() const {
    if (pose_override_) return true;
    return model_ && model_->isPoseLoaded();
}

bool HandGestureRecognizing::predictPose(const cg_handshot& shot,
                                         int* pose_id, float* confidence) {
    if (pose_override_) return pose_override_->predict(shot, pose_id, confidence);
    return model_->predictPose(shot, pose_id, confidence) != 0;
}

const CgPoseManifest* HandGestureRecognizing::poseManifest() const {
    if (pose_override_) return &pose_override_->manifest;
    return model_ ? model_->poseManifest() : nullptr;
}

// ---------------------------------------------------------------------------
// Private — telemetry
// ---------------------------------------------------------------------------

void HandGestureRecognizing::reportFrameTelemetry(const cg_handshot& shot) {
    if (!on_frame_telemetry) return;

    cg_frame_telemetry row{};
    row.timestamp    = shot.timestamp;
    if (pending_commit_) {
        row.commit_deadline     = pending_commit_->commit_deadline;
        row.min_buffer_deadline = pending_commit_->min_buffer_deadline;
    }
    row.raw_energy            = frame_.raw_energy;
    row.raw_energy_valid      = frame_.raw_energy_valid ? 1 : 0;
    row.smoothed_energy_valid = frame_.smoothed_valid ? 1 : 0;
    if (frame_.smoothed_valid) {
        row.smoothed_energy = frame_.smoothed_energy;
        row.hold_run_frames = frame_.hold_run_frames;
        row.hold_run_ms     = static_cast<float>(frame_.hold_run_ms);
    }
    row.track_index  = -1;
    row.buffer_count = gate_->bufferCount();
    row.handedness   = static_cast<uint8_t>(shot.handedness);
    row.is_absent    = shot.is_absent ? 1 : 0;
    row.gate_enabled = config_.gate_enabled ? 1 : 0;
    row.gate_open    = gate_->state() == MotionGateState::open ? 1 : 0;
    on_frame_telemetry(row);
}

void HandGestureRecognizing::reportEvent(const cg_decision_event& ev) {
    if (on_decision_event) on_decision_event(ev);
}

void HandGestureRecognizing::reportCycleEnded(cg_cycle_end_reason reason,
                                              int buffer_count, double now) {
    if (!on_decision_event) return;
    cg_decision_event ev{};
    ev.kind         = CG_EVENT_CYCLE_ENDED;
    ev.reason       = reason;
    ev.timestamp    = now;
    ev.buffer_count = buffer_count;
    on_decision_event(ev);
}
