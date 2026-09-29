/*
 * test_telemetry.cpp — session capture Stage 1: frame telemetry and decision events.
 *
 * Synthetic hand streams with a controlled per-frame motion energy drive the
 * recognizer through each decision. Phase 2 runs on PoseOverrideForTesting, so
 * no TFLite model is needed and the pose sequence is exact.
 */

#include <gtest/gtest.h>
#include "CameraGestures/HandGestureRecognizing.h"
#include "HandGestureRecognizing.hpp"
#include "MotionGate.hpp"
#include <cmath>
#include <set>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Synthetic hand stream
// ---------------------------------------------------------------------------

namespace {

constexpr double kDt = 0.04; // 25 fps; keeps k_open (33 ms) and k_hold (100 ms) off frame boundaries

// A straight "hand": landmark i at (0.5 + 0.01 i, 0.5 + 0.02 i). The wrist (0)
// and middle MCP (9) fix the normalisation, so shifting every other landmark
// by `dx` changes MotionGate energy by exactly 19 * |dx| / scale.
cg_handshot poseShot(double t, float dx) {
    cg_handshot s{};
    s.timestamp  = t;
    s.handedness = CG_HAND_RIGHT;
    for (int i = 0; i < 21; ++i) {
        s.landmarks[i].x = 0.5f + 0.01f * i + ((i == 0 || i == 9) ? 0.0f : dx);
        s.landmarks[i].y = 0.5f + 0.02f * i;
        s.landmarks[i].z = 0.0f;
    }
    return s;
}

cg_handshot absentShot(double t) {
    cg_handshot s{};
    s.timestamp  = t;
    s.handedness = CG_HAND_RIGHT;
    s.is_absent  = 1;
    return s;
}

// Landmark shift that produces the given per-frame energy between two poses.
float shiftForEnergy(float energy) {
    const float scale = std::sqrt(0.09f * 0.09f + 0.18f * 0.18f);
    return energy * scale / 19.0f;
}

// Builds a stream frame by frame; oscillating between two poses holds the
// per-frame energy constant.
struct Stream {
    std::vector<cg_handshot> shots;
    double t    = 1000.0;
    bool   flip = false;
    float  dx   = 0.0f; // the last pose's shift

    Stream& move(int frames, float energy) {
        for (int i = 0; i < frames; ++i) {
            flip = !flip;
            dx   = flip ? shiftForEnergy(energy) : 0.0f;
            shots.push_back(poseShot(t, dx));
            t += kDt;
        }
        return *this;
    }
    Stream& still(int frames) { // repeats the last pose: zero energy
        for (int i = 0; i < frames; ++i) {
            shots.push_back(poseShot(t, dx));
            t += kDt;
        }
        return *this;
    }
    Stream& absent(int frames = 1) {
        for (int i = 0; i < frames; ++i) { shots.push_back(absentShot(t)); t += kDt; }
        return *this;
    }
};

// Records everything the recognizer reports.
struct Recorder {
    std::vector<cg_frame_telemetry> rows;
    struct Event { int kind; int reason; double t; double commit_dl; double min_buf_dl; };
    std::vector<Event> events;

    void attach(HandGestureRecognizing& r) {
        r.on_frame_telemetry = [this](const cg_frame_telemetry& row) { rows.push_back(row); };
        r.on_decision_event  = [this](const cg_decision_event& e) {
            events.push_back({e.kind, e.reason, e.timestamp,
                              e.commit_deadline, e.min_buffer_deadline});
        };
    }
    std::vector<Event> ofKind(int kind) const {
        std::vector<Event> out;
        for (const auto& e : events) if (e.kind == kind) out.push_back(e);
        return out;
    }
};

// Feed shots and tick timers at each shot time, as the replay rig does.
void run(HandGestureRecognizing& r, const std::vector<cg_handshot>& shots) {
    for (const auto& s : shots) {
        r.processShot(s);
        r.tickTimers(s.timestamp);
    }
}

HandGestureRecognizingConfig gateConfig() {
    HandGestureRecognizingConfig cfg;
    cfg.gate_enabled = true;
    return cfg;
}

HandGestureRecognizingConfig holdsConfig() {
    auto cfg  = gateConfig();
    cfg.holds = HoldsConfig{};
    return cfg;
}

// Two regular poses. Pose 1 completes gesture "a" with nothing longer, so a
// hold on it commits at once. Pose 2 completes "b" but also starts "c", so a
// hold on it starts the T_commit timer.
PoseOverrideForTesting fakePoseModel(int pose_id, float confidence = 0.95f) {
    PoseOverrideForTesting o;
    o.manifest.version = 2;
    o.manifest.pose_clusters["1"] = CgPoseCluster{"p1", CgClusterKind::regular};
    o.manifest.pose_clusters["2"] = CgPoseCluster{"p2", CgClusterKind::regular};
    o.manifest.gesture_templates["a"] = {{1}};
    o.manifest.gesture_templates["b"] = {{2}};
    o.manifest.gesture_templates["c"] = {{2, 1}};
    o.predict = [pose_id, confidence](const cg_handshot&, int* id, float* conf) {
        *id   = pose_id;
        *conf = confidence;
        return true;
    };
    return o;
}

} // namespace

// ---------------------------------------------------------------------------
// One row per processed shot (through the public C API)
// ---------------------------------------------------------------------------

TEST(FrameTelemetry, OneRowPerShotThroughCApi) {
    cg_recognizer_config cfg = cg_recognizer_default_config();
    cfg.gate_enabled = 1;
    cg_recognizer_ref rec = cg_recognizer_create(&cfg, nullptr);
    ASSERT_NE(rec, nullptr);

    std::vector<cg_frame_telemetry> rows;
    cg_recognizer_set_frame_telemetry_callback(rec,
        [](void* ctx, const cg_frame_telemetry* row) {
            static_cast<std::vector<cg_frame_telemetry>*>(ctx)->push_back(*row);
        }, &rows);

    Stream s;
    s.move(40, 3.0f).still(20).absent(5).move(35, 1.2f); // 100 shots, gate opens and closes
    ASSERT_EQ(s.shots.size(), 100u);
    for (const auto& shot : s.shots) {
        cg_recognizer_process_shot(rec, &shot);
        cg_recognizer_tick_timers(rec, shot.timestamp);
    }

    ASSERT_EQ(rows.size(), 100u);
    for (size_t i = 0; i < rows.size(); ++i) {
        EXPECT_EQ(rows[i].timestamp, s.shots[i].timestamp);
        EXPECT_EQ(rows[i].track_index, -1);
        if (i > 0) EXPECT_GT(rows[i].timestamp, rows[i - 1].timestamp);
    }

    // Unsetting the callback stops the rows.
    cg_recognizer_set_frame_telemetry_callback(rec, nullptr, nullptr);
    cg_recognizer_process_shot(rec, &s.shots.back());
    EXPECT_EQ(rows.size(), 100u);

    cg_recognizer_destroy(rec);
}

// ---------------------------------------------------------------------------
// Raw energy is MotionGate's own value
// ---------------------------------------------------------------------------

TEST(FrameTelemetry, RawEnergyMatchesMotionGateEnergy) {
    HandGestureRecognizing r(gateConfig());
    Recorder rec;
    rec.attach(r);

    cg_handshot a = poseShot(10.0, 0.0f);
    cg_handshot b = poseShot(10.04, 0.013f);
    b.landmarks[4].y += 0.02f; // not just a uniform shift
    r.processShot(a);
    r.processShot(b);

    ASSERT_EQ(rec.rows.size(), 2u);
    EXPECT_EQ(rec.rows[0].raw_energy_valid, 0); // no previous frame: the gate used 0
    EXPECT_EQ(rec.rows[0].raw_energy, 0.0f);
    EXPECT_EQ(rec.rows[1].raw_energy_valid, 1);
    const float expected = MotionGate::energy(MotionGate::normalize(b), MotionGate::normalize(a));
    EXPECT_GT(expected, 0.0f);
    EXPECT_EQ(rec.rows[1].raw_energy, expected);
}

// ---------------------------------------------------------------------------
// Smoothed energy validity
// ---------------------------------------------------------------------------

TEST(FrameTelemetry, SmoothedEnergyAbsentWhileGateClosed) {
    HandGestureRecognizing r(holdsConfig());
    r.setPoseOverrideForTesting(fakePoseModel(1));
    Recorder rec;
    rec.attach(r);

    Stream s;
    s.move(60, 0.8f).still(20); // below t_open: the gate never opens
    run(r, s.shots);

    ASSERT_EQ(rec.rows.size(), s.shots.size());
    for (const auto& row : rec.rows) {
        EXPECT_EQ(row.gate_open, 0);
        EXPECT_EQ(row.smoothed_energy_valid, 0);
        EXPECT_EQ(row.hold_run_frames, 0);
    }
    EXPECT_TRUE(rec.ofKind(CG_EVENT_GATE_OPENED).empty());
}

TEST(FrameTelemetry, SmoothedEnergyAbsentWithoutPoseModel) {
    HandGestureRecognizing r(holdsConfig()); // holds mode, but no pose model
    Recorder rec;
    rec.attach(r);

    Stream s;
    s.move(20, 3.0f);
    run(r, s.shots);

    int open_rows = 0;
    for (const auto& row : rec.rows) {
        if (row.gate_open) ++open_rows;
        EXPECT_EQ(row.smoothed_energy_valid, 0);
    }
    EXPECT_GT(open_rows, 10);
}

TEST(FrameTelemetry, SmoothedEnergyPresentWhileGateOpenWithPoseModel) {
    // Positive control for the two tests above: a pose model is present. Its
    // confidence stays under tau_pose_confidence, so holds never commit and
    // the gate is not reset mid-stream.
    HandGestureRecognizing r(holdsConfig());
    r.setPoseOverrideForTesting(fakePoseModel(1, 0.1f));
    Recorder rec;
    rec.attach(r);

    Stream s;
    s.move(12, 3.0f);
    run(r, s.shots);

    int valid = 0;
    for (const auto& row : rec.rows) {
        if (row.smoothed_energy_valid) {
            ++valid;
            EXPECT_EQ(row.gate_open, 1);
        }
    }
    EXPECT_GT(valid, 5);
}

// ---------------------------------------------------------------------------
// Cycle-end reasons
// ---------------------------------------------------------------------------

namespace {

std::vector<Recorder::Event> cycleEnds(HandGestureRecognizingConfig cfg,
                                       const std::vector<cg_handshot>& shots) {
    HandGestureRecognizing r(cfg);
    Recorder rec;
    rec.attach(r);
    run(r, shots);
    return rec.ofKind(CG_EVENT_CYCLE_ENDED);
}

} // namespace

TEST(DecisionEvents, CycleEndReasonsAreDistinct) {
    Stream absent_end;
    absent_end.move(10, 3.0f).absent();
    auto a = cycleEnds(gateConfig(), absent_end.shots);

    Stream low_energy_end;
    low_energy_end.move(10, 3.0f).still(10);
    auto b = cycleEnds(gateConfig(), low_energy_end.shots);

    auto capped = gateConfig();
    capped.gesture_buffer_size = 5;
    Stream cap_end;
    cap_end.move(12, 3.0f);
    auto c = cycleEnds(capped, cap_end.shots);

    ASSERT_EQ(a.size(), 1u);
    ASSERT_GE(b.size(), 1u);
    ASSERT_GE(c.size(), 1u);
    EXPECT_EQ(a[0].reason, CG_CYCLE_END_ABSENT_FRAME);
    EXPECT_EQ(b[0].reason, CG_CYCLE_END_LOW_ENERGY);
    EXPECT_EQ(c[0].reason, CG_CYCLE_END_BUFFER_CAP);

    // With no gesture model the cycle is skipped, and says why.
    HandGestureRecognizing r(gateConfig());
    Recorder rec;
    rec.attach(r);
    run(r, absent_end.shots);
    auto skipped = rec.ofKind(CG_EVENT_CYCLE_SKIPPED);
    ASSERT_EQ(skipped.size(), 1u);
    EXPECT_EQ(skipped[0].reason, CG_CYCLE_SKIP_NO_MODEL);
}

// ---------------------------------------------------------------------------
// Commit deadline attribution
// ---------------------------------------------------------------------------

namespace {

// Moderate motion: opens the gate (> t_open), keeps it open (> t_close), and
// smooths below t_hold, so a hold completes ~120 ms after the gate opens —
// well inside T_min_buffer (200 ms).
std::vector<Recorder::Event> commitsFor(int pose_id, Recorder& rec) {
    HandGestureRecognizing r(holdsConfig());
    r.setPoseOverrideForTesting(fakePoseModel(pose_id));
    rec.attach(r);
    Stream s;
    s.move(25, 1.2f);
    run(r, s.shots);
    return rec.ofKind(CG_EVENT_COMMIT_FIRED);
}

} // namespace

TEST(DecisionEvents, CommitReportsTheGatingDeadline) {
    // Pose 1 → commit_now, deferred only by T_min_buffer.
    Recorder rec_min;
    auto by_min_buffer = commitsFor(1, rec_min);
    // Pose 2 → start_commit_timer; T_commit ends after T_min_buffer.
    Recorder rec_commit;
    auto by_t_commit = commitsFor(2, rec_commit);

    ASSERT_FALSE(by_min_buffer.empty());
    ASSERT_FALSE(by_t_commit.empty());

    const auto& m = by_min_buffer[0];
    EXPECT_EQ(m.reason, CG_COMMIT_T_MIN_BUFFER);
    EXPECT_GT(m.min_buf_dl, m.commit_dl);
    EXPECT_GE(m.t, m.min_buf_dl);

    const auto& c = by_t_commit[0];
    EXPECT_EQ(c.reason, CG_COMMIT_T_COMMIT);
    EXPECT_GE(c.commit_dl, c.min_buf_dl);
    EXPECT_GE(c.t, c.commit_dl);

    EXPECT_NE(m.reason, c.reason);

    // The hold that led there was reported with the prefix action taken.
    auto holds_min = rec_min.ofKind(CG_EVENT_HOLD_COMPLETED);
    auto holds_commit = rec_commit.ofKind(CG_EVENT_HOLD_COMPLETED);
    ASSERT_FALSE(holds_min.empty());
    ASSERT_FALSE(holds_commit.empty());
    EXPECT_EQ(holds_min[0].reason, CG_PREFIX_COMMIT_NOW);
    EXPECT_EQ(holds_commit[0].reason, CG_PREFIX_START_COMMIT_TIMER);

    // Pending deadlines show up in the frame rows while the commit waits.
    bool saw_pending = false;
    for (const auto& row : rec_commit.rows) {
        if (row.commit_deadline > 0.0 && row.timestamp < c.t) saw_pending = true;
    }
    EXPECT_TRUE(saw_pending);

    // The commit resets the gate, and that cycle end is reported as a commit.
    auto ends = rec_min.ofKind(CG_EVENT_CYCLE_ENDED);
    ASSERT_FALSE(ends.empty());
    EXPECT_EQ(ends[0].reason, CG_CYCLE_END_COMMITTED);
}

// ---------------------------------------------------------------------------
// Event names
// ---------------------------------------------------------------------------

TEST(DecisionEvents, NamesAreStable) {
    EXPECT_STREQ(cg_decision_event_kind_name(CG_EVENT_CYCLE_ENDED), "cycle_ended");
    EXPECT_STREQ(cg_decision_event_reason_name(CG_EVENT_CYCLE_ENDED, CG_CYCLE_END_BUFFER_CAP),
                 "buffer_cap");
    EXPECT_STREQ(cg_decision_event_reason_name(CG_EVENT_COMMIT_FIRED, CG_COMMIT_T_MIN_BUFFER),
                 "t_min_buffer");
    EXPECT_STREQ(cg_decision_event_kind_name(99), "unknown");
    EXPECT_STREQ(cg_decision_event_reason_name(CG_EVENT_CYCLE_SKIPPED, 99), "unknown");
}
