#include <gtest/gtest.h>
#include "CameraGestures/Types.h"

static cg_handshot make_shot(double ts, bool absent = false) {
    cg_handshot s{};
    s.timestamp  = ts;
    s.handedness = CG_HAND_RIGHT;
    s.is_absent  = absent ? 1 : 0;
    return s;
}

TEST(HandFilm, EmptyFilm) {
    cg_handfilm_ref f = cg_handfilm_create(1000.0);
    EXPECT_EQ(cg_handfilm_shot_count(f), 0u);
    EXPECT_DOUBLE_EQ(cg_handfilm_start_time(f), 1000.0);
    EXPECT_DOUBLE_EQ(cg_handfilm_gesture_duration(f), 0.0);
    EXPECT_DOUBLE_EQ(cg_handfilm_in_view_duration(f), 0.0);
    EXPECT_EQ(cg_handfilm_in_view_frame_count(f), 0u);
    cg_handfilm_destroy(f);
}

TEST(HandFilm, GestureDuration) {
    cg_handfilm_ref f = cg_handfilm_create(1000.0);
    cg_handshot s1 = make_shot(1000.0);
    cg_handshot s2 = make_shot(1002.5);
    cg_handfilm_add_shot(f, &s1);
    cg_handfilm_add_shot(f, &s2);
    EXPECT_DOUBLE_EQ(cg_handfilm_gesture_duration(f), 2.5);
    cg_handfilm_destroy(f);
}

// inViewDuration mirrors iOS: sum of intervals between consecutive non-absent frames
TEST(HandFilm, InViewDuration_NoAbsent) {
    cg_handfilm_ref f = cg_handfilm_create(0.0);
    cg_handshot s0 = make_shot(0.0);
    cg_handshot s1 = make_shot(1.0);
    cg_handshot s2 = make_shot(1.5);
    cg_handfilm_add_shot(f, &s0);
    cg_handfilm_add_shot(f, &s1);
    cg_handfilm_add_shot(f, &s2);
    EXPECT_DOUBLE_EQ(cg_handfilm_in_view_duration(f), 1.5);
    cg_handfilm_destroy(f);
}

TEST(HandFilm, InViewDuration_WithAbsent) {
    cg_handfilm_ref f = cg_handfilm_create(0.0);
    cg_handshot s0 = make_shot(0.0, false);
    cg_handshot s1 = make_shot(1.0, true);   // absent — skipped
    cg_handshot s2 = make_shot(2.0, false);
    cg_handshot s3 = make_shot(3.0, false);
    cg_handfilm_add_shot(f, &s0);
    cg_handfilm_add_shot(f, &s1);
    cg_handfilm_add_shot(f, &s2);
    cg_handfilm_add_shot(f, &s3);
    // visible: ts 0.0, 2.0, 3.0 → intervals 2.0 + 1.0 = 3.0
    EXPECT_DOUBLE_EQ(cg_handfilm_in_view_duration(f), 3.0);
    EXPECT_EQ(cg_handfilm_in_view_frame_count(f), 3u);
    cg_handfilm_destroy(f);
}

TEST(HandFilm, SingleVisibleFrame_ZeroInView) {
    cg_handfilm_ref f = cg_handfilm_create(0.0);
    cg_handshot s = make_shot(0.0);
    cg_handfilm_add_shot(f, &s);
    EXPECT_DOUBLE_EQ(cg_handfilm_in_view_duration(f), 0.0);
    cg_handfilm_destroy(f);
}

TEST(HandFilm, Clear) {
    cg_handfilm_ref f = cg_handfilm_create(0.0);
    cg_handshot s = make_shot(1.0);
    cg_handfilm_add_shot(f, &s);
    EXPECT_EQ(cg_handfilm_shot_count(f), 1u);
    cg_handfilm_clear(f);
    EXPECT_EQ(cg_handfilm_shot_count(f), 0u);
    cg_handfilm_destroy(f);
}

TEST(HandFilm, GetShot) {
    cg_handfilm_ref f = cg_handfilm_create(0.0);
    cg_handshot s = make_shot(5.0);
    s.handedness = CG_HAND_LEFT;
    cg_handfilm_add_shot(f, &s);
    cg_handshot out{};
    EXPECT_EQ(cg_handfilm_get_shot(f, 0, &out), 1);
    EXPECT_DOUBLE_EQ(out.timestamp, 5.0);
    EXPECT_EQ(out.handedness, CG_HAND_LEFT);
    EXPECT_EQ(cg_handfilm_get_shot(f, 1, &out), 0);  // out of range
    cg_handfilm_destroy(f);
}
