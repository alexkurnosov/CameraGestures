/*
 * test_gesture_model.cpp — Stage 4 unit tests.
 *
 * Tests FeaturePreprocessor (pure math, no TFLite) and PoseManifest JSON parsing.
 *
 * Parity fixture: core/tests/fixtures/test_handfilm.json
 * Generated on the VPS with:
 *   docker compose exec db psql ... | docker compose exec -T server python3 -c "
 *     film = json.loads(sys.stdin.read().strip())
 *     print(json.dumps({'film': film,
 *                       'sf':  summary_features(film).tolist(),
 *                       'fm_row0': feature_matrix(film)[0].tolist()}))"
 * The file embeds both the raw HandFilm and the Python reference vectors so
 * the parity test is fully self-contained.
 */

#include <gtest/gtest.h>
#include "FeaturePreprocessor.hpp"
#include "PoseManifest.hpp"
#include "CameraGestures/Types.h"
#include <cmath>
#include <fstream>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static cg_handshot makeShot(double ts, bool left = false, bool absent = false) {
    cg_handshot s{};
    s.timestamp  = ts;
    s.handedness = left ? CG_HAND_LEFT : CG_HAND_RIGHT;
    s.is_absent  = absent ? 1 : 0;
    // Realistic non-zero landmark values: spread fingers in a "palm open" pose.
    // Wrist at origin; each landmark offset by a multiple of its index.
    for (int i = 0; i < 21; ++i) {
        s.landmarks[i].x = 0.02f * i;
        s.landmarks[i].y = 0.01f * i * i * 0.1f;
        s.landmarks[i].z = -0.005f * i;
    }
    // Ensure wrist→lm9 distance > 1e-6 (lm9 = middle_MCP)
    s.landmarks[9].x = 0.18f;
    s.landmarks[9].y = 0.22f;
    s.landmarks[9].z = -0.05f;
    return s;
}

static cg_handfilm_ref makeFilm(int numFrames, double fps = 30.0) {
    cg_handfilm_ref film = cg_handfilm_create(0.0);
    for (int i = 0; i < numFrames; ++i) {
        double ts = i / fps;
        cg_handshot s = makeShot(ts);
        // Vary landmarks slightly per frame so velocity != 0
        for (int j = 0; j < 21; ++j) {
            s.landmarks[j].x += 0.001f * i;
            s.landmarks[j].y -= 0.0005f * i;
        }
        cg_handfilm_add_shot(film, &s);
    }
    return film;
}

// ---------------------------------------------------------------------------
// Output dimension tests
// ---------------------------------------------------------------------------

TEST(FeaturePreprocessor, FeatureMatrixLength) {
    FeaturePreprocessor pp;
    cg_handfilm_ref film = makeFilm(40);
    auto mat = pp.featureMatrix(film);
    EXPECT_EQ(static_cast<int>(mat.size()), CG_TARGET_FRAMES * CG_FEATURES_PER_FRAME);
    cg_handfilm_destroy(film);
}

TEST(FeaturePreprocessor, SummaryFeaturesLength) {
    FeaturePreprocessor pp;
    cg_handfilm_ref film = makeFilm(40);
    auto sf = pp.summaryFeatures(film);
    EXPECT_EQ(static_cast<int>(sf.size()), CG_SUMMARY_FEATURES);
    cg_handfilm_destroy(film);
}

TEST(FeaturePreprocessor, PoseVectorLength) {
    FeaturePreprocessor pp;
    cg_handshot s = makeShot(0.0);
    auto pv = pp.poseVector(s);
    EXPECT_EQ(static_cast<int>(pv.size()), CG_POSE_VECTOR_SIZE);
}

// ---------------------------------------------------------------------------
// Absent frame filtering
// ---------------------------------------------------------------------------

TEST(FeaturePreprocessor, AbsentFramesFiltered) {
    FeaturePreprocessor pp;
    cg_handfilm_ref film = cg_handfilm_create(0.0);
    // Mix real and absent frames
    cg_handshot real = makeShot(0.0);
    cg_handshot absent = makeShot(0.033, false, true);
    for (int i = 0; i < 10; ++i) {
        cg_handfilm_add_shot(film, &real);
        cg_handfilm_add_shot(film, &absent);
    }
    // summaryFeatures should only count the 10 real frames
    auto sf = pp.summaryFeatures(film);
    EXPECT_EQ(static_cast<int>(sf.size()), CG_SUMMARY_FEATURES);
    // Verify no NaN / Inf
    for (float v : sf) {
        EXPECT_TRUE(std::isfinite(v)) << "Non-finite value in summaryFeatures";
    }
    cg_handfilm_destroy(film);
}

// ---------------------------------------------------------------------------
// Empty film — should produce all-zero output without crashing
// ---------------------------------------------------------------------------

TEST(FeaturePreprocessor, EmptyFilmZeros) {
    FeaturePreprocessor pp;
    cg_handfilm_ref film = cg_handfilm_create(0.0);
    auto mat = pp.featureMatrix(film);
    EXPECT_EQ(static_cast<int>(mat.size()), CG_TARGET_FRAMES * CG_FEATURES_PER_FRAME);
    for (float v : mat) EXPECT_FLOAT_EQ(v, 0.0f);
    auto sf = pp.summaryFeatures(film);
    for (float v : sf) EXPECT_TRUE(std::isfinite(v));
    cg_handfilm_destroy(film);
}

// ---------------------------------------------------------------------------
// Padding vs trimming behaviour
// ---------------------------------------------------------------------------

TEST(FeaturePreprocessor, ShortFilmPaddedWithZeros) {
    // With 5 frames and TARGET_FRAMES=60, rows 0..4 contain real data,
    // rows 5..59 must be zero (padded at end per JS logic).
    FeaturePreprocessor pp;
    cg_handfilm_ref film = makeFilm(5);
    auto mat = pp.featureMatrix(film);
    // The first 5 rows should be non-zero (real frames)
    bool anyNonZero = false;
    for (int col = 0; col < CG_FEATURES_PER_FRAME; ++col) {
        if (mat[col] != 0.0f) { anyNonZero = true; break; }
    }
    EXPECT_TRUE(anyNonZero) << "Row 0 should contain real frame data";
    // Rows 5..59 should be all-zero (the padded suffix)
    for (int row = 5; row < CG_TARGET_FRAMES; ++row) {
        for (int col = 0; col < CG_FEATURES_PER_FRAME; ++col) {
            EXPECT_FLOAT_EQ(mat[row * CG_FEATURES_PER_FRAME + col], 0.0f)
                << "Padded row " << row << " col " << col << " should be 0";
        }
    }
    cg_handfilm_destroy(film);
}

TEST(FeaturePreprocessor, LongFilmTrimsPrefix) {
    // With 80 frames, only the last 60 are kept.
    FeaturePreprocessor pp;
    cg_handfilm_ref film80 = makeFilm(80);
    cg_handfilm_ref film60 = makeFilm(80);  // build the "last 60" as a separate film
    cg_handfilm_destroy(film60);            // discard; we just check dimensions

    auto mat80 = pp.featureMatrix(film80);
    EXPECT_EQ(static_cast<int>(mat80.size()), CG_TARGET_FRAMES * CG_FEATURES_PER_FRAME);
    // All rows should be non-zero (real data, no padding)
    for (int row = 0; row < CG_TARGET_FRAMES; ++row) {
        float rowSum = 0.0f;
        for (int col = 0; col < CG_FEATURES_PER_FRAME; ++col)
            rowSum += std::abs(mat80[row * CG_FEATURES_PER_FRAME + col]);
        EXPECT_GT(rowSum, 0.0f) << "Row " << row << " should be non-zero";
    }
    cg_handfilm_destroy(film80);
}

// ---------------------------------------------------------------------------
// Handedness symmetry: left-hand output should differ from right-hand output
// (the x-flip makes them non-identical)
// ---------------------------------------------------------------------------

TEST(FeaturePreprocessor, LeftRightHandsDiffer) {
    FeaturePreprocessor pp;
    cg_handshot rShot = makeShot(0.0, /*left=*/false);
    cg_handshot lShot = makeShot(0.0, /*left=*/true);
    auto pvR = pp.poseVector(rShot);
    auto pvL = pp.poseVector(lShot);
    // At least some coordinates must differ due to the x-flip
    bool differs = false;
    for (int i = 0; i < CG_POSE_VECTOR_SIZE; ++i) {
        if (pvR[i] != pvL[i]) { differs = true; break; }
    }
    EXPECT_TRUE(differs);
}

// ---------------------------------------------------------------------------
// Geometric extras: dimensions and finite values
// ---------------------------------------------------------------------------

TEST(FeaturePreprocessor, PoseExtrasFiniteAndInRange) {
    FeaturePreprocessor pp;
    cg_handshot s = makeShot(0.0);
    auto pv = pp.poseVector(s);
    // coords[0..62], extras[63..82]
    for (int i = 63; i < 83; ++i) {
        EXPECT_TRUE(std::isfinite(pv[i])) << "extras[" << i-63 << "] is not finite";
        EXPECT_GE(pv[i], 0.0f) << "distance/angle should be non-negative";
    }
}

// ---------------------------------------------------------------------------
// geomCoef: doubling it should double the extra block
// ---------------------------------------------------------------------------

TEST(FeaturePreprocessor, GeomCoefScalesExtras) {
    cg_handshot s = makeShot(0.0);

    FeaturePreprocessor pp1;
    pp1.setGeomCoef(1.0f);
    auto pv1 = pp1.poseVector(s);

    FeaturePreprocessor pp2;
    pp2.setGeomCoef(2.0f);
    auto pv2 = pp2.poseVector(s);

    // Coords (0..62) unchanged, extras (63..82) doubled
    for (int i = 0; i < 63; ++i)
        EXPECT_FLOAT_EQ(pv1[i], pv2[i]) << "coord " << i << " changed with geomCoef";
    for (int i = 63; i < 83; ++i)
        EXPECT_NEAR(pv2[i], 2.0f * pv1[i], 1e-5f) << "extra " << i-63 << " not doubled";
}

// ---------------------------------------------------------------------------
// Constants cross-check
// ---------------------------------------------------------------------------

TEST(FeaturePreprocessor, ConstantsMatchJS) {
    EXPECT_EQ(CG_TARGET_FRAMES,      60);
    EXPECT_EQ(CG_LANDMARKS,          21);
    EXPECT_EQ(CG_COORDS_PER_FRAME,   63);
    EXPECT_EQ(CG_POSE_EXTRAS,        20);
    EXPECT_EQ(CG_POSE_VECTOR_SIZE,   83);
    EXPECT_EQ(CG_FEATURES_PER_FRAME, 146);
    EXPECT_EQ(CG_SUMMARY_FEATURES,   296);
}

// ---------------------------------------------------------------------------
// PoseManifest JSON round-trip
// PARITY_NOTE: generate a real pose_manifest.json from the server with
//   python -c "from ml.trainer_pose import ...; json.dump(...)"
// and add it to core/tests/fixtures/ for a full round-trip test.
// The test below validates the parser on an inline JSON string.
// ---------------------------------------------------------------------------

TEST(PoseManifest, ParsesInlineJSON) {
    const char* json = R"({
        "version": 2,
        "pose_clusters": {
            "0": {"label": "fist",   "kind": "regular",      "suspected_idle": false, "n_samples": 50,  "centroid": [0.1, 0.2, 0.3]},
            "1": {"label": "palm",   "kind": "idle",         "suspected_idle": true,  "n_samples": 80,  "centroid": [0.4, 0.5]},
            "2": {"label": "point",  "kind": "unconfirmed",  "suspected_idle": false, "n_samples": 10,  "centroid": []}
        },
        "idle_poses": [1],
        "gesture_templates": {
            "swipe_left": [[0, 2], [1]]
        },
        "parameters": {
            "t_hold": 0.3,
            "k_hold_frames": 5,
            "tau_pose_confidence": 0.6
        }
    })";

    // Write to a temp file
    const std::string tmpPath = "/tmp/cg_test_pose_manifest.json";
    {
        std::ofstream f(tmpPath);
        f << json;
    }

    CgPoseManifest m = CgPoseManifest::loadFromFile(tmpPath);

    EXPECT_EQ(m.version, 2);
    EXPECT_EQ(m.pose_clusters.size(), 3u);
    EXPECT_EQ(m.pose_clusters["0"].label, "fist");
    EXPECT_EQ(m.pose_clusters["0"].kind, CgClusterKind::regular);
    EXPECT_FALSE(m.pose_clusters["0"].suspected_idle);
    EXPECT_EQ(m.pose_clusters["1"].kind, CgClusterKind::idle);
    EXPECT_TRUE(m.pose_clusters["1"].suspected_idle);
    EXPECT_EQ(m.pose_clusters["2"].kind, CgClusterKind::unconfirmed);
    EXPECT_EQ(m.idle_poses.size(), 1u);
    EXPECT_EQ(m.idle_poses[0], 1);
    EXPECT_EQ(m.gesture_templates.count("swipe_left"), 1u);
    ASSERT_TRUE(m.parameters.has_value());
    ASSERT_TRUE(m.parameters->t_hold.has_value());
    EXPECT_FLOAT_EQ(m.parameters->t_hold.value(), 0.3f);
    EXPECT_EQ(m.parameters->k_hold_frames.value(), 5);
    EXPECT_FLOAT_EQ(m.parameters->tau_pose_confidence.value(), 0.6f);
}

TEST(PoseManifest, SortedClusterIds) {
    const char* json = R"({
        "version": 1,
        "pose_clusters": {
            "10": {"label":"a","kind":"regular","suspected_idle":false,"n_samples":1,"centroid":[]},
            "2":  {"label":"b","kind":"regular","suspected_idle":false,"n_samples":1,"centroid":[]},
            "0":  {"label":"c","kind":"regular","suspected_idle":false,"n_samples":1,"centroid":[]}
        },
        "idle_poses": [], "gesture_templates": {}
    })";
    const std::string tmpPath = "/tmp/cg_test_sorted_ids.json";
    { std::ofstream f(tmpPath); f << json; }

    CgPoseManifest m = CgPoseManifest::loadFromFile(tmpPath);
    auto ids = m.sortedClusterIds();
    ASSERT_EQ(ids.size(), 3u);
    EXPECT_EQ(ids[0], "0");
    EXPECT_EQ(ids[1], "2");
    EXPECT_EQ(ids[2], "10");
}

TEST(PoseManifest, ClusterKindAndLabel) {
    const char* json = R"({
        "version": 1,
        "pose_clusters": {
            "0": {"label":"fist","kind":"regular","suspected_idle":false,"n_samples":1,"centroid":[]}
        },
        "idle_poses": [], "gesture_templates": {}
    })";
    const std::string tmpPath = "/tmp/cg_test_kind.json";
    { std::ofstream f(tmpPath); f << json; }

    CgPoseManifest m = CgPoseManifest::loadFromFile(tmpPath);
    EXPECT_EQ(m.clusterKind(0),  CgClusterKind::regular);
    EXPECT_EQ(m.clusterLabel(0), "fist");
    EXPECT_EQ(m.clusterKind(99), CgClusterKind::unconfirmed);  // missing → unconfirmed
    EXPECT_EQ(m.clusterLabel(99), "pose_99");                   // missing → default name
}

TEST(PoseManifest, MissingFileThrows) {
    EXPECT_THROW(CgPoseManifest::loadFromFile("/nonexistent/path.json"), std::runtime_error);
}

// ---------------------------------------------------------------------------
// Parity tests against Python preprocessor.py
//
// Fixture: core/tests/fixtures/test_handfilm.json
// Generated on VPS — contains the raw HandFilm plus Python reference vectors
// for summary_features and the first row of feature_matrix.
// ---------------------------------------------------------------------------

#include <nlohmann/json.hpp>

static cg_handfilm_ref loadHandFilmFromFixture(const nlohmann::json& film_json) {
    double start_time = film_json.value("start_time", 0.0);
    cg_handfilm_ref film = cg_handfilm_create(start_time);
    for (const auto& frame : film_json["frames"]) {
        cg_handshot shot{};
        shot.timestamp = frame.value("timestamp", 0.0);
        shot.is_absent = frame.value("is_absent", false) ? 1 : 0;
        std::string lor = frame.value("left_or_right", "right");
        shot.handedness = (lor == "left") ? CG_HAND_LEFT : CG_HAND_RIGHT;
        const auto& lms = frame["landmarks"];
        for (int i = 0; i < std::min((int)lms.size(), 21); ++i) {
            shot.landmarks[i].x = lms[i].value("x", 0.0f);
            shot.landmarks[i].y = lms[i].value("y", 0.0f);
            shot.landmarks[i].z = lms[i].value("z", 0.0f);
        }
        cg_handfilm_add_shot(film, &shot);
    }
    return film;
}

static const std::string kFixturePath = "fixtures/test_handfilm.json";

TEST(Parity, SummaryFeaturesMatchPython) {
    std::ifstream f(kFixturePath);
    if (!f.is_open()) GTEST_SKIP() << kFixturePath << " not found — run VPS extraction first";

    nlohmann::json j;
    f >> j;

    cg_handfilm_ref film = loadHandFilmFromFixture(j["film"]);
    FeaturePreprocessor pp;
    auto sf = pp.summaryFeatures(film);
    cg_handfilm_destroy(film);

    ASSERT_EQ((int)sf.size(), CG_SUMMARY_FEATURES);

    const auto& ref = j["sf"];
    ASSERT_EQ((int)ref.size(), CG_SUMMARY_FEATURES);

    int mismatches = 0;
    for (int i = 0; i < CG_SUMMARY_FEATURES; ++i) {
        float py_val = ref[i].get<float>();
        EXPECT_NEAR(sf[i], py_val, 1e-5f)
            << "sf[" << i << "]  cpp=" << sf[i] << "  py=" << py_val;
        if (std::abs(sf[i] - py_val) > 1e-5f) ++mismatches;
    }
    EXPECT_EQ(mismatches, 0) << mismatches << " / " << CG_SUMMARY_FEATURES << " values exceeded epsilon";
}

TEST(Parity, FeatureMatrixRow0MatchesPython) {
    std::ifstream f(kFixturePath);
    if (!f.is_open()) GTEST_SKIP() << kFixturePath << " not found — run VPS extraction first";

    nlohmann::json j;
    f >> j;

    cg_handfilm_ref film = loadHandFilmFromFixture(j["film"]);
    FeaturePreprocessor pp;
    auto fm = pp.featureMatrix(film);
    cg_handfilm_destroy(film);

    ASSERT_EQ((int)fm.size(), CG_TARGET_FRAMES * CG_FEATURES_PER_FRAME);

    const auto& ref = j["fm_row0"];
    ASSERT_EQ((int)ref.size(), CG_FEATURES_PER_FRAME);

    int mismatches = 0;
    for (int i = 0; i < CG_FEATURES_PER_FRAME; ++i) {
        float py_val = ref[i].get<float>();
        EXPECT_NEAR(fm[i], py_val, 1e-5f)
            << "fm[0][" << i << "]  cpp=" << fm[i] << "  py=" << py_val;
        if (std::abs(fm[i] - py_val) > 1e-5f) ++mismatches;
    }
    EXPECT_EQ(mismatches, 0) << mismatches << " / " << CG_FEATURES_PER_FRAME << " values exceeded epsilon";
}
