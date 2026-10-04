/*
 * test_session.cpp — session capture Stage 2: recorder, manifest, reader.
 *
 * Sessions are written to a scratch directory under testing::TempDir() and
 * read back through the public C API. The recognizer runs without a model:
 * Phase 1 and the telemetry it reports are all a round trip needs.
 */

#include <gtest/gtest.h>
#include "CameraGestures/CameraGestures.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using json   = nlohmann::json;

namespace {

// ---------------------------------------------------------------------------
// Scratch space
// ---------------------------------------------------------------------------

struct ScratchDir {
    fs::path dir;
    ScratchDir() {
        std::string tmpl = (fs::path(testing::TempDir()) / "cgsession_XXXXXX").string();
        char* made = mkdtemp(tmpl.data());
        EXPECT_NE(made, nullptr);
        dir = made ? fs::path(made) : fs::path();
    }
    ~ScratchDir() {
        std::error_code ec;
        fs::remove_all(dir, ec);
    }
    std::string bundle(const std::string& name = "s.cgsession") const {
        return (dir / name).string();
    }
};

std::string readFile(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), {});
}

void writeFile(const fs::path& p, const std::string& content) {
    std::ofstream out(p, std::ios::binary | std::ios::trunc);
    out << content;
}

// ---------------------------------------------------------------------------
// Synthetic shots
// ---------------------------------------------------------------------------

// A hand whose landmarks vary every frame; `base_x` separates two hands.
cg_handshot handShot(double t, float base_x, std::mt19937& rng, cg_handedness hand = CG_HAND_RIGHT) {
    std::uniform_real_distribution<float> jitter(-0.02f, 0.02f);
    cg_handshot s{};
    s.timestamp  = t;
    s.handedness = hand;
    for (int i = 0; i < 21; ++i) {
        s.landmarks[i].x = base_x + 0.01f * i + jitter(rng);
        s.landmarks[i].y = 0.5f + 0.02f * i + jitter(rng);
        s.landmarks[i].z = -0.03f + jitter(rng);
    }
    return s;
}

cg_handshot absentShot(double t) {
    cg_handshot s{};
    s.timestamp = t;
    s.handedness = CG_HAND_RIGHT;
    s.is_absent = 1;
    return s;
}

cg_session_shot sessionShot(const cg_handshot& h, int track, double pts, bool has_pts) {
    cg_session_shot s{};
    s.shot        = h;
    s.track_index = track;
    s.pts         = pts;
    s.has_pts     = has_pts ? 1 : 0;
    return s;
}

// Bit-for-bit comparison, field by field (struct padding is not compared).
template <typename T>
bool sameBits(const T& a, const T& b) { return std::memcmp(&a, &b, sizeof(T)) == 0; }

::testing::AssertionResult shotsEqual(const cg_session_shot& a, const cg_session_shot& b) {
    if (!sameBits(a.shot.timestamp, b.shot.timestamp)) return ::testing::AssertionFailure() << "timestamp";
    for (int i = 0; i < 21; ++i) {
        if (!sameBits(a.shot.landmarks[i].x, b.shot.landmarks[i].x)
                || !sameBits(a.shot.landmarks[i].y, b.shot.landmarks[i].y)
                || !sameBits(a.shot.landmarks[i].z, b.shot.landmarks[i].z)) {
            return ::testing::AssertionFailure() << "landmark " << i;
        }
    }
    if (a.shot.handedness != b.shot.handedness) return ::testing::AssertionFailure() << "handedness";
    if (a.shot.is_absent != b.shot.is_absent)   return ::testing::AssertionFailure() << "is_absent";
    if (a.track_index != b.track_index)         return ::testing::AssertionFailure() << "track_index";
    if (a.has_pts != b.has_pts)                 return ::testing::AssertionFailure() << "has_pts";
    if (!sameBits(a.pts, b.pts))                return ::testing::AssertionFailure() << "pts";
    return ::testing::AssertionSuccess();
}

::testing::AssertionResult rowsEqual(const cg_frame_telemetry& a, const cg_frame_telemetry& b) {
#define CG_CMP(f) if (!sameBits(a.f, b.f)) return ::testing::AssertionFailure() << #f;
    CG_CMP(timestamp) CG_CMP(commit_deadline) CG_CMP(min_buffer_deadline)
    CG_CMP(raw_energy) CG_CMP(smoothed_energy) CG_CMP(hold_run_ms)
    CG_CMP(hold_run_frames) CG_CMP(track_index) CG_CMP(buffer_count)
    CG_CMP(handedness) CG_CMP(is_absent) CG_CMP(gate_enabled) CG_CMP(gate_open)
    CG_CMP(raw_energy_valid) CG_CMP(smoothed_energy_valid)
#undef CG_CMP
    return ::testing::AssertionSuccess();
}

// A stream that opens and closes the gate: bursts of motion, rests, absences.
struct Feed {
    std::vector<cg_session_shot> shots; // in feed order
};

Feed motionFeed(int frames, unsigned seed, bool two_hands) {
    std::mt19937 rng(seed);
    Feed f;
    double t = 1'700'000'000.0;
    for (int i = 0; i < frames; ++i, t += 1.0 / 30.0) {
        const int phase = (i / 25) % 4; // move, rest, move, absent
        if (phase == 3 && i % 25 < 5) {
            f.shots.push_back(sessionShot(absentShot(t), CG_SESSION_ABSENT_TRACK, 0.0, false));
            continue;
        }
        std::mt19937 still_rng(seed); // phase 1: the same pose every frame (zero energy)
        std::mt19937& r = (phase == 1) ? still_rng : rng;
        f.shots.push_back(sessionShot(handShot(t, 0.2f, r, CG_HAND_RIGHT), 0, t - 12.5, i % 7 != 0));
        if (two_hands) {
            f.shots.push_back(sessionShot(handShot(t, 0.7f, r, CG_HAND_LEFT), 1, t - 12.5, true));
        }
    }
    return f;
}

cg_recognizer_config gateConfig() {
    cg_recognizer_config cfg = cg_recognizer_default_config();
    cfg.gate_enabled = 1;
    return cfg;
}

// What a plain recognizer reports for a feed, with the track stamp the
// recorder is expected to apply.
struct Reference {
    std::vector<cg_frame_telemetry> rows;
    std::vector<std::string>        event_names;
};

Reference referenceRun(const Feed& feed) {
    cg_recognizer_config cfg = gateConfig();
    cg_recognizer_ref rec = cg_recognizer_create(&cfg, nullptr);
    struct Ctx { Reference ref; int track = -1; } ctx;
    cg_recognizer_set_frame_telemetry_callback(rec, [](void* c, const cg_frame_telemetry* r) {
        auto* x = static_cast<Ctx*>(c);
        cg_frame_telemetry row = *r;
        row.track_index = x->track;
        x->ref.rows.push_back(row);
    }, &ctx);
    cg_recognizer_set_decision_event_callback(rec, [](void* c, const cg_decision_event* e) {
        static_cast<Ctx*>(c)->ref.event_names.push_back(cg_decision_event_kind_name(e->kind));
    }, &ctx);
    for (const auto& s : feed.shots) {
        ctx.track = s.track_index;
        cg_recognizer_process_shot(rec, &s.shot);
        cg_recognizer_tick_timers(rec, s.shot.timestamp);
    }
    cg_recognizer_destroy(rec);
    return ctx.ref;
}

// Records a feed with the recorder attached to a recognizer.
bool recordFeed(const std::string& bundle, const Feed& feed, int chunk_records,
                const cg_session_provenance* prov = nullptr,
                const cg_session_model_files* files = nullptr,
                bool stop = true) {
    cg_recognizer_config cfg = gateConfig();
    cg_recognizer_ref rec = cg_recognizer_create(&cfg, nullptr);
    cg_session_recorder_ref r = cg_session_recorder_create(bundle.c_str());
    cg_session_recorder_set_chunk_records(r, chunk_records);
    bool ok = cg_session_recorder_start(r, rec, prov, files) == 1;
    EXPECT_TRUE(ok) << cg_session_recorder_last_error(r);
    for (const auto& s : feed.shots) {
        if (!ok) break;
        ok = cg_session_recorder_record_shot(r, &s) == 1;
        cg_recognizer_process_shot(rec, &s.shot);
        cg_recognizer_tick_timers(rec, s.shot.timestamp);
    }
    if (ok && stop) ok = cg_session_recorder_stop(r) == 1;
    if (stop) cg_session_recorder_destroy(r);
    // !stop: leak the recorder deliberately — the process "died" mid-session.
    cg_recognizer_destroy(rec);
    return ok;
}

std::vector<cg_session_shot> readAllShots(cg_session_reader_ref rd, int track) {
    std::vector<cg_session_shot> out(cg_session_reader_shot_count(rd, track));
    const size_t n = cg_session_reader_read_shots(rd, track, 0, out.size(), out.data());
    out.resize(n);
    return out;
}

std::vector<cg_frame_telemetry> readAllRows(cg_session_reader_ref rd) {
    std::vector<cg_frame_telemetry> out(cg_session_reader_telemetry_count(rd));
    const size_t n = cg_session_reader_read_telemetry(rd, 0, out.size(), out.data());
    out.resize(n);
    return out;
}

cg_session_reader_ref openOrFail(const std::string& bundle) {
    char err[512];
    cg_session_reader_ref rd = cg_session_reader_open(bundle.c_str(), err, sizeof(err));
    EXPECT_NE(rd, nullptr) << err;
    return rd;
}

} // namespace

// ---------------------------------------------------------------------------
// Round trip is lossless
// ---------------------------------------------------------------------------

TEST(SessionRoundTrip, ShotsRowsAndEventsReadBackBitForBit) {
    ScratchDir tmp;
    const Feed feed = motionFeed(300, 7, /*two_hands=*/false);
    const Reference ref = referenceRun(feed);
    ASSERT_TRUE(recordFeed(tmp.bundle(), feed, 16));

    cg_session_reader_ref rd = openOrFail(tmp.bundle());
    ASSERT_NE(rd, nullptr);
    EXPECT_EQ(cg_session_reader_format_version(rd), CG_SESSION_FORMAT_VERSION);
    EXPECT_EQ(cg_session_reader_is_complete(rd), 1);
    EXPECT_EQ(cg_session_reader_is_truncated(rd), 0);

    // Shots, per track, in feed order.
    size_t matched = 0;
    for (int track : {CG_SESSION_ABSENT_TRACK, 0}) {
        std::vector<cg_session_shot> expected;
        for (const auto& s : feed.shots) if (s.track_index == track) expected.push_back(s);
        const auto got = readAllShots(rd, track);
        ASSERT_EQ(got.size(), expected.size()) << "track " << track;
        for (size_t i = 0; i < got.size(); ++i) {
            ASSERT_TRUE(shotsEqual(got[i], expected[i])) << "track " << track << " shot " << i;
        }
        matched += got.size();
    }
    EXPECT_EQ(matched, feed.shots.size());

    // Telemetry: exactly what the recognizer reported, stamped with the track.
    const auto rows = readAllRows(rd);
    ASSERT_EQ(rows.size(), ref.rows.size());
    ASSERT_EQ(rows.size(), feed.shots.size());
    for (size_t i = 0; i < rows.size(); ++i) {
        ASSERT_TRUE(rowsEqual(rows[i], ref.rows[i])) << "row " << i;
    }

    // Events: same sequence, each a parseable line.
    ASSERT_EQ(cg_session_reader_event_count(rd), ref.event_names.size());
    ASSERT_GT(ref.event_names.size(), 4u); // the feed opens and closes the gate
    for (size_t i = 0; i < ref.event_names.size(); ++i) {
        const json j = json::parse(cg_session_reader_event_json(rd, i));
        EXPECT_EQ(j["event"], ref.event_names[i]) << "event " << i;
        EXPECT_EQ(j["source"], "core");
    }
    EXPECT_EQ(cg_session_reader_event_json(rd, ref.event_names.size()), nullptr);

    // The recognizer config in effect is in the manifest.
    cg_recognizer_config cfg{};
    int bypass = -1;
    ASSERT_EQ(cg_session_reader_get_config(rd, &cfg, &bypass), 1);
    const cg_recognizer_config want = gateConfig();
    EXPECT_EQ(cfg.gate_enabled, 1);
    EXPECT_EQ(cfg.motion_gate.t_open, want.motion_gate.t_open);
    EXPECT_EQ(cfg.motion_gate.t_close, want.motion_gate.t_close);
    EXPECT_EQ(cfg.motion_gate.k_close_ms, want.motion_gate.k_close_ms);
    EXPECT_EQ(cfg.holds.t_hold, want.holds.t_hold);
    EXPECT_EQ(cfg.holds.t_min_buffer_ms, want.holds.t_min_buffer_ms);
    EXPECT_EQ(cfg.gesture_buffer_size, want.gesture_buffer_size);
    EXPECT_EQ(bypass, 0);

    double t0 = 0, t1 = 0;
    ASSERT_EQ(cg_session_reader_time_range(rd, &t0, &t1), 1);
    EXPECT_EQ(t0, feed.shots.front().shot.timestamp);
    EXPECT_EQ(t1, feed.shots.back().shot.timestamp);

    EXPECT_TRUE(fs::exists(fs::path(tmp.bundle()) / "labels.jsonl"));
    EXPECT_EQ(fs::file_size(fs::path(tmp.bundle()) / "labels.jsonl"), 0u);
    cg_session_reader_close(rd);
}

TEST(SessionRoundTrip, StopDetachesFromTheRecognizer) {
    ScratchDir tmp;
    cg_recognizer_config cfg = gateConfig();
    cg_recognizer_ref rec = cg_recognizer_create(&cfg, nullptr);
    cg_session_recorder_ref r = cg_session_recorder_create(tmp.bundle().c_str());
    ASSERT_EQ(cg_session_recorder_start(r, rec, nullptr, nullptr), 1);
    std::mt19937 rng(1);
    const cg_handshot h = handShot(100.0, 0.3f, rng);
    const cg_session_shot s = sessionShot(h, 0, 0.0, false);
    cg_session_recorder_record_shot(r, &s);
    cg_recognizer_process_shot(rec, &h);
    ASSERT_EQ(cg_session_recorder_stop(r), 1);
    EXPECT_EQ(cg_session_recorder_is_recording(r), 0);
    cg_session_recorder_destroy(r);

    // The recorder is gone; the recognizer must no longer call into it.
    cg_recognizer_process_shot(rec, &h);
    cg_recognizer_destroy(rec);

    cg_session_reader_ref rd = openOrFail(tmp.bundle());
    ASSERT_NE(rd, nullptr);
    EXPECT_EQ(cg_session_reader_telemetry_count(rd), 1u);
    cg_session_reader_close(rd);
}

// ---------------------------------------------------------------------------
// Manifest hashes and provenance
// ---------------------------------------------------------------------------

TEST(SessionManifest, HashesMatchKnownVectorsAndShasum) {
    ScratchDir tmp;
    const fs::path models = tmp.dir / "models";
    fs::create_directory(models);

    // FIPS 180-4 test vectors, plus a random file checked against shasum.
    writeFile(models / "gesture_model.tflite", "abc");
    writeFile(models / "gesture_ids.json", "");
    writeFile(models / "pose_model.tflite", std::string(1'000'000, 'a'));
    writeFile(models / "pose_manifest.json",
              "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq");
    std::string random_bytes(200'003, '\0');
    std::mt19937 rng(42);
    for (auto& c : random_bytes) c = static_cast<char>(rng() & 0xff);
    writeFile(models / "preprocessor.js", random_bytes);

    const std::string p_gm = (models / "gesture_model.tflite").string();
    const std::string p_gi = (models / "gesture_ids.json").string();
    const std::string p_pm = (models / "pose_model.tflite").string();
    const std::string p_pf = (models / "pose_manifest.json").string();
    const std::string p_pp = (models / "preprocessor.js").string();
    cg_session_model_files files{p_gm.c_str(), p_gi.c_str(), p_pm.c_str(), p_pf.c_str(), p_pp.c_str()};

    cg_session_kv extra[] = {{"build", "debug"}, {"note", "unit test"}};
    cg_session_provenance prov{};
    prov.app_version  = "2.3 (41)";
    prov.device_model = "iPhone15,2";
    prov.os_version   = "iOS 18.1";
    prov.camera       = {"AVCaptureSessionPreset640x480", 640, 480, 30.0, CG_CAMERA_POSITION_FRONT};
    prov.extra        = extra;
    prov.n_extra      = 2;

    ASSERT_TRUE(recordFeed(tmp.bundle(), motionFeed(10, 3, false), 64, &prov, &files));

    cg_session_reader_ref rd = openOrFail(tmp.bundle());
    ASSERT_NE(rd, nullptr);
    cg_session_file_records rec{};
    ASSERT_EQ(cg_session_reader_get_model_files(rd, &rec), 1);

    EXPECT_STREQ(rec.gesture_model.sha256,
                 "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    EXPECT_STREQ(rec.gesture_ids.sha256,
                 "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    EXPECT_STREQ(rec.pose_model.sha256,
                 "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    EXPECT_STREQ(rec.pose_manifest.sha256,
                 "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    EXPECT_EQ(rec.pose_model.size, 1'000'000u);
    EXPECT_EQ(rec.preprocessor.size, 200'003u);

    // File names only — never the path.
    EXPECT_STREQ(rec.gesture_model.name, "gesture_model.tflite");
    EXPECT_STREQ(rec.preprocessor.name, "preprocessor.js");
    EXPECT_EQ(std::string(cg_session_reader_manifest_json(rd)).find(models.string()),
              std::string::npos);

    // The random file against the system's shasum, when it is available.
    FILE* p = popen(("shasum -a 256 '" + p_pp + "' 2>/dev/null").c_str(), "r");
    char line[256] = {0};
    const bool have_shasum = p && std::fgets(line, sizeof(line), p) != nullptr;
    if (p) pclose(p);
    if (have_shasum) {
        EXPECT_EQ(std::string(line).substr(0, 64), std::string(rec.preprocessor.sha256));
    } else {
        ADD_FAILURE() << "shasum not available to cross-check the random file";
    }

    cg_session_provenance got{};
    ASSERT_EQ(cg_session_reader_get_provenance(rd, &got), 1);
    EXPECT_STREQ(got.app_version, "2.3 (41)");
    EXPECT_STREQ(got.device_model, "iPhone15,2");
    EXPECT_STREQ(got.os_version, "iOS 18.1");
    EXPECT_STREQ(got.camera.preset, "AVCaptureSessionPreset640x480");
    EXPECT_EQ(got.camera.width, 640);
    EXPECT_EQ(got.camera.height, 480);
    EXPECT_EQ(got.camera.fps, 30.0);
    EXPECT_EQ(got.camera.position, CG_CAMERA_POSITION_FRONT);
    ASSERT_EQ(got.n_extra, 2);
    for (int i = 0; i < 2; ++i) {
        const std::string k = got.extra[i].key;
        EXPECT_STREQ(got.extra[i].value, k == "build" ? "debug" : "unit test");
    }
    EXPECT_STREQ(cg_session_reader_library_version(rd), cg_version());
    EXPECT_GT(cg_session_reader_created_at(rd), 1.7e9);
    cg_session_reader_close(rd);
}

TEST(SessionManifest, UnknownProvenanceAndNoModelsAreNull) {
    ScratchDir tmp;
    ASSERT_TRUE(recordFeed(tmp.bundle(), motionFeed(5, 3, false), 64));
    cg_session_reader_ref rd = openOrFail(tmp.bundle());
    ASSERT_NE(rd, nullptr);
    cg_session_provenance got{};
    cg_session_reader_get_provenance(rd, &got);
    EXPECT_EQ(got.app_version, nullptr);
    EXPECT_EQ(got.camera.preset, nullptr);
    EXPECT_EQ(got.camera.position, CG_CAMERA_POSITION_UNKNOWN);
    EXPECT_EQ(got.n_extra, 0);
    cg_session_file_records rec{};
    cg_session_reader_get_model_files(rd, &rec);
    EXPECT_EQ(rec.gesture_model.name, nullptr);
    EXPECT_EQ(rec.preprocessor.sha256, nullptr);
    cg_session_reader_close(rd);
}

TEST(SessionManifest, UnreadableModelFileFailsStartAndCreatesNothing) {
    ScratchDir tmp;
    const std::string missing = (tmp.dir / "nope.tflite").string();
    cg_session_model_files files{missing.c_str(), nullptr, nullptr, nullptr, nullptr};
    cg_session_recorder_ref r = cg_session_recorder_create(tmp.bundle().c_str());
    EXPECT_EQ(cg_session_recorder_start(r, nullptr, nullptr, &files), 0);
    EXPECT_NE(std::string(cg_session_recorder_last_error(r)).find("gesture_model"), std::string::npos);
    EXPECT_FALSE(fs::exists(tmp.bundle()));
    cg_session_recorder_destroy(r);
}

// ---------------------------------------------------------------------------
// Truncated bundles recover the valid prefix
// ---------------------------------------------------------------------------

TEST(SessionTruncation, KilledRecorderLeavesEveryFlushedChunk) {
    ScratchDir tmp;
    const Feed feed = motionFeed(50, 9, false); // includes a few absent frames
    // Never stopped: the manifest stays "recording", unflushed records are lost.
    ASSERT_TRUE(recordFeed(tmp.bundle(), feed, 8, nullptr, nullptr, /*stop=*/false));

    cg_session_reader_ref rd = openOrFail(tmp.bundle());
    ASSERT_NE(rd, nullptr);
    EXPECT_EQ(cg_session_reader_is_complete(rd), 0);
    EXPECT_EQ(cg_session_reader_is_truncated(rd), 0); // clean chunk boundaries
    const size_t rows = cg_session_reader_telemetry_count(rd);
    EXPECT_EQ(rows % 8, 0u);
    EXPECT_GE(rows, feed.shots.size() - 8); // lost at most one chunk
    EXPECT_LT(rows, feed.shots.size());

    // What survived is an exact prefix of the feed.
    const auto got = readAllShots(rd, 0);
    std::vector<cg_session_shot> expected;
    for (const auto& s : feed.shots) if (s.track_index == 0) expected.push_back(s);
    ASSERT_LE(got.size(), expected.size());
    for (size_t i = 0; i < got.size(); ++i) ASSERT_TRUE(shotsEqual(got[i], expected[i]));
    cg_session_reader_close(rd);
}

TEST(SessionTruncation, TornLastChunkIsDroppedAndReported) {
    ScratchDir tmp;
    const Feed feed = motionFeed(50, 11, false);
    ASSERT_TRUE(recordFeed(tmp.bundle(), feed, 8));

    const fs::path tel = fs::path(tmp.bundle()) / "telemetry.bin";
    const size_t full = fs::file_size(tel);
    fs::resize_file(tel, full - 10); // cut inside the last chunk's payload
    const fs::path ev = fs::path(tmp.bundle()) / "events.jsonl";
    std::string events = readFile(ev);
    const size_t n_events = std::count(events.begin(), events.end(), '\n');
    writeFile(ev, events.substr(0, events.size() - 5)); // cut the last line

    cg_session_reader_ref rd = openOrFail(tmp.bundle());
    ASSERT_NE(rd, nullptr);
    EXPECT_EQ(cg_session_reader_is_truncated(rd), 1);
    EXPECT_EQ(cg_session_reader_is_complete(rd), 1); // the manifest still says so
    const size_t last_chunk = feed.shots.size() % 8 ? feed.shots.size() % 8 : 8;
    EXPECT_EQ(cg_session_reader_telemetry_count(rd), feed.shots.size() - last_chunk);
    EXPECT_EQ(cg_session_reader_event_count(rd), n_events - 1);
    cg_session_reader_close(rd);
}

TEST(SessionTruncation, CorruptChunkEndsTheReadablePrefix) {
    ScratchDir tmp;
    const Feed feed = motionFeed(64, 13, false);
    ASSERT_TRUE(recordFeed(tmp.bundle(), feed, 8));

    // Flip one byte inside the third chunk's payload (header 16, chunk 16 + 8*56).
    const fs::path tel = fs::path(tmp.bundle()) / "telemetry.bin";
    std::string bytes = readFile(tel);
    const size_t chunk_bytes = 16 + 8 * 56;
    bytes[16 + 2 * chunk_bytes + 16 + 100] ^= 0x01;
    writeFile(tel, bytes);

    cg_session_reader_ref rd = openOrFail(tmp.bundle());
    ASSERT_NE(rd, nullptr);
    EXPECT_EQ(cg_session_reader_is_truncated(rd), 1);
    EXPECT_EQ(cg_session_reader_telemetry_count(rd), 16u);
    cg_session_reader_close(rd);
}

// ---------------------------------------------------------------------------
// Two hands: separate tracks on a shared time axis
// ---------------------------------------------------------------------------

TEST(SessionTracks, TwoHandsStaySeparateOnASharedTimeAxis) {
    ScratchDir tmp;
    const Feed feed = motionFeed(200, 21, /*two_hands=*/true);
    ASSERT_TRUE(recordFeed(tmp.bundle(), feed, 16));

    cg_session_reader_ref rd = openOrFail(tmp.bundle());
    ASSERT_NE(rd, nullptr);
    ASSERT_EQ(cg_session_reader_track_count(rd), 3);
    EXPECT_EQ(cg_session_reader_track_at(rd, 0), CG_SESSION_ABSENT_TRACK);
    EXPECT_EQ(cg_session_reader_track_at(rd, 1), 0);
    EXPECT_EQ(cg_session_reader_track_at(rd, 2), 1);

    std::vector<double> all_times;
    for (int track : {0, 1}) {
        const auto shots = readAllShots(rd, track);
        ASSERT_FALSE(shots.empty());
        for (size_t i = 0; i < shots.size(); ++i) {
            // Hand 0 sits near x = 0.2, hand 1 near x = 0.7: no interleaving.
            const float wrist_x = shots[i].shot.landmarks[0].x;
            EXPECT_EQ(shots[i].track_index, track);
            EXPECT_TRUE(track == 0 ? wrist_x < 0.45f : wrist_x > 0.45f) << "track " << track;
            EXPECT_EQ(shots[i].shot.handedness, track == 0 ? CG_HAND_RIGHT : CG_HAND_LEFT);
            if (i > 0) EXPECT_GT(shots[i].shot.timestamp, shots[i - 1].shot.timestamp);
            all_times.push_back(shots[i].shot.timestamp);
        }
    }
    // Both tracks carry the same frame times: one clock.
    const auto t0 = readAllShots(rd, 0), t1 = readAllShots(rd, 1);
    ASSERT_EQ(t0.size(), t1.size());
    for (size_t i = 0; i < t0.size(); ++i) {
        EXPECT_EQ(t0[i].shot.timestamp, t1[i].shot.timestamp);
    }

    // Telemetry rows follow the feed and carry its track.
    const auto rows = readAllRows(rd);
    ASSERT_EQ(rows.size(), feed.shots.size());
    for (size_t i = 0; i < rows.size(); ++i) {
        EXPECT_EQ(rows[i].track_index, feed.shots[i].track_index) << "row " << i;
        EXPECT_EQ(rows[i].timestamp, feed.shots[i].shot.timestamp);
    }

    double lo = 0, hi = 0;
    ASSERT_EQ(cg_session_reader_time_range(rd, &lo, &hi), 1);
    EXPECT_EQ(lo, feed.shots.front().shot.timestamp);
    EXPECT_EQ(hi, feed.shots.back().shot.timestamp);
    cg_session_reader_close(rd);
}

TEST(SessionTracks, AbsentShotsGoToTheAbsentTrackAndPresentOnesNeedAnIndex) {
    ScratchDir tmp;
    cg_session_recorder_ref r = cg_session_recorder_create(tmp.bundle().c_str());
    ASSERT_EQ(cg_session_recorder_start(r, nullptr, nullptr, nullptr), 1);

    const cg_session_shot absent = sessionShot(absentShot(5.0), 3, 0.0, false);
    EXPECT_EQ(cg_session_recorder_record_shot(r, &absent), 1);
    std::mt19937 rng(2);
    const cg_session_shot unindexed = sessionShot(handShot(5.1, 0.3f, rng), -1, 0.0, false);
    EXPECT_EQ(cg_session_recorder_record_shot(r, &unindexed), 0);
    EXPECT_NE(std::string(cg_session_recorder_last_error(r)).find("track_index"), std::string::npos);
    ASSERT_EQ(cg_session_recorder_stop(r), 1);
    cg_session_recorder_destroy(r);

    cg_session_reader_ref rd = openOrFail(tmp.bundle());
    ASSERT_NE(rd, nullptr);
    ASSERT_EQ(cg_session_reader_track_count(rd), 1);
    EXPECT_EQ(cg_session_reader_track_at(rd, 0), CG_SESSION_ABSENT_TRACK);
    cg_session_shot got{};
    ASSERT_EQ(cg_session_reader_read_shots(rd, CG_SESSION_ABSENT_TRACK, 0, 1, &got), 1u);
    EXPECT_EQ(got.track_index, CG_SESSION_ABSENT_TRACK);
    EXPECT_EQ(got.shot.is_absent, 1);
    EXPECT_EQ(cg_session_reader_get_config(rd, nullptr, nullptr), 0); // no recognizer
    cg_session_reader_close(rd);
}

TEST(SessionTracks, StartRefusesAnExistingBundle) {
    ScratchDir tmp;
    fs::create_directory(tmp.bundle());
    cg_session_recorder_ref r = cg_session_recorder_create(tmp.bundle().c_str());
    EXPECT_EQ(cg_session_recorder_start(r, nullptr, nullptr, nullptr), 0);
    EXPECT_NE(std::string(cg_session_recorder_last_error(r)).find("exists"), std::string::npos);
    cg_session_recorder_destroy(r);
}

// ---------------------------------------------------------------------------
// Seek: random access matches a sequential read
// ---------------------------------------------------------------------------

TEST(SessionSeek, RandomAccessMatchesSequentialRead) {
    ScratchDir tmp;
    const Feed feed = motionFeed(600, 31, /*two_hands=*/true); // ~1150 shots, uneven chunks
    ASSERT_TRUE(recordFeed(tmp.bundle(), feed, 16));

    cg_session_reader_ref rd = openOrFail(tmp.bundle());
    ASSERT_NE(rd, nullptr);
    const auto seq_shots = readAllShots(rd, 1);
    const auto seq_rows  = readAllRows(rd);
    ASSERT_GT(seq_shots.size(), 500u);

    std::mt19937 rng(99);
    for (int k = 0; k < 100; ++k) {
        // Shots: random range, possibly running past the end.
        const size_t first = rng() % (seq_shots.size() + 5);
        const size_t count = 1 + rng() % 40;
        std::vector<cg_session_shot> got(count);
        const size_t n = cg_session_reader_read_shots(rd, 1, first, count, got.data());
        const size_t want = first >= seq_shots.size() ? 0 : std::min(count, seq_shots.size() - first);
        ASSERT_EQ(n, want) << "offset " << first;
        for (size_t i = 0; i < n; ++i) ASSERT_TRUE(shotsEqual(got[i], seq_shots[first + i]));

        // Telemetry.
        const size_t rfirst = rng() % seq_rows.size();
        std::vector<cg_frame_telemetry> rows(count);
        const size_t rn = cg_session_reader_read_telemetry(rd, rfirst, count, rows.data());
        ASSERT_EQ(rn, std::min(count, seq_rows.size() - rfirst));
        for (size_t i = 0; i < rn; ++i) ASSERT_TRUE(rowsEqual(rows[i], seq_rows[rfirst + i]));

        // Seek by time equals a linear lower bound. Telemetry has two rows per
        // frame time, so this also checks it lands on the first of equal times.
        const double t0 = seq_rows.front().timestamp, t1 = seq_rows.back().timestamp;
        const double t = t0 - 1.0 + (t1 - t0 + 2.0) * (rng() % 10000) / 10000.0;
        const double exact = seq_rows[rng() % seq_rows.size()].timestamp;
        for (double q : {t, exact}) {
            size_t lin_rows = 0;
            while (lin_rows < seq_rows.size() && seq_rows[lin_rows].timestamp < q) ++lin_rows;
            EXPECT_EQ(cg_session_reader_find_telemetry(rd, q), lin_rows) << "t " << q;
            size_t lin_shots = 0;
            while (lin_shots < seq_shots.size() && seq_shots[lin_shots].shot.timestamp < q) ++lin_shots;
            EXPECT_EQ(cg_session_reader_find_shot(rd, 1, q), lin_shots) << "t " << q;
        }
    }
    cg_session_reader_close(rd);
}

// ---------------------------------------------------------------------------
// Format version is enforced
// ---------------------------------------------------------------------------

TEST(SessionFormatVersion, FutureManifestVersionIsRejectedClearly) {
    ScratchDir tmp;
    ASSERT_TRUE(recordFeed(tmp.bundle(), motionFeed(20, 5, false), 8));
    const fs::path mf = fs::path(tmp.bundle()) / "manifest.json";
    json j = json::parse(readFile(mf));
    j["format_version"] = CG_SESSION_FORMAT_VERSION + 1;
    writeFile(mf, j.dump());

    char err[512] = {0};
    EXPECT_EQ(cg_session_reader_open(tmp.bundle().c_str(), err, sizeof(err)), nullptr);
    const std::string msg = err;
    EXPECT_NE(msg.find("format version " + std::to_string(CG_SESSION_FORMAT_VERSION + 1)),
              std::string::npos) << msg;
    EXPECT_NE(msg.find("newer"), std::string::npos) << msg;
}

TEST(SessionFormatVersion, FutureStreamVersionIsRejectedClearly) {
    ScratchDir tmp;
    ASSERT_TRUE(recordFeed(tmp.bundle(), motionFeed(20, 5, false), 8));
    const fs::path tel = fs::path(tmp.bundle()) / "telemetry.bin";
    std::string bytes = readFile(tel);
    bytes[4] = static_cast<char>(CG_SESSION_FORMAT_VERSION + 1); // u16 LE at offset 4
    writeFile(tel, bytes);

    char err[512] = {0};
    EXPECT_EQ(cg_session_reader_open(tmp.bundle().c_str(), err, sizeof(err)), nullptr);
    EXPECT_NE(std::string(err).find("format version"), std::string::npos) << err;
}

TEST(SessionFormatVersion, MissingManifestIsAnError) {
    ScratchDir tmp;
    fs::create_directory(tmp.bundle());
    char err[512] = {0};
    EXPECT_EQ(cg_session_reader_open(tmp.bundle().c_str(), err, sizeof(err)), nullptr);
    EXPECT_NE(std::string(err).find("manifest"), std::string::npos) << err;
}

// ---------------------------------------------------------------------------
// Committed fixtures keep opening (tests run from core/tests/)
// ---------------------------------------------------------------------------

TEST(SessionFixtures, CommittedBundlesOpenCompleteAndMatchTheirCounts) {
    const fs::path dir = "fixtures/sessions";
    ASSERT_TRUE(fs::is_directory(dir)) << "run from core/tests/";
    int bundles = 0;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.path().extension() != ".cgsession") continue;
        ++bundles;
        SCOPED_TRACE(entry.path().string());
        cg_session_reader_ref rd = openOrFail(entry.path().string());
        ASSERT_NE(rd, nullptr);
        EXPECT_EQ(cg_session_reader_is_complete(rd), 1);
        EXPECT_EQ(cg_session_reader_is_truncated(rd), 0);

        const json counts = json::parse(cg_session_reader_manifest_json(rd))["counts"];
        EXPECT_EQ(cg_session_reader_telemetry_count(rd), counts["telemetry"].get<size_t>());
        EXPECT_EQ(cg_session_reader_event_count(rd), counts["events"].get<size_t>());
        size_t shots = 0;
        for (int i = 0; i < cg_session_reader_track_count(rd); ++i) {
            const int track = cg_session_reader_track_at(rd, i);
            const size_t n = cg_session_reader_shot_count(rd, track);
            EXPECT_EQ(n, counts["shots"][std::to_string(track)].get<size_t>());
            EXPECT_EQ(readAllShots(rd, track).size(), n);
            shots += n;
        }
        EXPECT_EQ(readAllRows(rd).size(), shots); // one row per shot
        cg_session_reader_close(rd);
    }
    EXPECT_GE(bundles, 1);
}

// ---------------------------------------------------------------------------
// Binding events
// ---------------------------------------------------------------------------

TEST(SessionBindingEvents, AppendedLinesReadBackWithTheirFields) {
    ScratchDir tmp;
    cg_session_recorder_ref rec = cg_session_recorder_create(tmp.bundle().c_str());
    ASSERT_EQ(cg_session_recorder_start(rec, nullptr, nullptr, nullptr), 1);

    EXPECT_EQ(cg_session_recorder_append_binding_event(rec, 10.5, "cooldown_started",
                                                       "{\"duration\": 1.0}"), 1);
    // The reserved members cannot be overridden by the fields.
    EXPECT_EQ(cg_session_recorder_append_binding_event(rec, 11.0, "gesture_suppressed",
        "{\"gesture_id\": \"wave\", \"source\": \"core\", \"t\": 0}"), 1);
    EXPECT_EQ(cg_session_recorder_append_binding_event(rec, 12.0, "status_changed", nullptr), 1);

    EXPECT_EQ(cg_session_recorder_append_binding_event(rec, 13.0, "bad", "[1, 2]"), 0);
    EXPECT_EQ(cg_session_recorder_append_binding_event(rec, 13.0, "bad", "{not json"), 0);
    ASSERT_EQ(cg_session_recorder_stop(rec), 1);
    cg_session_recorder_destroy(rec);

    cg_session_reader_ref rd = openOrFail(tmp.bundle());
    ASSERT_NE(rd, nullptr);
    ASSERT_EQ(cg_session_reader_event_count(rd), 3u);

    const json first = json::parse(cg_session_reader_event_json(rd, 0));
    EXPECT_EQ(first["source"], "binding");
    EXPECT_EQ(first["event"], "cooldown_started");
    EXPECT_EQ(first["t"], 10.5);
    EXPECT_EQ(first["duration"], 1.0);

    const json second = json::parse(cg_session_reader_event_json(rd, 1));
    EXPECT_EQ(second["source"], "binding");
    EXPECT_EQ(second["t"], 11.0);
    EXPECT_EQ(second["gesture_id"], "wave");

    EXPECT_EQ(json::parse(cg_session_reader_event_json(rd, 2))["event"], "status_changed");
    cg_session_reader_close(rd);
}
