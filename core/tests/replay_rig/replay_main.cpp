// Replay rig for Stage 5 parity testing.
//
// Reads recorded HandFilm JSONs produced by the iOS Training App (stored under
// apps/training-ios/ModelTraining/trainingData/), runs each film through the
// C++ HandGestureRecognizing pipeline, and writes the resulting DetectedGesture
// stream to stdout as JSON (one object per line).
//
// Usage:
//   replay_rig --model   <path/to/gesture_model.tflite>
//              (--gesture-ids <path/to/gesture_ids.json>   # server sidecar, preferred
//               | --registry  <path/to/gestures.json>)     # legacy class list
//              [--pose-model   <path/to/pose_model.tflite>]
//              [--pose-manifest <path/to/pose_manifest.json>]
//              [--holds]           # enable Phase-2 holds mode
//              [--bypass-phase2]   # run Phase-3 unrestricted
//              [--telemetry <path>] # write the per-shot decision record (see below)
//              [--session-out <bundle.cgsession>]  # also record a session (see below)
//              [--pair-tracks]     # with --session-out: two films at a time as hands 0 and 1
//              <handfilm.json> [<handfilm.json> ...]
//
// Output format (JSON Lines):
//   {"film_path":"...", "gesture_id":"...", "gesture_name":"...",
//    "confidence":0.92, "candidate_set_size":3}
//
// If no gesture is detected for a film, a JSON object with gesture_id="" is emitted.
//
// --telemetry writes a second JSON Lines file, in emission order: one
//   {"film_path":..., "type":"frame", "t":..., "gate_open":..., "raw_energy":..., ...}
// row per processed shot, preceded by that shot's decision events
//   {"film_path":..., "type":"event", "event":"cycle_ended", "reason":"low_energy", "t":..., ...}
// Events fired by timer ticks appear between the shots they fall after. The
// gesture output on stdout is unchanged by the flag.
//
// --session-out records the whole run as one .cgsession bundle through the
//   library's session recorder, then reads it back and checks every shot,
//   telemetry row and event count bit-for-bit (exit 5 on a mismatch). Films are
//   laid end to end on one timeline: each is shifted by a whole number of
//   seconds so it starts ~1.5 s after the previous one ended. A whole-second
//   shift is exact in double precision, so every frame interval, and with it
//   every decision, is unchanged. The flush tick after each film is at +1 s
//   rather than +10 s, well past any T_commit / T_min_buffer deadline, so it
//   stays before the next film. Non-absent shots go to track 0.
//
// --pair-tracks takes the films two at a time and plays both at once through
//   the one recognizer, as hands 0 and 1 of a two-hand session: both are
//   shifted to the same start and merged by time. The stdout line for a pair
//   names both films, "a.json + b.json". An odd last film plays alone.
//   Absent frames from either film are recorded as absent frames — an
//   approximation of the device, which reports absence only when no hand is
//   in view.
//
// Compare two runs with:
//   diff <(replay_rig --model m.tflite --gesture-ids ids.json films/*.json) \
//        <expected_output.jsonl>

#include "CameraGestures/CameraGestures.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

using json = nlohmann::json;

// ---------------------------------------------------------------------------
// JSON → cg_handshot
// ---------------------------------------------------------------------------

static cg_handshot shot_from_json(const json& j) {
    cg_handshot s{};
    s.timestamp   = j.value("timestamp", 0.0);
    s.is_absent   = j.value("isAbsent", false) ? 1 : 0;
    std::string hand = j.value("leftOrRight", "unknown");
    if (hand == "left")       s.handedness = CG_HAND_LEFT;
    else if (hand == "right") s.handedness = CG_HAND_RIGHT;
    else                      s.handedness = CG_HAND_UNKNOWN;

    const auto& lms = j["landmarks"];
    for (int i = 0; i < 21 && i < static_cast<int>(lms.size()); ++i) {
        s.landmarks[i].x = lms[i].value("x", 0.0f);
        s.landmarks[i].y = lms[i].value("y", 0.0f);
        s.landmarks[i].z = lms[i].value("z", 0.0f);
    }
    return s;
}

// ---------------------------------------------------------------------------
// Load HandFilm JSON → cg_handfilm_ref
// ---------------------------------------------------------------------------

static cg_handfilm_ref film_from_file(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("cannot open: " + path);
    json j = json::parse(f);

    double start = j.value("startTime", 0.0);
    cg_handfilm_ref film = cg_handfilm_create(start);

    for (const auto& frame : j["frames"]) {
        cg_handshot shot = shot_from_json(frame);
        cg_handfilm_add_shot(film, &shot);
    }
    return film;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

struct Args {
    std::string              model_path;
    std::string              registry_path;
    std::string              gesture_ids_path;
    std::string              pose_model_path;
    std::string              pose_manifest_path;
    bool                     holds          = false;
    bool                     bypass_phase2  = false;
    std::string              telemetry_path;
    std::string              session_path;
    bool                     pair_tracks    = false;
    std::vector<std::string> film_paths;
};

// ---------------------------------------------------------------------------
// --telemetry output, and forwarding to the --session-out recorder
// ---------------------------------------------------------------------------

struct TelemetrySink {
    std::ofstream out;
    std::string   film_path;

    // --session-out: the recorder, and what the rig handed it, kept for the
    // read-back check.
    cg_session_recorder_ref         recorder = nullptr;
    int                             track    = CG_SESSION_ABSENT_TRACK; // last recorded shot's
    std::vector<cg_session_shot>    shots;
    std::vector<cg_frame_telemetry> rows;
    size_t                          events   = 0;
};

static void on_frame_telemetry(void* ctx, const cg_frame_telemetry* r) {
    auto* sink = static_cast<TelemetrySink*>(ctx);
    if (sink->recorder) {
        cg_frame_telemetry row = *r;
        row.track_index = sink->track; // the stamp the recorder applies
        sink->rows.push_back(row);
        cg_session_recorder_append_telemetry(sink->recorder, r);
    }
    if (!sink->out.is_open()) return;
    json j;
    j["film_path"]        = sink->film_path;
    j["type"]             = "frame";
    j["t"]                = r->timestamp;
    j["handedness"]       = r->handedness;
    j["is_absent"]        = r->is_absent != 0;
    j["track_index"]      = r->track_index;
    j["gate_enabled"]     = r->gate_enabled != 0;
    j["gate_open"]        = r->gate_open != 0;
    j["buffer_count"]     = r->buffer_count;
    j["raw_energy"]       = r->raw_energy_valid ? json(r->raw_energy) : json(nullptr);
    if (r->smoothed_energy_valid) {
        j["smoothed_energy"] = r->smoothed_energy;
        j["hold_run_frames"] = r->hold_run_frames;
        j["hold_run_ms"]     = r->hold_run_ms;
    } else {
        j["smoothed_energy"] = nullptr;
    }
    j["commit_deadline"]     = r->commit_deadline;
    j["min_buffer_deadline"] = r->min_buffer_deadline;
    sink->out << j.dump() << "\n";
}

static json candidate_list(const cg_decision_event* e) {
    json ids = json::array();
    for (int i = 0; i < e->n_candidates; ++i) ids.push_back(e->candidate_ids[i]);
    return ids;
}

static void on_decision_event(void* ctx, const cg_decision_event* e) {
    auto* sink = static_cast<TelemetrySink*>(ctx);
    if (sink->recorder) {
        cg_session_recorder_append_decision_event(sink->recorder, e);
        ++sink->events;
    }
    if (!sink->out.is_open()) return;
    json j;
    j["film_path"] = sink->film_path;
    j["type"]      = "event";
    j["event"]     = cg_decision_event_kind_name(e->kind);
    j["reason"]    = cg_decision_event_reason_name(e->kind, e->reason);
    j["t"]         = e->timestamp;
    switch (e->kind) {
    case CG_EVENT_CYCLE_ENDED:
    case CG_EVENT_CYCLE_SKIPPED:
        j["buffer_count"] = e->buffer_count;
        break;
    case CG_EVENT_HOLD_COMPLETED:
        j["pose_id"]         = e->pose_id;
        j["confidence"]      = e->confidence;
        j["accepted"]        = e->accepted != 0;
        j["hold_start_time"] = e->hold_start_time;
        j["rep_shot_time"]   = e->rep_shot_time;
        j["observed_seq"]    = std::vector<int>(e->observed_seq, e->observed_seq + e->n_observed);
        break;
    case CG_EVENT_PHASE3_PREDICTION:
        j["buffer_count"] = e->buffer_count;
        j["restricted"]   = e->restricted != 0;
        j["accepted"]     = e->accepted != 0;
        j["gesture_id"]   = e->gesture_id ? e->gesture_id : "";
        j["confidence"]   = e->confidence;
        if (e->restricted) j["candidates"] = candidate_list(e);
        break;
    case CG_EVENT_COMMIT_FIRED:
        j["candidates"]          = candidate_list(e);
        j["commit_deadline"]     = e->commit_deadline;
        j["min_buffer_deadline"] = e->min_buffer_deadline;
        break;
    default:
        break;
    }
    sink->out << j.dump() << "\n";
}

// ---------------------------------------------------------------------------
// --session-out
// ---------------------------------------------------------------------------

static std::string base_name(const std::string& path) {
    const size_t slash = path.find_last_of('/');
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

// Creates and starts the recorder; the manifest carries the rig's model files,
// the recognizer config, and the film list.
static cg_session_recorder_ref start_session(const Args& a, cg_recognizer_ref rec) {
    cg_session_recorder_ref r = cg_session_recorder_create(a.session_path.c_str());
    if (!r) {
        std::cerr << "Cannot create session recorder for " << a.session_path << "\n";
        return nullptr;
    }
    std::string films;
    for (const auto& p : a.film_paths) films += (films.empty() ? "" : ",") + base_name(p);
    const std::string film_count = std::to_string(a.film_paths.size());
    const std::string mode = std::string(a.holds ? "holds" : "plain")
                           + (a.bypass_phase2 ? "+bypass_phase2" : "")
                           + (a.pair_tracks ? "+pair_tracks" : "");
    cg_session_kv extra[] = {
        {"generator",  "replay_rig"},
        {"mode",       mode.c_str()},
        {"film_count", film_count.c_str()},
        {"films",      films.c_str()},
    };
    cg_session_provenance prov{};
    prov.app_version = "replay_rig";
    prov.extra       = extra;
    prov.n_extra     = 4;

    auto opt = [](const std::string& s) { return s.empty() ? nullptr : s.c_str(); };
    cg_session_model_files files{};
    files.gesture_model = opt(a.model_path);
    files.gesture_ids   = opt(a.gesture_ids_path);
    files.pose_model    = opt(a.pose_model_path);
    files.pose_manifest = opt(a.pose_manifest_path);

    if (!cg_session_recorder_start(r, rec, &prov, &files)) {
        std::cerr << "Cannot start session: " << cg_session_recorder_last_error(r) << "\n";
        cg_session_recorder_destroy(r);
        return nullptr;
    }
    return r;
}

template <typename T>
static bool same_bits(const T& a, const T& b) { return std::memcmp(&a, &b, sizeof(T)) == 0; }

static bool same_shot(const cg_session_shot& a, const cg_session_shot& b) {
    if (!same_bits(a.shot.timestamp, b.shot.timestamp) || !same_bits(a.pts, b.pts)
            || a.shot.handedness != b.shot.handedness || a.shot.is_absent != b.shot.is_absent
            || a.track_index != b.track_index || a.has_pts != b.has_pts) {
        return false;
    }
    for (int i = 0; i < 21; ++i) {
        if (!same_bits(a.shot.landmarks[i].x, b.shot.landmarks[i].x)
                || !same_bits(a.shot.landmarks[i].y, b.shot.landmarks[i].y)
                || !same_bits(a.shot.landmarks[i].z, b.shot.landmarks[i].z)) {
            return false;
        }
    }
    return true;
}

static bool same_row(const cg_frame_telemetry& a, const cg_frame_telemetry& b) {
    return same_bits(a.timestamp, b.timestamp) && same_bits(a.commit_deadline, b.commit_deadline)
        && same_bits(a.min_buffer_deadline, b.min_buffer_deadline)
        && same_bits(a.raw_energy, b.raw_energy) && same_bits(a.smoothed_energy, b.smoothed_energy)
        && same_bits(a.hold_run_ms, b.hold_run_ms) && a.hold_run_frames == b.hold_run_frames
        && a.track_index == b.track_index && a.buffer_count == b.buffer_count
        && a.handedness == b.handedness && a.is_absent == b.is_absent
        && a.gate_enabled == b.gate_enabled && a.gate_open == b.gate_open
        && a.raw_energy_valid == b.raw_energy_valid
        && a.smoothed_energy_valid == b.smoothed_energy_valid;
}

// Reads the bundle back and compares it with what the rig recorded.
static bool verify_session(const std::string& path, const TelemetrySink& sink) {
    char err[512];
    cg_session_reader_ref rd = cg_session_reader_open(path.c_str(), err, sizeof(err));
    if (!rd) {
        std::cerr << "Session read-back failed to open: " << err << "\n";
        return false;
    }
    std::string problem;
    if (!cg_session_reader_is_complete(rd))  problem = "manifest not complete";
    if (cg_session_reader_is_truncated(rd))  problem = "truncated";

    std::map<int, std::vector<cg_session_shot>> by_track;
    for (const auto& s : sink.shots) by_track[s.track_index].push_back(s);
    if (problem.empty() && cg_session_reader_track_count(rd) != static_cast<int>(by_track.size())) {
        problem = "track count differs";
    }
    for (const auto& [track, expected] : by_track) {
        if (!problem.empty()) break;
        std::vector<cg_session_shot> got(expected.size());
        if (cg_session_reader_shot_count(rd, track) != expected.size()
                || cg_session_reader_read_shots(rd, track, 0, got.size(), got.data()) != got.size()) {
            problem = "shot count differs on track " + std::to_string(track);
            break;
        }
        for (size_t i = 0; i < got.size(); ++i) {
            if (!same_shot(got[i], expected[i])) {
                problem = "shot " + std::to_string(i) + " differs on track " + std::to_string(track);
                break;
            }
        }
    }
    if (problem.empty()) {
        std::vector<cg_frame_telemetry> rows(sink.rows.size());
        if (cg_session_reader_telemetry_count(rd) != rows.size()
                || cg_session_reader_read_telemetry(rd, 0, rows.size(), rows.data()) != rows.size()) {
            problem = "telemetry row count differs";
        } else {
            for (size_t i = 0; i < rows.size(); ++i) {
                if (!same_row(rows[i], sink.rows[i])) {
                    problem = "telemetry row " + std::to_string(i) + " differs";
                    break;
                }
            }
        }
    }
    if (problem.empty() && cg_session_reader_event_count(rd) != sink.events) {
        problem = "event count differs";
    }

    if (problem.empty()) {
        std::cerr << "Session " << path << ": " << sink.shots.size() << " shots on "
                  << by_track.size() << " tracks, " << sink.rows.size() << " telemetry rows, "
                  << sink.events << " events; read back bit-for-bit: OK\n";
    } else {
        std::cerr << "Session " << path << " read-back MISMATCH: " << problem << "\n";
    }
    cg_session_reader_close(rd);
    return problem.empty();
}

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model"        && i+1 < argc) { a.model_path        = argv[++i]; }
        else if (arg == "--registry"  && i+1 < argc) { a.registry_path     = argv[++i]; }
        else if (arg == "--gesture-ids" && i+1 < argc) { a.gesture_ids_path = argv[++i]; }
        else if (arg == "--pose-model"   && i+1 < argc) { a.pose_model_path   = argv[++i]; }
        else if (arg == "--pose-manifest" && i+1 < argc) { a.pose_manifest_path = argv[++i]; }
        else if (arg == "--holds")        { a.holds         = true; }
        else if (arg == "--bypass-phase2") { a.bypass_phase2 = true; }
        else if (arg == "--telemetry" && i+1 < argc) { a.telemetry_path = argv[++i]; }
        else if (arg == "--session-out" && i+1 < argc) { a.session_path = argv[++i]; }
        else if (arg == "--pair-tracks")  { a.pair_tracks = true; }
        else if (arg[0] != '-')           { a.film_paths.push_back(arg); }
        else {
            std::cerr << "Unknown argument: " << arg << "\n";
        }
    }
    return a;
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    // Exactly one class-list source.
    if (args.model_path.empty() || args.film_paths.empty()
            || args.registry_path.empty() == args.gesture_ids_path.empty()) {
        std::cerr <<
            "Usage: replay_rig --model <tflite> (--gesture-ids <json> | --registry <json>)\n"
            "                  [--pose-model <tflite>] [--pose-manifest <json>]\n"
            "                  [--holds] [--bypass-phase2] [--telemetry <path>]\n"
            "                  <film.json>...\n";
        return 1;
    }

    // Load gesture model. --gesture-ids is the server's sidecar, a JSON array in
    // the model's output order, passed through verbatim; --registry derives the
    // list locally (legacy, see cg_gesture_model_load).
    cg_gesture_model_ref model = nullptr;
    if (!args.gesture_ids_path.empty()) {
        std::vector<std::string> ids;
        try {
            std::ifstream f(args.gesture_ids_path);
            if (!f.is_open()) throw std::runtime_error("cannot open");
            ids = json::parse(f).get<std::vector<std::string>>();
        } catch (const std::exception& e) {
            std::cerr << "Bad --gesture-ids file " << args.gesture_ids_path
                      << " (expected a JSON array of strings): " << e.what() << "\n";
            return 2;
        }
        std::vector<const char*> id_ptrs;
        for (const auto& id : ids) id_ptrs.push_back(id.c_str());
        model = cg_gesture_model_load_with_ids(
            args.model_path.c_str(), id_ptrs.data(), static_cast<int>(id_ptrs.size()));
    } else {
        model = cg_gesture_model_load(
            args.model_path.c_str(), args.registry_path.c_str());
    }
    if (!model) {
        std::cerr << "Failed to load gesture model from: " << args.model_path << "\n";
        return 2;
    }

    // Load pose model (optional).
    if (!args.pose_model_path.empty() && !args.pose_manifest_path.empty()) {
        if (!cg_gesture_model_load_pose(model,
                args.pose_model_path.c_str(),
                args.pose_manifest_path.c_str())) {
            std::cerr << "Warning: failed to load pose model — holds mode disabled.\n";
        }
    }

    // Build recognizer config.
    cg_recognizer_config cfg = cg_recognizer_default_config();
    cfg.gate_enabled          = 1;
    cfg.holds_enabled         = (args.holds && !args.pose_model_path.empty()) ? 1 : 0;

    cg_recognizer_ref rec = cg_recognizer_create(&cfg, model);
    if (!rec) {
        std::cerr << "Failed to create recognizer.\n";
        cg_gesture_model_destroy(model);
        return 3;
    }

    if (args.bypass_phase2) cg_recognizer_set_bypass_phase2(rec, 1);

    TelemetrySink sink;
    if (!args.telemetry_path.empty()) {
        sink.out.open(args.telemetry_path);
        if (!sink.out.is_open()) {
            std::cerr << "Cannot open --telemetry file: " << args.telemetry_path << "\n";
            cg_recognizer_destroy(rec);
            cg_gesture_model_destroy(model);
            return 4;
        }
    }

    const bool session = !args.session_path.empty();
    if (args.pair_tracks && !session) {
        std::cerr << "--pair-tracks needs --session-out\n";
        cg_recognizer_destroy(rec);
        cg_gesture_model_destroy(model);
        return 1;
    }
    if (session) {
        sink.recorder = start_session(args, rec);
        if (!sink.recorder) {
            cg_recognizer_destroy(rec);
            cg_gesture_model_destroy(model);
            return 4;
        }
    }
    // Installed after the recorder's own: the rig forwards to it, so it can
    // keep what it handed over for the read-back check.
    if (sink.out.is_open() || session) {
        cg_recognizer_set_frame_telemetry_callback(rec, on_frame_telemetry, &sink);
        cg_recognizer_set_decision_event_callback(rec, on_decision_event, &sink);
    }

    // Per-film (or per-pair) replay: push shots one at a time, then check for a
    // result. Because the replay rig is offline (no timer-based T_commit), we
    // tick timers with synthetic timestamps after each shot.
    const size_t step = args.pair_tracks ? 2 : 1;
    double next_start = 0.0; // --session-out: where the next film starts on the timeline
    for (size_t fi = 0; fi < args.film_paths.size(); fi += step) {
        std::vector<std::string> paths(args.film_paths.begin() + fi,
            args.film_paths.begin() + std::min(fi + step, args.film_paths.size()));

        std::vector<std::vector<cg_handshot>> films;
        bool skipped = false;
        for (const auto& path : paths) {
            try {
                cg_handfilm_ref film = film_from_file(path);
                std::vector<cg_handshot> shots(cg_handfilm_shot_count(film));
                for (size_t i = 0; i < shots.size(); ++i) cg_handfilm_get_shot(film, i, &shots[i]);
                cg_handfilm_destroy(film);
                films.push_back(std::move(shots));
            } catch (const std::exception& e) {
                std::cerr << "Skipping " << path << ": " << e.what() << "\n";
                skipped = true;
            }
        }
        if (skipped) continue;
        const std::string label = paths.size() == 1 ? paths[0] : paths[0] + " + " + paths[1];

        // Shots in feed order with their track: film i's hand is track i.
        if (session) {
            double base = next_start;
            if (base == 0.0) {
                for (const auto& f : films) {
                    if (!f.empty() && (base == 0.0 || f.front().timestamp < base)) base = f.front().timestamp;
                }
            }
            for (auto& f : films) {
                if (f.empty()) continue;
                const double shift = std::ceil(base - f.front().timestamp); // whole seconds: exact
                for (auto& shot : f) shot.timestamp += shift;
            }
        }
        std::vector<std::pair<cg_handshot, int>> feed;
        for (size_t t = 0; t < films.size(); ++t) {
            for (const auto& shot : films[t]) feed.push_back({shot, static_cast<int>(t)});
        }
        if (films.size() > 1) { // merge a pair by time; a single film keeps its order
            std::stable_sort(feed.begin(), feed.end(), [](const auto& a, const auto& b) {
                return a.first.timestamp < b.first.timestamp;
            });
        }

        // Detected results for this film.
        struct Result {
            std::string gesture_id;
            std::string gesture_name;
            float       confidence     = 0.0f;
            int         candidate_size = -1;
        };
        std::vector<Result> results;

        auto gesture_cb = [](void* ctx,
                              const cg_gesture_prediction* pred,
                              cg_handfilm_ref /*film*/,
                              int cand_size) {
            auto* v = reinterpret_cast<std::vector<Result>*>(ctx);
            v->push_back({pred->gesture_id, pred->gesture_name,
                          pred->confidence, cand_size});
        };
        cg_recognizer_set_gesture_callback(rec, gesture_cb, &results);

        // Reset gate state between films. A cycle still open from the previous
        // film ends here, so its event is attributed to that film.
        cg_recognizer_reset_gate(rec);
        cg_recognizer_set_gate_enabled(rec, 1);
        sink.film_path = label;

        double last_ts = 0.0;
        for (const auto& [shot, track] : feed) {
            last_ts = shot.timestamp;
            if (sink.recorder) {
                cg_session_shot ss{};
                ss.shot        = shot;
                ss.track_index = shot.is_absent ? CG_SESSION_ABSENT_TRACK : track;
                cg_session_recorder_record_shot(sink.recorder, &ss);
                sink.shots.push_back(ss);
                sink.track = ss.track_index;
            }
            cg_recognizer_process_shot(rec, &shot);
            // Tick timers at each shot timestamp.
            cg_recognizer_tick_timers(rec, shot.timestamp);
        }
        // Final tick after the film to flush pending commits: far future
        // normally, +1 s on a session timeline (see --session-out).
        cg_recognizer_tick_timers(rec, last_ts + (session ? 1.0 : 10.0));
        next_start = last_ts + 1.5;

        // Emit one JSON line per result (or one empty-result line if none).
        if (results.empty()) {
            json out;
            out["film_path"]          = label;
            out["gesture_id"]         = "";
            out["gesture_name"]       = "";
            out["confidence"]         = 0.0;
            out["candidate_set_size"] = -1;
            std::cout << out.dump() << "\n";
        } else {
            for (const auto& r : results) {
                json out;
                out["film_path"]          = label;
                out["gesture_id"]         = r.gesture_id;
                out["gesture_name"]       = r.gesture_name;
                out["confidence"]         = r.confidence;
                out["candidate_set_size"] = r.candidate_size;
                std::cout << out.dump() << "\n";
            }
        }
    }

    int status = 0;
    if (sink.recorder) {
        if (!cg_session_recorder_stop(sink.recorder)) {
            std::cerr << "Session stop failed: " << cg_session_recorder_last_error(sink.recorder) << "\n";
            status = 5;
        }
        cg_session_recorder_destroy(sink.recorder);
        sink.recorder = nullptr;
        if (status == 0 && !verify_session(args.session_path, sink)) status = 5;
    }

    cg_recognizer_destroy(rec);
    cg_gesture_model_destroy(model);
    return status;
}
