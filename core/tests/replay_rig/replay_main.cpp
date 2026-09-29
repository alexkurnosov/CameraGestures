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
    std::vector<std::string> film_paths;
};

// ---------------------------------------------------------------------------
// --telemetry output
// ---------------------------------------------------------------------------

struct TelemetrySink {
    std::ofstream out;
    std::string   film_path;
};

static void on_frame_telemetry(void* ctx, const cg_frame_telemetry* r) {
    auto* sink = static_cast<TelemetrySink*>(ctx);
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
        cg_recognizer_set_frame_telemetry_callback(rec, on_frame_telemetry, &sink);
        cg_recognizer_set_decision_event_callback(rec, on_decision_event, &sink);
    }

    // Per-film replay: push shots one at a time, then check for a result.
    // Because the replay rig is offline (no timer-based T_commit), we tick
    // timers with synthetic timestamps after each shot.
    for (const auto& film_path : args.film_paths) {
        cg_handfilm_ref film = nullptr;
        try {
            film = film_from_file(film_path);
        } catch (const std::exception& e) {
            std::cerr << "Skipping " << film_path << ": " << e.what() << "\n";
            continue;
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
        sink.film_path = film_path;

        const size_t n = cg_handfilm_shot_count(film);
        double last_ts = 0.0;
        for (size_t i = 0; i < n; ++i) {
            cg_handshot shot{};
            if (!cg_handfilm_get_shot(film, i, &shot)) continue;
            last_ts = shot.timestamp;
            cg_recognizer_process_shot(rec, &shot);
            // Tick timers at each shot timestamp.
            cg_recognizer_tick_timers(rec, shot.timestamp);
        }
        // Final tick with a far-future timestamp to flush pending commits.
        cg_recognizer_tick_timers(rec, last_ts + 10.0);

        cg_handfilm_destroy(film);

        // Emit one JSON line per result (or one empty-result line if none).
        if (results.empty()) {
            json out;
            out["film_path"]          = film_path;
            out["gesture_id"]         = "";
            out["gesture_name"]       = "";
            out["confidence"]         = 0.0;
            out["candidate_set_size"] = -1;
            std::cout << out.dump() << "\n";
        } else {
            for (const auto& r : results) {
                json out;
                out["film_path"]          = film_path;
                out["gesture_id"]         = r.gesture_id;
                out["gesture_name"]       = r.gesture_name;
                out["confidence"]         = r.confidence;
                out["candidate_set_size"] = r.candidate_size;
                std::cout << out.dump() << "\n";
            }
        }
    }

    cg_recognizer_destroy(rec);
    cg_gesture_model_destroy(model);
    return 0;
}
