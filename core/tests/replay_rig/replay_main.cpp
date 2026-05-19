// Replay rig for Stage 5 parity testing.
//
// Reads recorded HandFilm JSONs produced by the iOS Training App (stored under
// apps/training-ios/ModelTraining/trainingData/), runs each film through the
// C++ HandGestureRecognizing pipeline, and writes the resulting DetectedGesture
// stream to stdout as JSON (one object per line).
//
// Usage:
//   replay_rig --model   <path/to/gesture_model.tflite>
//              --registry <path/to/gestures.json>
//              [--pose-model   <path/to/pose_model.tflite>]
//              [--pose-manifest <path/to/pose_manifest.json>]
//              [--holds]           # enable Phase-2 holds mode
//              [--bypass-phase2]   # run Phase-3 unrestricted
//              <handfilm.json> [<handfilm.json> ...]
//
// Output format (JSON Lines):
//   {"film_path":"...", "gesture_id":"...", "gesture_name":"...",
//    "confidence":0.92, "candidate_set_size":3}
//
// If no gesture is detected for a film, a JSON object with gesture_id="" is emitted.
//
// Compare two runs with:
//   diff <(replay_rig --model m.tflite --registry g.json films/*.json) \
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
    std::string              pose_model_path;
    std::string              pose_manifest_path;
    bool                     holds          = false;
    bool                     bypass_phase2  = false;
    std::vector<std::string> film_paths;
};

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model"        && i+1 < argc) { a.model_path        = argv[++i]; }
        else if (arg == "--registry"  && i+1 < argc) { a.registry_path     = argv[++i]; }
        else if (arg == "--pose-model"   && i+1 < argc) { a.pose_model_path   = argv[++i]; }
        else if (arg == "--pose-manifest" && i+1 < argc) { a.pose_manifest_path = argv[++i]; }
        else if (arg == "--holds")        { a.holds         = true; }
        else if (arg == "--bypass-phase2") { a.bypass_phase2 = true; }
        else if (arg[0] != '-')           { a.film_paths.push_back(arg); }
        else {
            std::cerr << "Unknown argument: " << arg << "\n";
        }
    }
    return a;
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    if (args.model_path.empty() || args.registry_path.empty() || args.film_paths.empty()) {
        std::cerr <<
            "Usage: replay_rig --model <tflite> --registry <json> [--pose-model <tflite>]\n"
            "                  [--pose-manifest <json>] [--holds] [--bypass-phase2]\n"
            "                  <film.json>...\n";
        return 1;
    }

    // Load gesture model.
    cg_gesture_model_ref model = cg_gesture_model_load(
        args.model_path.c_str(), args.registry_path.c_str());
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

        // Reset gate state between films.
        cg_recognizer_reset_gate(rec);
        cg_recognizer_set_gate_enabled(rec, 1);

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
