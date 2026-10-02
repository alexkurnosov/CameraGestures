#pragma once
#include "CameraGestures/SessionCapture.h"
#include <nlohmann/json.hpp>

// manifest.json and events.jsonl encoding, shared by the recorder and reader
// so the two cannot drift. Key names follow the C struct field names.
//
// manifest.json, format version 1:
//   {
//     "format": "cgsession", "format_version": 1,
//     "status": "recording" | "complete",
//     "library_version": "...", "created_at": <epoch s>, "stopped_at": <epoch s> | null,
//     "chunk_records": 64,
//     "provenance": { "app_version", "device_model", "os_version",
//                     "camera": { "preset", "width", "height", "fps", "position" },
//                     "extra": { "<key>": "<value>", ... } },
//     "model_files": { "gesture_model": { "name", "sha256", "size" } | null, ... },
//     "config": { ...cg_recognizer_config..., "bypass_phase2": bool } | null,
//     "files": { "telemetry": "telemetry.bin", "events": "events.jsonl",
//                "labels": "labels.jsonl", "shots_dir": "shots" },
//     // written by stop:
//     "counts": { "telemetry": n, "events": n, "shots": { "<track>": n, ... } },
//     "time_range": [start, end] | null
//   }

namespace cgsession {

nlohmann::json configToJson(const cg_recognizer_config& c, bool bypass_phase2);
// Returns false if a required key is missing or mistyped.
bool configFromJson(const nlohmann::json& j, cg_recognizer_config* c, bool* bypass_phase2);

const char* cameraPositionName(int position);
int         cameraPositionFromName(const std::string& name);

// One events.jsonl line (without the newline) for a decision event.
std::string decisionEventLine(const cg_decision_event& e, int track_index);

} // namespace cgsession
