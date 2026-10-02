#include "SessionManifest.hpp"
#include <vector>

using json = nlohmann::json;

namespace cgsession {

json configToJson(const cg_recognizer_config& c, bool bypass_phase2) {
    json j;
    j["gate_enabled"] = c.gate_enabled != 0;
    j["motion_gate"]  = {
        {"t_open",      c.motion_gate.t_open},
        {"k_open_ms",   c.motion_gate.k_open_ms},
        {"t_close",     c.motion_gate.t_close},
        {"k_close_ms",  c.motion_gate.k_close_ms},
        {"cooldown_ms", c.motion_gate.cooldown_ms},
    };
    j["gesture_buffer_size"] = c.gesture_buffer_size;
    j["holds_enabled"]       = c.holds_enabled != 0;
    j["holds"] = {
        {"t_hold",                c.holds.t_hold},
        {"k_hold_ms",             c.holds.k_hold_ms},
        {"smooth_k_ms",           c.holds.smooth_k_ms},
        {"t_commit_ms",           c.holds.t_commit_ms},
        {"t_min_buffer_ms",       c.holds.t_min_buffer_ms},
        {"tau_pose_confidence",   c.holds.tau_pose_confidence},
        {"tau_phase3_confidence", c.holds.tau_phase3_confidence},
    };
    j["confidence_threshold"]        = c.confidence_threshold;
    j["retain_landmarks_for_review"] = c.retain_landmarks_for_review != 0;
    j["bypass_phase2"]               = bypass_phase2;
    return j;
}

bool configFromJson(const json& j, cg_recognizer_config* c, bool* bypass_phase2) {
    try {
        cg_recognizer_config out{};
        out.gate_enabled               = j.at("gate_enabled").get<bool>() ? 1 : 0;
        const json& g                  = j.at("motion_gate");
        out.motion_gate.t_open         = g.at("t_open").get<float>();
        out.motion_gate.k_open_ms      = g.at("k_open_ms").get<double>();
        out.motion_gate.t_close        = g.at("t_close").get<float>();
        out.motion_gate.k_close_ms     = g.at("k_close_ms").get<double>();
        out.motion_gate.cooldown_ms    = g.at("cooldown_ms").get<double>();
        out.gesture_buffer_size        = j.at("gesture_buffer_size").get<int>();
        out.holds_enabled              = j.at("holds_enabled").get<bool>() ? 1 : 0;
        const json& h                  = j.at("holds");
        out.holds.t_hold               = h.at("t_hold").get<float>();
        out.holds.k_hold_ms            = h.at("k_hold_ms").get<double>();
        out.holds.smooth_k_ms          = h.at("smooth_k_ms").get<double>();
        out.holds.t_commit_ms          = h.at("t_commit_ms").get<double>();
        out.holds.t_min_buffer_ms      = h.at("t_min_buffer_ms").get<double>();
        out.holds.tau_pose_confidence  = h.at("tau_pose_confidence").get<float>();
        out.holds.tau_phase3_confidence = h.at("tau_phase3_confidence").get<float>();
        out.confidence_threshold       = j.at("confidence_threshold").get<float>();
        out.retain_landmarks_for_review = j.at("retain_landmarks_for_review").get<bool>() ? 1 : 0;
        *c             = out;
        *bypass_phase2 = j.value("bypass_phase2", false);
        return true;
    } catch (const json::exception&) {
        return false;
    }
}

const char* cameraPositionName(int position) {
    switch (position) {
    case CG_CAMERA_POSITION_FRONT:    return "front";
    case CG_CAMERA_POSITION_BACK:     return "back";
    case CG_CAMERA_POSITION_EXTERNAL: return "external";
    }
    return "unknown";
}

int cameraPositionFromName(const std::string& name) {
    if (name == "front")    return CG_CAMERA_POSITION_FRONT;
    if (name == "back")     return CG_CAMERA_POSITION_BACK;
    if (name == "external") return CG_CAMERA_POSITION_EXTERNAL;
    return CG_CAMERA_POSITION_UNKNOWN;
}

static json candidateList(const cg_decision_event& e) {
    json ids = json::array();
    for (int i = 0; i < e.n_candidates; ++i) ids.push_back(e.candidate_ids[i]);
    return ids;
}

// Same fields per kind as replay_rig --telemetry, plus "track".
std::string decisionEventLine(const cg_decision_event& e, int track_index) {
    json j;
    j["t"]      = e.timestamp;
    j["track"]  = track_index;
    j["source"] = "core";
    j["event"]  = cg_decision_event_kind_name(e.kind);
    j["reason"] = cg_decision_event_reason_name(e.kind, e.reason);
    switch (e.kind) {
    case CG_EVENT_CYCLE_ENDED:
    case CG_EVENT_CYCLE_SKIPPED:
        j["buffer_count"] = e.buffer_count;
        break;
    case CG_EVENT_HOLD_COMPLETED:
        j["pose_id"]         = e.pose_id;
        j["confidence"]      = e.confidence;
        j["accepted"]        = e.accepted != 0;
        j["hold_start_time"] = e.hold_start_time;
        j["rep_shot_time"]   = e.rep_shot_time;
        j["observed_seq"]    = e.n_observed > 0
            ? std::vector<int>(e.observed_seq, e.observed_seq + e.n_observed)
            : std::vector<int>();
        break;
    case CG_EVENT_PHASE3_PREDICTION:
        j["buffer_count"] = e.buffer_count;
        j["restricted"]   = e.restricted != 0;
        j["accepted"]     = e.accepted != 0;
        j["gesture_id"]   = e.gesture_id ? e.gesture_id : "";
        j["confidence"]   = e.confidence;
        if (e.restricted) j["candidates"] = candidateList(e);
        break;
    case CG_EVENT_COMMIT_FIRED:
        j["candidates"]          = candidateList(e);
        j["commit_deadline"]     = e.commit_deadline;
        j["min_buffer_deadline"] = e.min_buffer_deadline;
        break;
    default:
        break;
    }
    return j.dump(-1, ' ', false, json::error_handler_t::replace);
}

} // namespace cgsession
