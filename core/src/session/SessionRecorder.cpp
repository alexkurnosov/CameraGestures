#include "CameraGestures/CameraGestures.h"
#include "SessionFormat.hpp"
#include "SessionManifest.hpp"
#include "Sha256.hpp"

#include <chrono>
#include <cstdio>
#include <map>
#include <string>
#include <vector>
#include <sys/stat.h>

using json = nlohmann::json;
using namespace cgsession;

namespace {

double wallClockNow() {
    using namespace std::chrono;
    return duration<double>(system_clock::now().time_since_epoch()).count();
}

std::string baseName(const std::string& path) {
    const size_t slash = path.find_last_of('/');
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

// One binary stream file. Records collect in `pending` and go to disk as one
// CRC-checked chunk, flushed to the OS, so a killed process loses only what is
// still pending.
struct StreamWriter {
    FILE*                f = nullptr;
    size_t               record_size = 0;
    std::vector<uint8_t> pending;
    uint32_t             pending_count = 0;
    uint64_t             written = 0; // records on disk

    bool open(const std::string& path, uint16_t kind, size_t rec_size, int32_t track) {
        record_size = rec_size;
        f = std::fopen(path.c_str(), "wb");
        if (!f) return false;
        uint8_t header[kStreamHeaderSize];
        encodeStreamHeader(header, CG_SESSION_FORMAT_VERSION, kind,
                           static_cast<uint32_t>(rec_size), track);
        return std::fwrite(header, 1, sizeof(header), f) == sizeof(header)
            && std::fflush(f) == 0;
    }

    uint8_t* addRecord() {
        pending.resize(pending.size() + record_size);
        ++pending_count;
        return pending.data() + pending.size() - record_size;
    }

    uint64_t total() const { return written + pending_count; }

    bool flushChunk() {
        if (!f || pending_count == 0) return true;
        // Header and payload in one buffer: one write, so a kill rarely tears a chunk.
        std::vector<uint8_t> chunk(kChunkHeaderSize + pending.size());
        encodeChunkHeader(chunk.data(), pending_count, crc32(pending.data(), pending.size()));
        std::copy(pending.begin(), pending.end(), chunk.begin() + kChunkHeaderSize);
        const bool ok = std::fwrite(chunk.data(), 1, chunk.size(), f) == chunk.size()
                     && std::fflush(f) == 0;
        if (ok) written += pending_count;
        pending.clear();
        pending_count = 0;
        return ok;
    }

    bool close() {
        const bool ok = flushChunk();
        if (f) std::fclose(f);
        f = nullptr;
        return ok;
    }
};

bool writeFileAtomically(const std::string& path, const std::string& content) {
    const std::string tmp = path + ".tmp";
    FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) return false;
    bool ok = std::fwrite(content.data(), 1, content.size(), f) == content.size();
    ok = (std::fclose(f) == 0) && ok;
    return ok && std::rename(tmp.c_str(), path.c_str()) == 0;
}

json optionalString(const char* s) { return s ? json(s) : json(nullptr); }

} // namespace

struct cg_session_recorder_s {
    std::string bundle;
    int         chunk_records = CG_SESSION_DEFAULT_CHUNK_RECORDS;
    bool        recording = false;
    bool        started   = false; // start succeeded once; a recorder records one session
    std::string error;

    cg_recognizer_ref recognizer = nullptr;
    json              manifest;

    StreamWriter                telemetry;
    std::map<int, StreamWriter> shots;
    FILE*                       events = nullptr;
    std::string                 events_pending;
    uint64_t                    event_count = 0;

    int    current_track = CG_SESSION_ABSENT_TRACK;
    bool   have_range = false;
    double t_start = 0.0, t_end = 0.0;

    bool fail(const std::string& message) {
        error = message;
        return false;
    }

    std::string path(const std::string& rel) const { return joinPath(bundle, rel); }

    bool writeManifest() {
        return writeFileAtomically(path(kManifestFile),
            manifest.dump(2, ' ', false, json::error_handler_t::replace) + "\n");
    }

    bool flushAll() {
        bool ok = telemetry.flushChunk();
        for (auto& kv : shots) ok = kv.second.flushChunk() && ok;
        if (events && !events_pending.empty()) {
            ok = std::fwrite(events_pending.data(), 1, events_pending.size(), events)
                     == events_pending.size()
              && std::fflush(events) == 0 && ok;
            events_pending.clear();
        }
        return ok;
    }

    // Called before each shot is added: once any stream holds a full chunk,
    // every stream is flushed. Chunks therefore break only between shots — a
    // shot, its telemetry row and its events reach disk together — and no track
    // sits on unflushed data longer than one chunk of the busiest stream.
    bool flushIfFull() {
        const auto full = static_cast<uint32_t>(chunk_records);
        bool due = telemetry.pending_count >= full;
        for (const auto& kv : shots) due = due || kv.second.pending_count >= full;
        return !due || flushAllOrAbort();
    }

    bool flushAllOrAbort() {
        if (flushAll()) return true;
        abort("write failed while recording (disk full?)");
        return false;
    }

    void detach() {
        if (!recognizer) return;
        cg_recognizer_set_frame_telemetry_callback(recognizer, nullptr, nullptr);
        cg_recognizer_set_decision_event_callback(recognizer, nullptr, nullptr);
        recognizer = nullptr;
    }

    // An I/O failure mid-session: stop accepting data and keep what is on disk.
    void abort(const std::string& message) {
        error = message;
        detach();
        telemetry.close();
        for (auto& kv : shots) kv.second.close();
        if (events) std::fclose(events);
        events    = nullptr;
        recording = false;
        manifest["status"] = "aborted";
        manifest["error"]  = message;
        writeManifest(); // best effort; the reader works without it being updated
    }

    StreamWriter* shotStream(int track) {
        auto it = shots.find(track);
        if (it != shots.end()) return &it->second;
        StreamWriter w;
        const std::string rel = std::string(kShotsDir) + "/" + shotFileName(track);
        if (!w.open(path(rel), kStreamShots, kShotRecordSize, track)) {
            abort("cannot create " + rel);
            return nullptr;
        }
        return &shots.emplace(track, w).first->second;
    }
};

namespace {

void onFrameTelemetry(void* ctx, const cg_frame_telemetry* row) {
    cg_session_recorder_append_telemetry(static_cast<cg_session_recorder_ref>(ctx), row);
}

void onDecisionEvent(void* ctx, const cg_decision_event* event) {
    cg_session_recorder_append_decision_event(static_cast<cg_session_recorder_ref>(ctx), event);
}

json provenanceJson(const cg_session_provenance* p) {
    cg_session_provenance empty{};
    if (!p) p = &empty;
    json extra = json::object();
    for (int i = 0; i < p->n_extra && p->extra; ++i) {
        if (p->extra[i].key) extra[p->extra[i].key] = optionalString(p->extra[i].value);
    }
    return {
        {"app_version",  optionalString(p->app_version)},
        {"device_model", optionalString(p->device_model)},
        {"os_version",   optionalString(p->os_version)},
        {"camera", {
            {"preset",   optionalString(p->camera.preset)},
            {"width",    p->camera.width},
            {"height",   p->camera.height},
            {"fps",      p->camera.fps},
            {"position", cameraPositionName(p->camera.position)},
        }},
        {"extra", extra},
    };
}

} // namespace

extern "C" {

cg_session_recorder_ref cg_session_recorder_create(const char* bundle_path) {
    if (!bundle_path || !*bundle_path) return nullptr;
    auto* r = new (std::nothrow) cg_session_recorder_s();
    if (!r) return nullptr;
    r->bundle = bundle_path;
    return r;
}

void cg_session_recorder_destroy(cg_session_recorder_ref recorder) {
    if (!recorder) return;
    if (recorder->recording) cg_session_recorder_stop(recorder);
    delete recorder;
}

void cg_session_recorder_set_chunk_records(cg_session_recorder_ref recorder, int records) {
    if (!recorder || recorder->started || records < 1) return;
    recorder->chunk_records = records;
}

int cg_session_recorder_start(cg_session_recorder_ref       recorder,
                              cg_recognizer_ref             recognizer,
                              const cg_session_provenance*  provenance,
                              const cg_session_model_files* files) {
    if (!recorder) return 0;
    auto& r = *recorder;
    if (r.started) return r.fail("a recorder records one session; create another");

    struct stat st;
    if (stat(r.bundle.c_str(), &st) == 0) return r.fail("bundle already exists: " + r.bundle);

    // Hash before creating anything, so a bad path leaves no bundle behind.
    json model_files = json::object();
    const std::pair<const char*, const char*> roles[] = {
        {"gesture_model", files ? files->gesture_model : nullptr},
        {"gesture_ids",   files ? files->gesture_ids   : nullptr},
        {"pose_model",    files ? files->pose_model    : nullptr},
        {"pose_manifest", files ? files->pose_manifest : nullptr},
        {"preprocessor",  files ? files->preprocessor  : nullptr},
    };
    for (const auto& role : roles) {
        if (!role.second) { model_files[role.first] = nullptr; continue; }
        std::string hex;
        uint64_t    size = 0;
        if (!Sha256::hashFile(role.second, &hex, &size)) {
            return r.fail(std::string("cannot read ") + role.first + " file: " + role.second);
        }
        model_files[role.first] = {{"name", baseName(role.second)}, {"sha256", hex}, {"size", size}};
    }

    json config = nullptr;
    if (recognizer) {
        cg_recognizer_config c{};
        cg_recognizer_get_config(recognizer, &c);
        config = configToJson(c, cg_recognizer_get_bypass_phase2(recognizer) != 0);
    }

    if (mkdir(r.bundle.c_str(), 0755) != 0) return r.fail("cannot create bundle: " + r.bundle);
    if (mkdir(r.path(kShotsDir).c_str(), 0755) != 0) {
        return r.fail("cannot create " + r.path(kShotsDir));
    }

    r.manifest = {
        {"format",          "cgsession"},
        {"format_version",  CG_SESSION_FORMAT_VERSION},
        {"status",          "recording"},
        {"library_version", cg_version()},
        {"created_at",      wallClockNow()},
        {"stopped_at",      nullptr},
        {"chunk_records",   r.chunk_records},
        {"provenance",      provenanceJson(provenance)},
        {"model_files",     model_files},
        {"config",          config},
        {"files", {
            {"telemetry", kTelemetryFile},
            {"events",    kEventsFile},
            {"labels",    kLabelsFile},
            {"shots_dir", kShotsDir},
        }},
    };
    if (!r.writeManifest()) return r.fail("cannot write manifest");

    FILE* labels = std::fopen(r.path(kLabelsFile).c_str(), "wb");
    if (!labels) return r.fail("cannot create labels.jsonl");
    std::fclose(labels);

    if (!r.telemetry.open(r.path(kTelemetryFile), kStreamTelemetry, kTelemetryRecordSize, -1)) {
        return r.fail("cannot create telemetry.bin");
    }
    r.events = std::fopen(r.path(kEventsFile).c_str(), "wb");
    if (!r.events) {
        r.telemetry.close();
        return r.fail("cannot create events.jsonl");
    }

    r.started   = true;
    r.recording = true;
    r.error.clear();
    if (recognizer) {
        r.recognizer = recognizer;
        cg_recognizer_set_frame_telemetry_callback(recognizer, onFrameTelemetry, recorder);
        cg_recognizer_set_decision_event_callback(recognizer, onDecisionEvent, recorder);
    }
    return 1;
}

int cg_session_recorder_record_shot(cg_session_recorder_ref recorder,
                                    const cg_session_shot*  shot) {
    if (!recorder || !shot || !recorder->recording) return 0;
    auto& r = *recorder;
    const int track = shot->shot.is_absent ? CG_SESSION_ABSENT_TRACK : shot->track_index;
    if (track < 0 && !shot->shot.is_absent) {
        return r.fail("a present shot needs track_index >= 0");
    }
    if (!r.flushIfFull()) return 0;
    StreamWriter* w = r.shotStream(track);
    if (!w) return 0;

    cg_session_shot s = *shot;
    s.track_index = track;
    encodeShot(w->addRecord(), s);
    r.current_track = track;

    const double t = s.shot.timestamp;
    if (!r.have_range) { r.t_start = r.t_end = t; r.have_range = true; }
    if (t < r.t_start) r.t_start = t;
    if (t > r.t_end)   r.t_end   = t;
    return 1;
}

int cg_session_recorder_append_telemetry(cg_session_recorder_ref   recorder,
                                         const cg_frame_telemetry* row) {
    if (!recorder || !row || !recorder->recording) return 0;
    auto& r = *recorder;
    cg_frame_telemetry stamped = *row;
    if (stamped.track_index == -1) stamped.track_index = r.current_track;
    encodeTelemetry(r.telemetry.addRecord(), stamped);
    // Rows normally flush with the next shot; this bounds a caller feeding
    // rows without shots.
    if (r.telemetry.pending_count >= 2u * static_cast<uint32_t>(r.chunk_records)) {
        return r.flushAllOrAbort();
    }
    return 1;
}

int cg_session_recorder_append_decision_event(cg_session_recorder_ref  recorder,
                                              const cg_decision_event* event) {
    if (!recorder || !event || !recorder->recording) return 0;
    auto& r = *recorder;
    r.events_pending += decisionEventLine(*event, r.current_track);
    r.events_pending += '\n';
    ++r.event_count;
    return 1;
}

int cg_session_recorder_append_binding_event(cg_session_recorder_ref recorder,
                                             double      timestamp,
                                             const char* event,
                                             const char* fields_json) {
    if (!recorder || !event || !recorder->recording) return 0;
    auto& r = *recorder;
    json line = json::object();
    if (fields_json) {
        line = json::parse(fields_json, nullptr, false);
        if (!line.is_object()) return r.fail("binding event fields are not a JSON object");
    }
    line["t"]      = timestamp;
    line["source"] = "binding";
    line["event"]  = event;
    r.events_pending += line.dump(-1, ' ', false, json::error_handler_t::replace);
    r.events_pending += '\n';
    ++r.event_count;
    return 1;
}

int cg_session_recorder_stop(cg_session_recorder_ref recorder) {
    if (!recorder || !recorder->recording) return 0;
    auto& r = *recorder;
    r.detach();

    bool ok = r.flushAll();
    ok = r.telemetry.close() && ok;
    json shot_counts = json::object();
    for (auto& kv : r.shots) {
        ok = kv.second.close() && ok;
        shot_counts[std::to_string(kv.first)] = kv.second.written;
    }
    if (r.events) ok = (std::fclose(r.events) == 0) && ok;
    r.events    = nullptr;
    r.recording = false;
    if (!ok) {
        r.abort("write failed while stopping (disk full?)");
        return 0;
    }

    r.manifest["status"]     = "complete";
    r.manifest["stopped_at"] = wallClockNow();
    r.manifest["counts"]     = {
        {"telemetry", r.telemetry.written},
        {"events",    r.event_count},
        {"shots",     shot_counts},
    };
    r.manifest["time_range"] = r.have_range ? json::array({r.t_start, r.t_end}) : json(nullptr);
    if (!r.writeManifest()) return r.fail("cannot finalise manifest");
    return 1;
}

int cg_session_recorder_is_recording(cg_session_recorder_ref recorder) {
    return recorder && recorder->recording ? 1 : 0;
}

const char* cg_session_recorder_last_error(cg_session_recorder_ref recorder) {
    return recorder ? recorder->error.c_str() : "";
}

} // extern "C"
