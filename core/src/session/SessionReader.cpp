#include "CameraGestures/CameraGestures.h"
#include "SessionFormat.hpp"
#include "SessionManifest.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <dirent.h>

using json = nlohmann::json;
using namespace cgsession;

namespace {

// One binary stream, indexed by chunk at open. Every chunk is length- and
// CRC-checked once; the first bad one ends the readable prefix. Records are
// decoded only when read.
struct StreamReader {
    FILE*    f = nullptr;
    size_t   record_size = 0;
    int32_t  track_index = 0;
    bool     truncated = false;

    struct Chunk {
        long     payload_offset;
        uint64_t first_record;
        uint32_t count;
        double   first_timestamp; // every record starts with its f64 timestamp
    };
    std::vector<Chunk> chunks;
    uint64_t           total = 0;

    // Cache of the last decoded chunk's payload: sequential reads stay cheap.
    size_t               cached_chunk = SIZE_MAX;
    std::vector<uint8_t> cache;

    ~StreamReader() { if (f) std::fclose(f); }

    // Returns false with *error set on a bad or unsupported header.
    bool open(const std::string& path, uint16_t kind, size_t expected_record_size,
              std::string* error) {
        f = std::fopen(path.c_str(), "rb");
        if (!f) { *error = "cannot open " + path; return false; }

        uint8_t h[kStreamHeaderSize];
        if (std::fread(h, 1, sizeof(h), f) != sizeof(h)) {
            // A kill between creating the file and writing its header.
            truncated = true;
            record_size = expected_record_size;
            return true;
        }
        if (std::memcmp(h, kStreamMagic, 4) != 0) {
            *error = path + " is not a session stream";
            return false;
        }
        const uint16_t version = getU16(h + 4);
        if (version > CG_SESSION_FORMAT_VERSION) {
            *error = path + " has format version " + std::to_string(version)
                   + "; this library reads up to " + std::to_string(CG_SESSION_FORMAT_VERSION);
            return false;
        }
        record_size = getU32(h + 8);
        track_index = static_cast<int32_t>(getU32(h + 12));
        if (getU16(h + 6) != kind || record_size != expected_record_size) {
            *error = path + " has an unexpected stream kind or record size";
            return false;
        }
        scan();
        return true;
    }

    void scan() {
        std::fseek(f, 0, SEEK_END);
        const long file_size = std::ftell(f);
        long pos = static_cast<long>(kStreamHeaderSize);
        std::vector<uint8_t> payload;
        while (pos < file_size) {
            uint8_t ch[kChunkHeaderSize];
            std::fseek(f, pos, SEEK_SET);
            if (std::fread(ch, 1, sizeof(ch), f) != sizeof(ch)
                    || std::memcmp(ch, kChunkMagic, 4) != 0) {
                truncated = true;
                return;
            }
            const uint32_t count = getU32(ch + 4);
            const uint32_t crc   = getU32(ch + 8);
            const long     bytes = static_cast<long>(count) * static_cast<long>(record_size);
            const long     payload_offset = pos + static_cast<long>(kChunkHeaderSize);
            if (count == 0 || payload_offset + bytes > file_size) {
                truncated = true;
                return;
            }
            payload.resize(static_cast<size_t>(bytes));
            if (std::fread(payload.data(), 1, payload.size(), f) != payload.size()
                    || crc32(payload.data(), payload.size()) != crc) {
                truncated = true;
                return;
            }
            chunks.push_back({payload_offset, total, count, getF64(payload.data())});
            total += count;
            pos = payload_offset + bytes;
        }
    }

    size_t chunkFor(uint64_t record) const {
        auto it = std::upper_bound(chunks.begin(), chunks.end(), record,
            [](uint64_t r, const Chunk& c) { return r < c.first_record; });
        return static_cast<size_t>(it - chunks.begin()) - 1;
    }

    const uint8_t* chunkPayload(size_t ci) {
        if (ci != cached_chunk) {
            const Chunk& c = chunks[ci];
            cache.resize(size_t(c.count) * record_size);
            std::fseek(f, c.payload_offset, SEEK_SET);
            if (std::fread(cache.data(), 1, cache.size(), f) != cache.size()) {
                cached_chunk = SIZE_MAX;
                return nullptr;
            }
            cached_chunk = ci;
        }
        return cache.data();
    }

    // Calls decode(record_bytes, i) for records [first, first + count).
    template <typename Decode>
    size_t read(uint64_t first, size_t count, Decode decode) {
        if (first >= total) return 0;
        count = static_cast<size_t>(std::min<uint64_t>(count, total - first));
        size_t done = 0;
        while (done < count) {
            const uint64_t rec = first + done;
            const size_t   ci  = chunkFor(rec);
            const uint8_t* p   = chunkPayload(ci);
            if (!p) break;
            const Chunk& c   = chunks[ci];
            uint64_t     off = rec - c.first_record;
            for (; off < c.count && done < count; ++off, ++done) {
                decode(p + off * record_size, done);
            }
        }
        return done;
    }

    double timestampAt(uint64_t i) {
        const size_t   ci = chunkFor(i);
        const uint8_t* p  = chunkPayload(ci);
        return p ? getF64(p + (i - chunks[ci].first_record) * record_size) : 0.0;
    }

    // First record with timestamp >= t, assuming timestamps never decrease.
    uint64_t find(double t) {
        if (chunks.empty()) return 0;
        // Last chunk whose first timestamp is < t holds the answer, or the answer
        // is the first record of the chunk after it.
        auto it = std::lower_bound(chunks.begin(), chunks.end(), t,
            [](const Chunk& c, double v) { return c.first_timestamp < v; });
        if (it == chunks.begin()) return 0;
        const Chunk& c = *(it - 1);
        uint64_t lo = c.first_record, hi = c.first_record + c.count;
        while (lo < hi) {
            const uint64_t mid = lo + (hi - lo) / 2;
            if (timestampAt(mid) < t) lo = mid + 1; else hi = mid;
        }
        return lo;
    }
};

bool readWholeFile(const std::string& path, std::string* out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    std::ostringstream ss;
    ss << in.rdbuf();
    *out = ss.str();
    return true;
}

// Copies a manifest string into owned storage; NULL for a JSON null.
const char* keep(std::vector<std::unique_ptr<std::string>>& store, const json& v) {
    if (!v.is_string()) return nullptr;
    store.push_back(std::make_unique<std::string>(v.get<std::string>()));
    return store.back()->c_str();
}

} // namespace

struct cg_session_reader_s {
    std::string bundle;
    json        manifest;
    std::string manifest_text;
    int         format_version = 0;
    bool        complete = false;

    StreamReader                                   telemetry;
    std::map<int, std::unique_ptr<StreamReader>>   shots;
    std::vector<int>                               track_ids;
    std::vector<std::string>                       events;

    std::vector<std::unique_ptr<std::string>> strings;
    std::string                               library_version;
    cg_session_provenance                     provenance{};
    std::vector<cg_session_kv>                extra;
    cg_session_file_records                   files{};
    bool                                      has_config = false;
    cg_recognizer_config                      config{};
    bool                                      bypass_phase2 = false;

    bool truncated() const {
        if (telemetry.truncated) return true;
        for (const auto& kv : shots) if (kv.second->truncated) return true;
        return false;
    }

    StreamReader* track(int t) {
        auto it = shots.find(t);
        return it == shots.end() ? nullptr : it->second.get();
    }

    bool load(std::string* error) {
        if (!readWholeFile(joinPath(bundle, kManifestFile), &manifest_text)) {
            *error = "no manifest.json in " + bundle;
            return false;
        }
        try {
            manifest = json::parse(manifest_text);
        } catch (const json::exception& e) {
            *error = std::string("manifest.json does not parse: ") + e.what();
            return false;
        }
        if (manifest.value("format", "") != "cgsession" || !manifest.contains("format_version")
                || !manifest["format_version"].is_number_integer()) {
            *error = "manifest.json is not a cgsession manifest";
            return false;
        }
        format_version = manifest["format_version"].get<int>();
        if (format_version > CG_SESSION_FORMAT_VERSION) {
            *error = "session format version " + std::to_string(format_version)
                   + " is newer than this library supports ("
                   + std::to_string(CG_SESSION_FORMAT_VERSION) + "); update CameraGestures";
            return false;
        }
        complete = manifest.value("status", "") == "complete";

        try {
            loadManifestFields();
        } catch (const json::exception& e) {
            *error = std::string("manifest.json has a malformed field: ") + e.what();
            return false;
        }

        if (!telemetry.open(joinPath(bundle, kTelemetryFile), kStreamTelemetry,
                            kTelemetryRecordSize, error)) {
            return false;
        }
        if (!loadShots(error)) return false;
        loadEvents();
        return true;
    }

    void loadManifestFields() {
        library_version = manifest.value("library_version", "");

        const json p = manifest.value("provenance", json::object());
        provenance.app_version  = keep(strings, p.value("app_version", json()));
        provenance.device_model = keep(strings, p.value("device_model", json()));
        provenance.os_version   = keep(strings, p.value("os_version", json()));
        const json cam = p.value("camera", json::object());
        provenance.camera.preset   = keep(strings, cam.value("preset", json()));
        provenance.camera.width    = cam.value("width", 0);
        provenance.camera.height   = cam.value("height", 0);
        provenance.camera.fps      = cam.value("fps", 0.0);
        provenance.camera.position = cameraPositionFromName(cam.value("position", "unknown"));
        const json ex = p.value("extra", json::object());
        for (auto it = ex.begin(); it != ex.end(); ++it) {
            extra.push_back({keep(strings, json(it.key())), keep(strings, it.value())});
        }
        provenance.extra   = extra.empty() ? nullptr : extra.data();
        provenance.n_extra = static_cast<int32_t>(extra.size());

        const json mf = manifest.value("model_files", json::object());
        auto file = [&](const char* role, cg_session_file_record* out) {
            const json v = mf.value(role, json());
            if (!v.is_object()) return;
            out->name   = keep(strings, v.value("name", json()));
            out->sha256 = keep(strings, v.value("sha256", json()));
            out->size   = v.value("size", uint64_t(0));
        };
        file("gesture_model", &files.gesture_model);
        file("gesture_ids",   &files.gesture_ids);
        file("pose_model",    &files.pose_model);
        file("pose_manifest", &files.pose_manifest);
        file("preprocessor",  &files.preprocessor);

        const json cfg = manifest.value("config", json());
        has_config = cfg.is_object() && configFromJson(cfg, &config, &bypass_phase2);
    }

    bool loadShots(std::string* error) {
        const std::string dir = joinPath(bundle, kShotsDir);
        DIR* d = opendir(dir.c_str());
        if (!d) return true; // no shots recorded
        std::vector<std::pair<int, std::string>> found;
        while (dirent* e = readdir(d)) {
            int t;
            if (parseShotFileName(e->d_name, &t)) found.push_back({t, e->d_name});
        }
        closedir(d);
        std::sort(found.begin(), found.end());
        for (const auto& tf : found) {
            auto s = std::make_unique<StreamReader>();
            if (!s->open(joinPath(dir, tf.second), kStreamShots, kShotRecordSize, error)) {
                return false;
            }
            track_ids.push_back(tf.first);
            shots[tf.first] = std::move(s);
        }
        return true;
    }

    void loadEvents() {
        std::string text;
        if (!readWholeFile(joinPath(bundle, kEventsFile), &text)) return;
        size_t start = 0;
        while (start < text.size()) {
            const size_t nl = text.find('\n', start);
            if (nl == std::string::npos) break; // partial last line
            if (nl > start) events.push_back(text.substr(start, nl - start));
            start = nl + 1;
        }
    }
};

extern "C" {

cg_session_reader_ref cg_session_reader_open(const char* bundle_path,
                                             char* error_buf, size_t error_buf_len) {
    auto set_error = [&](const std::string& msg) {
        if (error_buf && error_buf_len > 0) {
            std::snprintf(error_buf, error_buf_len, "%s", msg.c_str());
        }
    };
    if (!bundle_path) { set_error("no bundle path"); return nullptr; }
    auto r = std::make_unique<cg_session_reader_s>();
    r->bundle = bundle_path;
    std::string error;
    if (!r->load(&error)) { set_error(error); return nullptr; }
    set_error("");
    return r.release();
}

void cg_session_reader_close(cg_session_reader_ref reader) { delete reader; }

int cg_session_reader_format_version(cg_session_reader_ref reader) {
    return reader ? reader->format_version : 0;
}

int cg_session_reader_is_complete(cg_session_reader_ref reader) {
    return reader && reader->complete ? 1 : 0;
}

int cg_session_reader_is_truncated(cg_session_reader_ref reader) {
    return reader && reader->truncated() ? 1 : 0;
}

const char* cg_session_reader_manifest_json(cg_session_reader_ref reader) {
    return reader ? reader->manifest_text.c_str() : "";
}

const char* cg_session_reader_library_version(cg_session_reader_ref reader) {
    return reader ? reader->library_version.c_str() : "";
}

double cg_session_reader_created_at(cg_session_reader_ref reader) {
    return reader ? reader->manifest.value("created_at", 0.0) : 0.0;
}

int cg_session_reader_get_provenance(cg_session_reader_ref reader, cg_session_provenance* out) {
    if (!reader || !out) return 0;
    *out = reader->provenance;
    return 1;
}

int cg_session_reader_get_model_files(cg_session_reader_ref reader, cg_session_file_records* out) {
    if (!reader || !out) return 0;
    *out = reader->files;
    return 1;
}

int cg_session_reader_get_config(cg_session_reader_ref reader, cg_recognizer_config* out,
                                 int* bypass_phase2_out) {
    if (!reader || !reader->has_config) return 0;
    if (out) *out = reader->config;
    if (bypass_phase2_out) *bypass_phase2_out = reader->bypass_phase2 ? 1 : 0;
    return 1;
}

int cg_session_reader_time_range(cg_session_reader_ref reader, double* start_out, double* end_out) {
    if (!reader) return 0;
    bool   any = false;
    double lo = 0.0, hi = 0.0;
    for (const auto& kv : reader->shots) {
        StreamReader& s = *kv.second;
        if (s.total == 0) continue;
        const double first = s.timestampAt(0);
        const double last  = s.timestampAt(s.total - 1);
        if (!any || first < lo) lo = first;
        if (!any || last > hi)  hi = last;
        any = true;
    }
    if (!any) return 0;
    if (start_out) *start_out = lo;
    if (end_out)   *end_out   = hi;
    return 1;
}

int cg_session_reader_track_count(cg_session_reader_ref reader) {
    return reader ? static_cast<int>(reader->track_ids.size()) : 0;
}

int cg_session_reader_track_at(cg_session_reader_ref reader, int i) {
    if (!reader || i < 0 || i >= static_cast<int>(reader->track_ids.size())) return 0;
    return reader->track_ids[static_cast<size_t>(i)];
}

size_t cg_session_reader_shot_count(cg_session_reader_ref reader, int track_index) {
    StreamReader* s = reader ? reader->track(track_index) : nullptr;
    return s ? static_cast<size_t>(s->total) : 0;
}

size_t cg_session_reader_read_shots(cg_session_reader_ref reader, int track_index,
                                    size_t first, size_t count, cg_session_shot* out) {
    StreamReader* s = reader ? reader->track(track_index) : nullptr;
    if (!s || !out) return 0;
    return s->read(first, count, [&](const uint8_t* p, size_t i) {
        decodeShot(p, track_index, &out[i]);
    });
}

size_t cg_session_reader_find_shot(cg_session_reader_ref reader, int track_index, double t) {
    StreamReader* s = reader ? reader->track(track_index) : nullptr;
    return s ? static_cast<size_t>(s->find(t)) : 0;
}

size_t cg_session_reader_telemetry_count(cg_session_reader_ref reader) {
    return reader ? static_cast<size_t>(reader->telemetry.total) : 0;
}

size_t cg_session_reader_read_telemetry(cg_session_reader_ref reader, size_t first,
                                        size_t count, cg_frame_telemetry* out) {
    if (!reader || !out) return 0;
    return reader->telemetry.read(first, count, [&](const uint8_t* p, size_t i) {
        decodeTelemetry(p, &out[i]);
    });
}

size_t cg_session_reader_find_telemetry(cg_session_reader_ref reader, double t) {
    return reader ? static_cast<size_t>(reader->telemetry.find(t)) : 0;
}

size_t cg_session_reader_event_count(cg_session_reader_ref reader) {
    return reader ? reader->events.size() : 0;
}

const char* cg_session_reader_event_json(cg_session_reader_ref reader, size_t i) {
    if (!reader || i >= reader->events.size()) return nullptr;
    return reader->events[i].c_str();
}

} // extern "C"
