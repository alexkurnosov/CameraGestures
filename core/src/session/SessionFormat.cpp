#include "SessionFormat.hpp"
#include <array>
#include <cstdlib>
#include <cstring>

namespace cgsession {

namespace {

void putU8(uint8_t* p, uint8_t v) { *p = v; }

void putF32(uint8_t* p, float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, 4);
    putU32(p, bits);
}

float getF32(const uint8_t* p) {
    uint32_t bits = getU32(p);
    float v;
    std::memcpy(&v, &bits, 4);
    return v;
}

void    putI32(uint8_t* p, int32_t v) { putU32(p, static_cast<uint32_t>(v)); }
int32_t getI32(const uint8_t* p)      { return static_cast<int32_t>(getU32(p)); }

} // namespace

void putU16(uint8_t* p, uint16_t v) {
    p[0] = uint8_t(v);
    p[1] = uint8_t(v >> 8);
}

void putU32(uint8_t* p, uint32_t v) {
    for (int i = 0; i < 4; ++i) p[i] = uint8_t(v >> (8 * i));
}

void putF64(uint8_t* p, double v) {
    uint64_t bits;
    std::memcpy(&bits, &v, 8);
    for (int i = 0; i < 8; ++i) p[i] = uint8_t(bits >> (8 * i));
}

uint16_t getU16(const uint8_t* p) { return uint16_t(p[0] | (p[1] << 8)); }

uint32_t getU32(const uint8_t* p) {
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) v |= uint32_t(p[i]) << (8 * i);
    return v;
}

double getF64(const uint8_t* p) {
    uint64_t bits = 0;
    for (int i = 0; i < 8; ++i) bits |= uint64_t(p[i]) << (8 * i);
    double v;
    std::memcpy(&v, &bits, 8);
    return v;
}

std::string shotFileName(int track_index) {
    if (track_index == CG_SESSION_ABSENT_TRACK) return "absent.bin";
    return std::to_string(track_index) + ".bin";
}

bool parseShotFileName(const std::string& name, int* track_index) {
    if (name == "absent.bin") {
        *track_index = CG_SESSION_ABSENT_TRACK;
        return true;
    }
    const size_t dot = name.find(".bin");
    if (dot == std::string::npos || dot == 0 || dot + 4 != name.size()) return false;
    for (size_t i = 0; i < dot; ++i) {
        if (name[i] < '0' || name[i] > '9') return false;
    }
    *track_index = std::atoi(name.substr(0, dot).c_str());
    return true;
}

uint32_t crc32(const void* data, size_t len) {
    // Function-local static: initialised once, thread-safe.
    static const std::array<uint32_t, 256> table = [] {
        std::array<uint32_t, 256> t{};
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            t[i] = c;
        }
        return t;
    }();
    uint32_t c = 0xFFFFFFFFu;
    const auto* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < len; ++i) c = table[(c ^ p[i]) & 0xff] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

void encodeStreamHeader(uint8_t out[kStreamHeaderSize], uint16_t version,
                        uint16_t kind, uint32_t record_size, int32_t track_index) {
    std::memcpy(out, kStreamMagic, 4);
    putU16(out + 4, version);
    putU16(out + 6, kind);
    putU32(out + 8, record_size);
    putI32(out + 12, track_index);
}

void encodeChunkHeader(uint8_t out[kChunkHeaderSize], uint32_t record_count, uint32_t crc) {
    std::memcpy(out, kChunkMagic, 4);
    putU32(out + 4, record_count);
    putU32(out + 8, crc);
    putU32(out + 12, 0);
}

void encodeShot(uint8_t out[kShotRecordSize], const cg_session_shot& s) {
    putF64(out + 0, s.shot.timestamp);
    putF64(out + 8, s.pts);
    for (int i = 0; i < 21; ++i) {
        putF32(out + 16 + 12 * i + 0, s.shot.landmarks[i].x);
        putF32(out + 16 + 12 * i + 4, s.shot.landmarks[i].y);
        putF32(out + 16 + 12 * i + 8, s.shot.landmarks[i].z);
    }
    putU8(out + 268, uint8_t(s.shot.handedness));
    putU8(out + 269, s.shot.is_absent ? 1 : 0);
    putU8(out + 270, s.has_pts ? 1 : 0);
    putU8(out + 271, 0);
}

void decodeShot(const uint8_t in[kShotRecordSize], int32_t track_index, cg_session_shot* s) {
    std::memset(s, 0, sizeof(*s));
    s->shot.timestamp = getF64(in + 0);
    s->pts            = getF64(in + 8);
    for (int i = 0; i < 21; ++i) {
        s->shot.landmarks[i].x = getF32(in + 16 + 12 * i + 0);
        s->shot.landmarks[i].y = getF32(in + 16 + 12 * i + 4);
        s->shot.landmarks[i].z = getF32(in + 16 + 12 * i + 8);
    }
    s->shot.handedness = static_cast<cg_handedness>(in[268]);
    s->shot.is_absent  = in[269];
    s->has_pts         = in[270];
    s->track_index     = track_index;
}

void encodeTelemetry(uint8_t out[kTelemetryRecordSize], const cg_frame_telemetry& r) {
    putF64(out + 0,  r.timestamp);
    putF64(out + 8,  r.commit_deadline);
    putF64(out + 16, r.min_buffer_deadline);
    putF32(out + 24, r.raw_energy);
    putF32(out + 28, r.smoothed_energy);
    putF32(out + 32, r.hold_run_ms);
    putI32(out + 36, r.hold_run_frames);
    putI32(out + 40, r.track_index);
    putI32(out + 44, r.buffer_count);
    putU8(out + 48, r.handedness);
    putU8(out + 49, r.is_absent);
    putU8(out + 50, r.gate_enabled);
    putU8(out + 51, r.gate_open);
    putU8(out + 52, r.raw_energy_valid);
    putU8(out + 53, r.smoothed_energy_valid);
    putU8(out + 54, 0);
    putU8(out + 55, 0);
}

void decodeTelemetry(const uint8_t in[kTelemetryRecordSize], cg_frame_telemetry* r) {
    std::memset(r, 0, sizeof(*r));
    r->timestamp             = getF64(in + 0);
    r->commit_deadline       = getF64(in + 8);
    r->min_buffer_deadline   = getF64(in + 16);
    r->raw_energy            = getF32(in + 24);
    r->smoothed_energy       = getF32(in + 28);
    r->hold_run_ms           = getF32(in + 32);
    r->hold_run_frames       = getI32(in + 36);
    r->track_index           = getI32(in + 40);
    r->buffer_count          = getI32(in + 44);
    r->handedness            = in[48];
    r->is_absent             = in[49];
    r->gate_enabled          = in[50];
    r->gate_open             = in[51];
    r->raw_energy_valid      = in[52];
    r->smoothed_energy_valid = in[53];
}

std::string joinPath(const std::string& a, const std::string& b) {
    if (a.empty() || a.back() == '/') return a + b;
    return a + "/" + b;
}

} // namespace cgsession
