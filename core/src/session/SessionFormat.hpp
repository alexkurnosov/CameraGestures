#pragma once
#include "CameraGestures/SessionCapture.h"
#include <cstddef>
#include <cstdint>
#include <string>

// On-disk layout of a .cgsession bundle, format version 1. Shared by the
// recorder and the reader; nothing here is public ABI.
//
// Every binary stream file:
//   stream header (16 bytes)
//     "CGSB"  u16 format_version  u16 stream_kind  u32 record_size  i32 track_index
//   then chunks, each:
//     chunk header (16 bytes)
//       "CGCK"  u32 record_count  u32 crc32(payload)  u32 reserved (0)
//     payload: record_count * record_size bytes
//
// All integers and floats are little-endian; floats are stored as their IEEE
// bit patterns, so a round trip is bit-exact.

namespace cgsession {

constexpr char     kStreamMagic[4] = {'C', 'G', 'S', 'B'};
constexpr char     kChunkMagic[4]  = {'C', 'G', 'C', 'K'};
constexpr size_t   kStreamHeaderSize = 16;
constexpr size_t   kChunkHeaderSize  = 16;

enum StreamKind : uint16_t {
    kStreamShots     = 1,
    kStreamTelemetry = 2,
};

// Shot record, 272 bytes:
//   0   f64 timestamp        8   f64 pts
//   16  f32 landmarks[21][3] (x, y, z)
//   268 u8 handedness  269 u8 is_absent  270 u8 has_pts  271 u8 reserved
constexpr size_t kShotRecordSize = 272;

// Telemetry record, 56 bytes, cg_frame_telemetry field order:
//   0 f64 timestamp  8 f64 commit_deadline  16 f64 min_buffer_deadline
//   24 f32 raw_energy  28 f32 smoothed_energy  32 f32 hold_run_ms
//   36 i32 hold_run_frames  40 i32 track_index  44 i32 buffer_count
//   48 u8 handedness  49 is_absent  50 gate_enabled  51 gate_open
//   52 raw_energy_valid  53 smoothed_energy_valid  54..55 reserved
constexpr size_t kTelemetryRecordSize = 56;

constexpr const char* kManifestFile  = "manifest.json";
constexpr const char* kTelemetryFile = "telemetry.bin";
constexpr const char* kEventsFile    = "events.jsonl";
constexpr const char* kLabelsFile    = "labels.jsonl";
constexpr const char* kShotsDir      = "shots";

// "shots/0.bin", "shots/absent.bin"
std::string shotFileName(int track_index);
// Parses a file name inside shots/; false if it is not a track file.
bool parseShotFileName(const std::string& name, int* track_index);

uint32_t crc32(const void* data, size_t len);

void encodeStreamHeader(uint8_t out[kStreamHeaderSize], uint16_t version,
                        uint16_t kind, uint32_t record_size, int32_t track_index);
void encodeChunkHeader(uint8_t out[kChunkHeaderSize], uint32_t record_count, uint32_t crc);

void encodeShot(uint8_t out[kShotRecordSize], const cg_session_shot& s);
void decodeShot(const uint8_t in[kShotRecordSize], int32_t track_index, cg_session_shot* s);

void encodeTelemetry(uint8_t out[kTelemetryRecordSize], const cg_frame_telemetry& r);
void decodeTelemetry(const uint8_t in[kTelemetryRecordSize], cg_frame_telemetry* r);

// Little-endian primitives.
void     putU16(uint8_t* p, uint16_t v);
void     putU32(uint8_t* p, uint32_t v);
void     putF64(uint8_t* p, double v);
uint16_t getU16(const uint8_t* p);
uint32_t getU32(const uint8_t* p);
double   getF64(const uint8_t* p);

std::string joinPath(const std::string& a, const std::string& b);

} // namespace cgsession
