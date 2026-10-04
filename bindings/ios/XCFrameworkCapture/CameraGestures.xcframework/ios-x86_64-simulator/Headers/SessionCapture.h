#ifndef CG_SESSION_CAPTURE_H
#define CG_SESSION_CAPTURE_H

#include "Types.h"
#include "HandGestureRecognizing.h"

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * Session capture — record a recognition session to a .cgsession bundle and
 * read it back.
 *
 * Bundle layout (a directory):
 *   manifest.json        provenance, model file hashes, full recognizer config
 *   shots/<track>.bin    one packed shot track per MediaPipe hand index
 *                        (0.bin, 1.bin, ...); frames with no hand go to absent.bin
 *   telemetry.bin        one packed cg_frame_telemetry row per processed shot
 *   events.jsonl         decision events ("source": "core") and binding events
 *                        ("source": "binding"), one JSON object per line
 *   labels.jsonl         reserved, empty in format version 1
 *
 * Binary streams are little-endian, written in CRC-checked chunks of up to
 * CG_SESSION_DEFAULT_CHUNK_RECORDS records and flushed to the OS per chunk.
 * All streams flush together, between shots, so a shot, its telemetry row and
 * its events reach disk together, and a killed process loses at most the one
 * chunk it had not yet written. The reader keeps every complete chunk and
 * reports a torn or corrupt one as truncation.
 * ---------------------------------------------------------------------- */

#define CG_SESSION_FORMAT_VERSION        1
#define CG_SESSION_DEFAULT_CHUNK_RECORDS 64
#define CG_SESSION_ABSENT_TRACK          (-1)

/* One recorded shot. */
typedef struct cg_session_shot {
    cg_handshot shot;         /* as fed to the recognizer; timestamp untouched */
    double      pts;          /* capture presentation timestamp (seconds); valid only if has_pts */
    int32_t     track_index;  /* MediaPipe per-frame hand index; CG_SESSION_ABSENT_TRACK for absent frames */
    int32_t     has_pts;      /* bool */
} cg_session_shot;

/* -------------------------------------------------------------------------
 * Provenance and model files — supplied by the caller at start
 * ---------------------------------------------------------------------- */

typedef enum cg_camera_position {
    CG_CAMERA_POSITION_UNKNOWN  = 0,
    CG_CAMERA_POSITION_FRONT    = 1,
    CG_CAMERA_POSITION_BACK     = 2,
    CG_CAMERA_POSITION_EXTERNAL = 3
} cg_camera_position;

typedef struct cg_session_camera_info {
    const char* preset;   /* platform preset name, e.g. "AVCaptureSessionPreset640x480"; NULL = unknown */
    int32_t     width;    /* delivered frame size in pixels; 0 = unknown */
    int32_t     height;
    double      fps;      /* nominal capture rate; 0 = unknown */
    int32_t     position; /* cg_camera_position */
} cg_session_camera_info;

/* Platform-specific notes nobody depends on. Anything a reader relies on
 * belongs in a named field instead. */
typedef struct cg_session_kv {
    const char* key;
    const char* value;
} cg_session_kv;

typedef struct cg_session_provenance {
    const char*            app_version;   /* NULL = unknown */
    const char*            device_model;
    const char*            os_version;
    cg_session_camera_info camera;
    const cg_session_kv*   extra;         /* may be NULL when n_extra == 0 */
    int32_t                n_extra;
} cg_session_provenance;

/* Paths of the files that shaped the session's decisions. The recorder
 * hashes each (SHA-256) at start and stores the file name, size and hash —
 * never the full path. NULL = not loaded; stored as null. */
typedef struct cg_session_model_files {
    const char* gesture_model;   /* gesture_model.tflite */
    const char* gesture_ids;     /* gesture_ids.json */
    const char* pose_model;      /* pose_model.tflite */
    const char* pose_manifest;   /* pose_manifest.json */
    const char* preprocessor;    /* preprocessor.js */
} cg_session_model_files;

/* -------------------------------------------------------------------------
 * Recorder
 *
 * Not thread-safe: call every function on the queue that drives the
 * recognizer. Per shot, the order is
 *     cg_session_recorder_record_shot(rec, &s);
 *     cg_recognizer_process_shot(recognizer, &s.shot);
 * so the telemetry row and events the shot produces are attributed to its
 * track.
 * ---------------------------------------------------------------------- */

typedef struct cg_session_recorder_s* cg_session_recorder_ref;

/* bundle_path: the .cgsession directory to create. It must not exist yet;
 * its parent must. Nothing is written until start. */
cg_session_recorder_ref cg_session_recorder_create(const char* bundle_path);

/* Stops (finalising the manifest) if still recording, then frees. */
void cg_session_recorder_destroy(cg_session_recorder_ref recorder);

/* Records per chunk; call before start. Values < 1 are ignored. */
void cg_session_recorder_set_chunk_records(cg_session_recorder_ref recorder, int records);

/* Creates the bundle, hashes the model files, writes the manifest and opens
 * the streams.
 *
 * recognizer: when non-NULL, its config is read into the manifest and the
 *   recorder installs itself as the recognizer's frame-telemetry and
 *   decision-event callbacks until stop — replacing any set before. When
 *   NULL, the manifest carries no config and the caller feeds rows and events
 *   through cg_session_recorder_append_telemetry / _append_decision_event.
 * provenance, files: may be NULL (everything unknown).
 * Returns 1 on success, 0 on failure (see cg_session_recorder_last_error). */
int cg_session_recorder_start(cg_session_recorder_ref       recorder,
                              cg_recognizer_ref             recognizer,
                              const cg_session_provenance*  provenance,
                              const cg_session_model_files* files);

/* Records one shot. An absent shot goes to the absent track whatever its
 * track_index; a present shot needs track_index >= 0. Returns 1 on success. */
int cg_session_recorder_record_shot(cg_session_recorder_ref recorder,
                                    const cg_session_shot*  shot);

/* Manual feeding (recognizer == NULL at start, or to chain callbacks).
 * A row whose track_index is -1 is stamped with the track of the last
 * recorded shot. Events are attributed to that track too. */
int cg_session_recorder_append_telemetry(cg_session_recorder_ref   recorder,
                                         const cg_frame_telemetry* row);
int cg_session_recorder_append_decision_event(cg_session_recorder_ref  recorder,
                                              const cg_decision_event* event);

/* Appends an event from the platform binding (cooldown windows, suppressed
 * gestures, status changes) to events.jsonl as
 *     {"t": timestamp, "source": "binding", "event": event, ...fields}
 * fields_json: a JSON object whose members are added to the line, or NULL.
 * Its "t", "source" and "event" members, if any, are ignored. Binding events
 * carry no track. Returns 0 if fields_json is not a JSON object. */
int cg_session_recorder_append_binding_event(cg_session_recorder_ref recorder,
                                             double      timestamp,
                                             const char* event,
                                             const char* fields_json);

/* Flushes every stream, detaches from the recognizer and rewrites the
 * manifest as complete, with record counts and the session time range.
 * Returns 1 on success. */
int cg_session_recorder_stop(cg_session_recorder_ref recorder);

int cg_session_recorder_is_recording(cg_session_recorder_ref recorder);

/* Human-readable reason for the last failure; "" if none. Valid until the
 * next call on this recorder. */
const char* cg_session_recorder_last_error(cg_session_recorder_ref recorder);

/* -------------------------------------------------------------------------
 * Reader
 *
 * Shots and telemetry are read on demand, one chunk at a time; nothing is
 * decoded up front. All returned strings live as long as the reader.
 * ---------------------------------------------------------------------- */

typedef struct cg_session_reader_s* cg_session_reader_ref;

/* A model file as recorded. name == NULL: not loaded when recording. */
typedef struct cg_session_file_record {
    const char* name;     /* file name only */
    const char* sha256;   /* 64 lowercase hex digits */
    uint64_t    size;     /* bytes */
} cg_session_file_record;

typedef struct cg_session_file_records {
    cg_session_file_record gesture_model;
    cg_session_file_record gesture_ids;
    cg_session_file_record pose_model;
    cg_session_file_record pose_manifest;
    cg_session_file_record preprocessor;
} cg_session_file_records;

/* Returns NULL on failure and writes the reason to error_buf (may be NULL).
 * Fails on a missing or unparsable manifest and on a format version newer
 * than CG_SESSION_FORMAT_VERSION. A truncated stream is not a failure. */
cg_session_reader_ref cg_session_reader_open(const char* bundle_path,
                                             char*       error_buf,
                                             size_t      error_buf_len);
void cg_session_reader_close(cg_session_reader_ref reader);

int cg_session_reader_format_version(cg_session_reader_ref reader);
/* 1 when the recorder stopped cleanly and finalised the manifest. */
int cg_session_reader_is_complete(cg_session_reader_ref reader);
/* 1 when any stream ends in a partial or corrupt chunk; the records before
 * it are still readable. */
int cg_session_reader_is_truncated(cg_session_reader_ref reader);

const char* cg_session_reader_manifest_json(cg_session_reader_ref reader);
const char* cg_session_reader_library_version(cg_session_reader_ref reader);
/* Wall-clock time the recording started (seconds since Unix epoch). */
double      cg_session_reader_created_at(cg_session_reader_ref reader);

/* extra points into reader-owned storage. Returns 1. */
int cg_session_reader_get_provenance(cg_session_reader_ref reader,
                                     cg_session_provenance* out);
int cg_session_reader_get_model_files(cg_session_reader_ref    reader,
                                      cg_session_file_records* out);
/* Returns 0 when the session was recorded without a recognizer. */
int cg_session_reader_get_config(cg_session_reader_ref reader,
                                 cg_recognizer_config* out,
                                 int*                  bypass_phase2_out);

/* Earliest and latest shot timestamp over all tracks. Returns 0 if no shots. */
int cg_session_reader_time_range(cg_session_reader_ref reader,
                                 double* start_out, double* end_out);

/* Tracks present, ascending (the absent track, -1, first when present). */
int cg_session_reader_track_count(cg_session_reader_ref reader);
int cg_session_reader_track_at(cg_session_reader_ref reader, int i);

/* Shots of one track. read_* return the number of records written to out. */
size_t cg_session_reader_shot_count(cg_session_reader_ref reader, int track_index);
size_t cg_session_reader_read_shots(cg_session_reader_ref reader, int track_index,
                                    size_t first, size_t count, cg_session_shot* out);
/* Index of the first shot with timestamp >= t; shot_count when none. */
size_t cg_session_reader_find_shot(cg_session_reader_ref reader, int track_index, double t);

size_t cg_session_reader_telemetry_count(cg_session_reader_ref reader);
size_t cg_session_reader_read_telemetry(cg_session_reader_ref reader,
                                        size_t first, size_t count,
                                        cg_frame_telemetry* out);
size_t cg_session_reader_find_telemetry(cg_session_reader_ref reader, double t);

/* Events as their JSON lines, in recorded order. A trailing partial line is
 * dropped. Returns NULL when i is out of range. */
size_t      cg_session_reader_event_count(cg_session_reader_ref reader);
const char* cg_session_reader_event_json(cg_session_reader_ref reader, size_t i);

#ifdef __cplusplus
}
#endif

#endif /* CG_SESSION_CAPTURE_H */
