#ifndef CG_HAND_GESTURE_RECOGNIZING_H
#define CG_HAND_GESTURE_RECOGNIZING_H

#include "Types.h"
#include "GestureModel.h"

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * Opaque handle
 * ---------------------------------------------------------------------- */

typedef struct cg_recognizer_s* cg_recognizer_ref;

/* -------------------------------------------------------------------------
 * Configuration
 * ---------------------------------------------------------------------- */

/* Phase-1 motion gate parameters. */
typedef struct cg_motion_gate_config {
    float  t_open;       /* energy threshold to open the gate */
    double k_open_ms;    /* gate opens after exceeding t_open for this many ms */
    float  t_close;      /* energy threshold to close the gate */
    double k_close_ms;   /* gate closes after falling below t_close for this many ms */
    double cooldown_ms;  /* post-cycle cooldown (handled in Swift layer) */
} cg_motion_gate_config;

/* Phase-2 holds-mode parameters. */
typedef struct cg_holds_config {
    float  t_hold;                /* smoothed energy threshold for a hold run */
    double k_hold_ms;             /* minimum hold duration (ms) */
    double smooth_k_ms;           /* smoothing window for energy (ms) */
    double t_commit_ms;           /* T_commit timer duration (ms) */
    double t_min_buffer_ms;       /* minimum gate-open duration before commit (ms) */
    float  tau_pose_confidence;   /* min pose-classifier confidence */
    float  tau_phase3_confidence; /* Phase-3 masked-argmax threshold */
} cg_holds_config;

/* Full recognizer configuration. */
typedef struct cg_recognizer_config {
    int                  gate_enabled;       /* bool: 1 to enable Phase-1 gate */
    cg_motion_gate_config motion_gate;
    int                  gesture_buffer_size; /* rolling buffer cap (frames) */
    int                  holds_enabled;       /* bool: 1 to enable Phase-2 holds mode */
    cg_holds_config      holds;
    float                confidence_threshold; /* Phase-3 threshold (unrestricted path) */
    int                  retain_landmarks_for_review; /* bool: 1 to populate hold landmarks
                                                         in holds-telemetry callback */
} cg_recognizer_config;

/* Returns a config struct pre-filled with the same defaults as the iOS V1 pod. */
cg_recognizer_config cg_recognizer_default_config(void);

/* -------------------------------------------------------------------------
 * Telemetry — what the library decided, as it decided it
 *
 * These report the recognizer's own values; they are never recomputed. A
 * frame's decision events are delivered before its telemetry row.
 * ---------------------------------------------------------------------- */

/* One row per cg_recognizer_process_shot call, emitted after the shot has been
 * fully processed (gate, Phase 2, any commit it triggered). Fixed-width fields:
 * the row is meant to be written to disk verbatim. */
typedef struct cg_frame_telemetry {
    double  timestamp;            /* the shot's timestamp */
    double  commit_deadline;      /* pending T_commit deadline after this shot; 0 = none pending */
    double  min_buffer_deadline;  /* pending T_min_buffer deadline; 0 = none or already satisfied */
    float   raw_energy;           /* the energy MotionGate compared against t_open / t_close */
    float   smoothed_energy;      /* HoldDetector's smoothed energy, compared against t_hold */
    float   hold_run_ms;          /* duration of the current below-t_hold run; 0 = not in a run */
    int32_t hold_run_frames;      /* frames in that run; 0 = not in a run */
    int32_t track_index;          /* MediaPipe per-frame hand index; -1 = not known to the recognizer */
    int32_t buffer_count;         /* MotionGate buffer count after this shot */
    uint8_t handedness;           /* the shot's cg_handedness */
    uint8_t is_absent;
    uint8_t gate_enabled;         /* 0 = Phase 1 disabled; the shot was not processed */
    uint8_t gate_open;            /* gate state after this shot, including Phase-2 resets */
    uint8_t raw_energy_valid;     /* 0 = no previous frame to diff against (the gate used 0),
                                     absent shot, or gate disabled */
    uint8_t smoothed_energy_valid;/* 1 only when HoldDetector ran on this shot: gate open,
                                     holds mode on, pose model loaded. hold_run_* are
                                     meaningful only when this is 1. */
    uint8_t reserved[2];
} cg_frame_telemetry;

typedef enum cg_decision_event_kind {
    CG_EVENT_GATE_OPENED       = 0,
    CG_EVENT_CYCLE_ENDED       = 1, /* reason: cg_cycle_end_reason */
    CG_EVENT_CYCLE_SKIPPED     = 2, /* reason: cg_cycle_skip_reason — no Phase 3 ran */
    CG_EVENT_HOLD_COMPLETED    = 3, /* reason: cg_prefix_action */
    CG_EVENT_PHASE3_PREDICTION = 4, /* reason: 0 */
    CG_EVENT_COMMIT_FIRED      = 5  /* reason: cg_commit_gate */
} cg_decision_event_kind;

typedef enum cg_cycle_end_reason {
    CG_CYCLE_END_ABSENT_FRAME   = 0, /* gate: the hand left the frame */
    CG_CYCLE_END_LOW_ENERGY     = 1, /* gate: energy below t_close for k_close_ms */
    CG_CYCLE_END_BUFFER_CAP     = 2, /* gate: buffer reached gesture_buffer_size */
    CG_CYCLE_END_PHASE2_DISCARD = 3, /* Phase 2 reset the gate: no prefix, idle reset or discard */
    CG_CYCLE_END_COMMITTED      = 4, /* a Phase-2 commit reset the gate */
    CG_CYCLE_END_EXTERNAL_RESET = 5  /* cg_recognizer_reset_gate / gate disabled while open */
} cg_cycle_end_reason;

typedef enum cg_cycle_skip_reason {
    CG_CYCLE_SKIP_ALREADY_COMMITTED = 0, /* Phase 2 already committed this cycle */
    CG_CYCLE_SKIP_EMPTY_BUFFER      = 1,
    CG_CYCLE_SKIP_NO_MODEL          = 2, /* no gesture model loaded */
    CG_CYCLE_SKIP_TOO_FEW_FRAMES    = 3  /* fewer frames than recognition needs */
} cg_cycle_skip_reason;

typedef enum cg_commit_gate {
    CG_COMMIT_IMMEDIATE    = 0, /* fired on the hold itself; no deadline was pending */
    CG_COMMIT_T_COMMIT     = 1, /* fired by the timer; T_commit was the later deadline */
    CG_COMMIT_T_MIN_BUFFER = 2  /* fired by the timer; T_min_buffer was the later deadline */
} cg_commit_gate;

typedef enum cg_prefix_action {
    CG_PREFIX_NOT_OBSERVED       = 0, /* pose rejected (low confidence / prediction failed) */
    CG_PREFIX_NO_PREFIX          = 1,
    CG_PREFIX_LIVE_PREFIX        = 2,
    CG_PREFIX_COMMIT_NOW         = 3,
    CG_PREFIX_START_COMMIT_TIMER = 4,
    CG_PREFIX_IDLE_RESET         = 5,
    CG_PREFIX_IDLE_DISCARD       = 6,
    CG_PREFIX_IDLE_COMMIT        = 7
} cg_prefix_action;

/* A discrete decision. Fields not listed for an event's kind are zero / NULL.
 * All pointers are valid only for the duration of the callback. */
typedef struct cg_decision_event {
    int    kind;          /* cg_decision_event_kind */
    int    reason;        /* per kind, see cg_decision_event_kind */
    double timestamp;     /* shot timestamp, or the tick's `now` for timer-fired events */
    int    buffer_count;  /* CYCLE_ENDED, CYCLE_SKIPPED, PHASE3_PREDICTION: frames in the cycle */

    /* HOLD_COMPLETED */
    int    pose_id;             /* -1 if the pose prediction failed */
    double hold_start_time;
    double rep_shot_time;       /* the representative (minimum-energy) frame classified */
    const int* observed_seq;    /* pose sequence after this hold */
    int    n_observed;

    /* HOLD_COMPLETED: pose confidence. PHASE3_PREDICTION: confidence of the
     * emitted gesture, 0 when nothing was emitted. */
    float  confidence;
    /* HOLD_COMPLETED: confidence >= tau_pose_confidence.
     * PHASE3_PREDICTION: a gesture was emitted. */
    int    accepted;

    /* PHASE3_PREDICTION */
    int         restricted;     /* 1 = masked argmax over candidate_ids, 0 = unrestricted */
    const char* gesture_id;     /* emitted gesture, "" when none */

    /* PHASE3_PREDICTION (restricted) and COMMIT_FIRED */
    const char* const* candidate_ids;
    int                n_candidates;

    /* COMMIT_FIRED: the pending deadlines at the moment it fired */
    double commit_deadline;
    double min_buffer_deadline;
} cg_decision_event;

/* Stable lowercase names for logs and files, e.g. "cycle_ended", "low_energy".
 * Never NULL; unknown values return "unknown". */
const char* cg_decision_event_kind_name(int kind);
const char* cg_decision_event_reason_name(int kind, int reason);

/* -------------------------------------------------------------------------
 * Lifecycle
 * ---------------------------------------------------------------------- */

/* Create a recognizer with the given config.
 * gesture_model: a fully loaded cg_gesture_model_ref (may be NULL for testing;
 *                recognizer will not emit gestures until a model is attached).
 * Returns NULL on allocation failure. */
cg_recognizer_ref cg_recognizer_create(const cg_recognizer_config* config,
                                        cg_gesture_model_ref gesture_model);

void cg_recognizer_destroy(cg_recognizer_ref recognizer);

/* -------------------------------------------------------------------------
 * Callbacks
 * ---------------------------------------------------------------------- */

/* Called when a gesture is detected.
 * context: opaque pointer forwarded to callback.
 * film:    transient cg_handfilm_ref — valid only for the duration of the call.
 *          Do NOT destroy it; do NOT retain it past the callback. */
typedef void (*cg_gesture_callback)(void* context,
                                    const cg_gesture_prediction* prediction,
                                    cg_handfilm_ref              film,
                                    int                          candidate_set_size);

/* Called when Phase-1 gate state or buffer count changes. */
typedef void (*cg_gate_update_callback)(void* context, int gate_open, int buffer_count);

/* Called when Phase-2 detects a hold (holds mode only).
 * observed_seq / n_observed: current pose ID sequence.
 * rep_shot / normalized_coords: non-null only when retain_landmarks_for_review=1;
 *   both pointers are valid only for the duration of the call. */
typedef void (*cg_holds_telemetry_callback)(void* context,
                                             int    pose_id,
                                             float  confidence,
                                             const int* observed_seq,
                                             int    n_observed,
                                             const char* matched_gesture,
                                             const cg_handshot* rep_shot,
                                             const float* normalized_coords,
                                             int    normalized_coords_len);

void cg_recognizer_set_gesture_callback(cg_recognizer_ref         recognizer,
                                         cg_gesture_callback       callback,
                                         void*                     context);

void cg_recognizer_set_gate_update_callback(cg_recognizer_ref       recognizer,
                                             cg_gate_update_callback callback,
                                             void*                   context);

void cg_recognizer_set_holds_telemetry_callback(cg_recognizer_ref           recognizer,
                                                 cg_holds_telemetry_callback callback,
                                                 void*                       context);

/* Called once per processed shot. row is valid only for the duration of the call. */
typedef void (*cg_frame_telemetry_callback)(void* context, const cg_frame_telemetry* row);

/* Called for each decision event. event is valid only for the duration of the call. */
typedef void (*cg_decision_event_callback)(void* context, const cg_decision_event* event);

void cg_recognizer_set_frame_telemetry_callback(cg_recognizer_ref           recognizer,
                                                 cg_frame_telemetry_callback callback,
                                                 void*                       context);

void cg_recognizer_set_decision_event_callback(cg_recognizer_ref          recognizer,
                                                cg_decision_event_callback callback,
                                                void*                      context);

/* -------------------------------------------------------------------------
 * Runtime control
 * ---------------------------------------------------------------------- */

/* Push one camera frame through the pipeline.
 * Must be called from a single serial queue/thread. */
void cg_recognizer_process_shot(cg_recognizer_ref    recognizer,
                                 const cg_handshot*   shot);

/* Tick T_commit / T_min_buffer timers.
 * now: current time (seconds since Unix epoch).
 * Must be called periodically (e.g. every 10 ms) from the same serial queue.
 * Returns 1 if a commit was triggered this tick. */
int cg_recognizer_tick_timers(cg_recognizer_ref recognizer, double now);

/* Reset all Phase-1 and Phase-2 state. */
void cg_recognizer_reset_gate(cg_recognizer_ref recognizer);

/* Enable / disable Phase-1 gate at runtime. */
void cg_recognizer_set_gate_enabled(cg_recognizer_ref recognizer, int enabled);

/* Enable / disable Phase-2 bypass (forces Phase-3 to run unrestricted). */
void cg_recognizer_set_bypass_phase2(cg_recognizer_ref recognizer, int bypass);

/* Attach a (re)loaded gesture model. */
void cg_recognizer_set_gesture_model(cg_recognizer_ref    recognizer,
                                      cg_gesture_model_ref model);

/* The config in effect: the one passed to cg_recognizer_create, with
 * gate_enabled as last set by cg_recognizer_set_gate_enabled.
 * Returns 1 on success. */
int cg_recognizer_get_config(cg_recognizer_ref recognizer, cg_recognizer_config* out);

/* Returns 1 while Phase-2 bypass is on. */
int cg_recognizer_get_bypass_phase2(cg_recognizer_ref recognizer);

#ifdef __cplusplus
}
#endif

#endif /* CG_HAND_GESTURE_RECOGNIZING_H */
