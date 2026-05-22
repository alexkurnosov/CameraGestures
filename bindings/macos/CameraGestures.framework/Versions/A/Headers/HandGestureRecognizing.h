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
} cg_recognizer_config;

/* Returns a config struct pre-filled with the same defaults as the iOS V1 pod. */
cg_recognizer_config cg_recognizer_default_config(void);

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
 * observed_seq / n_observed: current pose ID sequence. */
typedef void (*cg_holds_telemetry_callback)(void* context,
                                             int    pose_id,
                                             float  confidence,
                                             const int* observed_seq,
                                             int    n_observed,
                                             const char* matched_gesture);

void cg_recognizer_set_gesture_callback(cg_recognizer_ref         recognizer,
                                         cg_gesture_callback       callback,
                                         void*                     context);

void cg_recognizer_set_gate_update_callback(cg_recognizer_ref       recognizer,
                                             cg_gate_update_callback callback,
                                             void*                   context);

void cg_recognizer_set_holds_telemetry_callback(cg_recognizer_ref           recognizer,
                                                 cg_holds_telemetry_callback callback,
                                                 void*                       context);

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

#ifdef __cplusplus
}
#endif

#endif /* CG_HAND_GESTURE_RECOGNIZING_H */
