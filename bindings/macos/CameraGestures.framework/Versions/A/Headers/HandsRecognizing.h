#ifndef CG_HANDS_RECOGNIZING_H
#define CG_HANDS_RECOGNIZING_H

#include "Types.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle for a HandsRecognizer instance.
 * Manages a rolling HandFilm buffer; platform binding drives landmark detection. */
typedef struct cg_hands_recognizer_s* cg_hands_recognizer_ref;

/* Fired for every handshot pushed (real and absent), once a film is in progress.
 * Absent shots with no prior real frame are silently dropped (mirrors V1 behavior). */
typedef void (*cg_handshot_callback)(const cg_handshot* shot, void* userdata);

/* Create / destroy. */
cg_hands_recognizer_ref cg_hands_recognizer_create(void);
void                    cg_hands_recognizer_destroy(cg_hands_recognizer_ref);

/* Register a callback fired on each accepted push_handshot. */
void cg_hands_recognizer_set_callback(
    cg_hands_recognizer_ref,
    cg_handshot_callback callback,
    void* userdata);

/* Push a handshot (landmark result already converted by the platform binding).
 * On iOS/Android: Swift/Kotlin calls MediaPipe and converts results before calling this.
 * On macOS (stretch goal): C++ calls this internally after LiteRT inference. */
void cg_hands_recognizer_push_handshot(
    cg_hands_recognizer_ref,
    const cg_handshot* shot);

/* Harvest the accumulated film and reset the internal buffer.
 * Returns a heap-allocated cg_handfilm_ref; caller MUST call cg_handfilm_destroy. */
cg_handfilm_ref cg_hands_recognizer_harvest(cg_hands_recognizer_ref);

/* Reset without returning the film. */
void cg_hands_recognizer_reset(cg_hands_recognizer_ref);

/* ---- macOS LiteRT HandLandmarker (macOS only) ----------------------------
 * Used by bindings/macos/Sources/HandsRecognizing.swift to run two-stage
 * TFLite inference (palm detector → landmark model) without MediaPipe SDK.
 * Only compiled when CG_BUILD_MACOS_LANDMARKER is defined (set by the macOS
 * CMake build and the macOS Xcode framework target). */
#ifdef CG_BUILD_MACOS_LANDMARKER

typedef struct cg_hand_landmarker_lrt_s* cg_hand_landmarker_lrt_ref;

/* Create and load. Returns NULL if the .task bundle cannot be opened. */
cg_hand_landmarker_lrt_ref cg_hand_landmarker_lrt_create(
    const char*             task_path,
    cg_hands_recognizer_ref recognizer);

void cg_hand_landmarker_lrt_destroy(cg_hand_landmarker_lrt_ref);

/* Push a BGRA8 frame. stride = bytes per row (>= width*4). timestamp in seconds. */
void cg_hand_landmarker_lrt_push_frame(
    cg_hand_landmarker_lrt_ref,
    const uint8_t* bgra,
    int width, int height, int stride,
    double timestamp);

#endif /* CG_BUILD_MACOS_LANDMARKER */

#ifdef __cplusplus
}
#endif

#endif /* CG_HANDS_RECOGNIZING_H */
