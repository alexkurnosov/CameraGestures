#pragma once
#include "CameraGestures/HandsRecognizing.h"
#include <mutex>

/* Internal C++ class backing cg_hands_recognizer_ref.
 *
 * Manages a rolling HandFilm buffer: real shots anchor the start time and
 * open the film; absent shots are only appended after the first real shot
 * (mirrors iOS V1 HandsRecognizing.currentHandfilm logic exactly).
 */
class HandsRecognizer {
public:
    HandsRecognizer();
    ~HandsRecognizer();

    void setCallback(cg_handshot_callback cb, void* userdata);

    /* Push a landmark result. Fires callback unless absent with no prior real frame. */
    void pushHandshot(const cg_handshot& shot);

    /* Snapshot the film, reset buffer, return heap-allocated ref (caller destroys). */
    cg_handfilm_ref harvest();

    /* Discard current film. */
    void reset();

private:
    cg_handshot_callback callback_  = nullptr;
    void*                userdata_  = nullptr;
    cg_handfilm_ref      film_      = nullptr;
    bool                 has_frames_ = false;
    std::mutex           mutex_;
};
