#include "HandsRecognizer.hpp"
#include "CameraGestures/Types.h"

// ---- C++ class implementation -------------------------------------------

HandsRecognizer::HandsRecognizer() = default;

HandsRecognizer::~HandsRecognizer() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (film_) { cg_handfilm_destroy(film_); film_ = nullptr; }
}

void HandsRecognizer::setCallback(cg_handshot_callback cb, void* userdata) {
    std::lock_guard<std::mutex> lock(mutex_);
    callback_ = cb;
    userdata_ = userdata;
}

void HandsRecognizer::pushHandshot(const cg_handshot& shot) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (shot.is_absent) {
        // V1: absent frames are only added once a real frame has opened the film.
        if (!has_frames_) return;
        cg_handfilm_add_shot(film_, &shot);
        if (callback_) callback_(&shot, userdata_);
        return;
    }

    // Real frame: anchor film start to the first real frame's timestamp.
    if (!has_frames_) {
        film_ = cg_handfilm_create(shot.timestamp);
        has_frames_ = true;
    }
    cg_handfilm_add_shot(film_, &shot);
    if (callback_) callback_(&shot, userdata_);
}

cg_handfilm_ref HandsRecognizer::harvest() {
    std::lock_guard<std::mutex> lock(mutex_);
    cg_handfilm_ref result = film_ ? film_ : cg_handfilm_create(0.0);
    film_       = nullptr;
    has_frames_ = false;
    return result;
}

void HandsRecognizer::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (film_) { cg_handfilm_destroy(film_); film_ = nullptr; }
    has_frames_ = false;
}

// ---- C ABI (thin wrapper over HandsRecognizer) --------------------------

struct cg_hands_recognizer_s {
    HandsRecognizer impl;
};

extern "C" {

cg_hands_recognizer_ref cg_hands_recognizer_create(void) {
    return new cg_hands_recognizer_s{};
}

void cg_hands_recognizer_destroy(cg_hands_recognizer_ref ref) {
    delete ref;
}

void cg_hands_recognizer_set_callback(
    cg_hands_recognizer_ref ref,
    cg_handshot_callback    callback,
    void*                   userdata)
{
    if (ref) ref->impl.setCallback(callback, userdata);
}

void cg_hands_recognizer_push_handshot(
    cg_hands_recognizer_ref ref,
    const cg_handshot*      shot)
{
    if (ref && shot) ref->impl.pushHandshot(*shot);
}

cg_handfilm_ref cg_hands_recognizer_harvest(cg_hands_recognizer_ref ref) {
    return ref ? ref->impl.harvest() : nullptr;
}

void cg_hands_recognizer_reset(cg_hands_recognizer_ref ref) {
    if (ref) ref->impl.reset();
}

} // extern "C"
