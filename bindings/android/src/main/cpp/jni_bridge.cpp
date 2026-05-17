// JNI bridge — exposes the C ABI cg_hands_recognizer_* functions to Kotlin.
//
// Stage 3: HandsRecognizing only. The Kotlin layer calls MediaPipe tasks-vision
// (Java API) and then calls these JNI functions to push handshots into the C++
// film buffer.

#include <jni.h>
#include <android/log.h>
#include "CameraGestures/HandsRecognizing.h"

#define TAG "CameraGestures"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

// ---- Helpers ----------------------------------------------------------------

// Kotlin/Java long ↔ C++ pointer (safe on all ABIs CocoaPods targets).
static inline cg_hands_recognizer_ref fromHandle(jlong h) {
    return reinterpret_cast<cg_hands_recognizer_ref>(static_cast<uintptr_t>(h));
}

// ---- JNI exports ------------------------------------------------------------

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_cameragestures_HandsRecognizerNative_create(JNIEnv*, jobject) {
    auto* ref = cg_hands_recognizer_create();
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(ref));
}

JNIEXPORT void JNICALL
Java_com_cameragestures_HandsRecognizerNative_destroy(JNIEnv*, jobject, jlong handle) {
    cg_hands_recognizer_destroy(fromHandle(handle));
}

// Push a single handshot from Kotlin.
// landmarks: float[63] (21 × {x,y,z}), timestamp: double seconds,
// handedness: 0=left 1=right 2=unknown, isAbsent: 1 or 0.
JNIEXPORT void JNICALL
Java_com_cameragestures_HandsRecognizerNative_pushHandshot(
    JNIEnv*  env,
    jobject,
    jlong    handle,
    jfloatArray landmarks,
    jdouble  timestamp,
    jint     handedness,
    jint     isAbsent)
{
    auto* ref = fromHandle(handle);
    if (!ref) return;

    cg_handshot shot{};
    shot.timestamp  = timestamp;
    shot.handedness = static_cast<cg_handedness>(handedness);
    shot.is_absent  = isAbsent;

    if (!isAbsent) {
        jfloat* pts = env->GetFloatArrayElements(landmarks, nullptr);
        if (pts) {
            for (int i = 0; i < 21; ++i) {
                shot.landmarks[i].x = pts[i * 3 + 0];
                shot.landmarks[i].y = pts[i * 3 + 1];
                shot.landmarks[i].z = pts[i * 3 + 2];
            }
            env->ReleaseFloatArrayElements(landmarks, pts, JNI_ABORT);
        }
    }

    cg_hands_recognizer_push_handshot(ref, &shot);
}

// Harvest the film into a flat float array: [startTime, shotCount, shot0…].
// Each shot: [ts, handedness, isAbsent, lm0x, lm0y, lm0z, …, lm20z] = 3 + 63 = 66 floats.
// Caller converts to Kotlin data classes.
JNIEXPORT jfloatArray JNICALL
Java_com_cameragestures_HandsRecognizerNative_harvest(
    JNIEnv* env,
    jobject,
    jlong   handle)
{
    auto* ref = fromHandle(handle);
    cg_handfilm_ref film = ref ? cg_hands_recognizer_harvest(ref) : nullptr;
    if (!film) {
        return env->NewFloatArray(0);
    }

    size_t count   = cg_handfilm_shot_count(film);
    double startTs = cg_handfilm_start_time(film);

    // Layout: [startTime, shotCount, shot0…shotN]  shot = 66 floats
    jsize  total   = static_cast<jsize>(2 + count * 66);
    jfloatArray arr = env->NewFloatArray(total);
    if (!arr) { cg_handfilm_destroy(film); return env->NewFloatArray(0); }

    jfloat* buf = env->GetFloatArrayElements(arr, nullptr);
    buf[0] = static_cast<jfloat>(startTs);
    buf[1] = static_cast<jfloat>(count);

    for (size_t i = 0; i < count; ++i) {
        cg_handshot shot{};
        if (!cg_handfilm_get_shot(film, i, &shot)) continue;
        jfloat* base = buf + 2 + i * 66;
        base[0] = static_cast<jfloat>(shot.timestamp);
        base[1] = static_cast<jfloat>(shot.handedness);
        base[2] = static_cast<jfloat>(shot.is_absent);
        for (int j = 0; j < 21; ++j) {
            base[3 + j * 3 + 0] = shot.landmarks[j].x;
            base[3 + j * 3 + 1] = shot.landmarks[j].y;
            base[3 + j * 3 + 2] = shot.landmarks[j].z;
        }
    }

    env->ReleaseFloatArrayElements(arr, buf, 0);
    cg_handfilm_destroy(film);
    return arr;
}

JNIEXPORT void JNICALL
Java_com_cameragestures_HandsRecognizerNative_reset(JNIEnv*, jobject, jlong handle) {
    cg_hands_recognizer_reset(fromHandle(handle));
}

} // extern "C"
