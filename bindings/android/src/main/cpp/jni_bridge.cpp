// JNI bridge — exposes the C ABI to Kotlin.
//
// Stage 3: cg_hands_recognizer_* (HandsRecognizing film buffer)
// Stage 6: cg_gesture_model_* (GestureModel) + cg_recognizer_* (HandGestureRecognizing)
//
// The Kotlin layer calls MediaPipe tasks-vision for landmark detection and then
// pushes handshots here; the C++ side runs GestureModel + 3-phase recognizer.

#include <jni.h>
#include <android/log.h>
#include <mutex>
#include <cstring>
#include <cstdio>
#include "CameraGestures/HandsRecognizing.h"
#include "CameraGestures/GestureModel.h"
#include "CameraGestures/HandGestureRecognizing.h"

#define TAG "CameraGestures"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  TAG, __VA_ARGS__)

// ---- Helpers ----------------------------------------------------------------

static inline cg_hands_recognizer_ref fromHsHandle(jlong h) {
    return reinterpret_cast<cg_hands_recognizer_ref>(static_cast<uintptr_t>(h));
}

static inline cg_gesture_model_ref fromModelHandle(jlong h) {
    return reinterpret_cast<cg_gesture_model_ref>(static_cast<uintptr_t>(h));
}

// ---- Recognizer wrapper (Stage 6) ------------------------------------------
// Wraps cg_recognizer_ref and stores the latest gesture for polling from Kotlin.

struct RecognizerHandle {
    cg_recognizer_ref  recognizer  = nullptr;
    std::mutex         mu;
    bool               has_gesture = false;
    cg_gesture_prediction last_gesture{};

    static void onGesture(void*                    ctx,
                           const cg_gesture_prediction* pred,
                           cg_handfilm_ref,
                           int)
    {
        if (!pred) return;
        auto* self = static_cast<RecognizerHandle*>(ctx);
        std::lock_guard<std::mutex> g(self->mu);
        self->last_gesture = *pred;
        self->has_gesture  = true;
    }
};

static inline RecognizerHandle* fromRecHandle(jlong h) {
    return reinterpret_cast<RecognizerHandle*>(static_cast<uintptr_t>(h));
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
    cg_hands_recognizer_destroy(fromHsHandle(handle));
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
    auto* ref = fromHsHandle(handle);
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
    auto* ref = fromHsHandle(handle);
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
    cg_hands_recognizer_reset(fromHsHandle(handle));
}

// ============================================================================
// GestureModel JNI (Stage 6)
// ============================================================================

JNIEXPORT jlong JNICALL
Java_com_cameragestures_GestureModelNative_load(
    JNIEnv* env, jobject, jstring jtflite, jstring jregistry)
{
    const char* tflite   = env->GetStringUTFChars(jtflite,   nullptr);
    const char* registry = env->GetStringUTFChars(jregistry, nullptr);
    LOGI("GestureModel: loading tflite=%s registry=%s", tflite, registry);

    // Step 1: check registry parses
    cg_registry_ref reg = cg_registry_create(registry);
    if (!reg) {
        LOGE("GestureModel: cg_registry_create failed — bad gestures.json at %s", registry);
        env->ReleaseStringUTFChars(jtflite,   tflite);
        env->ReleaseStringUTFChars(jregistry, registry);
        return 0L;
    }
    int reg_count = (int)cg_registry_count(reg);
    LOGI("GestureModel: registry loaded ok, %d gestures", reg_count);
    cg_registry_destroy(reg);

    auto* ref = cg_gesture_model_load(tflite, registry);
    env->ReleaseStringUTFChars(jtflite,   tflite);
    env->ReleaseStringUTFChars(jregistry, registry);
    if (!ref) LOGE("GestureModel: cg_gesture_model_load failed (TFLite alloc or dim mismatch)");
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(ref));
}

JNIEXPORT void JNICALL
Java_com_cameragestures_GestureModelNative_destroy(JNIEnv*, jobject, jlong handle) {
    cg_gesture_model_destroy(fromModelHandle(handle));
}

JNIEXPORT jint JNICALL
Java_com_cameragestures_GestureModelNative_loadPose(
    JNIEnv* env, jobject, jlong handle, jstring jtflite, jstring jmanifest)
{
    const char* tflite   = env->GetStringUTFChars(jtflite,   nullptr);
    const char* manifest = env->GetStringUTFChars(jmanifest, nullptr);
    int ok = cg_gesture_model_load_pose(fromModelHandle(handle), tflite, manifest);
    env->ReleaseStringUTFChars(jtflite,   tflite);
    env->ReleaseStringUTFChars(jmanifest, manifest);
    return ok;
}

// ============================================================================
// HandGestureRecognizing JNI (Stage 6)
// ============================================================================

JNIEXPORT jlong JNICALL
Java_com_cameragestures_RecognizerNative_create(JNIEnv*, jobject, jlong modelHandle) {
    auto* model = fromModelHandle(modelHandle);
    auto  cfg   = cg_recognizer_default_config();
    // For the demo, enable the motion gate so recognition is gated on motion.
    cfg.gate_enabled = 1;

    auto* w = new RecognizerHandle{};
    w->recognizer = cg_recognizer_create(&cfg, model);
    if (!w->recognizer) {
        LOGE("Recognizer: cg_recognizer_create failed");
        delete w;
        return 0L;
    }
    cg_recognizer_set_gesture_callback(w->recognizer, RecognizerHandle::onGesture, w);
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(w));
}

JNIEXPORT void JNICALL
Java_com_cameragestures_RecognizerNative_destroy(JNIEnv*, jobject, jlong handle) {
    auto* w = fromRecHandle(handle);
    if (!w) return;
    cg_recognizer_destroy(w->recognizer);
    delete w;
}

JNIEXPORT void JNICALL
Java_com_cameragestures_RecognizerNative_processShot(
    JNIEnv*     env,
    jobject,
    jlong       handle,
    jfloatArray landmarks,
    jdouble     timestamp,
    jint        handedness,
    jint        isAbsent)
{
    auto* w = fromRecHandle(handle);
    if (!w) return;

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

    cg_recognizer_process_shot(w->recognizer, &shot);
}

JNIEXPORT jint JNICALL
Java_com_cameragestures_RecognizerNative_tickTimers(JNIEnv*, jobject, jlong handle, jdouble now) {
    auto* w = fromRecHandle(handle);
    if (!w) return 0;
    return cg_recognizer_tick_timers(w->recognizer, now);
}

JNIEXPORT void JNICALL
Java_com_cameragestures_RecognizerNative_reset(JNIEnv*, jobject, jlong handle) {
    auto* w = fromRecHandle(handle);
    if (w) cg_recognizer_reset_gate(w->recognizer);
}

// Returns "gesture_name|confidence" if a gesture is pending, null otherwise.
JNIEXPORT jstring JNICALL
Java_com_cameragestures_RecognizerNative_pollGesture(JNIEnv* env, jobject, jlong handle) {
    auto* w = fromRecHandle(handle);
    if (!w) return nullptr;

    std::lock_guard<std::mutex> g(w->mu);
    if (!w->has_gesture) return nullptr;

    char buf[400];
    snprintf(buf, sizeof(buf), "%s|%.4f",
             w->last_gesture.gesture_name,
             static_cast<double>(w->last_gesture.confidence));
    w->has_gesture = false;
    return env->NewStringUTF(buf);
}

} // extern "C"
