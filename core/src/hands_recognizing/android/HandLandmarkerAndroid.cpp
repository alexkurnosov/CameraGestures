// Android HandLandmarker implementation — Stage 3 stub.
//
// On Android, landmark detection is driven by the Kotlin layer (tasks-vision
// Java API) and results are pushed into the C++ film buffer via JNI
// (bindings/android/src/main/cpp/jni_bridge.cpp → cg_hands_recognizer_push_handshot).
//
// This file is a placeholder for any future C++-side Android helpers
// (e.g., NV21 → BGRA conversion, frame rate throttle logic) that might
// be lifted out of jni_bridge.cpp as the binding matures.
