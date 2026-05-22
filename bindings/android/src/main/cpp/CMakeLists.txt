cmake_minimum_required(VERSION 3.22)
project(cameragestures_jni)

# Prebuilt core static library (built by core/scripts/build-android.sh).
set(PREBUILT_DIR "${CAMERAGESTURES_PREBUILT_DIR}/${ANDROID_ABI}")

add_library(CameraGesturesCore STATIC IMPORTED)
set_target_properties(CameraGesturesCore PROPERTIES
    IMPORTED_LOCATION "${PREBUILT_DIR}/libCameraGestures.a"
)

# TFLite shared library — built alongside the core .a by build-android.sh.
# Importing it here causes AGP to package it into the APK automatically.
# CMAKE_CURRENT_SOURCE_DIR = bindings/android/src/main/cpp — 5 levels up is repo root.
get_filename_component(TFLITE_SO
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../core/third_party/tflite/android/${ANDROID_ABI}/libtensorflowlite_jni.so"
    ABSOLUTE)
if(EXISTS "${TFLITE_SO}")
    add_library(tensorflowlite_jni SHARED IMPORTED)
    set_target_properties(tensorflowlite_jni PROPERTIES
        IMPORTED_LOCATION "${TFLITE_SO}"
    )
    set(TFLITE_LINK_TARGET tensorflowlite_jni)
else()
    message(WARNING "libtensorflowlite_jni.so not found for ${ANDROID_ABI} — TFLite inference disabled")
    set(TFLITE_LINK_TARGET "")
endif()

add_library(cameragestures SHARED
    jni_bridge.cpp
)

target_include_directories(cameragestures PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../core/include"
)

target_link_libraries(cameragestures
    CameraGesturesCore
    ${TFLITE_LINK_TARGET}
    android
    log
)
