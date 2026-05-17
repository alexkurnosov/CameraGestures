cmake_minimum_required(VERSION 3.22)
project(cameragestures_jni)

# Prebuilt core static library (built by core/scripts/build-android.sh).
set(PREBUILT_DIR "${CAMERAGESTURES_PREBUILT_DIR}/${ANDROID_ABI}")

add_library(CameraGesturesCore STATIC IMPORTED)
set_target_properties(CameraGesturesCore PROPERTIES
    IMPORTED_LOCATION "${PREBUILT_DIR}/libCameraGestures.a"
)

add_library(cameragestures SHARED
    jni_bridge.cpp
)

target_include_directories(cameragestures PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../../../core/include"
)

target_link_libraries(cameragestures
    CameraGesturesCore
    android
    log
)
