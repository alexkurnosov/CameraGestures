cmake_minimum_required(VERSION 3.21)

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../VERSION" _version_raw)
string(STRIP "${_version_raw}" PROJECT_VERSION_STR)

project(CameraGestures VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Options ---------------------------------------------------------------

# Enable TFLite inference backend. Required for gesture/pose classification at runtime.
# OFF by default so the library and tests build without a vendored TFLite binary
# (FeaturePreprocessor and PoseManifest compile and test fine without it).
option(CG_ENABLE_TFLITE "Build with TFLite inference backend" OFF)

# ---- Third-party includes --------------------------------------------------

set(NLOHMANN_JSON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/nlohmann_json")
set(TFLITE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/tflite/include")

# ---- Library sources (accumulated per stage) --------------------------------

set(CG_SOURCES
    src/CameraGestures.cpp
    # Stage 2: HandGestureTypes
    src/types/types.cpp
    src/types/gesture_registry.cpp
    # Stage 3: HandsRecognizing (platform-agnostic buffer manager)
    src/hands_recognizing/HandsRecognizer.cpp
    # Stage 4: GestureModel
    src/gesture_model/FeaturePreprocessor.cpp
    src/gesture_model/PoseManifest.cpp
    src/gesture_model/TFLiteBackend.cpp
    src/gesture_model/GestureModel.cpp
    # Stage 5: HandGestureRecognizing (3-phase pipeline)
    src/hand_gesture_recognizing/MotionGate.cpp
    src/hand_gesture_recognizing/HoldDetector.cpp
    src/hand_gesture_recognizing/PrefixMatcher.cpp
    src/hand_gesture_recognizing/HandGestureRecognizing.cpp
)

# Stage 7: macOS LiteRT HandLandmarker (only when APPLE and not IOS)
if(APPLE AND NOT IOS)
    list(APPEND CG_SOURCES
        src/hands_recognizing/macos/TaskBundleParser.cpp
        src/hands_recognizing/macos/AnchorDecoder.cpp
        src/hands_recognizing/macos/HandROICropper.cpp
        src/hands_recognizing/macos/HandLandmarkerLiteRT.cpp
    )
endif()

add_library(CameraGestures STATIC ${CG_SOURCES})

target_include_directories(CameraGestures
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${NLOHMANN_JSON_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src/hands_recognizing
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gesture_model
        ${CMAKE_CURRENT_SOURCE_DIR}/src/hand_gesture_recognizing
        ${CMAKE_CURRENT_SOURCE_DIR}/src/hands_recognizing/macos
)

if(CG_ENABLE_TFLITE)
    target_compile_definitions(CameraGestures PRIVATE CG_ENABLE_TFLITE)
    target_include_directories(CameraGestures PRIVATE ${TFLITE_INCLUDE_DIR})

    # Platform-specific TFLite prebuilt library
    if(IOS)
        # TFLite is linked via the XCFramework in the podspec; nothing needed here.
    elseif(ANDROID)
        find_library(TFLITE_LIB tensorflowlite_jni
            PATHS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tflite/android/${ANDROID_ABI}
            NO_DEFAULT_PATH)
        if(TFLITE_LIB)
            target_link_libraries(CameraGestures PRIVATE ${TFLITE_LIB})
        endif()
    elseif(APPLE AND NOT IOS)
        # macOS: static TFLite re-platformed from iOS Simulator xcframework.
        # Run core/scripts/vendor-tflite-macos.sh to generate this file.
        find_library(TFLITE_LIB TensorFlowLiteC
            PATHS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tflite/macos
            NO_DEFAULT_PATH)
        if(TFLITE_LIB)
            target_link_libraries(CameraGestures PRIVATE ${TFLITE_LIB})
        endif()
        # Enable macOS LiteRT HandLandmarker C ABI
        target_compile_definitions(CameraGestures PUBLIC CG_BUILD_MACOS_LANDMARKER)
    endif()
endif()

install(TARGETS CameraGestures
    ARCHIVE DESTINATION lib
)
install(DIRECTORY include/CameraGestures DESTINATION include)

# ---- Tests -----------------------------------------------------------------

option(CG_BUILD_TESTS "Build gtest unit tests" OFF)
if(CG_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
