cmake_minimum_required(VERSION 3.21)

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../VERSION" _version_raw)
string(STRIP "${_version_raw}" PROJECT_VERSION_STR)

project(CameraGestures VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Third-party includes ------------------------------------------------

set(NLOHMANN_JSON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/nlohmann_json")

# ---- Library sources (accumulated per stage) ----------------------------

add_library(CameraGestures STATIC
    src/CameraGestures.cpp
    # Stage 2: HandGestureTypes
    src/types/types.cpp
    src/types/gesture_registry.cpp
    # Stage 3: HandsRecognizing (platform-agnostic buffer manager)
    src/hands_recognizing/HandsRecognizer.cpp
)

target_include_directories(CameraGestures
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${NLOHMANN_JSON_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src/hands_recognizing
)

install(TARGETS CameraGestures
    ARCHIVE DESTINATION lib
)
install(DIRECTORY include/CameraGestures DESTINATION include)

# ---- Tests ---------------------------------------------------------------

option(CG_BUILD_TESTS "Build gtest unit tests" OFF)
if(CG_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
