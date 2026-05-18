cmake_minimum_required(VERSION 3.21)

include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
    GIT_SHALLOW    TRUE
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(cg_tests
    test_types.cpp
    test_gesture_registry.cpp
    # Stage 4: GestureModel unit tests (preprocessor parity + PoseManifest JSON)
    test_gesture_model.cpp
)

target_link_libraries(cg_tests PRIVATE
    CameraGestures
    GTest::gtest_main
)

target_include_directories(cg_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party/nlohmann_json
    ${CMAKE_SOURCE_DIR}/src/gesture_model
)

include(GoogleTest)
# Run tests from the source tests/ directory so "fixtures/" is a valid relative path.
gtest_discover_tests(cg_tests WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
