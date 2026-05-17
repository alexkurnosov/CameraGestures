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
)

target_link_libraries(cg_tests PRIVATE
    CameraGestures
    GTest::gtest_main
)

target_include_directories(cg_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party/nlohmann_json
)

include(GoogleTest)
gtest_discover_tests(cg_tests)
