cmake_minimum_required(VERSION 3.21)

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../VERSION" _version_raw)
string(STRIP "${_version_raw}" PROJECT_VERSION_STR)

project(CameraGestures VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(CameraGestures STATIC
    src/CameraGestures.cpp
)

target_include_directories(CameraGestures
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

install(TARGETS CameraGestures
    ARCHIVE DESTINATION lib
)
install(DIRECTORY include/CameraGestures DESTINATION include)
