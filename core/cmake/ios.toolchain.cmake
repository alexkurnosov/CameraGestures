# iOS CMake toolchain — delegates to the community ios-cmake toolchain.
# Download ios.toolchain.cmake from:
#   https://github.com/leetal/ios-cmake  (tested with tag 4.4.1)
# and place it alongside this file, or set CMAKE_TOOLCHAIN_FILE to its path.
#
# build-ios.sh passes -DCMAKE_TOOLCHAIN_FILE pointing here; this file then
# includes the real toolchain if present, or errors out with a clear message.

# build-ios.sh clones leetal/ios-cmake into this directory automatically.
# You can also supply your own toolchain by setting CMAKE_TOOLCHAIN_FILE directly.
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/ios-cmake/ios.toolchain.cmake")
    include("${CMAKE_CURRENT_LIST_DIR}/ios-cmake/ios.toolchain.cmake")
else()
    message(FATAL_ERROR
        "iOS toolchain not found. build-ios.sh will clone it automatically;\n"
        "or run manually: git clone --depth 1 --branch 4.4.1 \\\n"
        "  https://github.com/leetal/ios-cmake core/cmake/ios-cmake"
    )
endif()
