cmake_minimum_required(VERSION 3.20)
project(NumHalide VERSION 1.0.0 LANGUAGES CXX)

# --- CMake module path ---
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/buildsystem/cmake")
include(NumHalideOptions)

# --- Options ---
option(NUMHALIDE_BUILD_TESTS    "Build unit tests"  ON)
option(NUMHALIDE_BUILD_EXAMPLES "Build examples"    ON)

# --- Find Halide ---
find_package(Halide REQUIRED)

# --- Header-only library target ---
add_library(numhalide INTERFACE)
target_include_directories(numhalide INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include/numhalide>
)
target_link_libraries(numhalide INTERFACE Halide::Halide)
target_compile_features(numhalide INTERFACE cxx_std_20)
target_compile_definitions(numhalide INTERFACE NOMINMAX)
add_library(NumHalide::numhalide ALIAS numhalide)

# --- Tests ---
if(NUMHALIDE_BUILD_TESTS)
    enable_testing()
    find_package(GTest CONFIG REQUIRED)

    file(GLOB TEST_SOURCES "${CMAKE_SOURCE_DIR}/tests/test_*.cpp")

    add_executable(numhalide_tests ${TEST_SOURCES})
    target_link_libraries(numhalide_tests PRIVATE
        numhalide
        GTest::gtest
        GTest::gtest_main
    )
    numhalide_set_target_options(numhalide_tests)
    numhalide_copy_halide_dll(numhalide_tests)

    # Register with CTest
    include(GoogleTest)
    gtest_discover_tests(numhalide_tests)
endif()

# --- Examples ---
if(NUMHALIDE_BUILD_EXAMPLES)
    # stb image write implementation (compiled once, shared by all examples)
    add_library(stbi_impl STATIC
        "${CMAKE_SOURCE_DIR}/examples/_common/stbi_impl.cpp"
    )
    target_include_directories(stbi_impl PUBLIC
        "${CMAKE_SOURCE_DIR}/extern/stb"
        "${CMAKE_SOURCE_DIR}/examples/_common"
    )
    numhalide_set_target_options(stbi_impl)

    # Collect all example directories (00_gradient, 01_shape_debug, ...)
    file(GLOB EXAMPLE_DIRS RELATIVE "${CMAKE_SOURCE_DIR}/examples"
        "${CMAKE_SOURCE_DIR}/examples/[0-9][0-9]_*"
    )
    list(SORT EXAMPLE_DIRS)

    foreach(EXAMPLE_DIR ${EXAMPLE_DIRS})
        # Extract the .cpp file (same name as directory)
        file(GLOB EXAMPLE_SRC "${CMAKE_SOURCE_DIR}/examples/${EXAMPLE_DIR}/*.cpp")
        if(NOT EXAMPLE_SRC)
            continue()
        endif()

        # Target name: example_00_gradient, example_01_shape_debug, ...
        string(REPLACE "/" "_" EXAMPLE_TARGET "example_${EXAMPLE_DIR}")

        add_executable(${EXAMPLE_TARGET} ${EXAMPLE_SRC})
        target_link_libraries(${EXAMPLE_TARGET} PRIVATE
            numhalide
            stbi_impl
        )
        target_include_directories(${EXAMPLE_TARGET} PRIVATE
            "${CMAKE_SOURCE_DIR}/extern/stb"
            "${CMAKE_SOURCE_DIR}/examples/_common"
        )
        numhalide_set_target_options(${EXAMPLE_TARGET})
        numhalide_copy_halide_dll(${EXAMPLE_TARGET})

        # Set working directory for Visual Studio debugging
        set_target_properties(${EXAMPLE_TARGET} PROPERTIES
            VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/working_dir"
            FOLDER "Examples"
        )
    endforeach()

    # Organize in IDE folders
    set_target_properties(stbi_impl PROPERTIES FOLDER "Examples")
endif()

if(NUMHALIDE_BUILD_TESTS)
    set_target_properties(numhalide_tests PROPERTIES FOLDER "Tests")
endif()

# Enable IDE folder grouping
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# --- Install ---
include(GNUInstallDirs)

install(DIRECTORY src/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/numhalide
    FILES_MATCHING PATTERN "*.h"
)

install(TARGETS numhalide
    EXPORT NumHalideTargets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT NumHalideTargets
    FILE NumHalideTargets.cmake
    NAMESPACE NumHalide::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NumHalide
)
