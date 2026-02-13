# FindHalide.cmake
# Finds the pre-built Halide library in extern/Halide
#
# Sets:
#   Halide_FOUND        - TRUE if Halide is found
#   Halide_INCLUDE_DIRS - Halide include directories
#   Halide_LIBRARIES    - Halide library path
#   Halide_DLL          - Halide DLL path (for runtime copy)
#
# Creates imported target: Halide::Halide

if(Halide_FOUND)
    return()
endif()

set(_HALIDE_ROOT "${CMAKE_SOURCE_DIR}/extern/Halide")

find_path(Halide_INCLUDE_DIR
    NAMES Halide.h
    PATHS "${_HALIDE_ROOT}/include"
    NO_DEFAULT_PATH
)

find_library(Halide_LIBRARY
    NAMES Halide
    PATHS "${_HALIDE_ROOT}/lib"
    NO_DEFAULT_PATH
)

find_file(Halide_DLL
    NAMES Halide.dll
    PATHS "${_HALIDE_ROOT}/bin/RelWithDebInfo"
    NO_DEFAULT_PATH
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Halide
    REQUIRED_VARS Halide_LIBRARY Halide_INCLUDE_DIR
)

if(Halide_FOUND AND NOT TARGET Halide::Halide)
    add_library(Halide::Halide SHARED IMPORTED)
    set_target_properties(Halide::Halide PROPERTIES
        IMPORTED_IMPLIB "${Halide_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${Halide_INCLUDE_DIR}"
    )
    if(Halide_DLL)
        set_target_properties(Halide::Halide PROPERTIES
            IMPORTED_LOCATION "${Halide_DLL}"
        )
    endif()
endif()

set(Halide_INCLUDE_DIRS "${Halide_INCLUDE_DIR}")
set(Halide_LIBRARIES "${Halide_LIBRARY}")
