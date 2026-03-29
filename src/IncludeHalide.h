/// @file IncludeHalide.h
/// @brief Wrapper that includes Halide.h with all compiler warnings suppressed.
///
/// Include this file instead of <Halide.h> anywhere in NumHalide.
/// The suppression ensures Halide's internal headers do not pollute
/// the project's warning output.

#pragma once

#ifdef _MSC_VER
#  pragma warning(push, 0)
#endif

#include <Halide.h>

#ifdef _MSC_VER
#  pragma warning(pop)
#endif
