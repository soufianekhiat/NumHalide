/// @file array_compare.h
/// @brief Array comparison utilities for Halide::Func objects
///
/// Provides: array_equal, array_equiv

#pragma once

#include "common.h"
#include "shape.h"
#include "reduce.h"

NS_NUM_HALIDE_BEGIN

// -----------------------------------------------------------------------------
// Array Comparison
// -----------------------------------------------------------------------------

/// @brief Check if two arrays are element-wise equal
/// @param a First input Func
/// @param b Second input Func
/// @param shape Shape of both arrays (must be identical)
/// @param name Function name
/// @return 1D scalar Func (1 if all elements equal, 0 otherwise)
inline Halide::Func array_equal(Halide::Func a, Halide::Func b, const shape_t& shape, std::string const& name = "array_equal") {
	// Create element-wise equality
	Halide::Func eq("eq");
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) vars.push_back(Halide::Var());
	eq(vars) = Halide::cast<int32_t>(a(vars) == b(vars));
	// reduce_all returns 1 if ALL nonzero
	return reduce_all(eq, shape, name);
}

/// @brief Check if two arrays are element-wise equivalent
/// @param a First input Func
/// @param b Second input Func
/// @param shape Shape of both arrays (must be identical)
/// @param name Function name
/// @return 1D scalar Func (1 if all elements equal, 0 otherwise)
///
/// Note: Broadcasting is not implemented; this is an alias for array_equal.
inline Halide::Func array_equiv(Halide::Func a, Halide::Func b, const shape_t& shape, std::string const& name = "array_equiv") {
	return array_equal(a, b, shape, name);
}

NS_NUM_HALIDE_END
