/// @file compare_ext.h
/// @brief Extended comparison and tolerance functions
///
/// Provides: isclose, allclose, isneginf, isposinf

#pragma once

#include "common.h"
#include "shape.h"
#include "reduce.h"

NS_NUM_HALIDE_BEGIN

// -----------------------------------------------------------------------------
// Element-wise Approximate Equality
// -----------------------------------------------------------------------------

/// @brief Check if two arrays are element-wise equal within a tolerance
/// @param a First input Func
/// @param b Second input Func
/// @param shape Shape of inputs (must be identical)
/// @param rtol Relative tolerance (default 1e-5)
/// @param atol Absolute tolerance (default 1e-8)
/// @param name Function name
/// @return Func returning 1 where |a-b| <= atol + rtol*|b|, 0 otherwise
///
/// Follows NumPy semantics: |a - b| <= atol + rtol * |b|
///
/// Usage:
///   Func close = isclose(a, b, {3, 4});
///   Func close = isclose(a, b, {3, 4}, 1e-3f, 1e-6f);
inline
Halide::Func isclose(Halide::Func a, Halide::Func b, const shape_t& shape,
                     Halide::Expr rtol = 1e-5f, Halide::Expr atol = 1e-8f,
                     std::string const& name = "isclose")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	Halide::Expr diff = Halide::abs(a(vars) - b(vars));
	Halide::Expr tol = atol + rtol * Halide::abs(b(vars));
	ret(vars) = Halide::cast<int32_t>(diff <= tol);
	return ret;
}

/// @brief Check if two scalar Exprs are close (element-wise against scalar)
/// @param a Input Func
/// @param scalar Scalar value to compare against
/// @param shape Shape of input
/// @param rtol Relative tolerance
/// @param atol Absolute tolerance
/// @param name Function name
/// @return Func returning 1 where close, 0 otherwise
inline
Halide::Func isclose(Halide::Func a, Halide::Expr scalar, const shape_t& shape,
                     Halide::Expr rtol = 1e-5f, Halide::Expr atol = 1e-8f,
                     std::string const& name = "isclose")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	Halide::Expr diff = Halide::abs(a(vars) - scalar);
	Halide::Expr tol = atol + rtol * Halide::abs(scalar);
	ret(vars) = Halide::cast<int32_t>(diff <= tol);
	return ret;
}

// -----------------------------------------------------------------------------
// All-Close Reduction
// -----------------------------------------------------------------------------

/// @brief Check if all elements of two arrays are close within a tolerance
/// @param a First input Func
/// @param b Second input Func
/// @param shape Shape of inputs
/// @param rtol Relative tolerance (default 1e-5)
/// @param atol Absolute tolerance (default 1e-8)
/// @param name Function name
/// @return 1D Func with single element: 1 if all close, 0 otherwise
///
/// Equivalent to reduce_all(isclose(a, b, shape, rtol, atol)).
///
/// Usage:
///   Func ac = allclose(a, b, {3, 4});
inline
Halide::Func allclose(Halide::Func a, Halide::Func b, const shape_t& shape,
                      Halide::Expr rtol = 1e-5f, Halide::Expr atol = 1e-8f,
                      std::string const& name = "allclose")
{
	Halide::Func close = isclose(a, b, shape, rtol, atol, name + "_isclose");
	return reduce_all(close, shape, name);
}

// -----------------------------------------------------------------------------
// Signed Infinity Detection
// -----------------------------------------------------------------------------

/// @brief Check for negative infinity values (float/double only)
/// @param f Input Func
/// @param shape Shape of input
/// @param name Function name
/// @return Func returning 1 where value is -inf, 0 otherwise
///
/// Usage:
///   Func mask = isneginf(f, {3, 4});
inline
Halide::Func isneginf(Halide::Func f, const shape_t& shape, std::string const& name = "isneginf")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	Halide::Expr val = f(vars);
	ret(vars) = Halide::cast<int32_t>(Halide::is_inf(val) && (val < 0));
	return ret;
}

/// @brief Check for positive infinity values (float/double only)
/// @param f Input Func
/// @param shape Shape of input
/// @param name Function name
/// @return Func returning 1 where value is +inf, 0 otherwise
///
/// Usage:
///   Func mask = isposinf(f, {3, 4});
inline
Halide::Func isposinf(Halide::Func f, const shape_t& shape, std::string const& name = "isposinf")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	Halide::Expr val = f(vars);
	ret(vars) = Halide::cast<int32_t>(Halide::is_inf(val) && (val > 0));
	return ret;
}

NS_NUM_HALIDE_END
