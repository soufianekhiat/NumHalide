/// @file ops.h
/// @brief Element-wise operations: where, clip, cast, reshape
///
/// Provides: where, clip, astype, reshape_func, expand_dims, squeeze

#pragma once

#include "common.h"
#include "shape.h"

NS_NUM_HALIDE_BEGIN

// -----------------------------------------------------------------------------
// Conditional Selection
// -----------------------------------------------------------------------------

/// @brief Select elements from x or y based on condition
/// @param cond Boolean condition Func
/// @param x Values where condition is true
/// @param y Values where condition is false
/// @param shape Shape of the arrays
/// @param name Function name
/// @return Func with selected values
///
/// Usage:
///   Func result = where(mask, a, b, {3, 4});
inline
Halide::Func where(Halide::Func cond, Halide::Func x, Halide::Func y, const shape_t& shape, std::string const& name = "where")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	ret(vars) = Halide::select(cond(vars) != 0, x(vars), y(vars));
	return ret;
}

/// @brief Select elements from x or y based on scalar expression condition
/// @param cond Boolean expression (e.g., a(vars) > 0)
/// @param x Func for true values
/// @param y Func for false values
/// @param shape Shape of the arrays
/// @param name Function name
/// @return Func with selected values
inline
Halide::Func where_expr(Halide::Func x, const shape_t& shape, Halide::Expr threshold, Halide::Func y, std::string const& name = "where_expr")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	ret(vars) = Halide::select(x(vars) > threshold, x(vars), y(vars));
	return ret;
}

// -----------------------------------------------------------------------------
// Clipping
// -----------------------------------------------------------------------------

/// @brief Clip values to a range [lo, hi]
/// @param f Input Func
/// @param shape Shape of input
/// @param lo Lower bound
/// @param hi Upper bound
/// @param name Function name
/// @return Clipped Func
///
/// Usage:
///   Func clipped = clip(a, {3, 4}, 0.0f, 1.0f);
inline
Halide::Func clip(Halide::Func f, const shape_t& shape, Halide::Expr lo, Halide::Expr hi, std::string const& name = "clip")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	ret(vars) = Halide::clamp(f(vars), lo, hi);
	return ret;
}

/// @brief Clip values to minimum value
/// @param f Input Func
/// @param shape Shape of input
/// @param lo Lower bound
/// @param name Function name
/// @return Clipped Func
inline
Halide::Func clip_min(Halide::Func f, const shape_t& shape, Halide::Expr lo, std::string const& name = "clip_min")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	ret(vars) = Halide::max(f(vars), lo);
	return ret;
}

/// @brief Clip values to maximum value
/// @param f Input Func
/// @param shape Shape of input
/// @param hi Upper bound
/// @param name Function name
/// @return Clipped Func
inline
Halide::Func clip_max(Halide::Func f, const shape_t& shape, Halide::Expr hi, std::string const& name = "clip_max")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	ret(vars) = Halide::min(f(vars), hi);
	return ret;
}

// -----------------------------------------------------------------------------
// Type Casting
// -----------------------------------------------------------------------------

/// @brief Cast a Func to a different data type
/// @param f Input Func
/// @param shape Shape of input
/// @param dtype Target data type
/// @param name Function name
/// @return Func with new data type
///
/// Usage:
///   Func int_vals = astype(float_vals, {3, 4}, Int(32));
inline
Halide::Func astype(Halide::Func f, const shape_t& shape, Halide::Type dtype, std::string const& name = "astype")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	ret(vars) = Halide::cast(dtype, f(vars));
	return ret;
}

/// @brief Round and cast to integer type
/// @param f Input Func (float type)
/// @param shape Shape of input
/// @param dtype Target integer type
/// @param name Function name
/// @return Func with rounded integer values
inline
Halide::Func round_astype(Halide::Func f, const shape_t& shape, Halide::Type dtype, std::string const& name = "round_astype")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	ret(vars) = Halide::cast(dtype, Halide::round(f(vars)));
	return ret;
}

// -----------------------------------------------------------------------------
// Reshaping
// -----------------------------------------------------------------------------

/// @brief Reshape a Func to a new shape (must have same total elements)
/// @param f Input Func
/// @param in_shape Current shape
/// @param out_shape Target shape
/// @param name Function name
/// @return Reshaped Func
///
/// Usage:
///   // Reshape 2x6 to 3x4
///   Func reshaped = reshape_func(a, {2, 6}, {3, 4});
inline
Halide::Func reshape_func(Halide::Func f, const shape_t& in_shape, const shape_t& out_shape, std::string const& name = "reshape")
{
	// Validate total elements match
	int in_total = 1;
	for (int i = 0; i < in_shape.rank; ++i) {
		in_total *= in_shape.extents[i];
	}
	int out_total = 1;
	for (int i = 0; i < out_shape.rank; ++i) {
		out_total *= out_shape.extents[i];
	}
	nh_require(nullptr, in_total == out_total,
		"reshape: total elements must match (%d vs %d)", in_total, out_total);

	// Build sizes vectors for the existing reshape function
	std::vector<Halide::Expr> in_sizes;
	for (int i = 0; i < in_shape.rank; ++i) {
		in_sizes.push_back(in_shape.extents[i]);
	}
	std::vector<Halide::Expr> out_sizes;
	for (int i = 0; i < out_shape.rank; ++i) {
		out_sizes.push_back(out_shape.extents[i]);
	}

	// Use the existing reshape implementation
	return reshape(f, in_sizes, out_sizes, name);
}

// -----------------------------------------------------------------------------
// Mathematical Operations
// -----------------------------------------------------------------------------

/// @brief Compute absolute value
/// @param f Input Func
/// @param shape Shape of input
/// @param name Function name
/// @return Func with absolute values
inline
Halide::Func nh_abs(Halide::Func f, const shape_t& shape, std::string const& name = "abs")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	ret(vars) = Halide::abs(f(vars));
	return ret;
}

/// @brief Compute sign (-1, 0, or 1)
/// @param f Input Func
/// @param shape Shape of input
/// @param name Function name
/// @return Func with sign values
inline
Halide::Func sign(Halide::Func f, const shape_t& shape, std::string const& name = "sign")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	Halide::Expr val = f(vars);
	Halide::Type type = f.types()[0];
	ret(vars) = Halide::select(
		val > 0, Halide::cast(type, 1),
		val < 0, Halide::cast(type, -1),
		Halide::cast(type, 0)
	);
	return ret;
}

/// @brief Compute floor
/// @param f Input Func
/// @param shape Shape of input
/// @param name Function name
/// @return Func with floor values
inline
Halide::Func nh_floor(Halide::Func f, const shape_t& shape, std::string const& name = "nh_floor")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	ret(vars) = Halide::floor(f(vars));
	return ret;
}

/// @brief Compute ceiling
/// @param f Input Func
/// @param shape Shape of input
/// @param name Function name
/// @return Func with ceiling values
inline
Halide::Func nh_ceil(Halide::Func f, const shape_t& shape, std::string const& name = "nh_ceil")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	ret(vars) = Halide::ceil(f(vars));
	return ret;
}

/// @brief Round to nearest integer
/// @param f Input Func
/// @param shape Shape of input
/// @param name Function name
/// @return Func with rounded values
inline
Halide::Func nh_round(Halide::Func f, const shape_t& shape, std::string const& name = "round")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	ret(vars) = Halide::round(f(vars));
	return ret;
}

/// @brief Compute square root
/// @param f Input Func
/// @param shape Shape of input
/// @param name Function name
/// @return Func with sqrt values
inline
Halide::Func nh_sqrt(Halide::Func f, const shape_t& shape, std::string const& name = "sqrt")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	ret(vars) = Halide::sqrt(f(vars));
	return ret;
}

/// @brief Compute exponential
/// @param f Input Func
/// @param shape Shape of input
/// @param name Function name
/// @return Func with exp values
inline
Halide::Func nh_exp(Halide::Func f, const shape_t& shape, std::string const& name = "exp")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	ret(vars) = Halide::exp(f(vars));
	return ret;
}

/// @brief Compute natural logarithm
/// @param f Input Func
/// @param shape Shape of input
/// @param name Function name
/// @return Func with log values
inline
Halide::Func nh_log(Halide::Func f, const shape_t& shape, std::string const& name = "log")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	ret(vars) = Halide::log(f(vars));
	return ret;
}

/// @brief Compute power
/// @param f Base Func
/// @param shape Shape of input
/// @param exponent Power to raise to
/// @param name Function name
/// @return Func with power values
inline
Halide::Func nh_pow(Halide::Func f, const shape_t& shape, Halide::Expr exponent, std::string const& name = "pow")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	ret(vars) = Halide::pow(f(vars), exponent);
	return ret;
}

/// @brief Compute element-wise power
/// @param base Base Func
/// @param exp Exponent Func
/// @param shape Shape of inputs
/// @param name Function name
/// @return Func with power values
inline
Halide::Func nh_pow(Halide::Func base, Halide::Func exp, const shape_t& shape, std::string const& name = "pow")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	ret(vars) = Halide::pow(base(vars), exp(vars));
	return ret;
}

// -----------------------------------------------------------------------------
// Comparison Operations
// -----------------------------------------------------------------------------

/// @brief Element-wise equality comparison
/// @return Func returning 1 where equal, 0 otherwise
inline
Halide::Func equal(Halide::Func a, Halide::Func b, const shape_t& shape, std::string const& name = "equal")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}
	ret(vars) = Halide::cast<int32_t>(a(vars) == b(vars));
	return ret;
}

/// @brief Element-wise equality comparison with scalar
inline
Halide::Func equal(Halide::Func a, Halide::Expr scalar, const shape_t& shape, std::string const& name = "equal")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}
	ret(vars) = Halide::cast<int32_t>(a(vars) == scalar);
	return ret;
}

/// @brief Element-wise inequality comparison
inline
Halide::Func not_equal(Halide::Func a, Halide::Func b, const shape_t& shape, std::string const& name = "not_equal")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}
	ret(vars) = Halide::cast<int32_t>(a(vars) != b(vars));
	return ret;
}

/// @brief Element-wise inequality comparison with scalar
inline
Halide::Func not_equal(Halide::Func a, Halide::Expr scalar, const shape_t& shape, std::string const& name = "not_equal")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}
	ret(vars) = Halide::cast<int32_t>(a(vars) != scalar);
	return ret;
}

/// @brief Element-wise greater than comparison
inline
Halide::Func greater(Halide::Func a, Halide::Func b, const shape_t& shape, std::string const& name = "greater")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}
	ret(vars) = Halide::cast<int32_t>(a(vars) > b(vars));
	return ret;
}

/// @brief Element-wise greater than comparison with scalar
inline
Halide::Func greater(Halide::Func a, Halide::Expr scalar, const shape_t& shape, std::string const& name = "greater")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}
	ret(vars) = Halide::cast<int32_t>(a(vars) > scalar);
	return ret;
}

/// @brief Element-wise less than comparison
inline
Halide::Func less(Halide::Func a, Halide::Func b, const shape_t& shape, std::string const& name = "less")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}
	ret(vars) = Halide::cast<int32_t>(a(vars) < b(vars));
	return ret;
}

/// @brief Element-wise less than comparison with scalar
inline
Halide::Func less(Halide::Func a, Halide::Expr scalar, const shape_t& shape, std::string const& name = "less")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}
	ret(vars) = Halide::cast<int32_t>(a(vars) < scalar);
	return ret;
}

/// @brief Element-wise greater than or equal comparison
inline
Halide::Func greater_equal(Halide::Func a, Halide::Func b, const shape_t& shape, std::string const& name = "greater_equal")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}
	ret(vars) = Halide::cast<int32_t>(a(vars) >= b(vars));
	return ret;
}

/// @brief Element-wise greater than or equal comparison with scalar
inline
Halide::Func greater_equal(Halide::Func a, Halide::Expr scalar, const shape_t& shape, std::string const& name = "greater_equal")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}
	ret(vars) = Halide::cast<int32_t>(a(vars) >= scalar);
	return ret;
}

/// @brief Element-wise less than or equal comparison
inline
Halide::Func less_equal(Halide::Func a, Halide::Func b, const shape_t& shape, std::string const& name = "less_equal")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}
	ret(vars) = Halide::cast<int32_t>(a(vars) <= b(vars));
	return ret;
}

/// @brief Element-wise less than or equal comparison with scalar
inline
Halide::Func less_equal(Halide::Func a, Halide::Expr scalar, const shape_t& shape, std::string const& name = "less_equal")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}
	ret(vars) = Halide::cast<int32_t>(a(vars) <= scalar);
	return ret;
}

// -----------------------------------------------------------------------------
// Logical Operations
// -----------------------------------------------------------------------------

/// @brief Element-wise logical AND
inline
Halide::Func logical_and(Halide::Func a, Halide::Func b, const shape_t& shape, std::string const& name = "logical_and")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}
	ret(vars) = Halide::cast<int32_t>((a(vars) != 0) && (b(vars) != 0));
	return ret;
}

/// @brief Element-wise logical OR
inline
Halide::Func logical_or(Halide::Func a, Halide::Func b, const shape_t& shape, std::string const& name = "logical_or")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}
	ret(vars) = Halide::cast<int32_t>((a(vars) != 0) || (b(vars) != 0));
	return ret;
}

/// @brief Element-wise logical NOT
inline
Halide::Func nh_logical_not(Halide::Func a, const shape_t& shape, std::string const& name = "nh_logical_not")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}
	ret(vars) = Halide::cast<int32_t>(a(vars) == 0);
	return ret;
}

/// @brief Element-wise logical XOR
inline
Halide::Func logical_xor(Halide::Func a, Halide::Func b, const shape_t& shape, std::string const& name = "logical_xor")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}
	// XOR: exactly one is true
	Halide::Expr a_bool = a(vars) != 0;
	Halide::Expr b_bool = b(vars) != 0;
	ret(vars) = Halide::cast<int32_t>((a_bool && !b_bool) || (!a_bool && b_bool));
	return ret;
}

// -----------------------------------------------------------------------------
// Special Value Detection
// -----------------------------------------------------------------------------

/// @brief Check for NaN values (float/double only)
inline
Halide::Func isnan_func(Halide::Func a, const shape_t& shape, std::string const& name = "isnan")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}
	ret(vars) = Halide::cast<int32_t>(Halide::is_nan(a(vars)));
	return ret;
}

/// @brief Check for infinity values (float/double only)
inline
Halide::Func isinf_func(Halide::Func a, const shape_t& shape, std::string const& name = "isinf")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}
	ret(vars) = Halide::cast<int32_t>(Halide::is_inf(a(vars)));
	return ret;
}

/// @brief Check for finite values (not NaN and not infinity)
inline
Halide::Func isfinite_func(Halide::Func a, const shape_t& shape, std::string const& name = "isfinite")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}
	ret(vars) = Halide::cast<int32_t>(Halide::is_finite(a(vars)));
	return ret;
}

NS_NUM_HALIDE_END
