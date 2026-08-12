/// Note: exp, log, sqrt, abs, negative are already defined in broadcast.hnn// @file math_ext.h
/// @brief Extended math functions
///
/// Provides: exp2, log2, log10, expm1, log1p, square, cbrt,
///           reciprocal, sinc, heaviside, fmod, remainder, nan_to_num

#pragma once

#include "common.h"
#include "shape.h"

NS_NUM_HALIDE_BEGIN

// -----------------------------------------------------------------------------
// Exponential / Logarithmic Variants
// -----------------------------------------------------------------------------

/// @brief Compute 2^x element-wise
inline
Halide::Func exp2(Halide::Func f, const shape_t& shape, std::string const& name = "exp2")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	ret(vars) = Halide::pow(2.0f, f(vars));
	return ret;
}

/// @brief Compute base-2 logarithm: log(x) / log(2)
inline
Halide::Func log2(Halide::Func f, const shape_t& shape, std::string const& name = "log2")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	// ln(2) of a constant made in the input's own float type — for f32
	// this folds to the same value as the old log(2.0f); for f64 it is
	// full-precision ln(2) instead of a widened f32 constant.
	Halide::Type t = f.types()[0];
	if (!t.is_float()) t = Halide::Float(32);
	ret(vars) = Halide::log(f(vars)) / Halide::log(Halide::Internal::make_const(t, 2.0));
	return ret;
}

/// @brief Compute base-10 logarithm: log(x) / log(10)
inline
Halide::Func log10(Halide::Func f, const shape_t& shape, std::string const& name = "log10")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	// ln(10) of a constant made in the input's own float type — same
	// rationale as log2 above.
	Halide::Type t = f.types()[0];
	if (!t.is_float()) t = Halide::Float(32);
	ret(vars) = Halide::log(f(vars)) / Halide::log(Halide::Internal::make_const(t, 10.0));
	return ret;
}

/// @brief Compute exp(x) - 1, accurate for small x
inline
Halide::Func expm1(Halide::Func f, const shape_t& shape, std::string const& name = "expm1")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	ret(vars) = Halide::exp(f(vars)) - 1.0f;
	return ret;
}

/// @brief Compute log(1 + x), accurate for small x
inline
Halide::Func log1p(Halide::Func f, const shape_t& shape, std::string const& name = "log1p")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	ret(vars) = Halide::log(f(vars) + 1.0f);
	return ret;
}

// -----------------------------------------------------------------------------
// Power / Root Variants
// -----------------------------------------------------------------------------

/// @brief Compute x^2 element-wise
inline
Halide::Func square(Halide::Func f, const shape_t& shape, std::string const& name = "square")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	Halide::Expr val = f(vars);
	ret(vars) = val * val;
	return ret;
}

/// @brief Compute cube root: pow(x, 1/3)
inline
Halide::Func cbrt(Halide::Func f, const shape_t& shape, std::string const& name = "cbrt")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	Halide::Expr val = f(vars);
	Halide::Type t = f.types()[0];
	Halide::Expr third = Halide::Internal::make_const(t, 1.0 / 3.0);
	// Sign-aware: cbrt(-x) = -cbrt(x)
	ret(vars) = Halide::select(
		val >= 0,
		Halide::pow(val, third),
		-Halide::pow(-val, third)
	);
	return ret;
}

/// @brief Compute 1/x element-wise
inline
Halide::Func reciprocal(Halide::Func f, const shape_t& shape, std::string const& name = "reciprocal")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	ret(vars) = 1.0f / f(vars);
	return ret;
}

// -----------------------------------------------------------------------------
// Special Functions
// -----------------------------------------------------------------------------

/// @brief Compute normalized sinc: sin(pi*x) / (pi*x), with sinc(0) = 1
inline
Halide::Func sinc(Halide::Func f, const shape_t& shape, std::string const& name = "sinc")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	// pi is made in the input's own float type (f64 keeps full-precision
	// pi; the f32 constant is unchanged). The x == 0 arm is a multiplied
	// 0/1 indicator with a guarded denominator, NOT a select around the
	// division: a select arm containing sin(pix)/pix evaluates 0/0 = NaN
	// at x = 0 under CMOV and breaks reverse-mode AD (f32-zero arm vs f64
	// division derivative). Value-identical: x != 0 -> sin(pix)/(pix + 0);
	// x == 0 -> 0 * (0/1) + 1 = 1.
	Halide::Type t = f.types()[0];
	if (!t.is_float()) t = Halide::Float(32);
	Halide::Expr one = Halide::Internal::make_one(t);
	Halide::Expr pi_t = Halide::Internal::make_const(t, 3.14159265358979323846);
	Halide::Expr val = f(vars);
	Halide::Expr pix = pi_t * val;
	Halide::Expr is0 = Halide::cast(t, val == Halide::Internal::make_zero(t));
	ret(vars) = (one - is0) * (Halide::sin(pix) / (pix + is0)) + is0;
	return ret;
}

/// @brief Compute Heaviside step function
/// @param f Input Func
/// @param h0 Value at x == 0 (typically 0.5)
/// @param shape Shape of input
/// @param name Function name
inline
Halide::Func heaviside(Halide::Func f, Halide::Expr h0, const shape_t& shape, std::string const& name = "heaviside")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	Halide::Expr val = f(vars);
	ret(vars) = Halide::select(val < 0.0f, 0.0f, val > 0.0f, 1.0f, h0);
	return ret;
}

// -----------------------------------------------------------------------------
// Modular Arithmetic
// -----------------------------------------------------------------------------

/// @brief Compute floating-point modulus: a - floor(a/b) * b
inline
Halide::Func fmod(Halide::Func a, Halide::Func b, const shape_t& shape, std::string const& name = "fmod")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	Halide::Expr av = a(vars);
	Halide::Expr bv = b(vars);
	ret(vars) = av - Halide::floor(av / bv) * bv;
	return ret;
}

/// @brief Compute IEEE remainder: a - round(a/b) * b
inline
Halide::Func remainder(Halide::Func a, Halide::Func b, const shape_t& shape, std::string const& name = "remainder")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	Halide::Expr av = a(vars);
	Halide::Expr bv = b(vars);
	ret(vars) = av - Halide::round(av / bv) * bv;
	return ret;
}

// -----------------------------------------------------------------------------
// NaN / Inf Handling
// -----------------------------------------------------------------------------

/// @brief Replace NaN and Inf values
/// @param f Input Func
/// @param shape Shape of input
/// @param nan_val Replacement for NaN (default 0)
/// @param posinf_val Replacement for +Inf (default a large float)
/// @param neginf_val Replacement for -Inf (default a large negative float)
/// @param name Function name
inline
Halide::Func nan_to_num(Halide::Func f, const shape_t& shape,
	Halide::Expr nan_val = 0.0f,
	Halide::Expr posinf_val = 3.402823466e+38f,
	Halide::Expr neginf_val = -3.402823466e+38f,
	std::string const& name = "nan_to_num")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	Halide::Expr val = f(vars);
	ret(vars) = Halide::select(
		Halide::is_nan(val), nan_val,
		Halide::is_inf(val) && val > 0, posinf_val,
		Halide::is_inf(val) && val < 0, neginf_val,
		val
	);
	return ret;
}

// Note: exp, log, sqrt, abs, negative are already defined in broadcast.h

// -----------------------------------------------------------------------------
// Rounding Functions
// -----------------------------------------------------------------------------

/// @brief Round down to nearest integer element-wise
inline
Halide::Func floor(Halide::Func f, const shape_t& shape, std::string const& name = "floor_f")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	ret(vars) = Halide::floor(f(vars));
	return ret;
}

/// @brief Round up to nearest integer element-wise
inline
Halide::Func ceil(Halide::Func f, const shape_t& shape, std::string const& name = "ceil_f")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	ret(vars) = Halide::ceil(f(vars));
	return ret;
}

/// @brief Round to nearest integer element-wise
inline
Halide::Func round(Halide::Func f, const shape_t& shape, std::string const& name = "round_f")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	ret(vars) = Halide::round(f(vars));
	return ret;
}

/// @brief Round to nearest integer element-wise (alias for round)
inline
Halide::Func rint(Halide::Func f, const shape_t& shape, std::string const& name = "rint_f")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	ret(vars) = Halide::round(f(vars));
	return ret;
}

/// @brief Truncate toward zero element-wise
inline
Halide::Func trunc(Halide::Func f, const shape_t& shape, std::string const& name = "trunc_f")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	ret(vars) = Halide::trunc(f(vars));
	return ret;
}

/// @brief Truncate toward zero element-wise (NumPy's fix, alias for trunc)
inline
Halide::Func fix(Halide::Func f, const shape_t& shape, std::string const& name = "fix_f")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	ret(vars) = Halide::trunc(f(vars));
	return ret;
}

// -----------------------------------------------------------------------------
// Power Functions
// -----------------------------------------------------------------------------

/// @brief Compute base^exponent element-wise (Func exponent)
inline
Halide::Func power(Halide::Func base, Halide::Func exponent, const shape_t& shape, std::string const& name = "power_f")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	ret(vars) = Halide::pow(base(vars), exponent(vars));
	return ret;
}

/// @brief Compute base^exponent element-wise (scalar Expr exponent)
inline
Halide::Func power(Halide::Func base, Halide::Expr exponent, const shape_t& shape, std::string const& name = "power_f")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	ret(vars) = Halide::pow(base(vars), exponent);
	return ret;
}

NS_NUM_HALIDE_END
