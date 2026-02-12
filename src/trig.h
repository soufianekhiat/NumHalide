/// @file trig.h
/// @brief Trigonometric functions
///
/// Provides: sin, cos, tan, asin, acos, atan, sinh, cosh, tanh,
///           asinh, acosh, atanh, degrees, radians, atan2, hypot

#pragma once

#include "common.h"
#include "shape.h"

NS_NUM_HALIDE_BEGIN

// -----------------------------------------------------------------------------
// Basic Trigonometric Functions
// -----------------------------------------------------------------------------

/// @brief Compute sine
inline
Halide::Func nh_sin(Halide::Func f, const shape_t& shape, std::string const& name = "sin")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	ret(vars) = Halide::sin(f(vars));
	return ret;
}

/// @brief Compute cosine
inline
Halide::Func nh_cos(Halide::Func f, const shape_t& shape, std::string const& name = "cos")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	ret(vars) = Halide::cos(f(vars));
	return ret;
}

/// @brief Compute tangent
inline
Halide::Func nh_tan(Halide::Func f, const shape_t& shape, std::string const& name = "tan")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	ret(vars) = Halide::tan(f(vars));
	return ret;
}

// -----------------------------------------------------------------------------
// Inverse Trigonometric Functions
// -----------------------------------------------------------------------------

/// @brief Compute arc sine
inline
Halide::Func nh_asin(Halide::Func f, const shape_t& shape, std::string const& name = "asin")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	ret(vars) = Halide::asin(f(vars));
	return ret;
}

/// @brief Compute arc cosine
inline
Halide::Func nh_acos(Halide::Func f, const shape_t& shape, std::string const& name = "acos")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	ret(vars) = Halide::acos(f(vars));
	return ret;
}

/// @brief Compute arc tangent
inline
Halide::Func nh_atan(Halide::Func f, const shape_t& shape, std::string const& name = "atan")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	ret(vars) = Halide::atan(f(vars));
	return ret;
}

// -----------------------------------------------------------------------------
// Hyperbolic Functions
// -----------------------------------------------------------------------------

/// @brief Compute hyperbolic sine
inline
Halide::Func nh_sinh(Halide::Func f, const shape_t& shape, std::string const& name = "sinh")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	ret(vars) = Halide::sinh(f(vars));
	return ret;
}

/// @brief Compute hyperbolic cosine
inline
Halide::Func nh_cosh(Halide::Func f, const shape_t& shape, std::string const& name = "cosh")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	ret(vars) = Halide::cosh(f(vars));
	return ret;
}

/// @brief Compute hyperbolic tangent
inline
Halide::Func nh_tanh(Halide::Func f, const shape_t& shape, std::string const& name = "tanh")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	ret(vars) = Halide::tanh(f(vars));
	return ret;
}

// -----------------------------------------------------------------------------
// Inverse Hyperbolic Functions
// -----------------------------------------------------------------------------

/// @brief Compute inverse hyperbolic sine: log(x + sqrt(x^2 + 1))
inline
Halide::Func nh_asinh(Halide::Func f, const shape_t& shape, std::string const& name = "asinh")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	Halide::Expr val = f(vars);
	ret(vars) = Halide::log(val + Halide::sqrt(val * val + 1.0f));
	return ret;
}

/// @brief Compute inverse hyperbolic cosine: log(x + sqrt(x^2 - 1))
inline
Halide::Func nh_acosh(Halide::Func f, const shape_t& shape, std::string const& name = "acosh")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	Halide::Expr val = f(vars);
	ret(vars) = Halide::log(val + Halide::sqrt(val * val - 1.0f));
	return ret;
}

/// @brief Compute inverse hyperbolic tangent: 0.5 * log((1+x)/(1-x))
inline
Halide::Func nh_atanh(Halide::Func f, const shape_t& shape, std::string const& name = "atanh")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	Halide::Expr val = f(vars);
	ret(vars) = 0.5f * Halide::log((1.0f + val) / (1.0f - val));
	return ret;
}

// -----------------------------------------------------------------------------
// Angle Conversions
// -----------------------------------------------------------------------------

/// @brief Convert radians to degrees
inline
Halide::Func degrees(Halide::Func f, const shape_t& shape, std::string const& name = "degrees")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	ret(vars) = f(vars) * (180.0f / 3.14159265358979323846f);
	return ret;
}

/// @brief Convert degrees to radians
inline
Halide::Func radians(Halide::Func f, const shape_t& shape, std::string const& name = "radians")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	ret(vars) = f(vars) * (3.14159265358979323846f / 180.0f);
	return ret;
}

// -----------------------------------------------------------------------------
// Two-argument Trigonometric Functions
// -----------------------------------------------------------------------------

/// @brief Compute atan2(y, x) element-wise
inline
Halide::Func nh_atan2(Halide::Func f_y, Halide::Func f_x, const shape_t& shape, std::string const& name = "atan2")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	ret(vars) = Halide::atan2(f_y(vars), f_x(vars));
	return ret;
}

/// @brief Compute hypotenuse: sqrt(x^2 + y^2) element-wise
inline
Halide::Func hypot(Halide::Func f_x, Halide::Func f_y, const shape_t& shape, std::string const& name = "hypot")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	Halide::Expr xv = f_x(vars);
	Halide::Expr yv = f_y(vars);
	ret(vars) = Halide::sqrt(xv * xv + yv * yv);
	return ret;
}

NS_NUM_HALIDE_END
