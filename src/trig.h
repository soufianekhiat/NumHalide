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
// Inverse Hyperbolic Functions
// -----------------------------------------------------------------------------

/// @brief Compute inverse hyperbolic sine: log(x + sqrt(x^2 + 1))
inline
Halide::Func asinh(Halide::Func f, const shape_t& shape, std::string const& name = "asinh_f")
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
Halide::Func acosh(Halide::Func f, const shape_t& shape, std::string const& name = "acosh_f")
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
Halide::Func atanh(Halide::Func f, const shape_t& shape, std::string const& name = "atanh_f")
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

// -----------------------------------------------------------------------------
// Basic Trigonometric Functions
// -----------------------------------------------------------------------------

/// @brief Compute sine element-wise
inline
Halide::Func sin(Halide::Func f, const shape_t& shape, std::string const& name = "sin_f")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	ret(vars) = Halide::sin(f(vars));
	return ret;
}

/// @brief Compute cosine element-wise
inline
Halide::Func cos(Halide::Func f, const shape_t& shape, std::string const& name = "cos_f")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	ret(vars) = Halide::cos(f(vars));
	return ret;
}

/// @brief Compute tangent element-wise
inline
Halide::Func tan(Halide::Func f, const shape_t& shape, std::string const& name = "tan_f")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	ret(vars) = Halide::tan(f(vars));
	return ret;
}

/// @brief Compute inverse sine element-wise
inline
Halide::Func asin(Halide::Func f, const shape_t& shape, std::string const& name = "asin_f")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	ret(vars) = Halide::asin(f(vars));
	return ret;
}

/// @brief Compute inverse cosine element-wise
inline
Halide::Func acos(Halide::Func f, const shape_t& shape, std::string const& name = "acos_f")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	ret(vars) = Halide::acos(f(vars));
	return ret;
}

/// @brief Compute inverse tangent element-wise
inline
Halide::Func atan(Halide::Func f, const shape_t& shape, std::string const& name = "atan_f")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	ret(vars) = Halide::atan(f(vars));
	return ret;
}

/// @brief Compute hyperbolic sine element-wise
inline
Halide::Func sinh(Halide::Func f, const shape_t& shape, std::string const& name = "sinh_f")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	ret(vars) = Halide::sinh(f(vars));
	return ret;
}

/// @brief Compute hyperbolic cosine element-wise
inline
Halide::Func cosh(Halide::Func f, const shape_t& shape, std::string const& name = "cosh_f")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	ret(vars) = Halide::cosh(f(vars));
	return ret;
}

/// @brief Compute hyperbolic tangent element-wise
inline
Halide::Func tanh(Halide::Func f, const shape_t& shape, std::string const& name = "tanh_f")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	ret(vars) = Halide::tanh(f(vars));
	return ret;
}

/// @brief Compute four-quadrant arctangent of f_y/f_x element-wise
inline
Halide::Func atan2(Halide::Func f_y, Halide::Func f_x, const shape_t& shape, std::string const& name = "atan2_f")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	ret(vars) = Halide::atan2(f_y(vars), f_x(vars));
	return ret;
}

/// @brief Wrap angles into [-pi, pi] element-wise (single-pass approximation)
/// @note Uses clamp(x - 2*pi*floor((x+pi)/(2*pi)), -pi, pi)
inline
Halide::Func unwrap(Halide::Func f, const shape_t& shape, std::string const& name = "unwrap_f")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	constexpr float pi = 3.14159265358979323846f;
	constexpr float two_pi = 2.0f * pi;
	Halide::Expr val = f(vars);
	ret(vars) = Halide::clamp(
		val - two_pi * Halide::floor((val + pi) / two_pi),
		-pi, pi
	);
	return ret;
}

NS_NUM_HALIDE_END
