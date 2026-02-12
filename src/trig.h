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

NS_NUM_HALIDE_END
