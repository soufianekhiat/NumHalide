/// @file gradient.h
/// @brief Gradient and differential operators for arrays
///
/// Provides: gradient_1d, gradient_2d, laplacian, divergence

#pragma once

#include "common.h"
#include "shape.h"

NS_NUM_HALIDE_BEGIN

// -----------------------------------------------------------------------------
// 1D Gradient (Central Differences)
// -----------------------------------------------------------------------------

/// @brief Compute gradient along a single axis using central differences
/// @param f Input Func
/// @param shape Shape of input (1D or 2D)
/// @param axis Axis along which to compute the gradient (default 0)
/// @param name Function name
/// @return Func with gradient values: (f[i+1] - f[i-1]) / 2 with boundary clamping
///
/// For 1D: ret(x) = (f(clamp(x+1)) - f(clamp(x-1))) / 2.0f
/// For 2D axis=1 (cols): ret(x,y) = (f(clamp(x+1),y) - f(clamp(x-1),y)) / 2.0f
/// For 2D axis=0 (rows): ret(x,y) = (f(x,clamp(y+1)) - f(x,clamp(y-1))) / 2.0f
inline
Halide::Func gradient_1d(Halide::Func f, const shape_t& shape, int axis = 0, std::string const& name = "gradient")
{
	nh_require(nullptr, shape.rank >= 1 && shape.rank <= 2,
		"gradient_1d requires 1D or 2D input, got rank %d", shape.rank);

	int norm_axis = normalized_axis(axis, shape.rank);

	Halide::Func ret(name);

	if (shape.rank == 1) {
		Halide::Var x;
		int n = shape.extents[0];

		Halide::Expr xp = Halide::clamp(x + 1, 0, n - 1);
		Halide::Expr xm = Halide::clamp(x - 1, 0, n - 1);

		ret(x) = (f(xp) - f(xm)) / 2.0f;
	} else { // rank == 2
		Halide::Var x, y;
		int rows = shape.extents[0]; // y dimension
		int cols = shape.extents[1]; // x dimension

		if (norm_axis == 1) {
			// Along cols (Halide x axis)
			Halide::Expr xp = Halide::clamp(x + 1, 0, cols - 1);
			Halide::Expr xm = Halide::clamp(x - 1, 0, cols - 1);
			ret(x, y) = (f(xp, y) - f(xm, y)) / 2.0f;
		} else {
			// Along rows (Halide y axis)
			Halide::Expr yp = Halide::clamp(y + 1, 0, rows - 1);
			Halide::Expr ym = Halide::clamp(y - 1, 0, rows - 1);
			ret(x, y) = (f(x, yp) - f(x, ym)) / 2.0f;
		}
	}

	return ret;
}

// -----------------------------------------------------------------------------
// 2D Gradient (Both Axes)
// -----------------------------------------------------------------------------

/// @brief Compute 2D gradient returning both partial derivatives
/// @param f Input 2D Func
/// @param shape Shape of input (must be 2D)
/// @param name Function name
/// @return Pair of {grad_x, grad_y} Funcs using central differences
inline
std::pair<Halide::Func, Halide::Func> gradient_2d(Halide::Func f, const shape_t& shape, std::string const& name = "gradient2d")
{
	nh_require(nullptr, shape.rank == 2, "gradient_2d requires 2D input, got rank %d", shape.rank);

	auto grad_x = gradient_1d(f, shape, 1, name + "_x");
	auto grad_y = gradient_1d(f, shape, 0, name + "_y");

	return { grad_x, grad_y };
}

// -----------------------------------------------------------------------------
// Laplacian
// -----------------------------------------------------------------------------

/// @brief Compute discrete Laplacian of a 2D array
/// @param f Input 2D Func
/// @param shape Shape of input (must be 2D)
/// @param name Function name
/// @return Func with Laplacian: f(x+1,y) + f(x-1,y) + f(x,y+1) + f(x,y-1) - 4*f(x,y)
///
/// Uses the standard 5-point stencil with boundary clamping.
inline
Halide::Func laplacian(Halide::Func f, const shape_t& shape, std::string const& name = "laplacian")
{
	nh_require(nullptr, shape.rank == 2, "laplacian requires 2D input, got rank %d", shape.rank);

	int rows = shape.extents[0];
	int cols = shape.extents[1];

	Halide::Func ret(name);
	Halide::Var x, y;

	Halide::Expr xp = Halide::clamp(x + 1, 0, cols - 1);
	Halide::Expr xm = Halide::clamp(x - 1, 0, cols - 1);
	Halide::Expr yp = Halide::clamp(y + 1, 0, rows - 1);
	Halide::Expr ym = Halide::clamp(y - 1, 0, rows - 1);

	ret(x, y) = f(xp, y) + f(xm, y) + f(x, yp) + f(x, ym) - 4.0f * f(x, y);

	return ret;
}

// -----------------------------------------------------------------------------
// Divergence
// -----------------------------------------------------------------------------

/// @brief Compute divergence of a 2D vector field: dfx/dx + dfy/dy
/// @param fx X-component of vector field (2D Func)
/// @param fy Y-component of vector field (2D Func)
/// @param shape Shape of input (must be 2D)
/// @param name Function name
/// @return Func with divergence values using central differences
inline
Halide::Func divergence(Halide::Func fx, Halide::Func fy, const shape_t& shape, std::string const& name = "divergence")
{
	nh_require(nullptr, shape.rank == 2, "divergence requires 2D input, got rank %d", shape.rank);

	// dfx/dx: gradient of fx along x (axis=1)
	auto gx = gradient_1d(fx, shape, 1, name + "_dfdx");
	// dfy/dy: gradient of fy along y (axis=0)
	auto gy = gradient_1d(fy, shape, 0, name + "_dfdy");

	Halide::Func ret(name);
	Halide::Var x, y;

	ret(x, y) = gx(x, y) + gy(x, y);

	return ret;
}

NS_NUM_HALIDE_END
