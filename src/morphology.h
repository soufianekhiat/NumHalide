/// @file morphology.h
/// @brief Morphological operations for 2D images
///
/// Provides: dilate, erode, morph_open, morph_close, morph_gradient, top_hat

#pragma once

#include "common.h"
#include "shape.h"

NS_NUM_HALIDE_BEGIN

// -----------------------------------------------------------------------------
// Dilation
// -----------------------------------------------------------------------------

/// @brief Morphological dilation (local maximum) with square structuring element
/// @param f Input 2D Func
/// @param shape Shape of input (must be 2D)
/// @param kernel_size Size of square kernel (e.g., 3 for 3x3)
/// @param name Function name
/// @return Func with local maximum values
inline
Halide::Func dilate(Halide::Func f, const shape_t& shape, int kernel_size, std::string const& name = "dilate")
{
	nh_require(shape.rank == 2, "dilate requires 2D input, got rank %d", shape.rank);
	nh_require(kernel_size > 0 && (kernel_size % 2) == 1,
		"dilate requires odd positive kernel_size, got %d", kernel_size);

	int rows = shape.extents[0];
	int cols = shape.extents[1];
	int half = kernel_size / 2;

	Halide::Func ret(name);
	Halide::Var x, y;
	Halide::RDom r(-half, kernel_size, -half, kernel_size);

	Halide::Expr ix = Halide::clamp(x + r.x, 0, cols - 1);
	Halide::Expr iy = Halide::clamp(y + r.y, 0, rows - 1);

	// Initialize to type minimum, then take max over neighborhood
	ret(x, y) = f.types()[0].min();
	ret(x, y) = Halide::max(ret(x, y), f(ix, iy));

	return ret;
}

// -----------------------------------------------------------------------------
// Erosion
// -----------------------------------------------------------------------------

/// @brief Morphological erosion (local minimum) with square structuring element
/// @param f Input 2D Func
/// @param shape Shape of input (must be 2D)
/// @param kernel_size Size of square kernel (e.g., 3 for 3x3)
/// @param name Function name
/// @return Func with local minimum values
inline
Halide::Func erode(Halide::Func f, const shape_t& shape, int kernel_size, std::string const& name = "erode")
{
	nh_require(shape.rank == 2, "erode requires 2D input, got rank %d", shape.rank);
	nh_require(kernel_size > 0 && (kernel_size % 2) == 1,
		"erode requires odd positive kernel_size, got %d", kernel_size);

	int rows = shape.extents[0];
	int cols = shape.extents[1];
	int half = kernel_size / 2;

	Halide::Func ret(name);
	Halide::Var x, y;
	Halide::RDom r(-half, kernel_size, -half, kernel_size);

	Halide::Expr ix = Halide::clamp(x + r.x, 0, cols - 1);
	Halide::Expr iy = Halide::clamp(y + r.y, 0, rows - 1);

	// Initialize to type maximum, then take min over neighborhood
	ret(x, y) = f.types()[0].max();
	ret(x, y) = Halide::min(ret(x, y), f(ix, iy));

	return ret;
}

// -----------------------------------------------------------------------------
// Compound Morphological Operations
// -----------------------------------------------------------------------------

/// @brief Morphological opening: erosion followed by dilation
/// @param f Input 2D Func
/// @param shape Shape of input (must be 2D)
/// @param kernel_size Size of square kernel
/// @param name Function name
/// @return Opened image (removes small bright regions)
inline
Halide::Func morph_open(Halide::Func f, const shape_t& shape, int kernel_size, std::string const& name = "morph_open")
{
	auto eroded = erode(f, shape, kernel_size, name + "_erode");
	return dilate(eroded, shape, kernel_size, name);
}

/// @brief Morphological closing: dilation followed by erosion
/// @param f Input 2D Func
/// @param shape Shape of input (must be 2D)
/// @param kernel_size Size of square kernel
/// @param name Function name
/// @return Closed image (fills small dark regions)
inline
Halide::Func morph_close(Halide::Func f, const shape_t& shape, int kernel_size, std::string const& name = "morph_close")
{
	auto dilated = dilate(f, shape, kernel_size, name + "_dilate");
	return erode(dilated, shape, kernel_size, name);
}

/// @brief Morphological gradient: dilation minus erosion
/// @param f Input 2D Func
/// @param shape Shape of input (must be 2D)
/// @param kernel_size Size of square kernel
/// @param name Function name
/// @return Edge-like image highlighting boundaries
inline
Halide::Func morph_gradient(Halide::Func f, const shape_t& shape, int kernel_size, std::string const& name = "morph_gradient")
{
	nh_require(shape.rank == 2, "morph_gradient requires 2D input, got rank %d", shape.rank);

	auto dilated = dilate(f, shape, kernel_size, name + "_dilate");
	auto eroded = erode(f, shape, kernel_size, name + "_erode");

	Halide::Func ret(name);
	Halide::Var x, y;

	ret(x, y) = dilated(x, y) - eroded(x, y);

	return ret;
}

/// @brief Top-hat transform: original minus opening
/// @param f Input 2D Func
/// @param shape Shape of input (must be 2D)
/// @param kernel_size Size of square kernel
/// @param name Function name
/// @return Small bright features extracted from image
inline
Halide::Func top_hat(Halide::Func f, const shape_t& shape, int kernel_size, std::string const& name = "top_hat")
{
	nh_require(shape.rank == 2, "top_hat requires 2D input, got rank %d", shape.rank);

	auto opened = morph_open(f, shape, kernel_size, name + "_open");

	Halide::Func ret(name);
	Halide::Var x, y;

	ret(x, y) = f(x, y) - opened(x, y);

	return ret;
}

NS_NUM_HALIDE_END
