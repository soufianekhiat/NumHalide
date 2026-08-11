/// @file cumulative.h
/// @brief Cumulative and discrete-difference operations
///
/// Provides: cumsum, cumprod, diff

#pragma once

#include "common.h"
#include "shape.h"

NS_NUM_HALIDE_BEGIN

// -----------------------------------------------------------------------------
// Cumulative Sum
// -----------------------------------------------------------------------------

/// @brief Cumulative sum along an axis (1D only)
/// @param f Input Func (1D)
/// @param shape Shape of input
/// @param axis Axis to accumulate along (must be 0 for 1D)
/// @param name Function name
/// @return Func with cumulative sums
///
/// For a 1D array [a, b, c, d], returns [a, a+b, a+b+c, a+b+c+d].
/// Uses a serial scan pattern: ret(0) = f(0), ret(i) = ret(i-1) + f(i).
///
/// Usage:
///   Func cs = cumsum(f, {8});
inline
Halide::Func cumsum(Halide::Func f, const shape_t& shape, int axis = 0, std::string const& name = "cumsum")
{
	nh_require(shape.rank == 1, "cumsum currently supports 1D arrays only");
	int norm_axis = normalized_axis(axis, shape.rank);
	nh_require(norm_axis == 0, "cumsum: axis must be 0 for 1D arrays");

	int n = shape.extents[0];

	Halide::Func ret(name);
	Halide::Var x;

	// Pure definition: copy input
	ret(x) = f(x);

	// Serial scan update: ret(r) = ret(r-1) + f(r) for r = 1..n-1
	// At r=1: ret(1) = ret(0) + f(1) = f(0) + f(1)
	// At r=2: ret(2) = ret(1) + f(2) = f(0) + f(1) + f(2)
	// etc.
	Halide::RDom r(1, n - 1);
	ret(r) = ret(r - 1) + f(r);

	// Force sequential evaluation for correctness
	ret.compute_root();

	return ret;
}

/// @brief Cumulative sum with a RUNTIME length (1D)
/// @param f Input Func (1D)
/// @param n Number of elements as a runtime expression (e.g. a buffer extent)
/// @param name Function name
/// @return Func with cumulative sums
///
/// Same sequential scan as the compile-time overload:
/// ret(0) = f(0), ret(i) = ret(i-1) + f(i). Reverse-mode derivability of
/// this scan form is verified. Type follows f.types()[0].
inline
Halide::Func cumsum(Halide::Func f, Halide::Expr n, std::string const& name = "cumsum_rt")
{
	Halide::Func ret(name);
	Halide::Var x("x");

	// Pure definition: copy input
	ret(x) = f(x);

	// Serial scan update: ret(r) = ret(r-1) + f(r) for r = 1..n-1
	Halide::RDom r(1, Halide::max(n - 1, 0), "r_" + name);
	ret(r) = ret(r - 1) + f(r);

	// Force sequential evaluation for correctness
	ret.compute_root();

	return ret;
}

// -----------------------------------------------------------------------------
// Cumulative Product
// -----------------------------------------------------------------------------

/// @brief Cumulative product along an axis (1D only)
/// @param f Input Func (1D)
/// @param shape Shape of input
/// @param axis Axis to accumulate along (must be 0 for 1D)
/// @param name Function name
/// @return Func with cumulative products
///
/// For a 1D array [a, b, c, d], returns [a, a*b, a*b*c, a*b*c*d].
///
/// Usage:
///   Func cp = cumprod(f, {8});
inline
Halide::Func cumprod(Halide::Func f, const shape_t& shape, int axis = 0, std::string const& name = "cumprod")
{
	nh_require(shape.rank == 1, "cumprod currently supports 1D arrays only");
	int norm_axis = normalized_axis(axis, shape.rank);
	nh_require(norm_axis == 0, "cumprod: axis must be 0 for 1D arrays");

	int n = shape.extents[0];

	Halide::Func ret(name);
	Halide::Var x;

	// Pure definition: copy input
	ret(x) = f(x);

	// Serial scan update: ret(r) = ret(r-1) * f(r) for r = 1..n-1
	Halide::RDom r(1, n - 1);
	ret(r) = ret(r - 1) * f(r);

	// Force sequential evaluation for correctness
	ret.compute_root();

	return ret;
}

// -----------------------------------------------------------------------------
// Discrete Difference
// -----------------------------------------------------------------------------

/// @brief Discrete difference along an axis (1D only)
/// @param f Input Func (1D)
/// @param shape Shape of input
/// @param axis Axis to difference along (must be 0 for 1D)
/// @param n Order of the difference (currently only n=1 supported)
/// @param name Function name
/// @return Func with discrete differences (output has size - 1 elements)
///
/// For a 1D array [a, b, c, d], returns [b-a, c-b, d-c].
/// Output has shape.extents[0]-1 elements.
///
/// Usage:
///   Func d = diff(f, {8});  // returns 7 elements
inline
Halide::Func diff(Halide::Func f, const shape_t& shape, int axis = 0, int n = 1, std::string const& name = "diff")
{
	nh_require(shape.rank == 1, "diff currently supports 1D arrays only");
	int norm_axis = normalized_axis(axis, shape.rank);
	nh_require(norm_axis == 0, "diff: axis must be 0 for 1D arrays");
	nh_require(n >= 1, "diff: n must be >= 1");
	nh_require(shape.extents[0] > n, "diff: array size must be greater than n");

	Halide::Func ret(name);
	Halide::Var x;

	// First-order difference: ret(x) = f(x+1) - f(x)
	ret(x) = f(x + 1) - f(x);

	// For higher orders, apply diff recursively
	Halide::Func current = ret;
	int current_size = shape.extents[0] - 1;
	for (int i = 1; i < n; ++i) {
		Halide::Func next(name + "_order" + std::to_string(i + 1));
		Halide::Var y;
		next(y) = current(y + 1) - current(y);
		current.compute_root();
		current = next;
		current_size--;
	}

	return current;
}

NS_NUM_HALIDE_END
