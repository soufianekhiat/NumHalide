/// @file stats_ext.h
/// @brief Extended statistical operations
///
/// Provides: median, ptp, average (weighted mean), histogram, digitize

#pragma once

#include "common.h"
#include "shape.h"
#include "reduce.h"
#include "sort.h"

NS_NUM_HALIDE_BEGIN

namespace stats {

// -----------------------------------------------------------------------------
// Median (1D, power-of-2 size via bitonic sort)
// -----------------------------------------------------------------------------

/// @brief Compute median of a 1D array
/// @param f Input Func (1D)
/// @param shape Shape of input (must be 1D with power-of-2 size)
/// @param name Function name
/// @return 1D Func with single element containing the median
///
/// Uses bitonic sort internally. Array size must be a power of 2.
/// For even-sized arrays, returns the lower-middle element (integer-style median).
/// For float types, returns the average of the two middle elements.
///
/// Usage:
///   Func med = median(f, {8});
inline
Halide::Func median(Halide::Func f, const shape_t& shape, std::string const& name = "median")
{
	nh_require(shape.rank == 1, "median currently supports 1D arrays only");
	int n = shape.extents[0];
	nh_require(n > 0, "median: array must have at least 1 element");
	nh_require((n & (n - 1)) == 0,
		"median: size must be a power of 2 for bitonic sort, got %d", n);

	Halide::Func sorted = bitonic_sort(f, n, name + "_sorted");

	Halide::Func ret(name);
	Halide::Var x;

	Halide::Type t = f.types()[0];

	if (n % 2 == 1) {
		// Odd: middle element
		ret(x) = sorted(n / 2);
	}
	else {
		// Even: average of two middle elements
		// For integer types this will truncate; for float types it gives the true median
		Halide::Expr lo = sorted(n / 2 - 1);
		Halide::Expr hi = sorted(n / 2);
		if (t.is_float()) {
			ret(x) = (lo + hi) / Halide::cast(t, 2);
		}
		else {
			ret(x) = (lo + hi) / Halide::cast(t, 2);
		}
	}

	return ret;
}

// -----------------------------------------------------------------------------
// Peak-to-Peak (ptp)
// -----------------------------------------------------------------------------

/// @brief Compute peak-to-peak (max - min) along specified axes
/// @param f Input Func
/// @param in_shape Input shape
/// @param axes Axes to reduce over
/// @param keepdims If true, reduced axes have size 1
/// @param name Function name
/// @return Func with peak-to-peak values
///
/// Equivalent to reduce_max(f, ...) - reduce_min(f, ...).
///
/// Usage:
///   Func p = ptp(f, {3, 4}, {0});
inline
Halide::Func ptp(Halide::Func f, const shape_t& in_shape,
                 const std::vector<int>& axes,
                 bool keepdims = false,
                 std::string const& name = "ptp")
{
	Halide::Func f_max = reduce_max(f, in_shape, axes, keepdims, name + "_max");
	Halide::Func f_min = reduce_min(f, in_shape, axes, keepdims, name + "_min");

	// Normalize axes for output shape
	std::vector<int> norm_axes;
	for (int ax : axes) {
		norm_axes.push_back(normalized_axis(ax, in_shape.rank));
	}
	shape_t out_shape = infer_reduce(in_shape, norm_axes, keepdims);

	std::vector<Halide::Var> out_vars;
	for (int i = 0; i < out_shape.rank; ++i) {
		out_vars.push_back(Halide::Var());
	}

	Halide::Func ret(name);
	ret(out_vars) = f_max(out_vars) - f_min(out_vars);

	return ret;
}

/// @brief Compute peak-to-peak of all elements (full reduction)
/// @param f Input Func
/// @param in_shape Input shape
/// @param name Function name
/// @return 1D Func with single element
inline
Halide::Func ptp(Halide::Func f, const shape_t& in_shape,
                 std::string const& name = "ptp")
{
	std::vector<int> all_axes;
	for (int i = 0; i < in_shape.rank; ++i) all_axes.push_back(i);

	Halide::Func reduced = ptp(f, in_shape, all_axes, true, name + "_inner");

	Halide::Func ret(name);
	Halide::Var x;
	std::vector<Halide::Expr> zeros(in_shape.rank, 0);
	ret(x) = reduced(zeros);

	return ret;
}

/// @brief Compute peak-to-peak along a single axis
/// @param f Input Func
/// @param in_shape Input shape
/// @param axis Axis to reduce
/// @param keepdims If true, keep the reduced dimension
/// @param name Function name
/// @return Func with peak-to-peak values
inline
Halide::Func ptp(Halide::Func f, const shape_t& in_shape, int axis,
                 bool keepdims = false,
                 std::string const& name = "ptp")
{
	std::vector<int> axes = {axis};
	return ptp(f, in_shape, axes, keepdims, name);
}

// -----------------------------------------------------------------------------
// Weighted Average
// -----------------------------------------------------------------------------

/// @brief Compute weighted average of a 1D array
/// @param f Input Func (1D)
/// @param weights Weight Func (1D, same size as f)
/// @param shape Shape of input (must be 1D)
/// @param name Function name
/// @return 1D Func with single element containing the weighted average
///
/// Formula: average = sum(f * weights) / sum(weights)
///
/// Usage:
///   Func avg = average(f, w, {8});
inline
Halide::Func average(Halide::Func f, Halide::Func weights, const shape_t& shape,
                     std::string const& name = "average")
{
	nh_require(shape.rank == 1, "average currently supports 1D arrays only");
	int n = shape.extents[0];

	Halide::Var x;
	Halide::RDom r(0, n);
	Halide::Type t = f.types()[0];

	// Compute weighted sum
	Halide::Func wsum(name + "_wsum");
	wsum(x) = Halide::cast(t, 0);
	wsum(x) += f(r) * weights(r);

	// Compute sum of weights
	Halide::Func wtotal(name + "_wtotal");
	wtotal(x) = Halide::cast(t, 0);
	wtotal(x) += weights(r);

	// Weighted average = weighted_sum / sum_of_weights
	Halide::Func ret(name);
	ret(x) = wsum(x) / wtotal(x);

	return ret;
}

// -----------------------------------------------------------------------------
// Histogram
// -----------------------------------------------------------------------------

/// @brief Compute histogram of a 1D array
/// @param f Input Func (1D)
/// @param shape Shape of input (must be 1D)
/// @param n_bins Number of histogram bins
/// @param range_min Minimum value of the histogram range
/// @param range_max Maximum value of the histogram range
/// @param name Function name
/// @return 1D Func of size n_bins with bin counts (int32)
///
/// Values outside [range_min, range_max) are placed in the first/last bins.
/// Bin i covers [range_min + i*bin_width, range_min + (i+1)*bin_width).
///
/// Usage:
///   Func h = histogram(f, {100}, 10, 0.0f, 1.0f);
inline
Halide::Func histogram(Halide::Func f, const shape_t& shape,
                       int n_bins, Halide::Expr range_min, Halide::Expr range_max,
                       std::string const& name = "histogram")
{
	nh_require(shape.rank == 1, "histogram currently supports 1D arrays only");
	nh_require(n_bins > 0, "histogram: n_bins must be > 0");
	int n = shape.extents[0];

	Halide::Func ret(name);
	Halide::Var bin;
	Halide::RDom r(0, n);

	// Initialize all bins to 0
	ret(bin) = Halide::cast<int32_t>(0);

	// Compute bin width
	Halide::Expr bin_width = (range_max - range_min) / Halide::cast(f.types()[0], n_bins);

	// For each element, determine its bin and increment
	Halide::Expr val = f(r);
	Halide::Expr bin_idx = Halide::cast<int32_t>(Halide::floor((val - range_min) / bin_width));
	// Clamp to valid range [0, n_bins - 1]
	bin_idx = Halide::clamp(bin_idx, 0, n_bins - 1);

	// Scatter-add: increment the appropriate bin
	ret(bin_idx) += 1;

	return ret;
}

// -----------------------------------------------------------------------------
// Digitize
// -----------------------------------------------------------------------------

/// @brief Find bin indices for each value in a 1D array given sorted bin edges
/// @param f Input Func (1D) with values to digitize
/// @param bins Sorted Func (1D) containing bin edge values
/// @param shape Shape of input f (must be 1D)
/// @param n_bins Number of bin edges in bins Func
/// @param right If true, bins[i-1] <= x < bins[i]; if false, bins[i-1] < x <= bins[i]
/// @param name Function name
/// @return 1D Func with bin indices for each element of f
///
/// Returns indices such that bins[i-1] <= x < bins[i] (right=false) or
/// bins[i-1] < x <= bins[i] (right=true). Values below all edges return 0,
/// values above all edges return n_bins.
///
/// Usage:
///   Func idx = digitize(values, bin_edges, {100}, 5);
inline
Halide::Func digitize(Halide::Func f, Halide::Func bins, const shape_t& shape,
                      int n_bins, bool right = false,
                      std::string const& name = "digitize")
{
	nh_require(shape.rank == 1, "digitize currently supports 1D arrays only");
	nh_require(n_bins > 0, "digitize: n_bins must be > 0");

	Halide::Func ret(name);
	Halide::Var x;
	Halide::RDom r(0, n_bins);

	// For each value, count how many bin edges it exceeds
	// This gives the insertion index (equivalent to searchsorted)
	ret(x) = Halide::cast<int32_t>(0);

	if (right) {
		// right=true: bins[i-1] < x <= bins[i], so count bins[r] < val
		ret(x) += Halide::cast<int32_t>(Halide::select(bins(r) < f(x), 1, 0));
	}
	else {
		// right=false (default): bins[i-1] <= x < bins[i], so count bins[r] <= val
		ret(x) += Halide::cast<int32_t>(Halide::select(bins(r) <= f(x), 1, 0));
	}

	return ret;
}

} // namespace stats

NS_NUM_HALIDE_END
