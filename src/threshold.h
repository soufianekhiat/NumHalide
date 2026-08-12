/// @file threshold.h
/// @brief Thresholding operations
///
/// Provides: threshold_binary, threshold_trunc, threshold_tozero,
///           threshold_otsu, threshold_adaptive

#pragma once

#include "common.h"
#include "shape.h"
#include "histogram.h"

NS_NUM_HALIDE_BEGIN

// -----------------------------------------------------------------------------
// Basic Thresholding
// -----------------------------------------------------------------------------

/// @brief Binary threshold: x > thresh ? 1 : 0
/// @param f Input Func
/// @param shape Input shape
/// @param thresh Threshold value
/// @param name Function name
/// @return Func with binary output
inline
Halide::Func threshold_binary(Halide::Func f, const shape_t& shape, Halide::Expr thresh, std::string const& name = "thresh_bin")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	// Emit in the input's own type (f32-hardcoding breaks f64/half combos).
	Halide::Type type = f.types()[0];
	ret(vars) = Halide::select(f(vars) > Halide::cast(type, thresh),
		Halide::cast(type, 1), Halide::cast(type, 0));
	return ret;
}

/// @brief Truncation threshold: min(x, thresh)
/// @param f Input Func
/// @param shape Input shape
/// @param thresh Threshold value
/// @param name Function name
/// @return Func with values clamped to thresh
inline
Halide::Func threshold_trunc(Halide::Func f, const shape_t& shape, Halide::Expr thresh, std::string const& name = "thresh_trunc")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	Halide::Type type = f.types()[0];
	ret(vars) = Halide::min(f(vars), Halide::cast(type, thresh));
	return ret;
}

/// @brief To-zero threshold: x > thresh ? x : 0
/// @param f Input Func
/// @param shape Input shape
/// @param thresh Threshold value
/// @param name Function name
/// @return Func with values below threshold set to zero
inline
Halide::Func threshold_tozero(Halide::Func f, const shape_t& shape, Halide::Expr thresh, std::string const& name = "thresh_tozero")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	Halide::Type type = f.types()[0];
	ret(vars) = Halide::select(f(vars) > Halide::cast(type, thresh),
		f(vars), Halide::cast(type, 0));
	return ret;
}

// -----------------------------------------------------------------------------
// Otsu's Method
// -----------------------------------------------------------------------------

/// @brief Automatic thresholding using Otsu's method
/// Computes the optimal threshold that maximizes between-class variance,
/// then applies binary thresholding.
/// @param f Input Func with values in [0, 1]
/// @param shape Input shape
/// @param bins Number of histogram bins
/// @param name Function name
/// @return Func with binary output using optimal threshold
inline
Halide::Func threshold_otsu(Halide::Func f, const shape_t& shape, int bins = 256, std::string const& name = "otsu")
{
	// Compute histogram
	auto hist = histogram_1d(f, shape, bins, 0.0f, 1.0f, name + "_hist");
	hist.compute_root();

	// Total count
	int total = 1;
	for (int i = 0; i < shape.rank; ++i) {
		total *= shape.extents[i];
	}

	// Precompute cumulative count and cumulative weighted sum
	Halide::Func cum_count(name + "_cc");
	Halide::Func cum_sum(name + "_cs");
	Halide::Var x;

	cum_count(x) = hist(x);
	cum_sum(x) = Halide::cast<float>(x) * Halide::cast<float>(hist(x));

	Halide::RDom r(1, bins - 1);
	cum_count(r) = cum_count(r - 1) + hist(r);
	cum_sum(r) = cum_sum(r - 1) + Halide::cast<float>(r) * Halide::cast<float>(hist(r));
	cum_count.compute_root();
	cum_sum.compute_root();

	// Find threshold maximizing between-class variance
	Halide::Func best(name + "_best");
	Halide::Var dummy;
	best(dummy) = Halide::Tuple(0.0f, 0);  // (best_variance, best_threshold)

	Halide::RDom t(0, bins);
	Halide::Expr w0 = Halide::cast<float>(cum_count(t));
	Halide::Expr w1 = Halide::cast<float>(total) - w0;
	Halide::Expr mu0 = Halide::select(w0 > 0, cum_sum(t) / w0, 0.0f);
	Halide::Expr total_sum = cum_sum(bins - 1);
	Halide::Expr mu1 = Halide::select(w1 > 0, (total_sum - cum_sum(t)) / w1, 0.0f);
	Halide::Expr between_var = w0 * w1 * (mu0 - mu1) * (mu0 - mu1);

	Halide::Expr cur_best_var = best(0)[0];
	best(0) = Halide::select(between_var > cur_best_var,
		Halide::Tuple(between_var, t),
		best(0));
	best.compute_root();

	// Get optimal threshold as float in [0, 1]
	Halide::Expr opt_thresh = Halide::cast<float>(best(0)[1]) / Halide::cast<float>(bins);

	// Apply threshold — emit in the input's own type (f32-hardcoded output
	// arms broke f64 inputs). The histogram statistics above deliberately
	// stay f32: they operate on integer bin counts and the threshold is
	// quantized to 1/bins anyway.
	Halide::Type type = f.types()[0];
	if (!type.is_float()) type = Halide::Float(32);

	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	ret(vars) = Halide::select(f(vars) > opt_thresh,
		Halide::cast(type, 1), Halide::cast(type, 0));
	return ret;
}

// -----------------------------------------------------------------------------
// Adaptive Thresholding
// -----------------------------------------------------------------------------

/// @brief Adaptive (local mean) thresholding for 2D images
/// Computes a local mean using a box filter of given block_size,
/// then thresholds each pixel against its local mean.
/// @param f Input 2D Func
/// @param shape Input shape (must be 2D)
/// @param block_size Size of the local neighborhood (should be odd)
/// @param name Function name
/// @return Func with binary output: pixel > local_mean ? 1 : 0
inline
Halide::Func threshold_adaptive(Halide::Func f, const shape_t& shape, int block_size, std::string const& name = "adaptive")
{
	nh_require(shape.rank == 2, "adaptive threshold requires 2D");
	int rows = shape.extents[0];
	int cols = shape.extents[1];
	int half = block_size / 2;

	// Compute local mean using box filter
	Halide::Func local_mean(name + "_mean");
	Halide::Var x, y;
	Halide::RDom r(-half, block_size, -half, block_size);

	Halide::Expr ix = Halide::clamp(x + r.x, 0, cols - 1);
	Halide::Expr iy = Halide::clamp(y + r.y, 0, rows - 1);

	// Accumulate and emit in the input's own float type (f64 stays f64);
	// integer inputs keep the historical f32 path.
	Halide::Type type = f.types()[0];
	if (!type.is_float()) type = Halide::Float(32);

	local_mean(x, y) = Halide::cast(type, 0);
	local_mean(x, y) += f(ix, iy);

	// Normalize by block area
	Halide::Func mean_norm(name + "_mn");
	mean_norm(x, y) = local_mean(x, y) / Halide::cast(type, block_size * block_size);

	// Threshold: pixel > local_mean
	Halide::Func ret(name);
	ret(x, y) = Halide::select(f(x, y) > mean_norm(x, y),
		Halide::cast(type, 1), Halide::cast(type, 0));
	return ret;
}

NS_NUM_HALIDE_END
