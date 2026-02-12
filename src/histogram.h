/// @file histogram.h
/// @brief Histogram and LUT operations
///
/// Provides: histogram_1d, cumulative_histogram, histogram_equalize, apply_lut, gamma_correct

#pragma once

#include "common.h"
#include "shape.h"

NS_NUM_HALIDE_BEGIN

// -----------------------------------------------------------------------------
// Histogram Computation
// -----------------------------------------------------------------------------

/// @brief Compute 1D histogram of input data
/// @param f Input Func
/// @param shape Input shape (1D or 2D)
/// @param bins Number of histogram bins
/// @param range_min Minimum value of the range
/// @param range_max Maximum value of the range
/// @param name Function name
/// @return 1D Func with bin counts
inline
Halide::Func histogram_1d(Halide::Func f, const shape_t& shape, int bins,
                           Halide::Expr range_min = 0.0f, Halide::Expr range_max = 1.0f,
                           std::string const& name = "histogram")
{
	nh_require(nullptr, bins > 0, "histogram requires positive bin count");
	nh_require(nullptr, shape.rank >= 1 && shape.rank <= 2, "histogram supports 1D or 2D input");

	Halide::Func ret(name);
	Halide::Var bin;

	// Initialize bins to 0
	ret(bin) = 0;

	if (shape.rank == 1) {
		Halide::RDom r(0, shape.extents[0]);
		Halide::Expr val = f(r);
		Halide::Expr bin_idx = Halide::clamp(
			Halide::cast<int32_t>(Halide::floor((val - range_min) / (range_max - range_min) * bins)),
			0, bins - 1);
		ret(bin_idx) += 1;
	} else if (shape.rank == 2) {
		Halide::RDom r(0, shape.extents[1], 0, shape.extents[0]);
		Halide::Expr val = f(r.x, r.y);
		Halide::Expr bin_idx = Halide::clamp(
			Halide::cast<int32_t>(Halide::floor((val - range_min) / (range_max - range_min) * bins)),
			0, bins - 1);
		ret(bin_idx) += 1;
	}

	return ret;
}

// -----------------------------------------------------------------------------
// Cumulative Histogram (CDF)
// -----------------------------------------------------------------------------

/// @brief Compute cumulative histogram (cumulative distribution function)
/// @param f Input Func
/// @param shape Input shape (1D or 2D)
/// @param bins Number of histogram bins
/// @param range_min Minimum value of the range
/// @param range_max Maximum value of the range
/// @param name Function name
/// @return 1D Func with cumulative bin counts
inline
Halide::Func cumulative_histogram(Halide::Func f, const shape_t& shape, int bins,
                                   Halide::Expr range_min = 0.0f, Halide::Expr range_max = 1.0f,
                                   std::string const& name = "cum_hist")
{
	auto hist = histogram_1d(f, shape, bins, range_min, range_max, name + "_hist");
	hist.compute_root();

	// Cumulative sum of histogram
	Halide::Func ret(name);
	Halide::Var x;
	ret(x) = hist(x);
	Halide::RDom r(1, bins - 1);
	ret(r) = ret(r - 1) + hist(r);
	return ret;
}

// -----------------------------------------------------------------------------
// Histogram Equalization
// -----------------------------------------------------------------------------

/// @brief Histogram equalization for contrast enhancement
/// @param f Input Func with values in [0, 1]
/// @param shape Input shape
/// @param bins Number of histogram bins
/// @param name Function name
/// @return Func with equalized values in [0, 1]
inline
Halide::Func histogram_equalize(Halide::Func f, const shape_t& shape, int bins = 256, std::string const& name = "histeq")
{
	auto cdf = cumulative_histogram(f, shape, bins, 0.0f, 1.0f, name + "_cdf");
	cdf.compute_root();

	// Total elements
	int total = 1;
	for (int i = 0; i < shape.rank; ++i) {
		total *= shape.extents[i];
	}

	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	Halide::Expr val = f(vars);
	Halide::Expr bin_idx = Halide::clamp(Halide::cast<int32_t>(Halide::floor(val * bins)), 0, bins - 1);
	ret(vars) = Halide::cast<float>(cdf(bin_idx)) / Halide::cast<float>(total);
	return ret;
}

// -----------------------------------------------------------------------------
// Lookup Table
// -----------------------------------------------------------------------------

/// @brief Apply a lookup table to input data
/// @param f Input Func (integer-valued indices)
/// @param lut Lookup table Func
/// @param shape Input shape
/// @param name Function name
/// @return Func: ret(vars) = lut(cast<int>(f(vars)))
inline
Halide::Func apply_lut(Halide::Func f, Halide::Func lut, const shape_t& shape, std::string const& name = "lut")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	ret(vars) = lut(Halide::cast<int32_t>(f(vars)));
	return ret;
}

// -----------------------------------------------------------------------------
// Gamma Correction
// -----------------------------------------------------------------------------

/// @brief Apply gamma correction: pow(f, 1/gamma)
/// @param f Input Func with values in [0, 1]
/// @param shape Input shape
/// @param gamma Gamma value
/// @param name Function name
/// @return Func with gamma-corrected values
inline
Halide::Func gamma_correct(Halide::Func f, const shape_t& shape, Halide::Expr gamma, std::string const& name = "gamma")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	ret(vars) = Halide::pow(f(vars), 1.0f / gamma);
	return ret;
}

NS_NUM_HALIDE_END
