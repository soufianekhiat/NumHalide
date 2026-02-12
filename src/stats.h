/// @file stats.h
/// @brief Statistical operations for Halide::Func objects
///
/// Provides: var, std with axis control and keepdims
/// Note: median and percentile require sorting (see M16)

#pragma once

#include "common.h"
#include "numhalide.h"
#include "shape.h"
#include "reduce.h"
#include <algorithm>

NS_NUM_HALIDE_BEGIN

namespace stats {

// -----------------------------------------------------------------------------
// Variance
// -----------------------------------------------------------------------------

/// @brief Compute variance over specified axes
/// @param f Input Func
/// @param in_shape Input shape
/// @param axes Axes to reduce over
/// @param keepdims If true, reduced axes have size 1
/// @param ddof Delta degrees of freedom (0 for population variance, 1 for sample variance)
/// @param name Function name
/// @return Variance Func
///
/// Formula: var = E[(X - mean)^2] = E[X^2] - E[X]^2
/// With ddof correction: var = sum((X - mean)^2) / (N - ddof)
inline Halide::Func var(
    Halide::Func f,
    const shape_t& in_shape,
    const std::vector<int>& axes,
    bool keepdims = false,
    int ddof = 0,
    std::string const& name = "var")
{
    // Normalize axes
    std::vector<int> norm_axes;
    for (int ax : axes) {
        norm_axes.push_back(normalized_axis(ax, in_shape.rank));
    }
    std::sort(norm_axes.begin(), norm_axes.end());

    if (norm_axes.empty()) {
        for (int i = 0; i < in_shape.rank; ++i) {
            norm_axes.push_back(i);
        }
    }

    // Calculate count of elements being reduced
    int count = 1;
    for (int ax : norm_axes) {
        count *= in_shape.extents[ax];
    }

    // Build reduction domain
    std::vector<Halide::Range> rdom_ranges;
    for (int ax : norm_axes) {
        rdom_ranges.push_back({0, in_shape.extents[ax]});
    }
    Halide::RDom rdom(rdom_ranges);

    // Build output shape
    shape_t out_shape = infer_reduce(in_shape, norm_axes, keepdims);

    // Create output vars
    std::vector<Halide::Var> out_vars;
    for (int i = 0; i < out_shape.rank; ++i) {
        out_vars.push_back(Halide::Var());
    }

    // Build input arguments
    std::vector<Halide::Expr> in_args;
    int out_idx = 0;
    int rdom_idx = 0;

    for (int i = in_shape.rank - 1; i >= 0; --i) {
        bool is_reduced = std::find(norm_axes.begin(), norm_axes.end(), i) != norm_axes.end();
        if (is_reduced) {
            in_args.push_back(rdom[rdom_idx]);
            rdom_idx++;
            if (keepdims) out_idx++;
        } else {
            in_args.push_back(out_vars[out_idx]);
            out_idx++;
        }
    }

    // Compute E[X^2] and E[X]
    // Using the formula: var = E[X^2] - E[X]^2
    Halide::Type t = f.types()[0];

    // Sum of X
    Halide::Func sum_x(name + "_sum_x");
    sum_x(out_vars) = Halide::cast(t, 0);
    sum_x(out_vars) += f(in_args);

    // Sum of X^2
    Halide::Func sum_x2(name + "_sum_x2");
    sum_x2(out_vars) = Halide::cast(t, 0);
    sum_x2(out_vars) += f(in_args) * f(in_args);

    // Variance = E[X^2] - E[X]^2 = (sum_x2 / n) - (sum_x / n)^2
    // With ddof: var = (sum_x2 - sum_x^2/n) / (n - ddof)
    Halide::Func ret(name);
    Halide::Expr n = Halide::cast(t, count);
    Halide::Expr n_ddof = Halide::cast(t, count - ddof);

    // Using the numerically stable formula:
    // var = (sum_x2 - sum_x^2 / n) / (n - ddof)
    ret(out_vars) = (sum_x2(out_vars) - (sum_x(out_vars) * sum_x(out_vars)) / n) / n_ddof;

    return ret;
}

/// @brief Compute variance of all elements
/// @param f Input Func
/// @param in_shape Input shape
/// @param ddof Delta degrees of freedom
/// @param name Function name
/// @return 1D Func with single element
///
/// Note: Use this overload explicitly for full reduction with ddof.
/// For axis-specific reduction, use var(f, shape, axis, keepdims, ddof, name).
inline Halide::Func var_full(
    Halide::Func f,
    const shape_t& in_shape,
    int ddof = 0,
    std::string const& name = "var")
{
    std::vector<int> all_axes;
    for (int i = 0; i < in_shape.rank; ++i) all_axes.push_back(i);

    Halide::Func reduced = var(f, in_shape, all_axes, true, ddof, name + "_inner");

    Halide::Func ret(name);
    Halide::Var x;
    std::vector<Halide::Expr> zeros(in_shape.rank, 0);
    ret(x) = reduced(zeros);

    return ret;
}

/// @brief Compute variance of all elements (population variance, ddof=0)
/// @param f Input Func
/// @param in_shape Input shape
/// @param name Function name
/// @return 1D Func with single element
inline Halide::Func var(
    Halide::Func f,
    const shape_t& in_shape,
    std::string const& name = "var")
{
    return var_full(f, in_shape, 0, name);
}

/// @brief Compute variance along a single axis
/// @param f Input Func
/// @param in_shape Input shape
/// @param axis Axis to reduce
/// @param keepdims If true, keep the reduced dimension
/// @param ddof Delta degrees of freedom
/// @param name Function name
/// @return Variance Func
inline Halide::Func var(
    Halide::Func f,
    const shape_t& in_shape,
    int axis,
    bool keepdims = false,
    int ddof = 0,
    std::string const& name = "var")
{
    std::vector<int> axes = {axis};
    return var(f, in_shape, axes, keepdims, ddof, name);
}

// -----------------------------------------------------------------------------
// Standard Deviation
// -----------------------------------------------------------------------------

/// @brief Compute standard deviation over specified axes
/// @param f Input Func
/// @param in_shape Input shape
/// @param axes Axes to reduce over
/// @param keepdims If true, reduced axes have size 1
/// @param ddof Delta degrees of freedom
/// @param name Function name
/// @return Standard deviation Func
inline Halide::Func std(
    Halide::Func f,
    const shape_t& in_shape,
    const std::vector<int>& axes,
    bool keepdims = false,
    int ddof = 0,
    std::string const& name = "std")
{
    Halide::Func variance = var(f, in_shape, axes, keepdims, ddof, name + "_var");

    shape_t out_shape = infer_reduce(in_shape, axes, keepdims);

    std::vector<Halide::Var> out_vars;
    for (int i = 0; i < out_shape.rank; ++i) {
        out_vars.push_back(Halide::Var());
    }

    Halide::Func ret(name);
    ret(out_vars) = Halide::sqrt(variance(out_vars));

    return ret;
}

/// @brief Compute standard deviation of all elements
/// @param f Input Func
/// @param in_shape Input shape
/// @param ddof Delta degrees of freedom
/// @param name Function name
/// @return 1D Func with single element
///
/// Note: Use this overload explicitly for full reduction with ddof.
inline Halide::Func std_full(
    Halide::Func f,
    const shape_t& in_shape,
    int ddof = 0,
    std::string const& name = "std")
{
    Halide::Func variance = var_full(f, in_shape, ddof, name + "_var");

    Halide::Func ret(name);
    Halide::Var x;
    ret(x) = Halide::sqrt(variance(x));

    return ret;
}

/// @brief Compute standard deviation of all elements (population std, ddof=0)
/// @param f Input Func
/// @param in_shape Input shape
/// @param name Function name
/// @return 1D Func with single element
inline Halide::Func std(
    Halide::Func f,
    const shape_t& in_shape,
    std::string const& name = "std")
{
    return std_full(f, in_shape, 0, name);
}

/// @brief Compute standard deviation along a single axis
/// @param f Input Func
/// @param in_shape Input shape
/// @param axis Axis to reduce
/// @param keepdims If true, keep the reduced dimension
/// @param ddof Delta degrees of freedom
/// @param name Function name
/// @return Standard deviation Func
inline Halide::Func std(
    Halide::Func f,
    const shape_t& in_shape,
    int axis,
    bool keepdims = false,
    int ddof = 0,
    std::string const& name = "std")
{
    std::vector<int> axes = {axis};
    return std(f, in_shape, axes, keepdims, ddof, name);
}

// -----------------------------------------------------------------------------
// Sum of Squared Differences (helper for other stats)
// -----------------------------------------------------------------------------

/// @brief Compute sum of squared differences from mean
/// @param f Input Func
/// @param in_shape Input shape
/// @param axes Axes to reduce over
/// @param keepdims If true, reduced axes have size 1
/// @param name Function name
/// @return SSD Func
inline Halide::Func sum_squared_diff(
    Halide::Func f,
    const shape_t& in_shape,
    const std::vector<int>& axes,
    bool keepdims = false,
    std::string const& name = "ssd")
{
    std::vector<int> norm_axes;
    for (int ax : axes) {
        norm_axes.push_back(normalized_axis(ax, in_shape.rank));
    }
    std::sort(norm_axes.begin(), norm_axes.end());

    if (norm_axes.empty()) {
        for (int i = 0; i < in_shape.rank; ++i) {
            norm_axes.push_back(i);
        }
    }

    int count = 1;
    for (int ax : norm_axes) {
        count *= in_shape.extents[ax];
    }

    std::vector<Halide::Range> rdom_ranges;
    for (int ax : norm_axes) {
        rdom_ranges.push_back({0, in_shape.extents[ax]});
    }
    Halide::RDom rdom(rdom_ranges);

    shape_t out_shape = infer_reduce(in_shape, norm_axes, keepdims);

    std::vector<Halide::Var> out_vars;
    for (int i = 0; i < out_shape.rank; ++i) {
        out_vars.push_back(Halide::Var());
    }

    std::vector<Halide::Expr> in_args;
    int out_idx = 0;
    int rdom_idx = 0;

    for (int i = in_shape.rank - 1; i >= 0; --i) {
        bool is_reduced = std::find(norm_axes.begin(), norm_axes.end(), i) != norm_axes.end();
        if (is_reduced) {
            in_args.push_back(rdom[rdom_idx]);
            rdom_idx++;
            if (keepdims) out_idx++;
        } else {
            in_args.push_back(out_vars[out_idx]);
            out_idx++;
        }
    }

    Halide::Type t = f.types()[0];

    // Sum of X for mean
    Halide::Func sum_x(name + "_sum_x");
    sum_x(out_vars) = Halide::cast(t, 0);
    sum_x(out_vars) += f(in_args);

    // Sum of X^2
    Halide::Func sum_x2(name + "_sum_x2");
    sum_x2(out_vars) = Halide::cast(t, 0);
    sum_x2(out_vars) += f(in_args) * f(in_args);

    // SSD = sum_x2 - sum_x^2 / n
    Halide::Func ret(name);
    Halide::Expr n = Halide::cast(t, count);
    ret(out_vars) = sum_x2(out_vars) - (sum_x(out_vars) * sum_x(out_vars)) / n;

    return ret;
}

} // namespace stats

NS_NUM_HALIDE_END
