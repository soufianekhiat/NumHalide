/// @file reduce.h
/// @brief Reduction operations for Halide::Func objects
///
/// Provides: sum, mean, min, max, prod, argmin, argmax with axis control and keepdims

#pragma once

#include "common.h"
#include "numhalide.h"
#include "shape.h"
#include <algorithm>

NS_NUM_HALIDE_BEGIN

// -----------------------------------------------------------------------------
// Reduction Operations
// -----------------------------------------------------------------------------

/// @brief Reduce by sum over specified axes
/// @param f Input Func
/// @param in_shape Input shape
/// @param axes Axes to reduce over
/// @param keepdims If true, reduced axes have size 1; if false, they are removed
/// @param name Function name
/// @return Reduced Func
inline Halide::Func reduce_sum(
    Halide::Func f,
    const shape_t& in_shape,
    const std::vector<int>& axes,
    bool keepdims = false,
    std::string const& name = "reduce_sum")
{
    // Normalize axes
    std::vector<int> norm_axes;
    for (int ax : axes) {
        norm_axes.push_back(normalized_axis(ax, in_shape.rank));
    }
    std::sort(norm_axes.begin(), norm_axes.end());

    // Handle full reduction (all axes)
    if (norm_axes.empty()) {
        // Reduce over all axes
        for (int i = 0; i < in_shape.rank; ++i) {
            norm_axes.push_back(i);
        }
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
    // Map output vars and rdom vars to input dimensions.
    // Each reduced axis at position j in norm_axes maps to rdom[j] (same sorted order).
    std::vector<Halide::Expr> in_args;
    int out_idx = 0;

    for (int i = in_shape.rank - 1; i >= 0; --i) {
        int rdom_pos = -1;
        for (int j = 0; j < (int)norm_axes.size(); ++j) {
            if (norm_axes[j] == i) { rdom_pos = j; break; }
        }
        if (rdom_pos >= 0) {
            in_args.push_back(rdom[rdom_pos]);
            if (keepdims) out_idx++;
        } else {
            in_args.push_back(out_vars[out_idx]);
            out_idx++;
        }
    }

    // Define the reduction
    Halide::Func raw(out_shape.rank == 0 ? name + "_raw" : name);
    raw(out_vars) = Halide::cast(f.types()[0], 0);
    raw(out_vars) += f(in_args);

    if (out_shape.rank == 0) {
        // Full reduction: wrap scalar in 1D for consistent buffer handling
        Halide::Func ret(name);
        Halide::Var _x;
        ret(_x) = raw();
        return ret;
    }
    return raw;
}

/// @brief Reduce by product over specified axes
/// @param f Input Func
/// @param in_shape Input shape
/// @param axes Axes to reduce over
/// @param keepdims If true, reduced axes have size 1
/// @param name Function name
/// @return Reduced Func
inline Halide::Func reduce_prod(
    Halide::Func f,
    const shape_t& in_shape,
    const std::vector<int>& axes,
    bool keepdims = false,
    std::string const& name = "reduce_prod")
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

    for (int i = in_shape.rank - 1; i >= 0; --i) {
        int rdom_pos = -1;
        for (int j = 0; j < (int)norm_axes.size(); ++j) {
            if (norm_axes[j] == i) { rdom_pos = j; break; }
        }
        if (rdom_pos >= 0) {
            in_args.push_back(rdom[rdom_pos]);
            if (keepdims) out_idx++;
        } else {
            in_args.push_back(out_vars[out_idx]);
            out_idx++;
        }
    }

    Halide::Func raw(out_shape.rank == 0 ? name + "_raw" : name);
    raw(out_vars) = Halide::cast(f.types()[0], 1);
    raw(out_vars) *= f(in_args);

    if (out_shape.rank == 0) {
        Halide::Func ret(name); Halide::Var _x;
        ret(_x) = raw(); return ret;
    }
    return raw;
}

/// @brief Reduce by minimum over specified axes
/// @param f Input Func
/// @param in_shape Input shape
/// @param axes Axes to reduce over
/// @param keepdims If true, reduced axes have size 1
/// @param name Function name
/// @return Reduced Func
inline Halide::Func reduce_min(
    Halide::Func f,
    const shape_t& in_shape,
    const std::vector<int>& axes,
    bool keepdims = false,
    std::string const& name = "reduce_min")
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

    for (int i = in_shape.rank - 1; i >= 0; --i) {
        int rdom_pos = -1;
        for (int j = 0; j < (int)norm_axes.size(); ++j) {
            if (norm_axes[j] == i) { rdom_pos = j; break; }
        }
        if (rdom_pos >= 0) {
            in_args.push_back(rdom[rdom_pos]);
            if (keepdims) out_idx++;
        } else {
            in_args.push_back(out_vars[out_idx]);
            out_idx++;
        }
    }

    Halide::Func raw(out_shape.rank == 0 ? name + "_raw" : name);
    Halide::Type t = f.types()[0];
    raw(out_vars) = t.max();
    raw(out_vars) = Halide::min(raw(out_vars), f(in_args));

    if (out_shape.rank == 0) {
        Halide::Func ret(name); Halide::Var _x;
        ret(_x) = raw(); return ret;
    }
    return raw;
}

/// @brief Reduce by maximum over specified axes
/// @param f Input Func
/// @param in_shape Input shape
/// @param axes Axes to reduce over
/// @param keepdims If true, reduced axes have size 1
/// @param name Function name
/// @return Reduced Func
inline Halide::Func reduce_max(
    Halide::Func f,
    const shape_t& in_shape,
    const std::vector<int>& axes,
    bool keepdims = false,
    std::string const& name = "reduce_max")
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

    for (int i = in_shape.rank - 1; i >= 0; --i) {
        int rdom_pos = -1;
        for (int j = 0; j < (int)norm_axes.size(); ++j) {
            if (norm_axes[j] == i) { rdom_pos = j; break; }
        }
        if (rdom_pos >= 0) {
            in_args.push_back(rdom[rdom_pos]);
            if (keepdims) out_idx++;
        } else {
            in_args.push_back(out_vars[out_idx]);
            out_idx++;
        }
    }

    Halide::Func raw(out_shape.rank == 0 ? name + "_raw" : name);
    Halide::Type t = f.types()[0];
    raw(out_vars) = t.min();
    raw(out_vars) = Halide::max(raw(out_vars), f(in_args));

    if (out_shape.rank == 0) {
        Halide::Func ret(name); Halide::Var _x;
        ret(_x) = raw(); return ret;
    }
    return raw;
}

/// @brief Reduce by mean over specified axes
/// @param f Input Func
/// @param in_shape Input shape
/// @param axes Axes to reduce over
/// @param keepdims If true, reduced axes have size 1
/// @param name Function name
/// @return Reduced Func
inline Halide::Func reduce_mean(
    Halide::Func f,
    const shape_t& in_shape,
    const std::vector<int>& axes,
    bool keepdims = false,
    std::string const& name = "reduce_mean")
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

    // Calculate count of elements being reduced
    int count = 1;
    for (int ax : norm_axes) {
        count *= in_shape.extents[ax];
    }

    // Get sum
    Halide::Func sum_f = reduce_sum(f, in_shape, axes, keepdims, name + "_sum");

    // Divide by count
    shape_t out_shape = infer_reduce(in_shape, norm_axes, keepdims);
    Halide::Type t = f.types()[0];

    if (out_shape.rank == 0) {
        // sum_f is now 1D (scalar wrapped in 1D); return 1D mean
        Halide::Func ret(name);
        Halide::Var _x;
        ret(_x) = sum_f(_x) / Halide::cast(t, count);
        return ret;
    }

    std::vector<Halide::Var> out_vars;
    for (int i = 0; i < out_shape.rank; ++i) {
        out_vars.push_back(Halide::Var());
    }

    Halide::Func ret(name);
    ret(out_vars) = sum_f(out_vars) / Halide::cast(t, count);

    return ret;
}

/// @brief Find index of minimum value along an axis
/// @param f Input Func
/// @param in_shape Input shape
/// @param axis Axis to reduce over
/// @param keepdims If true, reduced axis has size 1
/// @param name Function name
/// @return Func containing indices of minimum values
inline Halide::Func argmin(
    Halide::Func f,
    const shape_t& in_shape,
    int axis,
    bool keepdims = false,
    std::string const& name = "argmin")
{
    int norm_axis = normalized_axis(axis, in_shape.rank);
    std::vector<int> norm_axes = {norm_axis};

    Halide::RDom rdom(0, in_shape.extents[norm_axis]);

    shape_t out_shape = infer_reduce(in_shape, norm_axes, keepdims);

    std::vector<Halide::Var> out_vars;
    for (int i = 0; i < out_shape.rank; ++i) {
        out_vars.push_back(Halide::Var());
    }

    // Build input arguments - one set with rdom, one with 0 for initialization
    std::vector<Halide::Expr> in_args_r;
    std::vector<Halide::Expr> init_args;
    int out_idx = 0;

    for (int i = in_shape.rank - 1; i >= 0; --i) {
        if (i == norm_axis) {
            in_args_r.push_back(rdom.x);
            init_args.push_back(0);
            if (keepdims) out_idx++;
        } else {
            in_args_r.push_back(out_vars[out_idx]);
            init_args.push_back(out_vars[out_idx]);
            out_idx++;
        }
    }

    // Use tuple reduction for argmin
    Halide::Func ret(name + "_tuple");

    ret(out_vars) = Halide::Tuple(f(init_args), 0);

    Halide::Expr curr_val = f(in_args_r);
    Halide::Expr curr_min = ret(out_vars)[0];
    Halide::Expr curr_idx = ret(out_vars)[1];

    ret(out_vars) = Halide::Tuple(
        Halide::select(curr_val < curr_min, curr_val, curr_min),
        Halide::select(curr_val < curr_min, rdom.x, curr_idx)
    );

    // Extract just the index
    Halide::Func ret_idx(name);
    ret_idx(out_vars) = ret(out_vars)[1];

    return ret_idx;
}

/// @brief Find index of maximum value along an axis
/// @param f Input Func
/// @param in_shape Input shape
/// @param axis Axis to reduce over
/// @param keepdims If true, reduced axis has size 1
/// @param name Function name
/// @return Func containing indices of maximum values
inline Halide::Func argmax(
    Halide::Func f,
    const shape_t& in_shape,
    int axis,
    bool keepdims = false,
    std::string const& name = "argmax")
{
    int norm_axis = normalized_axis(axis, in_shape.rank);
    std::vector<int> norm_axes = {norm_axis};

    Halide::RDom rdom(0, in_shape.extents[norm_axis]);

    shape_t out_shape = infer_reduce(in_shape, norm_axes, keepdims);

    std::vector<Halide::Var> out_vars;
    for (int i = 0; i < out_shape.rank; ++i) {
        out_vars.push_back(Halide::Var());
    }

    std::vector<Halide::Expr> in_args_r;
    std::vector<Halide::Expr> init_args;
    int out_idx = 0;

    for (int i = in_shape.rank - 1; i >= 0; --i) {
        if (i == norm_axis) {
            in_args_r.push_back(rdom.x);
            init_args.push_back(0);
            if (keepdims) out_idx++;
        } else {
            in_args_r.push_back(out_vars[out_idx]);
            init_args.push_back(out_vars[out_idx]);
            out_idx++;
        }
    }

    Halide::Func ret(name + "_tuple");

    ret(out_vars) = Halide::Tuple(f(init_args), 0);

    Halide::Expr curr_val = f(in_args_r);
    Halide::Expr curr_max = ret(out_vars)[0];
    Halide::Expr curr_idx = ret(out_vars)[1];

    ret(out_vars) = Halide::Tuple(
        Halide::select(curr_val > curr_max, curr_val, curr_max),
        Halide::select(curr_val > curr_max, rdom.x, curr_idx)
    );

    Halide::Func ret_idx(name);
    ret_idx(out_vars) = ret(out_vars)[1];

    return ret_idx;
}

// -----------------------------------------------------------------------------
// Convenience overloads for full reduction (all axes)
// -----------------------------------------------------------------------------

/// @brief Sum all elements (returns 1D func with single element)
inline Halide::Func reduce_sum(Halide::Func f, const shape_t& in_shape, std::string const& name = "reduce_sum") {
    // Reduce to single axis with size 1 for easier buffer handling
    std::vector<int> all_axes;
    for (int i = 0; i < in_shape.rank; ++i) all_axes.push_back(i);

    // Keep all but one axis to ensure we have at least 1D output
    Halide::Func reduced = reduce_sum(f, in_shape, all_axes, true, name + "_inner");

    // Flatten to 1D
    Halide::Func ret(name);
    Halide::Var x;
    std::vector<Halide::Expr> zeros(in_shape.rank, 0);
    ret(x) = reduced(zeros);

    return ret;
}

/// @brief Product of all elements (returns 1D func with single element)
inline Halide::Func reduce_prod(Halide::Func f, const shape_t& in_shape, std::string const& name = "reduce_prod") {
    std::vector<int> all_axes;
    for (int i = 0; i < in_shape.rank; ++i) all_axes.push_back(i);

    Halide::Func reduced = reduce_prod(f, in_shape, all_axes, true, name + "_inner");

    Halide::Func ret(name);
    Halide::Var x;
    std::vector<Halide::Expr> zeros(in_shape.rank, 0);
    ret(x) = reduced(zeros);

    return ret;
}

/// @brief Minimum of all elements (returns 1D func with single element)
inline Halide::Func reduce_min(Halide::Func f, const shape_t& in_shape, std::string const& name = "reduce_min") {
    std::vector<int> all_axes;
    for (int i = 0; i < in_shape.rank; ++i) all_axes.push_back(i);

    Halide::Func reduced = reduce_min(f, in_shape, all_axes, true, name + "_inner");

    Halide::Func ret(name);
    Halide::Var x;
    std::vector<Halide::Expr> zeros(in_shape.rank, 0);
    ret(x) = reduced(zeros);

    return ret;
}

/// @brief Maximum of all elements (returns 1D func with single element)
inline Halide::Func reduce_max(Halide::Func f, const shape_t& in_shape, std::string const& name = "reduce_max") {
    std::vector<int> all_axes;
    for (int i = 0; i < in_shape.rank; ++i) all_axes.push_back(i);

    Halide::Func reduced = reduce_max(f, in_shape, all_axes, true, name + "_inner");

    Halide::Func ret(name);
    Halide::Var x;
    std::vector<Halide::Expr> zeros(in_shape.rank, 0);
    ret(x) = reduced(zeros);

    return ret;
}

/// @brief Mean of all elements (returns 1D func with single element)
inline Halide::Func reduce_mean(Halide::Func f, const shape_t& in_shape, std::string const& name = "reduce_mean") {
    std::vector<int> all_axes;
    for (int i = 0; i < in_shape.rank; ++i) all_axes.push_back(i);

    Halide::Func reduced = reduce_mean(f, in_shape, all_axes, true, name + "_inner");

    Halide::Func ret(name);
    Halide::Var x;
    std::vector<Halide::Expr> zeros(in_shape.rank, 0);
    ret(x) = reduced(zeros);

    return ret;
}

// -----------------------------------------------------------------------------
// Boolean Reductions
// -----------------------------------------------------------------------------

/// @brief Check if any element is non-zero along specified axes
/// @param f Input Func
/// @param in_shape Input shape
/// @param axes Axes to reduce over
/// @param keepdims If true, reduced axes have size 1
/// @param name Function name
/// @return Func with 1 (true) if any element is non-zero, 0 (false) otherwise
inline Halide::Func reduce_any(
    Halide::Func f,
    const shape_t& in_shape,
    const std::vector<int>& axes,
    bool keepdims = false,
    std::string const& name = "reduce_any")
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

    // any = max(x != 0) - if any element is non-zero, max will be 1
    Halide::Func ret(name);
    ret(out_vars) = Halide::cast<int32_t>(0);
    ret(out_vars) = Halide::max(ret(out_vars), Halide::cast<int32_t>(f(in_args) != 0));

    return ret;
}

/// @brief Check if any element is non-zero (full reduction)
/// @param f Input Func
/// @param in_shape Input shape
/// @param name Function name
/// @return 1D Func with single element (1 if any true, 0 otherwise)
inline Halide::Func reduce_any(Halide::Func f, const shape_t& in_shape, std::string const& name = "reduce_any") {
    std::vector<int> all_axes;
    for (int i = 0; i < in_shape.rank; ++i) all_axes.push_back(i);

    Halide::Func reduced = reduce_any(f, in_shape, all_axes, true, name + "_inner");

    Halide::Func ret(name);
    Halide::Var x;
    std::vector<Halide::Expr> zeros(in_shape.rank, 0);
    ret(x) = reduced(zeros);

    return ret;
}

/// @brief Check if any element is non-zero along a single axis
/// @param f Input Func
/// @param in_shape Input shape
/// @param axis Axis to reduce
/// @param keepdims If true, keep the reduced dimension
/// @param name Function name
/// @return Func with boolean results
inline Halide::Func reduce_any(
    Halide::Func f,
    const shape_t& in_shape,
    int axis,
    bool keepdims = false,
    std::string const& name = "reduce_any")
{
    std::vector<int> axes = {axis};
    return reduce_any(f, in_shape, axes, keepdims, name);
}

/// @brief Check if all elements are non-zero along specified axes
/// @param f Input Func
/// @param in_shape Input shape
/// @param axes Axes to reduce over
/// @param keepdims If true, reduced axes have size 1
/// @param name Function name
/// @return Func with 1 (true) if all elements are non-zero, 0 (false) otherwise
inline Halide::Func reduce_all(
    Halide::Func f,
    const shape_t& in_shape,
    const std::vector<int>& axes,
    bool keepdims = false,
    std::string const& name = "reduce_all")
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

    // all = min(x != 0) - if any element is zero, min will be 0
    Halide::Func ret(name);
    ret(out_vars) = Halide::cast<int32_t>(1);
    ret(out_vars) = Halide::min(ret(out_vars), Halide::cast<int32_t>(f(in_args) != 0));

    return ret;
}

/// @brief Check if all elements are non-zero (full reduction)
/// @param f Input Func
/// @param in_shape Input shape
/// @param name Function name
/// @return 1D Func with single element (1 if all true, 0 otherwise)
inline Halide::Func reduce_all(Halide::Func f, const shape_t& in_shape, std::string const& name = "reduce_all") {
    std::vector<int> all_axes;
    for (int i = 0; i < in_shape.rank; ++i) all_axes.push_back(i);

    Halide::Func reduced = reduce_all(f, in_shape, all_axes, true, name + "_inner");

    Halide::Func ret(name);
    Halide::Var x;
    std::vector<Halide::Expr> zeros(in_shape.rank, 0);
    ret(x) = reduced(zeros);

    return ret;
}

/// @brief Check if all elements are non-zero along a single axis
/// @param f Input Func
/// @param in_shape Input shape
/// @param axis Axis to reduce
/// @param keepdims If true, keep the reduced dimension
/// @param name Function name
/// @return Func with boolean results
inline Halide::Func reduce_all(
    Halide::Func f,
    const shape_t& in_shape,
    int axis,
    bool keepdims = false,
    std::string const& name = "reduce_all")
{
    std::vector<int> axes = {axis};
    return reduce_all(f, in_shape, axes, keepdims, name);
}

/// @brief Count non-zero elements along specified axes
/// @param f Input Func
/// @param in_shape Input shape
/// @param axes Axes to reduce over
/// @param keepdims If true, reduced axes have size 1
/// @param name Function name
/// @return Func with count of non-zero elements
inline Halide::Func count_nonzero(
    Halide::Func f,
    const shape_t& in_shape,
    const std::vector<int>& axes,
    bool keepdims = false,
    std::string const& name = "count_nonzero")
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

    // Count by summing (x != 0)
    Halide::Func ret(name);
    ret(out_vars) = Halide::cast<int32_t>(0);
    ret(out_vars) += Halide::cast<int32_t>(f(in_args) != 0);

    return ret;
}

/// @brief Count non-zero elements (full reduction)
/// @param f Input Func
/// @param in_shape Input shape
/// @param name Function name
/// @return 1D Func with count
inline Halide::Func count_nonzero(Halide::Func f, const shape_t& in_shape, std::string const& name = "count_nonzero") {
    std::vector<int> all_axes;
    for (int i = 0; i < in_shape.rank; ++i) all_axes.push_back(i);

    Halide::Func reduced = count_nonzero(f, in_shape, all_axes, true, name + "_inner");

    Halide::Func ret(name);
    Halide::Var x;
    std::vector<Halide::Expr> zeros(in_shape.rank, 0);
    ret(x) = reduced(zeros);

    return ret;
}

/// @brief Count non-zero elements along a single axis
/// @param f Input Func
/// @param in_shape Input shape
/// @param axis Axis to reduce
/// @param keepdims If true, keep the reduced dimension
/// @param name Function name
/// @return Func with counts
inline Halide::Func count_nonzero(
    Halide::Func f,
    const shape_t& in_shape,
    int axis,
    bool keepdims = false,
    std::string const& name = "count_nonzero")
{
    std::vector<int> axes = {axis};
    return count_nonzero(f, in_shape, axes, keepdims, name);
}

NS_NUM_HALIDE_END
