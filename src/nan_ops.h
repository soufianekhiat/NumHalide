/// @file nan_ops.h
/// @brief NaN-safe reduction operations
///
/// Provides: nansum, nanmean, nanmin, nanmax, nanstd, nanvar, nanprod

#pragma once

#include "common.h"
#include "numhalide.h"
#include "shape.h"
#include "reduce.h"
#include "sort.h"
#include <algorithm>
#include <limits>
#include <climits>

NS_NUM_HALIDE_BEGIN

// -----------------------------------------------------------------------------
// NaN-safe Reductions
// -----------------------------------------------------------------------------

/// @brief Sum of non-NaN elements over specified axes
/// @param f Input Func
/// @param in_shape Input shape
/// @param axes Axes to reduce over
/// @param keepdims If true, reduced axes have size 1
/// @param name Function name
/// @return Reduced Func (NaN values treated as 0)
inline Halide::Func nansum(
    Halide::Func f,
    const shape_t& in_shape,
    const std::vector<int>& axes,
    bool keepdims = false,
    std::string const& name = "nansum")
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

    Halide::Type type = f.types()[0];
    Halide::Func ret(name);
    ret(out_vars) = Halide::cast(type, 0);
    Halide::Expr val = f(in_args);
    ret(out_vars) += Halide::select(Halide::is_nan(val), Halide::cast(type, 0), val);

    return ret;
}

/// @brief Product of non-NaN elements over specified axes
/// @param f Input Func
/// @param in_shape Input shape
/// @param axes Axes to reduce over
/// @param keepdims If true, reduced axes have size 1
/// @param name Function name
/// @return Reduced Func (NaN values treated as 1)
inline Halide::Func nanprod(
    Halide::Func f,
    const shape_t& in_shape,
    const std::vector<int>& axes,
    bool keepdims = false,
    std::string const& name = "nanprod")
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

    Halide::Type type = f.types()[0];
    Halide::Func ret(name);
    ret(out_vars) = Halide::cast(type, 1);
    Halide::Expr val = f(in_args);
    ret(out_vars) *= Halide::select(Halide::is_nan(val), Halide::cast(type, 1), val);

    return ret;
}

/// @brief Minimum of non-NaN elements over specified axes
/// @param f Input Func
/// @param in_shape Input shape
/// @param axes Axes to reduce over
/// @param keepdims If true, reduced axes have size 1
/// @param name Function name
/// @return Reduced Func (NaN values ignored; init = +inf for floats)
inline Halide::Func nanmin(
    Halide::Func f,
    const shape_t& in_shape,
    const std::vector<int>& axes,
    bool keepdims = false,
    std::string const& name = "nanmin")
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

    Halide::Type type = f.types()[0];
    Halide::Expr init_min = Halide::Internal::make_const(type,
        type.is_float() ? std::numeric_limits<float>::infinity() : (double)INT32_MAX);

    Halide::Func ret(name);
    ret(out_vars) = init_min;
    Halide::Expr val = f(in_args);
    // For NaN values, substitute +inf so they are never chosen as the minimum
    ret(out_vars) = Halide::min(ret(out_vars),
        Halide::select(Halide::is_nan(val), init_min, val));

    return ret;
}

/// @brief Maximum of non-NaN elements over specified axes
/// @param f Input Func
/// @param in_shape Input shape
/// @param axes Axes to reduce over
/// @param keepdims If true, reduced axes have size 1
/// @param name Function name
/// @return Reduced Func (NaN values ignored; init = -inf for floats)
inline Halide::Func nanmax(
    Halide::Func f,
    const shape_t& in_shape,
    const std::vector<int>& axes,
    bool keepdims = false,
    std::string const& name = "nanmax")
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

    Halide::Type type = f.types()[0];
    Halide::Expr init_max = Halide::Internal::make_const(type,
        type.is_float() ? -std::numeric_limits<float>::infinity() : (double)INT32_MIN);

    Halide::Func ret(name);
    ret(out_vars) = init_max;
    Halide::Expr val = f(in_args);
    // For NaN values, substitute -inf so they are never chosen as the maximum
    ret(out_vars) = Halide::max(ret(out_vars),
        Halide::select(Halide::is_nan(val), init_max, val));

    return ret;
}

/// @brief Mean of non-NaN elements over specified axes
/// @param f Input Func
/// @param in_shape Input shape
/// @param axes Axes to reduce over
/// @param keepdims If true, reduced axes have size 1
/// @param name Function name
/// @return Reduced Func: sum(non-NaN) / count(non-NaN)
inline Halide::Func nanmean(
    Halide::Func f,
    const shape_t& in_shape,
    const std::vector<int>& axes,
    bool keepdims = false,
    std::string const& name = "nanmean")
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

    // Build reduction domain
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

    Halide::Type type = f.types()[0];
    Halide::Expr val = f(in_args);
    Halide::Expr is_nan_val = Halide::is_nan(val);

    // Sum of non-NaN values
    Halide::Func nan_sum(name + "_sum");
    nan_sum(out_vars) = Halide::cast(type, 0);
    nan_sum(out_vars) += Halide::select(is_nan_val, Halide::cast(type, 0), val);

    // Count of non-NaN values
    Halide::Func nan_count(name + "_count");
    nan_count(out_vars) = Halide::cast(type, 0);
    nan_count(out_vars) += Halide::select(is_nan_val, Halide::cast(type, 0), Halide::cast(type, 1));

    // Mean = sum / count
    Halide::Func ret(name);
    ret(out_vars) = nan_sum(out_vars) / nan_count(out_vars);

    return ret;
}

/// @brief Variance of non-NaN elements over specified axes
/// @param f Input Func
/// @param in_shape Input shape
/// @param axes Axes to reduce over
/// @param keepdims If true, reduced axes have size 1
/// @param ddof Delta degrees of freedom (0 = population, 1 = sample)
/// @param name Function name
/// @return Variance Func: sum((x-mean)^2 for non-NaN x) / (count - ddof)
inline Halide::Func nanvar(
    Halide::Func f,
    const shape_t& in_shape,
    const std::vector<int>& axes,
    bool keepdims = false,
    int ddof = 0,
    std::string const& name = "nanvar")
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

    // Compute nanmean with keepdims=true for broadcasting back against f
    Halide::Func mean_f = nanmean(f, in_shape, axes, true, name + "_mean");

    // Build reduction domain
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

    // Build input args (pointing into f with rdom for reduced dims)
    std::vector<Halide::Expr> in_args;
    // Also build mean_args: always keepdims=true shape, so reduced dims are at size-1 slots
    std::vector<Halide::Expr> mean_args;

    {
        int out_idx = 0;
        int rdom_idx = 0;

        for (int i = in_shape.rank - 1; i >= 0; --i) {
            bool is_reduced = std::find(norm_axes.begin(), norm_axes.end(), i) != norm_axes.end();
            if (is_reduced) {
                in_args.push_back(rdom[rdom_idx]);
                mean_args.push_back(0); // keepdims size-1 dim: always index 0
                rdom_idx++;
                if (keepdims) out_idx++;
            } else {
                in_args.push_back(out_vars[out_idx]);
                mean_args.push_back(out_vars[out_idx]);
                out_idx++;
            }
        }
    }

    Halide::Type type = f.types()[0];
    Halide::Expr val = f(in_args);
    Halide::Expr is_nan_val = Halide::is_nan(val);
    Halide::Expr mean_val = mean_f(mean_args);

    // Count of non-NaN values
    Halide::Func nan_count(name + "_count");
    nan_count(out_vars) = Halide::cast(type, 0);
    nan_count(out_vars) += Halide::select(is_nan_val, Halide::cast(type, 0), Halide::cast(type, 1));

    // Sum of squared deviations from mean (only for non-NaN)
    Halide::Func ssd(name + "_ssd");
    ssd(out_vars) = Halide::cast(type, 0);
    Halide::Expr dev = val - mean_val;
    ssd(out_vars) += Halide::select(is_nan_val, Halide::cast(type, 0), dev * dev);

    // Variance = ssd / (count - ddof)
    Halide::Func ret(name);
    ret(out_vars) = ssd(out_vars) / (nan_count(out_vars) - Halide::cast(type, ddof));

    return ret;
}

/// @brief Standard deviation of non-NaN elements over specified axes
/// @param f Input Func
/// @param in_shape Input shape
/// @param axes Axes to reduce over
/// @param keepdims If true, reduced axes have size 1
/// @param ddof Delta degrees of freedom (0 = population, 1 = sample)
/// @param name Function name
/// @return Standard deviation Func: sqrt(nanvar(...))
inline Halide::Func nanstd(
    Halide::Func f,
    const shape_t& in_shape,
    const std::vector<int>& axes,
    bool keepdims = false,
    int ddof = 0,
    std::string const& name = "nanstd")
{
    Halide::Func variance = nanvar(f, in_shape, axes, keepdims, ddof, name + "_var");

    // Normalize axes to compute output shape
    std::vector<int> norm_axes;
    for (int ax : axes) {
        norm_axes.push_back(normalized_axis(ax, in_shape.rank));
    }
    if (norm_axes.empty()) {
        for (int i = 0; i < in_shape.rank; ++i) {
            norm_axes.push_back(i);
        }
    }
    shape_t out_shape = infer_reduce(in_shape, norm_axes, keepdims);

    std::vector<Halide::Var> out_vars;
    for (int i = 0; i < out_shape.rank; ++i) {
        out_vars.push_back(Halide::Var());
    }

    Halide::Func ret(name);
    ret(out_vars) = Halide::sqrt(variance(out_vars));

    return ret;
}

// -----------------------------------------------------------------------------
// NaN-safe Cumulative Operations
// -----------------------------------------------------------------------------

/// @brief Cumulative sum ignoring NaN (treated as 0)
inline
Halide::Func nancumsum(Halide::Func f, const shape_t& shape, int axis = 0,
    std::string const& name = "nancumsum")
{
    nh_require(nullptr, shape.rank == 1, "nancumsum: 1D only");
    (void)axis;
    int n = shape.extents[0];
    Halide::Var x;

    Halide::Func clean(name + "_clean");
    clean(x) = Halide::select(Halide::is_nan(f(x)), 0.0f, f(x));
    clean.compute_root();

    Halide::Func ret(name);
    ret(x) = clean(x);
    Halide::RDom r(1, n - 1, "r_ncs");
    ret(r) = ret(r - 1) + clean(r);
    ret.compute_root();
    return ret;
}

/// @brief Cumulative product ignoring NaN (treated as 1)
inline
Halide::Func nancumprod(Halide::Func f, const shape_t& shape, int axis = 0,
    std::string const& name = "nancumprod")
{
    nh_require(nullptr, shape.rank == 1, "nancumprod: 1D only");
    (void)axis;
    int n = shape.extents[0];
    Halide::Var x;

    Halide::Func clean(name + "_clean");
    clean(x) = Halide::select(Halide::is_nan(f(x)), 1.0f, f(x));
    clean.compute_root();

    Halide::Func ret(name);
    ret(x) = clean(x);
    Halide::RDom r(1, n - 1, "r_ncp");
    ret(r) = ret(r - 1) * clean(r);
    ret.compute_root();
    return ret;
}

// -----------------------------------------------------------------------------
// NaN-aware Median
// -----------------------------------------------------------------------------

/// @brief Median of a 1D array, ignoring NaN values
/// @param f  Input Func (1D)
/// @param n  Number of elements (including NaN)
/// @return 0D (scalar) Func with median of non-NaN values
inline
Halide::Func nanmedian(Halide::Func f, int n,
    std::string const& name = "nanmedian")
{
    Halide::Var i("i");

    // Replace NaN with +inf so they sort to the end
    Halide::Func clean(name + "_clean");
    clean(i) = Halide::select(Halide::is_nan(f(i)),
        std::numeric_limits<float>::infinity(), f(i));
    clean.compute_root();

    // Sort ascending (NaN/inf go to end)
    Halide::Func sorted = sort_1d(clean, n, true, name + "_sorted");
    sorted.compute_root();

    // Count non-NaN elements
    Halide::Func cnt(name + "_cnt");
    Halide::RDom rc(0, n, "rc_nm");
    cnt() = Halide::cast<int32_t>(0);
    cnt() += Halide::cast<int32_t>(Halide::select(Halide::is_nan(f(rc)), 0, 1));
    cnt.compute_root();

    // lo = sorted[(k-1)/2], hi = sorted[k/2]; median = (lo + hi) / 2
    // (for odd k: (k-1)/2 == k/2, so lo==hi and median is just that element)
    Halide::Expr k = cnt();
    Halide::Func lo_e(name + "_lo");
    Halide::RDom rlo(0, n, "rlo_nm");
    lo_e() = 0.0f;
    lo_e() += Halide::select(rlo == (k - 1) / 2, sorted(rlo), 0.0f);
    lo_e.compute_root();

    Halide::Func hi_e(name + "_hi");
    Halide::RDom rhi(0, n, "rhi_nm");
    hi_e() = 0.0f;
    hi_e() += Halide::select(rhi == k / 2, sorted(rhi), 0.0f);
    hi_e.compute_root();

    Halide::Func ret(name);
    ret() = (lo_e() + hi_e()) * 0.5f;
    return ret;
}

NS_NUM_HALIDE_END
