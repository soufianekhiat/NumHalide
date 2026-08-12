/// @file statistics2.h
/// @brief Extended statistical operations
///
/// Provides: percentile, quantile, corrcoef, cov, histogram2d, bincount

#pragma once

#include "common.h"
#include "numhalide.h"
#include "shape.h"
#include "sort.h"
#include "reduce.h"
#include "stats.h"
#include <algorithm>

NS_NUM_HALIDE_BEGIN

namespace stats {

// -----------------------------------------------------------------------------
// Percentile / Quantile (1D, power-of-2 size via bitonic sort)
// -----------------------------------------------------------------------------

/// @brief Compute the q-th percentile of a 1D array via linear interpolation
/// @param f Input Func (1D, must be power-of-2 size)
/// @param n Size of the input array (must be a power of 2)
/// @param q Percentile in [0.0, 100.0]
/// @param name Function name
/// @return 1D Func with a single element containing the interpolated percentile
///
/// Uses bitonic_sort internally. Requires power-of-2 array size.
/// Linear interpolation: pos = q/100*(n-1), result = f[lo]*(1-frac) + f[hi]*frac.
inline
Halide::Func percentile(Halide::Func f, int n, float q,
                        std::string const& name = "percentile")
{
    nh_require(n > 0, "percentile: array must have at least 1 element");
    nh_require((n & (n - 1)) == 0,
        "percentile: size must be a power of 2 for bitonic sort, got %d", n);
    nh_require(q >= 0.0f && q <= 100.0f,
        "percentile: q must be in [0.0, 100.0], got %f", (double)q);

    Halide::Func sorted = bitonic_sort(f, n, name + "_sorted");

    // Continuous position in [0, n-1]. Host math in double so the
    // interpolation fraction carries full precision into f64 pipelines
    // (it was f32-computed, which quantized f64 interpolation weights).
    double pos = (static_cast<double>(q) / 100.0) * static_cast<double>(n - 1);
    int lo = static_cast<int>(pos);
    int hi = lo + 1;
    double frac = pos - static_cast<double>(lo);

    // Clamp indices to valid range
    lo = std::max(0, std::min(lo, n - 1));
    hi = std::max(0, std::min(hi, n - 1));

    Halide::Type t = f.types()[0];

    Halide::Func ret(name);
    Halide::Var x;
    ret(x) = Halide::cast(t, sorted(lo)) * Halide::Internal::make_const(t, 1.0 - frac)
           + Halide::cast(t, sorted(hi)) * Halide::Internal::make_const(t, frac);

    return ret;
}

/// @brief Compute the q-th quantile of a 1D array via linear interpolation
/// @param f Input Func (1D, must be power-of-2 size)
/// @param n Size of the input array (must be a power of 2)
/// @param q Quantile in [0.0, 1.0]
/// @param name Function name
/// @return 1D Func with a single element containing the interpolated quantile
///
/// Equivalent to percentile(f, n, q * 100.0f, name).
inline
Halide::Func quantile(Halide::Func f, int n, float q,
                      std::string const& name = "quantile")
{
    nh_require(q >= 0.0f && q <= 1.0f,
        "quantile: q must be in [0.0, 1.0], got %f", (double)q);
    return percentile(f, n, q * 100.0f, name);
}

// -----------------------------------------------------------------------------
// Covariance (1D scalar)
// -----------------------------------------------------------------------------

/// @brief Compute scalar covariance of two 1D arrays
/// @param a First input Func (1D, size n)
/// @param b Second input Func (1D, size n)
/// @param n Number of elements
/// @param ddof Delta degrees of freedom (1 = sample covariance)
/// @param name Function name
/// @return 1D Func with a single element: cov(a, b)
///
/// Formula: sum((a[i] - mean_a) * (b[i] - mean_b)) / (n - ddof)
inline
Halide::Func cov(Halide::Func a, Halide::Func b, int n, int ddof = 1,
                 std::string const& name = "cov")
{
    nh_require(n > ddof, "cov: n must be > ddof");

    Halide::Var x;
    Halide::RDom r(0, n);
    Halide::Type t = a.types()[0];

    // Compute sum of a and b to derive means
    Halide::Func sum_a(name + "_sum_a");
    sum_a(x) = Halide::cast(t, 0);
    sum_a(x) += a(r);

    Halide::Func sum_b(name + "_sum_b");
    sum_b(x) = Halide::cast(t, 0);
    sum_b(x) += b(r);

    // mean_a = sum_a / n,  mean_b = sum_b / n
    Halide::Expr mean_a = sum_a(0) / Halide::cast(t, n);
    Halide::Expr mean_b = sum_b(0) / Halide::cast(t, n);

    // Sum of (a - mean_a) * (b - mean_b)
    Halide::Func cross(name + "_cross");
    cross(x) = Halide::cast(t, 0);
    cross(x) += (a(r) - mean_a) * (b(r) - mean_b);

    // Covariance
    Halide::Func ret(name);
    ret(x) = cross(0) / Halide::cast(t, n - ddof);

    return ret;
}

// -----------------------------------------------------------------------------
// Pearson Correlation Coefficient (1D scalar)
// -----------------------------------------------------------------------------

/// @brief Compute Pearson correlation coefficient of two 1D arrays
/// @param a First input Func (1D, size n)
/// @param b Second input Func (1D, size n)
/// @param n Number of elements
/// @param name Function name
/// @return 1D Func with a single element: r = cov(a,b) / sqrt(var(a)*var(b))
///
/// Uses ddof=1 (sample statistics) for both variance and covariance.
inline
Halide::Func corrcoef(Halide::Func a, Halide::Func b, int n,
                      std::string const& name = "corrcoef")
{
    nh_require(n > 1, "corrcoef: n must be > 1 for sample statistics");

    Halide::Var x;
    Halide::RDom r(0, n);
    Halide::Type t = a.types()[0];

    // Sums for means
    Halide::Func sum_a(name + "_sum_a");
    sum_a(x) = Halide::cast(t, 0);
    sum_a(x) += a(r);

    Halide::Func sum_b(name + "_sum_b");
    sum_b(x) = Halide::cast(t, 0);
    sum_b(x) += b(r);

    Halide::Expr mean_a = sum_a(0) / Halide::cast(t, n);
    Halide::Expr mean_b = sum_b(0) / Halide::cast(t, n);

    // Sums of squared deviations and cross-deviations
    Halide::Func sum_sq_a(name + "_sq_a");
    sum_sq_a(x) = Halide::cast(t, 0);
    sum_sq_a(x) += (a(r) - mean_a) * (a(r) - mean_a);

    Halide::Func sum_sq_b(name + "_sq_b");
    sum_sq_b(x) = Halide::cast(t, 0);
    sum_sq_b(x) += (b(r) - mean_b) * (b(r) - mean_b);

    Halide::Func sum_cross(name + "_cross");
    sum_cross(x) = Halide::cast(t, 0);
    sum_cross(x) += (a(r) - mean_a) * (b(r) - mean_b);

    // r = cov(a,b,ddof=1) / sqrt(var(a,ddof=1) * var(b,ddof=1))
    //   = sum_cross / sqrt(sum_sq_a * sum_sq_b)   [n-1 cancels]
    Halide::Func ret(name);
    ret(x) = sum_cross(0) / Halide::sqrt(sum_sq_a(0) * sum_sq_b(0));

    return ret;
}

// -----------------------------------------------------------------------------
// 2D Histogram
// -----------------------------------------------------------------------------

/// @brief Compute a 2D histogram from paired 1D arrays
/// @param x_vals 1D Func of x values (size n)
/// @param y_vals 1D Func of y values (size n)
/// @param n Number of input pairs
/// @param x_bins Number of bins along the x axis
/// @param y_bins Number of bins along the y axis
/// @param x_min Left edge of the x range
/// @param x_max Right edge of the x range
/// @param y_min Left edge of the y range
/// @param y_max Right edge of the y range
/// @param name Function name
/// @return 2D Func of shape (x_bins, y_bins) with int32 counts
///
/// Pairs outside the given ranges are discarded (not clamped into edge bins).
/// Halide dim 0 = x bins (innermost), dim 1 = y bins (outermost).
inline
Halide::Func histogram2d(
    Halide::Func x_vals, Halide::Func y_vals, int n,
    int x_bins, int y_bins,
    Halide::Expr x_min, Halide::Expr x_max,
    Halide::Expr y_min, Halide::Expr y_max,
    std::string const& name = "histogram2d")
{
    nh_require(n > 0,      "histogram2d: n must be > 0");
    nh_require(x_bins > 0, "histogram2d: x_bins must be > 0");
    nh_require(y_bins > 0, "histogram2d: y_bins must be > 0");

    Halide::Func ret(name);
    Halide::Var bx("bx"), by("by");

    // Initialize all bins to 0
    ret(bx, by) = Halide::cast<int32_t>(0);

    Halide::RDom r(0, n);

    // Compute bin indices using floor division
    Halide::Expr xi = Halide::cast<int32_t>(
        Halide::floor((x_vals(r) - x_min) / (x_max - x_min) * Halide::cast(x_vals.types()[0], x_bins)));
    Halide::Expr yi = Halide::cast<int32_t>(
        Halide::floor((y_vals(r) - y_min) / (y_max - y_min) * Halide::cast(y_vals.types()[0], y_bins)));

    // Only count pairs that fall strictly within bounds
    Halide::Expr in_bounds = (xi >= 0) && (xi < x_bins) && (yi >= 0) && (yi < y_bins);

    ret(Halide::clamp(xi, 0, x_bins - 1), Halide::clamp(yi, 0, y_bins - 1)) +=
        Halide::select(in_bounds, Halide::cast<int32_t>(1), Halide::cast<int32_t>(0));

    return ret;
}

// -----------------------------------------------------------------------------
// Bincount
// -----------------------------------------------------------------------------

/// @brief Count occurrences of each non-negative integer value in a 1D array
/// @param f Input Func (1D) containing non-negative integer values
/// @param n Number of input elements
/// @param out_size Size of the output array (values >= out_size are ignored)
/// @param name Function name
/// @return 1D Func of size out_size with int32 counts
///
/// Values < 0 or >= out_size are silently ignored.
/// Equivalent to numpy.bincount(f, minlength=out_size) when max(f) < out_size.
inline
Halide::Func bincount(Halide::Func f, int n, int out_size,
                      std::string const& name = "bincount")
{
    nh_require(n > 0,        "bincount: n must be > 0");
    nh_require(out_size > 0, "bincount: out_size must be > 0");

    Halide::Func ret(name);
    Halide::Var bin("bin");

    // Initialize all bins to 0
    ret(bin) = Halide::cast<int32_t>(0);

    Halide::RDom r(0, n);

    // Only count values that fall in [0, out_size)
    Halide::Expr val = f(r);
    Halide::Expr in_bounds = (val >= 0) && (val < out_size);

    ret(Halide::clamp(val, 0, out_size - 1)) +=
        Halide::select(in_bounds, Halide::cast<int32_t>(1), Halide::cast<int32_t>(0));

    return ret;
}

// -----------------------------------------------------------------------------
// NaN-aware Percentile / Quantile
// -----------------------------------------------------------------------------

/// @brief Percentile of a 1D array ignoring NaN values
/// @param f  Input Func (1D, n elements)
/// @param n  Total element count (including NaN)
/// @param q  Percentile in [0, 100]
/// @return 0D (scalar) Func with the q-th percentile of non-NaN values
inline
Halide::Func nanpercentile(Halide::Func f, int n, float q,
    std::string const& name = "nanpercentile")
{
    Halide::Var i("i");

    // Replace NaN with +inf so they sort to the end. Sentinel made in f's
    // own float type (an f32 infinity against an f64 arm is a select-type
    // mismatch at definition).
    Halide::Type tc = f.types()[0];
    if (!tc.is_float()) tc = Halide::Float(32);
    Halide::Func clean(name + "_clean");
    clean(i) = Halide::select(Halide::is_nan(f(i)),
        Halide::Internal::make_const(tc, std::numeric_limits<double>::infinity()),
        f(i));
    clean.compute_root();

    // Sort ascending
    Halide::Func sorted = sort_1d(clean, n, true, name + "_sorted");
    sorted.compute_root();

    // Count non-NaN elements
    Halide::Func cnt(name + "_cnt");
    Halide::RDom rc(0, n, "rc_np");
    cnt() = Halide::cast<int32_t>(0);
    cnt() += Halide::cast<int32_t>(Halide::select(Halide::is_nan(f(rc)), 0, 1));
    cnt.compute_root();

    // Linear interpolation at position pos = q/100 * (k-1), computed in
    // the sorted values' own float type (f64 stays f64 — the interpolation
    // fraction was f32-precision on f64 inputs). q/100 keeps the host-f32
    // quotient (float API parameter), so f32 results are unchanged.
    Halide::Type t = sorted.types()[0];
    if (!t.is_float()) t = Halide::Float(32);
    Halide::Expr zero_t = Halide::Internal::make_zero(t);
    Halide::Expr one_t  = Halide::Internal::make_one(t);

    Halide::Expr k   = Halide::cast(t, cnt());
    Halide::Expr pos = Halide::Internal::make_const(t, static_cast<double>(q / 100.0f))
                       * (k - one_t);
    Halide::Expr lo_idx = Halide::cast<int32_t>(Halide::floor(pos));
    Halide::Expr hi_idx = Halide::min(lo_idx + 1, cnt() - 1);
    Halide::Expr frac   = pos - Halide::floor(pos);

    Halide::Func lo_e(name + "_lo");
    Halide::RDom rlo(0, n, "rlo_np");
    lo_e() = zero_t;
    lo_e() += Halide::select(rlo == lo_idx, sorted(rlo), zero_t);
    lo_e.compute_root();

    Halide::Func hi_e(name + "_hi");
    Halide::RDom rhi(0, n, "rhi_np");
    hi_e() = zero_t;
    hi_e() += Halide::select(rhi == hi_idx, sorted(rhi), zero_t);
    hi_e.compute_root();

    Halide::Func ret(name);
    ret() = lo_e() * (one_t - frac) + hi_e() * frac;
    return ret;
}

/// @brief Quantile of a 1D array ignoring NaN values (q in [0, 1])
inline
Halide::Func nanquantile(Halide::Func f, int n, float q,
    std::string const& name = "nanquantile")
{
    return nanpercentile(f, n, q * 100.0f, name);
}

} // namespace stats

NS_NUM_HALIDE_END
