/// @file inplace.h
/// @brief In-place element-wise operations on Halide::Runtime::Buffer<float>
///
/// All functions modify the buffer directly (no new allocation).
/// Uses C++ element-wise loops that work correctly for any buffer layout,
/// including views created by view_transpose / view_slice / view_reshape.

#pragma once

#include "shape.h"

#include <string>
#include <algorithm>
#include <limits>
#include <cmath>

NS_NUM_HALIDE_BEGIN

// =============================================================================
// Internal helpers
// =============================================================================

namespace inplace_detail {

/// Apply fn(float) -> float to every element of a 1-3D buffer, respecting strides.
template<typename Fn>
inline void apply_nd(Halide::Runtime::Buffer<float>& src, Fn fn)
{
    int ndim = src.dimensions();
    if (ndim == 1) {
        int x0 = src.dim(0).min(), xn = x0 + src.dim(0).extent();
        for (int x = x0; x < xn; ++x)
            src(x) = fn(src(x));
    } else if (ndim == 2) {
        int x0 = src.dim(0).min(), xn = x0 + src.dim(0).extent();
        int y0 = src.dim(1).min(), yn = y0 + src.dim(1).extent();
        for (int y = y0; y < yn; ++y)
            for (int x = x0; x < xn; ++x)
                src(x, y) = fn(src(x, y));
    } else {
        int x0 = src.dim(0).min(), xn = x0 + src.dim(0).extent();
        int y0 = src.dim(1).min(), yn = y0 + src.dim(1).extent();
        int z0 = src.dim(2).min(), zn = z0 + src.dim(2).extent();
        for (int z = z0; z < zn; ++z)
            for (int y = y0; y < yn; ++y)
                for (int x = x0; x < xn; ++x)
                    src(x, y, z) = fn(src(x, y, z));
    }
}

/// Scan min and max over all elements (1-3D, stride-correct).
inline void scan_minmax(const Halide::Runtime::Buffer<float>& src,
                        float& out_min, float& out_max)
{
    out_min = std::numeric_limits<float>::max();
    out_max = std::numeric_limits<float>::lowest();

    const int ndim = src.dimensions();
    if (ndim == 1) {
        int x0 = src.dim(0).min(), xn = x0 + src.dim(0).extent();
        for (int x = x0; x < xn; ++x) {
            float v = src(x);
            out_min = std::min(out_min, v);
            out_max = std::max(out_max, v);
        }
    } else if (ndim == 2) {
        int x0 = src.dim(0).min(), xn = x0 + src.dim(0).extent();
        int y0 = src.dim(1).min(), yn = y0 + src.dim(1).extent();
        for (int y = y0; y < yn; ++y)
            for (int x = x0; x < xn; ++x) {
                float v = src(x, y);
                out_min = std::min(out_min, v);
                out_max = std::max(out_max, v);
            }
    } else {
        int x0 = src.dim(0).min(), xn = x0 + src.dim(0).extent();
        int y0 = src.dim(1).min(), yn = y0 + src.dim(1).extent();
        int z0 = src.dim(2).min(), zn = z0 + src.dim(2).extent();
        for (int z = z0; z < zn; ++z)
            for (int y = y0; y < yn; ++y)
                for (int x = x0; x < xn; ++x) {
                    float v = src(x, y, z);
                    out_min = std::min(out_min, v);
                    out_max = std::max(out_max, v);
                }
    }
}

} // namespace inplace_detail

// =============================================================================
// Public API
// =============================================================================

/// @brief Threshold from below: src[i] = max(src[i], thresh)
inline void inplace_threshold(Halide::Runtime::Buffer<float>& src, float thresh,
                               const std::string& /*name*/ = "inplace_threshold")
{
    nh_require(src.dimensions() >= 1 && src.dimensions() <= 3,
               "inplace_threshold: unsupported ndim=%d", src.dimensions());
    inplace_detail::apply_nd(src, [thresh](float v) { return std::max(v, thresh); });
}

/// @brief Clamp to [lo, hi]: src[i] = clamp(src[i], lo, hi)
inline void inplace_clamp(Halide::Runtime::Buffer<float>& src, float lo, float hi,
                          const std::string& /*name*/ = "inplace_clamp")
{
    nh_require(src.dimensions() >= 1 && src.dimensions() <= 3,
               "inplace_clamp: unsupported ndim=%d", src.dimensions());
    inplace_detail::apply_nd(src, [lo, hi](float v) {
        return v < lo ? lo : (v > hi ? hi : v);
    });
}

/// @brief Scale: src[i] = src[i] * factor
inline void inplace_scale(Halide::Runtime::Buffer<float>& src, float factor,
                          const std::string& /*name*/ = "inplace_scale")
{
    nh_require(src.dimensions() >= 1 && src.dimensions() <= 3,
               "inplace_scale: unsupported ndim=%d", src.dimensions());
    inplace_detail::apply_nd(src, [factor](float v) { return v * factor; });
}

/// @brief Add scalar: src[i] = src[i] + value
inline void inplace_add_scalar(Halide::Runtime::Buffer<float>& src, float value,
                               const std::string& /*name*/ = "inplace_add_scalar")
{
    nh_require(src.dimensions() >= 1 && src.dimensions() <= 3,
               "inplace_add_scalar: unsupported ndim=%d", src.dimensions());
    inplace_detail::apply_nd(src, [value](float v) { return v + value; });
}

/// @brief Elementwise exp: src[i] = exp(src[i])
inline void inplace_exp(Halide::Runtime::Buffer<float>& src,
                        const std::string& /*name*/ = "inplace_exp")
{
    nh_require(src.dimensions() >= 1 && src.dimensions() <= 3,
               "inplace_exp: unsupported ndim=%d", src.dimensions());
    inplace_detail::apply_nd(src, [](float v) { return std::exp(v); });
}

/// @brief Elementwise sqrt (clamped to 0): src[i] = sqrt(max(src[i], 0))
inline void inplace_sqrt(Halide::Runtime::Buffer<float>& src,
                         const std::string& /*name*/ = "inplace_sqrt")
{
    nh_require(src.dimensions() >= 1 && src.dimensions() <= 3,
               "inplace_sqrt: unsupported ndim=%d", src.dimensions());
    inplace_detail::apply_nd(src, [](float v) { return std::sqrt(std::max(v, 0.0f)); });
}

/// @brief Gamma correction: src[i] = pow(clamp(src[i], 0, 1), gamma)
inline void inplace_gamma(Halide::Runtime::Buffer<float>& src, float gamma,
                          const std::string& /*name*/ = "inplace_gamma")
{
    nh_require(src.dimensions() >= 1 && src.dimensions() <= 3,
               "inplace_gamma: unsupported ndim=%d", src.dimensions());
    inplace_detail::apply_nd(src, [gamma](float v) {
        float clamped = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
        return std::pow(clamped, gamma);
    });
}

/// @brief Normalize to [0, 1]: src[i] = (src[i] - min) / (max - min)
/// If all values are equal, all elements are set to 0.0f.
inline void inplace_normalize(Halide::Runtime::Buffer<float>& src,
                              const std::string& name = "inplace_normalize")
{
    nh_require(src.dimensions() >= 1 && src.dimensions() <= 3,
               "inplace_normalize: unsupported ndim=%d", src.dimensions());
    float mn, mx;
    inplace_detail::scan_minmax(src, mn, mx);
    float range = mx - mn;
    if (range < 1e-9f) {
        inplace_detail::apply_nd(src, [](float) { return 0.0f; });
        return;
    }
    inplace_add_scalar(src, -mn, name + "_shift");
    inplace_scale(src, 1.0f / range, name + "_scale");
}

NS_NUM_HALIDE_END
