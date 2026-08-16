/// @file interp.h
/// @brief Interpolation and resampling operations
///
/// Provides: interp1d, zoom, map_coordinates, resize

#pragma once

#include "common.h"
#include "numhalide.h"
#include "shape.h"

NS_NUM_HALIDE_BEGIN

// -----------------------------------------------------------------------------
// 1D Linear Interpolation
// -----------------------------------------------------------------------------

/// @brief 1D linear interpolation
/// @param x_new New x coordinates to interpolate at
/// @param x_coords Original x coordinates (1D Func)
/// @param y_values Values at original x coordinates (1D Func)
/// @param n Number of original points
/// @param name Function name
/// @return Interpolated values at x_new coordinates
inline
Halide::Func interp1d(Halide::Func x_new, Halide::Func x_coords, Halide::Func y_values,
                      int n, std::string const& name = "interp1d")
{
    Halide::Func ret(name);
    Halide::Var i("i");

    // For each x_new, find the bracketing indices and interpolate
    // Simple approach: scan to find where x fits

    // Use binary search to find index
    Halide::RDom r(0, n - 1, "r");

    // Find largest index where x_coords <= x_new
    Halide::Func find_idx("find_idx");
    find_idx(i) = 0;
    find_idx(i) = Halide::select(x_coords(r) <= x_new(i), r, find_idx(i));

    // Get the bracket
    Halide::Expr idx = Halide::clamp(find_idx(i), 0, n - 2);
    Halide::Expr x0 = x_coords(idx);
    Halide::Expr x1 = x_coords(idx + 1);
    Halide::Expr y0 = y_values(idx);
    Halide::Expr y1 = y_values(idx + 1);

    // Linear interpolation: y = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    Halide::Expr t = (x_new(i) - x0) / (x1 - x0);
    t = Halide::clamp(t, 0.0f, 1.0f);  // Clamp for extrapolation

    ret(i) = y0 + (y1 - y0) * t;

    return ret;
}

/// @brief Simple 1D interpolation with uniform spacing
/// @param values Original values (1D)
/// @param in_shape Shape of values
/// @param scale Scale factor for output
/// @param name Function name
/// @return Interpolated values
inline
Halide::Func interp1d_uniform(Halide::Func values, const shape_t& in_shape,
                               float scale, std::string const& name = "interp1d_uniform")
{
    nh_require(in_shape.rank == 1, "interp1d_uniform requires 1D input");

    int n = in_shape.extents[0];

    Halide::Func ret(name);
    Halide::Var x("x");

    // Interpolation weights are computed in the values' own float type
    // (f64 stays f64 — f32 weights silently truncated f64 interpolation);
    // integer values keep the historical f32 weights. scale itself is
    // f32-quantized by the float API parameter.
    Halide::Type ty = values.types()[0];
    if (!ty.is_float()) ty = Halide::Float(32);

    // Map output index to input coordinate
    Halide::Expr in_x = Halide::cast(ty, x) / Halide::cast(ty, scale);

    // Get integer indices
    Halide::Expr x0 = Halide::cast<int32_t>(Halide::floor(in_x));
    Halide::Expr x1 = x0 + 1;
    Halide::Expr t = in_x - Halide::cast(ty, x0);

    // Clamp indices
    x0 = Halide::clamp(x0, 0, n - 1);
    x1 = Halide::clamp(x1, 0, n - 1);

    // Interpolate
    ret(x) = values(x0) * (Halide::Internal::make_one(ty) - t) + values(x1) * t;

    return ret;
}

// -----------------------------------------------------------------------------
// 2D Bilinear Interpolation
// -----------------------------------------------------------------------------

/// @brief Bilinear interpolation for 2D image resize
/// @param input Input image
/// @param in_shape Input shape (rows x cols)
/// @param out_rows Output height
/// @param out_cols Output width
/// @param name Function name
/// @return Resized image
inline
Halide::Func resize_bilinear(Halide::Func input, const shape_t& in_shape,
                              int out_rows, int out_cols,
                              std::string const& name = "resize")
{
    nh_require(in_shape.rank == 2, "resize_bilinear requires 2D input");

    int in_rows = in_shape.extents[0];
    int in_cols = in_shape.extents[1];

    Halide::Func ret(name);
    Halide::Var x("x"), y("y");

    // Interpolation weights are computed in the input's own float type
    // (f64 stays f64 — f32 weights silently truncated f64 interpolation);
    // integer inputs keep the historical f32 weights.
    Halide::Type wt = input.types()[0];
    if (!wt.is_float()) wt = Halide::Float(32);
    Halide::Expr one = Halide::Internal::make_one(wt);

    // Map output coordinates to input coordinates
    Halide::Expr scale_x = Halide::cast(wt, in_cols - 1) / Halide::cast(wt, out_cols - 1);
    Halide::Expr scale_y = Halide::cast(wt, in_rows - 1) / Halide::cast(wt, out_rows - 1);

    Halide::Expr in_x = Halide::cast(wt, x) * scale_x;
    Halide::Expr in_y = Halide::cast(wt, y) * scale_y;

    // Get integer indices
    Halide::Expr x0 = Halide::cast<int32_t>(Halide::floor(in_x));
    Halide::Expr y0 = Halide::cast<int32_t>(Halide::floor(in_y));
    Halide::Expr x1 = x0 + 1;
    Halide::Expr y1 = y0 + 1;

    Halide::Expr tx = in_x - Halide::cast(wt, x0);
    Halide::Expr ty = in_y - Halide::cast(wt, y0);

    // Clamp indices
    x0 = Halide::clamp(x0, 0, in_cols - 1);
    x1 = Halide::clamp(x1, 0, in_cols - 1);
    y0 = Halide::clamp(y0, 0, in_rows - 1);
    y1 = Halide::clamp(y1, 0, in_rows - 1);

    // Bilinear interpolation
    Halide::Expr v00 = input(x0, y0);
    Halide::Expr v10 = input(x1, y0);
    Halide::Expr v01 = input(x0, y1);
    Halide::Expr v11 = input(x1, y1);

    Halide::Expr top = v00 * (one - tx) + v10 * tx;
    Halide::Expr bottom = v01 * (one - tx) + v11 * tx;

    ret(x, y) = top * (one - ty) + bottom * ty;

    return ret;
}

/// @brief Infer output shape for resize
inline shape_t infer_resize(const shape_t& /*in*/, int out_rows, int out_cols) {
    return shape_t{out_rows, out_cols};
}

// -----------------------------------------------------------------------------
// Nearest Neighbor Interpolation
// -----------------------------------------------------------------------------

/// @brief Nearest neighbor resize for 2D image
/// @param input Input image
/// @param in_shape Input shape
/// @param out_rows Output height
/// @param out_cols Output width
/// @param name Function name
/// @return Resized image
inline
Halide::Func resize_nearest(Halide::Func input, const shape_t& in_shape,
                            int out_rows, int out_cols,
                            std::string const& name = "resize_nearest")
{
    nh_require(in_shape.rank == 2, "resize_nearest requires 2D input");

    int in_rows = in_shape.extents[0];
    int in_cols = in_shape.extents[1];

    Halide::Func ret(name);
    Halide::Var x("x"), y("y");

    Halide::Expr scale_x = Halide::cast<float>(in_cols) / Halide::cast<float>(out_cols);
    Halide::Expr scale_y = Halide::cast<float>(in_rows) / Halide::cast<float>(out_rows);

    Halide::Expr in_x = Halide::cast<int32_t>(Halide::round(Halide::cast<float>(x) * scale_x));
    Halide::Expr in_y = Halide::cast<int32_t>(Halide::round(Halide::cast<float>(y) * scale_y));

    in_x = Halide::clamp(in_x, 0, in_cols - 1);
    in_y = Halide::clamp(in_y, 0, in_rows - 1);

    ret(x, y) = input(in_x, in_y);

    return ret;
}

// -----------------------------------------------------------------------------
// Zoom (Scale by Factor)
// -----------------------------------------------------------------------------

/// @brief Zoom (scale) a 2D image by a factor
/// @param input Input image
/// @param in_shape Input shape
/// @param factor Zoom factor (>1 = enlarge, <1 = shrink)
/// @param name Function name
/// @return Zoomed image
inline
Halide::Func zoom(Halide::Func input, const shape_t& in_shape, float factor,
                  std::string const& name = "zoom")
{
    int out_rows = static_cast<int>(in_shape.extents[0] * factor);
    int out_cols = static_cast<int>(in_shape.extents[1] * factor);
    return resize_bilinear(input, in_shape, out_rows, out_cols, name);
}

/// @brief Infer output shape for zoom
inline shape_t infer_zoom(const shape_t& in, float factor) {
    return shape_t{
        static_cast<int>(in.extents[0] * factor),
        static_cast<int>(in.extents[1] * factor)
    };
}

// -----------------------------------------------------------------------------
// Map Coordinates
// -----------------------------------------------------------------------------

/// @brief Sample image at arbitrary coordinates
/// @param input Input image
/// @param in_shape Input shape
/// @param coords_x X coordinates Func
/// @param coords_y Y coordinates Func
/// @param name Function name
/// @return Sampled values with bilinear interpolation
inline
Halide::Func map_coordinates(Halide::Func input, const shape_t& in_shape,
                              Halide::Func coords_x, Halide::Func coords_y,
                              std::string const& name = "map_coordinates")
{
    nh_require(in_shape.rank == 2, "map_coordinates requires 2D input");

    int rows = in_shape.extents[0];
    int cols = in_shape.extents[1];

    Halide::Func ret(name);
    Halide::Var x("x"), y("y");

    // Get float coordinates
    Halide::Expr cx = coords_x(x, y);
    Halide::Expr cy = coords_y(x, y);

    // Integer indices
    Halide::Expr x0 = Halide::cast<int32_t>(Halide::floor(cx));
    Halide::Expr y0 = Halide::cast<int32_t>(Halide::floor(cy));
    Halide::Expr x1 = x0 + 1;
    Halide::Expr y1 = y0 + 1;

    Halide::Expr tx = cx - Halide::cast<float>(x0);
    Halide::Expr ty = cy - Halide::cast<float>(y0);

    // Clamp
    x0 = Halide::clamp(x0, 0, cols - 1);
    x1 = Halide::clamp(x1, 0, cols - 1);
    y0 = Halide::clamp(y0, 0, rows - 1);
    y1 = Halide::clamp(y1, 0, rows - 1);

    // Bilinear
    Halide::Expr v00 = input(x0, y0);
    Halide::Expr v10 = input(x1, y0);
    Halide::Expr v01 = input(x0, y1);
    Halide::Expr v11 = input(x1, y1);

    Halide::Expr top = v00 * (1.0f - tx) + v10 * tx;
    Halide::Expr bottom = v01 * (1.0f - tx) + v11 * tx;

    ret(x, y) = top * (1.0f - ty) + bottom * ty;

    return ret;
}

NS_NUM_HALIDE_END
