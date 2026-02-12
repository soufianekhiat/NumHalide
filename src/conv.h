/// @file conv.h
/// @brief Convolution and correlation operations
///
/// Provides: convolve1d, convolve2d, correlate2d

#pragma once

#include "common.h"
#include "numhalide.h"
#include "shape.h"

NS_NUM_HALIDE_BEGIN

// -----------------------------------------------------------------------------
// 1D Convolution
// -----------------------------------------------------------------------------

/// @brief 1D convolution
/// @param input Input signal
/// @param in_shape Shape of input (1D)
/// @param kernel Convolution kernel
/// @param kernel_size Size of kernel
/// @param mode Padding mode: "valid", "same", "full"
/// @param name Function name
/// @return Convolved signal
inline
Halide::Func convolve1d(Halide::Func input, const shape_t& in_shape,
                        Halide::Func kernel, int kernel_size,
                        const std::string& mode = "same",
                        std::string const& name = "conv1d")
{
    nh_require(nullptr, in_shape.rank == 1, "convolve1d requires 1D input");

    int n = in_shape.extents[0];
    int half_k = kernel_size / 2;

    Halide::Func ret(name);
    Halide::Var x("x");
    Halide::RDom r(0, kernel_size, "r");

    // Convolution flips the kernel
    // output[x] = sum_k input[x + k - half_k] * kernel[kernel_size - 1 - k]
    // For correlation (no flip): kernel[k]

    Halide::Expr idx = x + r - half_k;

    // Boundary handling
    Halide::Expr clamped_idx = Halide::clamp(idx, 0, n - 1);

    ret(x) = Halide::cast(input.types()[0], 0);
    ret(x) += input(clamped_idx) * kernel(kernel_size - 1 - r);

    return ret;
}

// -----------------------------------------------------------------------------
// 2D Convolution
// -----------------------------------------------------------------------------

/// @brief 2D convolution with a kernel
/// @param input Input image
/// @param in_shape Shape of input (2D: rows x cols)
/// @param kernel Convolution kernel
/// @param kernel_rows Kernel height
/// @param kernel_cols Kernel width
/// @param name Function name
/// @return Convolved image
inline
Halide::Func convolve2d(Halide::Func input, const shape_t& in_shape,
                        Halide::Func kernel, int kernel_rows, int kernel_cols,
                        std::string const& name = "conv2d")
{
    nh_require(nullptr, in_shape.rank == 2, "convolve2d requires 2D input");

    int rows = in_shape.extents[0];
    int cols = in_shape.extents[1];
    int half_kr = kernel_rows / 2;
    int half_kc = kernel_cols / 2;

    Halide::Func ret(name);
    Halide::Var x("x"), y("y");
    Halide::RDom r(0, kernel_cols, 0, kernel_rows, "r");

    // Convolution flips the kernel
    Halide::Expr in_x = x + r.x - half_kc;
    Halide::Expr in_y = y + r.y - half_kr;

    // Clamp to boundary (edge extension)
    in_x = Halide::clamp(in_x, 0, cols - 1);
    in_y = Halide::clamp(in_y, 0, rows - 1);

    // Flipped kernel indices
    Halide::Expr k_x = kernel_cols - 1 - r.x;
    Halide::Expr k_y = kernel_rows - 1 - r.y;

    ret(x, y) = Halide::cast(input.types()[0], 0);
    ret(x, y) += input(in_x, in_y) * kernel(k_x, k_y);

    return ret;
}

/// @brief 2D convolution with separable kernel
/// @param input Input image
/// @param in_shape Shape of input
/// @param kernel_x Horizontal kernel (1D)
/// @param kernel_y Vertical kernel (1D)
/// @param kernel_size Size of each 1D kernel
/// @param name Function name
/// @return Convolved image
inline
Halide::Func convolve2d_separable(Halide::Func input, const shape_t& in_shape,
                                   Halide::Func kernel_x, Halide::Func kernel_y,
                                   int kernel_size,
                                   std::string const& name = "conv2d_sep")
{
    nh_require(nullptr, in_shape.rank == 2, "convolve2d_separable requires 2D input");

    int rows = in_shape.extents[0];
    int cols = in_shape.extents[1];
    int half_k = kernel_size / 2;

    // First pass: horizontal
    Halide::Func horiz("conv_horiz");
    Halide::Var x("x"), y("y");
    Halide::RDom rx(0, kernel_size, "rx");

    Halide::Expr in_x = Halide::clamp(x + rx - half_k, 0, cols - 1);
    horiz(x, y) = Halide::cast(input.types()[0], 0);
    horiz(x, y) += input(in_x, y) * kernel_x(kernel_size - 1 - rx);

    // Second pass: vertical
    Halide::Func ret(name);
    Halide::RDom ry(0, kernel_size, "ry");

    Halide::Expr in_y = Halide::clamp(y + ry - half_k, 0, rows - 1);
    ret(x, y) = Halide::cast(input.types()[0], 0);
    ret(x, y) += horiz(x, in_y) * kernel_y(kernel_size - 1 - ry);

    return ret;
}

// -----------------------------------------------------------------------------
// 2D Correlation
// -----------------------------------------------------------------------------

/// @brief 2D correlation (no kernel flip)
/// @param input Input image
/// @param in_shape Shape of input
/// @param kernel Correlation kernel
/// @param kernel_rows Kernel height
/// @param kernel_cols Kernel width
/// @param name Function name
/// @return Correlated image
inline
Halide::Func correlate2d(Halide::Func input, const shape_t& in_shape,
                         Halide::Func kernel, int kernel_rows, int kernel_cols,
                         std::string const& name = "corr2d")
{
    nh_require(nullptr, in_shape.rank == 2, "correlate2d requires 2D input");

    int rows = in_shape.extents[0];
    int cols = in_shape.extents[1];
    int half_kr = kernel_rows / 2;
    int half_kc = kernel_cols / 2;

    Halide::Func ret(name);
    Halide::Var x("x"), y("y");
    Halide::RDom r(0, kernel_cols, 0, kernel_rows, "r");

    Halide::Expr in_x = Halide::clamp(x + r.x - half_kc, 0, cols - 1);
    Halide::Expr in_y = Halide::clamp(y + r.y - half_kr, 0, rows - 1);

    // No flip for correlation
    ret(x, y) = Halide::cast(input.types()[0], 0);
    ret(x, y) += input(in_x, in_y) * kernel(r.x, r.y);

    return ret;
}

// -----------------------------------------------------------------------------
// Common Kernels
// -----------------------------------------------------------------------------

/// @brief Create a box (averaging) kernel
/// @param size Kernel size
/// @param name Function name
/// @return Box kernel Func
inline
Halide::Func box_kernel(int size, std::string const& name = "box_kernel")
{
    Halide::Func ret(name);
    Halide::Var x("x"), y("y");
    ret(x, y) = 1.0f / (size * size);
    return ret;
}

/// @brief Create a Gaussian kernel (1D)
/// @param size Kernel size
/// @param sigma Standard deviation
/// @param name Function name
/// @return Gaussian kernel Func
inline
Halide::Func gaussian_kernel_1d(int size, float sigma, std::string const& name = "gauss_kernel_1d")
{
    Halide::Func ret(name);
    Halide::Var x("x");

    float center = (size - 1) / 2.0f;
    float sigma_sq = sigma * sigma;

    // Gaussian: exp(-(x-center)^2 / (2*sigma^2))
    Halide::Expr dx = Halide::cast<float>(x) - center;
    Halide::Expr val = Halide::exp(-dx * dx / (2 * sigma_sq));

    ret(x) = val;
    return ret;
}

/// @brief Create a Sobel X kernel (edge detection)
/// @param name Function name
/// @return 3x3 Sobel X kernel
inline
Halide::Func sobel_x_kernel(std::string const& name = "sobel_x")
{
    Halide::Func ret(name);
    Halide::Var x("x"), y("y");

    // Sobel X: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    ret(x, y) = Halide::cast<float>(Halide::select(
        x == 0, Halide::select(y == 1, -2, -1),
        Halide::select(x == 2, Halide::select(y == 1, 2, 1), 0)
    ));

    return ret;
}

/// @brief Create a Sobel Y kernel (edge detection)
/// @param name Function name
/// @return 3x3 Sobel Y kernel
inline
Halide::Func sobel_y_kernel(std::string const& name = "sobel_y")
{
    Halide::Func ret(name);
    Halide::Var x("x"), y("y");

    // Sobel Y: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    ret(x, y) = Halide::cast<float>(Halide::select(
        y == 0, Halide::select(x == 1, -2, -1),
        Halide::select(y == 2, Halide::select(x == 1, 2, 1), 0)
    ));

    return ret;
}

/// @brief Create a Laplacian kernel
/// @param name Function name
/// @return 3x3 Laplacian kernel
inline
Halide::Func laplacian_kernel(std::string const& name = "laplacian")
{
    Halide::Func ret(name);
    Halide::Var x("x"), y("y");

    // Laplacian: [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    ret(x, y) = Halide::cast<float>(Halide::select(
        x == 1 && y == 1, -4,
        Halide::select((x == 1) || (y == 1), 1, 0)
    ));

    return ret;
}

NS_NUM_HALIDE_END
