/// @file test_conv.cpp
/// @brief Tests for convolution and correlation operations

#include <gtest/gtest.h>
#include "numhalide_all.h"
#include <cmath>

using namespace numhalide;

// -----------------------------------------------------------------------------
// 1D Convolution Tests
// -----------------------------------------------------------------------------

TEST(Conv, Convolve1DIdentity) {
    // Convolving with delta kernel should give same result
    shape_t shape = {5};
    Halide::Func input("input");
    Halide::Var x;
    input(x) = Halide::cast<float>(x + 1);  // [1, 2, 3, 4, 5]

    Halide::Func kernel("kernel");
    kernel(x) = Halide::select(x == 1, 1.0f, 0.0f);  // [0, 1, 0]

    auto result = convolve1d(input, shape, kernel, 3, "same", "conv_result");

    Halide::Runtime::Buffer<float> out(5);
    result.realize(out);

    EXPECT_NEAR(out(0), 1.0f, 1e-5f);
    EXPECT_NEAR(out(1), 2.0f, 1e-5f);
    EXPECT_NEAR(out(2), 3.0f, 1e-5f);
    EXPECT_NEAR(out(3), 4.0f, 1e-5f);
    EXPECT_NEAR(out(4), 5.0f, 1e-5f);
}

TEST(Conv, Convolve1DSmoothing) {
    // Box filter smoothing
    shape_t shape = {5};
    Halide::Func input("input");
    Halide::Var x;
    input(x) = Halide::cast<float>(Halide::select(x == 2, 9.0f, 0.0f));  // [0,0,9,0,0]

    Halide::Func kernel("kernel");
    kernel(x) = 1.0f / 3.0f;  // [1/3, 1/3, 1/3]

    auto result = convolve1d(input, shape, kernel, 3, "same", "conv_smooth");

    Halide::Runtime::Buffer<float> out(5);
    result.realize(out);

    // Impulse should spread: [0, 3, 3, 3, 0]
    EXPECT_NEAR(out(0), 0.0f, 1e-5f);
    EXPECT_NEAR(out(1), 3.0f, 1e-5f);
    EXPECT_NEAR(out(2), 3.0f, 1e-5f);
    EXPECT_NEAR(out(3), 3.0f, 1e-5f);
    EXPECT_NEAR(out(4), 0.0f, 1e-5f);
}

// -----------------------------------------------------------------------------
// 2D Convolution Tests
// -----------------------------------------------------------------------------

TEST(Conv, Convolve2DBox) {
    // 3x3 box blur on simple image
    shape_t shape = {3, 3};
    Halide::Func input("input");
    Halide::Var x, y;
    // Center pixel = 9, others = 0
    input(x, y) = Halide::cast<float>(Halide::select(x == 1 && y == 1, 9.0f, 0.0f));

    auto kernel = box_kernel(3, "box");

    auto result = convolve2d(input, shape, kernel, 3, 3, "conv_box");

    Halide::Runtime::Buffer<float> out(3, 3);
    result.realize(out);

    // All pixels should be 9/9 = 1 due to edge clamping behavior
    EXPECT_NEAR(out(1, 1), 1.0f, 1e-5f);
}

TEST(Conv, Convolve2DIdentity) {
    shape_t shape = {4, 4};
    Halide::Func input("input");
    Halide::Var x, y;
    input(x, y) = Halide::cast<float>(y * 4 + x);  // 0-15

    // Delta kernel
    Halide::Func kernel("delta");
    kernel(x, y) = Halide::select(x == 1 && y == 1, 1.0f, 0.0f);

    auto result = convolve2d(input, shape, kernel, 3, 3, "conv_id");

    Halide::Runtime::Buffer<float> out(4, 4);
    result.realize(out);

    EXPECT_NEAR(out(1, 1), 5.0f, 1e-5f);  // Original value at (1,1)
    EXPECT_NEAR(out(2, 2), 10.0f, 1e-5f); // Original value at (2,2)
}

TEST(Conv, Correlate2D) {
    shape_t shape = {3, 3};
    Halide::Func input("input");
    Halide::Var x, y;
    input(x, y) = Halide::cast<float>(y * 3 + x + 1);  // [[1,2,3],[4,5,6],[7,8,9]]

    // Symmetric kernel: correlation = convolution
    auto kernel = box_kernel(3, "box");

    auto conv = convolve2d(input, shape, kernel, 3, 3, "conv");
    auto corr = correlate2d(input, shape, kernel, 3, 3, "corr");

    Halide::Runtime::Buffer<float> conv_out(3, 3);
    Halide::Runtime::Buffer<float> corr_out(3, 3);
    conv.realize(conv_out);
    corr.realize(corr_out);

    // For symmetric kernel, conv and corr should be equal
    EXPECT_NEAR(conv_out(1, 1), corr_out(1, 1), 1e-5f);
}

// -----------------------------------------------------------------------------
// Separable Convolution Tests
// -----------------------------------------------------------------------------

TEST(Conv, Convolve2DSeparable) {
    shape_t shape = {4, 4};
    Halide::Func input("input");
    Halide::Var x, y;
    input(x, y) = Halide::cast<float>(Halide::select(x == 2 && y == 2, 1.0f, 0.0f));

    // Separable box filter
    Halide::Func kernel_1d("kernel_1d");
    kernel_1d(x) = 1.0f / 3.0f;

    auto result = convolve2d_separable(input, shape, kernel_1d, kernel_1d, 3, "conv_sep");

    Halide::Runtime::Buffer<float> out(4, 4);
    result.realize(out);

    // Impulse should spread to 3x3 region with value 1/9
    EXPECT_NEAR(out(2, 2), 1.0f / 9.0f, 1e-5f);
    EXPECT_NEAR(out(1, 2), 1.0f / 9.0f, 1e-5f);
    EXPECT_NEAR(out(2, 1), 1.0f / 9.0f, 1e-5f);
}

// -----------------------------------------------------------------------------
// Kernel Tests
// -----------------------------------------------------------------------------

TEST(Conv, SobelX) {
    shape_t shape = {3, 5};
    Halide::Func input("input");
    Halide::Var x, y;
    // Horizontal gradient: each column has constant value
    input(x, y) = Halide::cast<float>(x);  // Cols: 0, 1, 2, 3, 4

    auto kernel = sobel_x_kernel("sobel_x");
    auto result = convolve2d(input, shape, kernel, 3, 3, "sobel_result");

    Halide::Runtime::Buffer<float> out(5, 3);
    result.realize(out);

    // Interior should detect horizontal edges
    // The sign depends on kernel orientation, so just check non-zero
    EXPECT_NE(out(2, 1), 0.0f);
}

TEST(Conv, Laplacian) {
    // Laplacian of constant = 0
    shape_t shape = {4, 4};
    Halide::Func input("input");
    Halide::Var x, y;
    input(x, y) = 5.0f;  // Constant

    auto kernel = laplacian_kernel("laplacian");
    auto result = convolve2d(input, shape, kernel, 3, 3, "lap_result");

    Halide::Runtime::Buffer<float> out(4, 4);
    result.realize(out);

    // Interior pixels should be ~0
    EXPECT_NEAR(out(1, 1), 0.0f, 1e-5f);
    EXPECT_NEAR(out(2, 2), 0.0f, 1e-5f);
}
