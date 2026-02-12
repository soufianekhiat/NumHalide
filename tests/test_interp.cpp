/// @file test_interp.cpp
/// @brief Tests for interpolation operations

#include <gtest/gtest.h>
#include "numhalide_all.h"
#include <cmath>

using namespace numhalide;

// -----------------------------------------------------------------------------
// 1D Interpolation Tests
// -----------------------------------------------------------------------------

TEST(Interp, Interp1DUniform) {
    // Double the size with interpolation
    shape_t shape = {3};
    Halide::Func values("values");
    Halide::Var x;
    // [0, 2, 4]
    values(x) = Halide::cast<float>(x * 2);

    auto result = interp1d_uniform(values, shape, 2.0f, "interp_result");

    Halide::Runtime::Buffer<float> out(5);
    result.realize(out);

    // [0, 1, 2, 3, 4] approximately
    EXPECT_NEAR(out(0), 0.0f, 1e-5f);
    EXPECT_NEAR(out(2), 2.0f, 1e-5f);
    EXPECT_NEAR(out(4), 4.0f, 1e-5f);
}

TEST(Interp, Interp1DEndpoints) {
    shape_t shape = {2};
    Halide::Func values("values");
    Halide::Var x;
    // [0, 10]
    values(x) = Halide::cast<float>(x * 10);

    auto result = interp1d_uniform(values, shape, 2.0f, "interp_end");

    Halide::Runtime::Buffer<float> out(3);
    result.realize(out);

    EXPECT_NEAR(out(0), 0.0f, 1e-5f);   // Start
    EXPECT_NEAR(out(2), 10.0f, 1e-5f);  // End
    // Middle should be approximately 5
    EXPECT_GT(out(1), 0.0f);
    EXPECT_LT(out(1), 10.0f);
}

// -----------------------------------------------------------------------------
// 2D Resize Tests
// -----------------------------------------------------------------------------

TEST(Interp, ResizeBilinear2x) {
    // 2x upscale
    shape_t shape = {2, 2};
    Halide::Func input("input");
    Halide::Var x, y;
    // [[0, 10], [20, 30]]
    input(x, y) = Halide::cast<float>((y * 2 + x) * 10);

    auto result = resize_bilinear(input, shape, 3, 3, "resize_2x");

    Halide::Runtime::Buffer<float> out(3, 3);
    result.realize(out);

    // Corners should match
    EXPECT_NEAR(out(0, 0), 0.0f, 1e-5f);
    EXPECT_NEAR(out(2, 0), 10.0f, 1e-5f);
    EXPECT_NEAR(out(0, 2), 20.0f, 1e-5f);
    EXPECT_NEAR(out(2, 2), 30.0f, 1e-5f);

    // Center should be average
    EXPECT_NEAR(out(1, 1), 15.0f, 1e-5f);
}

TEST(Interp, ResizeNearest) {
    shape_t shape = {2, 2};
    Halide::Func input("input");
    Halide::Var x, y;
    input(x, y) = y * 2 + x;  // [[0, 1], [2, 3]]

    auto result = resize_nearest(input, shape, 4, 4, "resize_nn");

    Halide::Runtime::Buffer<int32_t> out(4, 4);
    result.realize(out);

    // Each 2x2 block in output should have same value
    EXPECT_EQ(out(0, 0), out(1, 1));  // Should be same quadrant
}

TEST(Interp, ZoomUp) {
    shape_t shape = {2, 2};
    Halide::Func input("input");
    Halide::Var x, y;
    input(x, y) = Halide::cast<float>(y * 2 + x);

    auto result = zoom(input, shape, 2.0f, "zoom_up");
    auto out_shape = infer_zoom(shape, 2.0f);

    EXPECT_EQ(out_shape.extents[0], 4);
    EXPECT_EQ(out_shape.extents[1], 4);

    Halide::Runtime::Buffer<float> out(4, 4);
    result.realize(out);

    // Corners should preserve values
    EXPECT_NEAR(out(0, 0), 0.0f, 1e-5f);
}

TEST(Interp, ZoomDown) {
    shape_t shape = {4, 4};
    Halide::Func input("input");
    Halide::Var x, y;
    input(x, y) = Halide::cast<float>(y * 4 + x);

    auto result = zoom(input, shape, 0.5f, "zoom_down");
    auto out_shape = infer_zoom(shape, 0.5f);

    EXPECT_EQ(out_shape.extents[0], 2);
    EXPECT_EQ(out_shape.extents[1], 2);

    Halide::Runtime::Buffer<float> out(2, 2);
    result.realize(out);

    // Check that values are reasonable
    EXPECT_GE(out(0, 0), 0.0f);
    EXPECT_LE(out(1, 1), 15.0f);
}

// -----------------------------------------------------------------------------
// Map Coordinates Tests
// -----------------------------------------------------------------------------

TEST(Interp, MapCoordinatesIdentity) {
    // Identity mapping
    shape_t shape = {4, 4};
    Halide::Func input("input");
    Halide::Var x, y;
    input(x, y) = Halide::cast<float>(y * 4 + x);

    Halide::Func coords_x("coords_x"), coords_y("coords_y");
    coords_x(x, y) = Halide::cast<float>(x);
    coords_y(x, y) = Halide::cast<float>(y);

    auto result = map_coordinates(input, shape, coords_x, coords_y, "map_id");

    Halide::Runtime::Buffer<float> out(4, 4);
    result.realize(out);

    // Should be same as input
    EXPECT_NEAR(out(0, 0), 0.0f, 1e-5f);
    EXPECT_NEAR(out(1, 1), 5.0f, 1e-5f);
    EXPECT_NEAR(out(2, 2), 10.0f, 1e-5f);
    EXPECT_NEAR(out(3, 3), 15.0f, 1e-5f);
}

TEST(Interp, MapCoordinatesShift) {
    // Shift by 0.5 pixels
    shape_t shape = {4, 4};
    Halide::Func input("input");
    Halide::Var x, y;
    input(x, y) = Halide::cast<float>(y * 4 + x);

    Halide::Func coords_x("coords_x"), coords_y("coords_y");
    coords_x(x, y) = Halide::cast<float>(x) + 0.5f;
    coords_y(x, y) = Halide::cast<float>(y) + 0.5f;

    auto result = map_coordinates(input, shape, coords_x, coords_y, "map_shift");

    Halide::Runtime::Buffer<float> out(3, 3);
    result.realize(out);

    // Center pixel should be interpolated between 4 values
    // (0,0) samples at (0.5, 0.5) which interpolates [0,1,4,5]
    // Expected: 0.25*(0+1+4+5) = 2.5
    EXPECT_NEAR(out(0, 0), 2.5f, 1e-5f);
}

TEST(Interp, MapCoordinatesSwirl) {
    // Simple rotation test
    shape_t shape = {8, 8};
    Halide::Func input("input");
    Halide::Var x, y;
    input(x, y) = Halide::cast<float>(x);  // Horizontal gradient

    // Small rotation around center
    float cx = 3.5f, cy = 3.5f;
    float angle = 0.1f;  // Small angle

    Halide::Func coords_x("coords_x"), coords_y("coords_y");
    Halide::Expr dx = Halide::cast<float>(x) - cx;
    Halide::Expr dy = Halide::cast<float>(y) - cy;
    coords_x(x, y) = cx + dx * std::cos(angle) - dy * std::sin(angle);
    coords_y(x, y) = cy + dx * std::sin(angle) + dy * std::cos(angle);

    auto result = map_coordinates(input, shape, coords_x, coords_y, "map_swirl");

    Halide::Runtime::Buffer<float> out(8, 8);
    result.realize(out);

    // Center should be approximately unchanged
    EXPECT_NEAR(out(4, 4), 4.0f, 0.5f);
}
