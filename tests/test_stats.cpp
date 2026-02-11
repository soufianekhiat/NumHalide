/// @file test_stats.cpp
/// @brief Tests for statistical operations

#include <gtest/gtest.h>
#include "numhalide_all.h"
#include <cmath>

using namespace numhalide;

// -----------------------------------------------------------------------------
// Variance Tests
// -----------------------------------------------------------------------------

TEST(Stats, Var1D) {
    // Create array [1, 2, 3, 4, 5]
    // Mean = 3, Variance = ((1-3)^2 + (2-3)^2 + (3-3)^2 + (4-3)^2 + (5-3)^2) / 5
    //                    = (4 + 1 + 0 + 1 + 4) / 5 = 10 / 5 = 2.0
    shape_t shape = {5};
    Halide::Func f("input");
    Halide::Var x;
    f(x) = Halide::cast<float>(x + 1);  // [1, 2, 3, 4, 5]

    auto result = stats::var(f, shape, "var_result");

    Halide::Runtime::Buffer<float> out(1);
    result.realize(out);

    EXPECT_NEAR(out(0), 2.0f, 1e-5f);
}

TEST(Stats, VarSampleDdof1) {
    // Same array but with ddof=1 (sample variance)
    // Sample variance = 10 / (5-1) = 10 / 4 = 2.5
    shape_t shape = {5};
    Halide::Func f("input");
    Halide::Var x;
    f(x) = Halide::cast<float>(x + 1);

    // Use var_full for explicit ddof control
    auto result = stats::var_full(f, shape, 1, "var_sample");

    Halide::Runtime::Buffer<float> out(1);
    result.realize(out);

    EXPECT_NEAR(out(0), 2.5f, 1e-5f);
}

TEST(Stats, Var2DAxis0) {
    // 2D array: [[1, 2], [3, 4], [5, 6]]
    // shape = {3, 2} (3 rows, 2 cols)
    // Variance along axis 0 (rows):
    //   col 0: [1, 3, 5] -> mean=3, var=((1-3)^2+(3-3)^2+(5-3)^2)/3 = 8/3
    //   col 1: [2, 4, 6] -> mean=4, var=((2-4)^2+(4-4)^2+(6-4)^2)/3 = 8/3
    shape_t shape = {3, 2};
    Halide::Func f("input");
    Halide::Var x, y;
    f(x, y) = Halide::cast<float>(y * 2 + x + 1);

    auto result = stats::var(f, shape, 0, false, 0, "var_axis0");

    Halide::Runtime::Buffer<float> out(2);
    result.realize(out);

    float expected = 8.0f / 3.0f;
    EXPECT_NEAR(out(0), expected, 1e-5f);
    EXPECT_NEAR(out(1), expected, 1e-5f);
}

TEST(Stats, Var2DAxis1) {
    // 2D array: [[1, 2], [3, 4], [5, 6]]
    // Variance along axis 1 (cols):
    //   row 0: [1, 2] -> mean=1.5, var=((1-1.5)^2+(2-1.5)^2)/2 = 0.5/2 = 0.25
    //   row 1: [3, 4] -> mean=3.5, var=0.25
    //   row 2: [5, 6] -> mean=5.5, var=0.25
    shape_t shape = {3, 2};
    Halide::Func f("input");
    Halide::Var x, y;
    f(x, y) = Halide::cast<float>(y * 2 + x + 1);

    auto result = stats::var(f, shape, 1, false, 0, "var_axis1");

    Halide::Runtime::Buffer<float> out(3);
    result.realize(out);

    EXPECT_NEAR(out(0), 0.25f, 1e-5f);
    EXPECT_NEAR(out(1), 0.25f, 1e-5f);
    EXPECT_NEAR(out(2), 0.25f, 1e-5f);
}

TEST(Stats, VarConstant) {
    // Constant array should have variance = 0
    shape_t shape = {4, 4};
    Halide::Func f("constant");
    Halide::Var x, y;
    f(x, y) = 5.0f;

    auto result = stats::var(f, shape, "var_constant");

    Halide::Runtime::Buffer<float> out(1);
    result.realize(out);

    EXPECT_NEAR(out(0), 0.0f, 1e-5f);
}

// -----------------------------------------------------------------------------
// Standard Deviation Tests
// -----------------------------------------------------------------------------

TEST(Stats, Std1D) {
    // Same as Var1D but take sqrt
    // Variance = 2.0, Std = sqrt(2.0) ≈ 1.414
    shape_t shape = {5};
    Halide::Func f("input");
    Halide::Var x;
    f(x) = Halide::cast<float>(x + 1);

    auto result = stats::std(f, shape, "std_result");

    Halide::Runtime::Buffer<float> out(1);
    result.realize(out);

    EXPECT_NEAR(out(0), std::sqrt(2.0f), 1e-5f);
}

TEST(Stats, StdSampleDdof1) {
    // Sample std with ddof=1
    // Sample variance = 2.5, Std = sqrt(2.5)
    shape_t shape = {5};
    Halide::Func f("input");
    Halide::Var x;
    f(x) = Halide::cast<float>(x + 1);

    // Use std_full for explicit ddof control
    auto result = stats::std_full(f, shape, 1, "std_sample");

    Halide::Runtime::Buffer<float> out(1);
    result.realize(out);

    EXPECT_NEAR(out(0), std::sqrt(2.5f), 1e-5f);
}

TEST(Stats, Std2DAxis0) {
    // Variance along axis 0 = 8/3, Std = sqrt(8/3)
    shape_t shape = {3, 2};
    Halide::Func f("input");
    Halide::Var x, y;
    f(x, y) = Halide::cast<float>(y * 2 + x + 1);

    auto result = stats::std(f, shape, 0, false, 0, "std_axis0");

    Halide::Runtime::Buffer<float> out(2);
    result.realize(out);

    float expected = std::sqrt(8.0f / 3.0f);
    EXPECT_NEAR(out(0), expected, 1e-5f);
    EXPECT_NEAR(out(1), expected, 1e-5f);
}

TEST(Stats, StdConstant) {
    // Constant array should have std = 0
    shape_t shape = {4, 4};
    Halide::Func f("constant");
    Halide::Var x, y;
    f(x, y) = 5.0f;

    auto result = stats::std(f, shape, "std_constant");

    Halide::Runtime::Buffer<float> out(1);
    result.realize(out);

    EXPECT_NEAR(out(0), 0.0f, 1e-5f);
}

TEST(Stats, VarStdRelation) {
    // Verify var = std^2
    shape_t shape = {10};
    Halide::Func f("input");
    Halide::Var x;
    f(x) = Halide::cast<float>(x * x);  // [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

    auto var_result = stats::var(f, shape, "var_check");
    auto std_result = stats::std(f, shape, "std_check");

    Halide::Runtime::Buffer<float> var_out(1);
    Halide::Runtime::Buffer<float> std_out(1);
    var_result.realize(var_out);
    std_result.realize(std_out);

    EXPECT_NEAR(var_out(0), std_out(0) * std_out(0), 1e-4f);
}

TEST(Stats, VarKeepdims) {
    // Test keepdims=true
    shape_t shape = {3, 4};
    Halide::Func f("input");
    Halide::Var x, y;
    f(x, y) = Halide::cast<float>(y * 4 + x);

    std::vector<int> axes = {0};
    auto result = stats::var(f, shape, axes, true, 0, "var_keepdims");

    // Output should be shape {1, 4}
    Halide::Runtime::Buffer<float> out(4, 1);
    result.realize(out);

    // Just verify it computes without error and produces reasonable values
    for (int i = 0; i < 4; ++i) {
        EXPECT_GE(out(i, 0), 0.0f);  // Variance is non-negative
    }
}

TEST(Stats, StdLargeValues) {
    // Test with larger values to check numerical stability
    // For uniform integers [0, n-1], variance = (n^2 - 1) / 12
    // For [1000, 1001, ..., 1099] (n=100), variance = (100^2 - 1) / 12 = 9999/12 = 833.25
    shape_t shape = {100};
    Halide::Func f("input");
    Halide::Var x;
    f(x) = Halide::cast<float>(x + 1000);  // [1000, 1001, ..., 1099]

    auto result = stats::std(f, shape, "std_large");

    Halide::Runtime::Buffer<float> out(1);
    result.realize(out);

    // Std = sqrt(833.25) ≈ 28.866
    float expected_var = (100.0f * 100.0f - 1.0f) / 12.0f;
    float expected_std = std::sqrt(expected_var);
    EXPECT_NEAR(out(0), expected_std, 0.5f);
}
