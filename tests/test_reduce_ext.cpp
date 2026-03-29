/// @file test_reduce_ext.cpp
/// @brief Tests for multi-axis reductions (T2-C)
#include <gtest/gtest.h>
#include "numhalide_all.h"
#include <vector>
#include <numeric>
#include <algorithm>
using namespace numhalide;

// Helper: make 2D float Func from row-major data
static Halide::Func make_f2d(const std::vector<float>& data, int rows, int cols, const std::string& nm) {
    Halide::Buffer<float> buf(cols, rows);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            buf(c, r) = data[(size_t)(r * cols + c)];
    Halide::Func f(nm); Halide::Var x, y;
    f(x, y) = buf(x, y);
    return f;
}

// Helper: make 3D float Func from data[d][r][c]
static Halide::Func make_f3d(const std::vector<float>& data, int D, int H, int W, const std::string& nm) {
    Halide::Buffer<float> buf(W, H, D);
    for (int d = 0; d < D; ++d)
        for (int r = 0; r < H; ++r)
            for (int c = 0; c < W; ++c)
                buf(c, r, d) = data[(size_t)(d * H * W + r * W + c)];
    Halide::Func f(nm); Halide::Var x, y, z;
    f(x, y, z) = buf(x, y, z);
    return f;
}

// 1. reduce_sum 2D axes {0,1} (both) -> scalar
// Note: reduce_sum(f, shape, {0,1}, false) produces a rank-0 output (keepdims=false, all axes reduced).
// The multi-axis overload with keepdims=false and all axes reduced gives a 0-dim Func.
// We realize as a 1-element buffer and access out(0).
TEST(ReduceExt, Sum2D_BothAxes) {
    // 3x4 matrix, values 1..12
    std::vector<float> data(12);
    std::iota(data.begin(), data.end(), 1.0f);
    auto f = make_f2d(data, 3, 4, "rs2d_01");
    auto result = reduce_sum(f, {3, 4}, {0, 1}, false, "rs2d_01_out");
    // sum = 1+2+...+12 = 78
    // Rank-0 Func realized as 1-element 1D buffer
    Halide::Runtime::Buffer<float> out(1);
    result.realize(out);
    EXPECT_NEAR(out(0), 78.0f, 1e-3f);
}

// 2. reduce_sum 2D axes {0,1} keepdims -> shape {1,1}
TEST(ReduceExt, Sum2D_BothAxes_Keepdims) {
    std::vector<float> data(12);
    std::iota(data.begin(), data.end(), 1.0f);
    auto f = make_f2d(data, 3, 4, "rs2d_01kd");
    auto result = reduce_sum(f, {3, 4}, {0, 1}, true, "rs2d_01kd_out");
    Halide::Runtime::Buffer<float> out(1, 1);  // {1,1}
    result.realize(out);
    EXPECT_NEAR(out(0, 0), 78.0f, 1e-3f);
}

// 3. reduce_min 2D axes {0,1} -> minimum element
TEST(ReduceExt, Min2D_BothAxes) {
    std::vector<float> data = {3,1,4,1, 5,9,2,6, 5,3,5,8};
    auto f = make_f2d(data, 3, 4, "rm2d_01");
    auto result = reduce_min(f, {3, 4}, {0, 1}, false, "rm2d_01_out");
    Halide::Runtime::Buffer<float> out(1);
    result.realize(out);
    EXPECT_NEAR(out(0), 1.0f, 1e-5f);
}

// 4. reduce_max 2D axes {0,1} -> maximum element
TEST(ReduceExt, Max2D_BothAxes) {
    std::vector<float> data = {3,1,4,1, 5,9,2,6, 5,3,5,8};
    auto f = make_f2d(data, 3, 4, "rx2d_01");
    auto result = reduce_max(f, {3, 4}, {0, 1}, false, "rx2d_01_out");
    Halide::Runtime::Buffer<float> out(1);
    result.realize(out);
    EXPECT_NEAR(out(0), 9.0f, 1e-5f);
}

// 5. reduce_mean 2D axes {0,1} -> mean of all elements
TEST(ReduceExt, Mean2D_BothAxes) {
    std::vector<float> data(12);
    std::iota(data.begin(), data.end(), 1.0f);
    auto f = make_f2d(data, 3, 4, "rmean2d_01");
    auto result = reduce_mean(f, {3, 4}, {0, 1}, false, "rmean2d_01_out");
    Halide::Runtime::Buffer<float> out(1);
    result.realize(out);
    EXPECT_NEAR(out(0), 6.5f, 1e-3f);  // mean of 1..12 = 6.5
}

// 6. reduce_sum 3D axis 0 (reduce depth D) -> shape {H, W} = {3, 4}
TEST(ReduceExt, Sum3D_Axis0) {
    // D=2, H=3, W=4
    // Data: slice 0 = 1..12, slice 1 = 13..24
    std::vector<float> data(24);
    std::iota(data.begin(), data.end(), 1.0f);
    auto f = make_f3d(data, 2, 3, 4, "rs3d_0");
    auto result = reduce_sum(f, {2, 3, 4}, {0}, false, "rs3d_0_out");
    // result(x, y) = f(x, y, 0) + f(x, y, 1) = data[r*4+c] + data[12 + r*4+c]
    Halide::Runtime::Buffer<float> out(4, 3);
    result.realize(out);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 4; ++c) {
            float expected = data[(size_t)(r * 4 + c)] + data[(size_t)(12 + r * 4 + c)];
            EXPECT_NEAR(out(c, r), expected, 1e-4f) << "(" << c << "," << r << ")";
        }
}

// 7. reduce_sum 3D axis 1 (reduce rows H) -> shape {D, W} = {2, 4}
TEST(ReduceExt, Sum3D_Axis1) {
    std::vector<float> data(24);
    std::iota(data.begin(), data.end(), 1.0f);
    auto f = make_f3d(data, 2, 3, 4, "rs3d_1");
    auto result = reduce_sum(f, {2, 3, 4}, {1}, false, "rs3d_1_out");
    Halide::Runtime::Buffer<float> out(4, 2);  // (W, D)
    result.realize(out);
    for (int d = 0; d < 2; ++d)
        for (int c = 0; c < 4; ++c) {
            float expected = 0.0f;
            for (int r = 0; r < 3; ++r)
                expected += data[(size_t)(d * 3 * 4 + r * 4 + c)];
            EXPECT_NEAR(out(c, d), expected, 1e-4f) << "d=" << d << " c=" << c;
        }
}

// 8. reduce_sum 3D axis 2 (reduce cols W) -> shape {D, H} = {2, 3}
TEST(ReduceExt, Sum3D_Axis2) {
    std::vector<float> data(24);
    std::iota(data.begin(), data.end(), 1.0f);
    auto f = make_f3d(data, 2, 3, 4, "rs3d_2");
    auto result = reduce_sum(f, {2, 3, 4}, {2}, false, "rs3d_2_out");
    Halide::Runtime::Buffer<float> out(3, 2);  // (H, D)
    result.realize(out);
    for (int d = 0; d < 2; ++d)
        for (int r = 0; r < 3; ++r) {
            float expected = 0.0f;
            for (int c = 0; c < 4; ++c)
                expected += data[(size_t)(d * 12 + r * 4 + c)];
            EXPECT_NEAR(out(r, d), expected, 1e-4f) << "d=" << d << " r=" << r;
        }
}

// 9. reduce_sum 3D axes {0, 2} (reduce D and W) -> shape {H} = {3}
TEST(ReduceExt, Sum3D_Axes02) {
    std::vector<float> data(24);
    std::iota(data.begin(), data.end(), 1.0f);
    auto f = make_f3d(data, 2, 3, 4, "rs3d_02");
    auto result = reduce_sum(f, {2, 3, 4}, {0, 2}, false, "rs3d_02_out");
    Halide::Runtime::Buffer<float> out(3);  // {H}
    result.realize(out);
    for (int r = 0; r < 3; ++r) {
        float expected = 0.0f;
        for (int d = 0; d < 2; ++d)
            for (int c = 0; c < 4; ++c)
                expected += data[(size_t)(d * 12 + r * 4 + c)];
        EXPECT_NEAR(out(r), expected, 1e-3f) << "r=" << r;
    }
}

// 10. reduce_sum 3D all axes {0,1,2} -> scalar
TEST(ReduceExt, Sum3D_AllAxes) {
    std::vector<float> data(24);
    std::iota(data.begin(), data.end(), 1.0f);
    auto f = make_f3d(data, 2, 3, 4, "rs3d_012");
    auto result = reduce_sum(f, {2, 3, 4}, {0, 1, 2}, false, "rs3d_012_out");
    // sum 1..24 = 300
    Halide::Runtime::Buffer<float> out(1);
    result.realize(out);
    EXPECT_NEAR(out(0), 300.0f, 1e-2f);
}

// 11. reduce_sum 3D axes {0,1} keepdims -> shape {1, 1, W} = {1,1,4}
TEST(ReduceExt, Sum3D_Axes01_Keepdims) {
    std::vector<float> data(24);
    std::iota(data.begin(), data.end(), 1.0f);
    auto f = make_f3d(data, 2, 3, 4, "rs3d_01kd");
    auto result = reduce_sum(f, {2, 3, 4}, {0, 1}, true, "rs3d_01kd_out");
    // Shape {1, 1, 4}: sum over D and H for each col
    Halide::Runtime::Buffer<float> out(4, 1, 1);  // (W, 1, 1)
    result.realize(out);
    for (int c = 0; c < 4; ++c) {
        float expected = 0.0f;
        for (int d = 0; d < 2; ++d)
            for (int r = 0; r < 3; ++r)
                expected += data[(size_t)(d * 12 + r * 4 + c)];
        EXPECT_NEAR(out(c, 0, 0), expected, 1e-3f) << "c=" << c;
    }
}

// 12. reduce_mean 3D axis 2 -> mean along cols
TEST(ReduceExt, Mean3D_Axis2) {
    std::vector<float> data(24);
    std::iota(data.begin(), data.end(), 1.0f);
    auto f = make_f3d(data, 2, 3, 4, "rmean3d_2");
    auto result = reduce_mean(f, {2, 3, 4}, {2}, false, "rmean3d_2_out");
    Halide::Runtime::Buffer<float> out(3, 2);  // (H, D)
    result.realize(out);
    for (int d = 0; d < 2; ++d)
        for (int r = 0; r < 3; ++r) {
            float expected = 0.0f;
            for (int c = 0; c < 4; ++c)
                expected += data[(size_t)(d * 12 + r * 4 + c)];
            expected /= 4.0f;
            EXPECT_NEAR(out(r, d), expected, 1e-4f) << "d=" << d << " r=" << r;
        }
}
