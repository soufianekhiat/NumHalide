/// @file test_reduce_ext2.cpp
/// @brief Tests for multi-axis reductions on a 3D tensor (ReduceMultiAxis3D suite)
///
/// Tensor: shape_t{4, 3, 2} using the same layout convention as test_reduce_ext.cpp.
///
/// NumHalide shape convention (same as test_reduce_ext.cpp):
///   shape_t{D, H, W} = {4, 3, 2} → D=4 (depth, axis 0), H=3 (rows, axis 1), W=2 (cols, axis 2)
///   make_f3d(data, D=4, H=3, W=2) → buf(W,H,D) = buf(2,3,4)
///   data stored row-major: data[d*H*W + r*W + c]
///
/// Using iota data: data[i] = float(i), so data[d*6 + r*2 + c] = 6d + 2r + c.
/// Values span 0..23.
#include <gtest/gtest.h>
#include "numhalide_all.h"
#include <vector>
#include <numeric>
using namespace numhalide;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static const int kD = 4, kH = 3, kW = 2;   // shape_t{4, 3, 2}
static const shape_t kShape{kD, kH, kW};    // extents: [4, 3, 2]

// ---------------------------------------------------------------------------
// Helper: build a buffer-backed Func f(x,y,z) = buf(x,y,z) for a D×H×W tensor.
// Matches the pattern in test_reduce_ext.cpp's make_f3d().
// data is row-major indexed as data[d*H*W + r*W + c].
// ---------------------------------------------------------------------------
static Halide::Func make_f3d_dhw(const std::vector<float>& data,
                                  int D, int H, int W,
                                  const std::string& nm)
{
    Halide::Buffer<float> buf(W, H, D);
    for (int d = 0; d < D; ++d)
        for (int r = 0; r < H; ++r)
            for (int c = 0; c < W; ++c)
                buf(c, r, d) = data[(size_t)(d * H * W + r * W + c)];
    Halide::Func f(nm);
    Halide::Var x("x"), y("y"), z("z");
    f(x, y, z) = buf(x, y, z);
    return f;
}

// Reference accessor: value at (col, row, depth) = data[d*H*W + r*W + c]
static float ref(const std::vector<float>& data, int c, int r, int d) {
    return data[(size_t)(d * kH * kW + r * kW + c)];
}

// ---------------------------------------------------------------------------
// 1. Axis0_3D — reduce axis 0 (D=4) → output shape {H=3, W=2}
//    Mirrors test_reduce_ext.cpp::Sum3D_Axis0 (but axis has extent 4, not 2).
//    result(c, r) = sum_{d=0}^{3} data[d*6 + r*2 + c]
//    Buffer(W, H) = Buffer(2, 3): out(c, r)
// ---------------------------------------------------------------------------
TEST(ReduceMultiAxis3D, Axis0_3D) {
    std::vector<float> data(kD * kH * kW);
    std::iota(data.begin(), data.end(), 0.0f);  // 0..23

    auto f = make_f3d_dhw(data, kD, kH, kW, "ra3d_ax0");
    auto result = reduce_sum(f, kShape, {0}, false, "ra3d_ax0_out");

    Halide::Runtime::Buffer<float> out(kW, kH);  // (2, 3)
    result.realize(out);

    for (int r = 0; r < kH; ++r) {
        for (int c = 0; c < kW; ++c) {
            float expected = 0.0f;
            for (int d = 0; d < kD; ++d)
                expected += ref(data, c, r, d);
            EXPECT_NEAR(out(c, r), expected, 1e-3f) << "c=" << c << " r=" << r;
        }
    }
}

// ---------------------------------------------------------------------------
// 2. Axis1_3D — reduce axis 1 (H=3) → output shape {D=4, W=2}
//    result(c, d) = sum_{r=0}^{2} data[d*6 + r*2 + c]
//    Buffer(W, D) = Buffer(2, 4): out(c, d)
// ---------------------------------------------------------------------------
TEST(ReduceMultiAxis3D, Axis1_3D) {
    std::vector<float> data(kD * kH * kW);
    std::iota(data.begin(), data.end(), 0.0f);

    auto f = make_f3d_dhw(data, kD, kH, kW, "ra3d_ax1");
    auto result = reduce_sum(f, kShape, {1}, false, "ra3d_ax1_out");

    Halide::Runtime::Buffer<float> out(kW, kD);  // (2, 4)
    result.realize(out);

    for (int d = 0; d < kD; ++d) {
        for (int c = 0; c < kW; ++c) {
            float expected = 0.0f;
            for (int r = 0; r < kH; ++r)
                expected += ref(data, c, r, d);
            EXPECT_NEAR(out(c, d), expected, 1e-3f) << "c=" << c << " d=" << d;
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Axis2_3D — reduce axis 2 (W=2) → output shape {D=4, H=3}
//    result(r, d) = sum_{c=0}^{1} data[d*6 + r*2 + c]
//    Buffer(H, D) = Buffer(3, 4): out(r, d)
// ---------------------------------------------------------------------------
TEST(ReduceMultiAxis3D, Axis2_3D) {
    std::vector<float> data(kD * kH * kW);
    std::iota(data.begin(), data.end(), 0.0f);

    auto f = make_f3d_dhw(data, kD, kH, kW, "ra3d_ax2");
    auto result = reduce_sum(f, kShape, {2}, false, "ra3d_ax2_out");

    Halide::Runtime::Buffer<float> out(kH, kD);  // (3, 4)
    result.realize(out);

    for (int d = 0; d < kD; ++d) {
        for (int r = 0; r < kH; ++r) {
            float expected = 0.0f;
            for (int c = 0; c < kW; ++c)
                expected += ref(data, c, r, d);
            EXPECT_NEAR(out(r, d), expected, 1e-3f) << "r=" << r << " d=" << d;
        }
    }
}

// ---------------------------------------------------------------------------
// 4. Axes01_3D — reduce {0, 1} (D and H) → output shape {W=2}
//    result(c) = sum_{d,r} data[d*6 + r*2 + c]
//    Buffer(W) = Buffer(2)
// ---------------------------------------------------------------------------
TEST(ReduceMultiAxis3D, Axes01_3D) {
    std::vector<float> data(kD * kH * kW);
    std::iota(data.begin(), data.end(), 0.0f);

    auto f = make_f3d_dhw(data, kD, kH, kW, "ra3d_ax01");
    auto result = reduce_sum(f, kShape, {0, 1}, false, "ra3d_ax01_out");

    Halide::Runtime::Buffer<float> out(kW);  // (2)
    result.realize(out);

    for (int c = 0; c < kW; ++c) {
        float expected = 0.0f;
        for (int d = 0; d < kD; ++d)
            for (int r = 0; r < kH; ++r)
                expected += ref(data, c, r, d);
        EXPECT_NEAR(out(c), expected, 1e-2f) << "c=" << c;
    }
}

// ---------------------------------------------------------------------------
// 5. Axes02_3D — reduce {0, 2} (D and W) → output shape {H=3}
//    result(r) = sum_{d,c} data[d*6 + r*2 + c]
//    Buffer(H) = Buffer(3)
// ---------------------------------------------------------------------------
TEST(ReduceMultiAxis3D, Axes02_3D) {
    std::vector<float> data(kD * kH * kW);
    std::iota(data.begin(), data.end(), 0.0f);

    auto f = make_f3d_dhw(data, kD, kH, kW, "ra3d_ax02");
    auto result = reduce_sum(f, kShape, {0, 2}, false, "ra3d_ax02_out");

    Halide::Runtime::Buffer<float> out(kH);  // (3)
    result.realize(out);

    for (int r = 0; r < kH; ++r) {
        float expected = 0.0f;
        for (int d = 0; d < kD; ++d)
            for (int c = 0; c < kW; ++c)
                expected += ref(data, c, r, d);
        EXPECT_NEAR(out(r), expected, 1e-2f) << "r=" << r;
    }
}

// ---------------------------------------------------------------------------
// 6. Axes12_3D — reduce {1, 2} (H and W) → output shape {D=4}
//    result(d) = sum_{r,c} data[d*6 + r*2 + c]
//    Buffer(D) = Buffer(4)
// ---------------------------------------------------------------------------
TEST(ReduceMultiAxis3D, Axes12_3D) {
    std::vector<float> data(kD * kH * kW);
    std::iota(data.begin(), data.end(), 0.0f);

    auto f = make_f3d_dhw(data, kD, kH, kW, "ra3d_ax12");
    auto result = reduce_sum(f, kShape, {1, 2}, false, "ra3d_ax12_out");

    Halide::Runtime::Buffer<float> out(kD);  // (4)
    result.realize(out);

    for (int d = 0; d < kD; ++d) {
        float expected = 0.0f;
        for (int r = 0; r < kH; ++r)
            for (int c = 0; c < kW; ++c)
                expected += ref(data, c, r, d);
        EXPECT_NEAR(out(d), expected, 1e-2f) << "d=" << d;
    }
}

// ---------------------------------------------------------------------------
// 7. AllAxes_3D — reduce {0, 1, 2} → scalar wrapped in Buffer(1)
//    total = sum(0..23) = 276
// ---------------------------------------------------------------------------
TEST(ReduceMultiAxis3D, AllAxes_3D) {
    std::vector<float> data(kD * kH * kW);
    std::iota(data.begin(), data.end(), 0.0f);

    auto f = make_f3d_dhw(data, kD, kH, kW, "ra3d_all");
    auto result = reduce_sum(f, kShape, {0, 1, 2}, false, "ra3d_all_out");

    Halide::Runtime::Buffer<float> out(1);
    result.realize(out);

    float expected = 0.0f;
    for (float v : data) expected += v;  // 0+1+...+23 = 276
    EXPECT_NEAR(out(0), expected, 0.5f);
}

// ---------------------------------------------------------------------------
// 8. Keepdims_Axis0 — reduce axis 0 (D=4), keepdims=true → shape {1, 3, 2}
//    Buffer(W, H, 1) = Buffer(2, 3, 1): out(c, r, 0)
// ---------------------------------------------------------------------------
TEST(ReduceMultiAxis3D, Keepdims_Axis0) {
    std::vector<float> data(kD * kH * kW);
    std::iota(data.begin(), data.end(), 0.0f);

    auto f = make_f3d_dhw(data, kD, kH, kW, "ra3d_kd0");
    auto result = reduce_sum(f, kShape, {0}, true, "ra3d_kd0_out");

    // Shape {1, 3, 2}: reduced D-axis kept as 1
    Halide::Runtime::Buffer<float> out(kW, kH, 1);  // (2, 3, 1)
    result.realize(out);

    for (int r = 0; r < kH; ++r) {
        for (int c = 0; c < kW; ++c) {
            float expected = 0.0f;
            for (int d = 0; d < kD; ++d)
                expected += ref(data, c, r, d);
            EXPECT_NEAR(out(c, r, 0), expected, 1e-3f) << "c=" << c << " r=" << r;
        }
    }
}

// ---------------------------------------------------------------------------
// 9. Keepdims_Axes01 — reduce {0, 1}, keepdims=true → shape {1, 1, 2}
//    Buffer(W, 1, 1) = Buffer(2, 1, 1): out(c, 0, 0)
// ---------------------------------------------------------------------------
TEST(ReduceMultiAxis3D, Keepdims_Axes01) {
    std::vector<float> data(kD * kH * kW);
    std::iota(data.begin(), data.end(), 0.0f);

    auto f = make_f3d_dhw(data, kD, kH, kW, "ra3d_kd01");
    auto result = reduce_sum(f, kShape, {0, 1}, true, "ra3d_kd01_out");

    Halide::Runtime::Buffer<float> out(kW, 1, 1);  // (2, 1, 1)
    result.realize(out);

    for (int c = 0; c < kW; ++c) {
        float expected = 0.0f;
        for (int d = 0; d < kD; ++d)
            for (int r = 0; r < kH; ++r)
                expected += ref(data, c, r, d);
        EXPECT_NEAR(out(c, 0, 0), expected, 1e-2f) << "c=" << c;
    }
}

// ---------------------------------------------------------------------------
// 10. ReduceMean_MultiAxis — reduce_mean along {0, 2} (D and W) → shape {H=3}
//     mean[r] = (sum_{d,c} data[d*6 + r*2 + c]) / (D*W)
//     Buffer(H) = Buffer(3): out(r)
// ---------------------------------------------------------------------------
TEST(ReduceMultiAxis3D, ReduceMean_MultiAxis) {
    std::vector<float> data(kD * kH * kW);
    std::iota(data.begin(), data.end(), 0.0f);

    auto f = make_f3d_dhw(data, kD, kH, kW, "ra3d_mean02");
    auto result = reduce_mean(f, kShape, {0, 2}, false, "ra3d_mean02_out");

    Halide::Runtime::Buffer<float> out(kH);  // (3)
    result.realize(out);

    const float count = static_cast<float>(kD * kW);  // 4 * 2 = 8
    for (int r = 0; r < kH; ++r) {
        float sum = 0.0f;
        for (int d = 0; d < kD; ++d)
            for (int c = 0; c < kW; ++c)
                sum += ref(data, c, r, d);
        float expected = sum / count;
        EXPECT_NEAR(out(r), expected, 1e-4f) << "r=" << r;
    }
}
