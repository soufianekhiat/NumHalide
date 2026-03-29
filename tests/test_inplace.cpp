/// @file test_inplace.cpp
/// @brief Tests for in-place operations (inplace.h)

#include <gtest/gtest.h>
#include "numhalide_all.h"
#include "inplace.h"

using namespace numhalide;

// =============================================================================
// Test helpers
// =============================================================================

static Halide::Runtime::Buffer<float> make_1d(const std::vector<float>& vals)
{
    Halide::Runtime::Buffer<float> b((int)vals.size());
    for (int i = 0; i < (int)vals.size(); ++i)
        b(i) = vals[i];
    return b;
}

static Halide::Runtime::Buffer<float> make_2d(int rows, int cols)
{
    // Halide convention: Buffer(cols, rows) so that b(col, row)
    Halide::Runtime::Buffer<float> b(cols, rows);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            b(c, r) = float(r * cols + c + 1);  // 1-indexed values
    return b;
}

// =============================================================================
// Tests
// =============================================================================

// 1. Threshold_1D
TEST(Inplace, Threshold_1D)
{
    // Values spanning above and below 0.5
    auto b = make_1d({0.1f, 0.3f, 0.5f, 0.7f, 0.9f});
    inplace_threshold(b, 0.5f);
    for (int i = 0; i < b.width(); ++i)
        EXPECT_GE(b(i), 0.5f) << "index " << i;
    // Values that were already >= 0.5 should be unchanged
    EXPECT_NEAR(b(2), 0.5f, 1e-6f);
    EXPECT_NEAR(b(3), 0.7f, 1e-6f);
    EXPECT_NEAR(b(4), 0.9f, 1e-6f);
    // Values below 0.5 should have been raised to 0.5
    EXPECT_NEAR(b(0), 0.5f, 1e-6f);
    EXPECT_NEAR(b(1), 0.5f, 1e-6f);
}

// 2. Clamp_1D
TEST(Inplace, Clamp_1D)
{
    auto b = make_1d({0.0f, 0.1f, 0.5f, 0.9f, 1.0f});
    inplace_clamp(b, 0.2f, 0.8f);
    for (int i = 0; i < b.width(); ++i) {
        EXPECT_GE(b(i), 0.2f) << "index " << i;
        EXPECT_LE(b(i), 0.8f) << "index " << i;
    }
    EXPECT_NEAR(b(0), 0.2f, 1e-6f);  // clamped up
    EXPECT_NEAR(b(1), 0.2f, 1e-6f);  // clamped up
    EXPECT_NEAR(b(2), 0.5f, 1e-6f);  // unchanged
    EXPECT_NEAR(b(3), 0.8f, 1e-6f);  // clamped down
    EXPECT_NEAR(b(4), 0.8f, 1e-6f);  // clamped down
}

// 3. Scale_2D
TEST(Inplace, Scale_2D)
{
    // 3 rows x 4 cols, values 1..12
    auto b = make_2d(3, 4);
    // Capture original values
    std::vector<float> orig(12);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 4; ++c)
            orig[r * 4 + c] = b(c, r);

    inplace_scale(b, 2.0f);

    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 4; ++c)
            EXPECT_NEAR(b(c, r), orig[r * 4 + c] * 2.0f, 1e-5f)
                << "row=" << r << " col=" << c;
}

// 4. AddScalar_1D
TEST(Inplace, AddScalar_1D)
{
    auto b = make_1d({1.0f, 2.0f, 3.0f, 4.0f});
    inplace_add_scalar(b, 3.0f);
    EXPECT_NEAR(b(0), 4.0f, 1e-6f);
    EXPECT_NEAR(b(1), 5.0f, 1e-6f);
    EXPECT_NEAR(b(2), 6.0f, 1e-6f);
    EXPECT_NEAR(b(3), 7.0f, 1e-6f);
}

// 5. Exp_1D
TEST(Inplace, Exp_1D)
{
    auto b = make_1d({0.0f, 1.0f, 2.0f});
    inplace_exp(b);
    EXPECT_NEAR(b(0), 1.0f,        1e-4f);
    EXPECT_NEAR(b(1), 2.71828f,    1e-4f);
    EXPECT_NEAR(b(2), 7.38906f,    1e-4f);
}

// 6. Sqrt_1D
TEST(Inplace, Sqrt_1D)
{
    auto b = make_1d({0.0f, 1.0f, 4.0f, 9.0f});
    inplace_sqrt(b);
    EXPECT_NEAR(b(0), 0.0f, 1e-5f);
    EXPECT_NEAR(b(1), 1.0f, 1e-5f);
    EXPECT_NEAR(b(2), 2.0f, 1e-5f);
    EXPECT_NEAR(b(3), 3.0f, 1e-5f);
}

// 7. Gamma_1D
TEST(Inplace, Gamma_1D)
{
    auto b = make_1d({0.0f, 0.5f, 1.0f});
    inplace_gamma(b, 2.0f);
    EXPECT_NEAR(b(0), 0.0f,  1e-5f);
    EXPECT_NEAR(b(1), 0.25f, 1e-5f);  // 0.5^2
    EXPECT_NEAR(b(2), 1.0f,  1e-5f);
}

// 8. Normalize_1D
TEST(Inplace, Normalize_1D)
{
    // [1, 2, 3, 4, 5] -> [0, 0.25, 0.5, 0.75, 1.0]
    auto b = make_1d({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    inplace_normalize(b);
    EXPECT_NEAR(b(0), 0.0f,  1e-5f);
    EXPECT_NEAR(b(1), 0.25f, 1e-5f);
    EXPECT_NEAR(b(2), 0.5f,  1e-5f);
    EXPECT_NEAR(b(3), 0.75f, 1e-5f);
    EXPECT_NEAR(b(4), 1.0f,  1e-5f);
}

// 9. WriteThrough_OriginalModified
TEST(Inplace, WriteThrough_OriginalModified)
{
    // Verify that in-place modifies the same memory (same data pointer)
    auto b = make_1d({1.0f, 2.0f, 3.0f});
    float* ptr_before = b.data();

    inplace_scale(b, 10.0f);

    // Pointer must be unchanged (same allocation)
    EXPECT_EQ(b.data(), ptr_before);

    // Data must have been modified
    EXPECT_NEAR(b(0), 10.0f, 1e-5f);
    EXPECT_NEAR(b(1), 20.0f, 1e-5f);
    EXPECT_NEAR(b(2), 30.0f, 1e-5f);
}

// 10. Chained_Ops_2D
TEST(Inplace, Chained_Ops_2D)
{
    // 4x4 buffer, values 1..16
    // scale(2) -> values 2..32
    // clamp(0, 3) -> values all clamped to [0, 3]
    // add(-1) -> values shifted down by 1, range [-1, 2]
    auto b = make_2d(4, 4);

    inplace_scale(b, 2.0f);
    inplace_clamp(b, 0.0f, 3.0f);
    inplace_add_scalar(b, -1.0f);

    // After scale(2): values are 2, 4, 6, ..., 32
    // After clamp(0, 3): all become 3 except first element which is 2 (2*1=2 <= 3)
    // After add(-1): first element = 2-1=1, rest = 3-1=2
    // b(col=0, row=0) was originally 1 -> *2=2 -> clamp->2 -> -1=1
    EXPECT_NEAR(b(0, 0), 1.0f, 1e-5f);
    // b(col=1, row=0) was originally 2 -> *2=4 -> clamp->3 -> -1=2
    EXPECT_NEAR(b(1, 0), 2.0f, 1e-5f);
    // All remaining values were >= 2 originally, so *2>=4, clamped to 3, then -1=2
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            float val = b(c, r);
            EXPECT_GE(val, -1.0f) << "row=" << r << " col=" << c;
            EXPECT_LE(val,  2.0f) << "row=" << r << " col=" << c;
        }
    }
    // Spot-check a few more
    // b(0,1) originally = 5 -> *2=10 -> clamp->3 -> -1=2
    EXPECT_NEAR(b(0, 1), 2.0f, 1e-5f);
    // b(3,3) originally = 16 -> *2=32 -> clamp->3 -> -1=2
    EXPECT_NEAR(b(3, 3), 2.0f, 1e-5f);
}
