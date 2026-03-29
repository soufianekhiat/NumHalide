/// @file test_sort_ext.cpp
/// @brief Tests for sort_1d_fast, argsort_1d_fast, sort_2d, argsort_2d

#include <gtest/gtest.h>
#include "numhalide_all.h"
#include <cmath>
#include <vector>
#include <algorithm>

using namespace numhalide;

// Helper: build a 1D Halide::Func from a float vector
static Halide::Func make_1d_float(const std::vector<float>& vals, const std::string& name)
{
    Halide::Buffer<float> buf((int)vals.size());
    for (int i = 0; i < (int)vals.size(); ++i) buf(i) = vals[i];
    Halide::Func f(name);
    Halide::Var x;
    f(x) = buf(x);
    return f;
}

// Helper: 4x4 test matrix:
//   row 0: [3,1,4,2]
//   row 1: [7,5,8,6]
//   row 2: [2,9,1,4]
//   row 3: [6,3,7,5]
static Halide::Func make_4x4_matrix(const std::string& name)
{
    Halide::Func f(name);
    Halide::Var x("x"), y("y");
    f(x, y) = Halide::cast<float>(
        Halide::select(
            y == 0, Halide::select(x == 0, 3, Halide::select(x == 1, 1, Halide::select(x == 2, 4, 2))),
            y == 1, Halide::select(x == 0, 7, Halide::select(x == 1, 5, Halide::select(x == 2, 8, 6))),
            y == 2, Halide::select(x == 0, 2, Halide::select(x == 1, 9, Halide::select(x == 2, 1, 4))),
                    Halide::select(x == 0, 6, Halide::select(x == 1, 3, Halide::select(x == 2, 7, 5)))));
    return f;
}

// -----------------------------------------------------------------------------
// Test 1: Sort1DFastPow2 — sort_1d_fast([3,1,4,2], 4) = [1,2,3,4]
// -----------------------------------------------------------------------------

TEST(SortExt, Sort1DFastPow2) {
    auto f = make_1d_float({3.0f, 1.0f, 4.0f, 2.0f}, "s1fp");
    auto sorted = sort_1d_fast(f, 4, true, "s1fp_out");
    Halide::Runtime::Buffer<float> out(4);
    sorted.realize(out);

    EXPECT_NEAR(out(0), 1.0f, 1e-5f);
    EXPECT_NEAR(out(1), 2.0f, 1e-5f);
    EXPECT_NEAR(out(2), 3.0f, 1e-5f);
    EXPECT_NEAR(out(3), 4.0f, 1e-5f);
}

// -----------------------------------------------------------------------------
// Test 2: Sort1DFastNonPow2 — sort_1d_fast([3,1,4,1,5,9], 6) = [1,1,3,4,5,9]
// -----------------------------------------------------------------------------

TEST(SortExt, Sort1DFastNonPow2) {
    auto f = make_1d_float({3.0f, 1.0f, 4.0f, 1.0f, 5.0f, 9.0f}, "s1fnp");
    auto sorted = sort_1d_fast(f, 6, true, "s1fnp_out");
    Halide::Runtime::Buffer<float> out(6);
    sorted.realize(out);

    EXPECT_NEAR(out(0), 1.0f, 1e-5f);
    EXPECT_NEAR(out(1), 1.0f, 1e-5f);
    EXPECT_NEAR(out(2), 3.0f, 1e-5f);
    EXPECT_NEAR(out(3), 4.0f, 1e-5f);
    EXPECT_NEAR(out(4), 5.0f, 1e-5f);
    EXPECT_NEAR(out(5), 9.0f, 1e-5f);
}

// -----------------------------------------------------------------------------
// Test 3: Sort1DFastDescending — sort_1d_fast([3,1,4,2], 4, false) = [4,3,2,1]
// -----------------------------------------------------------------------------

TEST(SortExt, Sort1DFastDescending) {
    auto f = make_1d_float({3.0f, 1.0f, 4.0f, 2.0f}, "s1fd");
    auto sorted = sort_1d_fast(f, 4, false, "s1fd_out");
    Halide::Runtime::Buffer<float> out(4);
    sorted.realize(out);

    EXPECT_NEAR(out(0), 4.0f, 1e-5f);
    EXPECT_NEAR(out(1), 3.0f, 1e-5f);
    EXPECT_NEAR(out(2), 2.0f, 1e-5f);
    EXPECT_NEAR(out(3), 1.0f, 1e-5f);
}

// -----------------------------------------------------------------------------
// Test 4: Argsort1DFast — argsort_1d_fast([3.0,1.0,4.0,2.0], 4) -> [1,3,0,2]
// -----------------------------------------------------------------------------

TEST(SortExt, Argsort1DFast) {
    // [3,1,4,2]: sorted order is 1<2<3<4 at indices 1,3,0,2
    auto f = make_1d_float({3.0f, 1.0f, 4.0f, 2.0f}, "as1f");
    auto indices = argsort_1d_fast(f, 4, true, "as1f_out");
    Halide::Runtime::Buffer<int32_t> out(4);
    indices.realize(out);

    // argsort ascending: out[0]=idx of smallest, ..., out[3]=idx of largest
    EXPECT_EQ(out(0), 1);  // value 1.0 is at index 1
    EXPECT_EQ(out(1), 3);  // value 2.0 is at index 3
    EXPECT_EQ(out(2), 0);  // value 3.0 is at index 0
    EXPECT_EQ(out(3), 2);  // value 4.0 is at index 2
}

// -----------------------------------------------------------------------------
// Test 5: Sort2DAxis1 — sort each row ascending
// -----------------------------------------------------------------------------

TEST(SortExt, Sort2DAxis1) {
    auto f = make_4x4_matrix("s2d_ax1");
    auto sorted = sort_2d(f, 4, 4, 1, true, "s2d_ax1_out");
    Halide::Runtime::Buffer<float> out(4, 4);
    sorted.realize(out);

    // Row 0: [3,1,4,2] -> [1,2,3,4]
    EXPECT_NEAR(out(0, 0), 1.0f, 1e-5f);
    EXPECT_NEAR(out(1, 0), 2.0f, 1e-5f);
    EXPECT_NEAR(out(2, 0), 3.0f, 1e-5f);
    EXPECT_NEAR(out(3, 0), 4.0f, 1e-5f);

    // Row 1: [7,5,8,6] -> [5,6,7,8]
    EXPECT_NEAR(out(0, 1), 5.0f, 1e-5f);
    EXPECT_NEAR(out(1, 1), 6.0f, 1e-5f);
    EXPECT_NEAR(out(2, 1), 7.0f, 1e-5f);
    EXPECT_NEAR(out(3, 1), 8.0f, 1e-5f);

    // Row 2: [2,9,1,4] -> [1,2,4,9]
    EXPECT_NEAR(out(0, 2), 1.0f, 1e-5f);
    EXPECT_NEAR(out(1, 2), 2.0f, 1e-5f);
    EXPECT_NEAR(out(2, 2), 4.0f, 1e-5f);
    EXPECT_NEAR(out(3, 2), 9.0f, 1e-5f);

    // Row 3: [6,3,7,5] -> [3,5,6,7]
    EXPECT_NEAR(out(0, 3), 3.0f, 1e-5f);
    EXPECT_NEAR(out(1, 3), 5.0f, 1e-5f);
    EXPECT_NEAR(out(2, 3), 6.0f, 1e-5f);
    EXPECT_NEAR(out(3, 3), 7.0f, 1e-5f);
}

// -----------------------------------------------------------------------------
// Test 6: Sort2DAxis0 — sort each column ascending
// -----------------------------------------------------------------------------

TEST(SortExt, Sort2DAxis0) {
    auto f = make_4x4_matrix("s2d_ax0");
    auto sorted = sort_2d(f, 4, 4, 0, true, "s2d_ax0_out");
    Halide::Runtime::Buffer<float> out(4, 4);
    sorted.realize(out);

    // Matrix columns:
    // col 0: [3,7,2,6] -> sorted [2,3,6,7]
    // col 1: [1,5,9,3] -> sorted [1,3,5,9]
    // col 2: [4,8,1,7] -> sorted [1,4,7,8]
    // col 3: [2,6,4,5] -> sorted [2,4,5,6]
    // Verify col 0:
    EXPECT_NEAR(out(0, 0), 2.0f, 1e-5f);
    EXPECT_NEAR(out(0, 1), 3.0f, 1e-5f);
    EXPECT_NEAR(out(0, 2), 6.0f, 1e-5f);
    EXPECT_NEAR(out(0, 3), 7.0f, 1e-5f);

    // Verify col 1:
    EXPECT_NEAR(out(1, 0), 1.0f, 1e-5f);
    EXPECT_NEAR(out(1, 1), 3.0f, 1e-5f);
    EXPECT_NEAR(out(1, 2), 5.0f, 1e-5f);
    EXPECT_NEAR(out(1, 3), 9.0f, 1e-5f);

    // Verify col 2:
    EXPECT_NEAR(out(2, 0), 1.0f, 1e-5f);
    EXPECT_NEAR(out(2, 1), 4.0f, 1e-5f);
    EXPECT_NEAR(out(2, 2), 7.0f, 1e-5f);
    EXPECT_NEAR(out(2, 3), 8.0f, 1e-5f);
}

// -----------------------------------------------------------------------------
// Test 7: Sort2DAxis1Descending — sort each row descending
// -----------------------------------------------------------------------------

TEST(SortExt, Sort2DAxis1Descending) {
    auto f = make_4x4_matrix("s2d_desc");
    auto sorted = sort_2d(f, 4, 4, 1, false, "s2d_desc_out");
    Halide::Runtime::Buffer<float> out(4, 4);
    sorted.realize(out);

    // Row 0: [3,1,4,2] -> descending [4,3,2,1]
    EXPECT_NEAR(out(0, 0), 4.0f, 1e-5f);
    EXPECT_NEAR(out(1, 0), 3.0f, 1e-5f);
    EXPECT_NEAR(out(2, 0), 2.0f, 1e-5f);
    EXPECT_NEAR(out(3, 0), 1.0f, 1e-5f);

    // Row 1: [7,5,8,6] -> descending [8,7,6,5]
    EXPECT_NEAR(out(0, 1), 8.0f, 1e-5f);
    EXPECT_NEAR(out(1, 1), 7.0f, 1e-5f);
    EXPECT_NEAR(out(2, 1), 6.0f, 1e-5f);
    EXPECT_NEAR(out(3, 1), 5.0f, 1e-5f);
}

// -----------------------------------------------------------------------------
// Test 8: Sort2DAxis1NonPow2 — 2x3 matrix, axis=1 (cols=3, non-power-of-2)
// -----------------------------------------------------------------------------

TEST(SortExt, Sort2DAxis1NonPow2) {
    // row 0: [5, 2, 8] -> sorted [2, 5, 8]
    // row 1: [3, 7, 1] -> sorted [1, 3, 7]
    Halide::Func f("s2d_np2");
    Halide::Var x("x"), y("y");
    f(x, y) = Halide::cast<float>(
        Halide::select(y == 0,
            Halide::select(x == 0, 5, Halide::select(x == 1, 2, 8)),
            Halide::select(x == 0, 3, Halide::select(x == 1, 7, 1))));

    auto sorted = sort_2d(f, 2, 3, 1, true, "s2d_np2_out");
    Halide::Runtime::Buffer<float> out(3, 2);
    sorted.realize(out);

    EXPECT_NEAR(out(0, 0), 2.0f, 1e-5f);
    EXPECT_NEAR(out(1, 0), 5.0f, 1e-5f);
    EXPECT_NEAR(out(2, 0), 8.0f, 1e-5f);

    EXPECT_NEAR(out(0, 1), 1.0f, 1e-5f);
    EXPECT_NEAR(out(1, 1), 3.0f, 1e-5f);
    EXPECT_NEAR(out(2, 1), 7.0f, 1e-5f);
}

// -----------------------------------------------------------------------------
// Test 9: Argsort2DAxis1 — argsort_2d(2x4 matrix, axis=1)
// -----------------------------------------------------------------------------

TEST(SortExt, Argsort2DAxis1) {
    // row 0: [3, 1, 4, 2] -> argsort ascending -> [1, 3, 0, 2]
    // row 1: [8, 5, 7, 6] -> argsort ascending -> [1, 3, 2, 0]
    Halide::Func f("as2d_ax1");
    Halide::Var x("x"), y("y");
    f(x, y) = Halide::cast<float>(
        Halide::select(y == 0,
            Halide::select(x == 0, 3, Halide::select(x == 1, 1, Halide::select(x == 2, 4, 2))),
            Halide::select(x == 0, 8, Halide::select(x == 1, 5, Halide::select(x == 2, 7, 6)))));

    auto indices = argsort_2d(f, 2, 4, 1, true, "as2d_ax1_out");
    Halide::Runtime::Buffer<int32_t> out(4, 2);
    indices.realize(out);

    // Row 0: [3,1,4,2] -> argsorted [1,3,0,2]
    EXPECT_EQ(out(0, 0), 1);
    EXPECT_EQ(out(1, 0), 3);
    EXPECT_EQ(out(2, 0), 0);
    EXPECT_EQ(out(3, 0), 2);

    // Row 1: [8,5,7,6] -> argsorted [1,3,2,0]
    EXPECT_EQ(out(0, 1), 1);
    EXPECT_EQ(out(1, 1), 3);
    EXPECT_EQ(out(2, 1), 2);
    EXPECT_EQ(out(3, 1), 0);
}

// -----------------------------------------------------------------------------
// Test 10: Sort2DAlreadySorted — already-sorted rows should be unchanged
// -----------------------------------------------------------------------------

TEST(SortExt, Sort2DAlreadySorted) {
    // row 0: [1, 2, 3, 4] (already sorted)
    // row 1: [5, 6, 7, 8] (already sorted)
    Halide::Func f("s2d_pre");
    Halide::Var x("x"), y("y");
    f(x, y) = Halide::cast<float>(y * 4 + x + 1);

    auto sorted = sort_2d(f, 2, 4, 1, true, "s2d_pre_out");
    Halide::Runtime::Buffer<float> out(4, 2);
    sorted.realize(out);

    for (int row = 0; row < 2; ++row)
        for (int col = 0; col < 4; ++col)
            EXPECT_NEAR(out(col, row), float(row * 4 + col + 1), 1e-5f);
}
