/// @file test_view.cpp
/// @brief Tests for zero-copy strided views (T2-D)
#include <gtest/gtest.h>
#include "numhalide_all.h"
#include <vector>
using namespace numhalide;

// Helper: allocate and fill a 2D buffer row-major
static Halide::Runtime::Buffer<float> make_buf2d(int cols, int rows, float base = 0.0f) {
    Halide::Runtime::Buffer<float> b(cols, rows);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            b(c, r) = base + (float)(r * cols + c);
    return b;
}

// 1. view_transpose: matches manual copy-transpose
TEST(View, Transpose2D_Values) {
    auto orig = make_buf2d(4, 3);  // 3 rows x 4 cols
    auto vt   = view_transpose(orig);
    // After transpose: 4 rows x 3 cols, vt(x, y) = orig(y, x)
    ASSERT_EQ(vt.dim(0).extent(), 3);  // new cols = old rows
    ASSERT_EQ(vt.dim(1).extent(), 4);  // new rows = old cols
    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 3; ++x)
            EXPECT_EQ(vt(x, y), orig(y, x)) << "(" << x << "," << y << ")";
}

// 2. view_transpose: same host pointer (zero copy)
TEST(View, Transpose2D_ZeroCopy) {
    auto orig = make_buf2d(5, 4);
    auto vt   = view_transpose(orig);
    EXPECT_EQ(static_cast<const void*>(vt.data()),
              static_cast<const void*>(orig.data()));
}

// 3. view_slice along axis 1 (rows): select rows 1..2 of 4x4
TEST(View, Slice2D_Axis0_Values) {
    auto orig = make_buf2d(4, 4);  // 4 cols x 4 rows, values 0..15
    auto vs   = view_slice(orig, 1, 1, 3);  // rows 1..2 (axis 1 = y = row)
    ASSERT_EQ(vs.dim(0).extent(), 4);  // cols unchanged
    ASSERT_EQ(vs.dim(1).extent(), 2);  // 2 rows
    for (int r = 0; r < 2; ++r)
        for (int c = 0; c < 4; ++c)
            EXPECT_EQ(vs(c, r), orig(c, r + 1)) << "(" << c << "," << r << ")";
}

// 4. view_slice along axis 0 (cols): select cols 1..2 of 4x3
TEST(View, Slice2D_Axis1_Values) {
    auto orig = make_buf2d(4, 3);  // 4 cols x 3 rows
    auto vs   = view_slice(orig, 0, 1, 3);  // cols 1..2 (axis 0 = x = col)
    ASSERT_EQ(vs.dim(0).extent(), 2);  // 2 cols
    ASSERT_EQ(vs.dim(1).extent(), 3);  // rows unchanged
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 2; ++c)
            EXPECT_EQ(vs(c, r), orig(c + 1, r)) << "(" << c << "," << r << ")";
}

// 5. view_slice: zero copy
TEST(View, Slice_ZeroCopy) {
    auto orig = make_buf2d(8, 8);
    auto vs   = view_slice(orig, 0, 2, 6);  // cols 2..5
    // Data pointer offset by 2*stride[0] = 2*1 = 2 floats
    EXPECT_EQ(static_cast<const void*>(vs.data()),
              static_cast<const void*>(orig.data() + 2));
}

// 6. view_reshape 2D to 1D
TEST(View, Reshape_2D_to_1D) {
    auto orig = make_buf2d(4, 3);  // 4 cols x 3 rows = 12 elements
    auto vr   = view_reshape(orig, {12});
    ASSERT_EQ(vr.dimensions(), 1);
    ASSERT_EQ(vr.dim(0).extent(), 12);
    // Elements should be in row-major order: 0,1,2,...,11
    for (int i = 0; i < 12; ++i)
        EXPECT_EQ(vr(i), (float)i) << "i=" << i;
}

// 7. view_reshape 1D to 2D
TEST(View, Reshape_1D_to_2D) {
    Halide::Runtime::Buffer<float> orig(12);
    for (int i = 0; i < 12; ++i) orig(i) = (float)i;
    auto vr = view_reshape(orig, {4, 3});  // 4 cols x 3 rows
    ASSERT_EQ(vr.dimensions(), 2);
    ASSERT_EQ(vr.dim(0).extent(), 4);
    ASSERT_EQ(vr.dim(1).extent(), 3);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 4; ++c)
            EXPECT_EQ(vr(c, r), (float)(r * 4 + c)) << "(" << c << "," << r << ")";
}

// 8. view_reshape: zero copy
TEST(View, Reshape_ZeroCopy) {
    auto orig = make_buf2d(6, 2);
    auto vr   = view_reshape(orig, {12});
    EXPECT_EQ(static_cast<const void*>(vr.data()),
              static_cast<const void*>(orig.data()));
}

// 9. view: chained transpose + slice
TEST(View, Chained_Transpose_Slice) {
    auto orig = make_buf2d(6, 4);  // 4 rows x 6 cols
    auto vt   = view_transpose(orig);    // now 6 rows x 4 cols
    auto vs   = view_slice(vt, 1, 1, 3); // rows 1..2 -> 2 rows x 4 cols
    ASSERT_EQ(vs.dim(0).extent(), 4);  // cols
    ASSERT_EQ(vs.dim(1).extent(), 2);  // rows
    for (int r = 0; r < 2; ++r)
        for (int c = 0; c < 4; ++c)
            // vs(c, r) = vt(c, r+1) = orig(r+1, c)
            EXPECT_EQ(vs(c, r), orig(r + 1, c)) << "(" << c << "," << r << ")";
}

// 10. view_slice write-through: modifying view affects original
TEST(View, Slice_WriteThrough) {
    auto orig = make_buf2d(4, 4);  // values 0..15
    auto vs   = view_slice(orig, 0, 2, 4);  // cols 2..3
    vs(0, 0) = 99.0f;  // sets col 2, row 0
    EXPECT_EQ(orig(2, 0), 99.0f);  // should reflect in original
}
