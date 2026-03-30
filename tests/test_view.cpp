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

// =============================================================================
// ViewInplace: combining view operations with inplace operations
// =============================================================================

// 1. Threshold on a 1D slice — write-through to original
TEST(ViewInplace, Threshold_On_Slice) {
    // buf = {-1, 0, 1, 2, 3}
    Halide::Runtime::Buffer<float> buf(5);
    buf(0) = -1.0f; buf(1) = 0.0f; buf(2) = 1.0f; buf(3) = 2.0f; buf(4) = 3.0f;

    // slice [1, 4) -> {0, 1, 2}
    auto slc = view_slice(buf, 0, 1, 4);
    ASSERT_EQ(slc.dim(0).extent(), 3);

    // threshold 0.5 -> each element becomes max(elem, 0.5)
    inplace_threshold(slc, 0.5f);

    // view should reflect new values
    EXPECT_NEAR(slc(0), 0.5f, 1e-5f);  // max(0, 0.5) = 0.5
    EXPECT_NEAR(slc(1), 1.0f, 1e-5f);  // max(1, 0.5) = 1.0
    EXPECT_NEAR(slc(2), 2.0f, 1e-5f);  // max(2, 0.5) = 2.0

    // original buf at positions 1,2,3 must have changed (write-through)
    EXPECT_NEAR(buf(1), 0.5f, 1e-5f);
    EXPECT_NEAR(buf(2), 1.0f, 1e-5f);
    EXPECT_NEAR(buf(3), 2.0f, 1e-5f);
    // positions 0 and 4 must be untouched
    EXPECT_NEAR(buf(0), -1.0f, 1e-5f);
    EXPECT_NEAR(buf(4),  3.0f, 1e-5f);
}

// 2. Scale on a transposed view — all original values doubled
TEST(ViewInplace, Scale_On_Transposed) {
    // 2 cols x 3 rows: values 0..5
    Halide::Runtime::Buffer<float> orig(2, 3);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 2; ++c)
            orig(c, r) = float(r * 2 + c);

    auto vt = view_transpose(orig);  // 3 cols x 2 rows
    ASSERT_EQ(vt.dim(0).extent(), 3);
    ASSERT_EQ(vt.dim(1).extent(), 2);

    inplace_scale(vt, 2.0f);

    // All original values must be doubled
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 2; ++c)
            EXPECT_NEAR(orig(c, r), float(r * 2 + c) * 2.0f, 1e-5f)
                << "orig(" << c << "," << r << ")";
}

// 3. Clamp on a 1D slice — write-through and other elements untouched
TEST(ViewInplace, Clamp_On_Slice) {
    // buf = {-5, -3, 0, 3, 5}
    Halide::Runtime::Buffer<float> buf(5);
    buf(0) = -5.0f; buf(1) = -3.0f; buf(2) = 0.0f; buf(3) = 3.0f; buf(4) = 5.0f;

    // slice [1, 4) -> {-3, 0, 3}
    auto slc = view_slice(buf, 0, 1, 4);
    ASSERT_EQ(slc.dim(0).extent(), 3);

    // clamp(-1, 2)
    inplace_clamp(slc, -1.0f, 2.0f);

    // view: max(-1, min(2, x))
    EXPECT_NEAR(slc(0), -1.0f, 1e-5f);  // clamp(-3, -1, 2) = -1
    EXPECT_NEAR(slc(1),  0.0f, 1e-5f);  // clamp(0,  -1, 2) =  0
    EXPECT_NEAR(slc(2),  2.0f, 1e-5f);  // clamp(3,  -1, 2) =  2

    // original at positions 1,2,3 must have changed
    EXPECT_NEAR(buf(1), -1.0f, 1e-5f);
    EXPECT_NEAR(buf(2),  0.0f, 1e-5f);
    EXPECT_NEAR(buf(3),  2.0f, 1e-5f);
    // positions 0 and 4 must be untouched
    EXPECT_NEAR(buf(0), -5.0f, 1e-5f);
    EXPECT_NEAR(buf(4),  5.0f, 1e-5f);
}

// 4. AddScalar on a reshaped view — all original 2D elements updated
TEST(ViewInplace, AddScalar_On_Reshape) {
    // 2 cols x 3 rows (6 elements), values 0..5
    Halide::Runtime::Buffer<float> orig(2, 3);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 2; ++c)
            orig(c, r) = float(r * 2 + c);

    // reshape to 1D (6 elements)
    auto vr = view_reshape(orig, {6});
    ASSERT_EQ(vr.dimensions(), 1);
    ASSERT_EQ(vr.dim(0).extent(), 6);

    inplace_add_scalar(vr, 10.0f);

    // every element in the original 2D buffer must be +10
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 2; ++c)
            EXPECT_NEAR(orig(c, r), float(r * 2 + c) + 10.0f, 1e-5f)
                << "orig(" << c << "," << r << ")";
}

// 5. Inplace on one slice does NOT affect the other slice
TEST(ViewInplace, Inplace_Does_Not_Affect_Other_Slice) {
    // buf of 6 elements: {1, 2, 3, 4, 5, 6}
    Halide::Runtime::Buffer<float> buf(6);
    for (int i = 0; i < 6; ++i) buf(i) = float(i + 1);

    // sliceA = [0, 3)  -> {1, 2, 3}
    // sliceB = [3, 6)  -> {4, 5, 6}
    auto slcA = view_slice(buf, 0, 0, 3);
    auto slcB = view_slice(buf, 0, 3, 6);

    // apply threshold=100 to sliceA (all 3 values become 100)
    inplace_threshold(slcA, 100.0f);

    // sliceB must be completely unchanged
    EXPECT_NEAR(slcB(0), 4.0f, 1e-5f);
    EXPECT_NEAR(slcB(1), 5.0f, 1e-5f);
    EXPECT_NEAR(slcB(2), 6.0f, 1e-5f);

    // and through to the original buffer tail
    EXPECT_NEAR(buf(3), 4.0f, 1e-5f);
    EXPECT_NEAR(buf(4), 5.0f, 1e-5f);
    EXPECT_NEAR(buf(5), 6.0f, 1e-5f);
}
