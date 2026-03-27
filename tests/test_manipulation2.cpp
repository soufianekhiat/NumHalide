/// @file test_manipulation2.cpp
/// @brief Tests for swapaxes, column_stack, row_stack, atleast_1d/2d/3d

#include <gtest/gtest.h>
#include "numhalide_all.h"
#include <cmath>

using namespace numhalide;

// -----------------------------------------------------------------------------
// Swapaxes Tests
// -----------------------------------------------------------------------------

TEST(Manipulation2, Swapaxes_2D) {
    // Input shape {2, 3}: 2 rows, 3 cols
    // In Halide: f(x, y) where x=col (0..2), y=row (0..1)
    // f(0,0)=1, f(1,0)=2, f(2,0)=3
    // f(0,1)=4, f(1,1)=5, f(2,1)=6
    shape_t in_shape = {2, 3};
    Halide::Func f("f");
    Halide::Var x("x"), y("y");
    f(x, y) = Halide::select(y == 0,
        Halide::select(x == 0, 1.0f, x == 1, 2.0f, 3.0f),
        Halide::select(x == 0, 4.0f, x == 1, 5.0f, 6.0f));

    // swapaxes(0, 1): swaps shape dim 0 (size 2) with shape dim 1 (size 3)
    // norm_ax1=0 -> halide_ax1=1 (y), norm_ax2=1 -> halide_ax2=0 (x)
    // result(x, y) = f(y, x), output shape {3, 2}, buffer size (2, 3)
    auto result = swapaxes(f, in_shape, 0, 1);
    Halide::Runtime::Buffer<float> out(2, 3);
    result.realize(out);

    // result(col, row): col=0..1, row=0..2
    // result(col, row) = f(row, col) [because x<->y swapped]
    EXPECT_NEAR(out(0, 0), 1.0f, 1e-5f);  // f(0,0) = 1
    EXPECT_NEAR(out(1, 0), 4.0f, 1e-5f);  // f(0,1) = 4
    EXPECT_NEAR(out(0, 1), 2.0f, 1e-5f);  // f(1,0) = 2
    EXPECT_NEAR(out(1, 1), 5.0f, 1e-5f);  // f(1,1) = 5
    EXPECT_NEAR(out(0, 2), 3.0f, 1e-5f);  // f(2,0) = 3
    EXPECT_NEAR(out(1, 2), 6.0f, 1e-5f);  // f(2,1) = 6
}

TEST(Manipulation2, Swapaxes_SameAxis) {
    // swapaxes with same axis should behave identically to the input
    shape_t s = {3, 4};
    Halide::Func f("f");
    Halide::Var x, y;
    // f(x, y) = x + y*4, buffer (4, 3)
    f(x, y) = Halide::cast<float>(x + y * 4);
    auto result = swapaxes(f, s, 0, 0);
    Halide::Runtime::Buffer<float> out(4, 3);
    result.realize(out);
    // out(2, 1) = f(2, 1) = 2 + 1*4 = 6
    EXPECT_NEAR(out(2, 1), 6.0f, 1e-5f);
    EXPECT_NEAR(out(0, 0), 0.0f, 1e-5f);
    EXPECT_NEAR(out(3, 2), 11.0f, 1e-5f);
}

// -----------------------------------------------------------------------------
// Column Stack Tests
// -----------------------------------------------------------------------------

TEST(Manipulation2, ColumnStack_1D) {
    // Stack 3 constant 1D arrays of size 4 as columns
    // Output shape {4, 3}: n_rows=4, n_cols=3
    // Halide buffer: (n_cols=3, n_rows=4) = Buffer<float>(3, 4)
    // ret(col, row): col=0..2, row=0..3
    Halide::Func a("a"), b("b"), c("c");
    Halide::Var x;
    a(x) = 1.0f;
    b(x) = 2.0f;
    c(x) = 3.0f;

    auto result = column_stack({a, b, c}, 4);
    Halide::Runtime::Buffer<float> out(3, 4);
    result.realize(out);

    EXPECT_NEAR(out(0, 0), 1.0f, 1e-5f);  // col 0 = array a, row 0
    EXPECT_NEAR(out(1, 0), 2.0f, 1e-5f);  // col 1 = array b, row 0
    EXPECT_NEAR(out(2, 0), 3.0f, 1e-5f);  // col 2 = array c, row 0
    EXPECT_NEAR(out(0, 3), 1.0f, 1e-5f);  // col 0 = array a, row 3
    EXPECT_NEAR(out(1, 3), 2.0f, 1e-5f);  // col 1 = array b, row 3
    EXPECT_NEAR(out(2, 3), 3.0f, 1e-5f);  // col 2 = array c, row 3
}

TEST(Manipulation2, ColumnStack_Varying) {
    // Stack 2 arrays of size 3: a(x)=x, b(x)=x*10
    // Output shape {3, 2}: buffer (2, 3)
    Halide::Func a("a2"), b("b2");
    Halide::Var x;
    a(x) = Halide::cast<float>(x);
    b(x) = Halide::cast<float>(x) * 10.0f;

    auto result = column_stack({a, b}, 3);
    Halide::Runtime::Buffer<float> out(2, 3);
    result.realize(out);

    // col 0 = a: [0,1,2], col 1 = b: [0,10,20]
    EXPECT_NEAR(out(0, 0), 0.0f, 1e-5f);
    EXPECT_NEAR(out(0, 1), 1.0f, 1e-5f);
    EXPECT_NEAR(out(0, 2), 2.0f, 1e-5f);
    EXPECT_NEAR(out(1, 0), 0.0f, 1e-5f);
    EXPECT_NEAR(out(1, 1), 10.0f, 1e-5f);
    EXPECT_NEAR(out(1, 2), 20.0f, 1e-5f);
}

// -----------------------------------------------------------------------------
// Row Stack Tests
// -----------------------------------------------------------------------------

TEST(Manipulation2, RowStack_1D) {
    // Stack 3 1D arrays of size 4 as rows → shape {3, 4}
    // Halide buffer: (n_cols=4, n_rows=3) = Buffer<float>(4, 3)
    // ret(col, row): col=0..3, row=0..2
    Halide::Func a("a3"), b("b3"), c("c3");
    Halide::Var x;
    a(x) = Halide::cast<float>(x);           // [0,1,2,3]
    b(x) = Halide::cast<float>(x) + 10.0f;  // [10,11,12,13]
    c(x) = Halide::cast<float>(x) + 20.0f;  // [20,21,22,23]

    std::vector<shape_t> shapes = {{4}, {4}, {4}};
    auto result = row_stack({a, b, c}, shapes);
    Halide::Runtime::Buffer<float> out(4, 3);
    result.realize(out);

    EXPECT_NEAR(out(0, 0), 0.0f, 1e-5f);   // row 0, col 0 = a(0)
    EXPECT_NEAR(out(2, 0), 2.0f, 1e-5f);   // row 0, col 2 = a(2)
    EXPECT_NEAR(out(0, 1), 10.0f, 1e-5f);  // row 1, col 0 = b(0)
    EXPECT_NEAR(out(3, 1), 13.0f, 1e-5f);  // row 1, col 3 = b(3)
    EXPECT_NEAR(out(0, 2), 20.0f, 1e-5f);  // row 2, col 0 = c(0)
    EXPECT_NEAR(out(3, 2), 23.0f, 1e-5f);  // row 2, col 3 = c(3)
}

TEST(Manipulation2, RowStack_SingleArray) {
    // A single 1D array stacked as rows yields 1 row
    Halide::Func a("a_single");
    Halide::Var x;
    a(x) = Halide::cast<float>(x) * 3.0f;  // [0, 3, 6]

    std::vector<shape_t> shapes = {{3}};
    auto result = row_stack({a}, shapes);
    Halide::Runtime::Buffer<float> out(3, 1);
    result.realize(out);

    EXPECT_NEAR(out(0, 0), 0.0f, 1e-5f);
    EXPECT_NEAR(out(1, 0), 3.0f, 1e-5f);
    EXPECT_NEAR(out(2, 0), 6.0f, 1e-5f);
}

// -----------------------------------------------------------------------------
// AtLeast1D Tests
// -----------------------------------------------------------------------------

TEST(Manipulation2, AtLeast1D_Passthrough) {
    // A 1D array should come back unchanged
    shape_t s = {5};
    Halide::Func f("f_1d");
    Halide::Var x;
    f(x) = Halide::cast<float>(x);
    auto [g, gs] = atleast_1d(f, s);
    EXPECT_EQ(gs.rank, 1);
    EXPECT_EQ(gs.extents[0], 5);

    Halide::Runtime::Buffer<float> out(5);
    g.realize(out);
    EXPECT_NEAR(out(3), 3.0f, 1e-5f);
}

TEST(Manipulation2, AtLeast1D_HigherRankPassthrough) {
    // A 2D array should also come back unchanged
    shape_t s = {3, 4};
    Halide::Func f("f_2d_pass");
    Halide::Var x, y;
    f(x, y) = 1.0f;
    auto [g, gs] = atleast_1d(f, s);
    EXPECT_EQ(gs.rank, 2);
    EXPECT_EQ(gs.extents[0], 3);
    EXPECT_EQ(gs.extents[1], 4);
}

// -----------------------------------------------------------------------------
// AtLeast2D Tests
// -----------------------------------------------------------------------------

TEST(Manipulation2, AtLeast2D_Promotes1D) {
    // 1D (4,) → 2D (1, 4): extents[0]=1 (rows), extents[1]=4 (cols)
    // Halide buffer: (cols=4, rows=1) = Buffer<float>(4, 1)
    shape_t s = {4};
    Halide::Func f("f_at2d");
    Halide::Var x;
    f(x) = Halide::cast<float>(x) * 2.0f;  // [0,2,4,6]
    auto [g, gs] = atleast_2d(f, s);
    EXPECT_EQ(gs.rank, 2);
    EXPECT_EQ(gs.extents[0], 1);  // 1 row
    EXPECT_EQ(gs.extents[1], 4);  // 4 cols

    // ret(x, y) = f(x), so out(col, row)
    Halide::Runtime::Buffer<float> out(4, 1);
    g.realize(out);
    EXPECT_NEAR(out(0, 0), 0.0f, 1e-5f);
    EXPECT_NEAR(out(2, 0), 4.0f, 1e-5f);
    EXPECT_NEAR(out(3, 0), 6.0f, 1e-5f);
}

TEST(Manipulation2, AtLeast2D_Passthrough) {
    // 2D input stays 2D, shape unchanged
    shape_t s = {3, 4};
    Halide::Func f("f_at2d_pass");
    Halide::Var x, y;
    f(x, y) = 1.0f;
    auto [g, gs] = atleast_2d(f, s);
    EXPECT_EQ(gs.rank, 2);
    EXPECT_EQ(gs.extents[0], 3);
    EXPECT_EQ(gs.extents[1], 4);
}

TEST(Manipulation2, AtLeast2D_3DPassthrough) {
    // 3D input stays 3D, shape unchanged
    shape_t s = {2, 3, 4};
    Halide::Func f("f_at2d_3d");
    Halide::Var x, y, z;
    f(x, y, z) = 1.0f;
    auto [g, gs] = atleast_2d(f, s);
    EXPECT_EQ(gs.rank, 3);
}

// -----------------------------------------------------------------------------
// AtLeast3D Tests
// -----------------------------------------------------------------------------

TEST(Manipulation2, AtLeast3D_From2D) {
    // 2D (2, 3) → 3D (2, 3, 1)
    // Input shape {2,3}: 2 rows, 3 cols; Halide: f(x=col, y=row)
    // Output shape {2, 3, 1}: ret(x,y,z) = f(y, z), buffer (1, 3, 2)
    shape_t s = {2, 3};
    Halide::Func f("f_at3d_2d");
    Halide::Var x, y;
    // f(x, y) = x + y*3 : col 0..2, row 0..1
    f(x, y) = Halide::cast<float>(x + y * 3);

    auto [g, gs] = atleast_3d(f, s);
    EXPECT_EQ(gs.rank, 3);
    EXPECT_EQ(gs.extents[0], 2);
    EXPECT_EQ(gs.extents[1], 3);
    EXPECT_EQ(gs.extents[2], 1);

    // buffer (inner=1, mid=3, outer=2)
    Halide::Runtime::Buffer<float> out(1, 3, 2);
    g.realize(out);
    // g(x, y, z) = f(y, z): out(0, col, row) = f(col, row) = col + row*3
    EXPECT_NEAR(out(0, 0, 0), 0.0f, 1e-5f);  // f(0,0) = 0
    EXPECT_NEAR(out(0, 1, 0), 1.0f, 1e-5f);  // f(1,0) = 1
    EXPECT_NEAR(out(0, 2, 0), 2.0f, 1e-5f);  // f(2,0) = 2
    EXPECT_NEAR(out(0, 0, 1), 3.0f, 1e-5f);  // f(0,1) = 3
    EXPECT_NEAR(out(0, 1, 1), 4.0f, 1e-5f);  // f(1,1) = 4
}

TEST(Manipulation2, AtLeast3D_From1D) {
    // 1D (n,) → 3D (1, n, 1): extents[0]=1, extents[1]=n, extents[2]=1
    // ret(x, y, z) = f(y), buffer (1, n, 1)
    shape_t s = {5};
    Halide::Func f("f_at3d_1d");
    Halide::Var x;
    f(x) = Halide::cast<float>(x) + 1.0f;  // [1,2,3,4,5]

    auto [g, gs] = atleast_3d(f, s);
    EXPECT_EQ(gs.rank, 3);
    EXPECT_EQ(gs.extents[0], 1);
    EXPECT_EQ(gs.extents[1], 5);
    EXPECT_EQ(gs.extents[2], 1);

    Halide::Runtime::Buffer<float> out(1, 5, 1);
    g.realize(out);
    EXPECT_NEAR(out(0, 0, 0), 1.0f, 1e-5f);
    EXPECT_NEAR(out(0, 4, 0), 5.0f, 1e-5f);
}

TEST(Manipulation2, AtLeast3D_Passthrough) {
    // 3D input stays 3D
    shape_t s = {2, 3, 4};
    Halide::Func f("f_at3d_pass");
    Halide::Var x, y, z;
    f(x, y, z) = 1.0f;
    auto [g, gs] = atleast_3d(f, s);
    EXPECT_EQ(gs.rank, 3);
    EXPECT_EQ(gs.extents[0], 2);
    EXPECT_EQ(gs.extents[1], 3);
    EXPECT_EQ(gs.extents[2], 4);
}
