/// @file test_broadcasting.cpp
/// @brief Tests for broadcasting binary operations from broadcast.h

#include <gtest/gtest.h>
#include "numhalide_all.h"
#include <cmath>

using namespace numhalide;

// -----------------------------------------------------------------------------
// Test 1: AddRowVectorBroadcast — (3,4)+(1,4)
// -----------------------------------------------------------------------------

TEST(Broadcast, AddRowVectorBroadcast) {
    // A(x,y) = y*4 + x  (3x4 matrix, rows=3, cols=4)
    Halide::Func A("A_rbv"), b("b_rbv");
    Halide::Var x("x"), y("y");
    A(x, y) = Halide::cast<float>(y * 4 + x);
    b(x, y) = Halide::cast<float>(x);  // [0,1,2,3] — row vector (1,4)

    auto result = add(A, {3, 4}, b, {1, 4}, "add_row_bcast");
    Halide::Runtime::Buffer<float> out(4, 3);  // Buffer(width=cols, height=rows)
    result.realize(out);

    // out(col, row) = (row*4 + col) + col = row*4 + 2*col
    EXPECT_NEAR(out(0, 0),  0.0f, 1e-5f);   // row=0, col=0: 0+0=0
    EXPECT_NEAR(out(1, 0),  2.0f, 1e-5f);   // row=0, col=1: 1+1=2
    EXPECT_NEAR(out(3, 1), 10.0f, 1e-5f);   // row=1, col=3: 7+3=10
    EXPECT_NEAR(out(2, 2), 12.0f, 1e-5f);   // row=2, col=2: 10+2=12
    EXPECT_NEAR(out(0, 1),  4.0f, 1e-5f);   // row=1, col=0: 4+0=4
    EXPECT_NEAR(out(3, 2), 14.0f, 1e-5f);   // row=2, col=3: 11+3=14
}

// -----------------------------------------------------------------------------
// Test 2: AddColVectorBroadcast — (3,4)+(3,1)
// -----------------------------------------------------------------------------

TEST(Broadcast, AddColVectorBroadcast) {
    // A(x,y) = y*4 + x  (3x4 matrix, rows=3, cols=4)
    // col_bias(x,y) = y*10  (column vector: {0,10,20} for rows 0,1,2)
    Halide::Func A("A_cbv"), col_bias("col_cbv");
    Halide::Var x("x"), y("y");
    A(x, y) = Halide::cast<float>(y * 4 + x);
    col_bias(x, y) = Halide::cast<float>(y * 10);  // shape {3,1}

    auto result = add(A, {3, 4}, col_bias, {3, 1}, "add_col_bcast");
    Halide::Runtime::Buffer<float> out(4, 3);
    result.realize(out);

    // out(col, row) = (row*4 + col) + row*10
    EXPECT_NEAR(out(0, 0),  0.0f, 1e-5f);   // row=0, col=0: 0 + 0 = 0
    EXPECT_NEAR(out(3, 0),  3.0f, 1e-5f);   // row=0, col=3: 3 + 0 = 3
    EXPECT_NEAR(out(0, 1), 14.0f, 1e-5f);   // row=1, col=0: 4 + 10 = 14
    EXPECT_NEAR(out(2, 2), 30.0f, 1e-5f);   // row=2, col=2: 10 + 20 = 30
    EXPECT_NEAR(out(3, 2), 31.0f, 1e-5f);   // row=2, col=3: 11 + 20 = 31
}

// -----------------------------------------------------------------------------
// Test 3: SubRowVector — (4,4)-(1,4)
// -----------------------------------------------------------------------------

TEST(Broadcast, SubRowVector) {
    // A(x,y) = x*2 + y  (4 rows x 4 cols)
    // row_vec(x,y) = x  (1 row x 4 cols)
    Halide::Func A("A_srv"), rv("rv_srv");
    Halide::Var x("x"), y("y");
    A(x, y) = Halide::cast<float>(x * 2 + y);
    rv(x, y) = Halide::cast<float>(x);

    auto result = sub(A, {4, 4}, rv, {1, 4}, "sub_row_vec");
    Halide::Runtime::Buffer<float> out(4, 4);
    result.realize(out);

    // out(col, row) = (col*2 + row) - col = col + row
    for (int row = 0; row < 4; ++row)
        for (int col = 0; col < 4; ++col)
            EXPECT_NEAR(out(col, row), float(col + row), 1e-5f);
}

// -----------------------------------------------------------------------------
// Test 4: MulColVector — (4,4)*(4,1)
// -----------------------------------------------------------------------------

TEST(Broadcast, MulColVector) {
    // A(x,y) = x + 1  (all rows same, values 1,2,3,4)
    // col_scale(x,y) = y + 1  (scale by row index + 1)
    Halide::Func A("A_mcv"), cs("cs_mcv");
    Halide::Var x("x"), y("y");
    A(x, y) = Halide::cast<float>(x + 1);      // rows=4, cols=4
    cs(x, y) = Halide::cast<float>(y + 1);     // shape {4,1}

    auto result = mul(A, {4, 4}, cs, {4, 1}, "mul_col_vec");
    Halide::Runtime::Buffer<float> out(4, 4);
    result.realize(out);

    // out(col, row) = (col+1) * (row+1)
    for (int row = 0; row < 4; ++row)
        for (int col = 0; col < 4; ++col)
            EXPECT_NEAR(out(col, row), float((col + 1) * (row + 1)), 1e-5f);
}

// -----------------------------------------------------------------------------
// Test 5: DivScalar — (3,3)/(1,1)
// -----------------------------------------------------------------------------

TEST(Broadcast, DivScalar) {
    // A(x,y) = (y*3 + x + 1) * 2.0f  (3x3 matrix, values 2,4,6,8,...)
    // scalar(x,y) = 2.0f  (shape {1,1})
    Halide::Func A("A_ds"), sc("sc_ds");
    Halide::Var x("x"), y("y");
    A(x, y) = Halide::cast<float>((y * 3 + x + 1) * 2);
    sc(x, y) = 2.0f;

    auto result = div(A, {3, 3}, sc, {1, 1}, "div_scalar");
    Halide::Runtime::Buffer<float> out(3, 3);
    result.realize(out);

    // out(col, row) = (row*3 + col + 1)
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col)
            EXPECT_NEAR(out(col, row), float(row * 3 + col + 1), 1e-5f);
}

// -----------------------------------------------------------------------------
// Test 6: MinimumBroadcast — minimum((3,3), (1,3))
// -----------------------------------------------------------------------------

TEST(Broadcast, MinimumBroadcast) {
    // A(x,y) = y*3 + x  (values 0..8)
    // row_floor(x,y) = x*2  (shape {1,3}: [0,2,4])
    Halide::Func A("A_mb"), rf("rf_mb");
    Halide::Var x("x"), y("y");
    A(x, y) = Halide::cast<float>(y * 3 + x);
    rf(x, y) = Halide::cast<float>(x * 2);  // shape {1,3} means 1 row, 3 cols

    auto result = minimum(A, {3, 3}, rf, {1, 3}, "min_bcast");
    Halide::Runtime::Buffer<float> out(3, 3);
    result.realize(out);

    // out(col, row) = min(row*3 + col, col*2)
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col) {
            float a_val = float(row * 3 + col);
            float b_val = float(col * 2);
            EXPECT_NEAR(out(col, row), std::min(a_val, b_val), 1e-5f);
        }
}

// -----------------------------------------------------------------------------
// Test 7: MaximumBroadcast — maximum((3,3), (3,1))
// -----------------------------------------------------------------------------

TEST(Broadcast, MaximumBroadcast) {
    // A(x,y) = y*3 + x  (values 0..8)
    // col_floor(x,y) = y  (shape {3,1}: 1 col, rows are [0,1,2])
    Halide::Func A("A_maxb"), cf("cf_maxb");
    Halide::Var x("x"), y("y");
    A(x, y) = Halide::cast<float>(y * 3 + x);
    cf(x, y) = Halide::cast<float>(y);  // shape {3,1}

    auto result = maximum(A, {3, 3}, cf, {3, 1}, "max_bcast");
    Halide::Runtime::Buffer<float> out(3, 3);
    result.realize(out);

    // out(col, row) = max(row*3 + col, row)
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col) {
            float a_val = float(row * 3 + col);
            float b_val = float(row);
            EXPECT_NEAR(out(col, row), std::max(a_val, b_val), 1e-5f);
        }
}

// -----------------------------------------------------------------------------
// Test 8: EqualBroadcast — equal((3,3), (1,3))
// -----------------------------------------------------------------------------

TEST(Broadcast, EqualBroadcast) {
    // A(x,y) = x  (each row is [0,1,2])
    // row_ref(x,y) = x  (shape {1,3}: also [0,1,2])
    // Result: all true because every element equals the reference
    Halide::Func A("A_eb"), rr("rr_eb");
    Halide::Var x("x"), y("y");
    A(x, y) = Halide::cast<float>(x);
    rr(x, y) = Halide::cast<float>(x);  // shape {1,3}

    auto result = equal(A, {3, 3}, rr, {1, 3}, "eq_bcast");
    Halide::Runtime::Buffer<bool> out(3, 3);
    result.realize(out);

    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col)
            EXPECT_TRUE(out(col, row));

    // Now with A(x,y) = y*3+x vs reference [0,1,2]: only row 0 matches
    Halide::Func A2("A2_eb");
    A2(x, y) = Halide::cast<float>(y * 3 + x);
    auto result2 = equal(A2, {3, 3}, rr, {1, 3}, "eq_bcast2");
    Halide::Runtime::Buffer<bool> out2(3, 3);
    result2.realize(out2);

    // Row 0: A(col,0) = col == col -> true
    EXPECT_TRUE(out2(0, 0));
    EXPECT_TRUE(out2(1, 0));
    EXPECT_TRUE(out2(2, 0));
    // Row 1: A(col,1) = 3+col != col -> false
    EXPECT_FALSE(out2(0, 1));
    EXPECT_FALSE(out2(1, 1));
}

// -----------------------------------------------------------------------------
// Test 9: GreaterBroadcast — greater((3,4), (3,1))
// -----------------------------------------------------------------------------

TEST(Broadcast, GreaterBroadcast) {
    // A(x,y) = x  (cols 0,1,2,3 for each row)
    // col_thresh(x,y) = y  (shape {3,1}: thresholds [0,1,2])
    // out(col, row) = col > row
    Halide::Func A("A_gb"), ct("ct_gb");
    Halide::Var x("x"), y("y");
    A(x, y) = Halide::cast<float>(x);
    ct(x, y) = Halide::cast<float>(y);  // shape {3,1}

    auto result = greater(A, {3, 4}, ct, {3, 1}, "gt_bcast");
    Halide::Runtime::Buffer<bool> out(4, 3);
    result.realize(out);

    // out(col, row) = (col > row)
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 4; ++col)
            EXPECT_EQ(out(col, row), col > row);
}

// -----------------------------------------------------------------------------
// Test 10: RankPromotion1DTo2D — add((3,4), {3,4}, f, {4})
// -----------------------------------------------------------------------------

TEST(Broadcast, RankPromotion1DTo2D) {
    // A(x,y) = y*4 + x  (3 rows x 4 cols)
    // row_vec is 1D: f(x) = x  (shape {4})
    // Broadcasting: {4} promoted to {1,4} to match {3,4}
    Halide::Func A("A_rp"), rv("rv_rp");
    Halide::Var x("x"), y("y");
    A(x, y) = Halide::cast<float>(y * 4 + x);
    rv(x) = Halide::cast<float>(x);  // 1D, shape {4}

    auto result = add(A, {3, 4}, rv, {4}, "add_1d_promo");
    Halide::Runtime::Buffer<float> out(4, 3);
    result.realize(out);

    // out(col, row) = (row*4 + col) + col = row*4 + 2*col
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 4; ++col)
            EXPECT_NEAR(out(col, row), float(row * 4 + 2 * col), 1e-5f);
}

// -----------------------------------------------------------------------------
// Test 11: SameShapeNoOp — add((3,4), (3,4)): no broadcast
// -----------------------------------------------------------------------------

TEST(Broadcast, SameShapeNoOp) {
    // A(x,y) = y*4 + x
    // B(x,y) = y*4 + x
    // Result: 2*(y*4+x)
    Halide::Func A("A_ssn"), B("B_ssn");
    Halide::Var x("x"), y("y");
    A(x, y) = Halide::cast<float>(y * 4 + x);
    B(x, y) = Halide::cast<float>(y * 4 + x);

    auto result = add(A, {3, 4}, B, {3, 4}, "add_same_shape");
    Halide::Runtime::Buffer<float> out(4, 3);
    result.realize(out);

    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 4; ++col)
            EXPECT_NEAR(out(col, row), float(2 * (row * 4 + col)), 1e-5f);
}

// -----------------------------------------------------------------------------
// Test 12: OuterProductViaMul — mul(col_vec {4,1}, row_vec {1,4}) -> {4,4}
// -----------------------------------------------------------------------------

TEST(Broadcast, OuterProductViaMul) {
    // col_vec(x,y) = y+1  (shape {4,1}: col values [1,2,3,4])
    // row_vec(x,y) = x+1  (shape {1,4}: row values [1,2,3,4])
    // outer(col, row) = (col+1) * (row+1)
    Halide::Func cv("cv_op"), rv("rv_op");
    Halide::Var x("x"), y("y");
    cv(x, y) = Halide::cast<float>(y + 1);  // shape {4,1}
    rv(x, y) = Halide::cast<float>(x + 1);  // shape {1,4}

    auto result = mul(cv, {4, 1}, rv, {1, 4}, "outer_prod");
    Halide::Runtime::Buffer<float> out(4, 4);
    result.realize(out);

    // out(col, row) = (row+1) * (col+1)
    for (int row = 0; row < 4; ++row)
        for (int col = 0; col < 4; ++col)
            EXPECT_NEAR(out(col, row), float((row + 1) * (col + 1)), 1e-5f);
}
