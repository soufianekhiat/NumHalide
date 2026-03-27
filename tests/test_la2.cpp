/// @file test_la2.cpp
/// @brief Tests for inner_1d, kron, and matrix_power operations

#include <gtest/gtest.h>
#include "numhalide_all.h"
#include <cmath>

using namespace numhalide;

// -----------------------------------------------------------------------------
// inner_1d Tests
// -----------------------------------------------------------------------------

TEST(LA2, Inner1D_Basic) {
    // a = [1.0, 2.0, 3.0], b = [4.0, 5.0, 6.0]
    // inner = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    Halide::Func a("a"), b("b");
    Halide::Var x;
    a(x) = Halide::cast<float>(x + 1);  // [1, 2, 3]
    b(x) = Halide::cast<float>(x + 4);  // [4, 5, 6]

    Halide::Func result = inner_1d(a, b, 3);

    Halide::Runtime::Buffer<float> out(1);
    result.realize(out);

    EXPECT_NEAR(out(0), 32.0f, 1e-5f);
}

TEST(LA2, Inner1D_Orthogonal) {
    // a = [1.0, 0.0], b = [0.0, 1.0]
    // inner = 1*0 + 0*1 = 0
    Halide::Func a("a"), b("b");
    Halide::Var x;
    a(x) = Halide::select(x == 0, 1.0f, 0.0f);
    b(x) = Halide::select(x == 0, 0.0f, 1.0f);

    Halide::Func result = inner_1d(a, b, 2);

    Halide::Runtime::Buffer<float> out(1);
    result.realize(out);

    EXPECT_NEAR(out(0), 0.0f, 1e-5f);
}

// -----------------------------------------------------------------------------
// kron Tests
// -----------------------------------------------------------------------------

TEST(LA2, Kron_Basic) {
    // A = [[1,0],[0,1]] (2x2 identity), B = [[1,2],[3,4]] (2x2)
    // kron(A, B) is a 4x4 matrix
    // shape_a = {Ma=2, Na=2}, shape_b = {Mb=2, Nb=2}
    // result shape = {4, 4}
    // kron(col, row) = a(col/Nb, row/Mb) * b(col%Nb, row%Mb)
    // Since A is identity: block at (i*2, j*2) = B if i==j, else 0
    shape_t sa = {2, 2};
    shape_t sb = {2, 2};

    Halide::Func A("A"), B("B");
    Halide::Var x("x"), y("y");
    // A(x, y): x=col, y=row; identity has 1 where x==y
    A(x, y) = Halide::select(x == y, 1.0f, 0.0f);
    // B(x, y): [[1,2],[3,4]] -> B(0,0)=1, B(1,0)=2, B(0,1)=3, B(1,1)=4
    B(x, y) = Halide::select(y == 0,
                  Halide::select(x == 0, 1.0f, 2.0f),
                  Halide::select(x == 0, 3.0f, 4.0f));

    Halide::Func result = kron(A, sa, B, sb);
    shape_t sout = infer_kron(sa, sb);  // {4, 4}

    // Buffer: cols = sout.extents[1] = 4, rows = sout.extents[0] = 4
    Halide::Runtime::Buffer<float> out(sout.extents[1], sout.extents[0]);
    result.realize(out);

    // Top-left 2x2 block: A(0,0)=1 * B = [[1,2],[3,4]]
    EXPECT_NEAR(out(0, 0), 1.0f, 1e-5f);  // col=0, row=0
    EXPECT_NEAR(out(1, 0), 2.0f, 1e-5f);  // col=1, row=0
    EXPECT_NEAR(out(0, 1), 3.0f, 1e-5f);  // col=0, row=1
    EXPECT_NEAR(out(1, 1), 4.0f, 1e-5f);  // col=1, row=1

    // Bottom-right 2x2 block: A(1,1)=1 * B = [[1,2],[3,4]]
    EXPECT_NEAR(out(2, 2), 1.0f, 1e-5f);  // col=2, row=2
    EXPECT_NEAR(out(3, 2), 2.0f, 1e-5f);  // col=3, row=2
    EXPECT_NEAR(out(2, 3), 3.0f, 1e-5f);  // col=2, row=3
    EXPECT_NEAR(out(3, 3), 4.0f, 1e-5f);  // col=3, row=3

    // Off-diagonal blocks: A(0,1)=0 and A(1,0)=0 -> zero blocks
    EXPECT_NEAR(out(0, 2), 0.0f, 1e-5f);  // top-right block
    EXPECT_NEAR(out(2, 0), 0.0f, 1e-5f);  // bottom-left block
}

TEST(LA2, Kron_InferShape) {
    // infer_kron({2,3}, {4,5}) should return shape {8, 15}
    shape_t sa = {2, 3};
    shape_t sb = {4, 5};
    shape_t sout = infer_kron(sa, sb);

    EXPECT_EQ(sout.extents[0], 8);
    EXPECT_EQ(sout.extents[1], 15);
}

// -----------------------------------------------------------------------------
// matrix_power Tests
// -----------------------------------------------------------------------------

TEST(LA2, MatrixPower_Zero) {
    // Any matrix to power 0 = identity
    // Use a 3x3 arbitrary matrix; result must be I_3
    Halide::Func A("A");
    Halide::Var x("x"), y("y");
    // Fill with non-trivial values
    A(x, y) = Halide::cast<float>(y * 3 + x + 1);  // [[1,2,3],[4,5,6],[7,8,9]]

    shape_t sa = {3, 3};
    Halide::Func result = matrix_power(A, sa, 0);

    Halide::Runtime::Buffer<float> out(3, 3);
    result.realize(out);

    // Check diagonal is 1, off-diagonal is 0
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            float expected = (col == row) ? 1.0f : 0.0f;
            EXPECT_NEAR(out(col, row), expected, 1e-5f);
        }
    }
}

TEST(LA2, MatrixPower_One) {
    // A^1 = A
    // A = [[2,1],[0,3]]
    // A(x, y): A(0,0)=2, A(1,0)=1, A(0,1)=0, A(1,1)=3
    Halide::Func A("A");
    Halide::Var x("x"), y("y");
    A(x, y) = Halide::select(y == 0,
                  Halide::select(x == 0, 2.0f, 1.0f),
                  Halide::select(x == 0, 0.0f, 3.0f));

    shape_t sa = {2, 2};
    Halide::Func result = matrix_power(A, sa, 1);

    Halide::Runtime::Buffer<float> out(2, 2);
    result.realize(out);

    EXPECT_NEAR(out(0, 0), 2.0f, 1e-5f);  // A(col=0, row=0)
    EXPECT_NEAR(out(1, 0), 1.0f, 1e-5f);  // A(col=1, row=0)
    EXPECT_NEAR(out(0, 1), 0.0f, 1e-5f);  // A(col=0, row=1)
    EXPECT_NEAR(out(1, 1), 3.0f, 1e-5f);  // A(col=1, row=1)
}

TEST(LA2, MatrixPower_Two) {
    // A = [[2,0],[0,3]], A^2 = [[4,0],[0,9]]
    // A(x, y): diagonal matrix, A(0,0)=2, A(1,1)=3, off-diagonal=0
    Halide::Func A("A");
    Halide::Var x("x"), y("y");
    A(x, y) = Halide::select(x == y,
                  Halide::select(x == 0, 2.0f, 3.0f),
                  0.0f);

    shape_t sa = {2, 2};
    Halide::Func result = matrix_power(A, sa, 2);

    Halide::Runtime::Buffer<float> out(2, 2);
    result.realize(out);

    EXPECT_NEAR(out(0, 0), 4.0f, 1e-5f);  // 2^2
    EXPECT_NEAR(out(1, 0), 0.0f, 1e-5f);
    EXPECT_NEAR(out(0, 1), 0.0f, 1e-5f);
    EXPECT_NEAR(out(1, 1), 9.0f, 1e-5f);  // 3^2
}
