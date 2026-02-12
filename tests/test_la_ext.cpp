/// @file test_la_ext.cpp
/// @brief Tests for extended linear algebra operations

#include <gtest/gtest.h>
#include "numhalide_all.h"
#include <cmath>

using namespace numhalide;

// -----------------------------------------------------------------------------
// Norm Tests
// -----------------------------------------------------------------------------

TEST(LAExt, NormVector) {
    shape_t shape = {3};
    Halide::Func v("v");
    Halide::Var x;
    // [3, 4, 0] -> norm = 5
    v(x) = Halide::cast<float>(Halide::select(x == 0, 3, Halide::select(x == 1, 4, 0)));

    auto result = norm(v, shape, "norm_result");

    Halide::Runtime::Buffer<float> out(1);
    result.realize(out);

    EXPECT_NEAR(out(0), 5.0f, 1e-5f);
}

TEST(LAExt, FrobeniusNorm) {
    shape_t shape = {2, 2};
    Halide::Func m("m");
    Halide::Var x, y;
    // [[1, 2], [3, 4]] -> frobenius = sqrt(1+4+9+16) = sqrt(30)
    m(x, y) = Halide::cast<float>(y * 2 + x + 1);

    auto result = frobenius_norm(m, shape, "frob_result");

    Halide::Runtime::Buffer<float> out(1);
    result.realize(out);

    EXPECT_NEAR(out(0), std::sqrt(30.0f), 1e-5f);
}

TEST(LAExt, NormAxis) {
    // 2x3 matrix, norm along axis 1 (columns)
    shape_t shape = {2, 3};
    Halide::Func m("m");
    Halide::Var x, y;
    // [[3, 4, 0], [0, 3, 4]] -> norms = [5, 5]
    m(x, y) = Halide::cast<float>(Halide::select(y == 0,
        Halide::select(x == 0, 3, Halide::select(x == 1, 4, 0)),
        Halide::select(x == 0, 0, Halide::select(x == 1, 3, 4))
    ));

    auto result = norm(m, shape, 1, "norm_axis1");

    Halide::Runtime::Buffer<float> out(2);
    result.realize(out);

    EXPECT_NEAR(out(0), 5.0f, 1e-5f);
    EXPECT_NEAR(out(1), 5.0f, 1e-5f);
}

// -----------------------------------------------------------------------------
// Triangular Matrix Tests
// -----------------------------------------------------------------------------

TEST(LAExt, Triu) {
    shape_t shape = {3, 3};
    Halide::Func m("m");
    Halide::Var x, y;
    m(x, y) = y * 3 + x + 1;  // [[1,2,3],[4,5,6],[7,8,9]]

    auto result = triu(m, shape, 0, "triu_result");

    Halide::Runtime::Buffer<int32_t> out(3, 3);
    result.realize(out);

    // Upper triangular: [[1,2,3],[0,5,6],[0,0,9]]
    EXPECT_EQ(out(0, 0), 1);
    EXPECT_EQ(out(1, 0), 2);
    EXPECT_EQ(out(2, 0), 3);
    EXPECT_EQ(out(0, 1), 0);  // Below diagonal
    EXPECT_EQ(out(1, 1), 5);
    EXPECT_EQ(out(2, 1), 6);
    EXPECT_EQ(out(0, 2), 0);  // Below diagonal
    EXPECT_EQ(out(1, 2), 0);  // Below diagonal
    EXPECT_EQ(out(2, 2), 9);
}

TEST(LAExt, Tril) {
    shape_t shape = {3, 3};
    Halide::Func m("m");
    Halide::Var x, y;
    m(x, y) = y * 3 + x + 1;  // [[1,2,3],[4,5,6],[7,8,9]]

    auto result = tril(m, shape, 0, "tril_result");

    Halide::Runtime::Buffer<int32_t> out(3, 3);
    result.realize(out);

    // Lower triangular: [[1,0,0],[4,5,0],[7,8,9]]
    EXPECT_EQ(out(0, 0), 1);
    EXPECT_EQ(out(1, 0), 0);  // Above diagonal
    EXPECT_EQ(out(2, 0), 0);  // Above diagonal
    EXPECT_EQ(out(0, 1), 4);
    EXPECT_EQ(out(1, 1), 5);
    EXPECT_EQ(out(2, 1), 0);  // Above diagonal
    EXPECT_EQ(out(0, 2), 7);
    EXPECT_EQ(out(1, 2), 8);
    EXPECT_EQ(out(2, 2), 9);
}

TEST(LAExt, TriuTriLSum) {
    // triu(k=0) + tril(k=-1) = original
    shape_t shape = {3, 3};
    Halide::Func m("m");
    Halide::Var x, y;
    m(x, y) = y * 3 + x + 1;

    auto upper = triu(m, shape, 0, "upper");
    auto lower = tril(m, shape, -1, "lower");  // Strict lower (below diagonal)

    Halide::Func combined("combined");
    combined(x, y) = upper(x, y) + lower(x, y);

    Halide::Runtime::Buffer<int32_t> out(3, 3);
    combined.realize(out);

    // Should equal original
    EXPECT_EQ(out(0, 0), 1);
    EXPECT_EQ(out(1, 1), 5);
    EXPECT_EQ(out(2, 2), 9);
    EXPECT_EQ(out(0, 1), 4);
    EXPECT_EQ(out(1, 0), 2);
}

// -----------------------------------------------------------------------------
// Determinant and Inverse Tests
// -----------------------------------------------------------------------------

TEST(LAExt, Det2x2) {
    Halide::Func m("m");
    Halide::Var x, y;
    // [[1, 2], [3, 4]] -> det = 1*4 - 2*3 = -2
    m(x, y) = y * 2 + x + 1;

    auto result = det2x2(m, "det_result");

    Halide::Runtime::Buffer<int32_t> out(1);
    result.realize(out);

    EXPECT_EQ(out(0), -2);
}

TEST(LAExt, Inv2x2) {
    Halide::Func m("m");
    Halide::Var x, y;
    // [[4, 7], [2, 6]] -> det = 24 - 14 = 10
    // inv = 1/10 * [[6, -7], [-2, 4]]
    m(x, y) = Halide::cast<float>(Halide::select(y == 0,
        Halide::select(x == 0, 4, 7),
        Halide::select(x == 0, 2, 6)
    ));

    auto result = inv2x2(m, "inv_result");

    Halide::Runtime::Buffer<float> out(2, 2);
    result.realize(out);

    EXPECT_NEAR(out(0, 0), 0.6f, 1e-5f);   // 6/10
    EXPECT_NEAR(out(1, 0), -0.7f, 1e-5f);  // -7/10
    EXPECT_NEAR(out(0, 1), -0.2f, 1e-5f);  // -2/10
    EXPECT_NEAR(out(1, 1), 0.4f, 1e-5f);   // 4/10
}

TEST(LAExt, InvIdentity) {
    // Inverse of identity = identity
    Halide::Func m("identity");
    Halide::Var x, y;
    m(x, y) = Halide::cast<float>(Halide::select(x == y, 1, 0));

    auto result = inv2x2(m, "inv_identity");

    Halide::Runtime::Buffer<float> out(2, 2);
    result.realize(out);

    EXPECT_NEAR(out(0, 0), 1.0f, 1e-5f);
    EXPECT_NEAR(out(1, 0), 0.0f, 1e-5f);
    EXPECT_NEAR(out(0, 1), 0.0f, 1e-5f);
    EXPECT_NEAR(out(1, 1), 1.0f, 1e-5f);
}

TEST(LAExt, Det3x3) {
    Halide::Func m("m");
    Halide::Var x, y;
    // [[1, 2, 3], [4, 5, 6], [7, 8, 9]] -> det = 0 (singular)
    m(x, y) = y * 3 + x + 1;

    auto result = det3x3(m, "det3_result");

    Halide::Runtime::Buffer<int32_t> out(1);
    result.realize(out);

    EXPECT_EQ(out(0), 0);
}

TEST(LAExt, Det3x3NonSingular) {
    Halide::Func m("m");
    Halide::Var x, y;
    // [[1, 0, 0], [0, 2, 0], [0, 0, 3]] -> det = 6
    m(x, y) = Halide::select(x == y,
        Halide::select(x == 0, 1, Halide::select(x == 1, 2, 3)),
        0);

    auto result = det3x3(m, "det3_diag");

    Halide::Runtime::Buffer<int32_t> out(1);
    result.realize(out);

    EXPECT_EQ(out(0), 6);
}
