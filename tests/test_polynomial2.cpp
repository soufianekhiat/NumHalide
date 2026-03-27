/// @file test_polynomial2.cpp
/// @brief Tests for polyadd, polysub, polymul, polyder, polyint operations
///
/// Coefficient arrays are highest-power-first:
///   [1, 2, 3] represents x^2 + 2x + 3

#include <gtest/gtest.h>
#include "numhalide_all.h"
#include <cmath>

using namespace numhalide;

// -----------------------------------------------------------------------------
// polyadd Tests
// -----------------------------------------------------------------------------

TEST(Polynomial2, PolyAdd_SameSize) {
    // a = [1, 2, 3] (x^2+2x+3), b = [4, 5, 6] (4x^2+5x+6)
    // result = [5, 7, 9]
    Halide::Func a("a"), b("b");
    Halide::Var i;
    a(i) = Halide::select(i == 0, 1.0f, i == 1, 2.0f, 3.0f);
    b(i) = Halide::select(i == 0, 4.0f, i == 1, 5.0f, 6.0f);

    Halide::Func result = polyadd(a, 3, b, 3);

    Halide::Runtime::Buffer<float> out(3);
    result.realize(out);

    EXPECT_NEAR(out(0), 5.0f, 1e-5f);
    EXPECT_NEAR(out(1), 7.0f, 1e-5f);
    EXPECT_NEAR(out(2), 9.0f, 1e-5f);
}

TEST(Polynomial2, PolyAdd_DifferentSize) {
    // a = [1, 0] (x), size 2
    // b = [3, 2, 1] (3x^2+2x+1), size 3
    // polyadd pads a to [0, 1, 0]: result = [3, 3, 1]
    Halide::Func a("a"), b("b");
    Halide::Var i;
    a(i) = Halide::select(i == 0, 1.0f, 0.0f);
    b(i) = Halide::select(i == 0, 3.0f, i == 1, 2.0f, 1.0f);

    Halide::Func result = polyadd(a, 2, b, 3);

    Halide::Runtime::Buffer<float> out(3);
    result.realize(out);

    EXPECT_NEAR(out(0), 3.0f, 1e-5f);
    EXPECT_NEAR(out(1), 3.0f, 1e-5f);
    EXPECT_NEAR(out(2), 1.0f, 1e-5f);
}

// -----------------------------------------------------------------------------
// polysub Tests
// -----------------------------------------------------------------------------

TEST(Polynomial2, PolySub_Basic) {
    // a = [5, 7, 9], b = [4, 5, 6]
    // result = [1, 2, 3]
    Halide::Func a("a"), b("b");
    Halide::Var i;
    a(i) = Halide::select(i == 0, 5.0f, i == 1, 7.0f, 9.0f);
    b(i) = Halide::select(i == 0, 4.0f, i == 1, 5.0f, 6.0f);

    Halide::Func result = polysub(a, 3, b, 3);

    Halide::Runtime::Buffer<float> out(3);
    result.realize(out);

    EXPECT_NEAR(out(0), 1.0f, 1e-5f);
    EXPECT_NEAR(out(1), 2.0f, 1e-5f);
    EXPECT_NEAR(out(2), 3.0f, 1e-5f);
}

// -----------------------------------------------------------------------------
// polymul Tests
// -----------------------------------------------------------------------------

TEST(Polynomial2, PolyMul_Basic) {
    // a = [1, 1] (x+1), b = [1, -1] (x-1)
    // result = [1, 0, -1] (x^2-1), size = 2+2-1 = 3
    Halide::Func a("a"), b("b");
    Halide::Var i;
    a(i) = Halide::select(i == 0, 1.0f, 1.0f);
    b(i) = Halide::select(i == 0, 1.0f, -1.0f);

    Halide::Func result = polymul(a, 2, b, 2);

    Halide::Runtime::Buffer<float> out(3);
    result.realize(out);

    EXPECT_NEAR(out(0), 1.0f, 1e-5f);   // x^2 coefficient
    EXPECT_NEAR(out(1), 0.0f, 1e-5f);   // x coefficient (1*(-1) + 1*1 = 0)
    EXPECT_NEAR(out(2), -1.0f, 1e-5f);  // constant
}

TEST(Polynomial2, PolyMul_Scalar) {
    // a = [1, 2, 3], b = [2] (constant 2)
    // result = [2, 4, 6], size = 3+1-1 = 3
    Halide::Func a("a"), b("b");
    Halide::Var i;
    a(i) = Halide::select(i == 0, 1.0f, i == 1, 2.0f, 3.0f);
    b(i) = 2.0f;

    Halide::Func result = polymul(a, 3, b, 1);

    Halide::Runtime::Buffer<float> out(3);
    result.realize(out);

    EXPECT_NEAR(out(0), 2.0f, 1e-5f);
    EXPECT_NEAR(out(1), 4.0f, 1e-5f);
    EXPECT_NEAR(out(2), 6.0f, 1e-5f);
}

// -----------------------------------------------------------------------------
// polyder Tests
// -----------------------------------------------------------------------------

TEST(Polynomial2, PolyDer_Linear) {
    // a = [3, 2, 1] (3x^2+2x+1)
    // polyder(a, 3, 1): d/dx = 6x+2 -> [6, 2], size 2
    Halide::Func a("a");
    Halide::Var i;
    a(i) = Halide::select(i == 0, 3.0f, i == 1, 2.0f, 1.0f);

    Halide::Func result = polyder(a, 3, 1);

    Halide::Runtime::Buffer<float> out(2);
    result.realize(out);

    EXPECT_NEAR(out(0), 6.0f, 1e-5f);  // 3 * 2 = 6 (power 2 -> 1)
    EXPECT_NEAR(out(1), 2.0f, 1e-5f);  // 2 * 1 = 2 (power 1 -> 0)
}

TEST(Polynomial2, PolyDer_Constant) {
    // a = [5] (constant), polyder(a, 1) = [0]
    Halide::Func a("a");
    Halide::Var i;
    a(i) = 5.0f;

    Halide::Func result = polyder(a, 1);

    Halide::Runtime::Buffer<float> out(1);
    result.realize(out);

    EXPECT_NEAR(out(0), 0.0f, 1e-5f);
}

TEST(Polynomial2, PolyDer_SecondOrder) {
    // a = [1, 0, 0] (x^2), polyder(a, 3, 2) = [2] (constant 2)
    // First derivative: 2x -> [2, 0]
    // Second derivative: 2 -> [2]
    Halide::Func a("a");
    Halide::Var i;
    a(i) = Halide::select(i == 0, 1.0f, 0.0f);

    Halide::Func result = polyder(a, 3, 2);

    Halide::Runtime::Buffer<float> out(1);
    result.realize(out);

    EXPECT_NEAR(out(0), 2.0f, 1e-5f);
}

// -----------------------------------------------------------------------------
// polyint Tests
// -----------------------------------------------------------------------------

TEST(Polynomial2, PolyInt_Basic) {
    // a = [3, 2] (3x+2), polyint(a, 2, 0.0f)
    // integral: 3x+2 -> (3/2)x^2 + 2x + C
    // result[0] = a[0]/(2-0) = 3/2 = 1.5
    // result[1] = a[1]/(2-1) = 2/1 = 2.0
    // result[2] = k = 0.0
    // Output: [1.5, 2.0, 0.0], size 3
    Halide::Func a("a");
    Halide::Var i;
    a(i) = Halide::select(i == 0, 3.0f, 2.0f);

    Halide::Func result = polyint(a, 2, 0.0f);

    Halide::Runtime::Buffer<float> out(3);
    result.realize(out);

    EXPECT_NEAR(out(0), 1.5f, 1e-5f);
    EXPECT_NEAR(out(1), 2.0f, 1e-5f);
    EXPECT_NEAR(out(2), 0.0f, 1e-5f);
}

TEST(Polynomial2, PolyInt_WithConstant) {
    // a = [1] (constant 1), polyint(a, 1, 5.0f)
    // integral: 1 -> x + C
    // result[0] = a[0]/(1-0) = 1/1 = 1.0
    // result[1] = k = 5.0
    // Output: [1.0, 5.0], size 2
    Halide::Func a("a");
    Halide::Var i;
    a(i) = 1.0f;

    Halide::Func result = polyint(a, 1, 5.0f);

    Halide::Runtime::Buffer<float> out(2);
    result.realize(out);

    EXPECT_NEAR(out(0), 1.0f, 1e-5f);
    EXPECT_NEAR(out(1), 5.0f, 1e-5f);
}
