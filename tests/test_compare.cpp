/// @file test_compare.cpp
/// @brief Tests for comparison and logical operations

#include <gtest/gtest.h>
#include "numhalide_all.h"
#include <cmath>
#include <limits>

using namespace numhalide;

// -----------------------------------------------------------------------------
// Equality Tests
// -----------------------------------------------------------------------------

TEST(Compare, Equal) {
    shape_t shape = {4};
    Halide::Func a("a"), b("b");
    Halide::Var x;
    a(x) = x;           // [0, 1, 2, 3]
    b(x) = 2 - x + x;   // [2, 2, 2, 2]

    // Func-Func comparison
    Halide::Func c("c");
    c(x) = x;
    auto result = equal(a, c, shape, "eq_result");

    Halide::Runtime::Buffer<int32_t> out(4);
    result.realize(out);

    EXPECT_EQ(out(0), 1);  // 0 == 0
    EXPECT_EQ(out(1), 1);  // 1 == 1
    EXPECT_EQ(out(2), 1);  // 2 == 2
    EXPECT_EQ(out(3), 1);  // 3 == 3
}

TEST(Compare, EqualScalar) {
    shape_t shape = {4};
    Halide::Func a("a");
    Halide::Var x;
    a(x) = x;  // [0, 1, 2, 3]

    auto result = equal(a, 2, shape, "eq_scalar");

    Halide::Runtime::Buffer<int32_t> out(4);
    result.realize(out);

    EXPECT_EQ(out(0), 0);  // 0 == 2? no
    EXPECT_EQ(out(1), 0);  // 1 == 2? no
    EXPECT_EQ(out(2), 1);  // 2 == 2? yes
    EXPECT_EQ(out(3), 0);  // 3 == 2? no
}

TEST(Compare, NotEqual) {
    shape_t shape = {4};
    Halide::Func a("a"), b("b");
    Halide::Var x;
    a(x) = x;      // [0, 1, 2, 3]
    b(x) = 3 - x;  // [3, 2, 1, 0]

    auto result = not_equal(a, b, shape, "neq_result");

    Halide::Runtime::Buffer<int32_t> out(4);
    result.realize(out);

    EXPECT_EQ(out(0), 1);  // 0 != 3
    EXPECT_EQ(out(1), 1);  // 1 != 2
    EXPECT_EQ(out(2), 1);  // 2 != 1
    EXPECT_EQ(out(3), 1);  // 3 != 0
}

TEST(Compare, Greater) {
    shape_t shape = {4};
    Halide::Func a("a"), b("b");
    Halide::Var x;
    a(x) = x;  // [0, 1, 2, 3]
    b(x) = 1;  // [1, 1, 1, 1]

    auto result = greater(a, b, shape, "gt_result");

    Halide::Runtime::Buffer<int32_t> out(4);
    result.realize(out);

    EXPECT_EQ(out(0), 0);  // 0 > 1? no
    EXPECT_EQ(out(1), 0);  // 1 > 1? no
    EXPECT_EQ(out(2), 1);  // 2 > 1? yes
    EXPECT_EQ(out(3), 1);  // 3 > 1? yes
}

TEST(Compare, Less) {
    shape_t shape = {4};
    Halide::Func a("a");
    Halide::Var x;
    a(x) = x;  // [0, 1, 2, 3]

    auto result = less(a, 2, shape, "lt_result");

    Halide::Runtime::Buffer<int32_t> out(4);
    result.realize(out);

    EXPECT_EQ(out(0), 1);  // 0 < 2? yes
    EXPECT_EQ(out(1), 1);  // 1 < 2? yes
    EXPECT_EQ(out(2), 0);  // 2 < 2? no
    EXPECT_EQ(out(3), 0);  // 3 < 2? no
}

TEST(Compare, GreaterEqual) {
    shape_t shape = {4};
    Halide::Func a("a");
    Halide::Var x;
    a(x) = x;  // [0, 1, 2, 3]

    auto result = greater_equal(a, 2, shape, "ge_result");

    Halide::Runtime::Buffer<int32_t> out(4);
    result.realize(out);

    EXPECT_EQ(out(0), 0);  // 0 >= 2? no
    EXPECT_EQ(out(1), 0);  // 1 >= 2? no
    EXPECT_EQ(out(2), 1);  // 2 >= 2? yes
    EXPECT_EQ(out(3), 1);  // 3 >= 2? yes
}

TEST(Compare, LessEqual) {
    shape_t shape = {4};
    Halide::Func a("a");
    Halide::Var x;
    a(x) = x;  // [0, 1, 2, 3]

    auto result = less_equal(a, 2, shape, "le_result");

    Halide::Runtime::Buffer<int32_t> out(4);
    result.realize(out);

    EXPECT_EQ(out(0), 1);  // 0 <= 2? yes
    EXPECT_EQ(out(1), 1);  // 1 <= 2? yes
    EXPECT_EQ(out(2), 1);  // 2 <= 2? yes
    EXPECT_EQ(out(3), 0);  // 3 <= 2? no
}

// -----------------------------------------------------------------------------
// Logical Tests
// -----------------------------------------------------------------------------

TEST(Compare, LogicalAnd) {
    shape_t shape = {4};
    Halide::Func a("a"), b("b");
    Halide::Var x;
    a(x) = Halide::select(x < 2, 1, 0);  // [1, 1, 0, 0]
    b(x) = Halide::select(x % 2 == 0, 1, 0);  // [1, 0, 1, 0]

    auto result = logical_and(a, b, shape, "and_result");

    Halide::Runtime::Buffer<int32_t> out(4);
    result.realize(out);

    EXPECT_EQ(out(0), 1);  // 1 && 1
    EXPECT_EQ(out(1), 0);  // 1 && 0
    EXPECT_EQ(out(2), 0);  // 0 && 1
    EXPECT_EQ(out(3), 0);  // 0 && 0
}

TEST(Compare, LogicalOr) {
    shape_t shape = {4};
    Halide::Func a("a"), b("b");
    Halide::Var x;
    a(x) = Halide::select(x < 2, 1, 0);  // [1, 1, 0, 0]
    b(x) = Halide::select(x % 2 == 0, 1, 0);  // [1, 0, 1, 0]

    auto result = logical_or(a, b, shape, "or_result");

    Halide::Runtime::Buffer<int32_t> out(4);
    result.realize(out);

    EXPECT_EQ(out(0), 1);  // 1 || 1
    EXPECT_EQ(out(1), 1);  // 1 || 0
    EXPECT_EQ(out(2), 1);  // 0 || 1
    EXPECT_EQ(out(3), 0);  // 0 || 0
}

TEST(Compare, LogicalNot) {
    shape_t shape = {4};
    Halide::Func a("a");
    Halide::Var x;
    a(x) = Halide::select(x < 2, 1, 0);  // [1, 1, 0, 0]

    auto result = logical_not(a, shape, "not_result");

    Halide::Runtime::Buffer<int32_t> out(4);
    result.realize(out);

    EXPECT_EQ(out(0), 0);  // !1
    EXPECT_EQ(out(1), 0);  // !1
    EXPECT_EQ(out(2), 1);  // !0
    EXPECT_EQ(out(3), 1);  // !0
}

TEST(Compare, LogicalXor) {
    shape_t shape = {4};
    Halide::Func a("a"), b("b");
    Halide::Var x;
    a(x) = Halide::select(x < 2, 1, 0);  // [1, 1, 0, 0]
    b(x) = Halide::select(x % 2 == 0, 1, 0);  // [1, 0, 1, 0]

    auto result = logical_xor(a, b, shape, "xor_result");

    Halide::Runtime::Buffer<int32_t> out(4);
    result.realize(out);

    EXPECT_EQ(out(0), 0);  // 1 xor 1 = 0
    EXPECT_EQ(out(1), 1);  // 1 xor 0 = 1
    EXPECT_EQ(out(2), 1);  // 0 xor 1 = 1
    EXPECT_EQ(out(3), 0);  // 0 xor 0 = 0
}

// -----------------------------------------------------------------------------
// Special Value Tests
// -----------------------------------------------------------------------------

TEST(Compare, IsNan) {
    shape_t shape = {4};
    Halide::Func a("a");
    Halide::Var x;
    // Create array with NaN at positions 1 and 3
    float nan_val = std::numeric_limits<float>::quiet_NaN();
    a(x) = Halide::select(x % 2 == 1, nan_val, Halide::cast<float>(x));

    auto result = isnan_func(a, shape, "isnan_result");

    Halide::Runtime::Buffer<int32_t> out(4);
    result.realize(out);

    EXPECT_EQ(out(0), 0);  // 0.0 is not NaN
    EXPECT_EQ(out(1), 1);  // NaN
    EXPECT_EQ(out(2), 0);  // 2.0 is not NaN
    EXPECT_EQ(out(3), 1);  // NaN
}

TEST(Compare, IsInf) {
    shape_t shape = {4};
    Halide::Func a("a");
    Halide::Var x;
    float inf_val = std::numeric_limits<float>::infinity();
    // [0, inf, 2, -inf]
    a(x) = Halide::select(x == 1, inf_val,
           Halide::select(x == 3, -inf_val, Halide::cast<float>(x)));

    auto result = isinf_func(a, shape, "isinf_result");

    Halide::Runtime::Buffer<int32_t> out(4);
    result.realize(out);

    EXPECT_EQ(out(0), 0);  // 0.0 is not inf
    EXPECT_EQ(out(1), 1);  // +inf
    EXPECT_EQ(out(2), 0);  // 2.0 is not inf
    EXPECT_EQ(out(3), 1);  // -inf
}

TEST(Compare, IsFinite) {
    shape_t shape = {4};
    Halide::Func a("a");
    Halide::Var x;
    float inf_val = std::numeric_limits<float>::infinity();
    float nan_val = std::numeric_limits<float>::quiet_NaN();
    // [0, inf, nan, 3]
    a(x) = Halide::select(x == 1, inf_val,
           Halide::select(x == 2, nan_val, Halide::cast<float>(x)));

    auto result = isfinite_func(a, shape, "isfinite_result");

    Halide::Runtime::Buffer<int32_t> out(4);
    result.realize(out);

    EXPECT_EQ(out(0), 1);  // 0.0 is finite
    EXPECT_EQ(out(1), 0);  // inf is not finite
    EXPECT_EQ(out(2), 0);  // nan is not finite
    EXPECT_EQ(out(3), 1);  // 3.0 is finite
}

// -----------------------------------------------------------------------------
// 2D Tests
// -----------------------------------------------------------------------------

TEST(Compare, Greater2D) {
    shape_t shape = {2, 3};
    Halide::Func a("a"), b("b");
    Halide::Var x, y;
    a(x, y) = y * 3 + x;      // [[0,1,2],[3,4,5]]
    b(x, y) = 2;              // [[2,2,2],[2,2,2]]

    auto result = greater(a, b, shape, "gt_2d");

    Halide::Runtime::Buffer<int32_t> out(3, 2);
    result.realize(out);

    // Row 0: [0>2, 1>2, 2>2] = [0, 0, 0]
    EXPECT_EQ(out(0, 0), 0);
    EXPECT_EQ(out(1, 0), 0);
    EXPECT_EQ(out(2, 0), 0);
    // Row 1: [3>2, 4>2, 5>2] = [1, 1, 1]
    EXPECT_EQ(out(0, 1), 1);
    EXPECT_EQ(out(1, 1), 1);
    EXPECT_EQ(out(2, 1), 1);
}
