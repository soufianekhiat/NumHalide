/// @file test_bool_reduce.cpp
/// @brief Tests for boolean reduction operations

#include <gtest/gtest.h>
#include "numhalide_all.h"

using namespace numhalide;

// -----------------------------------------------------------------------------
// Any Tests
// -----------------------------------------------------------------------------

TEST(BoolReduce, AnyTrue) {
    // Array with some non-zero values
    shape_t shape = {4};
    Halide::Func f("input");
    Halide::Var x;
    f(x) = Halide::select(x == 2, 1, 0);  // [0, 0, 1, 0]

    auto result = reduce_any(f, shape, "any_result");

    Halide::Runtime::Buffer<int32_t> out(1);
    result.realize(out);

    EXPECT_EQ(out(0), 1);  // Should be true (1)
}

TEST(BoolReduce, AnyFalse) {
    // Array with all zeros
    shape_t shape = {4};
    Halide::Func f("input");
    Halide::Var x;
    f(x) = 0;

    auto result = reduce_any(f, shape, "any_false");

    Halide::Runtime::Buffer<int32_t> out(1);
    result.realize(out);

    EXPECT_EQ(out(0), 0);  // Should be false (0)
}

TEST(BoolReduce, AnyAxis) {
    // 2D array:
    // [[0, 1],
    //  [0, 0],
    //  [1, 0]]
    // any along axis 0 (rows): [1, 1] (both cols have at least one non-zero)
    // any along axis 1 (cols): [1, 0, 1] (rows 0 and 2 have non-zeros)
    shape_t shape = {3, 2};
    Halide::Func f("input");
    Halide::Var x, y;
    // Create: row 0: [0,1], row 1: [0,0], row 2: [1,0]
    f(x, y) = Halide::select(
        (y == 0 && x == 1) || (y == 2 && x == 0),
        1, 0
    );

    // Test axis 0
    auto result0 = reduce_any(f, shape, 0, false, "any_axis0");
    Halide::Runtime::Buffer<int32_t> out0(2);
    result0.realize(out0);
    EXPECT_EQ(out0(0), 1);  // col 0: has 1 in row 2
    EXPECT_EQ(out0(1), 1);  // col 1: has 1 in row 0

    // Test axis 1
    auto result1 = reduce_any(f, shape, 1, false, "any_axis1");
    Halide::Runtime::Buffer<int32_t> out1(3);
    result1.realize(out1);
    EXPECT_EQ(out1(0), 1);  // row 0: has 1
    EXPECT_EQ(out1(1), 0);  // row 1: all zeros
    EXPECT_EQ(out1(2), 1);  // row 2: has 1
}

// -----------------------------------------------------------------------------
// All Tests
// -----------------------------------------------------------------------------

TEST(BoolReduce, AllTrue) {
    // Array with all non-zero values
    shape_t shape = {4};
    Halide::Func f("input");
    Halide::Var x;
    f(x) = x + 1;  // [1, 2, 3, 4]

    auto result = reduce_all(f, shape, "all_true");

    Halide::Runtime::Buffer<int32_t> out(1);
    result.realize(out);

    EXPECT_EQ(out(0), 1);  // Should be true (1)
}

TEST(BoolReduce, AllFalse) {
    // Array with at least one zero
    shape_t shape = {4};
    Halide::Func f("input");
    Halide::Var x;
    f(x) = Halide::select(x == 2, 0, 1);  // [1, 1, 0, 1]

    auto result = reduce_all(f, shape, "all_false");

    Halide::Runtime::Buffer<int32_t> out(1);
    result.realize(out);

    EXPECT_EQ(out(0), 0);  // Should be false (0)
}

TEST(BoolReduce, AllAxis) {
    // 2D array:
    // [[1, 1],
    //  [1, 0],
    //  [1, 1]]
    // all along axis 0 (rows): [1, 0] (col 0 all non-zero, col 1 has zero)
    // all along axis 1 (cols): [1, 0, 1]
    shape_t shape = {3, 2};
    Halide::Func f("input");
    Halide::Var x, y;
    // Create: all 1s except (x=1, y=1) which is 0
    f(x, y) = Halide::select(x == 1 && y == 1, 0, 1);

    // Test axis 0
    auto result0 = reduce_all(f, shape, 0, false, "all_axis0");
    Halide::Runtime::Buffer<int32_t> out0(2);
    result0.realize(out0);
    EXPECT_EQ(out0(0), 1);  // col 0: all non-zero
    EXPECT_EQ(out0(1), 0);  // col 1: has zero in row 1

    // Test axis 1
    auto result1 = reduce_all(f, shape, 1, false, "all_axis1");
    Halide::Runtime::Buffer<int32_t> out1(3);
    result1.realize(out1);
    EXPECT_EQ(out1(0), 1);  // row 0: all non-zero
    EXPECT_EQ(out1(1), 0);  // row 1: has zero
    EXPECT_EQ(out1(2), 1);  // row 2: all non-zero
}

// -----------------------------------------------------------------------------
// Count Nonzero Tests
// -----------------------------------------------------------------------------

TEST(BoolReduce, CountNonzero) {
    // Array with some zeros
    shape_t shape = {5};
    Halide::Func f("input");
    Halide::Var x;
    f(x) = Halide::select(x % 2 == 0, x + 1, 0);  // [1, 0, 3, 0, 5]

    auto result = count_nonzero(f, shape, "count_result");

    Halide::Runtime::Buffer<int32_t> out(1);
    result.realize(out);

    EXPECT_EQ(out(0), 3);  // 3 non-zero elements
}

TEST(BoolReduce, CountNonzeroAxis) {
    // 2D array:
    // [[1, 0, 2],
    //  [0, 0, 3],
    //  [4, 5, 0]]
    // count along axis 0: [2, 1, 2]
    // count along axis 1: [2, 1, 2]
    shape_t shape = {3, 3};
    Halide::Func f("input");
    Halide::Var x, y;
    // Row 0: [1, 0, 2], Row 1: [0, 0, 3], Row 2: [4, 5, 0]
    f(x, y) = Halide::select(
        (y == 0 && x == 0), 1,
        Halide::select(
            (y == 0 && x == 2), 2,
            Halide::select(
                (y == 1 && x == 2), 3,
                Halide::select(
                    (y == 2 && x == 0), 4,
                    Halide::select(
                        (y == 2 && x == 1), 5,
                        0
                    )
                )
            )
        )
    );

    // Test axis 0 (reduce rows, result per column)
    auto result0 = count_nonzero(f, shape, 0, false, "count_axis0");
    Halide::Runtime::Buffer<int32_t> out0(3);
    result0.realize(out0);
    EXPECT_EQ(out0(0), 2);  // col 0: 1, 0, 4 -> 2 non-zero
    EXPECT_EQ(out0(1), 1);  // col 1: 0, 0, 5 -> 1 non-zero
    EXPECT_EQ(out0(2), 2);  // col 2: 2, 3, 0 -> 2 non-zero

    // Test axis 1 (reduce cols, result per row)
    auto result1 = count_nonzero(f, shape, 1, false, "count_axis1");
    Halide::Runtime::Buffer<int32_t> out1(3);
    result1.realize(out1);
    EXPECT_EQ(out1(0), 2);  // row 0: [1, 0, 2] -> 2 non-zero
    EXPECT_EQ(out1(1), 1);  // row 1: [0, 0, 3] -> 1 non-zero
    EXPECT_EQ(out1(2), 2);  // row 2: [4, 5, 0] -> 2 non-zero
}

TEST(BoolReduce, CountNonzeroAllZeros) {
    // Array of all zeros
    shape_t shape = {4, 4};
    Halide::Func f("zeros");
    Halide::Var x, y;
    f(x, y) = 0;

    auto result = count_nonzero(f, shape, "count_zeros");

    Halide::Runtime::Buffer<int32_t> out(1);
    result.realize(out);

    EXPECT_EQ(out(0), 0);
}

TEST(BoolReduce, CountNonzeroAllNonzero) {
    // Array of all non-zeros
    shape_t shape = {3, 4};
    Halide::Func f("nonzeros");
    Halide::Var x, y;
    f(x, y) = y * 4 + x + 1;  // [1..12]

    auto result = count_nonzero(f, shape, "count_nonzeros");

    Halide::Runtime::Buffer<int32_t> out(1);
    result.realize(out);

    EXPECT_EQ(out(0), 12);  // All 12 elements are non-zero
}

TEST(BoolReduce, AnyWithFloats) {
    // Test with float values (any non-zero float should count)
    shape_t shape = {4};
    Halide::Func f("float_input");
    Halide::Var x;
    f(x) = Halide::select(x == 1, 0.5f, 0.0f);  // [0.0, 0.5, 0.0, 0.0]

    auto result = reduce_any(f, shape, "any_float");

    Halide::Runtime::Buffer<int32_t> out(1);
    result.realize(out);

    EXPECT_EQ(out(0), 1);  // 0.5 is non-zero
}

TEST(BoolReduce, AllWithNegatives) {
    // Test with negative values (negative is non-zero)
    shape_t shape = {4};
    Halide::Func f("neg_input");
    Halide::Var x;
    f(x) = Halide::cast<int32_t>(x) - 2;  // [-2, -1, 0, 1]

    auto result = reduce_all(f, shape, "all_neg");

    Halide::Runtime::Buffer<int32_t> out(1);
    result.realize(out);

    EXPECT_EQ(out(0), 0);  // There's a 0 at x=2
}
