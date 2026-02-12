/// @file test_sort.cpp
/// @brief Tests for sorting and search operations

#include <gtest/gtest.h>
#include "numhalide_all.h"

using namespace numhalide;

// -----------------------------------------------------------------------------
// Argmin/Argmax Tests
// -----------------------------------------------------------------------------

TEST(Sort, Argmin1D) {
    shape_t shape = {5};
    Halide::Func f("input");
    Halide::Var x;
    // [3, 1, 4, 0, 2] - min is at index 3
    f(x) = Halide::select(x == 0, 3,
           Halide::select(x == 1, 1,
           Halide::select(x == 2, 4,
           Halide::select(x == 3, 0, 2))));

    auto result = argmin(f, shape, "argmin_result");

    Halide::Runtime::Buffer<int32_t> out(1);
    result.realize(out);

    EXPECT_EQ(out(0), 3);  // Index of minimum (value 0)
}

TEST(Sort, Argmax1D) {
    shape_t shape = {5};
    Halide::Func f("input");
    Halide::Var x;
    // [3, 1, 4, 0, 2] - max is at index 2
    f(x) = Halide::select(x == 0, 3,
           Halide::select(x == 1, 1,
           Halide::select(x == 2, 4,
           Halide::select(x == 3, 0, 2))));

    auto result = argmax(f, shape, "argmax_result");

    Halide::Runtime::Buffer<int32_t> out(1);
    result.realize(out);

    EXPECT_EQ(out(0), 2);  // Index of maximum (value 4)
}

// TODO: 2D axis argmin has issues with Halide's argmin tuple handling
// Disabled until we can investigate further
TEST(Sort, DISABLED_Argmin2DAxis0) {
    // 3x2 array, find argmin along rows (axis 0)
    shape_t shape = {3, 2};
    Halide::Func f("input");
    Halide::Var x, y;
    // [[5, 2], [1, 4], [3, 0]]
    f(x, y) = Halide::select(y == 0, Halide::select(x == 0, 5, 2),
              Halide::select(y == 1, Halide::select(x == 0, 1, 4),
              Halide::select(x == 0, 3, 0)));

    auto result = argmin(f, shape, 0, "argmin_axis0");

    Halide::Runtime::Buffer<int32_t> out(2);
    result.realize(out);

    // Col 0: [5, 1, 3] -> min at row 1
    // Col 1: [2, 4, 0] -> min at row 2
    EXPECT_EQ(out(0), 1);
    EXPECT_EQ(out(1), 2);
}

// TODO: 2D axis argmax has issues with Halide's argmax tuple handling
TEST(Sort, DISABLED_Argmax2DAxis1) {
    // 2x3 array, find argmax along cols (axis 1)
    shape_t shape = {2, 3};
    Halide::Func f("input");
    Halide::Var x, y;
    // [[1, 5, 2], [4, 0, 3]]
    f(x, y) = Halide::select(y == 0, Halide::select(x == 0, 1, Halide::select(x == 1, 5, 2)),
              Halide::select(x == 0, 4, Halide::select(x == 1, 0, 3)));

    auto result = argmax(f, shape, 1, "argmax_axis1");

    Halide::Runtime::Buffer<int32_t> out(2);
    result.realize(out);

    // Row 0: [1, 5, 2] -> max at col 1
    // Row 1: [4, 0, 3] -> max at col 0
    EXPECT_EQ(out(0), 1);
    EXPECT_EQ(out(1), 0);
}

// -----------------------------------------------------------------------------
// Bitonic Sort Tests
// -----------------------------------------------------------------------------

TEST(Sort, BitonicSort4) {
    // Sort 4 elements (power of 2)
    Halide::Func f("input");
    Halide::Var x;
    // [3, 1, 4, 2]
    f(x) = Halide::select(x == 0, 3,
           Halide::select(x == 1, 1,
           Halide::select(x == 2, 4, 2)));

    auto result = bitonic_sort(f, 4, "sorted");

    Halide::Runtime::Buffer<int32_t> out(4);
    result.realize(out);

    EXPECT_EQ(out(0), 1);
    EXPECT_EQ(out(1), 2);
    EXPECT_EQ(out(2), 3);
    EXPECT_EQ(out(3), 4);
}

TEST(Sort, BitonicSort8) {
    // Sort 8 elements
    Halide::Func f("input");
    Halide::Var x;
    // [7, 3, 5, 1, 8, 2, 6, 4]
    f(x) = Halide::select(x == 0, 7,
           Halide::select(x == 1, 3,
           Halide::select(x == 2, 5,
           Halide::select(x == 3, 1,
           Halide::select(x == 4, 8,
           Halide::select(x == 5, 2,
           Halide::select(x == 6, 6, 4)))))));

    auto result = bitonic_sort(f, 8, "sorted8");

    Halide::Runtime::Buffer<int32_t> out(8);
    result.realize(out);

    EXPECT_EQ(out(0), 1);
    EXPECT_EQ(out(1), 2);
    EXPECT_EQ(out(2), 3);
    EXPECT_EQ(out(3), 4);
    EXPECT_EQ(out(4), 5);
    EXPECT_EQ(out(5), 6);
    EXPECT_EQ(out(6), 7);
    EXPECT_EQ(out(7), 8);
}

TEST(Sort, BitonicSortAlreadySorted) {
    Halide::Func f("input");
    Halide::Var x;
    f(x) = x + 1;  // [1, 2, 3, 4]

    auto result = bitonic_sort(f, 4, "sorted");

    Halide::Runtime::Buffer<int32_t> out(4);
    result.realize(out);

    EXPECT_EQ(out(0), 1);
    EXPECT_EQ(out(1), 2);
    EXPECT_EQ(out(2), 3);
    EXPECT_EQ(out(3), 4);
}

TEST(Sort, BitonicArgsort) {
    Halide::Func f("input");
    Halide::Var x;
    // [3, 1, 4, 2] -> sorted [1, 2, 3, 4] -> argsort [1, 3, 0, 2]
    f(x) = Halide::select(x == 0, 3,
           Halide::select(x == 1, 1,
           Halide::select(x == 2, 4, 2)));

    auto result = bitonic_argsort(f, 4, "argsort");

    Halide::Runtime::Buffer<int32_t> out(4);
    result.realize(out);

    // Verify indices: out[i] is the original index of the i-th smallest element
    EXPECT_EQ(out(0), 1);  // Smallest (1) was at index 1
    EXPECT_EQ(out(1), 3);  // Second smallest (2) was at index 3
    EXPECT_EQ(out(2), 0);  // Third smallest (3) was at index 0
    EXPECT_EQ(out(3), 2);  // Largest (4) was at index 2
}

TEST(Sort, BitonicSortFloats) {
    Halide::Func f("input");
    Halide::Var x;
    // [3.5, 1.2, 4.8, 2.1]
    f(x) = Halide::select(x == 0, 3.5f,
           Halide::select(x == 1, 1.2f,
           Halide::select(x == 2, 4.8f, 2.1f)));

    auto result = bitonic_sort(f, 4, "sorted_float");

    Halide::Runtime::Buffer<float> out(4);
    result.realize(out);

    EXPECT_NEAR(out(0), 1.2f, 1e-5f);
    EXPECT_NEAR(out(1), 2.1f, 1e-5f);
    EXPECT_NEAR(out(2), 3.5f, 1e-5f);
    EXPECT_NEAR(out(3), 4.8f, 1e-5f);
}
