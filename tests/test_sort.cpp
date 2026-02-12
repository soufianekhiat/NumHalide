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

TEST(Sort, Argmin2DAxis0) {
    // 3x2 array, find argmin along rows (axis 0)
    // shape = {rows=3, cols=2}
    // Halide: f(x, y) where x=cols, y=rows
    // [[5, 2], [1, 4], [3, 0]]
    const int rows = 3, cols = 2;

    Halide::Func f("input");
    Halide::Var x("x"), y("y");
    f(x, y) = Halide::select(y == 0, Halide::select(x == 0, 5, 2),
              Halide::select(y == 1, Halide::select(x == 0, 1, 4),
              Halide::select(x == 0, 3, 0)));
    f.compute_root();

    // Manual argmin along rows (y dimension)
    Halide::Func ret("argmin_axis0");
    Halide::RDom r(0, rows);
    ret(x) = 0;
    Halide::Expr best = f(x, Halide::clamp(ret(x), 0, rows - 1));
    ret(x) = Halide::select(f(x, r) < best, r, ret(x));

    Halide::Runtime::Buffer<int32_t> out(cols);
    ret.realize(out);

    // Col 0: [5, 1, 3] -> min at row 1
    // Col 1: [2, 4, 0] -> min at row 2
    EXPECT_EQ(out(0), 1);
    EXPECT_EQ(out(1), 2);
}

TEST(Sort, Argmax2DAxis1) {
    // 2x3 array, find argmax along cols (axis 1)
    // shape = {rows=2, cols=3}
    // Halide: f(x, y) where x=cols, y=rows
    // [[1, 5, 2], [4, 0, 3]]
    const int rows = 2, cols = 3;

    Halide::Func f("input");
    Halide::Var x("x"), y("y");
    f(x, y) = Halide::select(y == 0, Halide::select(x == 0, 1, Halide::select(x == 1, 5, 2)),
              Halide::select(x == 0, 4, Halide::select(x == 1, 0, 3)));
    f.compute_root();

    // Manual argmax along cols (x dimension)
    Halide::Func ret("argmax_axis1");
    Halide::RDom r(0, cols);
    ret(y) = 0;
    Halide::Expr best = f(Halide::clamp(ret(y), 0, cols - 1), y);
    ret(y) = Halide::select(f(r, y) > best, r, ret(y));

    Halide::Runtime::Buffer<int32_t> out(rows);
    ret.realize(out);

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
