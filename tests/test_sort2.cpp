/// @file test_sort2.cpp
/// @brief Tests for sort_1d and argsort_1d operations

#include <gtest/gtest.h>
#include "numhalide_all.h"
#include <cmath>

using namespace numhalide;

// -----------------------------------------------------------------------------
// sort_1d Tests
// -----------------------------------------------------------------------------

TEST(Sort2, Sort1D_Ascending) {
    // Input: [3.0, 1.0, 4.0, 2.0], sorted ascending: [1.0, 2.0, 3.0, 4.0]
    Halide::Func input("input");
    Halide::Var x;
    input(x) = Halide::select(x == 0, 3.0f,
               Halide::select(x == 1, 1.0f,
               Halide::select(x == 2, 4.0f, 2.0f)));

    Halide::Func result = sort_1d(input, 4, true);

    Halide::Runtime::Buffer<float> out(4);
    result.realize(out);

    EXPECT_NEAR(out(0), 1.0f, 1e-5f);
    EXPECT_NEAR(out(1), 2.0f, 1e-5f);
    EXPECT_NEAR(out(2), 3.0f, 1e-5f);
    EXPECT_NEAR(out(3), 4.0f, 1e-5f);
}

TEST(Sort2, Sort1D_Descending) {
    // Input: [3.0, 1.0, 4.0, 2.0], sorted descending: [4.0, 3.0, 2.0, 1.0]
    Halide::Func input("input");
    Halide::Var x;
    input(x) = Halide::select(x == 0, 3.0f,
               Halide::select(x == 1, 1.0f,
               Halide::select(x == 2, 4.0f, 2.0f)));

    Halide::Func result = sort_1d(input, 4, false);

    Halide::Runtime::Buffer<float> out(4);
    result.realize(out);

    EXPECT_NEAR(out(0), 4.0f, 1e-5f);
    EXPECT_NEAR(out(1), 3.0f, 1e-5f);
    EXPECT_NEAR(out(2), 2.0f, 1e-5f);
    EXPECT_NEAR(out(3), 1.0f, 1e-5f);
}

TEST(Sort2, Sort1D_Duplicates) {
    // Input: [3.0, 1.0, 3.0, 2.0], sorted ascending: [1.0, 2.0, 3.0, 3.0]
    Halide::Func input("input");
    Halide::Var x;
    input(x) = Halide::select(x == 0, 3.0f,
               Halide::select(x == 1, 1.0f,
               Halide::select(x == 2, 3.0f, 2.0f)));

    Halide::Func result = sort_1d(input, 4, true);

    Halide::Runtime::Buffer<float> out(4);
    result.realize(out);

    EXPECT_NEAR(out(0), 1.0f, 1e-5f);
    EXPECT_NEAR(out(1), 2.0f, 1e-5f);
    EXPECT_NEAR(out(2), 3.0f, 1e-5f);
    EXPECT_NEAR(out(3), 3.0f, 1e-5f);
}

TEST(Sort2, Sort1D_Singleton) {
    // Input: [5.0], sorted: [5.0]
    Halide::Func input("input");
    Halide::Var x;
    input(x) = 5.0f;

    Halide::Func result = sort_1d(input, 1);

    Halide::Runtime::Buffer<float> out(1);
    result.realize(out);

    EXPECT_NEAR(out(0), 5.0f, 1e-5f);
}

TEST(Sort2, Sort1D_NonPow2) {
    // Input: [3.0, 1.0, 2.0], n=3 (not a power of 2)
    // sorted ascending: [1.0, 2.0, 3.0]
    Halide::Func input("input");
    Halide::Var x;
    input(x) = Halide::select(x == 0, 3.0f,
               Halide::select(x == 1, 1.0f, 2.0f));

    Halide::Func result = sort_1d(input, 3, true);

    Halide::Runtime::Buffer<float> out(3);
    result.realize(out);

    EXPECT_NEAR(out(0), 1.0f, 1e-5f);
    EXPECT_NEAR(out(1), 2.0f, 1e-5f);
    EXPECT_NEAR(out(2), 3.0f, 1e-5f);
}

// -----------------------------------------------------------------------------
// argsort_1d Tests
// -----------------------------------------------------------------------------

TEST(Sort2, Argsort1D_Ascending) {
    // Input: [3.0, 1.0, 4.0, 2.0]
    // argsort ascending returns indices: [1, 3, 0, 2]
    // (index of 1.0=1, index of 2.0=3, index of 3.0=0, index of 4.0=2)
    Halide::Func input("input");
    Halide::Var x;
    input(x) = Halide::select(x == 0, 3.0f,
               Halide::select(x == 1, 1.0f,
               Halide::select(x == 2, 4.0f, 2.0f)));

    Halide::Func result = argsort_1d(input, 4, true);

    Halide::Runtime::Buffer<int32_t> out(4);
    result.realize(out);

    EXPECT_EQ(out(0), 1);  // smallest (1.0) was at index 1
    EXPECT_EQ(out(1), 3);  // second (2.0) was at index 3
    EXPECT_EQ(out(2), 0);  // third (3.0) was at index 0
    EXPECT_EQ(out(3), 2);  // largest (4.0) was at index 2
}

TEST(Sort2, Argsort1D_Descending) {
    // Input: [3.0, 1.0, 4.0, 2.0]
    // argsort descending returns indices: [2, 0, 3, 1]
    // (index of 4.0=2, index of 3.0=0, index of 2.0=3, index of 1.0=1)
    Halide::Func input("input");
    Halide::Var x;
    input(x) = Halide::select(x == 0, 3.0f,
               Halide::select(x == 1, 1.0f,
               Halide::select(x == 2, 4.0f, 2.0f)));

    Halide::Func result = argsort_1d(input, 4, false);

    Halide::Runtime::Buffer<int32_t> out(4);
    result.realize(out);

    EXPECT_EQ(out(0), 2);  // largest (4.0) was at index 2
    EXPECT_EQ(out(1), 0);  // second (3.0) was at index 0
    EXPECT_EQ(out(2), 3);  // third (2.0) was at index 3
    EXPECT_EQ(out(3), 1);  // smallest (1.0) was at index 1
}
