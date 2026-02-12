/// @file test_set_ops.cpp
/// @brief Tests for set operations

#include <gtest/gtest.h>
#include "numhalide_all.h"

using namespace numhalide;

// Helper to extract first element of tuple
Halide::Func extract_tuple_0(Halide::Func f, const std::string& name = "t0") {
    Halide::Func result(name);
    Halide::Var x("x");
    result(x) = f(x)[0];
    return result;
}

Halide::Func extract_tuple_1(Halide::Func f, const std::string& name = "t1") {
    Halide::Func result(name);
    Halide::Var x("x");
    result(x) = f(x)[1];
    return result;
}

// -----------------------------------------------------------------------------
// Unique Tests
// -----------------------------------------------------------------------------

TEST(SetOps, MarkUnique) {
    // Sorted array with duplicates: [1, 1, 2, 3, 3, 3, 4, 5]
    Halide::Func f("input");
    Halide::Var x;
    f(x) = Halide::select(x == 0 || x == 1, 1,
           Halide::select(x == 2, 2,
           Halide::select(x == 3 || x == 4 || x == 5, 3,
           Halide::select(x == 6, 4, 5))));

    auto marked = mark_unique(f, 8, "marked");
    auto vals = extract_tuple_0(marked, "vals");
    auto marks = extract_tuple_1(marked, "marks");

    Halide::Runtime::Buffer<int32_t> val_out(8);
    Halide::Runtime::Buffer<int32_t> mark_out(8);
    vals.realize(val_out);
    marks.realize(mark_out);

    // First occurrence of each value should be marked
    EXPECT_EQ(mark_out(0), 1);  // First 1
    EXPECT_EQ(mark_out(1), 0);  // Duplicate 1
    EXPECT_EQ(mark_out(2), 1);  // First 2
    EXPECT_EQ(mark_out(3), 1);  // First 3
    EXPECT_EQ(mark_out(4), 0);  // Duplicate 3
    EXPECT_EQ(mark_out(5), 0);  // Duplicate 3
    EXPECT_EQ(mark_out(6), 1);  // First 4
    EXPECT_EQ(mark_out(7), 1);  // First 5
}

TEST(SetOps, CountUnique) {
    // Sorted array: [1, 1, 2, 3, 3, 3, 4, 5] -> 5 unique elements
    Halide::Func f("input");
    Halide::Var x;
    f(x) = Halide::select(x == 0 || x == 1, 1,
           Halide::select(x == 2, 2,
           Halide::select(x == 3 || x == 4 || x == 5, 3,
           Halide::select(x == 6, 4, 5))));

    auto count = count_unique(f, 8, "count");

    Halide::Runtime::Buffer<int32_t> out(1);
    count.realize(out);

    EXPECT_EQ(out(0), 5);
}

TEST(SetOps, CountUniqueAllUnique) {
    // All unique: [0, 1, 2, 3, 4, 5, 6, 7]
    Halide::Func f("input");
    Halide::Var x;
    f(x) = x;

    auto count = count_unique(f, 8, "count");

    Halide::Runtime::Buffer<int32_t> out(1);
    count.realize(out);

    EXPECT_EQ(out(0), 8);
}

TEST(SetOps, CountUniqueAllSame) {
    // All same: [5, 5, 5, 5]
    Halide::Func f("input");
    Halide::Var x;
    f(x) = 5;

    auto count = count_unique(f, 4, "count");

    Halide::Runtime::Buffer<int32_t> out(1);
    count.realize(out);

    EXPECT_EQ(out(0), 1);
}

// -----------------------------------------------------------------------------
// In1d Tests
// -----------------------------------------------------------------------------

TEST(SetOps, In1dBasic) {
    // Test values: [2, 5, 7, 1]
    // Sorted array: [1, 2, 3, 4, 5, 6, 7, 8]
    Halide::Func test_vals("test_vals");
    Halide::Var x;
    test_vals(x) = Halide::select(x == 0, 2,
                   Halide::select(x == 1, 5,
                   Halide::select(x == 2, 7, 1)));

    Halide::Func sorted("sorted");
    sorted(x) = x + 1;  // [1, 2, 3, 4, 5, 6, 7, 8]

    auto result = in1d(test_vals, sorted, 4, 8, "in1d");

    Halide::Runtime::Buffer<int32_t> out(4);
    result.realize(out);

    EXPECT_EQ(out(0), 1);  // 2 is in array
    EXPECT_EQ(out(1), 1);  // 5 is in array
    EXPECT_EQ(out(2), 1);  // 7 is in array
    EXPECT_EQ(out(3), 1);  // 1 is in array
}

TEST(SetOps, In1dNotFound) {
    // Test values: [0, 10, 15, 20]
    // Sorted array: [1, 2, 3, 4, 5, 6, 7, 8]
    Halide::Func test_vals("test_vals");
    Halide::Var x;
    test_vals(x) = Halide::select(x == 0, 0,
                   Halide::select(x == 1, 10,
                   Halide::select(x == 2, 15, 20)));

    Halide::Func sorted("sorted");
    sorted(x) = x + 1;

    auto result = in1d(test_vals, sorted, 4, 8, "in1d");

    Halide::Runtime::Buffer<int32_t> out(4);
    result.realize(out);

    EXPECT_EQ(out(0), 0);  // 0 not in array
    EXPECT_EQ(out(1), 0);  // 10 not in array
    EXPECT_EQ(out(2), 0);  // 15 not in array
    EXPECT_EQ(out(3), 0);  // 20 not in array
}

// -----------------------------------------------------------------------------
// Intersect1d Tests
// -----------------------------------------------------------------------------

TEST(SetOps, Intersect1dSorted) {
    // a: [1, 2, 3, 4] (sorted)
    // b: [2, 3, 5, 6] (sorted)
    // intersection: [2, 3]
    Halide::Func a("a"), b("b");
    Halide::Var x;
    a(x) = x + 1;  // [1, 2, 3, 4]
    b(x) = Halide::select(x == 0, 2,
           Halide::select(x == 1, 3,
           Halide::select(x == 2, 5, 6)));

    auto result = intersect1d_sorted(a, b, 4, 4, "intersect");
    auto vals = extract_tuple_0(result, "vals");
    auto valid = extract_tuple_1(result, "valid");

    Halide::Runtime::Buffer<int32_t> val_out(4);
    Halide::Runtime::Buffer<int32_t> valid_out(4);
    vals.realize(val_out);
    valid.realize(valid_out);

    // Count valid intersection elements
    int count = 0;
    for (int i = 0; i < 4; i++) {
        if (valid_out(i) == 1) count++;
    }
    EXPECT_EQ(count, 2);  // 2 and 3 are in both
}

// -----------------------------------------------------------------------------
// Setdiff1d Tests
// -----------------------------------------------------------------------------

TEST(SetOps, Setdiff1dSorted) {
    // a: [1, 2, 3, 4] (sorted)
    // b: [2, 3, 5, 6] (sorted)
    // difference: [1, 4]
    Halide::Func a("a"), b("b");
    Halide::Var x;
    a(x) = x + 1;
    b(x) = Halide::select(x == 0, 2,
           Halide::select(x == 1, 3,
           Halide::select(x == 2, 5, 6)));

    auto result = setdiff1d_sorted(a, b, 4, 4, "setdiff");
    auto vals = extract_tuple_0(result, "vals");
    auto valid = extract_tuple_1(result, "valid");

    Halide::Runtime::Buffer<int32_t> val_out(4);
    Halide::Runtime::Buffer<int32_t> valid_out(4);
    vals.realize(val_out);
    valid.realize(valid_out);

    // Count valid difference elements
    int count = 0;
    for (int i = 0; i < 4; i++) {
        if (valid_out(i) == 1) count++;
    }
    EXPECT_EQ(count, 2);  // 1 and 4 are only in a
}

// -----------------------------------------------------------------------------
// Union1d Tests
// -----------------------------------------------------------------------------

TEST(SetOps, Union1dSorted) {
    // a: [1, 2, 3, 4] (sorted)
    // b: [3, 4, 5, 6] (sorted)
    // union: [1, 2, 3, 4, 5, 6]
    Halide::Func a("a"), b("b");
    Halide::Var x;
    a(x) = x + 1;  // [1, 2, 3, 4]
    b(x) = x + 3;  // [3, 4, 5, 6]

    auto result = union1d_sorted(a, b, 4, 4, "union");
    auto vals = extract_tuple_0(result, "vals");
    auto valid = extract_tuple_1(result, "valid");

    Halide::Runtime::Buffer<int32_t> val_out(8);
    Halide::Runtime::Buffer<int32_t> valid_out(8);
    vals.realize(val_out);
    valid.realize(valid_out);

    // Count valid union elements
    int count = 0;
    for (int i = 0; i < 8; i++) {
        if (valid_out(i) == 1) count++;
    }
    EXPECT_EQ(count, 6);  // 1, 2, 3, 4, 5, 6
}

// -----------------------------------------------------------------------------
// Full Pipeline Tests (with sorting)
// -----------------------------------------------------------------------------

// Note: Full pipeline tests with automatic sorting are expensive due to
// the bitonic sort scheduling overhead. The _sorted versions above are
// the primary tests. Users should pre-sort their data for best performance.
