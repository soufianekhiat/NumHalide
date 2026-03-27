/// @file test_set_ops2.cpp
/// @brief Tests for setxor1d_sorted and isin

#include <gtest/gtest.h>
#include "numhalide_all.h"
#include <cmath>

using namespace numhalide;

// Helper: extract tuple element 0 (value) from a 1D Tuple Func
static Halide::Func extract_vals(Halide::Func f, const std::string& name = "vals") {
    Halide::Func result(name);
    Halide::Var x("x");
    result(x) = f(x)[0];
    return result;
}

// Helper: extract tuple element 1 (mask) from a 1D Tuple Func
static Halide::Func extract_mask(Halide::Func f, const std::string& name = "mask") {
    Halide::Func result(name);
    Halide::Var x("x");
    result(x) = f(x)[1];
    return result;
}

// -----------------------------------------------------------------------------
// SetXor1D Tests
// -----------------------------------------------------------------------------

TEST(SetOps2, SetXor1D_Basic) {
    // a = [1,2,3,4] (sorted), b = [2,4,6,8] (sorted)
    // Symmetric difference: elements in exactly one array → {1, 3, 6, 8}
    // Output size = size_a + size_b = 8
    // Positions 0..3 are from a section, 4..7 from b section
    // a[0]=1 not in b → valid; a[1]=2 in b → invalid
    // a[2]=3 not in b → valid; a[3]=4 in b → invalid
    // b[0]=2 in a → invalid; b[1]=4 in a → invalid
    // b[2]=6 not in a → valid; b[3]=8 not in a → valid
    Halide::Func a("xor_a"), b("xor_b");
    Halide::Var x;
    a(x) = Halide::select(x == 0, 1.0f, x == 1, 2.0f, x == 2, 3.0f, 4.0f);
    b(x) = Halide::select(x == 0, 2.0f, x == 1, 4.0f, x == 2, 6.0f, 8.0f);

    auto result = setxor1d_sorted(a, b, 4, 4, "xor1d");
    result.compute_root();

    auto vals = extract_vals(result, "xor_vals");
    auto mask = extract_mask(result, "xor_mask");

    Halide::Runtime::Buffer<float>   vout(8);
    Halide::Runtime::Buffer<int32_t> mout(8);
    vals.realize(vout);
    mask.realize(mout);

    // Mask checks
    EXPECT_EQ(mout(0), 1);  // a[0]=1, not in b → valid
    EXPECT_EQ(mout(1), 0);  // a[1]=2, in b → invalid
    EXPECT_EQ(mout(2), 1);  // a[2]=3, not in b → valid
    EXPECT_EQ(mout(3), 0);  // a[3]=4, in b → invalid
    EXPECT_EQ(mout(4), 0);  // b[0]=2, in a → invalid
    EXPECT_EQ(mout(5), 0);  // b[1]=4, in a → invalid
    EXPECT_EQ(mout(6), 1);  // b[2]=6, not in a → valid
    EXPECT_EQ(mout(7), 1);  // b[3]=8, not in a → valid

    // Value checks at valid positions
    EXPECT_NEAR(vout(0), 1.0f, 1e-5f);
    EXPECT_NEAR(vout(2), 3.0f, 1e-5f);
    EXPECT_NEAR(vout(6), 6.0f, 1e-5f);
    EXPECT_NEAR(vout(7), 8.0f, 1e-5f);
}

TEST(SetOps2, SetXor1D_Disjoint) {
    // a = [1,2] (sorted), b = [3,4] (sorted)
    // Disjoint sets: every element is in the xor → all mask bits = 1
    Halide::Func a("xor_disj_a"), b("xor_disj_b");
    Halide::Var x;
    a(x) = Halide::select(x == 0, 1.0f, 2.0f);
    b(x) = Halide::select(x == 0, 3.0f, 4.0f);

    auto result = setxor1d_sorted(a, b, 2, 2, "xor_disj");
    result.compute_root();

    auto mask = extract_mask(result, "xor_disj_mask");
    Halide::Runtime::Buffer<int32_t> mout(4);
    mask.realize(mout);

    EXPECT_EQ(mout(0), 1);
    EXPECT_EQ(mout(1), 1);
    EXPECT_EQ(mout(2), 1);
    EXPECT_EQ(mout(3), 1);
}

TEST(SetOps2, SetXor1D_IdenticalSets) {
    // a = b = [1,2,4,8] (sorted): no element is unique to one set → all mask = 0
    Halide::Func a("xor_same_a"), b("xor_same_b");
    Halide::Var x;
    a(x) = Halide::select(x == 0, 1.0f, x == 1, 2.0f, x == 2, 4.0f, 8.0f);
    b(x) = Halide::select(x == 0, 1.0f, x == 1, 2.0f, x == 2, 4.0f, 8.0f);

    auto result = setxor1d_sorted(a, b, 4, 4, "xor_same");
    result.compute_root();

    auto mask = extract_mask(result, "xor_same_mask");
    Halide::Runtime::Buffer<int32_t> mout(8);
    mask.realize(mout);

    int valid_count = 0;
    for (int i = 0; i < 8; ++i) valid_count += mout(i);
    EXPECT_EQ(valid_count, 0);
}

// -----------------------------------------------------------------------------
// IsIn Tests
// -----------------------------------------------------------------------------

TEST(SetOps2, IsIn_1D) {
    // elements = [1,2,3,4,5], test_against = [2,4] (sorted)
    // expected: [0,1,0,1,0]
    shape_t elem_shape = {5};
    Halide::Func elements("isin_elems"), test_against("isin_test");
    Halide::Var x;
    elements(x)     = Halide::cast<float>(x + 1);       // [1,2,3,4,5]
    test_against(x) = Halide::select(x == 0, 2.0f, 4.0f); // [2,4]

    auto result = isin(elements, test_against, elem_shape, 2, "isin_1d");
    Halide::Runtime::Buffer<int32_t> out(5);
    result.realize(out);

    EXPECT_EQ(out(0), 0);  // 1 not in [2,4]
    EXPECT_EQ(out(1), 1);  // 2 in [2,4]
    EXPECT_EQ(out(2), 0);  // 3 not in [2,4]
    EXPECT_EQ(out(3), 1);  // 4 in [2,4]
    EXPECT_EQ(out(4), 0);  // 5 not in [2,4]
}

TEST(SetOps2, IsIn_AllMatch) {
    // elements = [3,3,3], test_against = [1,3] → all should match
    shape_t elem_shape = {3};
    Halide::Func elements("isin_allm"), test_against("isin_allm_t");
    Halide::Var x;
    elements(x)     = 3.0f;
    test_against(x) = Halide::select(x == 0, 1.0f, 3.0f);

    auto result = isin(elements, test_against, elem_shape, 2, "isin_allm");
    Halide::Runtime::Buffer<int32_t> out(3);
    result.realize(out);

    EXPECT_EQ(out(0), 1);
    EXPECT_EQ(out(1), 1);
    EXPECT_EQ(out(2), 1);
}

TEST(SetOps2, IsIn_NoneMatch) {
    // elements = [5,6,7], test_against = [1,2,3,4] → none should match
    shape_t elem_shape = {3};
    Halide::Func elements("isin_none"), test_against("isin_none_t");
    Halide::Var x;
    elements(x)     = Halide::cast<float>(x + 5);   // [5,6,7]
    test_against(x) = Halide::cast<float>(x + 1);   // [1,2,3,4]

    auto result = isin(elements, test_against, elem_shape, 4, "isin_none");
    Halide::Runtime::Buffer<int32_t> out(3);
    result.realize(out);

    EXPECT_EQ(out(0), 0);
    EXPECT_EQ(out(1), 0);
    EXPECT_EQ(out(2), 0);
}

TEST(SetOps2, IsIn_2D) {
    // 2D elements [[1,2],[3,4]], test_against = [1,4]
    // Expected: [[1,0],[0,1]]
    // In Halide: elements(x=col, y=row): (0,0)=1, (1,0)=2, (0,1)=3, (1,1)=4
    shape_t elem_shape = {2, 2};
    Halide::Func elements("isin_2d"), test_against("isin_2d_t");
    Halide::Var x, y, z;
    elements(x, y) = Halide::select(y == 0,
        Halide::select(x == 0, 1.0f, 2.0f),
        Halide::select(x == 0, 3.0f, 4.0f));
    test_against(z) = Halide::select(z == 0, 1.0f, 4.0f);

    auto result = isin(elements, test_against, elem_shape, 2, "isin_2d");
    Halide::Runtime::Buffer<int32_t> out(2, 2);
    result.realize(out);

    EXPECT_EQ(out(0, 0), 1);  // 1 in [1,4]
    EXPECT_EQ(out(1, 0), 0);  // 2 not in [1,4]
    EXPECT_EQ(out(0, 1), 0);  // 3 not in [1,4]
    EXPECT_EQ(out(1, 1), 1);  // 4 in [1,4]
}
