#include <gtest/gtest.h>
#include <Halide.h>
#include "../src/numhalide_all.h"
using namespace numhalide;

static Halide::Func make_1d(std::initializer_list<float> vals, const std::string& n = "f")
{
    std::vector<float> v(vals);
    Halide::Buffer<float> buf((int)v.size());
    for (int i = 0; i < (int)v.size(); ++i) buf(i) = v[i];
    Halide::Func f(n); Halide::Var x; f(x) = buf(x); return f;
}
static Halide::Func make_mat(int rows, int cols,
    std::initializer_list<float> vals, const std::string& n = "m")
{
    std::vector<float> v(vals);
    Halide::Buffer<float> buf(cols, rows);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            buf(c, r) = v[(size_t)(r * cols + c)];
    Halide::Func f(n); Halide::Var x, y; f(x, y) = buf(x, y); return f;
}

// ---- concat_1d ----

TEST(Join, Concat1D_Basic) {
    auto a = make_1d({1, 2, 3}, "a_c1");
    auto b = make_1d({4, 5},    "b_c1");
    auto r = concat_1d(a, 3, b, 2);
    Halide::Runtime::Buffer<float> out(5);
    r.realize(out);
    EXPECT_EQ(out(0), 1); EXPECT_EQ(out(1), 2); EXPECT_EQ(out(2), 3);
    EXPECT_EQ(out(3), 4); EXPECT_EQ(out(4), 5);
}

TEST(Join, Concat1D_Empty_First) {
    // First array of length 0 → only second array
    auto a = make_1d({},       "ae_c1");
    auto b = make_1d({7, 8, 9}, "be_c1");
    auto r = concat_1d(a, 0, b, 3);
    Halide::Runtime::Buffer<float> out(3);
    r.realize(out);
    EXPECT_EQ(out(0), 7); EXPECT_EQ(out(1), 8); EXPECT_EQ(out(2), 9);
}

// ---- concatenate (variadic) ----

TEST(Join, Concatenate_Three) {
    auto a = make_1d({1, 2},    "a_cv");
    auto b = make_1d({3},       "b_cv");
    auto c = make_1d({4, 5, 6}, "c_cv");
    auto r = concatenate({a, b, c}, {2, 1, 3});
    Halide::Runtime::Buffer<float> out(6);
    r.realize(out);
    for (int i = 0; i < 6; ++i)
        EXPECT_EQ(out(i), (float)(i + 1));
}

// ---- concat_2d ----

TEST(Join, Concat2D_Axis0) {
    // Stack rows: [[1,2],[3,4]] + [[5,6]] → [[1,2],[3,4],[5,6]]
    shape_t s1 = {2, 2}, s2 = {1, 2};
    auto A = make_mat(2, 2, {1,2, 3,4}, "A_2d0");
    auto B = make_mat(1, 2, {5, 6},     "B_2d0");
    auto r = concat_2d(A, s1, B, s2, 0);
    Halide::Runtime::Buffer<float> out(2, 3);
    r.realize(out);
    EXPECT_EQ(out(0,0),1); EXPECT_EQ(out(1,0),2);
    EXPECT_EQ(out(0,1),3); EXPECT_EQ(out(1,1),4);
    EXPECT_EQ(out(0,2),5); EXPECT_EQ(out(1,2),6);
}

TEST(Join, Concat2D_Axis1) {
    // Stack cols: [[1,2],[3,4]] | [[5],[6]] → [[1,2,5],[3,4,6]]
    shape_t s1 = {2, 2}, s2 = {2, 1};
    auto A = make_mat(2, 2, {1,2, 3,4}, "A_2d1");
    auto B = make_mat(2, 1, {5, 6},     "B_2d1");
    auto r = concat_2d(A, s1, B, s2, 1);
    Halide::Runtime::Buffer<float> out(3, 2);
    r.realize(out);
    EXPECT_EQ(out(0,0),1); EXPECT_EQ(out(1,0),2); EXPECT_EQ(out(2,0),5);
    EXPECT_EQ(out(0,1),3); EXPECT_EQ(out(1,1),4); EXPECT_EQ(out(2,1),6);
}

// ---- stack ----

TEST(Join, Stack_Axis0) {
    // Stack 3 arrays of length 4 → 3×4 matrix (row i = array i)
    auto a = make_1d({1,2,3,4}, "a_st");
    auto b = make_1d({5,6,7,8}, "b_st");
    auto r = stack({a, b}, 4, 0);
    // out(col, row): row 0 = a, row 1 = b
    Halide::Runtime::Buffer<float> out(4, 2);
    r.realize(out);
    EXPECT_EQ(out(0,0),1); EXPECT_EQ(out(3,0),4);
    EXPECT_EQ(out(0,1),5); EXPECT_EQ(out(3,1),8);
}

TEST(Join, Stack_Axis1) {
    // Stack 2 arrays of length 3 → 3×2 matrix (col j = array j)
    auto a = make_1d({1,2,3}, "a_st1");
    auto b = make_1d({4,5,6}, "b_st1");
    auto r = stack({a, b}, 3, 1);
    // out(col, row): col 0 = a, col 1 = b
    Halide::Runtime::Buffer<float> out(2, 3);
    r.realize(out);
    EXPECT_EQ(out(0,0),1); EXPECT_EQ(out(1,0),4);
    EXPECT_EQ(out(0,1),2); EXPECT_EQ(out(1,1),5);
    EXPECT_EQ(out(0,2),3); EXPECT_EQ(out(1,2),6);
}

TEST(Join, Stack_Three_Axis0) {
    auto a = make_1d({1,2}, "sa");
    auto b = make_1d({3,4}, "sb");
    auto c = make_1d({5,6}, "sc");
    auto r = stack({a, b, c}, 2, 0);
    Halide::Runtime::Buffer<float> out(2, 3);
    r.realize(out);
    EXPECT_EQ(out(0,0),1); EXPECT_EQ(out(1,0),2);
    EXPECT_EQ(out(0,1),3); EXPECT_EQ(out(1,1),4);
    EXPECT_EQ(out(0,2),5); EXPECT_EQ(out(1,2),6);
}
