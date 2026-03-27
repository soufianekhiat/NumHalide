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

// ---- flip 1D ----

TEST(FlipRoll, Flip1D) {
    auto f = make_1d({1,2,3,4,5}, "f_f1");
    auto r = flip(f, 5);
    Halide::Runtime::Buffer<float> out(5);
    r.realize(out);
    EXPECT_EQ(out(0), 5); EXPECT_EQ(out(1), 4); EXPECT_EQ(out(2), 3);
    EXPECT_EQ(out(3), 2); EXPECT_EQ(out(4), 1);
}

TEST(FlipRoll, Flip1D_Single) {
    auto f = make_1d({42.0f}, "f_f1s");
    auto r = flip(f, 1);
    Halide::Runtime::Buffer<float> out(1);
    r.realize(out);
    EXPECT_EQ(out(0), 42.0f);
}

// ---- flip 2D ----

TEST(FlipRoll, Flip2D_Axis0) {
    // flip rows: [[1,2],[3,4],[5,6]] → [[5,6],[3,4],[1,2]]
    shape_t s = {3, 2};
    auto A = make_mat(3, 2, {1,2, 3,4, 5,6}, "A_f2a0");
    auto r = flip(A, s, 0);
    Halide::Runtime::Buffer<float> out(2, 3);
    r.realize(out);
    EXPECT_EQ(out(0,0),5); EXPECT_EQ(out(1,0),6);
    EXPECT_EQ(out(0,1),3); EXPECT_EQ(out(1,1),4);
    EXPECT_EQ(out(0,2),1); EXPECT_EQ(out(1,2),2);
}

TEST(FlipRoll, Flip2D_Axis1) {
    // flip cols: [[1,2,3],[4,5,6]] → [[3,2,1],[6,5,4]]
    shape_t s = {2, 3};
    auto A = make_mat(2, 3, {1,2,3, 4,5,6}, "A_f2a1");
    auto r = flip(A, s, 1);
    Halide::Runtime::Buffer<float> out(3, 2);
    r.realize(out);
    EXPECT_EQ(out(0,0),3); EXPECT_EQ(out(1,0),2); EXPECT_EQ(out(2,0),1);
    EXPECT_EQ(out(0,1),6); EXPECT_EQ(out(1,1),5); EXPECT_EQ(out(2,1),4);
}

TEST(FlipRoll, Fliplr) {
    auto A = make_mat(2, 3, {1,2,3, 4,5,6}, "A_flr");
    auto r = fliplr(A, 3);
    Halide::Runtime::Buffer<float> out(3, 2);
    r.realize(out);
    EXPECT_EQ(out(0,0),3); EXPECT_EQ(out(2,0),1);
    EXPECT_EQ(out(0,1),6); EXPECT_EQ(out(2,1),4);
}

TEST(FlipRoll, Flipud) {
    auto A = make_mat(3, 2, {1,2, 3,4, 5,6}, "A_fud");
    auto r = flipud(A, 3);
    Halide::Runtime::Buffer<float> out(2, 3);
    r.realize(out);
    EXPECT_EQ(out(0,0),5); EXPECT_EQ(out(1,0),6);
    EXPECT_EQ(out(0,2),1); EXPECT_EQ(out(1,2),2);
}

// ---- roll 1D ----

TEST(FlipRoll, Roll1D_PositiveShift) {
    // roll([1,2,3,4,5], 2) → [4,5,1,2,3]
    auto f = make_1d({1,2,3,4,5}, "f_rp");
    auto r = roll(f, 5, 2);
    Halide::Runtime::Buffer<float> out(5);
    r.realize(out);
    EXPECT_EQ(out(0),4); EXPECT_EQ(out(1),5);
    EXPECT_EQ(out(2),1); EXPECT_EQ(out(3),2); EXPECT_EQ(out(4),3);
}

TEST(FlipRoll, Roll1D_NegativeShift) {
    // roll([1,2,3,4,5], -1) → [2,3,4,5,1]
    auto f = make_1d({1,2,3,4,5}, "f_rn");
    auto r = roll(f, 5, -1);
    Halide::Runtime::Buffer<float> out(5);
    r.realize(out);
    EXPECT_EQ(out(0),2); EXPECT_EQ(out(1),3); EXPECT_EQ(out(2),4);
    EXPECT_EQ(out(3),5); EXPECT_EQ(out(4),1);
}

TEST(FlipRoll, Roll1D_ZeroShift) {
    auto f = make_1d({1,2,3}, "f_rz");
    auto r = roll(f, 3, 0);
    Halide::Runtime::Buffer<float> out(3);
    r.realize(out);
    EXPECT_EQ(out(0),1); EXPECT_EQ(out(1),2); EXPECT_EQ(out(2),3);
}

TEST(FlipRoll, Roll1D_FullPeriod) {
    // shift by n = identity
    auto f = make_1d({1,2,3,4}, "f_rfp");
    auto r = roll(f, 4, 4);
    Halide::Runtime::Buffer<float> out(4);
    r.realize(out);
    EXPECT_EQ(out(0),1); EXPECT_EQ(out(1),2); EXPECT_EQ(out(2),3); EXPECT_EQ(out(3),4);
}

// ---- roll 2D ----

TEST(FlipRoll, Roll2D_Axis0) {
    // roll rows by 1: [[1,2],[3,4],[5,6]] → [[5,6],[1,2],[3,4]]
    shape_t s = {3, 2};
    auto A = make_mat(3, 2, {1,2, 3,4, 5,6}, "A_r2a0");
    auto r = roll(A, s, 1, 0);
    Halide::Runtime::Buffer<float> out(2, 3);
    r.realize(out);
    EXPECT_EQ(out(0,0),5); EXPECT_EQ(out(1,0),6);
    EXPECT_EQ(out(0,1),1); EXPECT_EQ(out(1,1),2);
    EXPECT_EQ(out(0,2),3); EXPECT_EQ(out(1,2),4);
}

TEST(FlipRoll, Roll2D_Axis1) {
    // roll cols by 1: [[1,2,3],[4,5,6]] → [[3,1,2],[6,4,5]]
    shape_t s = {2, 3};
    auto A = make_mat(2, 3, {1,2,3, 4,5,6}, "A_r2a1");
    auto r = roll(A, s, 1, 1);
    Halide::Runtime::Buffer<float> out(3, 2);
    r.realize(out);
    EXPECT_EQ(out(0,0),3); EXPECT_EQ(out(1,0),1); EXPECT_EQ(out(2,0),2);
    EXPECT_EQ(out(0,1),6); EXPECT_EQ(out(1,1),4); EXPECT_EQ(out(2,1),5);
}
