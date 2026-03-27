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

// ---- pad (constant) ----

TEST(PadFunc, Constant_SymmetricZero) {
    // [1,2,3] + 1 before + 2 after → [0,1,2,3,0,0]
    auto f = make_1d({1,2,3}, "f_pc");
    auto r = pad(f, 3, 1, 2);
    Halide::Runtime::Buffer<float> out(6);
    r.realize(out);
    EXPECT_EQ(out(0), 0); EXPECT_EQ(out(1), 1); EXPECT_EQ(out(2), 2);
    EXPECT_EQ(out(3), 3); EXPECT_EQ(out(4), 0); EXPECT_EQ(out(5), 0);
}

TEST(PadFunc, Constant_Value) {
    auto f = make_1d({5, 6}, "f_pv");
    auto r = pad(f, 2, 2, 1, 9.0f);
    Halide::Runtime::Buffer<float> out(5);
    r.realize(out);
    EXPECT_EQ(out(0), 9); EXPECT_EQ(out(1), 9);
    EXPECT_EQ(out(2), 5); EXPECT_EQ(out(3), 6);
    EXPECT_EQ(out(4), 9);
}

// ---- pad_edge ----

TEST(PadFunc, Edge) {
    // [1,2,3] + 2 before + 2 after → [1,1,1,2,3,3,3]
    auto f = make_1d({1,2,3}, "f_pe");
    auto r = pad_edge(f, 3, 2, 2);
    Halide::Runtime::Buffer<float> out(7);
    r.realize(out);
    EXPECT_EQ(out(0),1); EXPECT_EQ(out(1),1);
    EXPECT_EQ(out(2),1); EXPECT_EQ(out(3),2); EXPECT_EQ(out(4),3);
    EXPECT_EQ(out(5),3); EXPECT_EQ(out(6),3);
}

// ---- pad_reflect ----

TEST(PadFunc, Reflect) {
    // [1,2,3] + 2 before + 2 after → [3,2, 1,2,3, 2,1]
    auto f = make_1d({1,2,3}, "f_pr");
    auto r = pad_reflect(f, 3, 2, 2);
    Halide::Runtime::Buffer<float> out(7);
    r.realize(out);
    EXPECT_EQ(out(0),3); EXPECT_EQ(out(1),2);   // reflected left
    EXPECT_EQ(out(2),1); EXPECT_EQ(out(3),2); EXPECT_EQ(out(4),3);  // original
    EXPECT_EQ(out(5),2); EXPECT_EQ(out(6),1);   // reflected right
}

TEST(PadFunc, Reflect_One) {
    // [1,2,3,4] + 1 before + 1 after → [2, 1,2,3,4, 3]
    auto f = make_1d({1,2,3,4}, "f_pr1");
    auto r = pad_reflect(f, 4, 1, 1);
    Halide::Runtime::Buffer<float> out(6);
    r.realize(out);
    EXPECT_EQ(out(0),2);
    EXPECT_EQ(out(1),1); EXPECT_EQ(out(2),2); EXPECT_EQ(out(3),3); EXPECT_EQ(out(4),4);
    EXPECT_EQ(out(5),3);
}

// ---- pad_wrap ----

TEST(PadFunc, Wrap) {
    // [1,2,3] + 2 before + 2 after → [2,3, 1,2,3, 1,2]
    auto f = make_1d({1,2,3}, "f_pw");
    auto r = pad_wrap(f, 3, 2, 2);
    Halide::Runtime::Buffer<float> out(7);
    r.realize(out);
    EXPECT_EQ(out(0),2); EXPECT_EQ(out(1),3);
    EXPECT_EQ(out(2),1); EXPECT_EQ(out(3),2); EXPECT_EQ(out(4),3);
    EXPECT_EQ(out(5),1); EXPECT_EQ(out(6),2);
}

// ---- pad_2d ----

TEST(PadFunc, Pad2D_Constant) {
    // [[1,2],[3,4]] padded with 1 on each side → 4x4 with zeros
    shape_t s = {2, 2};
    auto A = make_mat(2, 2, {1,2, 3,4}, "A_p2d");
    auto r = pad_2d(A, s, 1, 1, 1, 1, 0.0f);
    Halide::Runtime::Buffer<float> out(4, 4);
    r.realize(out);
    // Corner and border should be 0
    EXPECT_EQ(out(0,0), 0); EXPECT_EQ(out(3,3), 0);
    // Inner 2x2
    EXPECT_EQ(out(1,1), 1); EXPECT_EQ(out(2,1), 2);
    EXPECT_EQ(out(1,2), 3); EXPECT_EQ(out(2,2), 4);
}

TEST(PadFunc, Pad2D_Edge) {
    shape_t s = {2, 2};
    auto A = make_mat(2, 2, {1,2, 3,4}, "A_p2de");
    auto r = pad_2d_edge(A, s, 1, 1, 1, 1);
    Halide::Runtime::Buffer<float> out(4, 4);
    r.realize(out);
    // Top-left corner replicates A(0,0)=1
    EXPECT_EQ(out(0,0), 1);
    // Bottom-right corner replicates A(1,1)=4
    EXPECT_EQ(out(3,3), 4);
    // Top-right replicates A(1,0)=2
    EXPECT_EQ(out(3,0), 2);
}

TEST(PadFunc, Pad2D_Reflect) {
    shape_t s = {2, 2};
    auto A = make_mat(2, 2, {1,2, 3,4}, "A_p2dr");
    auto r = pad_2d_reflect(A, s, 1, 1, 1, 1);
    Halide::Runtime::Buffer<float> out(4, 4);
    r.realize(out);
    // Inner 2x2 unchanged
    EXPECT_EQ(out(1,1), 1); EXPECT_EQ(out(2,1), 2);
    EXPECT_EQ(out(1,2), 3); EXPECT_EQ(out(2,2), 4);
    // out(0,1) = reflect col 0 → col 1 of A = A(1,0) = 2
    EXPECT_EQ(out(0,1), 2);
    // out(1,0) = reflect row 0 → row 1 of A = A(0,1) = 3
    EXPECT_EQ(out(1,0), 3);
}

TEST(PadFunc, Pad2D_Wrap) {
    shape_t s = {2, 2};
    auto A = make_mat(2, 2, {1,2, 3,4}, "A_p2dw");
    auto r = pad_2d_wrap(A, s, 1, 1, 1, 1);
    Halide::Runtime::Buffer<float> out(4, 4);
    r.realize(out);
    // out(0,0): wrap → A(1,1)=4
    EXPECT_EQ(out(0,0), 4);
    // out(3,3): wrap → A(0,0)=1
    EXPECT_EQ(out(3,3), 1);
}
