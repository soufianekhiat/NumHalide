#include <gtest/gtest.h>
#include <Halide.h>
#include "../src/numhalide_all.h"
using namespace numhalide;

static Halide::Func make_1d(std::initializer_list<float> vals, const std::string& n = "f") {
    std::vector<float> v(vals);
    Halide::Buffer<float> buf((int)v.size());
    for (int i = 0; i < (int)v.size(); ++i) buf(i) = v[i];
    Halide::Func f(n);
    Halide::Var x;
    f(x) = buf(x);
    return f;
}

static Halide::Func make_2d(int rows, int cols,
    std::initializer_list<float> vals, const std::string& n = "f") {
    std::vector<float> v(vals);
    Halide::Buffer<float> buf(cols, rows);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            buf(c, r) = v[(size_t)(r * cols + c)];
    Halide::Func f(n);
    Halide::Var x, y;
    f(x, y) = buf(x, y);
    return f;
}

// --- cross_3d ---
TEST(LA3, Cross_UnitVectors) {
    // [1,0,0] × [0,1,0] = [0,0,1]
    auto a = make_1d({1.0f, 0.0f, 0.0f}, "a");
    auto b = make_1d({0.0f, 1.0f, 0.0f}, "b");
    auto r = cross_3d(a, b);
    Halide::Runtime::Buffer<float> out(3);
    r.realize(out);
    EXPECT_NEAR(out(0), 0.0f, 1e-5f);
    EXPECT_NEAR(out(1), 0.0f, 1e-5f);
    EXPECT_NEAR(out(2), 1.0f, 1e-5f);
}

TEST(LA3, Cross_General) {
    // [1,2,3] × [4,5,6] = [2*6-3*5, 3*4-1*6, 1*5-2*4] = [-3, 6, -3]
    auto a = make_1d({1.0f, 2.0f, 3.0f}, "a");
    auto b = make_1d({4.0f, 5.0f, 6.0f}, "b");
    auto r = cross_3d(a, b);
    Halide::Runtime::Buffer<float> out(3);
    r.realize(out);
    EXPECT_NEAR(out(0), -3.0f, 1e-5f);
    EXPECT_NEAR(out(1),  6.0f, 1e-5f);
    EXPECT_NEAR(out(2), -3.0f, 1e-5f);
}

// --- tensordot axes=0 (outer product) ---
TEST(LA3, Tensordot_OuterProduct) {
    // [1,2] ⊗ [3,4] = [[3,4],[6,8]]
    auto a = make_1d({1.0f, 2.0f}, "a");
    auto b = make_1d({3.0f, 4.0f}, "b");
    shape_t sa = {2};
    shape_t sb = {2};
    auto r = tensordot(a, sa, b, sb, 0);
    // Output shape {2,2}: Halide Buffer(cols=2, rows=2)
    Halide::Runtime::Buffer<float> out(2, 2);
    r.realize(out);
    // out(col, row) = a(row) * b(col)
    EXPECT_NEAR(out(0, 0), 3.0f, 1e-5f); // a(0)*b(0) = 1*3
    EXPECT_NEAR(out(1, 0), 4.0f, 1e-5f); // a(0)*b(1) = 1*4
    EXPECT_NEAR(out(0, 1), 6.0f, 1e-5f); // a(1)*b(0) = 2*3
    EXPECT_NEAR(out(1, 1), 8.0f, 1e-5f); // a(1)*b(1) = 2*4
}

// --- tensordot axes=1 (matmul) ---
TEST(LA3, Tensordot_Matmul) {
    // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
    auto a = make_2d(2, 2, {1, 2, 3, 4}, "a");
    auto b = make_2d(2, 2, {5, 6, 7, 8}, "b");
    shape_t sa = {2, 2};
    shape_t sb = {2, 2};
    auto r = tensordot(a, sa, b, sb, 1);
    Halide::Runtime::Buffer<float> out(2, 2);
    r.realize(out);
    EXPECT_NEAR(out(0, 0), 19.0f, 1e-3f);
    EXPECT_NEAR(out(1, 0), 22.0f, 1e-3f);
    EXPECT_NEAR(out(0, 1), 43.0f, 1e-3f);
    EXPECT_NEAR(out(1, 1), 50.0f, 1e-3f);
}

TEST(LA3, Tensordot_DotProduct) {
    // dot([1,2,3], [4,5,6]) = 32
    auto a = make_1d({1.0f, 2.0f, 3.0f}, "a");
    auto b = make_1d({4.0f, 5.0f, 6.0f}, "b");
    shape_t sa = {3};
    shape_t sb = {3};
    auto r = tensordot(a, sa, b, sb, 1);
    auto out = Halide::Runtime::Buffer<float>::make_scalar();
    r.realize(out);
    EXPECT_NEAR(out(), 32.0f, 1e-3f);
}

// --- normalize ---
TEST(LA3, Normalize_1D) {
    // [3, 4] / 5 = [0.6, 0.8]
    auto f = make_1d({3.0f, 4.0f});
    shape_t s = {2};
    auto r = normalize(f, s, 0);
    Halide::Runtime::Buffer<float> out(2);
    r.realize(out);
    EXPECT_NEAR(out(0), 0.6f, 1e-4f);
    EXPECT_NEAR(out(1), 0.8f, 1e-4f);
}

TEST(LA3, Normalize_2D_ByRow) {
    // [[3,4],[0,1]] normalize each row (axis=1)
    auto f = make_2d(2, 2, {3.0f, 4.0f, 0.0f, 1.0f});
    shape_t s = {2, 2};
    auto r = normalize(f, s, 1);
    Halide::Runtime::Buffer<float> out(2, 2);
    r.realize(out);
    EXPECT_NEAR(out(0, 0), 0.6f, 1e-4f); // row 0: [3/5, 4/5]
    EXPECT_NEAR(out(1, 0), 0.8f, 1e-4f);
    EXPECT_NEAR(out(1, 1), 1.0f, 1e-3f); // row 1: [0/1, 1/1]
}
