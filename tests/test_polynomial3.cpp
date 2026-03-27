#include <gtest/gtest.h>
#include <Halide.h>
#include "../src/numhalide_all.h"
using namespace numhalide;

static Halide::Func make_poly(std::initializer_list<float> vals, const std::string& n = "p") {
    std::vector<float> v(vals);
    Halide::Buffer<float> buf((int)v.size());
    for (int i = 0; i < (int)v.size(); ++i) buf(i) = v[i];
    Halide::Func f(n);
    Halide::Var x;
    f(x) = buf(x);
    return f;
}

// --- polydiv ---
TEST(Polynomial3, PolyDiv_ExactDivision) {
    // (x^2 - 3x + 2) / (x - 1) = (x - 2) remainder 0
    // [1, -3, 2] / [1, -1] → q=[1,-2], r=[0]
    auto a = make_poly({1.0f, -3.0f, 2.0f}, "a");
    auto b = make_poly({1.0f, -1.0f}, "b");
    auto [q, r] = polydiv(a, 3, b, 2);
    Halide::Runtime::Buffer<float> qout(2), rout(1);
    q.realize(qout);
    r.realize(rout);
    EXPECT_NEAR(qout(0),  1.0f, 1e-4f);
    EXPECT_NEAR(qout(1), -2.0f, 1e-4f);
    EXPECT_NEAR(rout(0),  0.0f, 1e-4f);
}

TEST(Polynomial3, PolyDiv_WithRemainder) {
    // (x^2 + 2x + 3) / (x + 1) = (x + 1) remainder 2
    // [1, 2, 3] / [1, 1] → q=[1,1], r=[2]
    auto a = make_poly({1.0f, 2.0f, 3.0f}, "a");
    auto b = make_poly({1.0f, 1.0f}, "b");
    auto [q, r] = polydiv(a, 3, b, 2);
    Halide::Runtime::Buffer<float> qout(2), rout(1);
    q.realize(qout);
    r.realize(rout);
    EXPECT_NEAR(qout(0), 1.0f, 1e-4f);
    EXPECT_NEAR(qout(1), 1.0f, 1e-4f);
    EXPECT_NEAR(rout(0), 2.0f, 1e-4f);
}

TEST(Polynomial3, PolyDiv_ByConstant) {
    // (x^2 - 4) / 2 = (0.5x^2 - 2) remainder []
    // [1, 0, -4] / [2] → q=[0.5, 0, -2], r=[]
    auto a = make_poly({1.0f, 0.0f, -4.0f}, "a");
    auto b = make_poly({2.0f}, "b");
    auto [q, r] = polydiv(a, 3, b, 1);
    Halide::Runtime::Buffer<float> qout(3);
    q.realize(qout);
    EXPECT_NEAR(qout(0),  0.5f, 1e-4f);
    EXPECT_NEAR(qout(1),  0.0f, 1e-4f);
    EXPECT_NEAR(qout(2), -2.0f, 1e-4f);
    // remainder size = nb-1 = 0 (no actual remainder)
}

TEST(Polynomial3, PolyDiv_HigherDegree) {
    // (x^3 - 2x^2 - 4) / (x - 3) = (x^2 + x + 3) remainder 5
    // [1, -2, 0, -4] / [1, -3] → q=[1, 1, 3], r=[5]
    auto a = make_poly({1.0f, -2.0f, 0.0f, -4.0f}, "a");
    auto b = make_poly({1.0f, -3.0f}, "b");
    auto [q, r] = polydiv(a, 4, b, 2);
    Halide::Runtime::Buffer<float> qout(3), rout(1);
    q.realize(qout);
    r.realize(rout);
    EXPECT_NEAR(qout(0), 1.0f, 1e-3f);
    EXPECT_NEAR(qout(1), 1.0f, 1e-3f);
    EXPECT_NEAR(qout(2), 3.0f, 1e-3f);
    EXPECT_NEAR(rout(0), 5.0f, 1e-3f);
}
