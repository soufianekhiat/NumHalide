#include <gtest/gtest.h>
#include <Halide.h>
#include "../src/numhalide_all.h"
using namespace numhalide;

static Halide::Func make_1d(std::initializer_list<float> vals, const std::string& n = "f") {
    int sz = (int)vals.size();
    std::vector<float> v(vals);
    Halide::Func f(n);
    Halide::Var x;
    Halide::Buffer<float> buf(sz);
    for (int i = 0; i < sz; ++i) buf(i) = v[i];
    f(x) = buf(x);
    return f;
}

// --- logaddexp ---
TEST(Numeric, LogAddExp_Symmetric) {
    auto a = make_1d({1.0f, 2.0f});
    auto b = make_1d({2.0f, 1.0f});
    shape_t s = {2};
    auto r = logaddexp(a, b, s);
    Halide::Runtime::Buffer<float> out(2);
    r.realize(out);
    float expected = std::log(std::exp(1.0f) + std::exp(2.0f));
    EXPECT_NEAR(out(0), expected, 1e-4f);
    EXPECT_NEAR(out(1), expected, 1e-4f);
}

TEST(Numeric, LogAddExp_SameValue) {
    auto a = make_1d({0.0f});
    auto b = make_1d({0.0f});
    shape_t s = {1};
    auto r = logaddexp(a, b, s);
    Halide::Runtime::Buffer<float> out(1);
    r.realize(out);
    EXPECT_NEAR(out(0), std::log(2.0f), 1e-4f);
}

// --- logaddexp2 ---
TEST(Numeric, LogAddExp2_Basic) {
    auto a = make_1d({1.0f});
    auto b = make_1d({1.0f});
    shape_t s = {1};
    auto r = logaddexp2(a, b, s);
    Halide::Runtime::Buffer<float> out(1);
    r.realize(out);
    // log2(2^1 + 2^1) = log2(4) = 2
    EXPECT_NEAR(out(0), 2.0f, 1e-4f);
}

// --- copysign ---
TEST(Numeric, Copysign_Basic) {
    auto mag = make_1d({1.0f, -2.0f, 3.0f}, "mag");
    auto sgn = make_1d({-1.0f, 1.0f, -1.0f}, "sgn");
    shape_t s = {3};
    auto r = copysign(mag, sgn, s);
    Halide::Runtime::Buffer<float> out(3);
    r.realize(out);
    EXPECT_NEAR(out(0), -1.0f, 1e-5f);
    EXPECT_NEAR(out(1),  2.0f, 1e-5f);
    EXPECT_NEAR(out(2), -3.0f, 1e-5f);
}

// --- signbit ---
TEST(Numeric, Signbit_Basic) {
    auto f = make_1d({1.0f, -2.0f, 0.0f}, "sb_f");
    shape_t s = {3};
    auto r = signbit(f, s);
    Halide::Runtime::Buffer<uint8_t> out(3);
    r.realize(out);
    EXPECT_EQ(out(0), 0);
    EXPECT_EQ(out(1), 1);
    EXPECT_EQ(out(2), 0);
}

// --- trapz_1d uniform ---
TEST(Numeric, Trapz1D_Uniform) {
    auto f = make_1d({1.0f, 2.0f, 3.0f, 4.0f});
    auto r = trapz_1d(f, 4, 1.0f);
    auto out = Halide::Runtime::Buffer<float>::make_scalar();
    r.realize(out);
    // (1+2)/2 + (2+3)/2 + (3+4)/2 = 1.5 + 2.5 + 3.5 = 7.5
    EXPECT_NEAR(out(), 7.5f, 1e-4f);
}

TEST(Numeric, Trapz1D_DxHalf) {
    auto f = make_1d({0.0f, 1.0f, 4.0f});
    auto r = trapz_1d(f, 3, 0.5f);
    auto out = Halide::Runtime::Buffer<float>::make_scalar();
    r.realize(out);
    // (0+1)*0.5/2 + (1+4)*0.5/2 = 0.25 + 1.25 = 1.5
    EXPECT_NEAR(out(), 1.5f, 1e-4f);
}

// --- trapz_1d non-uniform ---
TEST(Numeric, Trapz1D_NonUniform) {
    auto f = make_1d({0.0f, 1.0f, 4.0f});
    auto x = make_1d({0.0f, 1.0f, 2.0f});
    auto r = trapz_1d(f, x, 3);
    auto out = Halide::Runtime::Buffer<float>::make_scalar();
    r.realize(out);
    // (0+1)*(1-0)/2 + (1+4)*(2-1)/2 = 0.5 + 2.5 = 3.0
    EXPECT_NEAR(out(), 3.0f, 1e-4f);
}

// --- i0 ---
TEST(Numeric, I0_Zero) {
    auto f = make_1d({0.0f});
    shape_t s = {1};
    auto r = i0(f, s);
    Halide::Runtime::Buffer<float> out(1);
    r.realize(out);
    EXPECT_NEAR(out(0), 1.0f, 1e-5f);
}

TEST(Numeric, I0_One) {
    auto f = make_1d({1.0f});
    shape_t s = {1};
    auto r = i0(f, s);
    Halide::Runtime::Buffer<float> out(1);
    r.realize(out);
    EXPECT_NEAR(out(0), 1.2660658f, 1e-3f);
}

TEST(Numeric, I0_Large) {
    auto f = make_1d({4.0f});
    shape_t s = {1};
    auto r = i0(f, s);
    Halide::Runtime::Buffer<float> out(1);
    r.realize(out);
    EXPECT_NEAR(out(0), 11.3019219f, 0.01f);
}

// --- correlate1d ---
TEST(Numeric, Correlate1D_Full) {
    // a=[1,2,3], v=[1,0]: full mode
    auto a = make_1d({1.0f, 2.0f, 3.0f});
    auto v = make_1d({1.0f, 0.0f});
    auto r = correlate1d(a, v, 3, 2, "full");
    // c[k] = sum_n a[n+k]*v[n]: output = [0,1,2,3]
    Halide::Runtime::Buffer<float> out(4);
    r.realize(out);
    EXPECT_NEAR(out(0), 0.0f, 1e-4f);
    EXPECT_NEAR(out(1), 1.0f, 1e-4f);
    EXPECT_NEAR(out(2), 2.0f, 1e-4f);
    EXPECT_NEAR(out(3), 3.0f, 1e-4f);
}

TEST(Numeric, Correlate1D_Same) {
    auto a = make_1d({1.0f, 2.0f, 3.0f});
    auto v = make_1d({1.0f, 0.0f, -1.0f});
    // v=[1,0,-1]: c[k] = a[k]*1 + a[k+1]*0 + a[k+2]*(-1)
    // same mode: output size=3, pad=(3-1)/2=1
    // k=0: a[-1]*1+a[0]*0+a[1]*(-1) = 0+0-2 = -2
    // k=1: a[0]*1+a[1]*0+a[2]*(-1) = 1+0-3 = -2
    // k=2: a[1]*1+a[2]*0+a[3]*(-1) = 2+0+0 = 2
    auto r = correlate1d(a, v, 3, 3, "same");
    Halide::Runtime::Buffer<float> out(3);
    r.realize(out);
    EXPECT_NEAR(out(0), -2.0f, 1e-4f);
    EXPECT_NEAR(out(1), -2.0f, 1e-4f);
    EXPECT_NEAR(out(2),  2.0f, 1e-4f);
}

TEST(Numeric, Correlate1D_Valid) {
    auto a = make_1d({1.0f, 2.0f, 3.0f, 4.0f});
    auto v = make_1d({1.0f, 0.0f, -1.0f});
    // valid mode: size = 4-3+1 = 2
    // k=0: a[0]*1+a[1]*0+a[2]*(-1) = 1-3 = -2
    // k=1: a[1]*1+a[2]*0+a[3]*(-1) = 2-4 = -2
    auto r = correlate1d(a, v, 4, 3, "valid");
    Halide::Runtime::Buffer<float> out(2);
    r.realize(out);
    EXPECT_NEAR(out(0), -2.0f, 1e-4f);
    EXPECT_NEAR(out(1), -2.0f, 1e-4f);
}
