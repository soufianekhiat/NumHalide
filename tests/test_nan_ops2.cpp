#include <gtest/gtest.h>
#include <Halide.h>
#include "../src/numhalide_all.h"
#include <limits>
using namespace numhalide;

// --- nancumsum ---
TEST(NanOps2, NanCumsum_Basic) {
    // [1, NaN, 3] → [1, 1, 4]
    Halide::ImageParam ip(Halide::Float(32), 1, "ip_ncs");
    float data[] = {1.0f, std::numeric_limits<float>::quiet_NaN(), 3.0f};
    Halide::Buffer<float> buf(data, 3);
    ip.set(buf);
    Halide::Func f("f"); Halide::Var x;
    f(x) = ip(x);

    shape_t s = {3};
    auto r = nancumsum(f, s);
    Halide::Runtime::Buffer<float> out(3);
    r.realize(out);
    EXPECT_NEAR(out(0), 1.0f, 1e-4f);
    EXPECT_NEAR(out(1), 1.0f, 1e-4f);
    EXPECT_NEAR(out(2), 4.0f, 1e-4f);
}

TEST(NanOps2, NanCumsum_NoNaN) {
    Halide::Func f("f"); Halide::Var x;
    Halide::Buffer<float> buf(3);
    buf(0) = 1.0f; buf(1) = 2.0f; buf(2) = 3.0f;
    f(x) = buf(x);
    shape_t s = {3};
    auto r = nancumsum(f, s);
    Halide::Runtime::Buffer<float> out(3);
    r.realize(out);
    EXPECT_NEAR(out(0), 1.0f, 1e-4f);
    EXPECT_NEAR(out(1), 3.0f, 1e-4f);
    EXPECT_NEAR(out(2), 6.0f, 1e-4f);
}

// --- nancumprod ---
TEST(NanOps2, NanCumprod_Basic) {
    // [2, NaN, 3] → [2, 2, 6]
    Halide::ImageParam ip(Halide::Float(32), 1, "ip_ncp");
    float data[] = {2.0f, std::numeric_limits<float>::quiet_NaN(), 3.0f};
    Halide::Buffer<float> buf(data, 3);
    ip.set(buf);
    Halide::Func f("f"); Halide::Var x;
    f(x) = ip(x);

    shape_t s = {3};
    auto r = nancumprod(f, s);
    Halide::Runtime::Buffer<float> out(3);
    r.realize(out);
    EXPECT_NEAR(out(0), 2.0f, 1e-4f);
    EXPECT_NEAR(out(1), 2.0f, 1e-4f);
    EXPECT_NEAR(out(2), 6.0f, 1e-4f);
}

// --- nanmedian ---
TEST(NanOps2, NanMedian_OddCount) {
    // [1, NaN, 3, NaN, 5] → median of [1,3,5] = 3
    Halide::ImageParam ip(Halide::Float(32), 1, "ip_nm1");
    float data[] = {1.0f, std::numeric_limits<float>::quiet_NaN(),
                    3.0f, std::numeric_limits<float>::quiet_NaN(), 5.0f};
    Halide::Buffer<float> buf(data, 5);
    ip.set(buf);
    Halide::Func f("f"); Halide::Var x;
    f(x) = ip(x);

    auto r = nanmedian(f, 5);
    auto out = Halide::Runtime::Buffer<float>::make_scalar();
    r.realize(out);
    EXPECT_NEAR(out(), 3.0f, 1e-4f);
}

TEST(NanOps2, NanMedian_EvenValidCount) {
    // [NaN, 1, 3, NaN, 5, 7] → valid [1,3,5,7], k=4 even
    // lo=(4-1)/2=1 → sorted[1]=3, hi=4/2=2 → sorted[2]=5, median=4
    Halide::ImageParam ip(Halide::Float(32), 1, "ip_nm2");
    float data[] = {std::numeric_limits<float>::quiet_NaN(),
                    1.0f, 3.0f,
                    std::numeric_limits<float>::quiet_NaN(),
                    5.0f, 7.0f};
    Halide::Buffer<float> buf(data, 6);
    ip.set(buf);
    Halide::Func f("f"); Halide::Var x;
    f(x) = ip(x);

    auto r = nanmedian(f, 6);
    auto out = Halide::Runtime::Buffer<float>::make_scalar();
    r.realize(out);
    EXPECT_NEAR(out(), 4.0f, 1e-4f);
}

TEST(NanOps2, NanMedian_AllValid) {
    // [2, 1, 4, 3] → sorted [1,2,3,4], median=(2+3)/2=2.5
    Halide::ImageParam ip(Halide::Float(32), 1, "ip_nm3");
    float data[] = {2.0f, 1.0f, 4.0f, 3.0f};
    Halide::Buffer<float> buf(data, 4);
    ip.set(buf);
    Halide::Func f("f"); Halide::Var x;
    f(x) = ip(x);

    auto r = nanmedian(f, 4);
    auto out = Halide::Runtime::Buffer<float>::make_scalar();
    r.realize(out);
    EXPECT_NEAR(out(), 2.5f, 1e-4f);
}
