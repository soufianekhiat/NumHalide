#include <gtest/gtest.h>
#include <Halide.h>
#include "../src/numhalide_all.h"
#include <limits>
using namespace numhalide;
using namespace numhalide::stats;

// Helper macro: create ImageParam + Buffer + Func in current scope
#define MAKE_IP_FUNC(name, data_arr, n)                                       \
    Halide::ImageParam ip_##name(Halide::Float(32), 1, #name "_ip");          \
    Halide::Buffer<float> buf_##name(data_arr, n);                            \
    ip_##name.set(buf_##name);                                                \
    Halide::Func name("f_" #name); { Halide::Var _x; name(_x) = ip_##name(_x); }

// --- nanpercentile ---
TEST(Statistics3, NanPercentile_Min) {
    float d[] = {1.0f, std::numeric_limits<float>::quiet_NaN(), 3.0f, 4.0f};
    MAKE_IP_FUNC(f1, d, 4)
    auto r = nanpercentile(f1, 4, 0.0f);
    auto out = Halide::Runtime::Buffer<float>::make_scalar();
    r.realize(out);
    EXPECT_NEAR(out(), 1.0f, 1e-4f);
}

TEST(Statistics3, NanPercentile_Max) {
    float d[] = {1.0f, std::numeric_limits<float>::quiet_NaN(), 3.0f, 4.0f};
    MAKE_IP_FUNC(f2, d, 4)
    auto r = nanpercentile(f2, 4, 100.0f);
    auto out = Halide::Runtime::Buffer<float>::make_scalar();
    r.realize(out);
    EXPECT_NEAR(out(), 4.0f, 1e-4f);
}

TEST(Statistics3, NanPercentile_Median_OddCount) {
    // valid [1, 3, 5]: 50th → pos=0.5*2=1.0, lo=hi=1 → sorted[1]=3
    float d[] = {1.0f, std::numeric_limits<float>::quiet_NaN(), 3.0f, 5.0f};
    MAKE_IP_FUNC(f3, d, 4)
    auto r = nanpercentile(f3, 4, 50.0f);
    auto out = Halide::Runtime::Buffer<float>::make_scalar();
    r.realize(out);
    EXPECT_NEAR(out(), 3.0f, 1e-4f);
}

TEST(Statistics3, NanPercentile_Interpolated) {
    // valid [1, 3, 5, 7], 50th: pos=1.5, lo=1,hi=2,frac=0.5 → 3*0.5+5*0.5=4
    float d[] = {std::numeric_limits<float>::quiet_NaN(), 1.0f, 3.0f, 5.0f, 7.0f};
    MAKE_IP_FUNC(f4, d, 5)
    auto r = nanpercentile(f4, 5, 50.0f);
    auto out = Halide::Runtime::Buffer<float>::make_scalar();
    r.realize(out);
    EXPECT_NEAR(out(), 4.0f, 1e-4f);
}

// --- nanquantile ---
TEST(Statistics3, NanQuantile_Half) {
    float d[] = {1.0f, std::numeric_limits<float>::quiet_NaN(), 3.0f, 5.0f};
    MAKE_IP_FUNC(f5, d, 4)
    auto r = nanquantile(f5, 4, 0.5f);
    auto out = Halide::Runtime::Buffer<float>::make_scalar();
    r.realize(out);
    EXPECT_NEAR(out(), 3.0f, 1e-4f);
}

TEST(Statistics3, NanQuantile_Zero) {
    float d[] = {std::numeric_limits<float>::quiet_NaN(), 2.0f, 4.0f};
    MAKE_IP_FUNC(f6, d, 3)
    auto r = nanquantile(f6, 3, 0.0f);
    auto out = Halide::Runtime::Buffer<float>::make_scalar();
    r.realize(out);
    EXPECT_NEAR(out(), 2.0f, 1e-4f);
}

TEST(Statistics3, NanQuantile_One) {
    float d[] = {std::numeric_limits<float>::quiet_NaN(), 2.0f, 4.0f};
    MAKE_IP_FUNC(f7, d, 3)
    auto r = nanquantile(f7, 3, 1.0f);
    auto out = Halide::Runtime::Buffer<float>::make_scalar();
    r.realize(out);
    EXPECT_NEAR(out(), 4.0f, 1e-4f);
}
