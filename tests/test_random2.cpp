/// @file test_random2.cpp
/// @brief Tests for rand_poisson, rand_gamma, rand_beta, rand_binomial

#include <gtest/gtest.h>
#include "numhalide_all.h"
#include <cmath>

using namespace numhalide;

// -----------------------------------------------------------------------------
// Range / Validity Tests
// -----------------------------------------------------------------------------

TEST(Random2, RandPoisson_Range) {
    // Poisson(lambda=4.0) values should be non-negative integers
    shape_t s = {64};
    auto f = rand_poisson(Halide::Int(32), s, 4.0f, 42);
    Halide::Runtime::Buffer<int32_t> out(64);
    f.realize(out);
    for (int i = 0; i < 64; ++i) {
        EXPECT_GE(out(i), 0);
    }
}

TEST(Random2, RandGamma_Positive) {
    // Gamma samples should be non-negative
    shape_t s = {64};
    auto f = rand_gamma(Halide::Float(32), s, 2.0f, 1.0f, 123);
    Halide::Runtime::Buffer<float> out(64);
    f.realize(out);
    for (int i = 0; i < 64; ++i) {
        EXPECT_GE(out(i), 0.0f);
    }
}

TEST(Random2, RandBeta_InRange) {
    // Beta samples should be in [0, 1]
    shape_t s = {64};
    auto f = rand_beta(Halide::Float(32), s, 2.0f, 3.0f, 456);
    Halide::Runtime::Buffer<float> out(64);
    f.realize(out);
    for (int i = 0; i < 64; ++i) {
        EXPECT_GE(out(i), 0.0f);
        EXPECT_LE(out(i), 1.0f);
    }
}

TEST(Random2, RandBinomial_Range) {
    // Binomial(n=10, p=0.5) values should be in [0, 10]
    shape_t s = {64};
    auto f = rand_binomial(Halide::Int(32), s, 10, 0.5f, 789);
    Halide::Runtime::Buffer<int32_t> out(64);
    f.realize(out);
    for (int i = 0; i < 64; ++i) {
        EXPECT_GE(out(i), 0);
        EXPECT_LE(out(i), 10);
    }
}

// -----------------------------------------------------------------------------
// Mean Approximation Tests (large sample, loose tolerance)
// -----------------------------------------------------------------------------

TEST(Random2, RandPoisson_Mean) {
    // Poisson(lambda=5.0): mean should be ~5.0
    const int N = 256;
    shape_t s = {N};
    auto f = rand_poisson(Halide::Int(32), s, 5.0f, 77);

    // Cast to float for summation
    Halide::Func ff("pp_float");
    Halide::Var x;
    ff(x) = Halide::cast<float>(f(x));

    auto sum_f = reduce_sum(ff, s, "pp_sum");
    Halide::Runtime::Buffer<float> sum_out(1);
    sum_f.realize(sum_out);
    float mean = sum_out(0) / static_cast<float>(N);

    // Normal approximation: mean=lambda=5, stddev=sqrt(5)~2.24
    // With 256 samples the sample-mean std ~ 2.24/16 ~ 0.14, so 0.5 is very safe
    EXPECT_NEAR(mean, 5.0f, 0.5f);
}

TEST(Random2, RandBinomial_Mean) {
    // Binomial(n=20, p=0.4): mean should be ~8.0
    const int N = 256;
    shape_t s = {N};
    auto f = rand_binomial(Halide::Int(32), s, 20, 0.4f, 55);

    Halide::Func ff("bi_float");
    Halide::Var x;
    ff(x) = Halide::cast<float>(f(x));

    auto sum_f = reduce_sum(ff, s, "bi_sum");
    Halide::Runtime::Buffer<float> sum_out(1);
    sum_f.realize(sum_out);
    float mean = sum_out(0) / static_cast<float>(N);

    // mean = n*p = 8, stddev of sample mean ~ sqrt(n*p*(1-p))/sqrt(N) ~ 0.14
    EXPECT_NEAR(mean, 8.0f, 1.0f);
}

TEST(Random2, RandGamma_Mean) {
    // Gamma(alpha=3, beta=2): mean = alpha*beta = 6
    const int N = 256;
    shape_t s = {N};
    auto f = rand_gamma(Halide::Float(32), s, 3.0f, 2.0f, 11);

    auto sum_f = reduce_sum(f, s, "ga_sum");
    Halide::Runtime::Buffer<float> sum_out(1);
    sum_f.realize(sum_out);
    float mean = sum_out(0) / static_cast<float>(N);

    // mean = 6, variance = alpha*beta^2 = 12, std of sample mean ~ sqrt(12)/16 ~ 0.22
    EXPECT_NEAR(mean, 6.0f, 1.5f);
}

TEST(Random2, RandBeta_Mean) {
    // Beta(alpha=2, beta=3): mean = alpha/(alpha+beta) = 0.4
    const int N = 256;
    shape_t s = {N};
    auto f = rand_beta(Halide::Float(32), s, 2.0f, 3.0f, 33);

    auto sum_f = reduce_sum(f, s, "be_sum");
    Halide::Runtime::Buffer<float> sum_out(1);
    sum_f.realize(sum_out);
    float mean = sum_out(0) / static_cast<float>(N);

    EXPECT_NEAR(mean, 0.4f, 0.1f);
}
