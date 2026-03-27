/// @file test_nan_ops.cpp
/// @brief Tests for NaN-safe reduction operations (nansum, nanprod, nanmin,
///        nanmax, nanmean, nanvar, nanstd)

#include <gtest/gtest.h>
#include "numhalide_all.h"
#include <cmath>
#include <limits>

using namespace numhalide;

// Helper: build a 1D Func backed by an ImageParam to prevent constant-folding.
// The caller owns `data` and `buf`; `ip` must outlive the returned Func.
static void make_nan_func(
    Halide::ImageParam& ip,
    Halide::Buffer<float>& buf,
    float* raw, int n,
    Halide::Func& out_func)
{
    buf = Halide::Buffer<float>(raw, n);
    ip.set(buf);
    Halide::Var x;
    out_func(x) = ip(x);
}

// -----------------------------------------------------------------------------
// NanSum Tests
// -----------------------------------------------------------------------------

TEST(NanOps, NanSum_1D) {
    // [1.0, NaN, 3.0, 4.0] -> nansum = 8.0
    shape_t s = {4};
    Halide::ImageParam ip(Halide::Float(32), 1, "ip_nansum");
    float data[] = {1.0f, std::numeric_limits<float>::quiet_NaN(), 3.0f, 4.0f};
    Halide::Buffer<float> buf(data, 4);
    ip.set(buf);
    Halide::Func f("f_nansum");
    Halide::Var x;
    f(x) = ip(x);

    auto result = nansum(f, s, {0}, /*keepdims=*/true);

    Halide::Runtime::Buffer<float> out(1);
    result.realize(out);

    EXPECT_NEAR(out(0), 8.0f, 1e-4f);
}

TEST(NanOps, NanSum_AllNaN) {
    // [NaN, NaN, NaN] -> nansum = 0.0 (NaN treated as 0)
    shape_t s = {3};
    Halide::ImageParam ip(Halide::Float(32), 1, "ip_nansum_all");
    float data[] = {
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::quiet_NaN()
    };
    Halide::Buffer<float> buf(data, 3);
    ip.set(buf);
    Halide::Func f("f_nansum_all");
    Halide::Var x;
    f(x) = ip(x);

    auto result = nansum(f, s, {0}, /*keepdims=*/true);

    Halide::Runtime::Buffer<float> out(1);
    result.realize(out);

    EXPECT_NEAR(out(0), 0.0f, 1e-4f);
}

// -----------------------------------------------------------------------------
// NanMin Tests
// -----------------------------------------------------------------------------

TEST(NanOps, NanMin_1D) {
    // [NaN, 5.0, 2.0, NaN] -> nanmin = 2.0
    shape_t s = {4};
    Halide::ImageParam ip(Halide::Float(32), 1, "ip_nanmin");
    float nan = std::numeric_limits<float>::quiet_NaN();
    float data[] = {nan, 5.0f, 2.0f, nan};
    Halide::Buffer<float> buf(data, 4);
    ip.set(buf);
    Halide::Func f("f_nanmin");
    Halide::Var x;
    f(x) = ip(x);

    auto result = nanmin(f, s, {0}, /*keepdims=*/true);

    Halide::Runtime::Buffer<float> out(1);
    result.realize(out);

    EXPECT_NEAR(out(0), 2.0f, 1e-4f);
}

TEST(NanOps, NanMin_NoNaN) {
    // [3.0, 1.0, 4.0, 2.0] -> nanmin = 1.0 (no NaN present)
    shape_t s = {4};
    Halide::ImageParam ip(Halide::Float(32), 1, "ip_nanmin_nonnan");
    float data[] = {3.0f, 1.0f, 4.0f, 2.0f};
    Halide::Buffer<float> buf(data, 4);
    ip.set(buf);
    Halide::Func f("f_nanmin_nonnan");
    Halide::Var x;
    f(x) = ip(x);

    auto result = nanmin(f, s, {0}, /*keepdims=*/true);

    Halide::Runtime::Buffer<float> out(1);
    result.realize(out);

    EXPECT_NEAR(out(0), 1.0f, 1e-4f);
}

// -----------------------------------------------------------------------------
// NanMax Tests
// -----------------------------------------------------------------------------

TEST(NanOps, NanMax_1D) {
    // [NaN, 5.0, 2.0, NaN] -> nanmax = 5.0
    shape_t s = {4};
    Halide::ImageParam ip(Halide::Float(32), 1, "ip_nanmax");
    float nan = std::numeric_limits<float>::quiet_NaN();
    float data[] = {nan, 5.0f, 2.0f, nan};
    Halide::Buffer<float> buf(data, 4);
    ip.set(buf);
    Halide::Func f("f_nanmax");
    Halide::Var x;
    f(x) = ip(x);

    auto result = nanmax(f, s, {0}, /*keepdims=*/true);

    Halide::Runtime::Buffer<float> out(1);
    result.realize(out);

    EXPECT_NEAR(out(0), 5.0f, 1e-4f);
}

TEST(NanOps, NanMax_NoNaN) {
    // [3.0, 1.0, 4.0, 2.0] -> nanmax = 4.0
    shape_t s = {4};
    Halide::ImageParam ip(Halide::Float(32), 1, "ip_nanmax_nonnan");
    float data[] = {3.0f, 1.0f, 4.0f, 2.0f};
    Halide::Buffer<float> buf(data, 4);
    ip.set(buf);
    Halide::Func f("f_nanmax_nonnan");
    Halide::Var x;
    f(x) = ip(x);

    auto result = nanmax(f, s, {0}, /*keepdims=*/true);

    Halide::Runtime::Buffer<float> out(1);
    result.realize(out);

    EXPECT_NEAR(out(0), 4.0f, 1e-4f);
}

// -----------------------------------------------------------------------------
// NanMean Tests
// -----------------------------------------------------------------------------

TEST(NanOps, NanMean_1D) {
    // [1.0, NaN, 3.0, NaN] -> nanmean = (1+3)/2 = 2.0
    shape_t s = {4};
    Halide::ImageParam ip(Halide::Float(32), 1, "ip_nanmean");
    float nan = std::numeric_limits<float>::quiet_NaN();
    float data[] = {1.0f, nan, 3.0f, nan};
    Halide::Buffer<float> buf(data, 4);
    ip.set(buf);
    Halide::Func f("f_nanmean");
    Halide::Var x;
    f(x) = ip(x);

    auto result = nanmean(f, s, {0}, /*keepdims=*/true);

    Halide::Runtime::Buffer<float> out(1);
    result.realize(out);

    EXPECT_NEAR(out(0), 2.0f, 1e-4f);
}

TEST(NanOps, NanMean_NoNaN) {
    // [1.0, 2.0, 3.0, 4.0] -> nanmean = 2.5
    shape_t s = {4};
    Halide::ImageParam ip(Halide::Float(32), 1, "ip_nanmean_nonnan");
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    Halide::Buffer<float> buf(data, 4);
    ip.set(buf);
    Halide::Func f("f_nanmean_nonnan");
    Halide::Var x;
    f(x) = ip(x);

    auto result = nanmean(f, s, {0}, /*keepdims=*/true);

    Halide::Runtime::Buffer<float> out(1);
    result.realize(out);

    EXPECT_NEAR(out(0), 2.5f, 1e-4f);
}

// -----------------------------------------------------------------------------
// NanProd Tests
// -----------------------------------------------------------------------------

TEST(NanOps, NanProd_1D) {
    // [2.0, NaN, 3.0] -> nanprod = 6.0  (NaN treated as 1)
    shape_t s = {3};
    Halide::ImageParam ip(Halide::Float(32), 1, "ip_nanprod");
    float nan = std::numeric_limits<float>::quiet_NaN();
    float data[] = {2.0f, nan, 3.0f};
    Halide::Buffer<float> buf(data, 3);
    ip.set(buf);
    Halide::Func f("f_nanprod");
    Halide::Var x;
    f(x) = ip(x);

    auto result = nanprod(f, s, {0}, /*keepdims=*/true);

    Halide::Runtime::Buffer<float> out(1);
    result.realize(out);

    EXPECT_NEAR(out(0), 6.0f, 1e-4f);
}

TEST(NanOps, NanProd_AllNaN) {
    // [NaN, NaN] -> nanprod = 1.0 (all NaN treated as 1, identity element)
    shape_t s = {2};
    Halide::ImageParam ip(Halide::Float(32), 1, "ip_nanprod_all");
    float nan = std::numeric_limits<float>::quiet_NaN();
    float data[] = {nan, nan};
    Halide::Buffer<float> buf(data, 2);
    ip.set(buf);
    Halide::Func f("f_nanprod_all");
    Halide::Var x;
    f(x) = ip(x);

    auto result = nanprod(f, s, {0}, /*keepdims=*/true);

    Halide::Runtime::Buffer<float> out(1);
    result.realize(out);

    EXPECT_NEAR(out(0), 1.0f, 1e-4f);
}

// -----------------------------------------------------------------------------
// NanVar Tests
// -----------------------------------------------------------------------------

TEST(NanOps, NanVar_1D) {
    // [2.0, NaN, 4.0, NaN, 6.0] -> non-NaN: [2,4,6], mean=4
    // ssd = (2-4)^2 + (4-4)^2 + (6-4)^2 = 4+0+4 = 8
    // nanvar (ddof=0) = 8/3 ≈ 2.6667
    shape_t s = {5};
    Halide::ImageParam ip(Halide::Float(32), 1, "ip_nanvar");
    float nan = std::numeric_limits<float>::quiet_NaN();
    float data[] = {2.0f, nan, 4.0f, nan, 6.0f};
    Halide::Buffer<float> buf(data, 5);
    ip.set(buf);
    Halide::Func f("f_nanvar");
    Halide::Var x;
    f(x) = ip(x);

    auto result = nanvar(f, s, {0}, /*keepdims=*/true, /*ddof=*/0);

    Halide::Runtime::Buffer<float> out(1);
    result.realize(out);

    EXPECT_NEAR(out(0), 8.0f / 3.0f, 1e-4f);
}

TEST(NanOps, NanVar_SampleDdof1) {
    // [2.0, NaN, 4.0, NaN, 6.0] -> non-NaN: [2,4,6], count=3, mean=4
    // nanvar (ddof=1) = 8/(3-1) = 4.0
    shape_t s = {5};
    Halide::ImageParam ip(Halide::Float(32), 1, "ip_nanvar_ddof1");
    float nan = std::numeric_limits<float>::quiet_NaN();
    float data[] = {2.0f, nan, 4.0f, nan, 6.0f};
    Halide::Buffer<float> buf(data, 5);
    ip.set(buf);
    Halide::Func f("f_nanvar_ddof1");
    Halide::Var x;
    f(x) = ip(x);

    auto result = nanvar(f, s, {0}, /*keepdims=*/true, /*ddof=*/1);

    Halide::Runtime::Buffer<float> out(1);
    result.realize(out);

    EXPECT_NEAR(out(0), 4.0f, 1e-4f);
}

// -----------------------------------------------------------------------------
// NanStd Tests
// -----------------------------------------------------------------------------

TEST(NanOps, NanStd_1D) {
    // [2.0, NaN, 4.0, NaN, 6.0] -> nanstd = sqrt(8/3) ≈ 1.6330
    shape_t s = {5};
    Halide::ImageParam ip(Halide::Float(32), 1, "ip_nanstd");
    float nan = std::numeric_limits<float>::quiet_NaN();
    float data[] = {2.0f, nan, 4.0f, nan, 6.0f};
    Halide::Buffer<float> buf(data, 5);
    ip.set(buf);
    Halide::Func f("f_nanstd");
    Halide::Var x;
    f(x) = ip(x);

    auto result = nanstd(f, s, {0}, /*keepdims=*/true, /*ddof=*/0);

    Halide::Runtime::Buffer<float> out(1);
    result.realize(out);

    EXPECT_NEAR(out(0), std::sqrt(8.0f / 3.0f), 1e-4f);
}

TEST(NanOps, NanStd_NoNaN) {
    // [1.0, 2.0, 3.0] -> mean=2, ssd=1+0+1=2, std=sqrt(2/3)
    shape_t s = {3};
    Halide::ImageParam ip(Halide::Float(32), 1, "ip_nanstd_nonnan");
    float data[] = {1.0f, 2.0f, 3.0f};
    Halide::Buffer<float> buf(data, 3);
    ip.set(buf);
    Halide::Func f("f_nanstd_nonnan");
    Halide::Var x;
    f(x) = ip(x);

    auto result = nanstd(f, s, {0}, /*keepdims=*/true, /*ddof=*/0);

    Halide::Runtime::Buffer<float> out(1);
    result.realize(out);

    EXPECT_NEAR(out(0), std::sqrt(2.0f / 3.0f), 1e-4f);
}

TEST(NanOps, NanVarStdRelation) {
    // Verify nanstd = sqrt(nanvar) on the same data
    // [1.0, NaN, 5.0] -> non-NaN: [1,5], mean=3, ssd=4+4=8, var=4.0, std=2.0
    shape_t s = {3};
    Halide::ImageParam ip_var(Halide::Float(32), 1, "ip_nanvsr_var");
    Halide::ImageParam ip_std(Halide::Float(32), 1, "ip_nanvsr_std");
    float nan = std::numeric_limits<float>::quiet_NaN();
    float data[] = {1.0f, nan, 5.0f};
    Halide::Buffer<float> buf_var(data, 3);
    Halide::Buffer<float> buf_std(data, 3);
    ip_var.set(buf_var);
    ip_std.set(buf_std);

    Halide::Func fv("f_nanvsr_var"), fs("f_nanvsr_std");
    Halide::Var x;
    fv(x) = ip_var(x);
    fs(x) = ip_std(x);

    auto var_result = nanvar(fv, s, {0}, /*keepdims=*/true, /*ddof=*/0);
    auto std_result = nanstd(fs, s, {0}, /*keepdims=*/true, /*ddof=*/0);

    Halide::Runtime::Buffer<float> var_out(1);
    Halide::Runtime::Buffer<float> std_out(1);
    var_result.realize(var_out);
    std_result.realize(std_out);

    EXPECT_NEAR(std_out(0) * std_out(0), var_out(0), 1e-4f);
}
