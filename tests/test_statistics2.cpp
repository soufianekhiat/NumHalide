/// @file test_statistics2.cpp
/// @brief Tests for extended statistical operations: percentile, quantile,
///        cov, corrcoef, histogram2d, bincount

#include <gtest/gtest.h>
#include "numhalide_all.h"
#include <cmath>
#include <limits>

using namespace numhalide;

// -----------------------------------------------------------------------------
// Percentile Tests (power-of-2 size required for bitonic sort)
// -----------------------------------------------------------------------------

TEST(Statistics2, Percentile_Median) {
    // Sorted input [1,2,3,4], q=50
    // pos = 0.5 * (4-1) = 1.5 -> lo=1, hi=2, frac=0.5
    // result = 2.0*0.5 + 3.0*0.5 = 2.5
    Halide::Func f("perc_median");
    Halide::Var x;
    f(x) = Halide::select(x == 0, 1.0f, x == 1, 2.0f, x == 2, 3.0f, 4.0f);

    auto result = stats::percentile(f, 4, 50.0f);

    Halide::Runtime::Buffer<float> out(1);
    result.realize(out);

    EXPECT_NEAR(out(0), 2.5f, 1e-4f);
}

TEST(Statistics2, Percentile_Min) {
    // q=0.0 -> minimum element
    // pos = 0.0 * 3 = 0.0 -> lo=hi=0, result = sorted[0] = 1.0
    Halide::Func f("perc_min");
    Halide::Var x;
    f(x) = Halide::select(x == 0, 1.0f, x == 1, 2.0f, x == 2, 3.0f, 4.0f);

    auto result = stats::percentile(f, 4, 0.0f);

    Halide::Runtime::Buffer<float> out(1);
    result.realize(out);

    EXPECT_NEAR(out(0), 1.0f, 1e-4f);
}

TEST(Statistics2, Percentile_Max) {
    // q=100.0 -> maximum element
    // pos = 1.0 * 3 = 3.0 -> lo=hi=3, result = sorted[3] = 4.0
    Halide::Func f("perc_max");
    Halide::Var x;
    f(x) = Halide::select(x == 0, 1.0f, x == 1, 2.0f, x == 2, 3.0f, 4.0f);

    auto result = stats::percentile(f, 4, 100.0f);

    Halide::Runtime::Buffer<float> out(1);
    result.realize(out);

    EXPECT_NEAR(out(0), 4.0f, 1e-4f);
}

TEST(Statistics2, Percentile_UnsortedInput) {
    // Unsorted [4,2,1,3] should sort to [1,2,3,4] first
    // q=50 -> same result: 2.5
    Halide::Func f("perc_unsorted");
    Halide::Var x;
    f(x) = Halide::select(x == 0, 4.0f, x == 1, 2.0f, x == 2, 1.0f, 3.0f);

    auto result = stats::percentile(f, 4, 50.0f);

    Halide::Runtime::Buffer<float> out(1);
    result.realize(out);

    EXPECT_NEAR(out(0), 2.5f, 1e-4f);
}

// -----------------------------------------------------------------------------
// Quantile Tests
// -----------------------------------------------------------------------------

TEST(Statistics2, Quantile_Half) {
    // quantile(q=0.5) == percentile(q=50) == 2.5 for [1,2,3,4]
    Halide::Func f("quant_half");
    Halide::Var x;
    f(x) = Halide::select(x == 0, 1.0f, x == 1, 2.0f, x == 2, 3.0f, 4.0f);

    auto result = stats::quantile(f, 4, 0.5f);

    Halide::Runtime::Buffer<float> out(1);
    result.realize(out);

    EXPECT_NEAR(out(0), 2.5f, 1e-4f);
}

TEST(Statistics2, Quantile_Zero) {
    // quantile(q=0.0) == minimum
    Halide::Func f("quant_zero");
    Halide::Var x;
    f(x) = Halide::select(x == 0, 4.0f, x == 1, 1.0f, x == 2, 3.0f, 2.0f);

    auto result = stats::quantile(f, 4, 0.0f);

    Halide::Runtime::Buffer<float> out(1);
    result.realize(out);

    EXPECT_NEAR(out(0), 1.0f, 1e-4f);
}

TEST(Statistics2, Quantile_One) {
    // quantile(q=1.0) == maximum
    Halide::Func f("quant_one");
    Halide::Var x;
    f(x) = Halide::select(x == 0, 4.0f, x == 1, 1.0f, x == 2, 3.0f, 2.0f);

    auto result = stats::quantile(f, 4, 1.0f);

    Halide::Runtime::Buffer<float> out(1);
    result.realize(out);

    EXPECT_NEAR(out(0), 4.0f, 1e-4f);
}

// -----------------------------------------------------------------------------
// Covariance Tests
// -----------------------------------------------------------------------------

TEST(Statistics2, Cov_Basic) {
    // a=[1,2,3], b=[4,5,6], ddof=1
    // mean_a=2, mean_b=5
    // sum((a-mean_a)*(b-mean_b)) = (-1)(-1) + 0*0 + 1*1 = 2
    // cov = 2 / (3-1) = 1.0
    Halide::Func fa("cov_a"), fb("cov_b");
    Halide::Var x;
    fa(x) = Halide::cast<float>(x + 1);   // 1, 2, 3
    fb(x) = Halide::cast<float>(x + 4);   // 4, 5, 6

    auto result = stats::cov(fa, fb, 3, 1);

    Halide::Runtime::Buffer<float> out(1);
    result.realize(out);

    EXPECT_NEAR(out(0), 1.0f, 1e-4f);
}

TEST(Statistics2, Cov_PopulationDdof0) {
    // a=[1,2,3], b=[4,5,6], ddof=0
    // cov = 2 / (3-0) = 2/3 ≈ 0.6667
    Halide::Func fa("cov_a_pop"), fb("cov_b_pop");
    Halide::Var x;
    fa(x) = Halide::cast<float>(x + 1);
    fb(x) = Halide::cast<float>(x + 4);

    auto result = stats::cov(fa, fb, 3, 0);

    Halide::Runtime::Buffer<float> out(1);
    result.realize(out);

    EXPECT_NEAR(out(0), 2.0f / 3.0f, 1e-4f);
}

TEST(Statistics2, Cov_Symmetric) {
    // cov(a, b) == cov(b, a)
    Halide::Func fa("cov_sym_a"), fb("cov_sym_b");
    Halide::Func fa2("cov_sym_a2"), fb2("cov_sym_b2");
    Halide::Var x;
    fa(x) = Halide::cast<float>(x + 1);
    fb(x) = Halide::cast<float>(x + 4);
    fa2(x) = Halide::cast<float>(x + 1);
    fb2(x) = Halide::cast<float>(x + 4);

    auto result_ab = stats::cov(fa, fb, 3, 1);
    auto result_ba = stats::cov(fb2, fa2, 3, 1);

    Halide::Runtime::Buffer<float> out_ab(1);
    Halide::Runtime::Buffer<float> out_ba(1);
    result_ab.realize(out_ab);
    result_ba.realize(out_ba);

    EXPECT_NEAR(out_ab(0), out_ba(0), 1e-4f);
}

// -----------------------------------------------------------------------------
// Pearson Correlation Coefficient Tests
// -----------------------------------------------------------------------------

TEST(Statistics2, Corrcoef_Perfect) {
    // a=[1,2,3], b=[2,4,6] (perfectly positively correlated)
    // corrcoef should be exactly 1.0
    Halide::Func fa("corr_pos_a"), fb("corr_pos_b");
    Halide::Var x;
    fa(x) = Halide::cast<float>(x + 1);        // 1, 2, 3
    fb(x) = Halide::cast<float>((x + 1) * 2);  // 2, 4, 6

    auto result = stats::corrcoef(fa, fb, 3);

    Halide::Runtime::Buffer<float> out(1);
    result.realize(out);

    EXPECT_NEAR(out(0), 1.0f, 1e-4f);
}

TEST(Statistics2, Corrcoef_Negative) {
    // a=[1,2,3], b=[3,2,1] (perfectly negatively correlated)
    // corrcoef should be exactly -1.0
    Halide::Func fa("corr_neg_a"), fb("corr_neg_b");
    Halide::Var x;
    fa(x) = Halide::cast<float>(x + 1);     // 1, 2, 3
    fb(x) = Halide::cast<float>(3 - x);     // 3, 2, 1

    auto result = stats::corrcoef(fa, fb, 3);

    Halide::Runtime::Buffer<float> out(1);
    result.realize(out);

    EXPECT_NEAR(out(0), -1.0f, 1e-4f);
}

TEST(Statistics2, Corrcoef_Uncorrelated) {
    // a=[1,2,3,4], b=[2,2,2,2] (b is constant, zero variance)
    // corrcoef = 0 / sqrt(var_a * 0) -> undefined (0/0), but
    // sum_sq_b = 0 so result is NaN or +/-Inf in practice.
    // This test just checks it doesn't crash; we don't assert a specific value.
    Halide::Func fa("corr_unc_a"), fb("corr_unc_b");
    Halide::Var x;
    fa(x) = Halide::cast<float>(x + 1);
    fb(x) = 2.0f;

    auto result = stats::corrcoef(fa, fb, 4);

    Halide::Runtime::Buffer<float> out(1);
    EXPECT_NO_THROW(result.realize(out));
    // Result is mathematically undefined; just verify it does not throw.
}

// -----------------------------------------------------------------------------
// Histogram2D Tests
// -----------------------------------------------------------------------------

TEST(Statistics2, Histogram2D_Basic) {
    // x=[0.5, 1.5], y=[0.5, 1.5], 2x2 bins over [0,2) x [0,2)
    // (0.5,0.5) -> xi=0, yi=0 -> bin (0,0)
    // (1.5,1.5) -> xi=1, yi=1 -> bin (1,1)
    Halide::Func x_vals("hist2d_x"), y_vals("hist2d_y");
    Halide::Var i;
    x_vals(i) = Halide::select(i == 0, 0.5f, 1.5f);
    y_vals(i) = Halide::select(i == 0, 0.5f, 1.5f);

    auto result = stats::histogram2d(
        x_vals, y_vals, 2,
        /*x_bins=*/2, /*y_bins=*/2,
        /*x_min=*/0.0f, /*x_max=*/2.0f,
        /*y_min=*/0.0f, /*y_max=*/2.0f);
    result.compute_root();

    Halide::Runtime::Buffer<int32_t> out(2, 2);
    result.realize(out);

    // out(bx, by) -- Halide dim0=x_bins, dim1=y_bins
    EXPECT_EQ(out(0, 0), 1);  // (0.5, 0.5)
    EXPECT_EQ(out(1, 1), 1);  // (1.5, 1.5)
    EXPECT_EQ(out(1, 0), 0);
    EXPECT_EQ(out(0, 1), 0);
}

TEST(Statistics2, Histogram2D_AllSameBin) {
    // All 4 points fall in the same bin (0,0)
    Halide::Func x_vals("hist2d_same_x"), y_vals("hist2d_same_y");
    Halide::Var i;
    x_vals(i) = 0.1f;  // always bin 0 on x
    y_vals(i) = 0.1f;  // always bin 0 on y

    auto result = stats::histogram2d(
        x_vals, y_vals, 4,
        /*x_bins=*/2, /*y_bins=*/2,
        /*x_min=*/0.0f, /*x_max=*/2.0f,
        /*y_min=*/0.0f, /*y_max=*/2.0f);
    result.compute_root();

    Halide::Runtime::Buffer<int32_t> out(2, 2);
    result.realize(out);

    EXPECT_EQ(out(0, 0), 4);
    EXPECT_EQ(out(1, 0), 0);
    EXPECT_EQ(out(0, 1), 0);
    EXPECT_EQ(out(1, 1), 0);
}

TEST(Statistics2, Histogram2D_OutOfRange) {
    // Points outside the range are discarded
    // x=[-1.0, 3.0], y=[0.5, 0.5]: both out of [0,2) on x -> both discarded
    Halide::Func x_vals("hist2d_oor_x"), y_vals("hist2d_oor_y");
    Halide::Var i;
    x_vals(i) = Halide::select(i == 0, -1.0f, 3.0f);
    y_vals(i) = 0.5f;

    auto result = stats::histogram2d(
        x_vals, y_vals, 2,
        /*x_bins=*/2, /*y_bins=*/2,
        /*x_min=*/0.0f, /*x_max=*/2.0f,
        /*y_min=*/0.0f, /*y_max=*/2.0f);
    result.compute_root();

    Halide::Runtime::Buffer<int32_t> out(2, 2);
    result.realize(out);

    EXPECT_EQ(out(0, 0), 0);
    EXPECT_EQ(out(1, 0), 0);
    EXPECT_EQ(out(0, 1), 0);
    EXPECT_EQ(out(1, 1), 0);
}

// -----------------------------------------------------------------------------
// Bincount Tests
// -----------------------------------------------------------------------------

TEST(Statistics2, Bincount_Basic) {
    // [0, 1, 1, 2, 2, 2], out_size=3
    // count(0)=1, count(1)=2, count(2)=3
    Halide::Func f("bincount_basic");
    Halide::Var x;
    f(x) = Halide::select(x == 0, 0,
           Halide::select(x == 1, 1,
           Halide::select(x == 2, 1,
           Halide::select(x == 3, 2,
           Halide::select(x == 4, 2, 2)))));

    auto result = stats::bincount(f, 6, 3);
    result.compute_root();

    Halide::Runtime::Buffer<int32_t> out(3);
    result.realize(out);

    EXPECT_EQ(out(0), 1);
    EXPECT_EQ(out(1), 2);
    EXPECT_EQ(out(2), 3);
}

TEST(Statistics2, Bincount_OutOfRange) {
    // Values >= out_size are silently ignored
    // [0, 5, 1, 5], out_size=3 -> only 0 and 1 count; 5 is ignored
    Halide::Func f("bincount_oor");
    Halide::Var x;
    f(x) = Halide::select(x == 0, 0,
           Halide::select(x == 1, 5,
           Halide::select(x == 2, 1, 5)));

    auto result = stats::bincount(f, 4, 3);
    result.compute_root();

    Halide::Runtime::Buffer<int32_t> out(3);
    result.realize(out);

    EXPECT_EQ(out(0), 1);
    EXPECT_EQ(out(1), 1);
    EXPECT_EQ(out(2), 0);
}

TEST(Statistics2, Bincount_SingleValue) {
    // All elements are 0; out_size=4
    Halide::Func f("bincount_single");
    Halide::Var x;
    f(x) = 0;

    auto result = stats::bincount(f, 5, 4);
    result.compute_root();

    Halide::Runtime::Buffer<int32_t> out(4);
    result.realize(out);

    EXPECT_EQ(out(0), 5);
    EXPECT_EQ(out(1), 0);
    EXPECT_EQ(out(2), 0);
    EXPECT_EQ(out(3), 0);
}
