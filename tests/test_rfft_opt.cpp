/// @file test_rfft_opt.cpp
/// @brief Tests for rfft Hermitian optimization (correctness verification)
#include <gtest/gtest.h>
#include "numhalide_all.h"
#include <cmath>
#include <vector>
using namespace numhalide;

static const float PI = 3.14159265358979323846f;

// Helper: real 1D input Func from buffer
static Halide::Func make_real_func(const std::vector<float>& data, const std::string& nm)
{
    Halide::Buffer<float> buf((int)data.size());
    for (int i = 0; i < (int)data.size(); ++i) buf(i) = data[i];
    Halide::Func f(nm);
    Halide::Var x;
    f(x) = buf(Halide::clamp(x, 0, (int)data.size() - 1));
    return f;
}

// Impulse input: x[0]=1, rest=0 → all bins = (1, 0)
TEST(RfftOpt, Impulse1D) {
    const int N = 16;
    std::vector<float> data(N, 0.0f); data[0] = 1.0f;
    auto f = make_real_func(data, "rfft_imp");
    auto spec = rfft(f, N, "rfft_imp_out");

    Halide::Runtime::Buffer<float> re_buf(N / 2 + 1), im_buf(N / 2 + 1);
    auto re_f = Halide::Func("re"); Halide::Var x; re_f(x) = spec(x)[0]; re_f.realize(re_buf);
    auto im_f = Halide::Func("im"); im_f(x) = spec(x)[1]; im_f.realize(im_buf);

    for (int k = 0; k < N / 2 + 1; ++k) {
        EXPECT_NEAR(re_buf(k), 1.0f, 1e-4f) << "bin " << k;
        EXPECT_NEAR(im_buf(k), 0.0f, 1e-4f) << "bin " << k;
    }
}

// DC input: x[n]=1 for all n → bin0 = N, bins 1..N/2 ≈ 0
TEST(RfftOpt, DCInput) {
    const int N = 16;
    std::vector<float> data(N, 1.0f);
    auto f = make_real_func(data, "rfft_dc");
    auto spec = rfft(f, N, "rfft_dc_out");

    Halide::Runtime::Buffer<float> re_buf(N / 2 + 1), im_buf(N / 2 + 1);
    Halide::Var x;
    auto re_f = Halide::Func("re2"); re_f(x) = spec(x)[0]; re_f.realize(re_buf);
    auto im_f = Halide::Func("im2"); im_f(x) = spec(x)[1]; im_f.realize(im_buf);

    EXPECT_NEAR(re_buf(0), (float)N, 1e-3f);
    EXPECT_NEAR(im_buf(0), 0.0f, 1e-3f);
    for (int k = 1; k < N / 2 + 1; ++k) {
        EXPECT_NEAR(re_buf(k), 0.0f, 1e-3f) << "re bin " << k;
        EXPECT_NEAR(im_buf(k), 0.0f, 1e-3f) << "im bin " << k;
    }
}

// Roundtrip 1D: rfft then irfft should recover original
TEST(RfftOpt, Roundtrip1D) {
    const int N = 32;
    std::vector<float> data(N);
    for (int i = 0; i < N; ++i)
        data[i] = std::sin(2.0f * PI * 3.0f * i / N) + 0.5f * std::cos(2.0f * PI * 7.0f * i / N);

    auto f = make_real_func(data, "rfft_rt");
    auto spec = rfft(f, N, "rfft_rt_fwd");
    auto rec  = irfft(spec, N, "rfft_rt_inv");

    Halide::Runtime::Buffer<float> out(N);
    rec.realize(out);

    for (int i = 0; i < N; ++i)
        EXPECT_NEAR(out(i), data[i], 1e-4f) << "sample " << i;
}

// Hermitian symmetry: rfft2d bin (0,0) = sum of all elements
TEST(RfftOpt, Rfft2dDC) {
    const int rows = 8, cols = 8;
    Halide::Buffer<float> buf2d(cols, rows);
    float total = 0.0f;
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            buf2d(c, r) = 1.0f + 0.1f * (r + c);
            total += buf2d(c, r);
        }
    Halide::Func f2d("f2d");
    Halide::Var x, y;
    f2d(x, y) = buf2d(x, y);

    auto spec2d = rfft2d(f2d, rows, cols, "rfft2d_dc");
    Halide::Runtime::Buffer<float> re_buf(cols, rows), im_buf(cols, rows);
    auto re_f = Halide::Func("re3"); re_f(x, y) = spec2d(x, y)[0];
    auto im_f = Halide::Func("im3"); im_f(x, y) = spec2d(x, y)[1];
    re_f.realize(re_buf);
    im_f.realize(im_buf);

    // DC bin (0,0) should equal sum of all elements
    EXPECT_NEAR(re_buf(0, 0), total, total * 1e-4f);
    EXPECT_NEAR(im_buf(0, 0), 0.0f, 0.1f);
}
