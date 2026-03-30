/// @file test_complex_type.cpp
/// @brief Tests for complex_type.h (complex_f32 and ComplexBuffer)

#include <gtest/gtest.h>
#include "numhalide_all.h"
#include <cmath>

using namespace numhalide;

static const float PI = 3.14159265358979323846f;

// =============================================================================
// 1. Value_Construction
// =============================================================================
TEST(ComplexType, Value_Construction)
{
    complex_f32 z(3.0f, 4.0f);
    EXPECT_NEAR(z.re, 3.0f, 1e-5f);
    EXPECT_NEAR(z.im, 4.0f, 1e-5f);
}

// =============================================================================
// 2. Value_Add
// =============================================================================
TEST(ComplexType, Value_Add)
{
    complex_f32 a(1.0f, 2.0f);
    complex_f32 b(3.0f, 4.0f);
    complex_f32 c = a + b;
    EXPECT_NEAR(c.re, 4.0f, 1e-5f);
    EXPECT_NEAR(c.im, 6.0f, 1e-5f);
}

// =============================================================================
// 3. Value_Mul
// =============================================================================
TEST(ComplexType, Value_Mul)
{
    // (1+2i)*(3+4i) = (3-8) + (4+6)i = -5 + 10i
    complex_f32 a(1.0f, 2.0f);
    complex_f32 b(3.0f, 4.0f);
    complex_f32 c = a * b;
    EXPECT_NEAR(c.re, -5.0f, 1e-5f);
    EXPECT_NEAR(c.im, 10.0f, 1e-5f);
}

// =============================================================================
// 4. Value_Div
// =============================================================================
TEST(ComplexType, Value_Div)
{
    // (2+4i) / (1+2i)
    // numerator * conj(denom) = (2+4i)*(1-2i) = (2+8) + (-4+4)i = 10 + 0i
    // |denom|² = 1 + 4 = 5
    // result = 10/5 + 0/5*i = 2 + 0i
    complex_f32 a(2.0f, 4.0f);
    complex_f32 b(1.0f, 2.0f);
    complex_f32 c = a / b;
    EXPECT_NEAR(c.re, 2.0f, 1e-5f);
    EXPECT_NEAR(c.im, 0.0f, 1e-5f);
}

// =============================================================================
// 5. Value_Abs
// =============================================================================
TEST(ComplexType, Value_Abs)
{
    complex_f32 z(3.0f, 4.0f);
    EXPECT_NEAR(z.abs(), 5.0f, 1e-5f);
}

// =============================================================================
// 6. Value_Phase
// =============================================================================
TEST(ComplexType, Value_Phase)
{
    // phase of (1 + i) should be pi/4
    complex_f32 z(1.0f, 1.0f);
    EXPECT_NEAR(z.phase(), PI / 4.0f, 1e-5f);
}

// =============================================================================
// 7. Value_Conj
// =============================================================================
TEST(ComplexType, Value_Conj)
{
    complex_f32 z(3.0f, -4.0f);
    complex_f32 c = z.conj();
    EXPECT_NEAR(c.re,  3.0f, 1e-5f);
    EXPECT_NEAR(c.im,  4.0f, 1e-5f);
}

// =============================================================================
// 8. Value_PolarRoundtrip
// =============================================================================
TEST(ComplexType, Value_PolarRoundtrip)
{
    float mag   = 5.0f;
    float phase = PI / 3.0f;  // 60 degrees
    complex_f32 z = complex_from_polar(mag, phase);

    // re = 5*cos(π/3) = 5*0.5 = 2.5
    // im = 5*sin(π/3) = 5*sqrt(3)/2 ≈ 4.330
    EXPECT_NEAR(z.re, mag * std::cos(phase), 1e-5f);
    EXPECT_NEAR(z.im, mag * std::sin(phase), 1e-5f);

    // Round-trip: abs and phase should recover mag and phase
    EXPECT_NEAR(z.abs(),   mag,   1e-5f);
    EXPECT_NEAR(z.phase(), phase, 1e-5f);
}

// =============================================================================
// 9. Buffer_Construction
// =============================================================================
TEST(ComplexType, Buffer_Construction)
{
    ComplexBuffer cb(8);
    EXPECT_EQ(cb.size(), 8);
    for (int i = 0; i < 8; ++i) {
        EXPECT_NEAR(cb(i).re, 0.0f, 1e-5f);
        EXPECT_NEAR(cb(i).im, 0.0f, 1e-5f);
    }
}

// =============================================================================
// 10. Buffer_Mul
// =============================================================================
TEST(ComplexType, Buffer_Mul)
{
    ComplexBuffer a(4);
    ComplexBuffer b(4);
    // a[0] = 1+2i, b[0] = 3+4i -> product = -5+10i
    a.set(0, complex_f32(1.0f, 2.0f));
    b.set(0, complex_f32(3.0f, 4.0f));
    a.set(1, complex_f32(1.0f, 0.0f));
    b.set(1, complex_f32(2.0f, 0.0f));

    ComplexBuffer c = complex_buf_mul(a, b);
    EXPECT_NEAR(c(0).re, -5.0f, 1e-5f);
    EXPECT_NEAR(c(0).im, 10.0f, 1e-5f);
    EXPECT_NEAR(c(1).re,  2.0f, 1e-5f);
    EXPECT_NEAR(c(1).im,  0.0f, 1e-5f);
}

// =============================================================================
// 11. Buffer_Abs
// =============================================================================
TEST(ComplexType, Buffer_Abs)
{
    ComplexBuffer cb(4);
    cb.set(0, complex_f32(3.0f, 4.0f));
    cb.set(1, complex_f32(0.0f, 1.0f));
    cb.set(2, complex_f32(1.0f, 0.0f));
    cb.set(3, complex_f32(5.0f, 12.0f));

    auto mag = complex_buf_abs(cb);
    EXPECT_NEAR(mag(0),  5.0f, 1e-5f);
    EXPECT_NEAR(mag(1),  1.0f, 1e-5f);
    EXPECT_NEAR(mag(2),  1.0f, 1e-5f);
    EXPECT_NEAR(mag(3), 13.0f, 1e-5f);
}

// =============================================================================
// 12. Buffer_Phase
// =============================================================================
TEST(ComplexType, Buffer_Phase)
{
    ComplexBuffer cb(2);
    cb.set(0, complex_f32(1.0f, 1.0f));  // phase = pi/4
    cb.set(1, complex_f32(0.0f, 1.0f));  // phase = pi/2

    auto phases = complex_buf_phase(cb);
    EXPECT_NEAR(phases(0), PI / 4.0f, 1e-5f);
    EXPECT_NEAR(phases(1), PI / 2.0f, 1e-5f);
}

// =============================================================================
// ComplexBuffer × FFT integration tests
// =============================================================================

// Helper: build a Halide Func from a ComplexBuffer's re/im Runtime buffers.
// The Func wraps re/im via ImageParam to avoid constant-folding.
static Halide::Func complex_buf_to_func(
    const ComplexBuffer& cb,
    Halide::ImageParam& ip_re,
    Halide::ImageParam& ip_im,
    const std::string& name)
{
    // ImageParam needs non-const Buffer — take a copy into a Halide::Buffer
    Halide::Buffer<float> b_re(cb.size()), b_im(cb.size());
    for (int i = 0; i < cb.size(); ++i) {
        b_re(i) = cb(i).re;
        b_im(i) = cb(i).im;
    }
    ip_re.set(b_re);
    ip_im.set(b_im);

    Halide::Func f(name);
    Halide::Var x;
    f(x) = Halide::Tuple(ip_re(x), ip_im(x));
    return f;
}

// 13. FFT_DCOffset: DC-only input → FFT[0] = N, FFT[k>0] = 0
TEST(ComplexType, FFT_DCOffset)
{
    // Input: constant 1.0 real signal of length 4
    // DFT: X[0] = sum(1) = N = 4, X[k>0] = 0
    const int N = 4;
    ComplexBuffer cb(N);
    for (int i = 0; i < N; ++i)
        cb.set(i, complex_f32(1.0f, 0.0f));

    Halide::ImageParam ip_re(Halide::Float(32), 1, "fft_dc_re");
    Halide::ImageParam ip_im(Halide::Float(32), 1, "fft_dc_im");
    Halide::Func f = complex_buf_to_func(cb, ip_re, ip_im, "fft_dc_in");

    Halide::Func F = fft(f, N, "fft_dc");

    Halide::Runtime::Buffer<float> out_re(N), out_im(N);
    Halide::Realization r = F.realize({N});
    auto buf_re = r[0].as<float>();
    auto buf_im = r[1].as<float>();

    EXPECT_NEAR(buf_re(0), (float)N, 1e-4f);  // DC = N
    EXPECT_NEAR(buf_im(0), 0.0f, 1e-4f);
    for (int k = 1; k < N; ++k) {
        EXPECT_NEAR(buf_re(k), 0.0f, 1e-4f) << "Re k=" << k;
        EXPECT_NEAR(buf_im(k), 0.0f, 1e-4f) << "Im k=" << k;
    }
}

// 14. FFT_IFFTRoundtrip: FFT followed by normalized IFFT recovers original signal
TEST(ComplexType, FFT_IFFTRoundtrip)
{
    const int N = 8;
    ComplexBuffer cb(N);
    // Arbitrary real signal
    for (int i = 0; i < N; ++i)
        cb.set(i, complex_f32((float)(i % 3) - 1.0f, 0.0f));

    Halide::ImageParam ip_re(Halide::Float(32), 1, "fft_rt_re");
    Halide::ImageParam ip_im(Halide::Float(32), 1, "fft_rt_im");
    Halide::Func f_in = complex_buf_to_func(cb, ip_re, ip_im, "fft_rt_in");

    Halide::Func F    = fft(f_in, N, "fft_rt_fwd");
    Halide::Func Finv = ifft_normalized(F, N, "fft_rt_inv");

    Halide::Realization r = Finv.realize({N});
    auto buf_re = r[0].as<float>();
    auto buf_im = r[1].as<float>();

    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(buf_re(i), cb(i).re, 1e-4f) << "Re i=" << i;
        EXPECT_NEAR(buf_im(i), 0.0f,     1e-4f) << "Im i=" << i;
    }
}

// 15. FFT_ComplexBuf_AddThenFFT: (a+b) FFT == FFT(a) + FFT(b)  (linearity)
TEST(ComplexType, FFT_Linearity)
{
    const int N = 4;

    ComplexBuffer ca(N), cb_buf(N);
    ca.set(0, complex_f32(1.0f, 0.0f));
    ca.set(1, complex_f32(0.0f, 1.0f));
    cb_buf.set(0, complex_f32(2.0f, 0.0f));
    cb_buf.set(2, complex_f32(0.0f, -1.0f));

    ComplexBuffer sum_buf = complex_buf_add(ca, cb_buf);

    // Build Funcs for each
    Halide::ImageParam ip_re_a(Halide::Float(32), 1, "fft_lin_a_re");
    Halide::ImageParam ip_im_a(Halide::Float(32), 1, "fft_lin_a_im");
    Halide::ImageParam ip_re_b(Halide::Float(32), 1, "fft_lin_b_re");
    Halide::ImageParam ip_im_b(Halide::Float(32), 1, "fft_lin_b_im");
    Halide::ImageParam ip_re_s(Halide::Float(32), 1, "fft_lin_s_re");
    Halide::ImageParam ip_im_s(Halide::Float(32), 1, "fft_lin_s_im");

    Halide::Func fa = complex_buf_to_func(ca,      ip_re_a, ip_im_a, "fft_lin_a");
    Halide::Func fb = complex_buf_to_func(cb_buf,  ip_re_b, ip_im_b, "fft_lin_b");
    Halide::Func fs = complex_buf_to_func(sum_buf, ip_re_s, ip_im_s, "fft_lin_s");

    Halide::Func Fa = fft(fa, N, "fft_la");
    Halide::Func Fb = fft(fb, N, "fft_lb");
    Halide::Func Fs = fft(fs, N, "fft_ls");

    // FFT(a+b) should equal FFT(a) + FFT(b) element-wise
    Halide::Func F_sum("fft_lin_sum");
    Halide::Var k;
    F_sum(k) = Halide::Tuple(Fa(k)[0] + Fb(k)[0], Fa(k)[1] + Fb(k)[1]);

    Halide::Realization rs = Fs.realize({N});
    Halide::Realization rsum = F_sum.realize({N});

    auto sr   = rs[0].as<float>();
    auto si   = rs[1].as<float>();
    auto sumr = rsum[0].as<float>();
    auto sumi = rsum[1].as<float>();

    for (int k_idx = 0; k_idx < N; ++k_idx) {
        EXPECT_NEAR(sr(k_idx),  sumr(k_idx), 1e-4f) << "Re k=" << k_idx;
        EXPECT_NEAR(si(k_idx),  sumi(k_idx), 1e-4f) << "Im k=" << k_idx;
    }
}
