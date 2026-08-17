/// @file test_fft.cpp
/// @brief Tests for FFT operations

#include <gtest/gtest.h>
#include "numhalide_all.h"
#include <cmath>

using namespace numhalide;

// Helper to extract real part from complex Func
Halide::Func extract_real(Halide::Func complex_func, const std::string& name = "real") {
    Halide::Func result(name);
    Halide::Var x("x");
    result(x) = complex_func(x)[0];
    return result;
}

Halide::Func extract_imag(Halide::Func complex_func, const std::string& name = "imag") {
    Halide::Func result(name);
    Halide::Var x("x");
    result(x) = complex_func(x)[1];
    return result;
}

Halide::Func extract_real_2d(Halide::Func complex_func, const std::string& name = "real") {
    Halide::Func result(name);
    Halide::Var x("x"), y("y");
    result(x, y) = complex_func(x, y)[0];
    return result;
}

Halide::Func extract_imag_2d(Halide::Func complex_func, const std::string& name = "imag") {
    Halide::Func result(name);
    Halide::Var x("x"), y("y");
    result(x, y) = complex_func(x, y)[1];
    return result;
}

// -----------------------------------------------------------------------------
// Complex Number Tests
// -----------------------------------------------------------------------------

TEST(FFT, ComplexAdd) {
    Halide::Func result("result");
    Halide::Var x;
    result(x) = complex_add(Halide::Tuple(1.0f, 2.0f), Halide::Tuple(3.0f, 4.0f));

    auto re = extract_real(result, "re");
    auto im = extract_imag(result, "im");

    Halide::Runtime::Buffer<float> re_out(1), im_out(1);
    re.realize(re_out);
    im.realize(im_out);

    EXPECT_NEAR(re_out(0), 4.0f, 1e-5f);  // 1 + 3
    EXPECT_NEAR(im_out(0), 6.0f, 1e-5f);  // 2 + 4
}

TEST(FFT, ComplexMul) {
    // (1 + 2i) * (3 + 4i) = 3 + 4i + 6i + 8i^2 = 3 + 10i - 8 = -5 + 10i
    Halide::Func result("result");
    Halide::Var x;
    result(x) = complex_mul(Halide::Tuple(1.0f, 2.0f), Halide::Tuple(3.0f, 4.0f));

    auto re = extract_real(result, "re");
    auto im = extract_imag(result, "im");

    Halide::Runtime::Buffer<float> re_out(1), im_out(1);
    re.realize(re_out);
    im.realize(im_out);

    EXPECT_NEAR(re_out(0), -5.0f, 1e-5f);
    EXPECT_NEAR(im_out(0), 10.0f, 1e-5f);
}

TEST(FFT, Expj) {
    // e^(j*0) = 1 + 0i
    // e^(j*pi/2) = 0 + 1i
    // e^(j*pi) = -1 + 0i
    Halide::Func result("result");
    Halide::Var x;

    float pi = static_cast<float>(M_PI);
    result(x) = expj(Halide::select(x == 0, 0.0f,
                     Halide::select(x == 1, pi / 2.0f, pi)));

    auto re = extract_real(result, "re");
    auto im = extract_imag(result, "im");

    Halide::Runtime::Buffer<float> re_out(3), im_out(3);
    re.realize(re_out);
    im.realize(im_out);

    EXPECT_NEAR(re_out(0), 1.0f, 1e-5f);
    EXPECT_NEAR(im_out(0), 0.0f, 1e-5f);

    EXPECT_NEAR(re_out(1), 0.0f, 1e-5f);
    EXPECT_NEAR(im_out(1), 1.0f, 1e-5f);

    EXPECT_NEAR(re_out(2), -1.0f, 1e-5f);
    EXPECT_NEAR(im_out(2), 0.0f, 1e-5f);
}

// -----------------------------------------------------------------------------
// 1D FFT Tests
// -----------------------------------------------------------------------------

TEST(FFT, FFT1D_DC) {
    // FFT of constant signal: all energy at DC
    // Input: [1, 1, 1, 1]
    // Output: [4, 0, 0, 0]
    Halide::Func input("input");
    Halide::Var x;
    input(x) = Halide::Tuple(1.0f, 0.0f);

    auto result = fft(input, 4, "fft");
    auto re = extract_real(result, "re");
    auto im = extract_imag(result, "im");

    Halide::Runtime::Buffer<float> re_out(4), im_out(4);
    re.realize(re_out);
    im.realize(im_out);

    EXPECT_NEAR(re_out(0), 4.0f, 1e-4f);  // DC component
    EXPECT_NEAR(im_out(0), 0.0f, 1e-4f);

    // Other components should be ~0
    for (int i = 1; i < 4; i++) {
        EXPECT_NEAR(re_out(i), 0.0f, 1e-4f);
        EXPECT_NEAR(im_out(i), 0.0f, 1e-4f);
    }
}

TEST(FFT, FFT1D_Impulse) {
    // FFT of impulse: flat spectrum
    // Input: [1, 0, 0, 0]
    // Output: [1, 1, 1, 1]
    Halide::Func input("input");
    Halide::Var x;
    input(x) = Halide::Tuple(
        Halide::select(x == 0, 1.0f, 0.0f),
        0.0f
    );

    auto result = fft(input, 4, "fft");
    auto re = extract_real(result, "re");
    auto im = extract_imag(result, "im");

    Halide::Runtime::Buffer<float> re_out(4), im_out(4);
    re.realize(re_out);
    im.realize(im_out);

    // All components should be 1
    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(re_out(i), 1.0f, 1e-4f);
        EXPECT_NEAR(im_out(i), 0.0f, 1e-4f);
    }
}

TEST(FFT, FFT1D_Roundtrip) {
    // Test FFT -> IFFT roundtrip
    // Input: [1, 2, 3, 4]
    Halide::Func input("input");
    Halide::Var x;
    input(x) = Halide::Tuple(Halide::cast<float>(x + 1), 0.0f);

    auto fft_result = fft(input, 4, "fft");
    auto ifft_result = ifft_normalized(fft_result, 4, "ifft");

    auto re = extract_real(ifft_result, "re");
    auto im = extract_imag(ifft_result, "im");

    Halide::Runtime::Buffer<float> re_out(4), im_out(4);
    re.realize(re_out);
    im.realize(im_out);

    // Should recover original signal
    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(re_out(i), static_cast<float>(i + 1), 1e-4f);
        EXPECT_NEAR(im_out(i), 0.0f, 1e-4f);
    }
}

TEST(FFT, FFT1D_Size8) {
    // Test larger FFT size
    Halide::Func input("input");
    Halide::Var x;
    input(x) = Halide::Tuple(1.0f, 0.0f);

    auto result = fft(input, 8, "fft8");
    auto re = extract_real(result, "re");
    auto im = extract_imag(result, "im");

    Halide::Runtime::Buffer<float> re_out(8), im_out(8);
    re.realize(re_out);
    im.realize(im_out);

    // DC should be 8, others should be 0
    EXPECT_NEAR(re_out(0), 8.0f, 1e-4f);
    for (int i = 1; i < 8; i++) {
        EXPECT_NEAR(re_out(i), 0.0f, 1e-4f);
        EXPECT_NEAR(im_out(i), 0.0f, 1e-4f);
    }
}

// -----------------------------------------------------------------------------
// 2D FFT Tests
// -----------------------------------------------------------------------------

TEST(FFT, FFT2D_DC) {
    // 2D FFT of constant: all energy at DC
    Halide::Func input("input");
    Halide::Var x, y;
    input(x, y) = Halide::Tuple(1.0f, 0.0f);

    auto result = fft2d(input, 4, 4, "fft2d");
    auto re = extract_real_2d(result, "re");
    auto im = extract_imag_2d(result, "im");

    Halide::Runtime::Buffer<float> re_out(4, 4), im_out(4, 4);
    re.realize(re_out);
    im.realize(im_out);

    // DC should be 16, others should be ~0
    EXPECT_NEAR(re_out(0, 0), 16.0f, 1e-3f);

    for (int j = 0; j < 4; j++) {
        for (int i = 0; i < 4; i++) {
            if (i != 0 || j != 0) {
                EXPECT_NEAR(re_out(i, j), 0.0f, 1e-3f);
                EXPECT_NEAR(im_out(i, j), 0.0f, 1e-3f);
            }
        }
    }
}

TEST(FFT, FFT2D_Roundtrip) {
    // Test 2D FFT -> IFFT roundtrip
    Halide::Func input("input");
    Halide::Var x, y;
    input(x, y) = Halide::Tuple(Halide::cast<float>(y * 4 + x + 1), 0.0f);

    auto fft_result = fft2d(input, 4, 4, "fft2d");
    auto ifft_result = ifft2d_normalized(fft_result, 4, 4, "ifft2d");

    auto re = extract_real_2d(ifft_result, "re");
    auto im = extract_imag_2d(ifft_result, "im");

    Halide::Runtime::Buffer<float> re_out(4, 4), im_out(4, 4);
    re.realize(re_out);
    im.realize(im_out);

    // Should recover original
    for (int j = 0; j < 4; j++) {
        for (int i = 0; i < 4; i++) {
            float expected = static_cast<float>(j * 4 + i + 1);
            EXPECT_NEAR(re_out(i, j), expected, 1e-3f);
            EXPECT_NEAR(im_out(i, j), 0.0f, 1e-3f);
        }
    }
}

// -----------------------------------------------------------------------------
// FFT Utility Tests
// -----------------------------------------------------------------------------

TEST(FFT, FFTShift1D) {
    // Input: [0, 1, 2, 3]
    // After shift: [2, 3, 0, 1]
    Halide::Func input("input");
    Halide::Var x;
    input(x) = Halide::Tuple(Halide::cast<float>(x), 0.0f);

    auto result = fftshift_1d(input, 4, "shift");
    auto re = extract_real(result, "re");

    Halide::Runtime::Buffer<float> re_out(4);
    re.realize(re_out);

    EXPECT_NEAR(re_out(0), 2.0f, 1e-5f);
    EXPECT_NEAR(re_out(1), 3.0f, 1e-5f);
    EXPECT_NEAR(re_out(2), 0.0f, 1e-5f);
    EXPECT_NEAR(re_out(3), 1.0f, 1e-5f);
}

TEST(FFT, PowerSpectrum) {
    // Input: 3 + 4i -> |z|^2 = 9 + 16 = 25
    Halide::Func input("input");
    Halide::Var x;
    input(x) = Halide::Tuple(3.0f, 4.0f);

    auto result = power_spectrum(input, "power");

    Halide::Runtime::Buffer<float> out(1);
    result.realize(out);

    EXPECT_NEAR(out(0), 25.0f, 1e-5f);
}

TEST(FFT, RealToComplex) {
    Halide::Func input("input");
    Halide::Var x;
    input(x) = Halide::cast<float>(x + 1);  // [1, 2, 3, 4]

    auto result = real_to_complex(input, "r2c");
    auto re = extract_real(result, "re");
    auto im = extract_imag(result, "im");

    Halide::Runtime::Buffer<float> re_out(4), im_out(4);
    re.realize(re_out);
    im.realize(im_out);

    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(re_out(i), static_cast<float>(i + 1), 1e-5f);
        EXPECT_NEAR(im_out(i), 0.0f, 1e-5f);
    }
}

// -----------------------------------------------------------------------------
// Real-input transforms with a HALF spectrum (numpy rfft / irfft semantics)
// -----------------------------------------------------------------------------
//
// dft_1d applied to real input returns ALL n bins; rdft/irdft use the n/2+1 the
// Hermitian symmetry makes independent. Both contracts are valid and they are
// NOT interchangeable -- their adjoints differ by a factor of 2 on every bin
// that stands for a conjugate pair. These pin the half-spectrum one.

TEST(FFTReal, IrfftRoundtrip1D) {
	const int N = 8;
	Halide::Buffer<float> sig(N);
	for (int i = 0; i < N; ++i)
		sig(i) = std::sin(0.7f * i) + 0.3f * std::cos(2.1f * i) + 0.1f * i;

	Halide::Var i;
	Halide::Func in("rt_in");
	in(i) = sig(i);

	Halide::Func half = rdft_1d(in, N, Halide::Float(32), "rt_half");
	Halide::Func back = irdft_1d_normalized(half, N, Halide::Float(32), "rt_back");

	Halide::Buffer<float> out = back.realize({N});
	for (int k = 0; k < N; ++k)
		EXPECT_NEAR(out(k), sig(k), 1e-4) << "roundtrip at " << k;
}

// The half spectrum must agree with the full DFT on the bins it keeps -- that
// is what makes it a HALF spectrum rather than a different transform.
TEST(FFTReal, HalfSpectrumMatchesFullDft1D) {
	const int N = 8;
	Halide::Buffer<float> sig(N);
	for (int i = 0; i < N; ++i) sig(i) = std::cos(1.3f * i) - 0.4f * i;

	Halide::Var i;
	Halide::Func in("hs_in");
	in(i) = sig(i);

	Halide::Func packed("hs_packed");
	packed(i) = complex(in(i), Halide::Expr(0.0f));
	Halide::Func full = dft_1d(packed, N, -1, Halide::Float(32), "hs_full");
	Halide::Func half = rdft_1d(in, N, Halide::Float(32), "hs_half");

	Halide::Realization fr = full.realize({N});
	Halide::Realization hr = half.realize({N / 2 + 1});
	Halide::Buffer<float> fre = fr[0], fim = fr[1];
	Halide::Buffer<float> hre = hr[0], him = hr[1];

	for (int k = 0; k <= N / 2; ++k) {
		EXPECT_NEAR(hre(k), fre(k), 1e-4) << "re bin " << k;
		EXPECT_NEAR(him(k), fim(k), 1e-4) << "im bin " << k;
	}
}

TEST(FFTReal, IrfftRoundtrip2D) {
	const int W = 4, H = 4;
	Halide::Buffer<float> img(W, H);
	for (int y = 0; y < H; ++y)
		for (int x = 0; x < W; ++x)
			img(x, y) = std::sin(0.9f * x) * std::cos(0.6f * y) + 0.2f * (x + y);

	Halide::Var x, y;
	Halide::Func in("rt2_in");
	in(x, y) = img(x, y);

	Halide::Func half = rdft_2d(in, W, H, Halide::Float(32), "rt2_half");
	Halide::Func back = irdft_2d_normalized(half, W, H, Halide::Float(32), "rt2_back");

	Halide::Buffer<float> out = back.realize({W, H});
	for (int yy = 0; yy < H; ++yy)
		for (int xx = 0; xx < W; ++xx)
			EXPECT_NEAR(out(xx, yy), img(xx, yy), 1e-3)
				<< "roundtrip at (" << xx << "," << yy << ")";
}
