/// @file test_fft_ext.cpp
/// @brief Tests for extended FFT operations

#include <gtest/gtest.h>
#include "numhalide_all.h"
#include <cmath>

using namespace numhalide;

// Helper to extract real/imag parts from 2D complex Func (Tuple)
static Halide::Func ext_real_2d(Halide::Func f, const std::string& name = "real") {
	Halide::Func ret(name);
	Halide::Var x("x"), y("y");
	ret(x, y) = f(x, y)[0];
	return ret;
}
static Halide::Func ext_imag_2d(Halide::Func f, const std::string& name = "imag") {
	Halide::Func ret(name);
	Halide::Var x("x"), y("y");
	ret(x, y) = f(x, y)[1];
	return ret;
}

TEST(FFTExt, CrossPowerSpectrumSelf) {
	// CPS of a signal with itself should have unit magnitude
	Halide::Func a("a");
	Halide::Var x, y;
	a(x, y) = Halide::Tuple(
		Halide::cast<float>(x + y),
		Halide::cast<float>(x - y)
	);

	Halide::Func result = cross_power_spectrum(a, a, 4, 4);

	auto re_func = ext_real_2d(result, "re");
	auto im_func = ext_imag_2d(result, "im");

	Halide::Runtime::Buffer<float> re(4, 4), im(4, 4);
	re_func.realize(re);
	im_func.realize(im);

	// CPS(A,A) = |A|^2/|A|^2 = 1+0j (normalized) where A != 0
	for (int j = 0; j < 4; ++j) {
		for (int i = 0; i < 4; ++i) {
			// Skip origin where input is (0,0) — CPS is undefined there
			if (i == 0 && j == 0) continue;
			float mag = std::sqrt(re(i, j) * re(i, j) + im(i, j) * im(i, j));
			EXPECT_NEAR(mag, 1.0f, 1e-3f);
		}
	}
}

TEST(FFTExt, CrossPowerSpectrumReal) {
	// Real-only signal: imaginary part of A*conj(A) is 0
	Halide::Func a("a");
	Halide::Var x, y;
	a(x, y) = Halide::Tuple(Halide::cast<float>(x + 1), 0.0f);

	Halide::Func result = cross_power_spectrum(a, a, 4, 4);

	auto re_func = ext_real_2d(result, "re");
	auto im_func = ext_imag_2d(result, "im");

	Halide::Runtime::Buffer<float> re(4, 4), im(4, 4);
	re_func.realize(re);
	im_func.realize(im);

	// For real signals: A*conj(A) = |A|^2 + 0i, so normalized = 1+0i
	for (int j = 0; j < 4; ++j) {
		for (int i = 0; i < 4; ++i) {
			EXPECT_NEAR(re(i, j), 1.0f, 1e-3f);
			EXPECT_NEAR(im(i, j), 0.0f, 1e-3f);
		}
	}
}

TEST(FFTExt, SpectralCentroidDC) {
	// Pure DC signal: all energy at bin 0 -> centroid = 0
	Halide::Func input("input");
	Halide::Var x;
	input(x) = Halide::Tuple(
		Halide::select(x == 0, 10.0f, 0.0f),
		0.0f
	);

	Halide::Func result = spectral_centroid(input, 8);

	Halide::Runtime::Buffer<float> out(1);
	result.realize(out);

	EXPECT_NEAR(out(0), 0.0f, 1e-4f);
}

TEST(FFTExt, SpectralCentroidHighFreq) {
	// All energy at highest bin -> centroid = N-1
	int N = 8;
	Halide::Func input("input");
	Halide::Var x;
	input(x) = Halide::Tuple(
		Halide::select(x == N - 1, 10.0f, 0.0f),
		0.0f
	);

	Halide::Func result = spectral_centroid(input, N);

	Halide::Runtime::Buffer<float> out(1);
	result.realize(out);

	EXPECT_NEAR(out(0), static_cast<float>(N - 1), 1e-4f);
}
