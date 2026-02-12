/// @file test_rfft.cpp
/// @brief Tests for real FFT operations

#include <gtest/gtest.h>
#include "numhalide_all.h"
#include <cmath>

using namespace numhalide;

TEST(RFFT, DCComponent) {
	int N = 8;
	Halide::Func input("input");
	Halide::Var x;
	input(x) = 3.0f;  // constant = DC only

	Halide::Func result = rfft(input, N);

	// Read real/imag parts
	Halide::Func re("re"), im("im");
	re(x) = result(x)[0];
	im(x) = result(x)[1];

	Halide::Runtime::Buffer<float> re_out(N / 2 + 1);
	Halide::Runtime::Buffer<float> im_out(N / 2 + 1);
	re.realize(re_out);
	im.realize(im_out);

	// DC bin should be N*3 = 24
	EXPECT_NEAR(re_out(0), 24.0f, 1e-3f);
	EXPECT_NEAR(im_out(0), 0.0f, 1e-3f);

	// Other bins should be ~0
	for (int i = 1; i <= N / 2; ++i) {
		EXPECT_NEAR(re_out(i), 0.0f, 1e-3f);
		EXPECT_NEAR(im_out(i), 0.0f, 1e-3f);
	}
}

TEST(RFFT, Roundtrip) {
	int N = 8;
	Halide::Func input("input");
	Halide::Var x;
	input(x) = Halide::select(x == 0, 1.0f, x == 1, 2.0f, x == 2, 3.0f, x == 3, 4.0f,
	                          x == 4, 5.0f, x == 5, 6.0f, x == 6, 7.0f, 8.0f);

	Halide::Func freq = rfft(input, N);
	Halide::Func reconstructed = irfft(freq, N);

	Halide::Runtime::Buffer<float> out(N);
	reconstructed.realize(out);

	float expected[] = { 1, 2, 3, 4, 5, 6, 7, 8 };
	for (int i = 0; i < N; ++i) {
		EXPECT_NEAR(out(i), expected[i], 1e-2f);
	}
}

TEST(RFFT, FFTFreqValues) {
	int N = 8;
	Halide::Func freq = fftfreq(N);

	Halide::Runtime::Buffer<float> out(N);
	freq.realize(out);

	// Expected: [0, 1, 2, 3, -4, -3, -2, -1] / 8
	float expected[] = { 0, 0.125f, 0.25f, 0.375f, -0.5f, -0.375f, -0.25f, -0.125f };
	for (int i = 0; i < N; ++i) {
		EXPECT_NEAR(out(i), expected[i], 1e-5f);
	}
}

TEST(RFFT, RFFTFreqValues) {
	int N = 8;
	Halide::Func freq = rfftfreq(N);

	Halide::Runtime::Buffer<float> out(N / 2 + 1);
	freq.realize(out);

	// Expected: [0, 1, 2, 3, 4] / 8
	float expected[] = { 0, 0.125f, 0.25f, 0.375f, 0.5f };
	for (int i = 0; i <= N / 2; ++i) {
		EXPECT_NEAR(out(i), expected[i], 1e-5f);
	}
}
