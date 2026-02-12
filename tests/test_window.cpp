/// @file test_window.cpp
/// @brief Tests for window functions

#include <gtest/gtest.h>
#include "numhalide_all.h"
#include <cmath>

using namespace numhalide;

TEST(Window, HanningEndpoints) {
	int N = 16;
	Halide::Func w = hanning(N);

	Halide::Runtime::Buffer<float> out(N);
	w.realize(out);

	EXPECT_NEAR(out(0), 0.0f, 1e-5f);
	EXPECT_NEAR(out(N - 1), 0.0f, 1e-5f);
}

TEST(Window, HanningSymmetry) {
	int N = 16;
	Halide::Func w = hanning(N);

	Halide::Runtime::Buffer<float> out(N);
	w.realize(out);

	for (int i = 0; i < N / 2; ++i) {
		EXPECT_NEAR(out(i), out(N - 1 - i), 1e-5f);
	}
}

TEST(Window, HanningPeak) {
	int N = 17;
	Halide::Func w = hanning(N);

	Halide::Runtime::Buffer<float> out(N);
	w.realize(out);

	EXPECT_NEAR(out(8), 1.0f, 1e-5f);  // center
}

TEST(Window, HammingEndpoints) {
	int N = 16;
	Halide::Func w = hamming(N);

	Halide::Runtime::Buffer<float> out(N);
	w.realize(out);

	EXPECT_NEAR(out(0), 0.08f, 1e-3f);
	EXPECT_NEAR(out(N - 1), 0.08f, 1e-3f);
}

TEST(Window, HammingSymmetry) {
	int N = 16;
	Halide::Func w = hamming(N);

	Halide::Runtime::Buffer<float> out(N);
	w.realize(out);

	for (int i = 0; i < N / 2; ++i) {
		EXPECT_NEAR(out(i), out(N - 1 - i), 1e-5f);
	}
}

TEST(Window, BlackmanEndpoints) {
	int N = 32;
	Halide::Func w = blackman(N);

	Halide::Runtime::Buffer<float> out(N);
	w.realize(out);

	EXPECT_NEAR(out(0), 0.0f, 1e-4f);
	EXPECT_NEAR(out(N - 1), 0.0f, 1e-4f);
}

TEST(Window, BlackmanSymmetry) {
	int N = 32;
	Halide::Func w = blackman(N);

	Halide::Runtime::Buffer<float> out(N);
	w.realize(out);

	for (int i = 0; i < N / 2; ++i) {
		EXPECT_NEAR(out(i), out(N - 1 - i), 1e-5f);
	}
}

TEST(Window, BartlettTriangle) {
	int N = 5;
	Halide::Func w = bartlett(N);

	Halide::Runtime::Buffer<float> out(N);
	w.realize(out);

	// Expected: 0, 0.5, 1.0, 0.5, 0
	EXPECT_NEAR(out(0), 0.0f, 1e-5f);
	EXPECT_NEAR(out(1), 0.5f, 1e-5f);
	EXPECT_NEAR(out(2), 1.0f, 1e-5f);
	EXPECT_NEAR(out(3), 0.5f, 1e-5f);
	EXPECT_NEAR(out(4), 0.0f, 1e-5f);
}

TEST(Window, BartlettPeak) {
	int N = 17;
	Halide::Func w = bartlett(N);

	Halide::Runtime::Buffer<float> out(N);
	w.realize(out);

	EXPECT_NEAR(out(8), 1.0f, 1e-5f);
}
