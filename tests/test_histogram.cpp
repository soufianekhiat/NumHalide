/// @file test_histogram.cpp
/// @brief Tests for histogram and LUT operations

#include <gtest/gtest.h>
#include "numhalide_all.h"

using namespace numhalide;

TEST(Histogram, UniformBins) {
	shape_t s = { 4 };
	Halide::Func input("input");
	Halide::Var x;
	// Values: 0.1, 0.3, 0.5, 0.7 -> bins 0, 1, 2, 3 with 4 bins in [0,1)
	input(x) = Halide::select(x == 0, 0.1f, x == 1, 0.3f, x == 2, 0.5f, 0.7f);

	Halide::Func result = histogram_1d(input, s, 4, 0.0f, 1.0f);

	Halide::Runtime::Buffer<int32_t> out(4);
	result.realize(out);

	// Each bin should have exactly 1 element
	for (int i = 0; i < 4; ++i)
		EXPECT_EQ(out(i), 1);
}

TEST(Histogram, AllSameBin) {
	shape_t s = { 4 };
	Halide::Func input("input");
	Halide::Var x;
	input(x) = 0.5f;  // all same value

	Halide::Func result = histogram_1d(input, s, 4, 0.0f, 1.0f);

	Halide::Runtime::Buffer<int32_t> out(4);
	result.realize(out);

	// All 4 elements should be in bin 2 (0.5 * 4 = 2)
	int total = 0;
	for (int i = 0; i < 4; ++i)
		total += out(i);
	EXPECT_EQ(total, 4);
}

TEST(Histogram, GammaCorrection) {
	shape_t s = { 4 };
	Halide::Func input("input");
	Halide::Var x;
	input(x) = Halide::select(x == 0, 0.0f, x == 1, 0.25f, x == 2, 0.5f, 1.0f);

	Halide::Func result = gamma_correct(input, s, 2.0f);

	Halide::Runtime::Buffer<float> out(4);
	result.realize(out);

	EXPECT_NEAR(out(0), 0.0f, 1e-5f);      // pow(0, 0.5) = 0
	EXPECT_NEAR(out(1), 0.5f, 1e-5f);      // pow(0.25, 0.5) = 0.5
	EXPECT_NEAR(out(2), std::sqrt(0.5f), 1e-5f);
	EXPECT_NEAR(out(3), 1.0f, 1e-5f);      // pow(1, 0.5) = 1
}

TEST(Histogram, ApplyLut) {
	shape_t s = { 4 };
	Halide::Func input("input");
	Halide::Var x;
	input(x) = Halide::cast<int32_t>(x);  // 0, 1, 2, 3

	Halide::Func lut("lut");
	lut(x) = Halide::cast<float>(x * x);  // 0, 1, 4, 9

	Halide::Func result = apply_lut(input, lut, s);

	Halide::Runtime::Buffer<float> out(4);
	result.realize(out);

	EXPECT_NEAR(out(0), 0.0f, 1e-5f);
	EXPECT_NEAR(out(1), 1.0f, 1e-5f);
	EXPECT_NEAR(out(2), 4.0f, 1e-5f);
	EXPECT_NEAR(out(3), 9.0f, 1e-5f);
}
