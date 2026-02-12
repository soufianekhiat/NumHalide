/// @file test_stats_ext.cpp
/// @brief Tests for extended statistics: ptp, average, histogram, digitize

#include <gtest/gtest.h>
#include "numhalide_all.h"
#include <cmath>

using namespace numhalide;

TEST(StatsExt, Ptp1D) {
	shape_t s = { 4 };
	Halide::Func input("input");
	Halide::Var x;
	input(x) = Halide::select(x == 0, 3.0f, x == 1, 1.0f, x == 2, 7.0f, 2.0f);

	Halide::Func result = stats::ptp(input, s);

	Halide::Runtime::Buffer<float> out(1);
	result.realize(out);

	EXPECT_NEAR(out(0), 6.0f, 1e-5f);  // 7 - 1
}

TEST(StatsExt, PtpConstant) {
	shape_t s = { 4 };
	Halide::Func input("input");
	Halide::Var x;
	input(x) = 5.0f;

	Halide::Func result = stats::ptp(input, s);

	Halide::Runtime::Buffer<float> out(1);
	result.realize(out);

	EXPECT_NEAR(out(0), 0.0f, 1e-5f);
}

TEST(StatsExt, Average) {
	shape_t s = { 3 };
	Halide::Func input("input"), weights("weights");
	Halide::Var x;
	input(x) = Halide::cast<float>(x + 1);    // 1, 2, 3
	weights(x) = Halide::select(x == 0, 3.0f, x == 1, 2.0f, 1.0f);  // 3, 2, 1

	Halide::Func result = stats::average(input, weights, s);

	Halide::Runtime::Buffer<float> out(1);
	result.realize(out);

	// (1*3 + 2*2 + 3*1) / (3+2+1) = 10/6 = 1.6667
	EXPECT_NEAR(out(0), 10.0f / 6.0f, 1e-4f);
}

TEST(StatsExt, AverageUniform) {
	shape_t s = { 4 };
	Halide::Func input("input"), weights("weights");
	Halide::Var x;
	input(x) = Halide::cast<float>(x + 1);  // 1, 2, 3, 4
	weights(x) = 1.0f;

	Halide::Func result = stats::average(input, weights, s);

	Halide::Runtime::Buffer<float> out(1);
	result.realize(out);

	EXPECT_NEAR(out(0), 2.5f, 1e-4f);  // (1+2+3+4)/4
}

TEST(StatsExt, Histogram) {
	shape_t s = { 8 };
	Halide::Func input("input");
	Halide::Var x;
	// Values: 0.1, 0.1, 0.3, 0.5, 0.5, 0.5, 0.9, 0.9
	input(x) = Halide::select(
		x < 2, 0.1f,
		x == 2, 0.3f,
		x < 6, 0.5f,
		0.9f
	);

	Halide::Func result = histogram_1d(input, s, 4, 0.0f, 1.0f);
	result.compute_root();

	Halide::Runtime::Buffer<int32_t> out(4);
	result.realize(out);

	// Bin 0: [0, 0.25) -> 0.1, 0.1 = 2
	// Bin 1: [0.25, 0.5) -> 0.3 = 1
	// Bin 2: [0.5, 0.75) -> 0.5, 0.5, 0.5 = 3
	// Bin 3: [0.75, 1.0) -> 0.9, 0.9 = 2
	EXPECT_EQ(out(0), 2);
	EXPECT_EQ(out(1), 1);
	EXPECT_EQ(out(2), 3);
	EXPECT_EQ(out(3), 2);
}

TEST(StatsExt, Digitize) {
	shape_t s = { 4 };
	Halide::Func input("input");
	Halide::Func bins("bins");
	Halide::Var x;
	input(x) = Halide::select(x == 0, 0.5f, x == 1, 1.5f, x == 2, 3.5f, 4.5f);
	// Bin edges: 0, 1, 2, 3, 4
	bins(x) = Halide::cast<float>(x);

	Halide::Func result = stats::digitize(input, bins, s, 5);

	Halide::Runtime::Buffer<int32_t> out(s.extents[0]);
	result.realize(out);

	// 0.5 -> bin 1 (between 0 and 1)
	// 1.5 -> bin 2 (between 1 and 2)
	// 3.5 -> bin 4 (between 3 and 4)
	// 4.5 -> bin 5 (past last edge, clamped to n_bins)
	EXPECT_EQ(out(0), 1);
	EXPECT_EQ(out(1), 2);
	EXPECT_EQ(out(2), 4);
	EXPECT_EQ(out(3), 5);
}
