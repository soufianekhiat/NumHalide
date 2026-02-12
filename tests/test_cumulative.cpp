/// @file test_cumulative.cpp
/// @brief Tests for cumulative operations: cumsum, cumprod, diff

#include <gtest/gtest.h>
#include "numhalide_all.h"
#include <cmath>

using namespace numhalide;

TEST(Cumulative, CumSum1D) {
	shape_t s = { 4 };
	Halide::Func input("input");
	Halide::Var x;
	input(x) = Halide::cast<float>(x + 1);  // 1, 2, 3, 4

	Halide::Func result = cumsum(input, s);
	result.compute_root();

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	result.realize(out);

	EXPECT_NEAR(out(0), 1.0f, 1e-5f);
	EXPECT_NEAR(out(1), 3.0f, 1e-5f);
	EXPECT_NEAR(out(2), 6.0f, 1e-5f);
	EXPECT_NEAR(out(3), 10.0f, 1e-5f);
}

TEST(Cumulative, CumProd1D) {
	shape_t s = { 4 };
	Halide::Func input("input");
	Halide::Var x;
	input(x) = Halide::cast<float>(x + 1);  // 1, 2, 3, 4

	Halide::Func result = cumprod(input, s);
	result.compute_root();

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	result.realize(out);

	EXPECT_NEAR(out(0), 1.0f, 1e-5f);
	EXPECT_NEAR(out(1), 2.0f, 1e-5f);
	EXPECT_NEAR(out(2), 6.0f, 1e-5f);
	EXPECT_NEAR(out(3), 24.0f, 1e-5f);
}

TEST(Cumulative, Diff1D) {
	shape_t s = { 5 };
	Halide::Func input("input");
	Halide::Var x;
	// Values: 1, 4, 2, 7, 3
	input(x) = Halide::select(x == 0, 1.0f, x == 1, 4.0f, x == 2, 2.0f, x == 3, 7.0f, 3.0f);

	Halide::Func result = diff(input, s);

	Halide::Runtime::Buffer<float> out(s.extents[0] - 1);
	result.realize(out);

	EXPECT_NEAR(out(0), 3.0f, 1e-5f);   // 4-1
	EXPECT_NEAR(out(1), -2.0f, 1e-5f);  // 2-4
	EXPECT_NEAR(out(2), 5.0f, 1e-5f);   // 7-2
	EXPECT_NEAR(out(3), -4.0f, 1e-5f);  // 3-7
}

TEST(Cumulative, CumSumConstant) {
	shape_t s = { 5 };
	Halide::Func input("input");
	Halide::Var x;
	input(x) = 3.0f;

	Halide::Func result = cumsum(input, s);
	result.compute_root();

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	result.realize(out);

	EXPECT_NEAR(out(0), 3.0f, 1e-5f);
	EXPECT_NEAR(out(1), 6.0f, 1e-5f);
	EXPECT_NEAR(out(2), 9.0f, 1e-5f);
	EXPECT_NEAR(out(3), 12.0f, 1e-5f);
	EXPECT_NEAR(out(4), 15.0f, 1e-5f);
}
