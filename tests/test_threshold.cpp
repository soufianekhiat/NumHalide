/// @file test_threshold.cpp
/// @brief Tests for thresholding operations

#include <gtest/gtest.h>
#include "numhalide_all.h"

using namespace numhalide;

TEST(Threshold, BinaryAbove) {
	shape_t s = { 4 };
	Halide::Func input("input");
	Halide::Var x;
	input(x) = Halide::select(x < 2, 0.3f, 0.7f);

	Halide::Func result = threshold_binary(input, s, 0.5f);

	Halide::Runtime::Buffer<float> out(4);
	result.realize(out);

	EXPECT_NEAR(out(0), 0.0f, 1e-5f);
	EXPECT_NEAR(out(1), 0.0f, 1e-5f);
	EXPECT_NEAR(out(2), 1.0f, 1e-5f);
	EXPECT_NEAR(out(3), 1.0f, 1e-5f);
}

TEST(Threshold, Truncate) {
	shape_t s = { 4 };
	Halide::Func input("input");
	Halide::Var x;
	input(x) = Halide::cast<float>(x) * 0.3f;  // 0.0, 0.3, 0.6, 0.9

	Halide::Func result = threshold_trunc(input, s, 0.5f);

	Halide::Runtime::Buffer<float> out(4);
	result.realize(out);

	EXPECT_NEAR(out(0), 0.0f, 1e-5f);
	EXPECT_NEAR(out(1), 0.3f, 1e-5f);
	EXPECT_NEAR(out(2), 0.5f, 1e-5f);  // clamped
	EXPECT_NEAR(out(3), 0.5f, 1e-5f);  // clamped
}

TEST(Threshold, ToZero) {
	shape_t s = { 4 };
	Halide::Func input("input");
	Halide::Var x;
	input(x) = Halide::cast<float>(x) * 0.3f;  // 0.0, 0.3, 0.6, 0.9

	Halide::Func result = threshold_tozero(input, s, 0.5f);

	Halide::Runtime::Buffer<float> out(4);
	result.realize(out);

	EXPECT_NEAR(out(0), 0.0f, 1e-5f);
	EXPECT_NEAR(out(1), 0.0f, 1e-5f);
	EXPECT_NEAR(out(2), 0.6f, 1e-5f);
	EXPECT_NEAR(out(3), 0.9f, 1e-5f);
}

TEST(Threshold, BinaryAllAbove) {
	shape_t s = { 4 };
	Halide::Func input("input");
	Halide::Var x;
	input(x) = 1.0f;

	Halide::Func result = threshold_binary(input, s, 0.5f);

	Halide::Runtime::Buffer<float> out(4);
	result.realize(out);

	for (int i = 0; i < 4; ++i)
		EXPECT_NEAR(out(i), 1.0f, 1e-5f);
}

TEST(Threshold, BinaryAllBelow) {
	shape_t s = { 4 };
	Halide::Func input("input");
	Halide::Var x;
	input(x) = 0.0f;

	Halide::Func result = threshold_binary(input, s, 0.5f);

	Halide::Runtime::Buffer<float> out(4);
	result.realize(out);

	for (int i = 0; i < 4; ++i)
		EXPECT_NEAR(out(i), 0.0f, 1e-5f);
}
