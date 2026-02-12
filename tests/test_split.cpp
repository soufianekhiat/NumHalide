/// @file test_split.cpp
/// @brief Tests for array splitting operations

#include <gtest/gtest.h>
#include "numhalide_all.h"

using namespace numhalide;

TEST(Split, EqualParts1D) {
	shape_t s = { 8 };
	Halide::Func input("input");
	Halide::Var x;
	input(x) = Halide::cast<float>(x);  // 0..7

	auto parts = split(input, s, 0, 4);
	ASSERT_EQ(parts.size(), 4u);

	for (int p = 0; p < 4; ++p) {
		Halide::Runtime::Buffer<float> out(2);
		parts[p].realize(out);
		EXPECT_NEAR(out(0), static_cast<float>(p * 2), 1e-5f);
		EXPECT_NEAR(out(1), static_cast<float>(p * 2 + 1), 1e-5f);
	}
}

TEST(Split, EqualParts2DAxis0) {
	shape_t s = { 6, 4 };
	Halide::Func input("input");
	Halide::Var x, y;
	input(x, y) = Halide::cast<float>(y * 10 + x);

	auto parts = split(input, s, 0, 3);  // Split 6 rows into 3 parts of 2
	ASSERT_EQ(parts.size(), 3u);

	// Check first element of each part
	Halide::Runtime::Buffer<float> out0(4, 2);
	parts[0].realize(out0);
	EXPECT_NEAR(out0(0, 0), 0.0f, 1e-5f);   // row 0, col 0

	Halide::Runtime::Buffer<float> out1(4, 2);
	parts[1].realize(out1);
	EXPECT_NEAR(out1(0, 0), 20.0f, 1e-5f);  // row 2, col 0

	Halide::Runtime::Buffer<float> out2(4, 2);
	parts[2].realize(out2);
	EXPECT_NEAR(out2(0, 0), 40.0f, 1e-5f);  // row 4, col 0
}

TEST(Split, EqualParts2DAxis1) {
	shape_t s = { 4, 6 };
	Halide::Func input("input");
	Halide::Var x, y;
	input(x, y) = Halide::cast<float>(y * 10 + x);

	auto parts = split(input, s, 1, 2);  // Split 6 cols into 2 parts of 3
	ASSERT_EQ(parts.size(), 2u);

	Halide::Runtime::Buffer<float> out0(3, 4);
	parts[0].realize(out0);
	EXPECT_NEAR(out0(0, 0), 0.0f, 1e-5f);

	Halide::Runtime::Buffer<float> out1(3, 4);
	parts[1].realize(out1);
	EXPECT_NEAR(out1(0, 0), 3.0f, 1e-5f);  // col 3
}

TEST(Split, HSplit) {
	shape_t s = { 4, 6 };
	Halide::Func input("input");
	Halide::Var x, y;
	input(x, y) = Halide::cast<float>(x);

	auto parts = hsplit(input, s, 3);  // Split 6 cols into 3 parts of 2
	ASSERT_EQ(parts.size(), 3u);

	Halide::Runtime::Buffer<float> out(2, 4);
	parts[1].realize(out);
	EXPECT_NEAR(out(0, 0), 2.0f, 1e-5f);  // col 2 of original
}

TEST(Split, VSplit) {
	shape_t s = { 6, 4 };
	Halide::Func input("input");
	Halide::Var x, y;
	input(x, y) = Halide::cast<float>(y);

	auto parts = vsplit(input, s, 2);  // Split 6 rows into 2 parts of 3
	ASSERT_EQ(parts.size(), 2u);

	Halide::Runtime::Buffer<float> out(4, 3);
	parts[1].realize(out);
	EXPECT_NEAR(out(0, 0), 3.0f, 1e-5f);  // row 3 of original
}

TEST(Split, SingleSection) {
	shape_t s = { 4 };
	Halide::Func input("input");
	Halide::Var x;
	input(x) = Halide::cast<float>(x);

	auto parts = split(input, s, 0, 1);
	ASSERT_EQ(parts.size(), 1u);

	Halide::Runtime::Buffer<float> out(4);
	parts[0].realize(out);
	for (int i = 0; i < 4; ++i) {
		EXPECT_NEAR(out(i), static_cast<float>(i), 1e-5f);
	}
}
