/// @file test_morphology.cpp
/// @brief Tests for morphological operations

#include <gtest/gtest.h>
#include "numhalide_all.h"

using namespace numhalide;

TEST(Morph, DilateMax) {
	shape_t s = { 4, 4 };
	Halide::Func input("input");
	Halide::Var x, y;
	input(x, y) = 5.0f;  // uniform

	Halide::Func result = dilate(input, s, 3);

	Halide::Runtime::Buffer<float> out(4, 4);
	result.realize(out);

	for (int j = 0; j < 4; ++j)
		for (int i = 0; i < 4; ++i)
			EXPECT_NEAR(out(i, j), 5.0f, 1e-5f);
}

TEST(Morph, ErodeMin) {
	shape_t s = { 4, 4 };
	Halide::Func input("input");
	Halide::Var x, y;
	input(x, y) = 5.0f;  // uniform

	Halide::Func result = erode(input, s, 3);

	Halide::Runtime::Buffer<float> out(4, 4);
	result.realize(out);

	for (int j = 0; j < 4; ++j)
		for (int i = 0; i < 4; ++i)
			EXPECT_NEAR(out(i, j), 5.0f, 1e-5f);
}

TEST(Morph, DilateExpands) {
	shape_t s = { 8, 8 };
	Halide::Func input("input");
	Halide::Var x, y;
	// Single bright pixel at center
	input(x, y) = Halide::select(x == 4 && y == 4, 1.0f, 0.0f);

	Halide::Func result = dilate(input, s, 3);

	Halide::Runtime::Buffer<float> out(8, 8);
	result.realize(out);

	// Center and neighbors should be 1
	EXPECT_NEAR(out(4, 4), 1.0f, 1e-5f);
	EXPECT_NEAR(out(3, 4), 1.0f, 1e-5f);
	EXPECT_NEAR(out(5, 4), 1.0f, 1e-5f);
	EXPECT_NEAR(out(4, 3), 1.0f, 1e-5f);
	EXPECT_NEAR(out(4, 5), 1.0f, 1e-5f);
	// Far away should still be 0
	EXPECT_NEAR(out(0, 0), 0.0f, 1e-5f);
}

TEST(Morph, ErodeContracts) {
	shape_t s = { 8, 8 };
	Halide::Func input("input");
	Halide::Var x, y;
	// Mostly bright, single dark pixel
	input(x, y) = Halide::select(x == 4 && y == 4, 0.0f, 1.0f);

	Halide::Func result = erode(input, s, 3);

	Halide::Runtime::Buffer<float> out(8, 8);
	result.realize(out);

	// Dark region should expand
	EXPECT_NEAR(out(4, 4), 0.0f, 1e-5f);
	EXPECT_NEAR(out(3, 4), 0.0f, 1e-5f);
	EXPECT_NEAR(out(5, 4), 0.0f, 1e-5f);
	// Far corner should still be bright
	EXPECT_NEAR(out(0, 0), 1.0f, 1e-5f);
}

TEST(Morph, GradientEdges) {
	shape_t s = { 16, 16 };
	Halide::Func input("input");
	Halide::Var x, y;
	// White square in center
	input(x, y) = Halide::select(x >= 4 && x < 12 && y >= 4 && y < 12, 1.0f, 0.0f);

	Halide::Func result = morph_gradient(input, s, 3);

	Halide::Runtime::Buffer<float> out(16, 16);
	result.realize(out);

	// Interior should be ~0 (dilate-erode cancels)
	EXPECT_NEAR(out(8, 8), 0.0f, 1e-5f);
	// Edge should be >0
	EXPECT_GT(out(4, 8), 0.0f);
}

TEST(Morph, OpenIdempotent) {
	shape_t s = { 8, 8 };
	Halide::Func input("input");
	Halide::Var x, y;
	input(x, y) = Halide::select(x >= 2 && x < 6 && y >= 2 && y < 6, 1.0f, 0.0f);

	Halide::Func opened = morph_open(input, s, 3);
	opened.compute_root();
	Halide::Func opened2 = morph_open(opened, s, 3);

	Halide::Runtime::Buffer<float> out1(8, 8);
	Halide::Runtime::Buffer<float> out2(8, 8);
	opened.realize(out1);
	opened2.realize(out2);

	for (int j = 0; j < 8; ++j)
		for (int i = 0; i < 8; ++i)
			EXPECT_NEAR(out1(i, j), out2(i, j), 1e-5f);
}
