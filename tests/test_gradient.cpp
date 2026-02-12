/// @file test_gradient.cpp
/// @brief Tests for gradient and differential operators

#include <gtest/gtest.h>
#include "numhalide_all.h"
#include <cmath>

using namespace numhalide;

TEST(Gradient, Linear1D) {
	shape_t s = { 8 };
	Halide::Func input("input");
	Halide::Var x;
	input(x) = Halide::cast<float>(x);  // 0, 1, 2, ..., 7

	Halide::Func result = gradient_1d(input, s, 0);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	result.realize(out);

	// Central difference of linear = 1 everywhere (interior)
	for (int i = 1; i < 7; ++i) {
		EXPECT_NEAR(out(i), 1.0f, 1e-5f);
	}
}

TEST(Gradient, Constant2D) {
	shape_t s = { 8, 8 };
	Halide::Func input("input");
	Halide::Var x, y;
	input(x, y) = 5.0f;

	auto [gx, gy] = gradient_2d(input, s);

	Halide::Runtime::Buffer<float> gx_out(8, 8);
	Halide::Runtime::Buffer<float> gy_out(8, 8);
	gx.realize(gx_out);
	gy.realize(gy_out);

	for (int j = 0; j < 8; ++j) {
		for (int i = 0; i < 8; ++i) {
			EXPECT_NEAR(gx_out(i, j), 0.0f, 1e-5f);
			EXPECT_NEAR(gy_out(i, j), 0.0f, 1e-5f);
		}
	}
}

TEST(Gradient, LinearX2D) {
	shape_t s = { 8, 8 };
	Halide::Func input("input");
	Halide::Var x, y;
	input(x, y) = Halide::cast<float>(x);  // f(x,y) = x

	auto [gx, gy] = gradient_2d(input, s);

	Halide::Runtime::Buffer<float> gx_out(8, 8);
	Halide::Runtime::Buffer<float> gy_out(8, 8);
	gx.realize(gx_out);
	gy.realize(gy_out);

	// grad_x = 1 (interior), grad_y = 0
	for (int j = 0; j < 8; ++j) {
		for (int i = 1; i < 7; ++i) {
			EXPECT_NEAR(gx_out(i, j), 1.0f, 1e-5f);
		}
		for (int i = 0; i < 8; ++i) {
			EXPECT_NEAR(gy_out(i, j), 0.0f, 1e-5f);
		}
	}
}

TEST(Gradient, Laplacian) {
	shape_t s = { 16, 16 };
	Halide::Func input("input");
	Halide::Var x, y;
	// f(x,y) = x^2 + y^2 -> laplacian = 2 + 2 = 4
	Halide::Expr fx = Halide::cast<float>(x);
	Halide::Expr fy = Halide::cast<float>(y);
	input(x, y) = fx * fx + fy * fy;

	Halide::Func result = laplacian(input, s);

	Halide::Runtime::Buffer<float> out(16, 16);
	result.realize(out);

	// Interior points should be ~4 (discrete Laplacian of x^2 = 2 exactly)
	for (int j = 2; j < 14; ++j) {
		for (int i = 2; i < 14; ++i) {
			EXPECT_NEAR(out(i, j), 4.0f, 1e-3f);
		}
	}
}

TEST(Gradient, Divergence) {
	shape_t s = { 16, 16 };
	Halide::Func fx("fx"), fy("fy");
	Halide::Var x, y;
	// F = (x, y) -> div(F) = 1 + 1 = 2
	fx(x, y) = Halide::cast<float>(x);
	fy(x, y) = Halide::cast<float>(y);

	Halide::Func result = divergence(fx, fy, s);

	Halide::Runtime::Buffer<float> out(16, 16);
	result.realize(out);

	// Interior: should be ~2
	for (int j = 2; j < 14; ++j) {
		for (int i = 2; i < 14; ++i) {
			EXPECT_NEAR(out(i, j), 2.0f, 1e-3f);
		}
	}
}
