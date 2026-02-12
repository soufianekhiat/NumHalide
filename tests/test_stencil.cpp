/// @file test_stencil.cpp
/// @brief Tests for stencil and finite difference operations

#include <gtest/gtest.h>
#include "numhalide_all.h"
#include <cmath>

using namespace numhalide;

TEST(Stencil, JacobiConstant) {
	shape_t s = { 8, 8 };
	Halide::Func input("input");
	Halide::Var x, y;
	input(x, y) = 5.0f;  // constant

	Halide::Func result = jacobi_step(input, s);

	Halide::Runtime::Buffer<float> out(8, 8);
	result.realize(out);

	for (int j = 0; j < 8; ++j)
		for (int i = 0; i < 8; ++i)
			EXPECT_NEAR(out(i, j), 5.0f, 1e-5f);
}

TEST(Stencil, JacobiConverges) {
	shape_t s = { 8, 8 };
	Halide::Func input("input");
	Halide::Var x, y;
	// Hot spot in center
	input(x, y) = Halide::select(x == 4 && y == 4, 100.0f, 0.0f);
	input.compute_root();

	// Run multiple iterations
	Halide::Func cur = input;
	for (int iter = 0; iter < 10; ++iter) {
		cur = jacobi_step(cur, s);
		cur.compute_root();
	}

	Halide::Runtime::Buffer<float> out(8, 8);
	cur.realize(out);

	// After iterations, heat should have spread - center should be less than 100
	EXPECT_LT(out(4, 4), 100.0f);
	// Neighbors should be > 0
	EXPECT_GT(out(3, 4), 0.0f);
	EXPECT_GT(out(5, 4), 0.0f);
}

TEST(Stencil, HeatEquilibrium) {
	shape_t s = { 8, 8 };
	Halide::Func input("input");
	Halide::Var x, y;
	input(x, y) = 10.0f;  // uniform temperature

	Halide::Func result = heat_diffusion_step(input, s, 0.1f, 1.0f);

	Halide::Runtime::Buffer<float> out(8, 8);
	result.realize(out);

	// Uniform -> no change (laplacian = 0)
	for (int j = 0; j < 8; ++j)
		for (int i = 0; i < 8; ++i)
			EXPECT_NEAR(out(i, j), 10.0f, 1e-4f);
}

TEST(Stencil, HeatDiffusion) {
	shape_t s = { 16, 16 };
	Halide::Func input("input");
	Halide::Var x, y;
	input(x, y) = Halide::select(x == 8 && y == 8, 100.0f, 0.0f);

	Halide::Func result = heat_diffusion_step(input, s, 0.1f, 1.0f);

	Halide::Runtime::Buffer<float> out(16, 16);
	result.realize(out);

	// Center should decrease (heat flowing out)
	EXPECT_LT(out(8, 8), 100.0f);
	// Neighbors should be > 0 (heat flowing in)
	EXPECT_GT(out(7, 8), 0.0f);
	EXPECT_GT(out(9, 8), 0.0f);
}
