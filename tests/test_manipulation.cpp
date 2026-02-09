#include "numhalide_all.h"
#include <gtest/gtest.h>

using namespace numhalide;

TEST(Manipulation, Concat1D) {
	// a = [0, 1, 2]
	Halide::Func a = arange(Halide::Float(32), 3);
	shape_t s_a = { 3 };
	
	// b = [10, 11]
	Halide::Func b = arange(Halide::Float(32), 10, 12);
	shape_t s_b = { 2 };
	
	// concat(a, b, 0) -> [0, 1, 2, 10, 11]
	Halide::Func c = concat(a, s_a, b, s_b, 0);
	
	Halide::Runtime::Buffer<float> result(5);
	c.realize(result);
	
	EXPECT_NEAR(result(0), 0.0f, 1e-5f);
	EXPECT_NEAR(result(1), 1.0f, 1e-5f);
	EXPECT_NEAR(result(2), 2.0f, 1e-5f);
	EXPECT_NEAR(result(3), 10.0f, 1e-5f);
	EXPECT_NEAR(result(4), 11.0f, 1e-5f);
}

TEST(Manipulation, VStack) {
	// a = [1, 1, 1]
	shape_t s_a = { 3 };
	Halide::Func a = ones(Halide::Float(32), s_a, "ones_a");
	
	// b = [0, 0, 0]
	shape_t s_b = { 3 };
	Halide::Func b = zeros(Halide::Float(32), s_b, "zeros_b");
	
	// vstack(a, b) -> [[1, 1, 1], [0, 0, 0]] (2x3)
	Halide::Func c = vstack(a, s_a, b, s_b);
	
	Halide::Runtime::Buffer<float> result(3, 2);
	c.realize(result);
	
	// Row 0 (a)
	EXPECT_NEAR(result(0, 0), 1.0f, 1e-5f);
	EXPECT_NEAR(result(1, 0), 1.0f, 1e-5f);
	EXPECT_NEAR(result(2, 0), 1.0f, 1e-5f);
	
	// Row 1 (b)
	EXPECT_NEAR(result(0, 1), 0.0f, 1e-5f);
	EXPECT_NEAR(result(1, 1), 0.0f, 1e-5f);
	EXPECT_NEAR(result(2, 1), 0.0f, 1e-5f);
}

TEST(Manipulation, HStack) {
	// a = [[1], [1]] (2x1)
	// Use explicit shape_t to avoid ambiguous overload
	shape_t s_a = { 2, 1 };
	Halide::Func a = ones(Halide::Float(32), s_a);
	
	// b = [[0], [0]] (2x1)
	shape_t s_b = { 2, 1 };
	Halide::Func b = zeros(Halide::Float(32), s_b);
	
	// hstack(a, b) -> [[1, 0], [1, 0]] (2x2)
	Halide::Func c = hstack(a, s_a, b, s_b);
	
	Halide::Runtime::Buffer<float> result(2, 2);
	c.realize(result);
	
	EXPECT_NEAR(result(0, 0), 1.0f, 1e-5f);
	EXPECT_NEAR(result(1, 0), 1.0f, 1e-5f);
	EXPECT_NEAR(result(0, 1), 0.0f, 1e-5f);
	EXPECT_NEAR(result(1, 1), 0.0f, 1e-5f);
}
