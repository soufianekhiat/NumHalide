#include "numhalide_all.h"
#include <gtest/gtest.h>

using namespace numhalide;

TEST(Broadcast, BroadcastTo) {
	// a = [1, 2, 3]
	shape_t s_a = { 3 };
	Halide::Func a = arange(Halide::Float(32), 1, 4);
	
	// Target: [2, 3] -> [[1, 2, 3], [1, 2, 3]]
	shape_t s_target = { 2, 3 };
	Halide::Func b = broadcast_to(a, s_a, s_target);
	
	Halide::Runtime::Buffer<float> result(3, 2); // Halide Buffer is (x, y) -> (3, 2)
	b.realize(result);
	
	EXPECT_NEAR(result(0, 0), 1.0f, 1e-5f);
	EXPECT_NEAR(result(1, 0), 2.0f, 1e-5f);
	EXPECT_NEAR(result(2, 0), 3.0f, 1e-5f);
	EXPECT_NEAR(result(0, 1), 1.0f, 1e-5f);
	EXPECT_NEAR(result(1, 1), 2.0f, 1e-5f);
	EXPECT_NEAR(result(2, 1), 3.0f, 1e-5f);
}

TEST(Broadcast, AddScalar) {
	// a = [[1, 1], [1, 1]]
	shape_t s_a = { 2, 2 };
	Halide::Func a = ones(Halide::Float(32), s_a);
	
	// b = 2.0 (scalar, rank 0)
	shape_t s_b = { }; // rank 0
	Halide::Func b = full(Halide::Float(32), 2.0f, s_b);
	
	// c = a + b -> [[3, 3], [3, 3]]
	Halide::Func c = add(a, s_a, b, s_b);
	
	Halide::Runtime::Buffer<float> result(2, 2);
	c.realize(result);
	
	EXPECT_NEAR(result(0, 0), 3.0f, 1e-5f);
	EXPECT_NEAR(result(1, 1), 3.0f, 1e-5f);
}

TEST(Broadcast, Add1DTo2D) {
	// a = [[0, 0], [10, 10]] (2x2)
	// Let's construct this manually or use vstack?
	// Let's use full and concat for setup or just trust arange + reshape?
	// We don't have reshape yet (well, we do but it's old).
	// Let's use broadcast_to to make 'a'.
	// a_row = [0, 10] (2x1)? No.
	// a_col = [[0], [10]] (2x1)
	shape_t s_col = { 2, 1 };
	Halide::Func col = arange(Halide::Float(32), 0, 20, 10); // 0, 10. Shape {2} -> reshape to {2, 1}
	// Wait, arange returns 1D.
	// Let's just use explicit fulls and vstack.
	shape_t s_1 = { 2 };
	Halide::Func row0 = zeros(Halide::Float(32), s_1); // [0, 0]
	Halide::Func row1 = full(Halide::Float(32), 10.0f, s_1); // [10, 10]
	Halide::Func a = vstack(row0, s_1, row1, s_1);
	shape_t s_a = { 2, 2 };
	
	// b = [1, 2] (1D)
	shape_t s_b = { 2 };
	Halide::Func b = arange(Halide::Float(32), 1, 3);
	
	// c = a + b
	// a: (2, 2)
	// b: (2) -> broadcasts to (1, 2) -> (2, 2)
	// [[0, 0], [10, 10]] + [[1, 2], [1, 2]] = [[1, 2], [11, 12]]
	
	Halide::Func c = add(a, s_a, b, s_b);
	
	Halide::Runtime::Buffer<float> result(2, 2);
	c.realize(result);
	
	EXPECT_NEAR(result(0, 0), 1.0f, 1e-5f); // row 0, col 0
	EXPECT_NEAR(result(1, 0), 2.0f, 1e-5f); // row 0, col 1
	EXPECT_NEAR(result(0, 1), 11.0f, 1e-5f); // row 1, col 0
	EXPECT_NEAR(result(1, 1), 12.0f, 1e-5f); // row 1, col 1
}

TEST(Broadcast, Negative) {
	// a = [1, -2]
	shape_t s_a = { 2 };
	Halide::Func a = arange(Halide::Float(32), 1, 3); // 1, 2
	// Wait, I want -2.
	// Let's use full.
	Halide::Func neg_two = full(Halide::Float(32), -2.0f, {1});
	Halide::Func one = full(Halide::Float(32), 1.0f, {1});
	Halide::Func mixed = concat(one, {1}, neg_two, {1}, 0);
	
	Halide::Func c = negative(mixed, s_a);
	
	Halide::Runtime::Buffer<float> result(2);
	c.realize(result);
	
	EXPECT_NEAR(result(0), -1.0f, 1e-5f);
	EXPECT_NEAR(result(1), 2.0f, 1e-5f);
}
