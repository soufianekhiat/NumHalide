#include "numhalide_all.h"
#include <gtest/gtest.h>

using namespace numhalide;

TEST(Reduce, Sum1D) {
	// a = [1, 2, 3, 4, 5]
	shape_t s_a = { 5 };
	Halide::Func a = arange(Halide::Float(32), 1, 6);

	// sum(a) = 15
	Halide::Func result = reduce_sum(a, s_a);

	Halide::Runtime::Buffer<float> out(1);
	result.realize(out);

	EXPECT_NEAR(out(0), 15.0f, 1e-5f);
}

TEST(Reduce, Sum2DAxis0) {
	// a = [[1, 2, 3],
	//      [4, 5, 6]]  (2x3)
	// sum(a, axis=0) = [5, 7, 9]
	shape_t s_a = { 2, 3 };
	Halide::Func a("input");
	Halide::Var x, y;
	a(x, y) = Halide::cast<float>(1 + x + y * 3);

	Halide::Func result = reduce_sum(a, s_a, {0}, false);

	Halide::Runtime::Buffer<float> out(3);
	result.realize(out);

	EXPECT_NEAR(out(0), 5.0f, 1e-5f);   // 1 + 4
	EXPECT_NEAR(out(1), 7.0f, 1e-5f);   // 2 + 5
	EXPECT_NEAR(out(2), 9.0f, 1e-5f);   // 3 + 6
}

TEST(Reduce, Sum2DAxis1) {
	// a = [[1, 2, 3],
	//      [4, 5, 6]]  (2x3)
	// sum(a, axis=1) = [6, 15]
	shape_t s_a = { 2, 3 };
	Halide::Func a("input");
	Halide::Var x, y;
	a(x, y) = Halide::cast<float>(1 + x + y * 3);

	Halide::Func result = reduce_sum(a, s_a, {1}, false);

	Halide::Runtime::Buffer<float> out(2);
	result.realize(out);

	EXPECT_NEAR(out(0), 6.0f, 1e-5f);   // 1 + 2 + 3
	EXPECT_NEAR(out(1), 15.0f, 1e-5f);  // 4 + 5 + 6
}

TEST(Reduce, SumKeepDims) {
	// a = [[1, 2, 3],
	//      [4, 5, 6]]  (2x3)
	// sum(a, axis=1, keepdims=True) = [[6], [15]]  (2x1)
	shape_t s_a = { 2, 3 };
	Halide::Func a("input");
	Halide::Var x, y;
	a(x, y) = Halide::cast<float>(1 + x + y * 3);

	Halide::Func result = reduce_sum(a, s_a, {1}, true);

	Halide::Runtime::Buffer<float> out(1, 2);
	result.realize(out);

	EXPECT_NEAR(out(0, 0), 6.0f, 1e-5f);
	EXPECT_NEAR(out(0, 1), 15.0f, 1e-5f);
}

TEST(Reduce, Mean2D) {
	// a = [[1, 2, 3],
	//      [4, 5, 6]]  (2x3)
	// mean(a, axis=1) = [2, 5]
	shape_t s_a = { 2, 3 };
	Halide::Func a("input");
	Halide::Var x, y;
	a(x, y) = Halide::cast<float>(1 + x + y * 3);

	Halide::Func result = reduce_mean(a, s_a, {1}, false);

	Halide::Runtime::Buffer<float> out(2);
	result.realize(out);

	EXPECT_NEAR(out(0), 2.0f, 1e-5f);   // (1+2+3)/3
	EXPECT_NEAR(out(1), 5.0f, 1e-5f);   // (4+5+6)/3
}

TEST(Reduce, Min2D) {
	// a = [[3, 1, 2],
	//      [6, 4, 5]]  (2x3)
	// min(a, axis=1) = [1, 4]
	shape_t s_a = { 2, 3 };
	Halide::Func a("input");
	Halide::Var x, y;
	// Row 0: 3, 1, 2
	// Row 1: 6, 4, 5
	a(x, y) = Halide::cast<float>(Halide::select(
		y == 0,
		Halide::select(x == 0, 3, Halide::select(x == 1, 1, 2)),
		Halide::select(x == 0, 6, Halide::select(x == 1, 4, 5))
	));

	Halide::Func result = reduce_min(a, s_a, {1}, false);

	Halide::Runtime::Buffer<float> out(2);
	result.realize(out);

	EXPECT_NEAR(out(0), 1.0f, 1e-5f);
	EXPECT_NEAR(out(1), 4.0f, 1e-5f);
}

TEST(Reduce, Max2D) {
	// a = [[3, 1, 2],
	//      [6, 4, 5]]  (2x3)
	// max(a, axis=1) = [3, 6]
	shape_t s_a = { 2, 3 };
	Halide::Func a("input");
	Halide::Var x, y;
	a(x, y) = Halide::cast<float>(Halide::select(
		y == 0,
		Halide::select(x == 0, 3, Halide::select(x == 1, 1, 2)),
		Halide::select(x == 0, 6, Halide::select(x == 1, 4, 5))
	));

	Halide::Func result = reduce_max(a, s_a, {1}, false);

	Halide::Runtime::Buffer<float> out(2);
	result.realize(out);

	EXPECT_NEAR(out(0), 3.0f, 1e-5f);
	EXPECT_NEAR(out(1), 6.0f, 1e-5f);
}

TEST(Reduce, Prod1D) {
	// a = [1, 2, 3, 4]
	// prod(a) = 24
	shape_t s_a = { 4 };
	Halide::Func a = arange(Halide::Float(32), 1, 5);

	Halide::Func result = reduce_prod(a, s_a);

	Halide::Runtime::Buffer<float> out(1);
	result.realize(out);

	EXPECT_NEAR(out(0), 24.0f, 1e-5f);
}

TEST(Reduce, FullReduction) {
	// a = [[1, 2], [3, 4]]  (2x2)
	// sum(a) = 10
	shape_t s_a = { 2, 2 };
	Halide::Func a("input");
	Halide::Var x, y;
	a(x, y) = Halide::cast<float>(1 + x + y * 2);

	Halide::Func result = reduce_sum(a, s_a);

	Halide::Runtime::Buffer<float> out(1);
	result.realize(out);

	EXPECT_NEAR(out(0), 10.0f, 1e-5f);  // 1+2+3+4
}
