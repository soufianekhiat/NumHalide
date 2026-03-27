/// @file test_basic_math.cpp
/// @brief Tests for basic math wrappers: trig, exp/log, sqrt, abs, rounding, power

#include <gtest/gtest.h>
#include "numhalide_all.h"
#include <cmath>

using namespace numhalide;

// -----------------------------------------------------------------------------
// Basic Trigonometric Wrappers (numhalide::sin, cos, tan, ...)
// -----------------------------------------------------------------------------

TEST(BasicMath, Sin) {
	shape_t s = { 4 };

	Halide::Func input("input");
	Halide::Var x;
	// 0, pi/6, pi/4, pi/2
	input(x) = Halide::select(
		x == 0, 0.0f,
		x == 1, 0.5235988f,
		x == 2, 0.7853982f,
		1.5707963f
	);

	Halide::Func result = sin(input, s);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	result.realize(out);

	EXPECT_NEAR(out(0), 0.0f, 1e-4f);
	EXPECT_NEAR(out(1), 0.5f, 1e-4f);
	EXPECT_NEAR(out(2), std::sqrt(2.0f) / 2.0f, 1e-4f);
	EXPECT_NEAR(out(3), 1.0f, 1e-4f);
}

TEST(BasicMath, Cos) {
	shape_t s = { 3 };

	Halide::Func input("input");
	Halide::Var x;
	// 0, pi/3, pi
	input(x) = Halide::select(
		x == 0, 0.0f,
		x == 1, 1.0471976f,
		3.14159265f
	);

	Halide::Func result = cos(input, s);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	result.realize(out);

	EXPECT_NEAR(out(0), 1.0f, 1e-4f);
	EXPECT_NEAR(out(1), 0.5f, 1e-4f);
	EXPECT_NEAR(out(2), -1.0f, 1e-4f);
}

TEST(BasicMath, Tan) {
	shape_t s = { 3 };

	Halide::Func input("input");
	Halide::Var x;
	// 0, pi/4, -pi/4
	input(x) = Halide::select(
		x == 0, 0.0f,
		x == 1, 0.7853982f,
		-0.7853982f
	);

	Halide::Func result = tan(input, s);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	result.realize(out);

	EXPECT_NEAR(out(0), 0.0f, 1e-4f);
	EXPECT_NEAR(out(1), 1.0f, 1e-4f);
	EXPECT_NEAR(out(2), -1.0f, 1e-4f);
}

TEST(BasicMath, Asin) {
	shape_t s = { 3 };

	Halide::Func input("input");
	Halide::Var x;
	input(x) = Halide::select(x == 0, 0.0f, x == 1, 0.5f, 1.0f);

	Halide::Func result = asin(input, s);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	result.realize(out);

	EXPECT_NEAR(out(0), 0.0f, 1e-4f);
	EXPECT_NEAR(out(1), 0.5235988f, 1e-4f);  // pi/6
	EXPECT_NEAR(out(2), 1.5707963f, 1e-4f);  // pi/2
}

TEST(BasicMath, Acos) {
	shape_t s = { 3 };

	Halide::Func input("input");
	Halide::Var x;
	input(x) = Halide::select(x == 0, 1.0f, x == 1, 0.5f, 0.0f);

	Halide::Func result = acos(input, s);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	result.realize(out);

	EXPECT_NEAR(out(0), 0.0f, 1e-4f);
	EXPECT_NEAR(out(1), 1.0471976f, 1e-4f);  // pi/3
	EXPECT_NEAR(out(2), 1.5707963f, 1e-4f);  // pi/2
}

TEST(BasicMath, Atan) {
	shape_t s = { 3 };

	Halide::Func input("input");
	Halide::Var x;
	input(x) = Halide::select(x == 0, 0.0f, x == 1, 1.0f, -1.0f);

	Halide::Func result = atan(input, s);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	result.realize(out);

	EXPECT_NEAR(out(0), 0.0f, 1e-4f);
	EXPECT_NEAR(out(1), 0.7853982f, 1e-4f);   // pi/4
	EXPECT_NEAR(out(2), -0.7853982f, 1e-4f);  // -pi/4
}

TEST(BasicMath, Atan2) {
	shape_t s = { 4 };

	Halide::Func fy("fy"), fx("fx");
	Halide::Var x;
	// (y, x) pairs: (0,1), (1,0), (1,1), (-1,0)
	fy(x) = Halide::select(x == 0, 0.0f, x == 1, 1.0f, x == 2, 1.0f, -1.0f);
	fx(x) = Halide::select(x == 0, 1.0f, x == 1, 0.0f, x == 2, 1.0f, 0.0f);

	Halide::Func result = atan2(fy, fx, s);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	result.realize(out);

	EXPECT_NEAR(out(0), 0.0f, 1e-4f);              // atan2(0,1) = 0
	EXPECT_NEAR(out(1), 1.5707963f, 1e-4f);         // atan2(1,0) = pi/2
	EXPECT_NEAR(out(2), 0.7853982f, 1e-4f);         // atan2(1,1) = pi/4
	EXPECT_NEAR(out(3), -1.5707963f, 1e-4f);        // atan2(-1,0) = -pi/2
}

TEST(BasicMath, Sinh) {
	shape_t s = { 3 };

	Halide::Func input("input");
	Halide::Var x;
	input(x) = Halide::select(x == 0, 0.0f, x == 1, 1.0f, -1.0f);

	Halide::Func result = sinh(input, s);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	result.realize(out);

	EXPECT_NEAR(out(0), 0.0f, 1e-4f);
	EXPECT_NEAR(out(1), std::sinh(1.0f), 1e-4f);
	EXPECT_NEAR(out(2), std::sinh(-1.0f), 1e-4f);
}

TEST(BasicMath, Cosh) {
	shape_t s = { 3 };

	Halide::Func input("input");
	Halide::Var x;
	input(x) = Halide::select(x == 0, 0.0f, x == 1, 1.0f, -1.0f);

	Halide::Func result = cosh(input, s);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	result.realize(out);

	EXPECT_NEAR(out(0), 1.0f, 1e-4f);
	EXPECT_NEAR(out(1), std::cosh(1.0f), 1e-4f);
	EXPECT_NEAR(out(2), std::cosh(-1.0f), 1e-4f);
}

TEST(BasicMath, Tanh) {
	shape_t s = { 3 };

	Halide::Func input("input");
	Halide::Var x;
	input(x) = Halide::select(x == 0, 0.0f, x == 1, 1.0f, -1.0f);

	Halide::Func result = tanh(input, s);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	result.realize(out);

	EXPECT_NEAR(out(0), 0.0f, 1e-4f);
	EXPECT_NEAR(out(1), std::tanh(1.0f), 1e-4f);
	EXPECT_NEAR(out(2), std::tanh(-1.0f), 1e-4f);
}

// -----------------------------------------------------------------------------
// Exponential / Logarithm
// -----------------------------------------------------------------------------

TEST(BasicMath, Exp) {
	shape_t s = { 3 };

	Halide::Func input("input");
	Halide::Var x;
	input(x) = Halide::select(x == 0, 0.0f, x == 1, 1.0f, 2.0f);

	Halide::Func result = exp(input, s);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	result.realize(out);

	const float e = std::exp(1.0f);
	EXPECT_NEAR(out(0), 1.0f, 1e-4f);         // e^0 = 1
	EXPECT_NEAR(out(1), e, 1e-4f);             // e^1 = e
	EXPECT_NEAR(out(2), e * e, 1e-3f);         // e^2
}

TEST(BasicMath, Log) {
	shape_t s = { 3 };

	const float e = std::exp(1.0f);

	Halide::Func input("input");
	Halide::Var x;
	input(x) = Halide::select(x == 0, 1.0f, x == 1, e, e * e);

	Halide::Func result = log(input, s);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	result.realize(out);

	EXPECT_NEAR(out(0), 0.0f, 1e-4f);   // log(1) = 0
	EXPECT_NEAR(out(1), 1.0f, 1e-4f);   // log(e) = 1
	EXPECT_NEAR(out(2), 2.0f, 1e-4f);   // log(e^2) = 2
}

// -----------------------------------------------------------------------------
// Square Root
// -----------------------------------------------------------------------------

TEST(BasicMath, Sqrt) {
	shape_t s = { 4 };

	Halide::Func input("input");
	Halide::Var x;
	input(x) = Halide::select(x == 0, 0.0f, x == 1, 1.0f, x == 2, 4.0f, 9.0f);

	Halide::Func result = sqrt(input, s);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	result.realize(out);

	EXPECT_NEAR(out(0), 0.0f, 1e-4f);
	EXPECT_NEAR(out(1), 1.0f, 1e-4f);
	EXPECT_NEAR(out(2), 2.0f, 1e-4f);
	EXPECT_NEAR(out(3), 3.0f, 1e-4f);
}

// -----------------------------------------------------------------------------
// Abs / Negative
// -----------------------------------------------------------------------------

TEST(BasicMath, Abs) {
	shape_t s = { 3 };

	Halide::Func input("input");
	Halide::Var x;
	input(x) = Halide::select(x == 0, -3.0f, x == 1, 0.0f, 4.0f);

	Halide::Func result = abs(input, s);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	result.realize(out);

	EXPECT_NEAR(out(0), 3.0f, 1e-4f);
	EXPECT_NEAR(out(1), 0.0f, 1e-4f);
	EXPECT_NEAR(out(2), 4.0f, 1e-4f);
}

TEST(BasicMath, Negative) {
	shape_t s = { 3 };

	Halide::Func input("input");
	Halide::Var x;
	input(x) = Halide::select(x == 0, 1.0f, x == 1, -2.0f, 3.0f);

	Halide::Func result = negative(input, s);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	result.realize(out);

	EXPECT_NEAR(out(0), -1.0f, 1e-4f);
	EXPECT_NEAR(out(1), 2.0f, 1e-4f);
	EXPECT_NEAR(out(2), -3.0f, 1e-4f);
}

// -----------------------------------------------------------------------------
// Rounding Functions
// -----------------------------------------------------------------------------

TEST(BasicMath, Floor) {
	shape_t s = { 3 };

	Halide::Func input("input");
	Halide::Var x;
	input(x) = Halide::select(x == 0, 1.2f, x == 1, -1.2f, 2.9f);

	Halide::Func result = floor(input, s);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	result.realize(out);

	EXPECT_NEAR(out(0), 1.0f, 1e-4f);
	EXPECT_NEAR(out(1), -2.0f, 1e-4f);
	EXPECT_NEAR(out(2), 2.0f, 1e-4f);
}

TEST(BasicMath, Ceil) {
	shape_t s = { 3 };

	Halide::Func input("input");
	Halide::Var x;
	input(x) = Halide::select(x == 0, 1.2f, x == 1, -1.2f, 2.1f);

	Halide::Func result = ceil(input, s);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	result.realize(out);

	EXPECT_NEAR(out(0), 2.0f, 1e-4f);
	EXPECT_NEAR(out(1), -1.0f, 1e-4f);
	EXPECT_NEAR(out(2), 3.0f, 1e-4f);
}

TEST(BasicMath, Round) {
	shape_t s = { 3 };

	Halide::Func input("input");
	Halide::Var x;
	// Halide uses round-half-to-even (banker's rounding)
	// 1.4 -> 1, 1.5 -> 2, 2.5 -> 2
	input(x) = Halide::select(x == 0, 1.4f, x == 1, 1.5f, 2.5f);

	Halide::Func result = round(input, s);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	result.realize(out);

	EXPECT_NEAR(out(0), 1.0f, 1e-4f);   // 1.4 -> 1
	EXPECT_NEAR(out(1), 2.0f, 1e-4f);   // 1.5 -> 2
	EXPECT_NEAR(out(2), 2.0f, 1e-4f);   // 2.5 -> 2 (round-half-to-even)
}

TEST(BasicMath, Trunc) {
	shape_t s = { 3 };

	Halide::Func input("input");
	Halide::Var x;
	input(x) = Halide::select(x == 0, 1.9f, x == 1, -1.9f, 2.1f);

	Halide::Func result = trunc(input, s);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	result.realize(out);

	EXPECT_NEAR(out(0), 1.0f, 1e-4f);
	EXPECT_NEAR(out(1), -1.0f, 1e-4f);
	EXPECT_NEAR(out(2), 2.0f, 1e-4f);
}

TEST(BasicMath, Fix) {
	shape_t s = { 3 };

	Halide::Func input("input");
	Halide::Var x;
	// fix is an alias for trunc (toward zero)
	input(x) = Halide::select(x == 0, 1.9f, x == 1, -1.9f, 2.1f);

	Halide::Func result = fix(input, s);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	result.realize(out);

	EXPECT_NEAR(out(0), 1.0f, 1e-4f);
	EXPECT_NEAR(out(1), -1.0f, 1e-4f);
	EXPECT_NEAR(out(2), 2.0f, 1e-4f);
}

// -----------------------------------------------------------------------------
// Power Functions
// -----------------------------------------------------------------------------

TEST(BasicMath, PowerFunc) {
	shape_t s = { 2 };

	Halide::Func base("base"), exp_f("exp_f");
	Halide::Var x;
	base(x)  = Halide::select(x == 0, 2.0f, 3.0f);
	exp_f(x) = Halide::select(x == 0, 3.0f, 2.0f);

	Halide::Func result = power(base, exp_f, s);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	result.realize(out);

	EXPECT_NEAR(out(0), 8.0f, 1e-4f);   // 2^3 = 8
	EXPECT_NEAR(out(1), 9.0f, 1e-4f);   // 3^2 = 9
}

TEST(BasicMath, PowerExpr) {
	shape_t s = { 2 };

	Halide::Func base("base");
	Halide::Var x;
	base(x) = Halide::select(x == 0, 2.0f, 3.0f);

	Halide::Func result = power(base, Halide::Expr(3.0f), s);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	result.realize(out);

	EXPECT_NEAR(out(0), 8.0f, 1e-4f);   // 2^3 = 8
	EXPECT_NEAR(out(1), 27.0f, 1e-4f);  // 3^3 = 27
}
