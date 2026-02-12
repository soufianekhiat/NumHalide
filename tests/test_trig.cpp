/// @file test_trig.cpp
/// @brief Tests for trigonometric functions

#include <gtest/gtest.h>
#include "numhalide_all.h"
#include <cmath>

using namespace numhalide;

// -----------------------------------------------------------------------------
// Basic Trig
// -----------------------------------------------------------------------------

TEST(Trig, Sin) {
	shape_t s = { 4 };

	Halide::Func input("input");
	Halide::Var x;
	// 0, pi/6, pi/4, pi/2
	float vals[] = { 0.0f, 0.5235988f, 0.7853982f, 1.5707963f };
	input(x) = Halide::select(
		x == 0, vals[0],
		x == 1, vals[1],
		x == 2, vals[2],
		vals[3]
	);

	Halide::Func result = nh_sin(input, s);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	result.realize(out);

	EXPECT_NEAR(out(0), 0.0f, 1e-5f);
	EXPECT_NEAR(out(1), 0.5f, 1e-5f);
	EXPECT_NEAR(out(2), std::sqrt(2.0f) / 2.0f, 1e-5f);
	EXPECT_NEAR(out(3), 1.0f, 1e-5f);
}

TEST(Trig, Cos) {
	shape_t s = { 3 };

	Halide::Func input("input");
	Halide::Var x;
	// 0, pi/3, pi
	input(x) = Halide::select(
		x == 0, 0.0f,
		x == 1, 1.0471976f,
		3.14159265f
	);

	Halide::Func result = nh_cos(input, s);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	result.realize(out);

	EXPECT_NEAR(out(0), 1.0f, 1e-5f);
	EXPECT_NEAR(out(1), 0.5f, 1e-5f);
	EXPECT_NEAR(out(2), -1.0f, 1e-5f);
}

TEST(Trig, Tan) {
	shape_t s = { 3 };

	Halide::Func input("input");
	Halide::Var x;
	// 0, pi/4, -pi/4
	input(x) = Halide::select(
		x == 0, 0.0f,
		x == 1, 0.7853982f,
		-0.7853982f
	);

	Halide::Func result = nh_tan(input, s);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	result.realize(out);

	EXPECT_NEAR(out(0), 0.0f, 1e-5f);
	EXPECT_NEAR(out(1), 1.0f, 1e-5f);
	EXPECT_NEAR(out(2), -1.0f, 1e-5f);
}

// -----------------------------------------------------------------------------
// Inverse Trig
// -----------------------------------------------------------------------------

TEST(Trig, Asin) {
	shape_t s = { 3 };

	Halide::Func input("input");
	Halide::Var x;
	input(x) = Halide::select(x == 0, 0.0f, x == 1, 0.5f, 1.0f);

	Halide::Func result = nh_asin(input, s);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	result.realize(out);

	EXPECT_NEAR(out(0), 0.0f, 1e-5f);
	EXPECT_NEAR(out(1), 0.5235988f, 1e-5f);  // pi/6
	EXPECT_NEAR(out(2), 1.5707963f, 1e-5f);  // pi/2
}

TEST(Trig, Acos) {
	shape_t s = { 3 };

	Halide::Func input("input");
	Halide::Var x;
	input(x) = Halide::select(x == 0, 1.0f, x == 1, 0.5f, 0.0f);

	Halide::Func result = nh_acos(input, s);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	result.realize(out);

	EXPECT_NEAR(out(0), 0.0f, 1e-5f);
	EXPECT_NEAR(out(1), 1.0471976f, 1e-5f);  // pi/3
	EXPECT_NEAR(out(2), 1.5707963f, 1e-5f);  // pi/2
}

TEST(Trig, Atan) {
	shape_t s = { 3 };

	Halide::Func input("input");
	Halide::Var x;
	input(x) = Halide::select(x == 0, 0.0f, x == 1, 1.0f, -1.0f);

	Halide::Func result = nh_atan(input, s);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	result.realize(out);

	EXPECT_NEAR(out(0), 0.0f, 1e-5f);
	EXPECT_NEAR(out(1), 0.7853982f, 1e-5f);   // pi/4
	EXPECT_NEAR(out(2), -0.7853982f, 1e-5f);  // -pi/4
}

// -----------------------------------------------------------------------------
// Hyperbolic
// -----------------------------------------------------------------------------

TEST(Trig, Sinh) {
	shape_t s = { 3 };

	Halide::Func input("input");
	Halide::Var x;
	input(x) = Halide::select(x == 0, 0.0f, x == 1, 1.0f, -1.0f);

	Halide::Func result = nh_sinh(input, s);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	result.realize(out);

	EXPECT_NEAR(out(0), 0.0f, 1e-5f);
	EXPECT_NEAR(out(1), std::sinh(1.0f), 1e-5f);
	EXPECT_NEAR(out(2), std::sinh(-1.0f), 1e-5f);
}

TEST(Trig, Cosh) {
	shape_t s = { 3 };

	Halide::Func input("input");
	Halide::Var x;
	input(x) = Halide::select(x == 0, 0.0f, x == 1, 1.0f, -1.0f);

	Halide::Func result = nh_cosh(input, s);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	result.realize(out);

	EXPECT_NEAR(out(0), 1.0f, 1e-5f);
	EXPECT_NEAR(out(1), std::cosh(1.0f), 1e-5f);
	EXPECT_NEAR(out(2), std::cosh(-1.0f), 1e-5f);
}

TEST(Trig, Tanh) {
	shape_t s = { 3 };

	Halide::Func input("input");
	Halide::Var x;
	input(x) = Halide::select(x == 0, 0.0f, x == 1, 1.0f, -1.0f);

	Halide::Func result = nh_tanh(input, s);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	result.realize(out);

	EXPECT_NEAR(out(0), 0.0f, 1e-5f);
	EXPECT_NEAR(out(1), std::tanh(1.0f), 1e-5f);
	EXPECT_NEAR(out(2), std::tanh(-1.0f), 1e-5f);
}

// -----------------------------------------------------------------------------
// Inverse Hyperbolic
// -----------------------------------------------------------------------------

TEST(Trig, Asinh) {
	shape_t s = { 3 };

	Halide::Func input("input");
	Halide::Var x;
	input(x) = Halide::select(x == 0, 0.0f, x == 1, 1.0f, -1.0f);

	Halide::Func result = nh_asinh(input, s);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	result.realize(out);

	EXPECT_NEAR(out(0), 0.0f, 1e-5f);
	EXPECT_NEAR(out(1), std::asinh(1.0f), 1e-4f);
	EXPECT_NEAR(out(2), std::asinh(-1.0f), 1e-4f);
}

TEST(Trig, Acosh) {
	shape_t s = { 3 };

	Halide::Func input("input");
	Halide::Var x;
	input(x) = Halide::select(x == 0, 1.0f, x == 1, 2.0f, 10.0f);

	Halide::Func result = nh_acosh(input, s);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	result.realize(out);

	EXPECT_NEAR(out(0), 0.0f, 1e-5f);
	EXPECT_NEAR(out(1), std::acosh(2.0f), 1e-4f);
	EXPECT_NEAR(out(2), std::acosh(10.0f), 1e-4f);
}

TEST(Trig, Atanh) {
	shape_t s = { 3 };

	Halide::Func input("input");
	Halide::Var x;
	input(x) = Halide::select(x == 0, 0.0f, x == 1, 0.5f, -0.5f);

	Halide::Func result = nh_atanh(input, s);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	result.realize(out);

	EXPECT_NEAR(out(0), 0.0f, 1e-5f);
	EXPECT_NEAR(out(1), std::atanh(0.5f), 1e-4f);
	EXPECT_NEAR(out(2), std::atanh(-0.5f), 1e-4f);
}

// -----------------------------------------------------------------------------
// Angle Conversions
// -----------------------------------------------------------------------------

TEST(Trig, DegreesRadians) {
	shape_t s = { 3 };

	Halide::Func input("input");
	Halide::Var x;
	// 0, pi/2, pi
	input(x) = Halide::select(x == 0, 0.0f, x == 1, 1.5707963f, 3.14159265f);

	Halide::Func deg = degrees(input, s);

	Halide::Runtime::Buffer<float> deg_out(s.extents[0]);
	deg.realize(deg_out);

	EXPECT_NEAR(deg_out(0), 0.0f, 1e-3f);
	EXPECT_NEAR(deg_out(1), 90.0f, 1e-3f);
	EXPECT_NEAR(deg_out(2), 180.0f, 1e-3f);

	// Round-trip: radians(degrees(x)) == x
	Halide::Func deg_input("deg_input");
	deg_input(x) = Halide::select(x == 0, 0.0f, x == 1, 90.0f, 180.0f);
	Halide::Func rad = radians(deg_input, s);

	Halide::Runtime::Buffer<float> rad_out(s.extents[0]);
	rad.realize(rad_out);

	EXPECT_NEAR(rad_out(0), 0.0f, 1e-5f);
	EXPECT_NEAR(rad_out(1), 1.5707963f, 1e-4f);
	EXPECT_NEAR(rad_out(2), 3.14159265f, 1e-4f);
}

// -----------------------------------------------------------------------------
// Two-argument functions
// -----------------------------------------------------------------------------

TEST(Trig, Atan2) {
	shape_t s = { 4 };

	Halide::Func fy("fy"), fx("fx");
	Halide::Var x;
	// (y, x) = (0,1), (1,0), (1,1), (-1,0)
	fy(x) = Halide::select(x == 0, 0.0f, x == 1, 1.0f, x == 2, 1.0f, -1.0f);
	fx(x) = Halide::select(x == 0, 1.0f, x == 1, 0.0f, x == 2, 1.0f, 0.0f);

	Halide::Func result = nh_atan2(fy, fx, s);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	result.realize(out);

	EXPECT_NEAR(out(0), 0.0f, 1e-5f);              // atan2(0,1) = 0
	EXPECT_NEAR(out(1), 1.5707963f, 1e-5f);         // atan2(1,0) = pi/2
	EXPECT_NEAR(out(2), 0.7853982f, 1e-5f);         // atan2(1,1) = pi/4
	EXPECT_NEAR(out(3), -1.5707963f, 1e-5f);        // atan2(-1,0) = -pi/2
}

TEST(Trig, Hypot) {
	shape_t s = { 3 };

	Halide::Func fa("fa"), fb("fb");
	Halide::Var x;
	fa(x) = Halide::select(x == 0, 3.0f, x == 1, 0.0f, 5.0f);
	fb(x) = Halide::select(x == 0, 4.0f, x == 1, 7.0f, 12.0f);

	Halide::Func result = hypot(fa, fb, s);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	result.realize(out);

	EXPECT_NEAR(out(0), 5.0f, 1e-5f);           // 3-4-5 triangle
	EXPECT_NEAR(out(1), 7.0f, 1e-5f);           // hypot(0,7) = 7
	EXPECT_NEAR(out(2), 13.0f, 1e-5f);          // 5-12-13 triangle
}
