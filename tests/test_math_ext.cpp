/// @file test_math_ext.cpp
/// @brief Tests for extended math functions

#include <gtest/gtest.h>
#include "numhalide_all.h"
#include <cmath>
#include <limits>

using namespace numhalide;

// -----------------------------------------------------------------------------
// Exponential / Log Variants
// -----------------------------------------------------------------------------

TEST(MathExt, Exp2) {
	shape_t s = { 4 };

	Halide::Func input("input");
	Halide::Var x;
	input(x) = Halide::select(x == 0, 0.0f, x == 1, 1.0f, x == 2, 2.0f, 3.0f);

	Halide::Func result = exp2(input, s);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	result.realize(out);

	EXPECT_NEAR(out(0), 1.0f, 1e-5f);   // 2^0
	EXPECT_NEAR(out(1), 2.0f, 1e-5f);   // 2^1
	EXPECT_NEAR(out(2), 4.0f, 1e-5f);   // 2^2
	EXPECT_NEAR(out(3), 8.0f, 1e-5f);   // 2^3
}

TEST(MathExt, Log2) {
	shape_t s = { 3 };

	Halide::Func input("input");
	Halide::Var x;
	input(x) = Halide::select(x == 0, 1.0f, x == 1, 4.0f, 8.0f);

	Halide::Func result = log2(input, s);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	result.realize(out);

	EXPECT_NEAR(out(0), 0.0f, 1e-5f);
	EXPECT_NEAR(out(1), 2.0f, 1e-4f);
	EXPECT_NEAR(out(2), 3.0f, 1e-4f);
}

TEST(MathExt, Log10) {
	shape_t s = { 3 };

	Halide::Func input("input");
	Halide::Var x;
	input(x) = Halide::select(x == 0, 1.0f, x == 1, 100.0f, 1000.0f);

	Halide::Func result = log10(input, s);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	result.realize(out);

	EXPECT_NEAR(out(0), 0.0f, 1e-5f);
	EXPECT_NEAR(out(1), 2.0f, 1e-4f);
	EXPECT_NEAR(out(2), 3.0f, 1e-4f);
}

TEST(MathExt, Expm1) {
	shape_t s = { 3 };

	Halide::Func input("input");
	Halide::Var x;
	input(x) = Halide::select(x == 0, 0.0f, x == 1, 1.0f, -1.0f);

	Halide::Func result = expm1(input, s);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	result.realize(out);

	EXPECT_NEAR(out(0), 0.0f, 1e-5f);
	EXPECT_NEAR(out(1), std::exp(1.0f) - 1.0f, 1e-5f);
	EXPECT_NEAR(out(2), std::exp(-1.0f) - 1.0f, 1e-5f);
}

TEST(MathExt, Log1p) {
	shape_t s = { 3 };

	Halide::Func input("input");
	Halide::Var x;
	input(x) = Halide::select(x == 0, 0.0f, x == 1, 1.0f, 9.0f);

	Halide::Func result = log1p(input, s);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	result.realize(out);

	EXPECT_NEAR(out(0), 0.0f, 1e-5f);                // log(1) = 0
	EXPECT_NEAR(out(1), std::log(2.0f), 1e-5f);      // log(2)
	EXPECT_NEAR(out(2), std::log(10.0f), 1e-5f);     // log(10)
}

// -----------------------------------------------------------------------------
// Power / Root
// -----------------------------------------------------------------------------

TEST(MathExt, Square) {
	shape_t s = { 4 };

	Halide::Func input("input");
	Halide::Var x;
	input(x) = Halide::cast<float>(x) - 1.0f;  // -1, 0, 1, 2

	Halide::Func result = square(input, s);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	result.realize(out);

	EXPECT_NEAR(out(0), 1.0f, 1e-5f);
	EXPECT_NEAR(out(1), 0.0f, 1e-5f);
	EXPECT_NEAR(out(2), 1.0f, 1e-5f);
	EXPECT_NEAR(out(3), 4.0f, 1e-5f);
}

TEST(MathExt, Cbrt) {
	shape_t s = { 3 };

	Halide::Func input("input");
	Halide::Var x;
	input(x) = Halide::select(x == 0, 8.0f, x == 1, 27.0f, -8.0f);

	Halide::Func result = cbrt(input, s);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	result.realize(out);

	EXPECT_NEAR(out(0), 2.0f, 1e-4f);
	EXPECT_NEAR(out(1), 3.0f, 1e-4f);
	EXPECT_NEAR(out(2), -2.0f, 1e-4f);  // sign-aware
}

TEST(MathExt, Reciprocal) {
	shape_t s = { 3 };

	Halide::Func input("input");
	Halide::Var x;
	input(x) = Halide::select(x == 0, 1.0f, x == 1, 2.0f, 4.0f);

	Halide::Func result = reciprocal(input, s);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	result.realize(out);

	EXPECT_NEAR(out(0), 1.0f, 1e-5f);
	EXPECT_NEAR(out(1), 0.5f, 1e-5f);
	EXPECT_NEAR(out(2), 0.25f, 1e-5f);
}

// -----------------------------------------------------------------------------
// Special Functions
// -----------------------------------------------------------------------------

TEST(MathExt, Sinc) {
	shape_t s = { 4 };

	Halide::Func input("input");
	Halide::Var x;
	input(x) = Halide::select(x == 0, 0.0f, x == 1, 1.0f, x == 2, -1.0f, 0.5f);

	Halide::Func result = sinc(input, s);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	result.realize(out);

	constexpr float pi = 3.14159265358979323846f;
	EXPECT_NEAR(out(0), 1.0f, 1e-5f);           // sinc(0) = 1
	EXPECT_NEAR(out(1), 0.0f, 1e-5f);           // sinc(1) = sin(pi)/pi = 0
	EXPECT_NEAR(out(2), 0.0f, 1e-5f);           // sinc(-1) = 0
	EXPECT_NEAR(out(3), std::sin(pi * 0.5f) / (pi * 0.5f), 1e-5f);
}

TEST(MathExt, Heaviside) {
	shape_t s = { 5 };

	Halide::Func input("input");
	Halide::Var x;
	input(x) = Halide::cast<float>(x) - 2.0f;  // -2, -1, 0, 1, 2

	Halide::Func result = heaviside(input, 0.5f, s);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	result.realize(out);

	EXPECT_NEAR(out(0), 0.0f, 1e-5f);   // x < 0
	EXPECT_NEAR(out(1), 0.0f, 1e-5f);   // x < 0
	EXPECT_NEAR(out(2), 0.5f, 1e-5f);   // x == 0 -> h0
	EXPECT_NEAR(out(3), 1.0f, 1e-5f);   // x > 0
	EXPECT_NEAR(out(4), 1.0f, 1e-5f);   // x > 0
}

// -----------------------------------------------------------------------------
// Modular Arithmetic
// -----------------------------------------------------------------------------

TEST(MathExt, Fmod) {
	shape_t s = { 3 };

	Halide::Func fa("fa"), fb("fb");
	Halide::Var x;
	fa(x) = Halide::select(x == 0, 5.0f, x == 1, 7.5f, 10.0f);
	fb(x) = Halide::select(x == 0, 3.0f, x == 1, 2.5f, 3.0f);

	Halide::Func result = fmod(fa, fb, s);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	result.realize(out);

	EXPECT_NEAR(out(0), 2.0f, 1e-5f);   // 5 mod 3
	EXPECT_NEAR(out(1), 0.0f, 1e-5f);   // 7.5 mod 2.5
	EXPECT_NEAR(out(2), 1.0f, 1e-5f);   // 10 mod 3
}

TEST(MathExt, Remainder) {
	shape_t s = { 3 };

	Halide::Func fa("fa"), fb("fb");
	Halide::Var x;
	fa(x) = Halide::select(x == 0, 5.0f, x == 1, 7.0f, -3.0f);
	fb(x) = Halide::select(x == 0, 3.0f, x == 1, 4.0f, 2.0f);

	Halide::Func result = remainder(fa, fb, s);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	result.realize(out);

	// IEEE remainder: a - round(a/b)*b
	EXPECT_NEAR(out(0), -1.0f, 1e-5f);  // 5 - round(5/3)*3 = 5 - 2*3 = -1
	EXPECT_NEAR(out(1), -1.0f, 1e-5f);  // 7 - round(7/4)*4 = 7 - 2*4 = -1
	EXPECT_NEAR(out(2), 1.0f, 1e-5f);  // -3 - round(-3/2)*2 = -3 - (-2)*2 = 1
}

// -----------------------------------------------------------------------------
// NaN / Inf Handling
// -----------------------------------------------------------------------------

TEST(MathExt, NanToNum) {
	shape_t s = { 3 };

	// Use ImageParam to avoid Halide constant-folding of special values
	Halide::ImageParam input_param(Halide::Float(32), 1, "nan_input");
	float nan_data[] = {
		std::numeric_limits<float>::quiet_NaN(),
		std::numeric_limits<float>::infinity(),
		-std::numeric_limits<float>::infinity()
	};
	Halide::Buffer<float> input_buf(nan_data, 3);
	input_param.set(input_buf);

	Halide::Func input("input");
	Halide::Var x;
	input(x) = input_param(x);

	Halide::Func result = nan_to_num(input, s, 0.0f, 999.0f, -999.0f);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	result.realize(out);

	EXPECT_NEAR(out(0), 0.0f, 1e-5f);     // NaN -> 0
	EXPECT_NEAR(out(1), 999.0f, 1e-5f);   // +Inf -> 999
	EXPECT_NEAR(out(2), -999.0f, 1e-5f);  // -Inf -> -999
}
