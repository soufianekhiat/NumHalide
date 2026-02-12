/// @file test_polynomial.cpp
/// @brief Tests for polynomial operations

#include <gtest/gtest.h>
#include "numhalide_all.h"
#include <cmath>

using namespace numhalide;

TEST(Poly, Constant) {
	shape_t s = { 3 };
	Halide::Func coeffs("coeffs"), xf("xf");
	Halide::Var x;
	coeffs(x) = 5.0f;  // p(x) = 5
	xf(x) = Halide::cast<float>(x);

	Halide::Func result = polyval(coeffs, 1, xf, s);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	result.realize(out);

	for (int i = 0; i < 3; ++i)
		EXPECT_NEAR(out(i), 5.0f, 1e-5f);
}

TEST(Poly, Linear) {
	shape_t s = { 4 };
	Halide::Func coeffs("coeffs"), xf("xf");
	Halide::Var x;
	// p(x) = 2x + 3 -> coeffs = [2, 3]
	coeffs(x) = Halide::select(x == 0, 2.0f, 3.0f);
	xf(x) = Halide::cast<float>(x);

	Halide::Func result = polyval(coeffs, 2, xf, s);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	result.realize(out);

	EXPECT_NEAR(out(0), 3.0f, 1e-5f);   // 2*0+3
	EXPECT_NEAR(out(1), 5.0f, 1e-5f);   // 2*1+3
	EXPECT_NEAR(out(2), 7.0f, 1e-5f);   // 2*2+3
	EXPECT_NEAR(out(3), 9.0f, 1e-5f);   // 2*3+3
}

TEST(Poly, Quadratic) {
	shape_t s = { 3 };
	Halide::Func coeffs("coeffs"), xf("xf");
	Halide::Var x;
	// p(x) = x^2 - 2x + 1 = (x-1)^2 -> coeffs = [1, -2, 1]
	coeffs(x) = Halide::select(x == 0, 1.0f, x == 1, -2.0f, 1.0f);
	xf(x) = Halide::cast<float>(x);

	Halide::Func result = polyval(coeffs, 3, xf, s);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	result.realize(out);

	EXPECT_NEAR(out(0), 1.0f, 1e-5f);   // (0-1)^2 = 1
	EXPECT_NEAR(out(1), 0.0f, 1e-5f);   // (1-1)^2 = 0
	EXPECT_NEAR(out(2), 1.0f, 1e-5f);   // (2-1)^2 = 1
}

TEST(Poly, ChebyshevT0) {
	shape_t s = { 3 };
	Halide::Func xf("xf");
	Halide::Var x;
	xf(x) = Halide::cast<float>(x) * 0.5f;

	Halide::Func result = chebyshev_t(0, xf, s);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	result.realize(out);

	for (int i = 0; i < 3; ++i)
		EXPECT_NEAR(out(i), 1.0f, 1e-5f);  // T_0(x) = 1
}

TEST(Poly, ChebyshevT1) {
	shape_t s = { 3 };
	Halide::Func xf("xf");
	Halide::Var x;
	xf(x) = Halide::cast<float>(x) * 0.5f;  // 0, 0.5, 1.0

	Halide::Func result = chebyshev_t(1, xf, s);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	result.realize(out);

	EXPECT_NEAR(out(0), 0.0f, 1e-5f);
	EXPECT_NEAR(out(1), 0.5f, 1e-5f);
	EXPECT_NEAR(out(2), 1.0f, 1e-5f);
}

TEST(Poly, ChebyshevT2) {
	shape_t s = { 3 };
	Halide::Func xf("xf");
	Halide::Var x;
	xf(x) = Halide::select(x == 0, 0.0f, x == 1, 0.5f, 1.0f);

	Halide::Func result = chebyshev_t(2, xf, s);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	result.realize(out);

	// T_2(x) = 2x^2 - 1
	EXPECT_NEAR(out(0), -1.0f, 1e-5f);   // 2*0-1
	EXPECT_NEAR(out(1), -0.5f, 1e-5f);   // 2*0.25-1
	EXPECT_NEAR(out(2), 1.0f, 1e-5f);    // 2*1-1
}

TEST(Poly, LegendreP0P1) {
	shape_t s = { 3 };
	Halide::Func xf("xf");
	Halide::Var x;
	xf(x) = Halide::select(x == 0, -1.0f, x == 1, 0.0f, 1.0f);

	Halide::Func p0 = legendre_p(0, xf, s);
	Halide::Func p1 = legendre_p(1, xf, s);

	Halide::Runtime::Buffer<float> p0_out(3), p1_out(3);
	p0.realize(p0_out);
	p1.realize(p1_out);

	// P_0 = 1
	for (int i = 0; i < 3; ++i)
		EXPECT_NEAR(p0_out(i), 1.0f, 1e-5f);

	// P_1 = x
	EXPECT_NEAR(p1_out(0), -1.0f, 1e-5f);
	EXPECT_NEAR(p1_out(1), 0.0f, 1e-5f);
	EXPECT_NEAR(p1_out(2), 1.0f, 1e-5f);
}
