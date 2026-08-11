/// @file test_polynomial_special_runtime.cpp
/// @brief Tests for the runtime-Expr special-polynomial overloads:
/// chebyshev_t / legendre_p (runtime degree, explicit Halide::Type) and
/// polyfit_linear (degree-1 normal equations, runtime point count).
/// Pins are closed-form hand values; where a compile-time twin exists the
/// runtime form is checked against it on the shared domain [-1, 1] (the
/// runtime chebyshev clamps outside it, the compile-time recurrence
/// extrapolates — they only agree inside).

#include <gtest/gtest.h>
#include "numhalide_all.h"

using namespace numhalide;

namespace {

Halide::Func make_points(std::initializer_list<float> vals, char const* nm) {
	Halide::Buffer<float> buf((int)vals.size());
	int i = 0;
	for (float v : vals) buf(i++) = v;
	Halide::Func f(nm);
	Halide::Var x;
	f(x) = buf(x);
	return f;
}

} // namespace

TEST(PolynomialSpecialRuntime, ChebyshevPinsAndClamp) {
	// T3(x) = 4x^3 - 3x: T3(0.5) = -1. Clamp contract: T3(1.5) evaluates at
	// the clamped x = 1, so T3(1.5) = T3(1) = 1.
	Halide::Func xs = make_points({0.5f, 1.5f}, "ch_xs");
	Halide::Func rt = chebyshev_t(Halide::Expr(3), xs, Halide::Float(32), "ch_rt");
	Halide::Runtime::Buffer<float> out(2);
	rt.realize(out);
	EXPECT_NEAR(out(0), -1.0f, 1e-5f) << "T3(0.5)";
	EXPECT_NEAR(out(1), 1.0f, 1e-5f) << "T3(1.5) clamped to T3(1)";
}

TEST(PolynomialSpecialRuntime, ChebyshevMatchesIntTwinOnSharedDomain) {
	// Inside [-1, 1] the clamped closed form and the compile-time recurrence
	// agree; check n = 0..4 on a shared grid.
	constexpr int kN = 5;
	Halide::Func xs = make_points({-1.0f, -0.5f, 0.0f, 0.5f, 1.0f}, "cht_xs");
	shape_t s = {kN};

	for (int n = 0; n <= 4; ++n) {
		Halide::Func rt = chebyshev_t(Halide::Expr(n), xs, Halide::Float(32),
		                              "cht_rt" + std::to_string(n));
		Halide::Func ct = chebyshev_t(n, xs, s, "cht_ct" + std::to_string(n));
		Halide::Runtime::Buffer<float> o_rt(kN), o_ct(kN);
		rt.realize(o_rt);
		ct.realize(o_ct);
		for (int i = 0; i < kN; ++i)
			EXPECT_NEAR(o_rt(i), o_ct(i), 1e-5f)
				<< "T_" << n << " twin at sample " << i;
	}
}

TEST(PolynomialSpecialRuntime, LegendrePinsAndZeroAboveThree) {
	// Closed forms at x = 0.5: P0 = 1, P1 = 0.5, P2 = (3/4 - 1)/2 = -0.125,
	// P3 = (5/8 - 3/2)/2 = -0.4375. Contract: n > 3 returns 0.
	Halide::Func xs = make_points({0.5f}, "lg_xs");
	float const expected[] = {1.0f, 0.5f, -0.125f, -0.4375f};

	for (int n = 0; n <= 3; ++n) {
		Halide::Func rt = legendre_p(Halide::Expr(n), xs, Halide::Float(32),
		                             "lg_rt" + std::to_string(n));
		Halide::Runtime::Buffer<float> out(1);
		rt.realize(out);
		EXPECT_NEAR(out(0), expected[n], 1e-6f) << "P" << n << "(0.5)";
	}

	Halide::Func rt5 = legendre_p(Halide::Expr(5), xs, Halide::Float(32), "lg_rt5");
	Halide::Runtime::Buffer<float> out5(1);
	rt5.realize(out5);
	EXPECT_EQ(out5(0), 0.0f) << "P5 must be exactly 0 (n > 3 contract)";
}

TEST(PolynomialSpecialRuntime, LegendreMatchesIntTwinForTableRange) {
	// The explicit table (runtime form) vs the compile-time recurrence for
	// n = 0..3 — algebraically identical, rounding differs slightly.
	constexpr int kN = 6;
	Halide::Func xs = make_points({-1.0f, -0.5f, 0.0f, 0.25f, 0.75f, 1.0f}, "lgt_xs");
	shape_t s = {kN};

	for (int n = 0; n <= 3; ++n) {
		Halide::Func rt = legendre_p(Halide::Expr(n), xs, Halide::Float(32),
		                             "lgt_rt" + std::to_string(n));
		Halide::Func ct = legendre_p(n, xs, s, "lgt_ct" + std::to_string(n));
		Halide::Runtime::Buffer<float> o_rt(kN), o_ct(kN);
		rt.realize(o_rt);
		ct.realize(o_ct);
		for (int i = 0; i < kN; ++i)
			EXPECT_NEAR(o_rt(i), o_ct(i), 1e-5f)
				<< "P_" << n << " twin at sample " << i;
	}
}

TEST(PolynomialSpecialRuntime, PolyfitLinearExactLine) {
	// y = 2x + 1 over x = {0,1,2,3}: Sx=6 Sy=16 Sxx=14 Sxy=34, D=20,
	// slope = 40/20 = 2, intercept = 20/20 = 1. All sums exact in f32.
	// Lowest-first: ret(0) = intercept, ret(1) = slope.
	Halide::Func xs = make_points({0.0f, 1.0f, 2.0f, 3.0f}, "pfl_x");
	Halide::Func ys = make_points({1.0f, 3.0f, 5.0f, 7.0f}, "pfl_y");
	Halide::Func rt = polyfit_linear(xs, ys, Halide::Expr(4), Halide::Float(32), "pfl");
	Halide::Runtime::Buffer<float> out(2);
	rt.realize(out);
	EXPECT_NEAR(out(0), 1.0f, 1e-6f) << "intercept";
	EXPECT_NEAR(out(1), 2.0f, 1e-6f) << "slope";
}

TEST(PolynomialSpecialRuntime, PolyfitLinearNoisyPointsByHand) {
	// Non-collinear points x = {0,1,2}, y = {0,1,3}: Sx=3 Sy=4 Sxx=5 Sxy=7,
	// D = 15 - 9 = 6, slope = (21 - 12)/6 = 1.5,
	// intercept = (20 - 21)/6 = -1/6.
	Halide::Func xs = make_points({0.0f, 1.0f, 2.0f}, "pfn_x");
	Halide::Func ys = make_points({0.0f, 1.0f, 3.0f}, "pfn_y");
	Halide::Func rt = polyfit_linear(xs, ys, Halide::Expr(3), Halide::Float(32), "pfn");
	Halide::Runtime::Buffer<float> out(2);
	rt.realize(out);
	EXPECT_NEAR(out(0), -1.0f / 6.0f, 1e-6f) << "intercept";
	EXPECT_NEAR(out(1), 1.5f, 1e-6f) << "slope";
}

TEST(PolynomialSpecialRuntime, F64TypePins) {
	// All three runtime forms compute in the requested type: Float(64)
	// declared, Float(64) out. polyfit_linear is realized at f64 to prove
	// the pipeline actually runs at that type (inputs are f32 Funcs — the
	// overloads cast internally).
	Halide::Func xs = make_points({0.0f, 1.0f, 2.0f, 3.0f}, "f64_x");
	Halide::Func ys = make_points({1.0f, 3.0f, 5.0f, 7.0f}, "f64_y");

	Halide::Func ch = chebyshev_t(Halide::Expr(3), xs, Halide::Float(64), "f64_ch");
	ASSERT_EQ(ch.types().size(), (size_t)1);
	EXPECT_EQ(ch.types()[0], Halide::Float(64));

	Halide::Func lg = legendre_p(Halide::Expr(2), xs, Halide::Float(64), "f64_lg");
	ASSERT_EQ(lg.types().size(), (size_t)1);
	EXPECT_EQ(lg.types()[0], Halide::Float(64));

	Halide::Func pf = polyfit_linear(xs, ys, Halide::Expr(4), Halide::Float(64), "f64_pf");
	ASSERT_EQ(pf.types().size(), (size_t)1);
	EXPECT_EQ(pf.types()[0], Halide::Float(64));

	Halide::Runtime::Buffer<double> out(2);
	pf.realize(out);
	EXPECT_NEAR(out(0), 1.0, 1e-12) << "f64 intercept";
	EXPECT_NEAR(out(1), 2.0, 1e-12) << "f64 slope";
}
