/// @file test_polynomial_runtime.cpp
/// @brief Tests for the runtime-Expr polynomial overloads
/// (polyval / polyder / polyint / polyadd / polysub with Halide::Expr
/// coefficient counts). Each runtime form is checked against numpy ground
/// truth and, where a compile-time twin exists, against that twin on
/// identical data.

#include <gtest/gtest.h>
#include "numhalide_all.h"

using namespace numhalide;

namespace {

Halide::Func make_coeffs(std::initializer_list<float> vals, char const* nm) {
	Halide::Buffer<float> buf((int)vals.size());
	int i = 0;
	for (float v : vals) buf(i++) = v;
	Halide::Func f(nm);
	Halide::Var x;
	f(x) = buf(x);
	return f;
}

} // namespace

TEST(PolynomialRuntime, PolyvalMatchesNumpyAndTwin) {
	// [2,3,4] highest-first = 2x^2+3x+4 at x=[0,1,2] -> [4,9,18]
	Halide::Func c = make_coeffs({2.0f, 3.0f, 4.0f}, "c");
	Halide::Func xs = make_coeffs({0.0f, 1.0f, 2.0f}, "xs");

	Halide::Func rt = polyval(c, Halide::Expr(3), xs, "pv_rt");
	Halide::Runtime::Buffer<float> out(3);
	rt.realize(out);

	float const expected[] = {4.0f, 9.0f, 18.0f};
	for (int i = 0; i < 3; ++i)
		EXPECT_NEAR(out(i), expected[i], 1e-5f) << "rt[" << i << "]";

	// Compile-time twin on the same data (rank-1 shape).
	shape_t s = {3};
	Halide::Func ct = polyval(c, 3, xs, s, "pv_ct");
	Halide::Runtime::Buffer<float> out_ct(3);
	ct.realize(out_ct);
	for (int i = 0; i < 3; ++i)
		EXPECT_NEAR(out(i), out_ct(i), 1e-5f) << "twin[" << i << "]";
}

TEST(PolynomialRuntime, PolyderMatchesNumpy) {
	// d/dx (2x^2+3x+4) = 4x+3 -> [4,3]
	Halide::Func c = make_coeffs({2.0f, 3.0f, 4.0f}, "cd");
	Halide::Func rt = polyder(c, Halide::Expr(3), "pd_rt");
	Halide::Runtime::Buffer<float> out(2);
	rt.realize(out);
	EXPECT_NEAR(out(0), 4.0f, 1e-6f);
	EXPECT_NEAR(out(1), 3.0f, 1e-6f);
}

TEST(PolynomialRuntime, PolyintMatchesNumpy) {
	// integral of 2x+4, k=7 -> x^2+4x+7 -> [1,4,7] (k at the lowest term)
	Halide::Func c = make_coeffs({2.0f, 4.0f}, "ci");
	Halide::Func rt = polyint(c, Halide::Expr(2), Halide::Expr(7.0f), "pi_rt");
	Halide::Runtime::Buffer<float> out(3);
	rt.realize(out);
	EXPECT_NEAR(out(0), 1.0f, 1e-6f);
	EXPECT_NEAR(out(1), 4.0f, 1e-6f);
	EXPECT_NEAR(out(2), 7.0f, 1e-6f);
}

TEST(PolynomialRuntime, PolyaddPolysubMatchTwins) {
	// Unequal lengths exercise the high-degree-side alignment.
	Halide::Func a = make_coeffs({1.0f, 2.0f}, "pa_a");
	Halide::Func b = make_coeffs({1.0f, 0.0f, 3.0f}, "pa_b");

	Halide::Func add_rt = polyadd(a, Halide::Expr(2), b, Halide::Expr(3), "pa_rt");
	Halide::Func add_ct = polyadd(a, 2, b, 3, "pa_ct");
	Halide::Runtime::Buffer<float> o_rt(3), o_ct(3);
	add_rt.realize(o_rt);
	add_ct.realize(o_ct);
	float const exp_add[] = {1.0f, 1.0f, 5.0f};
	for (int i = 0; i < 3; ++i) {
		EXPECT_NEAR(o_rt(i), exp_add[i], 1e-6f) << "add[" << i << "]";
		EXPECT_NEAR(o_rt(i), o_ct(i), 1e-6f) << "add twin[" << i << "]";
	}

	Halide::Func sub_rt = polysub(a, Halide::Expr(2), b, Halide::Expr(3), "ps_rt");
	Halide::Func sub_ct = polysub(a, 2, b, 3, "ps_ct");
	Halide::Runtime::Buffer<float> s_rt(3), s_ct(3);
	sub_rt.realize(s_rt);
	sub_ct.realize(s_ct);
	float const exp_sub[] = {-1.0f, 1.0f, -1.0f};
	for (int i = 0; i < 3; ++i) {
		EXPECT_NEAR(s_rt(i), exp_sub[i], 1e-6f) << "sub[" << i << "]";
		EXPECT_NEAR(s_rt(i), s_ct(i), 1e-6f) << "sub twin[" << i << "]";
	}
}
