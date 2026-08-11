/// @file test_polydiv_runtime.cpp
/// @brief Tests for the runtime-Expr polydiv overload (8-step synthetic
/// division chain, Halide::Expr coefficient counts, explicit Halide::Type).
/// Pins are numpy ground truth (numpy.polydiv) in the library's
/// highest-first convention, EXCEPT the remainder length: numpy trims
/// leading zeros down to at least one element, while this polydiv returns
/// a FIXED nv-1 remainder — the last nv-1 coefficients of the reduced
/// dividend, highest-first, zeros kept.

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

TEST(PolydivRuntime, LinearDivisorMatchesNumpy) {
	// numpy.polydiv([1,0,-1], [1,1]): (x^2 - 1)/(x + 1) = (x - 1) rem 0
	// -> q = [1, -1] highest-first, r = [0] (nv-1 = 1 slot, kept).
	Halide::Func u = make_coeffs({1.0f, 0.0f, -1.0f}, "pdl_u");
	Halide::Func v = make_coeffs({1.0f, 1.0f}, "pdl_v");

	polydiv_result res = polydiv(u, v, Halide::Expr(3), Halide::Expr(2),
	                             Halide::Float(32), "pdl");

	Halide::Runtime::Buffer<float> q(2);
	res.quotient.realize(q);
	EXPECT_NEAR(q(0), 1.0f, 1e-6f) << "q[0] (x coefficient)";
	EXPECT_NEAR(q(1), -1.0f, 1e-6f) << "q[1] (constant)";

	Halide::Runtime::Buffer<float> r(1);
	res.remainder.realize(r);
	EXPECT_NEAR(r(0), 0.0f, 1e-6f) << "remainder";
}

TEST(PolydivRuntime, Degree4ByDegree2NonzeroRemainder) {
	// (x^4 + 2x^3 + 3x^2 + 4x + 5) / (x^2 + x + 1), by hand:
	//   step 0: q0 = 1, w -> [0, 1, 2, 4, 5]
	//   step 1: q1 = 1, w -> [0, 0, 1, 3, 5]
	//   step 2: q2 = 1, w -> [0, 0, 0, 2, 4]
	// q = [1, 1, 1], r = [2, 4]  (check: (x^2+x+1)^2 + (2x+4) = u).
	Halide::Func u = make_coeffs({1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, "pd42_u");
	Halide::Func v = make_coeffs({1.0f, 1.0f, 1.0f}, "pd42_v");

	polydiv_result res = polydiv(u, v, Halide::Expr(5), Halide::Expr(3),
	                             Halide::Float(32), "pd42");

	Halide::Runtime::Buffer<float> q(3);
	res.quotient.realize(q);
	float const exp_q[] = {1.0f, 1.0f, 1.0f};
	for (int i = 0; i < 3; ++i)
		EXPECT_NEAR(q(i), exp_q[i], 1e-5f) << "q[" << i << "]";

	Halide::Runtime::Buffer<float> r(2);
	res.remainder.realize(r);
	EXPECT_NEAR(r(0), 2.0f, 1e-5f) << "r[0] (x coefficient)";
	EXPECT_NEAR(r(1), 4.0f, 1e-5f) << "r[1] (constant)";
}

TEST(PolydivRuntime, ExactMultipleRemainderAllZeros) {
	// u = (x + 2)(x^2 + 1) = x^3 + 2x^2 + x + 2 = [1, 2, 1, 2];
	// v = [1, 2] -> q = [1, 0, 1], r = [0]. numpy would return r = [0.]
	// too (it keeps at least one element); here it is the fixed nv-1 = 1
	// slot, which the exact division leaves at zero.
	Halide::Func u = make_coeffs({1.0f, 2.0f, 1.0f, 2.0f}, "pdx_u");
	Halide::Func v = make_coeffs({1.0f, 2.0f}, "pdx_v");

	polydiv_result res = polydiv(u, v, Halide::Expr(4), Halide::Expr(2),
	                             Halide::Float(32), "pdx");

	Halide::Runtime::Buffer<float> q(3);
	res.quotient.realize(q);
	float const exp_q[] = {1.0f, 0.0f, 1.0f};
	for (int i = 0; i < 3; ++i)
		EXPECT_NEAR(q(i), exp_q[i], 1e-5f) << "q[" << i << "]";

	Halide::Runtime::Buffer<float> r(1);
	res.remainder.realize(r);
	EXPECT_NEAR(r(0), 0.0f, 1e-5f) << "exact division leaves r = 0";
}

TEST(PolydivRuntime, ScalarDivisorQuotientOnly) {
	// nv = 1: v = [2] divides every coefficient; the chain supports it
	// (each step's subtraction window is the single position it just
	// consumed, and nq = nu = 3 <= 8). q = [1, 2, 3]. The remainder has
	// length nv-1 = 0 — an EMPTY Func by the fixed-length convention —
	// so it is not realized here.
	Halide::Func u = make_coeffs({2.0f, 4.0f, 6.0f}, "pds_u");
	Halide::Func v = make_coeffs({2.0f}, "pds_v");

	polydiv_result res = polydiv(u, v, Halide::Expr(3), Halide::Expr(1),
	                             Halide::Float(32), "pds");

	Halide::Runtime::Buffer<float> q(3);
	res.quotient.realize(q);
	float const exp_q[] = {1.0f, 2.0f, 3.0f};
	for (int i = 0; i < 3; ++i)
		EXPECT_NEAR(q(i), exp_q[i], 1e-6f) << "q[" << i << "]";
}

TEST(PolydivRuntime, F64TypePins) {
	// The `type` parameter drives the whole chain: f32 input Funcs, f64
	// declared -> both outputs typed f64, and the realized values match
	// the LinearDivisorMatchesNumpy pin at double precision.
	Halide::Func u = make_coeffs({1.0f, 0.0f, -1.0f}, "pd64_u");
	Halide::Func v = make_coeffs({1.0f, 1.0f}, "pd64_v");

	polydiv_result res = polydiv(u, v, Halide::Expr(3), Halide::Expr(2),
	                             Halide::Float(64), "pd64");

	ASSERT_EQ(res.quotient.types().size(), (size_t)1);
	EXPECT_EQ(res.quotient.types()[0], Halide::Float(64));
	ASSERT_EQ(res.remainder.types().size(), (size_t)1);
	EXPECT_EQ(res.remainder.types()[0], Halide::Float(64));

	Halide::Runtime::Buffer<double> q(2);
	res.quotient.realize(q);
	EXPECT_NEAR(q(0), 1.0, 1e-12) << "f64 q[0]";
	EXPECT_NEAR(q(1), -1.0, 1e-12) << "f64 q[1]";

	Halide::Runtime::Buffer<double> r(1);
	res.remainder.realize(r);
	EXPECT_NEAR(r(0), 0.0, 1e-12) << "f64 remainder";
}
