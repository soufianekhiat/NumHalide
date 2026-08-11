/// @file test_kaiser_runtime.cpp
/// @brief Tests for the RUNTIME typed kaiser overload
///        kaiser(Expr n, Expr beta, Type type) and the i0_expr refactor.
///
/// Reference values computed from the CONVERGED f64 Bessel series (I0 summed
/// to 1e-18 relative) — the true numpy.kaiser values. The A&S i0_expr is
/// ~1e-7 on I0 for all x, so the window ratio lands well inside the 1e-6 /
/// 2e-6 tolerances. The old compile-time kaiser overload's 6-term truncated
/// series would NOT: its ratio deviates from the Bessel truth at large beta
/// (the parent repo measured up to 2.5e-5 at beta=8.6 for a 10-term series;
/// 6 terms is worse). These pins are the Bessel-conformance contract.

#include <gtest/gtest.h>
#include "numhalide_all.h"

#include <cmath>

using namespace numhalide;

namespace {

Halide::Func make_1d(std::initializer_list<float> vals, const std::string& n = "f") {
	int sz = (int)vals.size();
	std::vector<float> v(vals);
	Halide::Func f(n);
	Halide::Var x;
	Halide::Buffer<float> buf(sz);
	for (int i = 0; i < sz; ++i) buf(i) = v[i];
	f(x) = buf(x);
	return f;
}

} // namespace

TEST(KaiserRuntime, N5Beta2ConvergedSeriesPins) {
	// n=5, beta=2: all I0 arguments are <= 2, deep inside the A&S small
	// branch (a plain polynomial), so the only error is the ~1e-7
	// approximation plus f32 rounding.
	Halide::Func w = kaiser(Halide::Expr(5), Halide::Expr(2.0f),
	                        Halide::Float(32), "k5");
	Halide::Runtime::Buffer<float> out(5);
	w.realize(out);

	double const expected[5] = { 0.438676280, 0.834761433, 1.0,
	                             0.834761433, 0.438676280 };
	for (int i = 0; i < 5; ++i)
		EXPECT_NEAR(out(i), expected[i], 1e-6) << "kaiser5[" << i << "]";
}

TEST(KaiserRuntime, N9Beta86ConvergedSeriesPins) {
	// n=9, beta=8.6: I0(8.6) ~ 750 exercises the A&S large branch
	// (exp(x)/sqrt(x) * poly). The converged-series truth at 2e-6 — this is
	// the pin the truncated-series forms fail.
	Halide::Func w = kaiser(Halide::Expr(9), Halide::Expr(8.6f),
	                        Halide::Float(32), "k9");
	Halide::Runtime::Buffer<float> out(9);
	w.realize(out);

	double const expected[9] = { 0.001332514, 0.067472079, 0.340393622,
	                             0.773829381, 1.0,         0.773829381,
	                             0.340393622, 0.067472079, 0.001332514 };
	for (int i = 0; i < 9; ++i)
		EXPECT_NEAR(out(i), expected[i], 2e-6) << "kaiser9[" << i << "]";
}

TEST(KaiserRuntime, SymmetryAndCenter) {
	// n=9: n-1 = 8 is a power of two, so t = i/4 - 1 is exact in f32 and
	// the mirrored indices produce bit-identical arguments — w[i] and
	// w[n-1-i] run the identical computation. EXPECT_FLOAT_EQ (4 ulps)
	// rather than EXPECT_EQ only to stay robust to compiler reassociation.
	// Center (t = 0): arg == beta, so the ratio is I0(beta)/I0(beta) == 1.
	Halide::Func w = kaiser(Halide::Expr(9), Halide::Expr(8.6f),
	                        Halide::Float(32), "ksym");
	Halide::Runtime::Buffer<float> out(9);
	w.realize(out);

	for (int i = 0; i < 4; ++i)
		EXPECT_FLOAT_EQ(out(i), out(8 - i)) << "mirror pair " << i;
	EXPECT_FLOAT_EQ(out(4), 1.0f);
}

TEST(KaiserRuntime, F64TypePin) {
	// The whole ratio is computed in the requested type: Float(64) in,
	// Float(64) out. Realize an f64 buffer to prove the pipeline runs at
	// that type, and re-check the endpoint/center values (the f32-literal
	// A&S coefficients promote to double; accuracy stays ~1e-7-bounded).
	Halide::Func w = kaiser(Halide::Expr(9), Halide::Expr(8.6),
	                        Halide::Float(64), "k64");
	ASSERT_EQ(w.types().size(), (size_t)1);
	EXPECT_EQ(w.types()[0], Halide::Float(64));

	Halide::Runtime::Buffer<double> out(9);
	w.realize(out);
	EXPECT_NEAR(out(0), 0.001332514, 2e-6);
	EXPECT_NEAR(out(2), 0.340393622, 2e-6);
	EXPECT_NEAR(out(4), 1.0, 1e-9);
}

// --- i0_expr refactor regression -------------------------------------------
// The Func form i0() now delegates to i0_expr. Same fixtures as
// test_numeric.cpp (I0(0), I0(1), I0(4)) — the refactor is value-identical,
// so the old pins must still hold.

TEST(KaiserRuntime, I0FuncFormStillMatchesOldFixtures) {
	auto f = make_1d({0.0f, 1.0f, 4.0f}, "i0reg_f");
	shape_t s = {3};
	auto r = i0(f, s);
	Halide::Runtime::Buffer<float> out(3);
	r.realize(out);
	EXPECT_NEAR(out(0), 1.0f, 1e-5f);        // I0(0) = 1
	EXPECT_NEAR(out(1), 1.2660658f, 1e-3f);  // I0(1)
	EXPECT_NEAR(out(2), 11.3019219f, 0.01f); // I0(4), large branch
}

TEST(KaiserRuntime, I0ExprMatchesFuncForm) {
	// Direct Expr-level use produces the same values as the Func wrapper —
	// the delegation is the identical computation.
	auto f = make_1d({0.0f, 1.0f, 2.5f, 4.0f, 8.6f}, "i0e_f");
	shape_t s = {5};
	auto func_form = i0(f, s, "i0e_func");

	Halide::Func expr_form("i0e_expr");
	Halide::Var x;
	expr_form(x) = i0_expr(f(x));

	Halide::Runtime::Buffer<float> a(5), b(5);
	func_form.realize(a);
	expr_form.realize(b);
	for (int i = 0; i < 5; ++i)
		EXPECT_FLOAT_EQ(a(i), b(i)) << "i0 point " << i;
}
