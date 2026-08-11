/// @file test_la_guards.cpp
/// @brief Tests for the eps / zero_on_singular guard parameters on the
///        small-matrix and triangular-solve routines
///
/// Pins the new defaulted parameters: inv2x2/inv3x3 zero_on_singular (a
/// singular input yields the ZERO matrix instead of inf/NaN), cholesky's
/// sqrt-floor eps, back_sub/fwd_sub pivot-guard eps (|pivot| <= eps divides
/// by 1 instead — finite fallback), and lstsq threading eps through qr_gs
/// and back_sub.

#include <gtest/gtest.h>
#include "numhalide_all.h"

#include <cmath>

using namespace numhalide;

namespace {

Halide::Func make_mat(int rows, int cols,
	std::initializer_list<float> vals, const std::string& n)
{
	std::vector<float> v(vals);
	Halide::Buffer<float> buf(cols, rows);
	for (int r = 0; r < rows; ++r)
		for (int c = 0; c < cols; ++c)
			buf(c, r) = v[(size_t)(r * cols + c)];
	Halide::Func f(n);
	Halide::Var x, y;
	f(x, y) = buf(x, y);
	return f;
}

Halide::Func make_vec(std::initializer_list<float> vals, const std::string& n)
{
	std::vector<float> v(vals);
	Halide::Buffer<float> buf((int)v.size());
	for (int i = 0; i < (int)v.size(); ++i)
		buf(i) = v[i];
	Halide::Func f(n);
	Halide::Var x;
	f(x) = buf(x);
	return f;
}

} // namespace

TEST(LAGuards, Inv2x2ZeroOnSingular) {
	// [[2,4],[1,2]]: det = 2*2 - 4*1 = 0 -> ALL FOUR outputs exactly 0
	auto m = make_mat(2, 2, { 2, 4, 1, 2 }, "m_i2s");
	auto inv = inv2x2(m, "inv2s", true);
	Halide::Runtime::Buffer<float> out(2, 2);
	inv.realize(out);
	EXPECT_EQ(out(0, 0), 0.0f);
	EXPECT_EQ(out(1, 0), 0.0f);
	EXPECT_EQ(out(0, 1), 0.0f);
	EXPECT_EQ(out(1, 1), 0.0f);
}

TEST(LAGuards, Inv2x2ZeroOnSingularRegularUnchanged) {
	// [[4,7],[2,6]]: det = 24 - 14 = 10; inv = [[6,-7],[-2,4]]/10
	// The guard must not perturb the regular path.
	auto m = make_mat(2, 2, { 4, 7, 2, 6 }, "m_i2r");
	auto inv = inv2x2(m, "inv2r", true);
	Halide::Runtime::Buffer<float> out(2, 2);
	inv.realize(out);
	EXPECT_NEAR(out(0, 0),  0.6f, 1e-5f);
	EXPECT_NEAR(out(1, 0), -0.7f, 1e-5f);
	EXPECT_NEAR(out(0, 1), -0.2f, 1e-5f);
	EXPECT_NEAR(out(1, 1),  0.4f, 1e-5f);
}

TEST(LAGuards, Inv3x3ZeroOnSingularRegularUnchanged) {
	// M = [[1,2,0],[0,1,3],[4,0,1]], det = 25
	// inv = adj/25 = [[0.04,-0.08,0.24],[0.48,0.04,-0.12],[-0.16,0.32,0.04]]
	auto m = make_mat(3, 3, { 1, 2, 0, 0, 1, 3, 4, 0, 1 }, "m_i3r");
	auto inv = inv3x3(m, "inv3r", true);
	Halide::Runtime::Buffer<float> out(3, 3);
	inv.realize(out);
	const float expected[9] = {
		 0.04f, -0.08f,  0.24f,
		 0.48f,  0.04f, -0.12f,
		-0.16f,  0.32f,  0.04f
	};
	for (int r = 0; r < 3; ++r)
		for (int c = 0; c < 3; ++c)
			EXPECT_NEAR(out(c, r), expected[r * 3 + c], 1e-5f)
				<< "inv3x3 at r=" << r << " c=" << c;
}

TEST(LAGuards, CholeskyEpsSPD) {
	// SPD fixture built from a known lower-triangular factor L0:
	// diag {1, 1.25, 1.5}, below-diag L0[r][c] = 0.1*((r*3+c*5)%7) - 0.2
	// -> L0 = [[1,0,0],[0.1,1.25,0],[0.4,0.2,1.5]], A = L0*L0^T
	double L0[3][3] = { { 0 } };
	const double d[3] = { 1.0, 1.25, 1.5 };
	for (int r = 0; r < 3; ++r) {
		L0[r][r] = d[r];
		for (int c = 0; c < r; ++c)
			L0[r][c] = 0.1 * ((r * 3 + c * 5) % 7) - 0.2;
	}
	double A[3][3];
	for (int r = 0; r < 3; ++r)
		for (int c = 0; c < 3; ++c) {
			A[r][c] = 0.0;
			for (int k = 0; k < 3; ++k)
				A[r][c] += L0[r][k] * L0[c][k];
		}

	// Buffer convention A(col, row)
	Halide::Buffer<float> abuf(3, 3);
	for (int r = 0; r < 3; ++r)
		for (int c = 0; c < 3; ++c)
			abuf(c, r) = static_cast<float>(A[r][c]);
	Halide::Func af("af_chol");
	Halide::Var x, y;
	af(x, y) = abuf(x, y);

	auto L = cholesky(af, 3, "chol_eps", 1e-30f);
	Halide::Runtime::Buffer<float> out(3, 3);   // L(col, row)
	L.realize(out);

	// Reconstruct L*L^T and compare to A
	for (int r = 0; r < 3; ++r)
		for (int c = 0; c < 3; ++c) {
			double rec = 0.0;
			for (int k = 0; k < 3; ++k)
				rec += static_cast<double>(out(k, r)) * static_cast<double>(out(k, c));
			EXPECT_NEAR(rec, A[r][c], 1e-4) << "L*L^T at r=" << r << " c=" << c;
		}

	// Strict upper triangle must be exactly zero (never written)
	for (int r = 0; r < 3; ++r)
		for (int c = r + 1; c < 3; ++c)
			EXPECT_EQ(out(c, r), 0.0f) << "upper at r=" << r << " c=" << c;
}

TEST(LAGuards, BackSubEpsRegularUnchanged) {
	// U = [[2,3],[0,4]], b = [8,8]: x1 = 2, x0 = (8 - 3*2)/2 = 1
	auto U = make_mat(2, 2, { 2, 3, 0, 4 }, "U_bse");
	auto b = make_vec({ 8.0f, 8.0f }, "b_bse");
	auto xf = back_sub(U, b, 2, "bs_eps", 1e-10f);
	Halide::Runtime::Buffer<float> out(2);
	xf.realize(out);
	EXPECT_NEAR(out(0), 1.0f, 1e-5f);
	EXPECT_NEAR(out(1), 2.0f, 1e-5f);
}

TEST(LAGuards, FwdSubEpsRegularUnchanged) {
	// L = [[2,0],[3,4]], b = [4,10]: y0 = 2, y1 = (10 - 3*2)/4 = 1
	auto L = make_mat(2, 2, { 2, 0, 3, 4 }, "L_fse");
	auto b = make_vec({ 4.0f, 10.0f }, "b_fse");
	auto yf = fwd_sub(L, b, 2, "fs_eps", 1e-10f);
	Halide::Runtime::Buffer<float> out(2);
	yf.realize(out);
	EXPECT_NEAR(out(0), 2.0f, 1e-5f);
	EXPECT_NEAR(out(1), 1.0f, 1e-5f);
}

TEST(LAGuards, BackSubEpsZeroPivotStaysFinite) {
	// U = [[0,3],[0,4]] has a ZERO diagonal pivot at (0,0). With eps=1e-10
	// the pivot is replaced by 1 for the division: outputs must be FINITE
	// (the unguarded path would produce inf/NaN). Values themselves are the
	// fallback's, so only finiteness is asserted.
	auto U = make_mat(2, 2, { 0, 3, 0, 4 }, "U_bs0");
	auto b = make_vec({ 8.0f, 8.0f }, "b_bs0");
	auto xf = back_sub(U, b, 2, "bs_zero", 1e-10f);
	Halide::Runtime::Buffer<float> out(2);
	xf.realize(out);
	EXPECT_TRUE(std::isfinite(out(0)));
	EXPECT_TRUE(std::isfinite(out(1)));
}

TEST(LAGuards, LstsqEpsRecovers4x4) {
	// A[r][c] = (r==c ? 4.0+r : 0.0) + 0.2*((r*7+c*2)%5) - 0.4
	// b = A*x_true computed here in double; lstsq with eps must recover x_true.
	const double x_true[4] = { 1.0, -2.0, 3.0, 0.5 };
	double A[4][4];
	double b[4];
	for (int r = 0; r < 4; ++r) {
		for (int c = 0; c < 4; ++c)
			A[r][c] = (r == c ? 4.0 + r : 0.0) + 0.2 * ((r * 7 + c * 2) % 5) - 0.4;
		b[r] = 0.0;
		for (int c = 0; c < 4; ++c)
			b[r] += A[r][c] * x_true[c];
	}

	Halide::Buffer<float> abuf(4, 4);   // A(col, row)
	Halide::Buffer<float> bbuf(4);
	for (int r = 0; r < 4; ++r) {
		for (int c = 0; c < 4; ++c)
			abuf(c, r) = static_cast<float>(A[r][c]);
		bbuf(r) = static_cast<float>(b[r]);
	}
	Halide::Func af("af_lsq");
	Halide::Func bf("bf_lsq");
	Halide::Var x, y;
	af(x, y) = abuf(x, y);
	bf(x) = bbuf(x);

	auto xf = lstsq(af, bf, 4, 4, "lsq_eps", 1e-30f);
	Halide::Runtime::Buffer<float> out(4);
	xf.realize(out);
	for (int i = 0; i < 4; ++i)
		EXPECT_NEAR(out(i), static_cast<float>(x_true[i]), 5e-3f) << "x(" << i << ")";
}
