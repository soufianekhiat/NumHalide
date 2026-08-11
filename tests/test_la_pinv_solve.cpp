/// @file test_la_pinv_solve.cpp
/// @brief Tests for pinv / solve, plus cond / matrix_rank sanity.
///
/// pinv and solve sit directly on the svd_jacobi / qr_gs substrate. The
/// Jacobi rotation SIGN bug (see test_la_svd_jacobi.cpp) meant U/Vt never
/// converged to orthogonality, so every SVD consumer — pinv, cond,
/// matrix_rank — was silently broken while reconstruction-style checks
/// still passed. These tests pin the consumers themselves:
///  - pinv on a well-conditioned matrix behaves as an inverse,
///  - pinv on a SINGULAR matrix delivers the Moore-Penrose answer
///    (the case a Gauss-Jordan inverse cannot produce),
///  - solve recovers a known x from b = A·x computed in double,
///  - cond and matrix_rank give textbook answers on fixtures with known
///    singular values.

#include <gtest/gtest.h>
#include "numhalide_all.h"

#include <cmath>

using namespace numhalide;

namespace {

// Well-conditioned asymmetric 3x3 (same fixture family as the svd tests):
// singular values ~ [3.80, 2.48, 1.89] — far from the pinv tol.
double fixture3(int r, int c)
{
	return (r == c ? 2.5 + 0.5 * r : 0.0) + 0.2 * ((r * 5 + c * 3) % 7) - 0.5;
}

// Diagonally dominant asymmetric 4x4: diag ~ 6+r (±0.6), off-diag in
// [-0.6, 0.6] — row-wise |diag| ≥ 5.4 > 1.8 ≥ Σ|off-diag|.
double fixture4_dd(int r, int c)
{
	return (r == c ? 6.0 + r : 0.0) + 0.3 * ((r * 7 + c * 5) % 5) - 0.6;
}

} // namespace

TEST(PinvSolve, PinvWellConditioned3x3) {
	// Buffer convention A(col, row): A(c, r) = M[r][c]
	Halide::Buffer<float> abuf(3, 3);
	for (int r = 0; r < 3; ++r)
		for (int c = 0; c < 3; ++c)
			abuf(c, r) = static_cast<float>(fixture3(r, c));
	Halide::Func af("af_pinv3");
	Halide::Var x, y;
	af(x, y) = abuf(x, y);

	auto Pi = pinv(af, 3, 3, 1e-10f, 10, "pinv3_wc");
	// pinv(col=j, row=i) = A⁺[i,j]
	Halide::Runtime::Buffer<float> P(3, 3);
	Pi.realize(P);

	// A is invertible, so A·A⁺ = A⁺·A = I.
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			double api = 0.0;  // (A·A⁺)[i][j] = Σ_k A[i][k]·A⁺[k][j]
			double pia = 0.0;  // (A⁺·A)[i][j] = Σ_k A⁺[i][k]·A[k][j]
			for (int k = 0; k < 3; ++k) {
				api += fixture3(i, k) * static_cast<double>(P(j, k));
				pia += static_cast<double>(P(k, i)) * fixture3(k, j);
			}
			double expected = (i == j) ? 1.0 : 0.0;
			EXPECT_NEAR(api, expected, 1e-3) << "A*pinv(A) at (" << i << "," << j << ")";
			EXPECT_NEAR(pia, expected, 1e-3) << "pinv(A)*A at (" << i << "," << j << ")";
		}
	}
}

TEST(PinvSolve, PinvSingular2x2) {
	// A = [[1,0],[0,0]] is SINGULAR: a Gauss-Jordan / closed-form inverse
	// cannot produce anything meaningful here, but the Moore-Penrose
	// pseudo-inverse is exactly A itself (σ = {1, 0}; the zero singular
	// value is dropped by tol, not inverted). This only works because
	// svd_jacobi's rotation sign was just fixed — with the old
	// non-converging sweeps, U/Vt were not orthogonal and the recomposed
	// pseudo-inverse was garbage on precisely this kind of input.
	Halide::Buffer<float> abuf(2, 2);
	abuf(0, 0) = 1.0f; abuf(1, 0) = 0.0f;
	abuf(0, 1) = 0.0f; abuf(1, 1) = 0.0f;
	Halide::Func af("af_pinv2s");
	Halide::Var x, y;
	af(x, y) = abuf(x, y);

	auto Pi = pinv(af, 2, 2, 1e-10f, 10, "pinv2_sing");
	Halide::Runtime::Buffer<float> P(2, 2);
	Pi.realize(P);

	EXPECT_NEAR(P(0, 0), 1.0f, 1e-5f);
	EXPECT_NEAR(P(1, 0), 0.0f, 1e-5f);
	EXPECT_NEAR(P(0, 1), 0.0f, 1e-5f);
	EXPECT_NEAR(P(1, 1), 0.0f, 1e-5f);
}

TEST(PinvSolve, SolveRecoversKnownX4x4) {
	// b = A·x_true computed in double IN-TEST, then solve must recover
	// x_true from the f32 pipeline.
	double const x_true[4] = { 1.0, -2.0, 3.0, 0.5 };

	Halide::Buffer<float> abuf(4, 4);
	Halide::Buffer<float> bbuf(4);
	for (int r = 0; r < 4; ++r) {
		double brow = 0.0;
		for (int c = 0; c < 4; ++c) {
			abuf(c, r) = static_cast<float>(fixture4_dd(r, c));
			brow += fixture4_dd(r, c) * x_true[c];
		}
		bbuf(r) = static_cast<float>(brow);
	}
	Halide::Func af("af_slv4"), bf("bf_slv4");
	Halide::Var x, y, i;
	af(x, y) = abuf(x, y);
	bf(i) = bbuf(i);

	// Exercise the threaded eps too (guards QR column norms + pivots).
	auto xf = solve(af, bf, 4, "slv4_known", 1e-10f);
	Halide::Runtime::Buffer<float> out(4);
	xf.realize(out);

	for (int k = 0; k < 4; ++k)
		EXPECT_NEAR(out(k), x_true[k], 5e-3) << "x(" << k << ")";
}

TEST(PinvSolve, CondDiagonal) {
	// diag(4, 2, 1, 0.5): singular values are the diagonal itself, so the
	// 2-norm condition number is σ_max/σ_min = 4/0.5 = 8. cond() was
	// silently broken before the svd_jacobi sign fix.
	Halide::Buffer<float> abuf(4, 4);
	float const d[4] = { 4.0f, 2.0f, 1.0f, 0.5f };
	for (int r = 0; r < 4; ++r)
		for (int c = 0; c < 4; ++c)
			abuf(c, r) = (r == c) ? d[r] : 0.0f;
	Halide::Func af("af_cond4");
	Halide::Var x, y;
	af(x, y) = abuf(x, y);

	auto cf = cond(af, 4, 4, 10, "cond4_diag");
	auto out = Halide::Runtime::Buffer<float>::make_scalar();
	cf.realize(out);

	EXPECT_NEAR(out(), 8.0f, 1e-2f);
}

TEST(PinvSolve, MatrixRankFullAndDeficient) {
	// Full-rank 3x3: singular values ~ [3.80, 2.48, 1.89], all far above
	// tol → rank 3.
	{
		Halide::Buffer<float> abuf(3, 3);
		for (int r = 0; r < 3; ++r)
			for (int c = 0; c < 3; ++c)
				abuf(c, r) = static_cast<float>(fixture3(r, c));
		Halide::Func af("af_rank3f");
		Halide::Var x, y;
		af(x, y) = abuf(x, y);

		auto rf = matrix_rank(af, 3, 3, 1e-4f, 10, "rank3_full");
		auto out = Halide::Runtime::Buffer<float>::make_scalar();
		rf.realize(out);
		EXPECT_NEAR(out(), 3.0f, 1e-3f);
	}

	// Rank-2 3x3: column 2 duplicates column 0, so one singular value is
	// zero in exact arithmetic. tol = 1e-4 (not the 1e-10 default): f32
	// Jacobi sweeps leave the null singular value at rounding level
	// (~1e-7·σ_max), which must still count as zero.
	{
		double const M[3][3] = {
			{ 1.0, 2.0, 1.0 },
			{ 3.0, 4.0, 3.0 },
			{ 5.0, 6.0, 5.0 },
		};
		Halide::Buffer<float> abuf(3, 3);
		for (int r = 0; r < 3; ++r)
			for (int c = 0; c < 3; ++c)
				abuf(c, r) = static_cast<float>(M[r][c]);
		Halide::Func af("af_rank3d");
		Halide::Var x, y;
		af(x, y) = abuf(x, y);

		auto rf = matrix_rank(af, 3, 3, 1e-4f, 10, "rank3_def");
		auto out = Halide::Runtime::Buffer<float>::make_scalar();
		rf.realize(out);
		EXPECT_NEAR(out(), 2.0f, 1e-3f);
	}
}
