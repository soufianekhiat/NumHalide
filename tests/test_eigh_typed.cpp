/// @file test_eigh_typed.cpp
/// @brief Regression tests for the typed eigh_jacobi overload
///
/// Pins the delegation-quality upgrade of eigh_jacobi (the same treatment
/// svd_jacobi received): type-generic computation (make_const, no f32
/// hardcoding), ALGEBRAIC zeta/t/c/s rotations instead of the legacy trig
/// theta = 0.5*atan2 form, an eps skip guard (identity rotation on tiny
/// off-diagonals — also what keeps reverse-mode derivatives finite at
/// diagonal inputs), ASCENDING eigenvalues via the AD-safe stable-rank
/// indicator gather, and sign-normalized eigenvector columns (last
/// component of each column >= 0).
///
/// The rotation solves t^2 + 2*zeta*t - 1 = 0 with
/// zeta = (A_qq - A_pp)/(2*A_pq), annihilating A'_pq only with the update
/// (p)' = c*(p) - s*(q), (q)' = s*(p) + c*(q) — the same sign pairing
/// svd_jacobi pins (the opposite pairing was the historic svd oscillation
/// bug). Sweep-invariance below is the anti-oscillation witness.

#include <gtest/gtest.h>
#include "numhalide_all.h"

#include <cmath>

using namespace numhalide;

namespace {

// Asymmetric svd-study fixture (test_la_svd_jacobi.cpp), symmetrized:
// Sym[r][c] = 0.5*(M[r][c] + M[c][r]) with
// M[r][c] = (r==c ? 2.5 + 0.5*r : 0.0) + 0.2*((r*5 + c*3) % 7) - 0.5
// giving Sym = [[2.0, 0.3, 0.4], [0.3, 2.7, 0.5], [0.4, 0.5, 3.4]].
double fixture3(int r, int c)
{
	return (r == c ? 2.5 + 0.5 * r : 0.0) + 0.2 * ((r * 5 + c * 3) % 7) - 0.5;
}

double sym3(int r, int c)
{
	return 0.5 * (fixture3(r, c) + fixture3(c, r));
}

// REFERENCE SOURCE for the sym3 pins below: a double-precision cyclic
// Jacobi mirroring the source's algebraic rotation and sweep order
// (p < q ascending) run to convergence; residual max|Sym*v - lambda*v|
// ~= 9e-16, i.e. these are the true eigenpairs to double rounding.
// Eigenvalues ascending:
double const kSym3Lam[3] = {
	1.8539617505394328, 2.4498442140738663, 3.7961940353866992
};
// Unit eigenvector columns, sign-normalized (last component >= 0),
// kSym3Vec[k][r] = component r of the eigenvector paired with kSym3Lam[k]:
double const kSym3Vec[3][3] = {
	{ -0.95612238297803620,  0.23840668177540217, 0.17027108636889379 },
	{ -0.12388629193474315, -0.85568885118379734, 0.50244280931307206 },
	{  0.26548479323505658,  0.45930256263665092, 0.84767858325921175 },
};

Halide::Expr eps_of(Halide::Type t, double v)
{
	return Halide::Internal::make_const(t, v);
}

} // namespace

TEST(EighTyped, Identity3x3) {
	// A = I: every off-diagonal is 0 < eps, so every rotation is the skip
	// guard's exact identity — V stays I, eigenvalues stay 1. The stable
	// rank keeps equal eigenvalues in original order, and the sign rule
	// maps a ZERO last component to +1 (0 >= 0), so the output is exactly
	// the identity eigendecomposition.
	Halide::Buffer<float> abuf(3, 3);
	for (int r = 0; r < 3; ++r)
		for (int c = 0; c < 3; ++c)
			abuf(c, r) = (r == c) ? 1.0f : 0.0f;
	Halide::Func af("af_eigh_id3");
	Halide::Var x, y;
	af(x, y) = abuf(x, y);

	auto eg = eigh_jacobi(af, 3, 4, Halide::Float(32),
		eps_of(Halide::Float(32), 1e-12), "eigh_id3");

	Halide::Runtime::Buffer<float> lam(3);
	Halide::Runtime::Buffer<float> V(3, 3);  // V(col=k, row=r)
	eg.eigenvalues.realize(lam);
	eg.eigenvectors.realize(V);

	for (int k = 0; k < 3; ++k)
		EXPECT_NEAR(lam(k), 1.0f, 1e-12) << "lambda(" << k << ")";
	for (int k = 0; k < 3; ++k)
		for (int r = 0; r < 3; ++r)
			EXPECT_NEAR(V(k, r), (k == r) ? 1.0f : 0.0f, 1e-12)
				<< "V(col=" << k << ", row=" << r << ")";
}

TEST(EighTyped, DiagonalAscendingAndSign3x3) {
	// A = diag(3, 1, 2): all rotations skip, V stays I; the ascending
	// gather must permute BOTH eigenvalues and eigenvector columns:
	// lam = [1, 2, 3], columns = [e_1, e_2, e_0]. The diag fixture forces
	// +-e_k columns and the sign rule makes the sign deterministic: e_1
	// and e_0 have a ZERO last component (-> +1), e_2 has last component
	// 1 (>= 0, kept) — so all columns come out as +e_k exactly.
	Halide::Buffer<float> abuf(3, 3);
	double const diag[3] = { 3.0, 1.0, 2.0 };
	for (int r = 0; r < 3; ++r)
		for (int c = 0; c < 3; ++c)
			abuf(c, r) = (r == c) ? static_cast<float>(diag[r]) : 0.0f;
	Halide::Func af("af_eigh_diag3");
	Halide::Var x, y;
	af(x, y) = abuf(x, y);

	auto eg = eigh_jacobi(af, 3, 4, Halide::Float(32),
		eps_of(Halide::Float(32), 1e-12), "eigh_diag3");

	Halide::Runtime::Buffer<float> lam(3);
	Halide::Runtime::Buffer<float> V(3, 3);
	eg.eigenvalues.realize(lam);
	eg.eigenvectors.realize(V);

	EXPECT_NEAR(lam(0), 1.0f, 1e-12);
	EXPECT_NEAR(lam(1), 2.0f, 1e-12);
	EXPECT_NEAR(lam(2), 3.0f, 1e-12);

	int const expect_col[3] = { 1, 2, 0 };  // output col k = e_{expect_col[k]}
	for (int k = 0; k < 3; ++k)
		for (int r = 0; r < 3; ++r)
			EXPECT_NEAR(V(k, r), (r == expect_col[k]) ? 1.0f : 0.0f, 1e-12)
				<< "V(col=" << k << ", row=" << r << ")";
}

TEST(EighTyped, GenericSymmetric3x3) {
	Halide::Buffer<float> abuf(3, 3);
	for (int r = 0; r < 3; ++r)
		for (int c = 0; c < 3; ++c)
			abuf(c, r) = static_cast<float>(sym3(r, c));
	Halide::Func af("af_eigh_gen3");
	Halide::Var x, y;
	af(x, y) = abuf(x, y);

	auto eg = eigh_jacobi(af, 3, 8, Halide::Float(32),
		eps_of(Halide::Float(32), 1e-12), "eigh_gen3");

	Halide::Runtime::Buffer<float> lam(3);
	Halide::Runtime::Buffer<float> V(3, 3);
	eg.eigenvalues.realize(lam);
	eg.eigenvectors.realize(V);

	// (a) Orthogonality V^T*V = I: columns are V(col=k); dot columns i,j
	// over rows.
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			double vtv = 0.0;
			for (int r = 0; r < 3; ++r)
				vtv += static_cast<double>(V(i, r)) * static_cast<double>(V(j, r));
			double expected = (i == j) ? 1.0 : 0.0;
			EXPECT_NEAR(vtv, expected, 1e-4) << "V^T*V at (" << i << "," << j << ")";
		}
	}

	// (b) Reconstruction: sum_k V(k,r)*lam(k)*V(k,c) = Sym[r][c]
	for (int r = 0; r < 3; ++r) {
		for (int c = 0; c < 3; ++c) {
			double rec = 0.0;
			for (int k = 0; k < 3; ++k)
				rec += static_cast<double>(V(k, r)) * static_cast<double>(lam(k))
					 * static_cast<double>(V(k, c));
			EXPECT_NEAR(rec, sym3(r, c), 1e-3) << "recon at r=" << r << " c=" << c;
		}
	}

	// (c) Ascending order + value pins vs the double-precision reference
	EXPECT_LE(lam(0), lam(1));
	EXPECT_LE(lam(1), lam(2));
	for (int k = 0; k < 3; ++k)
		EXPECT_NEAR(static_cast<double>(lam(k)), kSym3Lam[k], 1e-4)
			<< "lambda(" << k << ")";

	// (d) Sign rule: last component of every column >= 0
	for (int k = 0; k < 3; ++k)
		EXPECT_GE(V(k, 2), -1e-6f) << "sign rule on column " << k;
}

TEST(EighTyped, SweepInvariance3x3) {
	// Anti-oscillation pin: a converged decomposition must not depend on
	// the sweep count. The historic svd rotation-sign bug oscillated, so
	// outputs depended on sweep parity — this is the witness that the
	// eigh update pairing matches its t^2 + 2*zeta*t - 1 = 0 root. The
	// sign normalization makes eigenVECTORS comparable across sweep
	// counts too (without it +-v would be free to differ).
	Halide::Buffer<float> abuf(3, 3);
	for (int r = 0; r < 3; ++r)
		for (int c = 0; c < 3; ++c)
			abuf(c, r) = static_cast<float>(sym3(r, c));
	Halide::Func af("af_eigh_sw3");
	Halide::Var x, y;
	af(x, y) = abuf(x, y);

	auto eg6 = eigh_jacobi(af, 3, 6, Halide::Float(32),
		eps_of(Halide::Float(32), 1e-12), "eigh_sw6");
	auto eg12 = eigh_jacobi(af, 3, 12, Halide::Float(32),
		eps_of(Halide::Float(32), 1e-12), "eigh_sw12");

	Halide::Runtime::Buffer<float> lam6(3), lam12(3);
	Halide::Runtime::Buffer<float> V6(3, 3), V12(3, 3);
	eg6.eigenvalues.realize(lam6);
	eg6.eigenvectors.realize(V6);
	eg12.eigenvalues.realize(lam12);
	eg12.eigenvectors.realize(V12);

	for (int k = 0; k < 3; ++k)
		EXPECT_NEAR(static_cast<double>(lam6(k)), static_cast<double>(lam12(k)), 1e-4)
			<< "lambda(" << k << ") changed between sweeps=6 and sweeps=12";
	for (int k = 0; k < 3; ++k)
		for (int r = 0; r < 3; ++r)
			EXPECT_NEAR(static_cast<double>(V6(k, r)), static_cast<double>(V12(k, r)), 1e-3)
				<< "V(col=" << k << ", row=" << r << ") changed between sweeps";
}

TEST(EighTyped, RepeatedEigenvalue3x3) {
	// A = [[2,0,0],[0,3,1],[0,1,3]]: eigenvalues {2, 2, 4} (the 2x2 block
	// [[3,1],[1,3]] contributes 2 and 4). Within the repeated-eigenvalue
	// subspace the eigenvector DIRECTIONS are not unique, so this pins
	// only V^T*V = I and the eigenvalues.
	Halide::Buffer<float> abuf(3, 3);
	double const A[3][3] = { { 2, 0, 0 }, { 0, 3, 1 }, { 0, 1, 3 } };
	for (int r = 0; r < 3; ++r)
		for (int c = 0; c < 3; ++c)
			abuf(c, r) = static_cast<float>(A[r][c]);
	Halide::Func af("af_eigh_rep3");
	Halide::Var x, y;
	af(x, y) = abuf(x, y);

	auto eg = eigh_jacobi(af, 3, 8, Halide::Float(32),
		eps_of(Halide::Float(32), 1e-12), "eigh_rep3");

	Halide::Runtime::Buffer<float> lam(3);
	Halide::Runtime::Buffer<float> V(3, 3);
	eg.eigenvalues.realize(lam);
	eg.eigenvectors.realize(V);

	EXPECT_NEAR(lam(0), 2.0f, 1e-4f);
	EXPECT_NEAR(lam(1), 2.0f, 1e-4f);
	EXPECT_NEAR(lam(2), 4.0f, 1e-4f);

	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			double vtv = 0.0;
			for (int r = 0; r < 3; ++r)
				vtv += static_cast<double>(V(i, r)) * static_cast<double>(V(j, r));
			double expected = (i == j) ? 1.0 : 0.0;
			EXPECT_NEAR(vtv, expected, 1e-4) << "V^T*V at (" << i << "," << j << ")";
		}
	}
}

TEST(EighTyped, F64TypingAndValues2x2) {
	// FIX WITNESS for the f32 hardcoding: the legacy form cast everything
	// through float, so an f64 pipeline silently truncated (and could not
	// even realize into a double buffer). Hand-solvable fixture
	// [[2,1],[1,2]]: lambda = {1, 3}, eigenvectors (1,-1)/sqrt(2) and
	// (1,1)/sqrt(2). The sign rule FLIPS column 0 (last component -1/sqrt2
	// < 0) to (-1/sqrt2, +1/sqrt2) — pinning the flip machinery itself.
	// The 1e-12 tolerances are unreachable through an f32 pipeline
	// (~1e-7): passing here is the typed-computation witness.
	Halide::Buffer<double> abuf(2, 2);
	double const A[2][2] = { { 2, 1 }, { 1, 2 } };
	for (int r = 0; r < 2; ++r)
		for (int c = 0; c < 2; ++c)
			abuf(c, r) = A[r][c];
	Halide::Func af("af_eigh_d2");
	Halide::Var x, y;
	af(x, y) = abuf(x, y);

	auto eg = eigh_jacobi(af, 2, 4, Halide::Float(64),
		eps_of(Halide::Float(64), 1e-30), "eigh_d2");

	Halide::Runtime::Buffer<double> lam(2);
	Halide::Runtime::Buffer<double> V(2, 2);
	eg.eigenvalues.realize(lam);
	eg.eigenvectors.realize(V);

	EXPECT_NEAR(lam(0), 1.0, 1e-12);
	EXPECT_NEAR(lam(1), 3.0, 1e-12);

	double const inv_sqrt2 = 1.0 / std::sqrt(2.0);
	EXPECT_NEAR(V(0, 0), -inv_sqrt2, 1e-12);  // col 0 flipped by the sign rule
	EXPECT_NEAR(V(0, 1),  inv_sqrt2, 1e-12);
	EXPECT_NEAR(V(1, 0),  inv_sqrt2, 1e-12);
	EXPECT_NEAR(V(1, 1),  inv_sqrt2, 1e-12);
}

TEST(EighTyped, F64GenericSymmetric3x3) {
	// f64 typing + value pins on the generic fixture: eigenvalues AND
	// sign-normalized eigenvector columns against the double-precision
	// Jacobi reference (see kSym3Lam/kSym3Vec above). Tolerances far below
	// f32 resolution — fails against any f32-hardcoded path.
	Halide::Buffer<double> abuf(3, 3);
	for (int r = 0; r < 3; ++r)
		for (int c = 0; c < 3; ++c)
			abuf(c, r) = sym3(r, c);
	Halide::Func af("af_eigh_d3");
	Halide::Var x, y;
	af(x, y) = abuf(x, y);

	auto eg = eigh_jacobi(af, 3, 12, Halide::Float(64),
		eps_of(Halide::Float(64), 1e-30), "eigh_d3");

	Halide::Runtime::Buffer<double> lam(3);
	Halide::Runtime::Buffer<double> V(3, 3);
	eg.eigenvalues.realize(lam);
	eg.eigenvectors.realize(V);

	for (int k = 0; k < 3; ++k)
		EXPECT_NEAR(lam(k), kSym3Lam[k], 1e-10) << "lambda(" << k << ")";
	for (int k = 0; k < 3; ++k)
		for (int r = 0; r < 3; ++r)
			EXPECT_NEAR(V(k, r), kSym3Vec[k][r], 1e-8)
				<< "V(col=" << k << ", row=" << r << ")";

	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			double vtv = 0.0;
			double rec = 0.0;
			for (int r = 0; r < 3; ++r)
				vtv += V(i, r) * V(j, r);
			for (int k = 0; k < 3; ++k)
				rec += V(k, i) * lam(k) * V(k, j);
			EXPECT_NEAR(vtv, (i == j) ? 1.0 : 0.0, 1e-12)
				<< "V^T*V at (" << i << "," << j << ")";
			EXPECT_NEAR(rec, sym3(i, j), 1e-12)
				<< "recon at r=" << i << " c=" << j;
		}
	}
}
