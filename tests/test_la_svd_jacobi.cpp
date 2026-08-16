/// @file test_la_svd_jacobi.cpp
/// @brief Regression tests for svd_jacobi convergence (rotation-sign bug)
///
/// Pins the fix for a Jacobi rotation SIGN bug: the tangent t solves
/// t^2 + 2*zeta*t - 1 = 0, which annihilates the p.q column dot product
/// ONLY with the standard update wp' = c*wp - s*wq, wq' = s*wp + c*wq.
/// The previous opposite-sign update left gamma unreduced and the sweeps
/// OSCILLATED instead of converging: max |U^T*U - I| off-diagonal reached
/// ~0.39 at ANY sweep count.
///
/// IMPORTANT: reconstruction U*S*Vt = A held TRIVIALLY throughout the bug,
/// because W*V^T = A by construction for any accumulated rotations —
/// reconstruction ALONE cannot catch non-convergence. These tests therefore
/// assert ORTHOGONALITY of U and Vt (the property Jacobi sweeps must earn)
/// and sweep-invariance of S, in addition to reconstruction.

#include <gtest/gtest.h>
#include "numhalide_all.h"

#include <cmath>

using namespace numhalide;

namespace {

// Fixture A: M[r][c] = (r==c ? 2.5 + 0.5*r : 0.0) + 0.2*((r*5 + c*3) % 7) - 0.5
// Expected converged singular values (descending), verified against a
// double-precision reference implementing the identical sweep order:
// S ~ [3.8033, 2.4808, 1.8896]
double fixture3(int r, int c)
{
	return (r == c ? 2.5 + 0.5 * r : 0.0) + 0.2 * ((r * 5 + c * 3) % 7) - 0.5;
}

// 4x4 fixture: M[r][c] = (r==c ? 3.0 + 0.5*r : 0.0) + 0.15*((r*5 + c*3) % 9) - 0.6
double fixture4(int r, int c)
{
	return (r == c ? 3.0 + 0.5 * r : 0.0) + 0.15 * ((r * 5 + c * 3) % 9) - 0.6;
}

} // namespace

TEST(SVDJacobi, Convergence3x3) {
	// Buffer convention A(col, row): A(c, r) = M[r][c]
	Halide::Buffer<float> abuf(3, 3);
	for (int r = 0; r < 3; ++r)
		for (int c = 0; c < 3; ++c)
			abuf(c, r) = static_cast<float>(fixture3(r, c));
	Halide::Func af("af_svd3");
	Halide::Var x, y;
	af(x, y) = abuf(x, y);

	const int sweep_counts[] = { 4, 10 };
	double s_prev[3] = { 0, 0, 0 };

	for (int si = 0; si < 2; ++si) {
		int sweeps = sweep_counts[si];
		std::string base = "svd3_sw" + std::to_string(sweeps);
		auto svd = svd_jacobi(af, 3, 3, sweeps, base);

		Halide::Runtime::Buffer<float> U(3, 3);   // U(col=k, row=r)
		Halide::Runtime::Buffer<float> S(3);
		Halide::Runtime::Buffer<float> Vt(3, 3);  // Vt(col=c, row=k)
		svd.U.realize(U);
		svd.S.realize(S);
		svd.Vt.realize(Vt);

		// (a) Orthogonality — THE convergence witness the old sign bug failed.
		// U columns are U(col=k); dot columns i,j over rows: sum_k U(i,k)*U(j,k).
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				double utu = 0.0;
				double vvt = 0.0;
				for (int k = 0; k < 3; ++k) {
					utu += static_cast<double>(U(i, k)) * static_cast<double>(U(j, k));
					vvt += static_cast<double>(Vt(k, i)) * static_cast<double>(Vt(k, j));
				}
				double expected = (i == j) ? 1.0 : 0.0;
				EXPECT_NEAR(utu, expected, 1e-4) << "U^T*U at (" << i << "," << j << ") sweeps=" << sweeps;
				EXPECT_NEAR(vvt, expected, 1e-4) << "Vt*Vt^T at (" << i << "," << j << ") sweeps=" << sweeps;
			}
		}

		// (b) Reconstruction: sum_k U(k,r)*S(k)*Vt(c,k) = A(c,r)
		for (int r = 0; r < 3; ++r) {
			for (int c = 0; c < 3; ++c) {
				double rec = 0.0;
				for (int k = 0; k < 3; ++k)
					rec += static_cast<double>(U(k, r)) * static_cast<double>(S(k)) * static_cast<double>(Vt(c, k));
				EXPECT_NEAR(rec, fixture3(r, c), 1e-3) << "recon at r=" << r << " c=" << c << " sweeps=" << sweeps;
			}
		}

		// (c) Descending order (numpy convention)
		EXPECT_GE(S(0), S(1));
		EXPECT_GE(S(1), S(2));
		EXPECT_GE(S(2), 0.0f);

		// Converged values (hand-verified / double-reference cross-checked)
		EXPECT_NEAR(S(0), 3.8033f, 1e-4f);
		EXPECT_NEAR(S(1), 2.4808f, 1e-4f);
		EXPECT_NEAR(S(2), 1.8896f, 1e-4f);

		// (d) Sweep-invariance: 4 sweeps already converged; 10 must agree.
		// The buggy rotation oscillated, so S depended on sweep parity.
		if (si == 1) {
			for (int k = 0; k < 3; ++k)
				EXPECT_NEAR(static_cast<double>(S(k)), s_prev[k], 1e-4)
					<< "S(" << k << ") changed between sweeps=4 and sweeps=10";
		}
		for (int k = 0; k < 3; ++k)
			s_prev[k] = static_cast<double>(S(k));
	}
}

TEST(SVDJacobi, Convergence3x3_F64) {
	// Same fixture through a Float(64) Func — pins type-genericity of the
	// sweep (make_const-based constants, no f32 hardcoding).
	Halide::Buffer<double> abuf(3, 3);
	for (int r = 0; r < 3; ++r)
		for (int c = 0; c < 3; ++c)
			abuf(c, r) = fixture3(r, c);
	Halide::Func af("af_svd3d");
	Halide::Var x, y;
	af(x, y) = abuf(x, y);

	auto svd = svd_jacobi(af, 3, 3, 10, "svd3_f64");

	Halide::Runtime::Buffer<double> U(3, 3);
	Halide::Runtime::Buffer<double> S(3);
	Halide::Runtime::Buffer<double> Vt(3, 3);
	svd.U.realize(U);
	svd.S.realize(S);
	svd.Vt.realize(Vt);

	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			double utu = 0.0;
			double vvt = 0.0;
			for (int k = 0; k < 3; ++k) {
				utu += U(i, k) * U(j, k);
				vvt += Vt(k, i) * Vt(k, j);
			}
			double expected = (i == j) ? 1.0 : 0.0;
			EXPECT_NEAR(utu, expected, 1e-10) << "U^T*U at (" << i << "," << j << ")";
			EXPECT_NEAR(vvt, expected, 1e-10) << "Vt*Vt^T at (" << i << "," << j << ")";
		}
	}

	for (int r = 0; r < 3; ++r) {
		for (int c = 0; c < 3; ++c) {
			double rec = 0.0;
			for (int k = 0; k < 3; ++k)
				rec += U(k, r) * S(k) * Vt(c, k);
			EXPECT_NEAR(rec, fixture3(r, c), 1e-10) << "recon at r=" << r << " c=" << c;
		}
	}

	EXPECT_GE(S(0), S(1));
	EXPECT_GE(S(1), S(2));
	EXPECT_GE(S(2), 0.0);

	// Derived from a double-precision reference mirroring the source sweep
	// order exactly (not from the spec's 4-digit values): the f64 pipeline
	// should match the reference to rounding.
	EXPECT_NEAR(S(0), 3.8032704372057573, 1e-8);
	EXPECT_NEAR(S(1), 2.4808207726750653, 1e-8);
	EXPECT_NEAR(S(2), 1.8896196112817556, 1e-8);
}

TEST(SVDJacobi, Convergence4x4) {
	Halide::Buffer<float> abuf(4, 4);
	for (int r = 0; r < 4; ++r)
		for (int c = 0; c < 4; ++c)
			abuf(c, r) = static_cast<float>(fixture4(r, c));
	Halide::Func af("af_svd4");
	Halide::Var x, y;
	af(x, y) = abuf(x, y);

	auto svd = svd_jacobi(af, 4, 4, 6, "svd4");

	Halide::Runtime::Buffer<float> U(4, 4);
	Halide::Runtime::Buffer<float> S(4);
	Halide::Runtime::Buffer<float> Vt(4, 4);
	svd.U.realize(U);
	svd.S.realize(S);
	svd.Vt.realize(Vt);

	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			double utu = 0.0;
			double vvt = 0.0;
			for (int k = 0; k < 4; ++k) {
				utu += static_cast<double>(U(i, k)) * static_cast<double>(U(j, k));
				vvt += static_cast<double>(Vt(k, i)) * static_cast<double>(Vt(k, j));
			}
			double expected = (i == j) ? 1.0 : 0.0;
			EXPECT_NEAR(utu, expected, 1e-6) << "U^T*U at (" << i << "," << j << ")";
			EXPECT_NEAR(vvt, expected, 1e-6) << "Vt*Vt^T at (" << i << "," << j << ")";
		}
	}

	for (int r = 0; r < 4; ++r) {
		for (int c = 0; c < 4; ++c) {
			double rec = 0.0;
			for (int k = 0; k < 4; ++k)
				rec += static_cast<double>(U(k, r)) * static_cast<double>(S(k)) * static_cast<double>(Vt(c, k));
			EXPECT_NEAR(rec, fixture4(r, c), 1e-3) << "recon at r=" << r << " c=" << c;
		}
	}

	EXPECT_GE(S(0), S(1));
	EXPECT_GE(S(1), S(2));
	EXPECT_GE(S(2), S(3));
	EXPECT_GE(S(3), 0.0f);
}

/// SWEEPS NEEDED GROW WITH n, and under-convergence is SILENT: reconstruction
/// A ~ U S Vt stays accurate while U and Vt lose orthogonality, so a test that
/// checks only reconstruction passes on a factorization whose factors are not
/// orthogonal.
///
/// This is how a caller shipped 4x4 SVD at three sweeps with ||U^T U - I|| at
/// 1.4e-4 while every other forward in that package verified at ~1e-7. It went
/// unnoticed because the reconstruction check it had was fine.
///
/// SWEEPS SUFFICIENT FOR ONE MATRIX ARE NOT SUFFICIENT FOR ALL. On this file's
/// own fixture4, three sweeps already reaches 7.7e-8 -- so a test built on that
/// matrix reports convergence that other inputs do not get. Measured across
/// several matrices, three sweeps reaches only ~1e-4 on the worst of them while
/// six stays at ~1e-7.
///
/// So this sweeps a SPREAD of deterministic matrices and asserts the thing that
/// has to hold for a caller choosing a sweep count: six sweeps delivers
/// f32-grade orthogonality on ALL of them. The three-sweep figure is reported
/// rather than asserted, because how bad it gets is a property of the input.
TEST(SVDJacobi, SixSweepsOrthogonalOnHarderInputs4x4) {
	// A small deterministic LCG, so the fixtures vary without depending on the
	// platform's random implementation.
	auto orthogonality_error = [](int sweeps) -> double {
		double worst_all = 0.0;
		uint32_t seed = 12345u;
		auto next = [&seed]() -> float {
			seed = seed * 1664525u + 1013904223u;
			return static_cast<float>((seed >> 8) & 0xFFFF) / 32768.0f - 1.0f;
		};

		for (int trial = 0; trial < 6; ++trial) {
			Halide::Buffer<float> abuf(4, 4);
			for (int r = 0; r < 4; ++r)
				for (int c = 0; c < 4; ++c)
					abuf(c, r) = next();
			// Diagonally dominant: well conditioned, so what is measured is
			// convergence rather than the conditioning of a near-singular draw.
			for (int r = 0; r < 4; ++r) {
				float row = 0.0f;
				for (int c = 0; c < 4; ++c)
					if (c != r) row += std::abs(abuf(c, r));
				abuf(r, r) = row + 1.0f + 0.5f * static_cast<float>(r);
			}

		Halide::Func af("sw_af" + std::to_string(sweeps) + "_" +
		                std::to_string(trial));
		Halide::Var x, y;
		af(x, y) = abuf(x, y);

		auto svd = svd_jacobi(af, 4, 4, sweeps,
		                      "svd4_sw" + std::to_string(sweeps) + "_" +
		                          std::to_string(trial));
		Halide::Buffer<float> U = svd.U.realize({4, 4});

			for (int i = 0; i < 4; ++i) {
				for (int j = 0; j < 4; ++j) {
					double utu = 0.0;
					for (int k = 0; k < 4; ++k)
						utu += static_cast<double>(U(i, k)) *
						       static_cast<double>(U(j, k));
					double const expected = (i == j) ? 1.0 : 0.0;
					worst_all = std::max(worst_all, std::abs(utu - expected));
				}
			}
		}
		return worst_all;
	};

	double const at3 = orthogonality_error(3);
	double const at6 = orthogonality_error(6);

	std::cout << "[sweeps] worst ||U^T U - I|| over 6 matrices: 3 sweeps "
	          << at3 << ", 6 sweeps " << at6 << std::endl;

	// The only assertion, because it is the only part that must hold for every
	// input: six sweeps is enough. How far short three falls depends on the
	// matrix, so it is reported and not asserted.
	EXPECT_LT(at6, 1e-5)
		<< "six sweeps must reach f32-grade orthogonality at n=4 on every one "
		   "of these matrices";
}
