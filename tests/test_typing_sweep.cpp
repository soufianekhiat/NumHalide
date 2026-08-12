/// @file test_typing_sweep.cpp
/// @brief Pins for the f32/f64 typing sweep.
///
/// Every fixed site gets (a) an f64 pin — output type Float(64) where
/// applicable plus a value check at a tolerance only full f64 math can
/// meet (a value that needs more than f32 precision to distinguish, or an
/// accumulation an f32 path provably loses) — and (b) an f32 regression
/// pin (historical behavior preserved: the fixed forms fold to the same
/// f32 constants and seeds as the old literals).
///
/// The big/small accumulation fixture used repeatedly: a vector holding
/// one 1e7 plus seventeen 1.0 entries. Sum of squares = 1e14 + 17: exact
/// in f64 (1e14 < 2^53), while an f32 accumulator loses the +17 entirely
/// (f32 ulp at 1e14 is ~8.4e6) AND rounds 1e14 itself.

#include <gtest/gtest.h>
#include "numhalide_all.h"

#include <cmath>
#include <limits>

using namespace numhalide;

namespace {

double const kPi = 3.14159265358979323846;

Halide::Func wrap1d_f64(Halide::Buffer<double> const& buf, const std::string& n)
{
	Halide::Func f(n);
	Halide::Var x;
	f(x) = buf(x);
	return f;
}

Halide::Func wrap2d_f64(Halide::Buffer<double> const& buf, const std::string& n)
{
	Halide::Func f(n);
	Halide::Var x, y;
	f(x, y) = buf(x, y);
	return f;
}

Halide::Func wrap1d_f32(Halide::Buffer<float> const& buf, const std::string& n)
{
	Halide::Func f(n);
	Halide::Var x;
	f(x) = buf(x);
	return f;
}

// Real f64 signal -> complex Tuple Func (imag = 0).
Halide::Func cplx_f64(Halide::Buffer<double> const& re, const std::string& n)
{
	Halide::Func f(n);
	Halide::Var x;
	f(x) = Halide::Tuple(re(x), Halide::Internal::make_zero(Halide::Float(64)));
	return f;
}

Halide::Func cplx_f32(Halide::Buffer<float> const& re, const std::string& n)
{
	Halide::Func f(n);
	Halide::Var x;
	f(x) = Halide::Tuple(re(x), Halide::Internal::make_zero(Halide::Float(32)));
	return f;
}

Halide::Func part1d(Halide::Func c, int idx, const std::string& n)
{
	Halide::Func r(n);
	Halide::Var x;
	r(x) = c(x)[idx];
	return r;
}

// Big/small accumulation vector: [1e7, 1, 1, ..., 1] (17 ones), n = 18.
int const kAccN = 18;
double const kAccSumSq = 1.0e14 + 17.0;

Halide::Buffer<double> acc_vec_f64()
{
	Halide::Buffer<double> b(kAccN);
	b(0) = 1.0e7;
	for (int i = 1; i < kAccN; ++i) b(i) = 1.0;
	return b;
}

} // namespace

// ---------------------------------------------------------------------------
// la.h — svd_jacobi indicator skip guard (priority fix)
// ---------------------------------------------------------------------------

TEST(TypingSweep, SvdJacobiF64TypesAndReconstruction) {
	// Asymmetric well-conditioned 3x3 (the svd-study fixture). At f64 the
	// reconstruction U*diag(S)*Vt must hit ~1e-12; an f32 pipeline lands
	// around 1e-6 and fails this pin.
	Halide::Buffer<double> abuf(3, 3);
	for (int r = 0; r < 3; ++r)
		for (int c = 0; c < 3; ++c)
			abuf(c, r) = (r == c ? 2.5 + 0.5 * r : 0.0)
			           + 0.2 * ((r * 5 + c * 3) % 7) - 0.5;
	Halide::Func af = wrap2d_f64(abuf, "ts_svd_a64");

	auto svd = svd_jacobi(af, 3, 3, 10, "ts_svd64");
	ASSERT_EQ(svd.U.types()[0], Halide::Float(64));
	ASSERT_EQ(svd.S.types()[0], Halide::Float(64));
	ASSERT_EQ(svd.Vt.types()[0], Halide::Float(64));

	svd.U.compute_root();
	svd.S.compute_root();
	svd.Vt.compute_root();

	// R(c, r) = sum_k U(k, r) * S(k) * Vt(c, k)
	Halide::Func recon("ts_svd64_recon");
	Halide::Var x, y;
	Halide::RDom k(0, 3);
	recon(x, y) = Halide::Internal::make_zero(Halide::Float(64));
	recon(x, y) += svd.U(k, y) * svd.S(k) * svd.Vt(x, k);

	Halide::Buffer<double> out(3, 3);
	recon.realize(out);
	for (int r = 0; r < 3; ++r)
		for (int c = 0; c < 3; ++c)
			EXPECT_NEAR(out(c, r), abuf(c, r), 1e-9)
				<< "recon(" << c << "," << r << ")";
}

TEST(TypingSweep, SvdJacobiSkipGuardExactIdentityF32AndF64) {
	// diag(3, 2, 1): every column pair has gamma = 0, so every rotation
	// takes the skip path. The indicator form must produce the EXACT
	// identity rotation (cs = 1, sn = 0): S = [3, 2, 1] bit-exact, U = I,
	// Vt = I. This pins value-identity of the select -> indicator rewrite.
	{
		Halide::Buffer<float> abuf(3, 3);
		abuf.fill(0.0f);
		abuf(0, 0) = 3.0f; abuf(1, 1) = 2.0f; abuf(2, 2) = 1.0f;
		Halide::Func af("ts_svd_skip32_a");
		Halide::Var x, y;
		af(x, y) = abuf(x, y);

		auto svd = svd_jacobi(af, 3, 3, 4, "ts_svd_skip32");
		Halide::Buffer<float> s(3);
		svd.S.realize(s);
		EXPECT_EQ(s(0), 3.0f);
		EXPECT_EQ(s(1), 2.0f);
		EXPECT_EQ(s(2), 1.0f);

		Halide::Buffer<float> u(3, 3), vt(3, 3);
		svd.U.realize(u);
		svd.Vt.realize(vt);
		for (int r = 0; r < 3; ++r)
			for (int c = 0; c < 3; ++c) {
				EXPECT_EQ(u(c, r), (r == c) ? 1.0f : 0.0f) << "U(" << c << "," << r << ")";
				EXPECT_EQ(vt(c, r), (r == c) ? 1.0f : 0.0f) << "Vt(" << c << "," << r << ")";
			}
	}
	{
		Halide::Buffer<double> abuf(3, 3);
		abuf.fill(0.0);
		abuf(0, 0) = 3.0; abuf(1, 1) = 2.0; abuf(2, 2) = 1.0;
		Halide::Func af = wrap2d_f64(abuf, "ts_svd_skip64_a");

		auto svd = svd_jacobi(af, 3, 3, 4, "ts_svd_skip64");
		Halide::Buffer<double> s(3);
		svd.S.realize(s);
		EXPECT_EQ(s(0), 3.0);
		EXPECT_EQ(s(1), 2.0);
		EXPECT_EQ(s(2), 1.0);
	}
}

TEST(TypingSweep, MatrixRankF64TypeAndValue) {
	// diag(5, 3, 0): rank 2. Output type follows A (was hardcoded f32).
	Halide::Buffer<double> abuf(3, 3);
	abuf.fill(0.0);
	abuf(0, 0) = 5.0; abuf(1, 1) = 3.0;
	Halide::Func af = wrap2d_f64(abuf, "ts_mr_a");

	auto r = matrix_rank(af, 3, 3, 1e-10f, 4, "ts_mr");
	ASSERT_EQ(r.types()[0], Halide::Float(64));
	Halide::Buffer<double> out = Halide::Buffer<double>::make_scalar();
	r.realize(out);
	EXPECT_NEAR(out(), 2.0, 1e-12);
}

TEST(TypingSweep, DetLuF64KeepsSubUlpF32Perturbation) {
	// det(diag(1 + 1e-12, 1)) = 1 + 1e-12. The old cast<float> working
	// matrix flushed the perturbation (f32 ulp at 1 is 1.2e-7).
	Halide::Buffer<double> abuf(2, 2);
	abuf(0, 0) = 1.0 + 1e-12; abuf(1, 0) = 0.0;
	abuf(0, 1) = 0.0;         abuf(1, 1) = 1.0;
	Halide::Func af = wrap2d_f64(abuf, "ts_det_a");

	auto d = det_lu(af, 2, "ts_det");
	ASSERT_EQ(d.types()[0], Halide::Float(64));
	Halide::Buffer<double> out = Halide::Buffer<double>::make_scalar();
	d.realize(out);
	EXPECT_NEAR(out() - 1.0, 1e-12, 1e-14);
}

TEST(TypingSweep, SlogdetF64TypeAndLogPrecision) {
	// diag(2, 3): sign = +1, log|det| = ln 6. 1e-12 needs the f64 path
	// end-to-end (an f32 log is only good to ~1e-7).
	Halide::Buffer<double> abuf(2, 2);
	abuf(0, 0) = 2.0; abuf(1, 0) = 0.0;
	abuf(0, 1) = 0.0; abuf(1, 1) = 3.0;
	Halide::Func af = wrap2d_f64(abuf, "ts_sld_a");

	auto sld = slogdet(af, 2, "ts_sld");
	ASSERT_EQ(sld.sign.types()[0], Halide::Float(64));
	ASSERT_EQ(sld.logabsdet.types()[0], Halide::Float(64));
	Halide::Buffer<double> sg = Halide::Buffer<double>::make_scalar();
	Halide::Buffer<double> ld = Halide::Buffer<double>::make_scalar();
	sld.sign.realize(sg);
	sld.logabsdet.realize(ld);
	EXPECT_EQ(sg(), 1.0);
	EXPECT_NEAR(ld(), std::log(6.0), 1e-12);
}

TEST(TypingSweep, NormAndFrobeniusF64Accumulation) {
	// ||[1e7, 1 x17]|| = sqrt(1e14 + 17): the +17 is invisible to an f32
	// accumulator. 1e-6 absolute at 1e7 magnitude requires f64.
	Halide::Buffer<double> vbuf = acc_vec_f64();
	Halide::Func vf = wrap1d_f64(vbuf, "ts_norm_v");

	shape_t vs = { kAccN };
	auto nrm = norm(vf, vs, "ts_norm");
	ASSERT_EQ(nrm.types()[0], Halide::Float(64));
	Halide::Buffer<double> out(1);
	nrm.realize(out);
	EXPECT_NEAR(out(0), std::sqrt(kAccSumSq), 1e-6);

	// Frobenius: same trick in 2D — [[1e7, 1], [1, 1]], sumsq = 1e14 + 3.
	Halide::Buffer<double> mbuf(2, 2);
	mbuf(0, 0) = 1.0e7; mbuf(1, 0) = 1.0;
	mbuf(0, 1) = 1.0;   mbuf(1, 1) = 1.0;
	Halide::Func mf = wrap2d_f64(mbuf, "ts_frob_m");
	shape_t ms = { 2, 2 };
	auto fro = frobenius_norm(mf, ms, "ts_frob");
	ASSERT_EQ(fro.types()[0], Halide::Float(64));
	Halide::Buffer<double> fout(1);
	fro.realize(fout);
	EXPECT_NEAR(fout(0), std::sqrt(1.0e14 + 3.0), 1e-6);
}

TEST(TypingSweep, NormF32Regression) {
	// 3-4-5 stays 3-4-5 at f32 (historical behavior preserved).
	Halide::Buffer<float> vbuf(2);
	vbuf(0) = 3.0f; vbuf(1) = 4.0f;
	Halide::Func vf = wrap1d_f32(vbuf, "ts_norm32_v");
	shape_t vs = { 2 };
	auto nrm = norm(vf, vs, "ts_norm32");
	ASSERT_EQ(nrm.types()[0], Halide::Float(32));
	Halide::Buffer<float> out(1);
	nrm.realize(out);
	EXPECT_NEAR(out(0), 5.0f, 1e-5f);
}

// ---------------------------------------------------------------------------
// la_batched.h — batched SVD typed + indicator skip guard
// ---------------------------------------------------------------------------

TEST(TypingSweep, BatchedSvdF64TypesAndReconstruction) {
	// Batch of two 2x2 matrices; f64 reconstruction at 1e-9 (f32 ~1e-6).
	Halide::Buffer<double> abuf(2, 2, 2);
	double const vals[2][2][2] = { { {2.0, 0.5}, {0.5, 1.0} },
	                               { {1.5, -0.3}, {0.7, 2.5} } };
	for (int b = 0; b < 2; ++b)
		for (int r = 0; r < 2; ++r)
			for (int c = 0; c < 2; ++c)
				abuf(c, r, b) = vals[b][r][c];
	Halide::Func af("ts_bsvd_a");
	Halide::Var x, y, z;
	af(x, y, z) = abuf(x, y, z);

	auto svd = batched_svd_jacobi(af, 2, 2, 2, 10, "ts_bsvd");
	ASSERT_EQ(svd.U.types()[0], Halide::Float(64));
	ASSERT_EQ(svd.S.types()[0], Halide::Float(64));
	ASSERT_EQ(svd.Vt.types()[0], Halide::Float(64));

	svd.U.compute_root();
	svd.S.compute_root();
	svd.Vt.compute_root();

	Halide::Func recon("ts_bsvd_recon");
	Halide::RDom k(0, 2);
	recon(x, y, z) = Halide::Internal::make_zero(Halide::Float(64));
	recon(x, y, z) += svd.U(k, y, z) * svd.S(k, z) * svd.Vt(x, k, z);

	Halide::Buffer<double> out(2, 2, 2);
	recon.realize(out);
	for (int b = 0; b < 2; ++b)
		for (int r = 0; r < 2; ++r)
			for (int c = 0; c < 2; ++c)
				EXPECT_NEAR(out(c, r, b), abuf(c, r, b), 1e-9)
					<< "batch " << b << " (" << c << "," << r << ")";
}

TEST(TypingSweep, BatchedSvdF32Regression) {
	// Same fixture at f32: types stay f32 and the reconstruction still
	// meets the historical tolerance (indicator rewrite is value-identical).
	Halide::Buffer<float> abuf(2, 2, 2);
	float const vals[2][2][2] = { { {2.0f, 0.5f}, {0.5f, 1.0f} },
	                              { {1.5f, -0.3f}, {0.7f, 2.5f} } };
	for (int b = 0; b < 2; ++b)
		for (int r = 0; r < 2; ++r)
			for (int c = 0; c < 2; ++c)
				abuf(c, r, b) = vals[b][r][c];
	Halide::Func af("ts_bsvd32_a");
	Halide::Var x, y, z;
	af(x, y, z) = abuf(x, y, z);

	auto svd = batched_svd_jacobi(af, 2, 2, 2, 10, "ts_bsvd32");
	ASSERT_EQ(svd.U.types()[0], Halide::Float(32));

	svd.U.compute_root();
	svd.S.compute_root();
	svd.Vt.compute_root();

	Halide::Func recon("ts_bsvd32_recon");
	Halide::RDom k(0, 2);
	recon(x, y, z) = 0.0f;
	recon(x, y, z) += svd.U(k, y, z) * svd.S(k, z) * svd.Vt(x, k, z);

	Halide::Buffer<float> out(2, 2, 2);
	recon.realize(out);
	for (int b = 0; b < 2; ++b)
		for (int r = 0; r < 2; ++r)
			for (int c = 0; c < 2; ++c)
				EXPECT_NEAR(out(c, r, b), abuf(c, r, b), 1e-4f);
}

// ---------------------------------------------------------------------------
// fft.h — typed twiddles and normalization
// ---------------------------------------------------------------------------

TEST(TypingSweep, Fft1dF64TwiddlePrecision) {
	// x[n] = cos(2 pi n / 8): X[1] = X[7] = 4, all other bins 0. Hitting
	// 1e-12 needs f64 twiddles (f32 twiddles leave ~1e-6 residue).
	int const N = 8;
	Halide::Buffer<double> re(N);
	for (int n = 0; n < N; ++n) re(n) = std::cos(2.0 * kPi * n / N);
	Halide::Func cin = cplx_f64(re, "ts_fft64_in");

	auto X = fft_1d_c2c(cin, N, -1, "ts_fft64");
	ASSERT_EQ(X.types()[0], Halide::Float(64));
	ASSERT_EQ(X.types()[1], Halide::Float(64));

	Halide::Buffer<double> xr(N), xi(N);
	part1d(X, 0, "ts_fft64_re").realize(xr);
	part1d(X, 1, "ts_fft64_im").realize(xi);

	for (int k = 0; k < N; ++k) {
		double expect_re = (k == 1 || k == 7) ? 4.0 : 0.0;
		EXPECT_NEAR(xr(k), expect_re, 1e-12) << "re bin " << k;
		EXPECT_NEAR(xi(k), 0.0, 1e-12) << "im bin " << k;
	}
}

TEST(TypingSweep, Fft1dF32Regression) {
	// [1,2,3,4]: X = [10, -2+2j, -2, -2-2j] at the historical tolerance.
	int const N = 4;
	Halide::Buffer<float> re(N);
	for (int n = 0; n < N; ++n) re(n) = float(n + 1);
	Halide::Func cin = cplx_f32(re, "ts_fft32_in");

	auto X = fft_1d_c2c(cin, N, -1, "ts_fft32");
	ASSERT_EQ(X.types()[0], Halide::Float(32));

	Halide::Buffer<float> xr(N), xi(N);
	part1d(X, 0, "ts_fft32_re").realize(xr);
	part1d(X, 1, "ts_fft32_im").realize(xi);

	float const er[4] = { 10.0f, -2.0f, -2.0f, -2.0f };
	float const ei[4] = { 0.0f, 2.0f, 0.0f, -2.0f };
	for (int k = 0; k < N; ++k) {
		EXPECT_NEAR(xr(k), er[k], 1e-4f) << "re bin " << k;
		EXPECT_NEAR(xi(k), ei[k], 1e-4f) << "im bin " << k;
	}
}

TEST(TypingSweep, IfftNormalizedF64RoundTrip) {
	int const N = 8;
	Halide::Buffer<double> re(N);
	for (int n = 0; n < N; ++n) re(n) = 0.25 * n * n - 1.5 * n + 0.75;
	Halide::Func cin = cplx_f64(re, "ts_ifft64_in");

	auto fwd = fft_1d_c2c(cin, N, -1, "ts_ifft64_f");
	auto inv = ifft_normalized(fwd, N, "ts_ifft64_i");
	ASSERT_EQ(inv.types()[0], Halide::Float(64));

	Halide::Buffer<double> rr(N), ri(N);
	part1d(inv, 0, "ts_ifft64_re").realize(rr);
	part1d(inv, 1, "ts_ifft64_im").realize(ri);
	for (int n = 0; n < N; ++n) {
		EXPECT_NEAR(rr(n), re(n), 1e-12) << "roundtrip re " << n;
		EXPECT_NEAR(ri(n), 0.0, 1e-12) << "roundtrip im " << n;
	}
}

TEST(TypingSweep, Fft2dF64TypeAndRoundTrip) {
	int const R = 4, C = 4;
	Halide::Buffer<double> re(C, R);
	for (int r = 0; r < R; ++r)
		for (int c = 0; c < C; ++c)
			re(c, r) = 0.1 * (r * C + c) - 0.35;
	Halide::Func cin("ts_fft2d64_in");
	Halide::Var x, y;
	cin(x, y) = Halide::Tuple(re(x, y), Halide::Internal::make_zero(Halide::Float(64)));

	auto fwd = fft_2d_c2c(cin, R, C, -1, "ts_fft2d64_f");
	ASSERT_EQ(fwd.types()[0], Halide::Float(64));
	auto inv = ifft2d_normalized(fwd, R, C, "ts_fft2d64_i");
	ASSERT_EQ(inv.types()[0], Halide::Float(64));

	Halide::Func rr("ts_fft2d64_rr");
	rr(x, y) = inv(x, y)[0];
	Halide::Buffer<double> out(C, R);
	rr.realize(out);
	for (int r = 0; r < R; ++r)
		for (int c = 0; c < C; ++c)
			EXPECT_NEAR(out(c, r), re(c, r), 1e-12) << "(" << c << "," << r << ")";
}

// ---------------------------------------------------------------------------
// fft_fast.h — typed Cooley-Tukey
// ---------------------------------------------------------------------------

TEST(TypingSweep, FftFastF64MatchesDirectDft) {
	int const N = 8;
	Halide::Buffer<double> re(N);
	for (int n = 0; n < N; ++n) re(n) = std::sin(0.7 * n) + 0.2 * n;
	Halide::Func cin = cplx_f64(re, "ts_ff64_in");

	auto fast = fft_fast(cin, N, "ts_ff64_fast");
	ASSERT_EQ(fast.types()[0], Halide::Float(64));
	auto direct = fft_1d_c2c(cin, N, -1, "ts_ff64_direct");

	Halide::Buffer<double> fr(N), fi(N), dr(N), di(N);
	part1d(fast, 0, "ts_ff64_fr").realize(fr);
	part1d(fast, 1, "ts_ff64_fi").realize(fi);
	part1d(direct, 0, "ts_ff64_dr").realize(dr);
	part1d(direct, 1, "ts_ff64_di").realize(di);
	for (int k = 0; k < N; ++k) {
		EXPECT_NEAR(fr(k), dr(k), 1e-12) << "re bin " << k;
		EXPECT_NEAR(fi(k), di(k), 1e-12) << "im bin " << k;
	}

	// Normalized inverse roundtrip, also at f64 tolerance.
	auto inv = ifft_fast(fast, N, "ts_ff64_inv");
	ASSERT_EQ(inv.types()[0], Halide::Float(64));
	Halide::Buffer<double> rr(N), ri(N);
	part1d(inv, 0, "ts_ff64_rr").realize(rr);
	part1d(inv, 1, "ts_ff64_ri").realize(ri);
	for (int n = 0; n < N; ++n) {
		EXPECT_NEAR(rr(n), re(n), 1e-12) << "roundtrip re " << n;
		EXPECT_NEAR(ri(n), 0.0, 1e-12) << "roundtrip im " << n;
	}
}

// ---------------------------------------------------------------------------
// rfft.h — legacy fftfreq/rfftfreq delegate at Float(32)
// ---------------------------------------------------------------------------

TEST(TypingSweep, FftfreqLegacyDelegationF32Pins) {
	// numpy fftfreq(4) = [0, 0.25, -0.5, -0.25]; type stays Float(32).
	{
		auto f = fftfreq(4);
		ASSERT_EQ(f.types()[0], Halide::Float(32));
		Halide::Buffer<float> out(4);
		f.realize(out);
		EXPECT_FLOAT_EQ(out(0), 0.0f);
		EXPECT_FLOAT_EQ(out(1), 0.25f);
		EXPECT_FLOAT_EQ(out(2), -0.5f);
		EXPECT_FLOAT_EQ(out(3), -0.25f);
	}
	// Odd size with spacing: numpy fftfreq(5, 0.5) = [0, .4, .8, -.8, -.4].
	{
		auto f = fftfreq(5, 0.5f, "ts_ff5");
		Halide::Buffer<float> out(5);
		f.realize(out);
		float const e[5] = { 0.0f, 0.4f, 0.8f, -0.8f, -0.4f };
		for (int i = 0; i < 5; ++i) EXPECT_NEAR(out(i), e[i], 1e-6f) << i;
	}
	// numpy rfftfreq(5, 0.5) = [0, 0.4, 0.8].
	{
		auto f = rfftfreq(5, 0.5f, "ts_rf5");
		ASSERT_EQ(f.types()[0], Halide::Float(32));
		Halide::Buffer<float> out(3);
		f.realize(out);
		float const e[3] = { 0.0f, 0.4f, 0.8f };
		for (int i = 0; i < 3; ++i) EXPECT_NEAR(out(i), e[i], 1e-6f) << i;
	}
}

// ---------------------------------------------------------------------------
// window.h — typed windows + legacy kaiser delegation
// ---------------------------------------------------------------------------

TEST(TypingSweep, WindowsTypedF64Values) {
	// Value pins at 1e-12 against double references — the old widened-f32
	// pi constants miss by ~1e-8.
	int const N = 8;
	{
		auto w = hanning(Halide::Expr(N), Halide::Float(64), "ts_han64");
		ASSERT_EQ(w.types()[0], Halide::Float(64));
		Halide::Buffer<double> out(N);
		w.realize(out);
		for (int i = 0; i < N; ++i) {
			double e = 0.5 * (1.0 - std::cos(2.0 * kPi * i / (N - 1)));
			EXPECT_NEAR(out(i), e, 1e-12) << "hanning " << i;
		}
	}
	{
		auto w = hamming(Halide::Expr(N), Halide::Float(64), "ts_ham64");
		ASSERT_EQ(w.types()[0], Halide::Float(64));
		Halide::Buffer<double> out(N);
		w.realize(out);
		for (int i = 0; i < N; ++i) {
			double e = 0.54 - 0.46 * std::cos(2.0 * kPi * i / (N - 1));
			EXPECT_NEAR(out(i), e, 1e-12) << "hamming " << i;
		}
	}
	{
		auto w = blackman(Halide::Expr(N), Halide::Float(64), "ts_bla64");
		ASSERT_EQ(w.types()[0], Halide::Float(64));
		Halide::Buffer<double> out(N);
		w.realize(out);
		for (int i = 0; i < N; ++i) {
			double e = 0.42 - 0.5 * std::cos(2.0 * kPi * i / (N - 1))
			         + 0.08 * std::cos(4.0 * kPi * i / (N - 1));
			EXPECT_NEAR(out(i), e, 1e-12) << "blackman " << i;
		}
	}
	{
		auto w = bartlett(Halide::Expr(N), Halide::Float(64), "ts_bar64");
		ASSERT_EQ(w.types()[0], Halide::Float(64));
		Halide::Buffer<double> out(N);
		w.realize(out);
		for (int i = 0; i < N; ++i) {
			double e = 1.0 - std::abs(2.0 * i / double(N - 1) - 1.0);
			EXPECT_NEAR(out(i), e, 1e-12) << "bartlett " << i;
		}
	}
}

TEST(TypingSweep, WindowsLegacyF32Regression) {
	// Legacy (type-less) forms delegate at Float(32) — endpoints, symmetry
	// and peaks stay at their historical values.
	int const N = 16;
	{
		auto w = hanning(N, "ts_han32");
		ASSERT_EQ(w.types()[0], Halide::Float(32));
		Halide::Buffer<float> out(N);
		w.realize(out);
		EXPECT_NEAR(out(0), 0.0f, 1e-5f);
		EXPECT_NEAR(out(N - 1), 0.0f, 1e-5f);
		for (int i = 0; i < N / 2; ++i)
			EXPECT_NEAR(out(i), out(N - 1 - i), 1e-5f);
	}
	{
		auto w = hamming(N, "ts_ham32");
		Halide::Buffer<float> out(N);
		w.realize(out);
		EXPECT_NEAR(out(0), 0.08f, 1e-3f);
	}
	{
		auto w = bartlett(5, "ts_bar32");
		Halide::Buffer<float> out(5);
		w.realize(out);
		EXPECT_NEAR(out(0), 0.0f, 1e-5f);
		EXPECT_NEAR(out(1), 0.5f, 1e-5f);
		EXPECT_NEAR(out(2), 1.0f, 1e-5f);
	}
}

TEST(TypingSweep, KaiserLegacyNowMatchesBesselTruth) {
	// The legacy kaiser delegates to the A&S runtime form at Float(32).
	// These are the CONVERGED-series numpy.kaiser pins the old 6-term
	// truncated body could not meet at beta = 8.6 (documented accuracy
	// improvement; no NumHalide test pinned the old truncated values).
	{
		auto w = kaiser(5, 2.0f, "ts_k5");
		ASSERT_EQ(w.types()[0], Halide::Float(32));
		Halide::Buffer<float> out(5);
		w.realize(out);
		double const e[5] = { 0.438676280, 0.834761433, 1.0,
		                      0.834761433, 0.438676280 };
		for (int i = 0; i < 5; ++i)
			EXPECT_NEAR(out(i), e[i], 1e-6) << "kaiser5 " << i;
	}
	{
		auto w = kaiser(9, 8.6f, "ts_k9");
		Halide::Buffer<float> out(9);
		w.realize(out);
		double const e[9] = { 0.001332514, 0.067472079, 0.340393622,
		                      0.773829381, 1.0,         0.773829381,
		                      0.340393622, 0.067472079, 0.001332514 };
		for (int i = 0; i < 9; ++i)
			EXPECT_NEAR(out(i), e[i], 2e-6) << "kaiser9 " << i;
	}
}

// ---------------------------------------------------------------------------
// polynomial.h — compile-time polyval seed, degree-0 arms, Vandermonde
// ---------------------------------------------------------------------------

TEST(TypingSweep, PolyvalCompileTimeF64Seed) {
	// coeffs = [1 + 1e-12, 0], x = 1: result = 1 + 1e-12. The old
	// cast<float> seed flushed the perturbation to exactly 1.0.
	Halide::Buffer<double> cbuf(2);
	cbuf(0) = 1.0 + 1e-12; cbuf(1) = 0.0;
	Halide::Func cf = wrap1d_f64(cbuf, "ts_pv_c");

	Halide::Buffer<double> xbuf(1);
	xbuf(0) = 1.0;
	Halide::Func xf = wrap1d_f64(xbuf, "ts_pv_x");

	shape_t s = { 1 };
	auto r = polyval(cf, 2, xf, s, "ts_pv");
	ASSERT_EQ(r.types()[0], Halide::Float(64));
	Halide::Buffer<double> out(1);
	r.realize(out);
	EXPECT_NEAR(out(0) - 1.0, 1e-12, 1e-14);
}

TEST(TypingSweep, PolyvalCompileTimeF32Regression) {
	// 2x^2 + 3x + 4 at x = 2 -> 18 (historical f32 behavior).
	Halide::Buffer<float> cbuf(3);
	cbuf(0) = 2.0f; cbuf(1) = 3.0f; cbuf(2) = 4.0f;
	Halide::Func cf = wrap1d_f32(cbuf, "ts_pv32_c");
	Halide::Buffer<float> xbuf(1);
	xbuf(0) = 2.0f;
	Halide::Func xf = wrap1d_f32(xbuf, "ts_pv32_x");
	shape_t s = { 1 };
	auto r = polyval(cf, 3, xf, s, "ts_pv32");
	ASSERT_EQ(r.types()[0], Halide::Float(32));
	Halide::Buffer<float> out(1);
	r.realize(out);
	EXPECT_NEAR(out(0), 18.0f, 1e-4f);
}

TEST(TypingSweep, ChebyshevLegendreDegree0F64Type) {
	// The n == 0 arms used to emit a bare 1.0f (f32 output on f64 input).
	Halide::Buffer<double> xbuf(2);
	xbuf(0) = 0.25; xbuf(1) = -0.5;
	Halide::Func xf = wrap1d_f64(xbuf, "ts_cl_x");
	shape_t s = { 2 };

	auto c0 = chebyshev_t(0, xf, s, "ts_cheb0");
	ASSERT_EQ(c0.types()[0], Halide::Float(64));
	auto l0 = legendre_p(0, xf, s, "ts_leg0");
	ASSERT_EQ(l0.types()[0], Halide::Float(64));

	Halide::Buffer<double> out(2);
	c0.realize(out);
	EXPECT_EQ(out(0), 1.0);
	l0.realize(out);
	EXPECT_EQ(out(1), 1.0);
}

// ---------------------------------------------------------------------------
// einsum.h — typed accumulator seeds
// ---------------------------------------------------------------------------

TEST(TypingSweep, EinsumDotF64Accumulation) {
	// "i,i->" of the big/small vector with itself = 1e14 + 17 — the +17 is
	// invisible to the old f32 seed/accumulator (which was also a type
	// error for f64 updates).
	Halide::Buffer<double> vbuf = acc_vec_f64();
	Halide::Func vf = wrap1d_f64(vbuf, "ts_es_v");
	shape_t s = { kAccN };

	auto r = einsum("i,i->", vf, s, vf, s, "ts_es");
	ASSERT_EQ(r.types()[0], Halide::Float(64));
	Halide::Buffer<double> out(1);
	r.realize(out);
	EXPECT_NEAR(out(0), kAccSumSq, 1.0);
	EXPECT_NEAR(out(0) - 1.0e14, 17.0, 0.5);
}

TEST(TypingSweep, EinsumF32Regression) {
	// f32 matmul "ij,jk->ik" 2x2: historical values.
	Halide::Buffer<float> abuf(2, 2), bbuf(2, 2);
	abuf(0, 0) = 1.0f; abuf(1, 0) = 2.0f; abuf(0, 1) = 3.0f; abuf(1, 1) = 4.0f;
	bbuf(0, 0) = 5.0f; bbuf(1, 0) = 6.0f; bbuf(0, 1) = 7.0f; bbuf(1, 1) = 8.0f;
	Halide::Func af("ts_es32_a"), bf("ts_es32_b");
	Halide::Var x, y;
	af(x, y) = abuf(x, y);
	bf(x, y) = bbuf(x, y);
	shape_t s2 = { 2, 2 };
	auto r = einsum("ij,jk->ik", af, s2, bf, s2, "ts_es32");
	ASSERT_EQ(r.types()[0], Halide::Float(32));
	Halide::Buffer<float> out(2, 2);
	r.realize(out);
	// A(row i, col j) with buf(col, row): A = [[1,2],[3,4]], B = [[5,6],[7,8]]
	// C = A*B = [[19,22],[43,50]]; C(col, row).
	EXPECT_NEAR(out(0, 0), 19.0f, 1e-4f);
	EXPECT_NEAR(out(1, 0), 22.0f, 1e-4f);
	EXPECT_NEAR(out(0, 1), 43.0f, 1e-4f);
	EXPECT_NEAR(out(1, 1), 50.0f, 1e-4f);
}

// ---------------------------------------------------------------------------
// distance.h — typed accumulators
// ---------------------------------------------------------------------------

TEST(TypingSweep, CdistF64Precision) {
	// 1-D points 0 and 1e7 + 0.001: distance needs f64 to keep the 0.001
	// (f32 ulp at 1e7 is 1).
	Halide::Buffer<double> abuf(1, 1), bbuf(1, 1);
	abuf(0, 0) = 0.0;
	bbuf(0, 0) = 1.0e7 + 0.001;
	Halide::Func af("ts_cd_a"), bf("ts_cd_b");
	Halide::Var x, y;
	af(x, y) = abuf(x, y);
	bf(x, y) = bbuf(x, y);

	auto d = cdist_euclidean(af, bf, 1, 1, 1, "ts_cd");
	ASSERT_EQ(d.types()[0], Halide::Float(64));
	Halide::Buffer<double> out(1, 1);
	d.realize(out);
	EXPECT_NEAR(out(0, 0), 1.0e7 + 0.001, 1e-6);

	auto m = cdist_manhattan(af, bf, 1, 1, 1, "ts_cdm");
	ASSERT_EQ(m.types()[0], Halide::Float(64));
	m.realize(out);
	EXPECT_NEAR(out(0, 0), 1.0e7 + 0.001, 1e-6);
}

TEST(TypingSweep, CosineSimilarityTypesAndF32Regression) {
	{
		Halide::Buffer<double> abuf(2);
		abuf(0) = 1.0; abuf(1) = 0.0;
		Halide::Func af = wrap1d_f64(abuf, "ts_cs_a");
		shape_t s = { 2 };
		auto r = cosine_similarity(af, af, s, "ts_cs64");
		ASSERT_EQ(r.types()[0], Halide::Float(64));
		Halide::Buffer<double> out(1);
		r.realize(out);
		EXPECT_NEAR(out(0), 1.0, 1e-12);
	}
	{
		Halide::Buffer<float> abuf(2), bbuf(2);
		abuf(0) = 3.0f; abuf(1) = 4.0f;
		bbuf(0) = 3.0f; bbuf(1) = 4.0f;
		Halide::Func af = wrap1d_f32(abuf, "ts_cs32_a");
		Halide::Func bf = wrap1d_f32(bbuf, "ts_cs32_b");
		shape_t s = { 2 };
		auto r = cosine_similarity(af, bf, s, "ts_cs32");
		ASSERT_EQ(r.types()[0], Halide::Float(32));
		Halide::Buffer<float> out(1);
		r.realize(out);
		EXPECT_NEAR(out(0), 1.0f, 1e-6f);
	}
}

// ---------------------------------------------------------------------------
// math_ext.h / trig.h — typed constants (ln2, ln10, pi, ratios)
// ---------------------------------------------------------------------------

TEST(TypingSweep, Log2F64Ln2Precision) {
	// log2(2^40) = 40 at 1e-9: the old widened-f32 ln(2) constant leaves a
	// ~8e-7 absolute error at this magnitude.
	Halide::Buffer<double> xbuf(1);
	xbuf(0) = 1099511627776.0; // 2^40
	Halide::Func xf = wrap1d_f64(xbuf, "ts_l2_x");
	shape_t s = { 1 };
	auto r = numhalide::log2(xf, s, "ts_l2");
	ASSERT_EQ(r.types()[0], Halide::Float(64));
	Halide::Buffer<double> out(1);
	r.realize(out);
	EXPECT_NEAR(out(0), 40.0, 1e-9);
}

TEST(TypingSweep, Log2Log10F32Regression) {
	Halide::Buffer<float> xbuf(2);
	xbuf(0) = 8.0f; xbuf(1) = 100.0f;
	Halide::Func xf = wrap1d_f32(xbuf, "ts_l232_x");
	shape_t s = { 2 };
	auto r2 = numhalide::log2(xf, s, "ts_l232");
	auto r10 = numhalide::log10(xf, s, "ts_l1032");
	Halide::Buffer<float> out(2);
	r2.realize(out);
	EXPECT_NEAR(out(0), 3.0f, 1e-5f);
	r10.realize(out);
	EXPECT_NEAR(out(1), 2.0f, 1e-5f);
}

TEST(TypingSweep, RadiansDegreesF64Precision) {
	Halide::Buffer<double> xbuf(1);
	xbuf(0) = 180.0;
	Halide::Func xf = wrap1d_f64(xbuf, "ts_rad_x");
	shape_t s = { 1 };
	auto r = radians(xf, s, "ts_rad");
	ASSERT_EQ(r.types()[0], Halide::Float(64));
	Halide::Buffer<double> out(1);
	r.realize(out);
	// Old widened-f32 ratio misses pi by ~2e-7.
	EXPECT_NEAR(out(0), kPi, 1e-12);

	Halide::Buffer<double> ybuf(1);
	ybuf(0) = kPi;
	Halide::Func yf = wrap1d_f64(ybuf, "ts_deg_x");
	auto d = degrees(yf, s, "ts_deg");
	ASSERT_EQ(d.types()[0], Halide::Float(64));
	d.realize(out);
	EXPECT_NEAR(out(0), 180.0, 1e-9);
}

TEST(TypingSweep, RadiansF32Regression) {
	Halide::Buffer<float> xbuf(1);
	xbuf(0) = 180.0f;
	Halide::Func xf = wrap1d_f32(xbuf, "ts_rad32_x");
	shape_t s = { 1 };
	auto r = radians(xf, s, "ts_rad32");
	Halide::Buffer<float> out(1);
	r.realize(out);
	EXPECT_NEAR(out(0), float(kPi), 1e-4f);
}

TEST(TypingSweep, SincF64PrecisionAndExactUnity) {
	// sinc(0.5) = 2/pi at 1e-12 (old f32 pi leaves ~1e-8); sinc(0) = 1
	// EXACTLY through the indicator path in both types.
	{
		Halide::Buffer<double> xbuf(2);
		xbuf(0) = 0.5; xbuf(1) = 0.0;
		Halide::Func xf = wrap1d_f64(xbuf, "ts_sinc_x");
		shape_t s = { 2 };
		auto r = sinc(xf, s, "ts_sinc64");
		ASSERT_EQ(r.types()[0], Halide::Float(64));
		Halide::Buffer<double> out(2);
		r.realize(out);
		EXPECT_NEAR(out(0), 2.0 / kPi, 1e-12);
		EXPECT_EQ(out(1), 1.0);
	}
	{
		Halide::Buffer<float> xbuf(2);
		xbuf(0) = 0.5f; xbuf(1) = 0.0f;
		Halide::Func xf = wrap1d_f32(xbuf, "ts_sinc32_x");
		shape_t s = { 2 };
		auto r = sinc(xf, s, "ts_sinc32");
		Halide::Buffer<float> out(2);
		r.realize(out);
		EXPECT_NEAR(out(0), float(2.0 / kPi), 1e-6f);
		EXPECT_EQ(out(1), 1.0f);
	}
}

// ---------------------------------------------------------------------------
// numeric.h — trapz / correlate1d typed accumulators
// ---------------------------------------------------------------------------

TEST(TypingSweep, TrapzF64Accumulation) {
	// f = [2e14, 0, 34], dx = 1: integral = 1e14 + 17.
	Halide::Buffer<double> fbuf(3);
	fbuf(0) = 2.0e14; fbuf(1) = 0.0; fbuf(2) = 34.0;
	Halide::Func ff = wrap1d_f64(fbuf, "ts_tz_f");
	auto r = trapz_1d(ff, 3, 1.0f, "ts_tz");
	ASSERT_EQ(r.types()[0], Halide::Float(64));
	Halide::Buffer<double> out = Halide::Buffer<double>::make_scalar();
	r.realize(out);
	EXPECT_NEAR(out(), 1.0e14 + 17.0, 1.0);
	EXPECT_NEAR(out() - 1.0e14, 17.0, 0.5);
}

TEST(TypingSweep, Correlate1dTypesAndF32Regression) {
	{
		Halide::Buffer<double> abuf(3);
		abuf(0) = 1.0; abuf(1) = 2.0; abuf(2) = 3.0;
		Halide::Buffer<double> vbuf(2);
		vbuf(0) = 1.0; vbuf(1) = 1.0;
		Halide::Func af = wrap1d_f64(abuf, "ts_co_a");
		Halide::Func vf = wrap1d_f64(vbuf, "ts_co_v");
		auto r = correlate1d(af, vf, 3, 2, "full", "ts_co64");
		ASSERT_EQ(r.types()[0], Halide::Float(64));
		Halide::Buffer<double> out(4);
		r.realize(out);
		double const e[4] = { 1.0, 3.0, 5.0, 3.0 };
		for (int i = 0; i < 4; ++i) EXPECT_NEAR(out(i), e[i], 1e-12) << i;
	}
	{
		Halide::Buffer<float> abuf(3);
		abuf(0) = 1.0f; abuf(1) = 2.0f; abuf(2) = 3.0f;
		Halide::Buffer<float> vbuf(2);
		vbuf(0) = 1.0f; vbuf(1) = 1.0f;
		Halide::Func af = wrap1d_f32(abuf, "ts_co32_a");
		Halide::Func vf = wrap1d_f32(vbuf, "ts_co32_v");
		auto r = correlate1d(af, vf, 3, 2, "full", "ts_co32");
		ASSERT_EQ(r.types()[0], Halide::Float(32));
		Halide::Buffer<float> out(4);
		r.realize(out);
		float const e[4] = { 1.0f, 3.0f, 5.0f, 3.0f };
		for (int i = 0; i < 4; ++i) EXPECT_NEAR(out(i), e[i], 1e-5f) << i;
	}
}

// ---------------------------------------------------------------------------
// statistics2.h — percentile fraction precision
// ---------------------------------------------------------------------------

TEST(TypingSweep, PercentileF64FracPrecision) {
	// sorted = [0,1,2,3], q = 10: pos = 0.3, result = 0.3. The old
	// host-f32 fraction gave 0.30000001192...; 1e-12 needs the double
	// host math + typed constants.
	Halide::Buffer<double> fbuf(4);
	for (int i = 0; i < 4; ++i) fbuf(i) = double(i);
	Halide::Func ff = wrap1d_f64(fbuf, "ts_pc_f");
	auto r = stats::percentile(ff, 4, 10.0f, "ts_pc64");
	ASSERT_EQ(r.types()[0], Halide::Float(64));
	Halide::Buffer<double> out(1);
	r.realize(out);
	EXPECT_NEAR(out(0), 0.3, 1e-12);
}

TEST(TypingSweep, PercentileF32Regression) {
	// Median of [1,2,3,4] = 2.5 (historical pin).
	Halide::Buffer<float> fbuf(4);
	for (int i = 0; i < 4; ++i) fbuf(i) = float(i + 1);
	Halide::Func ff = wrap1d_f32(fbuf, "ts_pc32_f");
	auto r = stats::percentile(ff, 4, 50.0f, "ts_pc32");
	Halide::Buffer<float> out(1);
	r.realize(out);
	EXPECT_NEAR(out(0), 2.5f, 1e-4f);
}

TEST(TypingSweep, NanmedianNanpercentileF64Types) {
	// Seeds were bare 0.0f — a Halide type error for f64 pipelines.
	Halide::Buffer<double> fbuf(4);
	fbuf(0) = 3.0; fbuf(1) = 1.0;
	fbuf(2) = std::numeric_limits<double>::quiet_NaN(); fbuf(3) = 2.0;
	Halide::Func ff = wrap1d_f64(fbuf, "ts_nm_f");

	auto med = nanmedian(ff, 4, "ts_nm64");
	ASSERT_EQ(med.types()[0], Halide::Float(64));
	Halide::Buffer<double> out = Halide::Buffer<double>::make_scalar();
	med.realize(out);
	EXPECT_NEAR(out(), 2.0, 1e-12);

	auto pct = stats::nanpercentile(ff, 4, 50.0f, "ts_np64");
	ASSERT_EQ(pct.types()[0], Halide::Float(64));
	pct.realize(out);
	EXPECT_NEAR(out(), 2.0, 1e-12);
}

// ---------------------------------------------------------------------------
// Type-propagation checks for the remaining fixed seeds (definition-time
// pins: these very Func definitions were Halide type errors — or silent
// f32 truncations — before the sweep)
// ---------------------------------------------------------------------------

TEST(TypingSweep, RemainingFixedSitesF64Types) {
	Halide::Var x, y;

	// la.h normalize
	{
		Halide::Buffer<double> b(4);
		for (int i = 0; i < 4; ++i) b(i) = double(i + 1);
		Halide::Func f = wrap1d_f64(b, "ts_ty_norm_f");
		shape_t s = { 4 };
		auto r = normalize(f, s, -1, 1e-8f, "ts_ty_norm");
		EXPECT_EQ(r.types()[0], Halide::Float(64));
	}
	// la.h eig_qr (3x3 path; n=2 delegates to the already-typed eig2x2)
	{
		Halide::Buffer<double> b(3, 3);
		for (int r2 = 0; r2 < 3; ++r2)
			for (int c = 0; c < 3; ++c)
				b(c, r2) = (r2 == c) ? 2.0 + r2 : 0.25;
		Halide::Func f = wrap2d_f64(b, "ts_ty_eig_f");
		auto r = eig_qr(f, 3, 2, "ts_ty_eig");
		EXPECT_EQ(r.real.types()[0], Halide::Float(64));
		EXPECT_EQ(r.imag.types()[0], Halide::Float(64));
	}
	// threshold.h adaptive + otsu output
	{
		Halide::Buffer<double> b(4, 4);
		for (int r2 = 0; r2 < 4; ++r2)
			for (int c = 0; c < 4; ++c)
				b(c, r2) = 0.25 * ((r2 + c) % 4);
		Halide::Func f = wrap2d_f64(b, "ts_ty_th_f");
		shape_t s = { 4, 4 };
		auto ad = threshold_adaptive(f, s, 3, "ts_ty_thad");
		EXPECT_EQ(ad.types()[0], Halide::Float(64));
		auto ot = threshold_otsu(f, s, 16, "ts_ty_otsu");
		EXPECT_EQ(ot.types()[0], Halide::Float(64));
	}
	// histogram.h equalize + gamma_correct
	{
		Halide::Buffer<double> b(8);
		for (int i = 0; i < 8; ++i) b(i) = i / 8.0;
		Halide::Func f = wrap1d_f64(b, "ts_ty_he_f");
		shape_t s = { 8 };
		auto he = histogram_equalize(f, s, 16, "ts_ty_he");
		EXPECT_EQ(he.types()[0], Halide::Float(64));
		auto gc = gamma_correct(f, s, Halide::Expr(2.2f), "ts_ty_gc");
		EXPECT_EQ(gc.types()[0], Halide::Float(64));
	}
	// interp.h weights
	{
		Halide::Buffer<double> b(4);
		for (int i = 0; i < 4; ++i) b(i) = double(i);
		Halide::Func f = wrap1d_f64(b, "ts_ty_in_f");
		shape_t s = { 4 };
		auto iu = interp1d_uniform(f, s, 2.0f, "ts_ty_iu");
		EXPECT_EQ(iu.types()[0], Halide::Float(64));

		Halide::Buffer<double> b2(4, 4);
		for (int r2 = 0; r2 < 4; ++r2)
			for (int c = 0; c < 4; ++c)
				b2(c, r2) = double(r2 * 4 + c);
		Halide::Func f2 = wrap2d_f64(b2, "ts_ty_rb_f");
		shape_t s2 = { 4, 4 };
		auto rb = resize_bilinear(f2, s2, 8, 8, "ts_ty_rb");
		EXPECT_EQ(rb.types()[0], Halide::Float(64));
	}
	// soft_sort.h
	{
		Halide::Buffer<double> b(4);
		b(0) = 0.4; b(1) = 0.1; b(2) = 0.9; b(3) = 0.2;
		Halide::Func f = wrap1d_f64(b, "ts_ty_ss_f");
		auto sr = soft_rank(f, 4, 0.1f, "ts_ty_sr");
		EXPECT_EQ(sr.types()[0], Halide::Float(64));
		auto so = soft_sort(f, 4, 0.1f, 1.0f, "ts_ty_so");
		EXPECT_EQ(so.types()[0], Halide::Float(64));
	}
	// stencil.h stencil_apply
	{
		Halide::Buffer<double> b(4, 4);
		b.fill(1.0);
		Halide::Func f = wrap2d_f64(b, "ts_ty_st_f");
		Halide::Buffer<double> wb(1);
		wb(0) = 1.0;
		Halide::Func wf = wrap1d_f64(wb, "ts_ty_st_w");
		Halide::Buffer<int32_t> ob(1);
		ob(0) = 0;
		Halide::Func oxf("ts_ty_st_ox"), oyf("ts_ty_st_oy");
		oxf(x) = ob(x);
		oyf(x) = ob(x);
		shape_t s = { 4, 4 };
		auto st = stencil_apply(f, s, wf, oxf, oyf, 1, "ts_ty_st");
		EXPECT_EQ(st.types()[0], Halide::Float(64));
	}
	// numeric.h logaddexp2
	{
		Halide::Buffer<double> b(2);
		b(0) = 1.0; b(1) = 2.0;
		Halide::Func f = wrap1d_f64(b, "ts_ty_lae_f");
		shape_t s = { 2 };
		auto r = logaddexp2(f, f, s, "ts_ty_lae");
		EXPECT_EQ(r.types()[0], Halide::Float(64));
	}
	// fft_ext.h spectral_centroid + 2-D cross power spectrum
	{
		Halide::Buffer<double> b(4);
		b(0) = 1.0; b(1) = 0.5; b(2) = 0.25; b(3) = 0.125;
		Halide::Func cf("ts_ty_sc_f");
		cf(x) = Halide::Tuple(b(x), Halide::Internal::make_zero(Halide::Float(64)));
		auto sc = spectral_centroid(cf, 4, "ts_ty_sc");
		EXPECT_EQ(sc.types()[0], Halide::Float(64));

		Halide::Func c2("ts_ty_cps_f");
		Halide::Buffer<double> b2(2, 2);
		b2.fill(0.5);
		c2(x, y) = Halide::Tuple(b2(x, y), Halide::Internal::make_zero(Halide::Float(64)));
		auto cps = cross_power_spectrum(c2, c2, 2, 2, "ts_ty_cps");
		EXPECT_EQ(cps.types()[0], Halide::Float(64));
	}
	// polynomial.h polyfit (f64 Vandermonde exponent + typed lstsq chain)
	{
		Halide::Buffer<double> xb(4), yb(4);
		for (int i = 0; i < 4; ++i) { xb(i) = double(i); yb(i) = 2.0 * i + 1.0; }
		Halide::Func xf = wrap1d_f64(xb, "ts_ty_pf_x");
		Halide::Func yf = wrap1d_f64(yb, "ts_ty_pf_y");
		auto c = polyfit(xf, yf, 4, 1, "ts_ty_pf");
		EXPECT_EQ(c.types()[0], Halide::Float(64));
	}
	// la_batched.h cholesky / qr
	{
		Halide::Buffer<double> b(2, 2, 1);
		b(0, 0, 0) = 4.0; b(1, 0, 0) = 1.0;
		b(0, 1, 0) = 1.0; b(1, 1, 0) = 3.0;
		Halide::Func f("ts_ty_bch_f");
		Halide::Var z;
		f(x, y, z) = b(x, y, z);
		auto L = batched_cholesky(f, 2, 1, "ts_ty_bch");
		EXPECT_EQ(L.types()[0], Halide::Float(64));
		auto qr = batched_qr_gs(f, 2, 2, 1, "ts_ty_bqr");
		EXPECT_EQ(qr.Q.types()[0], Halide::Float(64));
		EXPECT_EQ(qr.R.types()[0], Halide::Float(64));
	}
	// factory_func.h rand_normal at f64 (typed two_pi constant)
	{
		shape_t s = { 8 };
		auto rn = rand_normal(Halide::Float(64), s, 0.0f, 1.0f, 7, "ts_ty_rn");
		EXPECT_EQ(rn.types()[0], Halide::Float(64));
	}
}
