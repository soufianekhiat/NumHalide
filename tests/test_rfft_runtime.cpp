/// @file test_rfft_runtime.cpp
/// @brief Tests for the runtime-Expr real-FFT overloads (rfft / irfft with
/// Halide::Expr sizes), the 1-D typed cross_power_spectrum, the
/// magnitude-weighted spectral centroid, and the same-cycle library fixes:
/// the int-N rfft/irfft/rfft2d/irfft2d accept any size (stale pow-2
/// requires dropped), rfft is type-preserving, and irfft reads ONLY bins
/// [0, N/2+1) (the old select body's bounds inference requested N+1
/// elements from a buffer documented to hold N/2+1).

#include <gtest/gtest.h>
#include "numhalide_all.h"
#include <cmath>

using namespace numhalide;

namespace {

// Wrap a real value list into a Buffer-backed 1D Func.
Halide::Func make_real_1d(std::initializer_list<float> vals, char const* nm) {
	int const n = (int)vals.size();
	Halide::Buffer<float> b(n);
	int i = 0;
	for (float v : vals) b(i++) = v;
	Halide::Func f(nm);
	Halide::Var x;
	f(x) = b(x);
	return f;
}

// Wrap re/im value lists into a Tuple-complex 1D Func backed by CONCRETE
// buffers of exactly re.size() elements (this is what makes the irfft
// bounds regression bite: the old body requested N+1 elements).
Halide::Func make_complex_1d(std::initializer_list<float> re,
		std::initializer_list<float> im, char const* nm) {
	int const n = (int)re.size();
	Halide::Buffer<float> rb(n), ib(n);
	int i = 0;
	for (float v : re) rb(i++) = v;
	i = 0;
	for (float v : im) ib(i++) = v;
	Halide::Func f(nm);
	Halide::Var x;
	f(x) = Halide::Tuple(rb(x), ib(x));
	return f;
}

// Type-generic re/im extraction (the Tuple element keeps its type).
Halide::Func extract_re(Halide::Func z, char const* nm) {
	Halide::Func f(nm);
	Halide::Var x;
	f(x) = z(x)[0];
	return f;
}

Halide::Func extract_im(Halide::Func z, char const* nm) {
	Halide::Func f(nm);
	Halide::Var x;
	f(x) = z(x)[1];
	return f;
}

} // namespace

TEST(RfftRuntime, ForwardHandPins) {
	// rfft([1,2,3,4]) -> [10, -2+2j, -2] realized over K = 3.
	Halide::Func in4 = make_real_1d({1.0f, 2.0f, 3.0f, 4.0f}, "rf4_in");
	Halide::Func rt4 = rfft(in4, Halide::Expr(4), Halide::Float(32), "rf4_rt");
	Halide::Buffer<float> re4 = extract_re(rt4, "rf4_re").realize({3});
	Halide::Buffer<float> im4 = extract_im(rt4, "rf4_im").realize({3});
	float const exp_re4[] = {10.0f, -2.0f, -2.0f};
	float const exp_im4[] = {0.0f, 2.0f, 0.0f};
	for (int i = 0; i < 3; ++i) {
		EXPECT_NEAR(re4(i), exp_re4[i], 1e-4f) << "n4 re[" << i << "]";
		EXPECT_NEAR(im4(i), exp_im4[i], 1e-4f) << "n4 im[" << i << "]";
	}

	// Non-pow-2 (pins the now-legal path): rfft([1,2,3]) ->
	// [6, -1.5+0.8660254j] over K = 2.
	Halide::Func in3 = make_real_1d({1.0f, 2.0f, 3.0f}, "rf3_in");
	Halide::Func rt3 = rfft(in3, Halide::Expr(3), Halide::Float(32), "rf3_rt");
	Halide::Buffer<float> re3 = extract_re(rt3, "rf3_re").realize({2});
	Halide::Buffer<float> im3 = extract_im(rt3, "rf3_im").realize({2});
	float const s3 = 0.8660254f; // sqrt(3)/2
	EXPECT_NEAR(re3(0), 6.0f, 1e-4f);
	EXPECT_NEAR(im3(0), 0.0f, 1e-4f);
	EXPECT_NEAR(re3(1), -1.5f, 1e-4f);
	EXPECT_NEAR(im3(1), s3, 1e-4f);
}

TEST(RfftRuntime, InverseAndRoundTrip) {
	// irfft([10, -2+2j, -2], K=3) -> [1,2,3,4].
	Halide::Func spec = make_complex_1d({10.0f, -2.0f, -2.0f},
			{0.0f, 2.0f, 0.0f}, "ir_spec");
	Halide::Func inv = irfft(spec, Halide::Expr(3), Halide::Float(32), "ir_rt");
	Halide::Buffer<float> out = inv.realize({4});
	float const expected[] = {1.0f, 2.0f, 3.0f, 4.0f};
	for (int i = 0; i < 4; ++i)
		EXPECT_NEAR(out(i), expected[i], 1e-4f) << "x[" << i << "]";

	// Round-trip irfft(rfft(x)) == x for a non-trivial x (n = 6, K = 4).
	float const x6[] = {0.5f, -1.25f, 2.0f, 3.5f, -0.75f, 1.0f};
	Halide::Func in = make_real_1d({0.5f, -1.25f, 2.0f, 3.5f, -0.75f, 1.0f}, "rt6_in");
	Halide::Func fwd = rfft(in, Halide::Expr(6), Halide::Float(32), "rt6_fwd");
	Halide::Func back = irfft(fwd, Halide::Expr(4), Halide::Float(32), "rt6_back");
	Halide::Buffer<float> rt = back.realize({6});
	for (int i = 0; i < 6; ++i)
		EXPECT_NEAR(rt(i), x6[i], 1e-3f) << "round-trip[" << i << "]";
}

TEST(RfftRuntime, IntNRfftTwinMatchesRuntime) {
	// The FIXED int-N rfft against the runtime form on identical data:
	// pow-2 n = 4 AND non-pow-2 n = 3 (legal only after the stale
	// power-of-2 require removal).
	{
		Halide::Func in = make_real_1d({1.0f, 2.0f, 3.0f, 4.0f}, "tw4_in");
		Halide::Func ct = rfft(in, 4, "tw4_ct");
		Halide::Func rt = rfft(in, Halide::Expr(4), Halide::Float(32), "tw4_rt");
		Halide::Buffer<float> ct_re = extract_re(ct, "tw4_ct_re").realize({3});
		Halide::Buffer<float> ct_im = extract_im(ct, "tw4_ct_im").realize({3});
		Halide::Buffer<float> rt_re = extract_re(rt, "tw4_rt_re").realize({3});
		Halide::Buffer<float> rt_im = extract_im(rt, "tw4_rt_im").realize({3});
		for (int i = 0; i < 3; ++i) {
			EXPECT_NEAR(ct_re(i), rt_re(i), 1e-4f) << "n4 re[" << i << "]";
			EXPECT_NEAR(ct_im(i), rt_im(i), 1e-4f) << "n4 im[" << i << "]";
		}
	}
	{
		Halide::Func in = make_real_1d({1.0f, 2.0f, 3.0f}, "tw3_in");
		Halide::Func ct = rfft(in, 3, "tw3_ct");
		Halide::Func rt = rfft(in, Halide::Expr(3), Halide::Float(32), "tw3_rt");
		Halide::Buffer<float> ct_re = extract_re(ct, "tw3_ct_re").realize({2});
		Halide::Buffer<float> ct_im = extract_im(ct, "tw3_ct_im").realize({2});
		Halide::Buffer<float> rt_re = extract_re(rt, "tw3_rt_re").realize({2});
		Halide::Buffer<float> rt_im = extract_im(rt, "tw3_rt_im").realize({2});
		for (int i = 0; i < 2; ++i) {
			EXPECT_NEAR(ct_re(i), rt_re(i), 1e-4f) << "n3 re[" << i << "]";
			EXPECT_NEAR(ct_im(i), rt_im(i), 1e-4f) << "n3 im[" << i << "]";
		}
	}
}

TEST(RfftRuntime, IntNIrfftTwinMatchesRuntime) {
	// The FIXED int-N irfft against the runtime form on identical
	// half-spectrum data (real DC and Nyquist — the documented contract).
	// Pow-2 N = 4 / K = 3:
	{
		Halide::Func spec = make_complex_1d({10.0f, -2.0f, -2.0f},
				{0.0f, 2.0f, 0.0f}, "it4_spec");
		Halide::Func ct = irfft(spec, 4, "it4_ct");
		Halide::Func rt = irfft(spec, Halide::Expr(3), Halide::Float(32), "it4_rt");
		Halide::Buffer<float> ct_out = ct.realize({4});
		Halide::Buffer<float> rt_out = rt.realize({4});
		float const expected[] = {1.0f, 2.0f, 3.0f, 4.0f};
		for (int i = 0; i < 4; ++i) {
			EXPECT_NEAR(ct_out(i), rt_out(i), 1e-4f) << "n4 twin[" << i << "]";
			EXPECT_NEAR(ct_out(i), expected[i], 1e-4f) << "n4 value[" << i << "]";
		}
	}
	// Non-pow-2 even N = 6 / K = 4 (legal only after the require removal):
	{
		Halide::Func spec = make_complex_1d({6.0f, 1.5f, -2.0f, 0.5f},
				{0.0f, -1.0f, 2.0f, 0.0f}, "it6_spec");
		Halide::Func ct = irfft(spec, 6, "it6_ct");
		Halide::Func rt = irfft(spec, Halide::Expr(4), Halide::Float(32), "it6_rt");
		Halide::Buffer<float> ct_out = ct.realize({6});
		Halide::Buffer<float> rt_out = rt.realize({6});
		for (int i = 0; i < 6; ++i)
			EXPECT_NEAR(ct_out(i), rt_out(i), 1e-4f) << "n6 twin[" << i << "]";
	}
}

TEST(RfftRuntime, IntNIrfftBoundedReads) {
	// Bounds regression: realize the FIXED int-N irfft against CONCRETE
	// Halide::Buffers holding exactly N/2+1 = 3 bins. The old select body
	// read input(N - x); bounds inference over both branches requested
	// input over [0, N] — N+1 elements — so this realization could not
	// bounds-check. The fixed body reads ONLY bins [0, K).
	Halide::Func spec = make_complex_1d({10.0f, -2.0f, -2.0f},
			{0.0f, 2.0f, 0.0f}, "bb_spec");
	Halide::Func inv = irfft(spec, 4, "bb_inv");
	Halide::Buffer<float> out = inv.realize({4});
	float const expected[] = {1.0f, 2.0f, 3.0f, 4.0f};
	for (int i = 0; i < 4; ++i)
		EXPECT_NEAR(out(i), expected[i], 1e-4f) << "x[" << i << "]";
}

TEST(RfftRuntime, Float64Typing) {
	// Runtime rfft with type=Float(64) keeps f64 end to end; values match
	// the hand DFT of [1,2,3,4] to double precision (angles in f64).
	int const n = 4;
	Halide::Buffer<double> b(n);
	for (int i = 0; i < n; ++i) b(i) = (double)(i + 1);
	Halide::Func in("f64_in");
	Halide::Var x;
	in(x) = b(x);

	Halide::Func rt = rfft(in, Halide::Expr(n), Halide::Float(64), "f64_rt");
	ASSERT_EQ(rt.types()[0], Halide::Float(64));
	ASSERT_EQ(rt.types()[1], Halide::Float(64));

	Halide::Buffer<double> re_out = extract_re(rt, "f64_re").realize({3});
	Halide::Buffer<double> im_out = extract_im(rt, "f64_im").realize({3});
	double const exp_re[] = {10.0, -2.0, -2.0};
	double const exp_im[] = {0.0, 2.0, 0.0};
	for (int i = 0; i < 3; ++i) {
		EXPECT_NEAR(re_out(i), exp_re[i], 1e-9) << "re[" << i << "]";
		EXPECT_NEAR(im_out(i), exp_im[i], 1e-9) << "im[" << i << "]";
	}

	// Fix pin: the int-N rfft no longer hard-casts to float — f64 input
	// keeps f64 (its twiddles stay f32, so values pin at 1e-4 only).
	Halide::Func ct = rfft(in, n, "f64_ct");
	ASSERT_EQ(ct.types()[0], Halide::Float(64));
	ASSERT_EQ(ct.types()[1], Halide::Float(64));
	Halide::Buffer<double> ct_re = extract_re(ct, "f64_ct_re").realize({3});
	for (int i = 0; i < 3; ++i)
		EXPECT_NEAR(ct_re(i), exp_re[i], 1e-4) << "ct re[" << i << "]";
}

TEST(RfftRuntime, CrossPowerSpectrum1DTyped) {
	// (1+0j, 0+1j): A*conj(B) = -j -> normalized (0, -1).
	Halide::Func a = make_complex_1d({1.0f}, {0.0f}, "cps_a");
	Halide::Func b = make_complex_1d({0.0f}, {1.0f}, "cps_b");
	Halide::Func cps = cross_power_spectrum(a, b, Halide::Float(32), "cps1");
	Halide::Buffer<float> re = extract_re(cps, "cps1_re").realize({1});
	Halide::Buffer<float> im = extract_im(cps, "cps1_im").realize({1});
	EXPECT_NEAR(re(0), 0.0f, 1e-5f);
	EXPECT_NEAR(im(0), -1.0f, 1e-5f);

	// Zero bin: the sqrt(max(mag2, 1e-20)) floor keeps the division
	// finite -> (0, 0), no NaN.
	Halide::Func z = make_complex_1d({0.0f}, {0.0f}, "cps_z");
	Halide::Func cps0 = cross_power_spectrum(z, z, Halide::Float(32), "cps0");
	Halide::Buffer<float> re0 = extract_re(cps0, "cps0_re").realize({1});
	Halide::Buffer<float> im0 = extract_im(cps0, "cps0_im").realize({1});
	EXPECT_FALSE(std::isnan(re0(0)));
	EXPECT_FALSE(std::isnan(im0(0)));
	EXPECT_NEAR(re0(0), 0.0f, 1e-6f);
	EXPECT_NEAR(im0(0), 0.0f, 1e-6f);
}

TEST(RfftRuntime, CrossPowerSpectrum1DMatches2DForm) {
	// Twin-check the 1-D typed form against the existing 2-D form on a
	// 1-row grid. The floors are value-identical:
	// sqrt(max(m^2, 1e-20)) == max(m, 1e-10) for m >= 0.
	int const K = 4;
	float const are[] = {1.0f, 2.0f, 0.0f, -1.0f};
	float const aim[] = {0.0f, -1.0f, 3.0f, -1.0f};
	float const bre[] = {0.0f, 1.0f, 2.0f, 3.0f};
	float const bim[] = {1.0f, 1.0f, 0.0f, -2.0f};

	Halide::Buffer<float> arb(K), aib(K), brb(K), bib(K);
	for (int i = 0; i < K; ++i) {
		arb(i) = are[i]; aib(i) = aim[i];
		brb(i) = bre[i]; bib(i) = bim[i];
	}

	Halide::Func a1("tw_a1"), b1("tw_b1");
	{
		Halide::Var x;
		a1(x) = Halide::Tuple(arb(x), aib(x));
		b1(x) = Halide::Tuple(brb(x), bib(x));
	}
	Halide::Func a2("tw_a2"), b2("tw_b2");
	{
		Halide::Var x, y;
		a2(x, y) = Halide::Tuple(arb(x), aib(x));
		b2(x, y) = Halide::Tuple(brb(x), bib(x));
	}

	Halide::Func c1 = cross_power_spectrum(a1, b1, Halide::Float(32), "tw_c1");
	Halide::Func c2 = cross_power_spectrum(a2, b2, K, 1, "tw_c2");

	Halide::Buffer<float> re1 = extract_re(c1, "tw_re1").realize({K});
	Halide::Buffer<float> im1 = extract_im(c1, "tw_im1").realize({K});

	Halide::Func re2f("tw_re2f"), im2f("tw_im2f");
	{
		Halide::Var x, y;
		re2f(x, y) = c2(x, y)[0];
		im2f(x, y) = c2(x, y)[1];
	}
	Halide::Buffer<float> re2 = re2f.realize({K, 1});
	Halide::Buffer<float> im2 = im2f.realize({K, 1});

	for (int i = 0; i < K; ++i) {
		EXPECT_NEAR(re1(i), re2(i, 0), 1e-6f) << "re[" << i << "]";
		EXPECT_NEAR(im1(i), im2(i, 0), 1e-6f) << "im[" << i << "]";
	}
}

TEST(RfftRuntime, SpectralCentroidMagnitude) {
	// [0,1,0,1]: total = 2, weighted = 1*1 + 3*1 = 4 -> centroid = 2.
	Halide::Func mags = make_real_1d({0.0f, 1.0f, 0.0f, 1.0f}, "scm_in");
	Halide::Func c = spectral_centroid_magnitude(mags, Halide::Expr(4),
			Halide::Float(32), "scm");
	Halide::Func wrap("scm_wrap");
	Halide::Var x;
	wrap(x) = c();
	Halide::Buffer<float> out = wrap.realize({1});
	EXPECT_NEAR(out(0), 2.0f, 1e-5f);

	// All zeros: the select guard substitutes 1 for the zero total -> 0.
	Halide::Func zeros = make_real_1d({0.0f, 0.0f, 0.0f, 0.0f}, "scm_z");
	Halide::Func cz = spectral_centroid_magnitude(zeros, Halide::Expr(4),
			Halide::Float(32), "scm_z_c");
	Halide::Func wrapz("scm_z_wrap");
	wrapz(x) = cz();
	Halide::Buffer<float> outz = wrapz.realize({1});
	EXPECT_NEAR(outz(0), 0.0f, 1e-6f);
}
