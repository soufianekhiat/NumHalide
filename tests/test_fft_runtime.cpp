/// @file test_fft_runtime.cpp
/// @brief Tests for the runtime-Expr general-DFT overloads (dft_1d /
/// idft_1d_normalized / dft_2d / idft_2d_normalized with Halide::Expr
/// sizes). Checked against the compile-time Cooley-Tukey twin on pow-2
/// data, hand-computed pins on non-pow-2 sizes, the unscaled-inverse
/// contract (dft_1d(+1) of dft_1d(-1) of x == n*x), normalized
/// round-trips, a non-square 2D impulse, and an f64 compute-type pin.

#include <gtest/gtest.h>
#include "numhalide_all.h"

using namespace numhalide;

namespace {

// Wrap re/im value lists into a Tuple-complex 1D Func.
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

// Asymmetric complex 2D grid so a w/h swap cannot cancel out.
float grid_re(int i, int j) { return (float)(i + 10 * j) + 0.25f * (float)(i * i); }
float grid_im(int i, int j) { return 1.0f + (float)i - (float)j; }

Halide::Func make_complex_grid(int w, int h, char const* nm) {
	Halide::Buffer<float> rb(w, h), ib(w, h);
	for (int j = 0; j < h; ++j)
		for (int i = 0; i < w; ++i) {
			rb(i, j) = grid_re(i, j);
			ib(i, j) = grid_im(i, j);
		}
	Halide::Func f(nm);
	Halide::Var x, y;
	f(x, y) = Halide::Tuple(rb(x, y), ib(x, y));
	return f;
}

// Type-generic re/im extraction (the Tuple element keeps the input type).
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

Halide::Func extract_re_2d(Halide::Func z, char const* nm) {
	Halide::Func f(nm);
	Halide::Var x, y;
	f(x, y) = z(x, y)[0];
	return f;
}

Halide::Func extract_im_2d(Halide::Func z, char const* nm) {
	Halide::Func f(nm);
	Halide::Var x, y;
	f(x, y) = z(x, y)[1];
	return f;
}

} // namespace

TEST(FFTRuntime, Pow2MatchesCompileTimeTwin) {
	// n = 8 complex data through the runtime dft_1d and the compile-time
	// fft_1d_c2c (same direct-DFT formula); elementwise compare.
	int const n = 8;
	Halide::Func in = make_complex_1d(
		{1.0f, 2.0f, -1.0f, 0.5f, 3.0f, -2.0f, 0.0f, 1.5f},
		{0.5f, -1.0f, 2.0f, 0.0f, -0.5f, 1.0f, -1.5f, 2.5f}, "p2_in");

	Halide::Func rt = dft_1d(in, Halide::Expr(n), -1, Halide::Float(32), "p2_rt");
	Halide::Runtime::Buffer<float> rt_re(n), rt_im(n);
	extract_re(rt, "p2_rt_re").realize(rt_re);
	extract_im(rt, "p2_rt_im").realize(rt_im);

	Halide::Func ct = fft_1d_c2c(in, n, -1, "p2_ct");
	Halide::Runtime::Buffer<float> ct_re(n), ct_im(n);
	extract_re(ct, "p2_ct_re").realize(ct_re);
	extract_im(ct, "p2_ct_im").realize(ct_im);

	for (int i = 0; i < n; ++i) {
		EXPECT_NEAR(rt_re(i), ct_re(i), 1e-4f) << "re[" << i << "]";
		EXPECT_NEAR(rt_im(i), ct_im(i), 1e-4f) << "im[" << i << "]";
	}
}

TEST(FFTRuntime, ForwardHandPins) {
	// [1,2,3,4] real -> X = [10, -2+2j, -2, -2-2j].
	Halide::Func in4 = make_complex_1d(
		{1.0f, 2.0f, 3.0f, 4.0f}, {0.0f, 0.0f, 0.0f, 0.0f}, "hp4_in");

	Halide::Func rt4 = dft_1d(in4, Halide::Expr(4), -1, Halide::Float(32), "hp4_rt");
	Halide::Runtime::Buffer<float> re4(4), im4(4);
	extract_re(rt4, "hp4_re").realize(re4);
	extract_im(rt4, "hp4_im").realize(im4);

	float const exp_re4[] = {10.0f, -2.0f, -2.0f, -2.0f};
	float const exp_im4[] = {0.0f, 2.0f, 0.0f, -2.0f};
	for (int i = 0; i < 4; ++i) {
		EXPECT_NEAR(re4(i), exp_re4[i], 1e-4f) << "n4 re[" << i << "]";
		EXPECT_NEAR(im4(i), exp_im4[i], 1e-4f) << "n4 im[" << i << "]";
	}

	// Non-pow-2: [1,2,3] real -> X = [6, -1.5+0.8660254j, -1.5-0.8660254j].
	Halide::Func in3 = make_complex_1d(
		{1.0f, 2.0f, 3.0f}, {0.0f, 0.0f, 0.0f}, "hp3_in");

	Halide::Func rt3 = dft_1d(in3, Halide::Expr(3), -1, Halide::Float(32), "hp3_rt");
	Halide::Runtime::Buffer<float> re3(3), im3(3);
	extract_re(rt3, "hp3_re").realize(re3);
	extract_im(rt3, "hp3_im").realize(im3);

	float const s3 = 0.8660254f; // sqrt(3)/2
	float const exp_re3[] = {6.0f, -1.5f, -1.5f};
	float const exp_im3[] = {0.0f, s3, -s3};
	for (int i = 0; i < 3; ++i) {
		EXPECT_NEAR(re3(i), exp_re3[i], 1e-4f) << "n3 re[" << i << "]";
		EXPECT_NEAR(im3(i), exp_im3[i], 1e-4f) << "n3 im[" << i << "]";
	}
}

TEST(FFTRuntime, UnscaledInverseIsNTimesInput) {
	// dft_1d(+1) of dft_1d(-1) of x == n * x on a non-pow-2 n with complex data.
	int const n = 5;
	float const in_re[] = {1.0f, -2.0f, 0.5f, 3.0f, -1.0f};
	float const in_im[] = {0.5f, 1.0f, -1.5f, 0.0f, 2.0f};
	Halide::Func in = make_complex_1d(
		{1.0f, -2.0f, 0.5f, 3.0f, -1.0f},
		{0.5f, 1.0f, -1.5f, 0.0f, 2.0f}, "ui_in");

	Halide::Func fwd = dft_1d(in, Halide::Expr(n), -1, Halide::Float(32), "ui_fwd");
	Halide::Func back = dft_1d(fwd, Halide::Expr(n), 1, Halide::Float(32), "ui_back");

	Halide::Runtime::Buffer<float> re(n), im(n);
	extract_re(back, "ui_re").realize(re);
	extract_im(back, "ui_im").realize(im);

	for (int i = 0; i < n; ++i) {
		EXPECT_NEAR(re(i), (float)n * in_re[i], 1e-3f) << "re[" << i << "]";
		EXPECT_NEAR(im(i), (float)n * in_im[i], 1e-3f) << "im[" << i << "]";
	}
}

TEST(FFTRuntime, NormalizedRoundTripNonPow2) {
	// idft_1d_normalized(dft_1d(-1, x)) == x on a non-pow-2 n with nonzero
	// imaginary parts.
	int const n = 6;
	float const in_re[] = {1.0f, 2.0f, -1.0f, 0.5f, -2.5f, 3.0f};
	float const in_im[] = {-0.5f, 1.5f, 2.0f, -1.0f, 0.25f, -3.0f};
	Halide::Func in = make_complex_1d(
		{1.0f, 2.0f, -1.0f, 0.5f, -2.5f, 3.0f},
		{-0.5f, 1.5f, 2.0f, -1.0f, 0.25f, -3.0f}, "nr_in");

	Halide::Func fwd = dft_1d(in, Halide::Expr(n), -1, Halide::Float(32), "nr_fwd");
	Halide::Func back = idft_1d_normalized(fwd, Halide::Expr(n), Halide::Float(32), "nr_back");

	Halide::Runtime::Buffer<float> re(n), im(n);
	extract_re(back, "nr_re").realize(re);
	extract_im(back, "nr_im").realize(im);

	for (int i = 0; i < n; ++i) {
		EXPECT_NEAR(re(i), in_re[i], 1e-4f) << "re[" << i << "]";
		EXPECT_NEAR(im(i), in_im[i], 1e-4f) << "im[" << i << "]";
	}
}

TEST(FFTRuntime, Impulse2DNonSquareAllOnes) {
	// Impulse at (0,0) on a NON-SQUARE 3x2 grid -> every bin is 1+0j.
	int const w = 3, h = 2;
	Halide::Func in("i2_in");
	Halide::Var x, y;
	in(x, y) = Halide::Tuple(
		Halide::select(x == 0 && y == 0, 1.0f, 0.0f), 0.0f);

	Halide::Func rt = dft_2d(in, Halide::Expr(w), Halide::Expr(h), -1,
		Halide::Float(32), "i2_rt");
	Halide::Runtime::Buffer<float> re(w, h), im(w, h);
	extract_re_2d(rt, "i2_re").realize(re);
	extract_im_2d(rt, "i2_im").realize(im);

	for (int j = 0; j < h; ++j)
		for (int i = 0; i < w; ++i) {
			EXPECT_NEAR(re(i, j), 1.0f, 1e-5f) << "re(" << i << "," << j << ")";
			EXPECT_NEAR(im(i, j), 0.0f, 1e-5f) << "im(" << i << "," << j << ")";
		}
}

TEST(FFTRuntime, Normalized2DRoundTripNonSquare) {
	// idft_2d_normalized(dft_2d(-1, x)) == x on 3x2 complex data. The
	// asymmetric grid catches a w/h swap in either pass.
	int const w = 3, h = 2;
	Halide::Func in = make_complex_grid(w, h, "n2_in");

	Halide::Func fwd = dft_2d(in, Halide::Expr(w), Halide::Expr(h), -1,
		Halide::Float(32), "n2_fwd");
	Halide::Func back = idft_2d_normalized(fwd, Halide::Expr(w), Halide::Expr(h),
		Halide::Float(32), "n2_back");

	Halide::Runtime::Buffer<float> re(w, h), im(w, h);
	extract_re_2d(back, "n2_re").realize(re);
	extract_im_2d(back, "n2_im").realize(im);

	for (int j = 0; j < h; ++j)
		for (int i = 0; i < w; ++i) {
			EXPECT_NEAR(re(i, j), grid_re(i, j), 1e-3f) << "re(" << i << "," << j << ")";
			EXPECT_NEAR(im(i, j), grid_im(i, j), 1e-3f) << "im(" << i << "," << j << ")";
		}
}

TEST(FFTRuntime, Float64TypeMatchesHandDFT) {
	// dft_1d with type=Float(64) on f64 data: realizes in f64 and matches
	// the hand DFT of [1,2,3,4] to double precision.
	int const n = 4;
	Halide::Buffer<double> rb(n), ib(n);
	for (int i = 0; i < n; ++i) {
		rb(i) = (double)(i + 1);
		ib(i) = 0.0;
	}
	Halide::Func in("f64_in");
	Halide::Var x;
	in(x) = Halide::Tuple(rb(x), ib(x));

	Halide::Func rt = dft_1d(in, Halide::Expr(n), -1, Halide::Float(64), "f64_rt");
	EXPECT_TRUE(rt.types()[0] == Halide::Float(64));
	EXPECT_TRUE(rt.types()[1] == Halide::Float(64));

	Halide::Runtime::Buffer<double> re(n), im(n);
	extract_re(rt, "f64_re").realize(re);
	extract_im(rt, "f64_im").realize(im);

	double const exp_re[] = {10.0, -2.0, -2.0, -2.0};
	double const exp_im[] = {0.0, 2.0, 0.0, -2.0};
	for (int i = 0; i < n; ++i) {
		EXPECT_NEAR(re(i), exp_re[i], 1e-9) << "re[" << i << "]";
		EXPECT_NEAR(im(i), exp_im[i], 1e-9) << "im[" << i << "]";
	}
}

// ---------------------------------------------------------------------------
// Library-fix pins (same cycle): the compile-time c2c forms are direct DFTs
// and now accept ANY size; fftshift/ifftshift follow the numpy ceil/floor
// pair on odd N; real_to_complex preserves float input types.
// ---------------------------------------------------------------------------

TEST(FftRuntime, CompileTimeC2cAcceptsNonPow2) {
	// fft_1d_c2c on n=3 real [1,2,3]: X = [6, -1.5 + (sqrt(3)/2) j, -1.5 - (sqrt(3)/2) j].
	Halide::Func in = make_complex_1d({1.0f, 2.0f, 3.0f}, {0.0f, 0.0f, 0.0f}, "np2_in");
	Halide::Func ct = fft_1d_c2c(in, 3, -1, "np2_ct");
	Halide::Buffer<float> re = extract_re(ct, "np2_re").realize({3});
	Halide::Buffer<float> im = extract_im(ct, "np2_im").realize({3});
	float const s3h = 0.8660254f; // sqrt(3)/2
	EXPECT_NEAR(re(0), 6.0f, 1e-4f);
	EXPECT_NEAR(im(0), 0.0f, 1e-4f);
	EXPECT_NEAR(re(1), -1.5f, 1e-4f);
	EXPECT_NEAR(im(1), s3h, 1e-4f);
	EXPECT_NEAR(re(2), -1.5f, 1e-4f);
	EXPECT_NEAR(im(2), -s3h, 1e-4f);
}

TEST(FftRuntime, FftshiftOddNMatchesNumpyAndRoundTrips) {
	// numpy: fftshift([0,1,2,3,4]) = [3,4,0,1,2]; ifftshift undoes it.
	Halide::Func in = make_complex_1d({0.0f, 1.0f, 2.0f, 3.0f, 4.0f},
			{0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, "sh_in");
	Halide::Func sh = fftshift_1d(in, 5, "sh_fwd");
	Halide::Buffer<float> re = extract_re(sh, "sh_re").realize({5});
	float const expected[5] = {3.0f, 4.0f, 0.0f, 1.0f, 2.0f};
	for (int i = 0; i < 5; ++i)
		EXPECT_NEAR(re(i), expected[i], 1e-6f) << "fftshift[" << i << "]";

	Halide::Func rt = ifftshift_1d(sh, 5, "sh_rt");
	Halide::Buffer<float> rt_re = extract_re(rt, "sh_rt_re").realize({5});
	for (int i = 0; i < 5; ++i)
		EXPECT_NEAR(rt_re(i), (float)i, 1e-6f) << "round-trip[" << i << "]";
}

TEST(FftRuntime, Fftshift2dOddRoundTrips) {
	// 3x5 grid: ifftshift_2d(fftshift_2d(x)) == x with both axes odd.
	Halide::Func in = make_complex_grid(3, 5, "sh2_in");
	Halide::Func sh = fftshift_2d(in, /*rows*/5, /*cols*/3, "sh2_fwd");
	Halide::Func rt = ifftshift_2d(sh, 5, 3, "sh2_rt");
	Halide::Func re2("sh2_re");
	{
		Halide::Var x, y;
		re2(x, y) = rt(x, y)[0];
	}
	Halide::Buffer<float> re = re2.realize({3, 5});
	for (int j = 0; j < 5; ++j)
		for (int i = 0; i < 3; ++i)
			EXPECT_NEAR(re(i, j), grid_re(i, j), 1e-6f)
				<< "round-trip[" << i << "," << j << "]";
}

TEST(FftRuntime, RealToComplexPreservesF64) {
	Halide::Buffer<double> b(3);
	b(0) = 1.0000000001;
	b(1) = -2.5;
	b(2) = 1e-12;
	Halide::Func in("r2c64_in");
	Halide::Var x;
	in(x) = b(x);
	Halide::Func z = real_to_complex(in, "r2c64");
	ASSERT_EQ(z.types()[0], Halide::Float(64));
	ASSERT_EQ(z.types()[1], Halide::Float(64));
	Halide::Func re("r2c64_re");
	re(x) = z(x)[0];
	Halide::Buffer<double> out = re.realize({3});
	EXPECT_NEAR(out(0), 1.0000000001, 1e-15);
	EXPECT_NEAR(out(1), -2.5, 1e-15);
	EXPECT_NEAR(out(2), 1e-12, 1e-20);
}
