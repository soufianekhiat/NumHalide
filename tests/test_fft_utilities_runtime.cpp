/// @file test_fft_utilities_runtime.cpp
/// @brief Tests for the runtime-Expr FFT utility overloads: fftshift_1d /
/// ifftshift_1d / fftshift_2d / ifftshift_2d with Halide::Expr sizes, and
/// fftfreq / rfftfreq with Halide::Expr n and d plus an explicit compute
/// type. Checked against the compile-time int twins (after the odd-N
/// fftfreq boundary fix), numpy value pins on odd and even sizes,
/// shift round-trips in 1-D and 2-D, and an f64 typing pin.

#include <gtest/gtest.h>
#include "numhalide_all.h"

using namespace numhalide;

namespace {

// Wrap re/im value lists into a Tuple-complex 1D Func. The imaginary
// channel is derived from the real one (10*re + 1) so a channel swap or
// a re/im mix-up cannot cancel out.
Halide::Func make_complex_1d(std::initializer_list<float> re, char const* nm) {
	int const n = (int)re.size();
	Halide::Buffer<float> rb(n), ib(n);
	int i = 0;
	for (float v : re) {
		rb(i) = v;
		ib(i) = 10.0f * v + 1.0f;
		++i;
	}
	Halide::Func f(nm);
	Halide::Var x;
	f(x) = Halide::Tuple(rb(x), ib(x));
	return f;
}

// Asymmetric complex 2D grid so an axis swap cannot cancel out.
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

TEST(FFTUtilitiesRuntime, Fftshift1dMatchesIntTwinAndNumpyPins) {
	// Odd n=5: numpy fftshift([0,1,2,3,4]) = [3,4,0,1,2].
	{
		int const n = 5;
		Halide::Func in = make_complex_1d({0.0f, 1.0f, 2.0f, 3.0f, 4.0f}, "s5_in");

		Halide::Func rt = fftshift_1d(in, Halide::Expr(n), "s5_rt");
		Halide::Buffer<float> rt_re = extract_re(rt, "s5_rt_re").realize({n});
		Halide::Buffer<float> rt_im = extract_im(rt, "s5_rt_im").realize({n});

		Halide::Func ct = fftshift_1d(in, n, "s5_ct");
		Halide::Buffer<float> ct_re = extract_re(ct, "s5_ct_re").realize({n});
		Halide::Buffer<float> ct_im = extract_im(ct, "s5_ct_im").realize({n});

		float const expected[5] = {3.0f, 4.0f, 0.0f, 1.0f, 2.0f};
		for (int i = 0; i < n; ++i) {
			EXPECT_NEAR(rt_re(i), expected[i], 1e-6f) << "n5 pin re[" << i << "]";
			EXPECT_NEAR(rt_re(i), ct_re(i), 1e-6f) << "n5 twin re[" << i << "]";
			EXPECT_NEAR(rt_im(i), ct_im(i), 1e-6f) << "n5 twin im[" << i << "]";
		}
	}

	// Even n=4: fftshift([0,1,2,3]) = [2,3,0,1].
	{
		int const n = 4;
		Halide::Func in = make_complex_1d({0.0f, 1.0f, 2.0f, 3.0f}, "s4_in");

		Halide::Func rt = fftshift_1d(in, Halide::Expr(n), "s4_rt");
		Halide::Buffer<float> rt_re = extract_re(rt, "s4_rt_re").realize({n});
		Halide::Buffer<float> rt_im = extract_im(rt, "s4_rt_im").realize({n});

		Halide::Func ct = fftshift_1d(in, n, "s4_ct");
		Halide::Buffer<float> ct_re = extract_re(ct, "s4_ct_re").realize({n});
		Halide::Buffer<float> ct_im = extract_im(ct, "s4_ct_im").realize({n});

		float const expected[4] = {2.0f, 3.0f, 0.0f, 1.0f};
		for (int i = 0; i < n; ++i) {
			EXPECT_NEAR(rt_re(i), expected[i], 1e-6f) << "n4 pin re[" << i << "]";
			EXPECT_NEAR(rt_re(i), ct_re(i), 1e-6f) << "n4 twin re[" << i << "]";
			EXPECT_NEAR(rt_im(i), ct_im(i), 1e-6f) << "n4 twin im[" << i << "]";
		}
	}
}

TEST(FFTUtilitiesRuntime, Ifftshift1dRoundTripsOddAndEven) {
	// ifftshift_1d(fftshift_1d(x)) == x for both odd and even sizes
	// (only the ceil/floor gather pair round-trips on odd n).
	for (int n : {5, 4}) {
		Halide::Func in = (n == 5)
			? make_complex_1d({0.0f, 1.0f, 2.0f, 3.0f, 4.0f}, "rt5_in")
			: make_complex_1d({0.0f, 1.0f, 2.0f, 3.0f}, "rt4_in");

		Halide::Func sh = fftshift_1d(in, Halide::Expr(n), "rt_fwd");
		Halide::Func back = ifftshift_1d(sh, Halide::Expr(n), "rt_back");
		Halide::Buffer<float> re = extract_re(back, "rt_re").realize({n});
		Halide::Buffer<float> im = extract_im(back, "rt_im").realize({n});

		for (int i = 0; i < n; ++i) {
			EXPECT_NEAR(re(i), (float)i, 1e-6f) << "n" << n << " round-trip re[" << i << "]";
			EXPECT_NEAR(im(i), 10.0f * (float)i + 1.0f, 1e-6f)
				<< "n" << n << " round-trip im[" << i << "]";
		}
	}
}

TEST(FFTUtilitiesRuntime, Fftshift2dRoundTripsBothAxesOdd) {
	// 3x5 grid (x extent 3, y extent 5 — both odd):
	// ifftshift_2d(fftshift_2d(x)) == x. NH signature is (rows, cols) =
	// (y extent, x extent).
	int const w = 3, h = 5;
	Halide::Func in = make_complex_grid(w, h, "g35_in");

	Halide::Func sh = fftshift_2d(in, Halide::Expr(h), Halide::Expr(w), "g35_fwd");
	Halide::Func back = ifftshift_2d(sh, Halide::Expr(h), Halide::Expr(w), "g35_back");

	Halide::Buffer<float> re = extract_re_2d(back, "g35_re").realize({w, h});
	Halide::Buffer<float> im = extract_im_2d(back, "g35_im").realize({w, h});

	for (int j = 0; j < h; ++j)
		for (int i = 0; i < w; ++i) {
			EXPECT_NEAR(re(i, j), grid_re(i, j), 1e-6f)
				<< "round-trip re(" << i << "," << j << ")";
			EXPECT_NEAR(im(i, j), grid_im(i, j), 1e-6f)
				<< "round-trip im(" << i << "," << j << ")";
		}
}

TEST(FFTUtilitiesRuntime, Fftshift2dValuePin3x2) {
	// 3 wide (cols) x 2 tall (rows), v(i,j) = i + 3j:
	//   [[0,1,2],    fftshift    [[5,3,4],
	//    [3,4,5]]      ---->      [2,0,1]]
	// (numpy.fft.fftshift on the (2,3) array; gather offsets ceil(3/2)=2
	// on x, ceil(2/2)=1 on y). Matches the int twin as well.
	int const w = 3, h = 2;
	Halide::Buffer<float> vb(w, h);
	for (int j = 0; j < h; ++j)
		for (int i = 0; i < w; ++i)
			vb(i, j) = (float)(i + 3 * j);
	Halide::Func in("p32_in");
	Halide::Var x, y;
	in(x, y) = Halide::Tuple(vb(x, y), 100.0f + vb(x, y));

	Halide::Func rt = fftshift_2d(in, Halide::Expr(h), Halide::Expr(w), "p32_rt");
	Halide::Buffer<float> re = extract_re_2d(rt, "p32_re").realize({w, h});
	Halide::Buffer<float> im = extract_im_2d(rt, "p32_im").realize({w, h});

	Halide::Func ct = fftshift_2d(in, h, w, "p32_ct");
	Halide::Buffer<float> ct_re = extract_re_2d(ct, "p32_ct_re").realize({w, h});

	float const expected[2][3] = {{5.0f, 3.0f, 4.0f}, {2.0f, 0.0f, 1.0f}};
	for (int j = 0; j < h; ++j)
		for (int i = 0; i < w; ++i) {
			EXPECT_NEAR(re(i, j), expected[j][i], 1e-6f)
				<< "pin re(" << i << "," << j << ")";
			EXPECT_NEAR(im(i, j), 100.0f + expected[j][i], 1e-6f)
				<< "pin im(" << i << "," << j << ")";
			EXPECT_NEAR(re(i, j), ct_re(i, j), 1e-6f)
				<< "twin re(" << i << "," << j << ")";
		}
}

TEST(FFTUtilitiesRuntime, FftfreqExprMatchesNumpyAndIntTwin) {
	// Even n=4, d=1: numpy fftfreq(4) = [0, 0.25, -0.5, -0.25].
	{
		int const n = 4;
		Halide::Func rt = fftfreq(Halide::Expr(n), Halide::Expr(1.0f),
			Halide::Float(32), "ff4_rt");
		Halide::Buffer<float> out = rt.realize({n});

		Halide::Func ct = fftfreq(n, 1.0f, "ff4_ct");
		Halide::Buffer<float> ct_out = ct.realize({n});

		float const expected[4] = {0.0f, 0.25f, -0.5f, -0.25f};
		for (int i = 0; i < n; ++i) {
			EXPECT_NEAR(out(i), expected[i], 1e-6f) << "n4 pin [" << i << "]";
			EXPECT_NEAR(out(i), ct_out(i), 1e-6f) << "n4 twin [" << i << "]";
		}
	}

	// Odd n=5, d=0.5: numpy fftfreq(5, 0.5) = [0, 0.4, 0.8, -0.8, -0.4].
	// k=2 is a POSITIVE frequency (positive branch through (n-1)//2 = 2);
	// the old n/2 boundary wrongly sent it to the negative branch.
	{
		int const n = 5;
		Halide::Func rt = fftfreq(Halide::Expr(n), Halide::Expr(0.5f),
			Halide::Float(32), "ff5_rt");
		Halide::Buffer<float> out = rt.realize({n});

		Halide::Func ct = fftfreq(n, 0.5f, "ff5_ct");
		Halide::Buffer<float> ct_out = ct.realize({n});

		float const expected[5] = {0.0f, 0.4f, 0.8f, -0.8f, -0.4f};
		for (int i = 0; i < n; ++i) {
			EXPECT_NEAR(out(i), expected[i], 1e-6f) << "n5 pin [" << i << "]";
			EXPECT_NEAR(out(i), ct_out(i), 1e-6f) << "n5 twin [" << i << "]";
		}
	}
}

TEST(FFTUtilitiesRuntime, RfftfreqExprMatchesNumpyAndIntTwin) {
	// n=5, d=0.5: numpy rfftfreq(5, 0.5) = [0, 0.4, 0.8] (n//2+1 = 3 bins).
	int const n = 5;
	int const bins = n / 2 + 1;
	Halide::Func rt = rfftfreq(Halide::Expr(n), Halide::Expr(0.5f),
		Halide::Float(32), "rf5_rt");
	Halide::Buffer<float> out = rt.realize({bins});

	Halide::Func ct = rfftfreq(n, 0.5f, "rf5_ct");
	Halide::Buffer<float> ct_out = ct.realize({bins});

	float const expected[3] = {0.0f, 0.4f, 0.8f};
	for (int i = 0; i < bins; ++i) {
		EXPECT_NEAR(out(i), expected[i], 1e-6f) << "pin [" << i << "]";
		EXPECT_NEAR(out(i), ct_out(i), 1e-6f) << "twin [" << i << "]";
	}
}

TEST(FFTUtilitiesRuntime, FftfreqFloat64TypeAndValues) {
	// type=Float(64) realizes as f64 and matches numpy fftfreq(4) to
	// double precision.
	int const n = 4;
	Halide::Func rt = fftfreq(Halide::Expr(n), Halide::Expr(1.0),
		Halide::Float(64), "ff64_rt");
	EXPECT_TRUE(rt.types()[0] == Halide::Float(64));

	Halide::Buffer<double> out = rt.realize({n});

	double const expected[4] = {0.0, 0.25, -0.5, -0.25};
	for (int i = 0; i < n; ++i)
		EXPECT_NEAR(out(i), expected[i], 1e-15) << "f64 [" << i << "]";
}
