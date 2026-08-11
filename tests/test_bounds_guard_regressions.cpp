/// @file test_bounds_guard_regressions.cpp
/// @brief Value-pins for every site rewritten off the select+clamp pattern
///
/// The hazard: select(cond, f(clamp(idx, lo, hi)), other) — when cond proves
/// the inner clamp redundant, the simplifier strips the clamp, and the CMOV
/// lowering evaluates BOTH arms, reading the unclamped index out of bounds.
/// The rewritten sites use either a MULTIPLIED 0/1 indicator with an
/// unconditional clamp (polymul, polyadd, polysub, polyint, correlate1d,
/// convolve1d "full", concat_*, pad, pad_2d) or pure-init + bounded-RDom
/// (sort_1d_fast / argsort_1d_fast padding, where the sentinel is +/-inf and
/// the 0/1-indicator form would compute inf*0 = NaN).
/// These tests pin the VALUES so any re-simplification of the guards shows
/// up as a wrong number, not just a sanitizer hit.

#include <gtest/gtest.h>
#include "numhalide_all.h"

#include <cmath>
#include <limits>

using namespace numhalide;

namespace {

Halide::Func make_vecf(std::initializer_list<float> vals, const std::string& n)
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

Halide::Func make_matf(int rows, int cols,
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

} // namespace

TEST(BoundsGuards, PolymulValues) {
	// Highest-first convention.
	// (x+1)(x-1) = x^2 - 1 -> [1, 0, -1]
	{
		auto a = make_vecf({ 1.0f,  1.0f }, "a_pm1");
		auto b = make_vecf({ 1.0f, -1.0f }, "b_pm1");
		auto r = polymul(a, 2, b, 2, "pm1");
		Halide::Runtime::Buffer<float> out(3);
		r.realize(out);
		EXPECT_NEAR(out(0),  1.0f, 1e-6f);
		EXPECT_NEAR(out(1),  0.0f, 1e-6f);
		EXPECT_NEAR(out(2), -1.0f, 1e-6f);
	}
	// Unequal lengths: (2x+3)(x^2+1) = 2x^3 + 3x^2 + 2x + 3 -> [2, 3, 2, 3]
	{
		auto a = make_vecf({ 2.0f, 3.0f }, "a_pm2");
		auto b = make_vecf({ 1.0f, 0.0f, 1.0f }, "b_pm2");
		auto r = polymul(a, 2, b, 3, "pm2");
		Halide::Runtime::Buffer<float> out(4);
		r.realize(out);
		EXPECT_NEAR(out(0), 2.0f, 1e-6f);
		EXPECT_NEAR(out(1), 3.0f, 1e-6f);
		EXPECT_NEAR(out(2), 2.0f, 1e-6f);
		EXPECT_NEAR(out(3), 3.0f, 1e-6f);
	}
}

TEST(BoundsGuards, PolyaddPolysubAlignment) {
	// Highest-first, right-aligned: result[i] = a[i-(nout-na)] + b[i-(nout-nb)]
	// a=[1,2] (na=2, offset 1), b=[1,0,3] (nb=3, offset 0), nout=3:
	//   add: r[0]=b[0]=1, r[1]=a[0]+b[1]=1, r[2]=a[1]+b[2]=5 -> [1, 1, 5]
	//   sub: r[0]=0-1=-1, r[1]=1-0=1, r[2]=2-3=-1 -> [-1, 1, -1]
	auto a = make_vecf({ 1.0f, 2.0f }, "a_pa");
	auto b = make_vecf({ 1.0f, 0.0f, 3.0f }, "b_pa");

	auto radd = polyadd(a, 2, b, 3, "pa_add");
	Halide::Runtime::Buffer<float> out_add(3);
	radd.realize(out_add);
	EXPECT_NEAR(out_add(0), 1.0f, 1e-6f);
	EXPECT_NEAR(out_add(1), 1.0f, 1e-6f);
	EXPECT_NEAR(out_add(2), 5.0f, 1e-6f);

	auto rsub = polysub(a, 2, b, 3, "pa_sub");
	Halide::Runtime::Buffer<float> out_sub(3);
	rsub.realize(out_sub);
	EXPECT_NEAR(out_sub(0), -1.0f, 1e-6f);
	EXPECT_NEAR(out_sub(1),  1.0f, 1e-6f);
	EXPECT_NEAR(out_sub(2), -1.0f, 1e-6f);
}

TEST(BoundsGuards, PolyintValues) {
	// Verified against the formula in src/polynomial.h:
	//   result[i] = a[i] / (na - i) for i < na, result[na] = k.
	// a=[2,4] (2x+4), k=7: r[0]=2/2=1, r[1]=4/1=4, r[2]=k=7 -> [1, 4, 7]
	// (integral of 2x+4 is x^2 + 4x + 7 — same by hand.)
	auto a = make_vecf({ 2.0f, 4.0f }, "a_pi");
	auto r = polyint(a, 2, 7.0f, "pi_int");
	Halide::Runtime::Buffer<float> out(3);
	r.realize(out);
	EXPECT_NEAR(out(0), 1.0f, 1e-6f);
	EXPECT_NEAR(out(1), 4.0f, 1e-6f);
	EXPECT_NEAR(out(2), 7.0f, 1e-6f);
}

TEST(BoundsGuards, Correlate1DSame) {
	// DERIVED FROM src/numeric.h: mode "same" uses pad = (nv-1)/2 and
	// c[k] = sum_n a[n + k - pad] * v[n], zero outside a.
	// a=[1,2,3], v=[1,1], nv=2 -> pad = 0:
	//   c[0] = a[0]+a[1] = 3, c[1] = a[1]+a[2] = 5, c[2] = a[2]+0 = 3
	auto a = make_vecf({ 1.0f, 2.0f, 3.0f }, "a_c1");
	auto v = make_vecf({ 1.0f, 1.0f }, "v_c1");
	auto r = correlate1d(a, v, 3, 2, "same", "corr_same");
	Halide::Runtime::Buffer<float> out(3);
	r.realize(out);
	EXPECT_NEAR(out(0), 3.0f, 1e-6f);
	EXPECT_NEAR(out(1), 5.0f, 1e-6f);
	EXPECT_NEAR(out(2), 3.0f, 1e-6f);
}

TEST(BoundsGuards, Convolve1DFull) {
	// [1,2,3] conv [1,1], mode "full" (zero-pad) -> [1, 3, 5, 3]
	shape_t s = { 3 };
	auto input = make_vecf({ 1.0f, 2.0f, 3.0f }, "in_cf");
	auto kernel = make_vecf({ 1.0f, 1.0f }, "k_cf");
	auto r = convolve1d(input, s, kernel, 2, "full", "conv_full");
	shape_t so = infer_convolve1d(s, 2, "full");
	EXPECT_EQ(so.extents[0], 4);
	Halide::Runtime::Buffer<float> out(4);
	r.realize(out);
	EXPECT_NEAR(out(0), 1.0f, 1e-6f);
	EXPECT_NEAR(out(1), 3.0f, 1e-6f);
	EXPECT_NEAR(out(2), 5.0f, 1e-6f);
	EXPECT_NEAR(out(3), 3.0f, 1e-6f);
}

TEST(BoundsGuards, Concat1DAndConcatenate) {
	// concat_1d: [1,2] ++ [3,4,5] -> [1,2,3,4,5]
	{
		auto f1 = make_vecf({ 1.0f, 2.0f }, "f1_cc");
		auto f2 = make_vecf({ 3.0f, 4.0f, 5.0f }, "f2_cc");
		auto r = concat_1d(f1, 2, f2, 3, "cc12");
		Halide::Runtime::Buffer<float> out(5);
		r.realize(out);
		for (int i = 0; i < 5; ++i)
			EXPECT_NEAR(out(i), static_cast<float>(i + 1), 1e-6f);
	}
	// concatenate: [1] ++ [2,3] ++ [4] -> [1,2,3,4]
	{
		std::vector<Halide::Func> fs = {
			make_vecf({ 1.0f }, "g1_cc"),
			make_vecf({ 2.0f, 3.0f }, "g2_cc"),
			make_vecf({ 4.0f }, "g3_cc"),
		};
		auto r = concatenate(fs, { 1, 2, 1 }, "cc3");
		Halide::Runtime::Buffer<float> out(4);
		r.realize(out);
		for (int i = 0; i < 4; ++i)
			EXPECT_NEAR(out(i), static_cast<float>(i + 1), 1e-6f);
	}
}

TEST(BoundsGuards, Concat2DBothAxes) {
	// f1 = [[1,2],[3,4]], f2 = [[5,6],[7,8]]
	shape_t s = { 2, 2 };
	auto f1 = make_matf(2, 2, { 1, 2, 3, 4 }, "f1_c2");
	auto f2 = make_matf(2, 2, { 5, 6, 7, 8 }, "f2_c2");

	// axis 0: stack rows -> 4x2, rows [1,2],[3,4],[5,6],[7,8]
	{
		auto r = concat_2d(f1, s, f2, s, 0, "c2_ax0");
		Halide::Runtime::Buffer<float> out(2, 4);
		r.realize(out);
		const float expected[4][2] = { { 1, 2 }, { 3, 4 }, { 5, 6 }, { 7, 8 } };
		for (int row = 0; row < 4; ++row)
			for (int col = 0; col < 2; ++col)
				EXPECT_NEAR(out(col, row), expected[row][col], 1e-6f)
					<< "axis0 at row=" << row << " col=" << col;
	}
	// axis 1: stack cols -> 2x4, rows [1,2,5,6],[3,4,7,8]
	{
		auto r = concat_2d(f1, s, f2, s, 1, "c2_ax1");
		Halide::Runtime::Buffer<float> out(4, 2);
		r.realize(out);
		const float expected[2][4] = { { 1, 2, 5, 6 }, { 3, 4, 7, 8 } };
		for (int row = 0; row < 2; ++row)
			for (int col = 0; col < 4; ++col)
				EXPECT_NEAR(out(col, row), expected[row][col], 1e-6f)
					<< "axis1 at row=" << row << " col=" << col;
	}
}

TEST(BoundsGuards, PadConstant1DAnd2D) {
	// pad([1,2,3], before=2, after=1, value=9) -> [9,9,1,2,3,9]
	{
		auto f = make_vecf({ 1.0f, 2.0f, 3.0f }, "f_pad");
		auto r = pad(f, 3, 2, 1, 9.0f, "pad1");
		Halide::Runtime::Buffer<float> out(6);
		r.realize(out);
		const float expected[6] = { 9, 9, 1, 2, 3, 9 };
		for (int i = 0; i < 6; ++i)
			EXPECT_NEAR(out(i), expected[i], 1e-6f) << "pad at " << i;
	}
	// pad_2d([[1,2],[3,4]], 1 all around, value=7) -> 4x4 with a 7 ring
	{
		shape_t s = { 2, 2 };
		auto f = make_matf(2, 2, { 1, 2, 3, 4 }, "f_pad2");
		auto r = pad_2d(f, s, 1, 1, 1, 1, 7.0f, "pad2");
		Halide::Runtime::Buffer<float> out(4, 4);
		r.realize(out);
		for (int row = 0; row < 4; ++row)
			for (int col = 0; col < 4; ++col) {
				bool interior = row >= 1 && row <= 2 && col >= 1 && col <= 2;
				float expected = interior
					? static_cast<float>((row - 1) * 2 + (col - 1) + 1)
					: 7.0f;
				EXPECT_NEAR(out(col, row), expected, 1e-6f)
					<< "pad2d at row=" << row << " col=" << col;
			}
	}
}

TEST(BoundsGuards, SortFastAndArgsortFastPadded) {
	// n=3 pads to 4 with a +inf sentinel (ascending).
	auto f = make_vecf({ 0.3f, 0.1f, 0.2f }, "f_sf");

	auto s = sort_1d_fast(f, 3, true, "sf1");
	Halide::Runtime::Buffer<float> out_s(3);
	s.realize(out_s);
	EXPECT_NEAR(out_s(0), 0.1f, 1e-6f);
	EXPECT_NEAR(out_s(1), 0.2f, 1e-6f);
	EXPECT_NEAR(out_s(2), 0.3f, 1e-6f);

	auto a = argsort_1d_fast(f, 3, true, "af1");
	Halide::Runtime::Buffer<int32_t> out_a(3);
	a.realize(out_a);
	EXPECT_EQ(out_a(0), 1);
	EXPECT_EQ(out_a(1), 2);
	EXPECT_EQ(out_a(2), 0);
}

TEST(BoundsGuards, SortFastInfRegression) {
	// The padding stage previously guarded the sentinel with a multiplied 0/1
	// indicator; with a +/-inf payload that computes inf*0 = NaN, and one NaN
	// poisons the whole comparison network. Now pure-init + bounded-RDom.
	// [0.5, -inf, inf] must sort to [-inf, 0.5, inf] with NO NaN anywhere,
	// including the sentinel lane (realized here as element 3 of the padded 4).
	const float inf = std::numeric_limits<float>::infinity();
	Halide::Buffer<float> buf(3);
	buf(0) = 0.5f;
	buf(1) = -inf;
	buf(2) = inf;
	Halide::Func f("f_sfi");
	Halide::Var x;
	f(x) = buf(x);

	auto s = sort_1d_fast(f, 3, true, "sfi");
	Halide::Runtime::Buffer<float> out(4);
	s.realize(out);
	for (int i = 0; i < 4; ++i)
		EXPECT_FALSE(std::isnan(out(i))) << "NaN at " << i;
	EXPECT_EQ(out(0), -inf);
	EXPECT_EQ(out(1), 0.5f);
	EXPECT_EQ(out(2), inf);

	// argsort over the same data: index lanes derived by simulating the
	// source's exact network in double (bitonic is not stable, so the tie
	// between the data +inf and the +inf sentinel is a network property, not
	// a convention): first three lanes = [1, 0, 2].
	auto ar = argsort_1d_fast(f, 3, true, "afi");
	Halide::Runtime::Buffer<int32_t> out_a(3);
	ar.realize(out_a);
	EXPECT_EQ(out_a(0), 1);
	EXPECT_EQ(out_a(1), 0);
	EXPECT_EQ(out_a(2), 2);
}
