/// @file test_stencil_histogram_runtime.cpp
/// @brief Tests for the runtime-Expr stencil / histogram / global-sort
/// overloads (jacobi_step / heat_diffusion_step with Halide::Expr extents,
/// cumsum with a Halide::Expr length, global_argmin / global_argmax /
/// searchsorted / searchsorted_single / stats::digitize with Halide::Expr
/// sizes). Each runtime form is checked against known values and, where a
/// compile-time twin exists, against that twin on identical data.

#include <gtest/gtest.h>
#include "numhalide_all.h"

using namespace numhalide;

namespace {

Halide::Func make_coeffs(std::initializer_list<float> vals, char const* nm) {
	Halide::Buffer<float> buf((int)vals.size());
	int i = 0;
	for (float v : vals) buf(i++) = v;
	Halide::Func f(nm);
	Halide::Var x;
	f(x) = buf(x);
	return f;
}

// Asymmetric 2D grid so a rows/cols swap cannot cancel out.
Halide::Func make_grid(int cols, int rows, char const* nm) {
	Halide::Buffer<float> buf(cols, rows);
	for (int j = 0; j < rows; ++j)
		for (int i = 0; i < cols; ++i)
			buf(i, j) = (float)(i + 10 * j) + 0.5f * (float)(i * i);
	Halide::Func f(nm);
	Halide::Var x, y;
	f(x, y) = buf(x, y);
	return f;
}

} // namespace

TEST(StencilHistogramRuntime, JacobiMatchesTwinAndKnownValue) {
	// Non-square: rows = 3 (y extent), cols = 4 (x extent).
	int const rows = 3, cols = 4;
	Halide::Func f = make_grid(cols, rows, "jac_f");

	Halide::Func rt = jacobi_step(f, Halide::Expr(rows), Halide::Expr(cols), "jac_rt");
	Halide::Runtime::Buffer<float> out(cols, rows);
	rt.realize(out);

	// Compile-time twin on the same data. shape_t = {rows, cols}.
	shape_t s = {rows, cols};
	Halide::Func ct = jacobi_step(f, s, "jac_ct");
	Halide::Runtime::Buffer<float> out_ct(cols, rows);
	ct.realize(out_ct);

	for (int j = 0; j < rows; ++j)
		for (int i = 0; i < cols; ++i)
			EXPECT_NEAR(out(i, j), out_ct(i, j), 1e-5f) << "twin(" << i << "," << j << ")";

	// Interior known value at (1,1): average of the 4 neighbors.
	auto v = [](int i, int j) { return (float)(i + 10 * j) + 0.5f * (float)(i * i); };
	float expected = 0.25f * (v(0, 1) + v(2, 1) + v(1, 0) + v(1, 2));
	EXPECT_NEAR(out(1, 1), expected, 1e-5f);

	// Corner known value at (0,0): repeat-edge boundary clamps x-1 and y-1 to 0.
	float corner = 0.25f * (v(0, 0) + v(1, 0) + v(0, 0) + v(0, 1));
	EXPECT_NEAR(out(0, 0), corner, 1e-5f);
}

TEST(StencilHistogramRuntime, HeatMatchesTwinAndBoundary) {
	int const rows = 3, cols = 4;
	float const dt = 0.1f, alpha = 0.5f;
	Halide::Func f = make_grid(cols, rows, "heat_f");

	Halide::Func rt = heat_diffusion_step(f, Halide::Expr(rows), Halide::Expr(cols),
		Halide::Expr(dt), Halide::Expr(alpha), "heat_rt2");
	Halide::Runtime::Buffer<float> out(cols, rows);
	rt.realize(out);

	shape_t s = {rows, cols};
	Halide::Func ct = heat_diffusion_step(f, s, dt, alpha, "heat_ct");
	Halide::Runtime::Buffer<float> out_ct(cols, rows);
	ct.realize(out_ct);

	for (int j = 0; j < rows; ++j)
		for (int i = 0; i < cols; ++i)
			EXPECT_NEAR(out(i, j), out_ct(i, j), 1e-4f) << "twin(" << i << "," << j << ")";

	// Corner known value at (0,0): clamped laplacian, then explicit Euler.
	auto v = [](int i, int j) { return (float)(i + 10 * j) + 0.5f * (float)(i * i); };
	float lap = v(0, 0) + v(1, 0) + v(0, 0) + v(0, 1) - 4.0f * v(0, 0);
	EXPECT_NEAR(out(0, 0), v(0, 0) + alpha * dt * lap, 1e-4f);
}

TEST(StencilHistogramRuntime, CumsumKnownAndTwin) {
	// [1,2,3,4,5] -> [1,3,6,10,15]
	Halide::Func f = make_coeffs({1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, "cs_f");

	Halide::Func rt = cumsum(f, Halide::Expr(5), "cs_rt");
	Halide::Runtime::Buffer<float> out(5);
	rt.realize(out);

	float const expected[] = {1.0f, 3.0f, 6.0f, 10.0f, 15.0f};
	for (int i = 0; i < 5; ++i)
		EXPECT_NEAR(out(i), expected[i], 1e-6f) << "rt[" << i << "]";

	shape_t s = {5};
	Halide::Func ct = cumsum(f, s);
	Halide::Runtime::Buffer<float> out_ct(5);
	ct.realize(out_ct);
	for (int i = 0; i < 5; ++i)
		EXPECT_NEAR(out(i), out_ct(i), 1e-6f) << "twin[" << i << "]";
}

TEST(StencilHistogramRuntime, GlobalArgminFirstTieAndTwin) {
	// Min value 1 appears at indices 1 and 3 -> FIRST extremum wins: 1.
	Halide::Func f = make_coeffs({3.0f, 1.0f, 4.0f, 1.0f, 5.0f}, "amin_f");

	Halide::Func rt = global_argmin(f, Halide::Expr(5), "amin_rt");
	Halide::Runtime::Buffer<int> out(1);
	rt.realize(out);
	EXPECT_EQ(out(0), 1);

	shape_t s = {5};
	Halide::Func ct = global_argmin(f, s, "amin_ct");
	Halide::Runtime::Buffer<int> out_ct(1);
	ct.realize(out_ct);
	EXPECT_EQ(out(0), out_ct(0));
}

TEST(StencilHistogramRuntime, GlobalArgmaxFirstTieAndTwin) {
	// Max value 7 appears at indices 1 and 3 -> FIRST extremum wins: 1.
	Halide::Func f = make_coeffs({2.0f, 7.0f, 4.0f, 7.0f, 1.0f}, "amax_f");

	Halide::Func rt = global_argmax(f, Halide::Expr(5), "amax_rt");
	Halide::Runtime::Buffer<int> out(1);
	rt.realize(out);
	EXPECT_EQ(out(0), 1);

	shape_t s = {5};
	Halide::Func ct = global_argmax(f, s, "amax_ct");
	Halide::Runtime::Buffer<int> out_ct(1);
	ct.realize(out_ct);
	EXPECT_EQ(out(0), out_ct(0));
}

TEST(StencilHistogramRuntime, SearchsortedPinsAndTwin) {
	// sorted = [1,3,5,7]; side='left' insertion indices.
	Halide::Func sorted = make_coeffs({1.0f, 3.0f, 5.0f, 7.0f}, "ss_sorted");
	Halide::Func vals   = make_coeffs({0.0f, 1.0f, 4.0f, 7.0f, 9.0f}, "ss_vals");

	Halide::Func rt = searchsorted(sorted, vals, Halide::Expr(4), "ss_rt");
	Halide::Runtime::Buffer<int> out(5);
	rt.realize(out);

	int const expected[] = {0, 0, 2, 3, 4};
	for (int i = 0; i < 5; ++i)
		EXPECT_EQ(out(i), expected[i]) << "rt[" << i << "]";

	// Compile-time twin (binary search) on the same data.
	Halide::Func ct = searchsorted(sorted, vals, 4, 5, "ss_ct");
	Halide::Runtime::Buffer<int> out_ct(5);
	ct.realize(out_ct);
	for (int i = 0; i < 5; ++i)
		EXPECT_EQ(out(i), out_ct(i)) << "twin[" << i << "]";
}

TEST(StencilHistogramRuntime, SearchsortedSinglePinAndTwin) {
	Halide::Func sorted = make_coeffs({1.0f, 3.0f, 5.0f, 7.0f}, "sss_sorted");

	Halide::Func rt = searchsorted_single(sorted, Halide::Expr(5.0f),
		Halide::Expr(4), "sss_rt");
	Halide::Runtime::Buffer<int> out(1);
	rt.realize(out);
	EXPECT_EQ(out(0), 2);

	Halide::Func ct = searchsorted_single(sorted, Halide::Expr(5.0f), 4, "sss_ct");
	Halide::Runtime::Buffer<int> out_ct(1);
	ct.realize(out_ct);
	EXPECT_EQ(out(0), out_ct(0));
}

TEST(StencilHistogramRuntime, DigitizePinsAndTwin) {
	// bins = [1,2,3]; right=false counts bins <= v: 0.5->0, 1.0->1, 2.5->2, 3.5->3.
	Halide::Func vals = make_coeffs({0.5f, 1.0f, 2.5f, 3.5f}, "dg_vals");
	Halide::Func bins = make_coeffs({1.0f, 2.0f, 3.0f}, "dg_bins");

	Halide::Func rt = stats::digitize(vals, bins, Halide::Expr(3), false, "dg_rt");
	Halide::Runtime::Buffer<int> out(4);
	rt.realize(out);

	int const expected[] = {0, 1, 2, 3};
	for (int i = 0; i < 4; ++i)
		EXPECT_EQ(out(i), expected[i]) << "rt[" << i << "]";

	shape_t s = {4};
	Halide::Func ct = stats::digitize(vals, bins, s, 3, false, "dg_ct");
	Halide::Runtime::Buffer<int> out_ct(4);
	ct.realize(out_ct);
	for (int i = 0; i < 4; ++i)
		EXPECT_EQ(out(i), out_ct(i)) << "twin[" << i << "]";
}
