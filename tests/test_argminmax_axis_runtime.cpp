/// @file test_argminmax_axis_runtime.cpp
/// @brief Tests for the runtime-axis argmin_axis / argmax_axis overloads
/// (reduce.h): argmin/argmax along a RUNTIME axis of a 2-D array. axis 0
/// reduces over Halide dimension 0 (output length n1), axis 1 over
/// dimension 1 (output length n0) — the caller's output buffer bounds the
/// result. Ties keep the FIRST extremum (strict compare), matching numpy.
/// Note: unlike the compile-time reduce.h::argmin/argmax (numpy shape-axis
/// order, axis 0 = LAST Halide dimension), here axis 0 = Halide dimension 0.

#include <gtest/gtest.h>
#include "numhalide_all.h"

#include <vector>

using namespace numhalide;

namespace {

// rows[j][i] = value at f(i, j): each initializer row is a run along
// Halide dimension 0.
Halide::Func make_grid(std::vector<std::vector<float>> const& rows, char const* nm) {
	int const n1 = (int)rows.size();
	int const n0 = (int)rows[0].size();
	Halide::Buffer<float> buf(n0, n1);
	for (int j = 0; j < n1; ++j)
		for (int i = 0; i < n0; ++i)
			buf(i, j) = rows[j][i];
	Halide::Func f(nm);
	Halide::Var x, y;
	f(x, y) = buf(x, y);
	return f;
}

Halide::Func make_grid_i32(std::vector<std::vector<int>> const& rows, char const* nm) {
	int const n1 = (int)rows.size();
	int const n0 = (int)rows[0].size();
	Halide::Buffer<int> buf(n0, n1);
	for (int j = 0; j < n1; ++j)
		for (int i = 0; i < n0; ++i)
			buf(i, j) = rows[j][i];
	Halide::Func f(nm);
	Halide::Var x, y;
	f(x, y) = buf(x, y);
	return f;
}

} // namespace

TEST(ArgMinMaxAxisRuntime, ArgminBothAxesKnownValues) {
	// 3x2: f(., 0) = [5,1,3], f(., 1) = [2,4,0].
	Halide::Func f = make_grid({{5.0f, 1.0f, 3.0f}, {2.0f, 4.0f, 0.0f}}, "amna_f");

	// axis 0 reduces over dimension 0 -> one index per x1 (length n1 = 2):
	// [5,1,3] -> 1; [2,4,0] -> 2.
	Halide::Func rt0 = argmin_axis(f, Halide::Expr(3), Halide::Expr(2),
		Halide::Expr(0), Halide::Float(32), "amna_ax0");
	Halide::Runtime::Buffer<int> out0(2);
	rt0.realize(out0);
	EXPECT_EQ(out0(0), 1);
	EXPECT_EQ(out0(1), 2);

	// axis 1 reduces over dimension 1 -> one index per x0 (length n0 = 3):
	// [5,2] -> 1; [1,4] -> 0; [3,0] -> 1.
	Halide::Func rt1 = argmin_axis(f, Halide::Expr(3), Halide::Expr(2),
		Halide::Expr(1), Halide::Float(32), "amna_ax1");
	Halide::Runtime::Buffer<int> out1(3);
	rt1.realize(out1);
	EXPECT_EQ(out1(0), 1);
	EXPECT_EQ(out1(1), 0);
	EXPECT_EQ(out1(2), 1);
}

TEST(ArgMinMaxAxisRuntime, ArgmaxFirstTieWins) {
	// f(., 0) = [1,7,7]: max 7 at indices 1 and 2 -> FIRST extremum wins: 1.
	// f(., 1) = [4,2,4]: max 4 at indices 0 and 2 -> FIRST extremum wins: 0.
	Halide::Func f = make_grid({{1.0f, 7.0f, 7.0f}, {4.0f, 2.0f, 4.0f}}, "amxa_f");

	Halide::Func rt0 = argmax_axis(f, Halide::Expr(3), Halide::Expr(2),
		Halide::Expr(0), Halide::Float(32), "amxa_ax0");
	Halide::Runtime::Buffer<int> out0(2);
	rt0.realize(out0);
	EXPECT_EQ(out0(0), 1);
	EXPECT_EQ(out0(1), 0);

	// axis 1: [1,4] -> 1; [7,2] -> 0; [7,4] -> 0.
	Halide::Func rt1 = argmax_axis(f, Halide::Expr(3), Halide::Expr(2),
		Halide::Expr(1), Halide::Float(32), "amxa_ax1");
	Halide::Runtime::Buffer<int> out1(3);
	rt1.realize(out1);
	EXPECT_EQ(out1(0), 1);
	EXPECT_EQ(out1(1), 0);
	EXPECT_EQ(out1(2), 0);
}

TEST(ArgMinMaxAxisRuntime, Int32ElementTypePin) {
	// Integer elements compared in Int32 — first-tie holds for ints too:
	// f(., 0) = [9,3,3] -> 1 (tie at 1,2); f(., 1) = [0,5,0] -> 0 (tie at 0,2).
	Halide::Func f = make_grid_i32({{9, 3, 3}, {0, 5, 0}}, "amni_f");

	Halide::Func rt = argmin_axis(f, Halide::Expr(3), Halide::Expr(2),
		Halide::Expr(0), Halide::Int(32), "amni_ax0");
	Halide::Runtime::Buffer<int> out(2);
	rt.realize(out);
	EXPECT_EQ(out(0), 1);
	EXPECT_EQ(out(1), 0);
}

TEST(ArgMinMaxAxisRuntime, NonSquareBothAxes) {
	// 2x4: n0 = 2, n1 = 4 — output length swaps between 4 (axis 0) and
	// 2 (axis 1), and n0 != n1 exercises the unselected scan's bounds.
	Halide::Func f = make_grid(
		{{3.0f, 8.0f}, {7.0f, 1.0f}, {2.0f, 2.0f}, {6.0f, 4.0f}}, "amns_f");

	// axis 0 -> length 4: [3,8] -> 0; [7,1] -> 1; [2,2] -> 0 (tie); [6,4] -> 1.
	Halide::Func rt0 = argmin_axis(f, Halide::Expr(2), Halide::Expr(4),
		Halide::Expr(0), Halide::Float(32), "amns_ax0");
	Halide::Runtime::Buffer<int> out0(4);
	rt0.realize(out0);
	EXPECT_EQ(out0(0), 0);
	EXPECT_EQ(out0(1), 1);
	EXPECT_EQ(out0(2), 0);
	EXPECT_EQ(out0(3), 1);

	// axis 1 -> length 2: [3,7,2,6] -> 2; [8,1,2,4] -> 1.
	Halide::Func rt1 = argmin_axis(f, Halide::Expr(2), Halide::Expr(4),
		Halide::Expr(1), Halide::Float(32), "amns_ax1");
	Halide::Runtime::Buffer<int> out1(2);
	rt1.realize(out1);
	EXPECT_EQ(out1(0), 2);
	EXPECT_EQ(out1(1), 1);
}

TEST(ArgMinMaxAxisRuntime, F64ComputeTypePin) {
	// f64 compute type: the result Func is still Int32 and the indices match
	// the f32 computation on the same data.
	Halide::Func f = make_grid({{5.0f, 1.0f, 3.0f}, {2.0f, 4.0f, 0.0f}}, "amn64_f");

	Halide::Func rt = argmin_axis(f, Halide::Expr(3), Halide::Expr(2),
		Halide::Expr(0), Halide::Float(64), "amn64_ax0");
	ASSERT_EQ((int)rt.types().size(), 1);
	EXPECT_TRUE(rt.types()[0] == Halide::Int(32));
	Halide::Runtime::Buffer<int> out(2);
	rt.realize(out);
	EXPECT_EQ(out(0), 1);
	EXPECT_EQ(out(1), 2);

	Halide::Func rtx = argmax_axis(f, Halide::Expr(3), Halide::Expr(2),
		Halide::Expr(0), Halide::Float(64), "amx64_ax0");
	ASSERT_EQ((int)rtx.types().size(), 1);
	EXPECT_TRUE(rtx.types()[0] == Halide::Int(32));
	Halide::Runtime::Buffer<int> outx(2);
	rtx.realize(outx);
	EXPECT_EQ(outx(0), 0);  // [5,1,3] -> max 5 at 0
	EXPECT_EQ(outx(1), 1);  // [2,4,0] -> max 4 at 1
}
