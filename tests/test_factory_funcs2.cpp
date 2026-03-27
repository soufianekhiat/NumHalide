/// @file test_factory_funcs2.cpp
/// @brief Tests for newer factory functions: logspace, geomspace, eye (with k), tri, vander, empty_like

#include <gtest/gtest.h>
#include "numhalide_all.h"
#include <cmath>

using namespace numhalide;

// -----------------------------------------------------------------------------
// Logspace
// -----------------------------------------------------------------------------

TEST(FactoryFuncs2, Logspace) {
	// logspace(type, 0, 2, 3) -> [10^0, 10^1, 10^2] = [1, 10, 100]
	Halide::Func f = logspace(Halide::Float(32), 0.0f, 2.0f, 3);

	Halide::Runtime::Buffer<float> out(3);
	f.realize(out);

	EXPECT_NEAR(out(0), 1.0f, 1e-4f);
	EXPECT_NEAR(out(1), 10.0f, 1e-3f);
	EXPECT_NEAR(out(2), 100.0f, 1e-2f);
}

TEST(FactoryFuncs2, LogspaceBase2) {
	// logspace(type, 0, 3, 4, base=2) -> [2^0, 2^1, 2^2, 2^3] = [1, 2, 4, 8]
	Halide::Func f = logspace(Halide::Float(32), 0.0f, 3.0f, 4, 2.0f);

	Halide::Runtime::Buffer<float> out(4);
	f.realize(out);

	EXPECT_NEAR(out(0), 1.0f, 1e-4f);
	EXPECT_NEAR(out(1), 2.0f, 1e-4f);
	EXPECT_NEAR(out(2), 4.0f, 1e-4f);
	EXPECT_NEAR(out(3), 8.0f, 1e-4f);
}

// -----------------------------------------------------------------------------
// Geomspace
// -----------------------------------------------------------------------------

TEST(FactoryFuncs2, Geomspace) {
	// geomspace(type, 1, 1000, 4) -> [1, 10, 100, 1000]
	Halide::Func f = geomspace(Halide::Float(32), 1.0f, 1000.0f, 4);

	Halide::Runtime::Buffer<float> out(4);
	f.realize(out);

	EXPECT_NEAR(out(0), 1.0f, 1e-3f);
	EXPECT_NEAR(out(1), 10.0f, 1e-3f);
	EXPECT_NEAR(out(2), 100.0f, 1e-2f);
	EXPECT_NEAR(out(3), 1000.0f, 1e-1f);
}

// -----------------------------------------------------------------------------
// Eye
// -----------------------------------------------------------------------------

TEST(FactoryFuncs2, EyeMainDiag) {
	// eye(type, 3) -> 3x3 identity matrix
	// In Halide: ret(col, row), so realize as Buffer<float>(cols=3, rows=3)
	Halide::Func f = eye(Halide::Float(32), 3);

	Halide::Runtime::Buffer<float> out(3, 3);
	f.realize(out);

	// Main diagonal: col == row -> 1
	EXPECT_NEAR(out(0, 0), 1.0f, 1e-4f);
	EXPECT_NEAR(out(1, 1), 1.0f, 1e-4f);
	EXPECT_NEAR(out(2, 2), 1.0f, 1e-4f);
	// Off-diagonal -> 0
	EXPECT_NEAR(out(1, 0), 0.0f, 1e-4f);
	EXPECT_NEAR(out(0, 1), 0.0f, 1e-4f);
	EXPECT_NEAR(out(2, 0), 0.0f, 1e-4f);
	EXPECT_NEAR(out(0, 2), 0.0f, 1e-4f);
}

TEST(FactoryFuncs2, EyeAboveDiag) {
	// eye(type, 3, 3, k=1) -> 1s one above main diagonal
	// Entry = 1 iff col - row == 1
	Halide::Func f = eye(Halide::Float(32), 3, 3, 1);

	Halide::Runtime::Buffer<float> out(3, 3);
	f.realize(out);

	// col - row == 1: (1,0) and (2,1) are 1
	EXPECT_NEAR(out(1, 0), 1.0f, 1e-4f);   // col=1, row=0: 1-0=1 -> 1
	EXPECT_NEAR(out(2, 1), 1.0f, 1e-4f);   // col=2, row=1: 2-1=1 -> 1
	// Main diagonal -> 0
	EXPECT_NEAR(out(0, 0), 0.0f, 1e-4f);
	EXPECT_NEAR(out(1, 1), 0.0f, 1e-4f);
	EXPECT_NEAR(out(2, 2), 0.0f, 1e-4f);
	// Below diagonal -> 0
	EXPECT_NEAR(out(0, 1), 0.0f, 1e-4f);
	EXPECT_NEAR(out(0, 2), 0.0f, 1e-4f);
}

// -----------------------------------------------------------------------------
// Tri
// -----------------------------------------------------------------------------

TEST(FactoryFuncs2, Tri) {
	// tri(type, 3) -> lower-triangular 3x3 (k=0 default)
	// ret(col, row) = 1 iff col <= row + k  (col <= row when k=0)
	Halide::Func f = tri(Halide::Float(32), 3);

	Halide::Runtime::Buffer<float> out(3, 3);
	f.realize(out);

	// col <= row -> 1 (lower triangle including diagonal)
	EXPECT_NEAR(out(0, 0), 1.0f, 1e-4f);   // col=0, row=0: 0<=0 -> 1
	EXPECT_NEAR(out(0, 1), 1.0f, 1e-4f);   // col=0, row=1: 0<=1 -> 1
	EXPECT_NEAR(out(1, 1), 1.0f, 1e-4f);   // col=1, row=1: 1<=1 -> 1
	EXPECT_NEAR(out(0, 2), 1.0f, 1e-4f);   // col=0, row=2: 0<=2 -> 1
	EXPECT_NEAR(out(1, 2), 1.0f, 1e-4f);   // col=1, row=2: 1<=2 -> 1
	EXPECT_NEAR(out(2, 2), 1.0f, 1e-4f);   // col=2, row=2: 2<=2 -> 1
	// col > row -> 0 (upper triangle above diagonal)
	EXPECT_NEAR(out(1, 0), 0.0f, 1e-4f);   // col=1, row=0: 1>0 -> 0
	EXPECT_NEAR(out(2, 0), 0.0f, 1e-4f);   // col=2, row=0: 2>0 -> 0
	EXPECT_NEAR(out(2, 1), 0.0f, 1e-4f);   // col=2, row=1: 2>1 -> 0
}

// -----------------------------------------------------------------------------
// Vander
// -----------------------------------------------------------------------------

TEST(FactoryFuncs2, Vander) {
	// x = [1, 2, 3], n_cols=3, increasing=false (NumPy default: decreasing powers)
	// ret(col, row) = x[row]^(n_cols-1-col)
	// row 0 (x=1): [1^2, 1^1, 1^0] = [1, 1, 1]
	// row 1 (x=2): [2^2, 2^1, 2^0] = [4, 2, 1]
	// row 2 (x=3): [3^2, 3^1, 3^0] = [9, 3, 1]
	Halide::Func x_vals("x_vals");
	Halide::Var i;
	x_vals(i) = Halide::select(i == 0, 1.0f, i == 1, 2.0f, 3.0f);

	Halide::Func result = vander(x_vals, 3, false);

	// Realize as Buffer<float>(cols=3, rows=3)
	Halide::Runtime::Buffer<float> out(3, 3);
	result.realize(out);

	// Row 0 (x=1)
	EXPECT_NEAR(out(0, 0), 1.0f, 1e-4f);   // 1^2 = 1
	EXPECT_NEAR(out(1, 0), 1.0f, 1e-4f);   // 1^1 = 1
	EXPECT_NEAR(out(2, 0), 1.0f, 1e-4f);   // 1^0 = 1
	// Row 1 (x=2)
	EXPECT_NEAR(out(0, 1), 4.0f, 1e-4f);   // 2^2 = 4
	EXPECT_NEAR(out(1, 1), 2.0f, 1e-4f);   // 2^1 = 2
	EXPECT_NEAR(out(2, 1), 1.0f, 1e-4f);   // 2^0 = 1
	// Row 2 (x=3)
	EXPECT_NEAR(out(0, 2), 9.0f, 1e-4f);   // 3^2 = 9
	EXPECT_NEAR(out(1, 2), 3.0f, 1e-4f);   // 3^1 = 3
	EXPECT_NEAR(out(2, 2), 1.0f, 1e-4f);   // 3^0 = 1
}

// -----------------------------------------------------------------------------
// EmptyLike
// -----------------------------------------------------------------------------

TEST(FactoryFuncs2, EmptyLike) {
	// empty_like should return a Func with the same type as the reference Func
	shape_t s = { 4 };

	Halide::Func ref("ref");
	Halide::Var x;
	ref(x) = Halide::cast<float>(x);

	Halide::Func result = empty_like(ref, s);

	// Verify the returned Func has the correct element type
	EXPECT_EQ(result.types()[0], Halide::Float(32));
}
