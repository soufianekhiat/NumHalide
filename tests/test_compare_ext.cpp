/// @file test_compare_ext.cpp
/// @brief Tests for tolerance comparisons and closeness

#include <gtest/gtest.h>
#include "numhalide_all.h"
#include <cmath>

using namespace numhalide;

TEST(CompareExt, IsCloseExact) {
	shape_t s = { 3 };
	Halide::Func a("a"), b("b");
	Halide::Var x;
	a(x) = Halide::cast<float>(x);
	b(x) = Halide::cast<float>(x);

	Halide::Func result = isclose(a, b, s);

	Halide::Runtime::Buffer<int32_t> out(s.extents[0]);
	result.realize(out);

	EXPECT_EQ(out(0), 1);
	EXPECT_EQ(out(1), 1);
	EXPECT_EQ(out(2), 1);
}

TEST(CompareExt, IsCloseWithinTol) {
	shape_t s = { 3 };
	Halide::Func a("a"), b("b");
	Halide::Var x;
	a(x) = Halide::cast<float>(x);
	b(x) = Halide::cast<float>(x) + 1e-6f;  // tiny difference

	Halide::Func result = isclose(a, b, s);

	Halide::Runtime::Buffer<int32_t> out(s.extents[0]);
	result.realize(out);

	EXPECT_EQ(out(0), 1);
	EXPECT_EQ(out(1), 1);
	EXPECT_EQ(out(2), 1);
}

TEST(CompareExt, IsCloseOutsideTol) {
	shape_t s = { 3 };
	Halide::Func a("a"), b("b");
	Halide::Var x;
	a(x) = Halide::cast<float>(x);
	b(x) = Halide::cast<float>(x) + 1.0f;  // large difference

	Halide::Func result = isclose(a, b, s);

	Halide::Runtime::Buffer<int32_t> out(s.extents[0]);
	result.realize(out);

	EXPECT_EQ(out(0), 0);
	EXPECT_EQ(out(1), 0);
	EXPECT_EQ(out(2), 0);
}

TEST(CompareExt, AllCloseTrue) {
	shape_t s = { 4 };
	Halide::Func a("a"), b("b");
	Halide::Var x;
	a(x) = Halide::cast<float>(x) * 1.0f;
	b(x) = Halide::cast<float>(x) * 1.0f + 1e-9f;

	Halide::Func result = allclose(a, b, s);

	Halide::Runtime::Buffer<int32_t> out(1);
	result.realize(out);

	EXPECT_EQ(out(0), 1);
}

TEST(CompareExt, AllCloseFalse) {
	shape_t s = { 4 };
	Halide::Func a("a"), b("b");
	Halide::Var x;
	a(x) = Halide::cast<float>(x);
	b(x) = Halide::cast<float>(x) + Halide::select(x == 2, 100.0f, 0.0f);

	Halide::Func result = allclose(a, b, s);

	Halide::Runtime::Buffer<int32_t> out(1);
	result.realize(out);

	EXPECT_EQ(out(0), 0);
}

TEST(CompareExt, IsPosInf) {
	shape_t s = { 3 };
	Halide::Func input("input");
	Halide::Var x;
	Halide::Expr zero = Halide::cast<float>(0);
	input(x) = Halide::select(x == 0, 1.0f / zero, x == 1, -1.0f / zero, 5.0f);

	Halide::Func result = isposinf(input, s);

	Halide::Runtime::Buffer<int32_t> out(s.extents[0]);
	result.realize(out);

	EXPECT_EQ(out(0), 1);  // +inf
	EXPECT_EQ(out(1), 0);  // -inf
	EXPECT_EQ(out(2), 0);  // normal
}

TEST(CompareExt, IsNegInf) {
	shape_t s = { 3 };
	Halide::Func input("input");
	Halide::Var x;
	Halide::Expr zero = Halide::cast<float>(0);
	input(x) = Halide::select(x == 0, 1.0f / zero, x == 1, -1.0f / zero, 5.0f);

	Halide::Func result = isneginf(input, s);

	Halide::Runtime::Buffer<int32_t> out(s.extents[0]);
	result.realize(out);

	EXPECT_EQ(out(0), 0);  // +inf
	EXPECT_EQ(out(1), 1);  // -inf
	EXPECT_EQ(out(2), 0);  // normal
}
