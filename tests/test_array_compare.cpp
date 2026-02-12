/// @file test_array_compare.cpp
/// @brief Tests for array comparison utilities

#include <gtest/gtest.h>
#include "numhalide_all.h"

using namespace numhalide;

TEST(ArrayCompare, EqualTrue) {
	shape_t s = { 4 };
	Halide::Func a("a"), b("b");
	Halide::Var x;
	a(x) = Halide::cast<float>(x);
	b(x) = Halide::cast<float>(x);

	Halide::Func result = array_equal(a, b, s);

	Halide::Runtime::Buffer<int32_t> out(1);
	result.realize(out);

	EXPECT_EQ(out(0), 1);
}

TEST(ArrayCompare, EqualFalse) {
	shape_t s = { 4 };
	Halide::Func a("a"), b("b");
	Halide::Var x;
	a(x) = Halide::cast<float>(x);
	b(x) = Halide::cast<float>(x) + Halide::select(x == 2, 1.0f, 0.0f);

	Halide::Func result = array_equal(a, b, s);

	Halide::Runtime::Buffer<int32_t> out(1);
	result.realize(out);

	EXPECT_EQ(out(0), 0);
}

TEST(ArrayCompare, EquivSame) {
	shape_t s = { 3 };
	Halide::Func a("a"), b("b");
	Halide::Var x;
	a(x) = 5.0f;
	b(x) = 5.0f;

	Halide::Func result = array_equiv(a, b, s);

	Halide::Runtime::Buffer<int32_t> out(1);
	result.realize(out);

	EXPECT_EQ(out(0), 1);
}

TEST(ArrayCompare, EqualFloat) {
	shape_t s = { 3 };
	Halide::Func a("a"), b("b");
	Halide::Var x;
	a(x) = Halide::cast<float>(x) * 0.1f;
	b(x) = Halide::cast<float>(x) * 0.1f;

	Halide::Func result = array_equal(a, b, s);

	Halide::Runtime::Buffer<int32_t> out(1);
	result.realize(out);

	EXPECT_EQ(out(0), 1);
}
