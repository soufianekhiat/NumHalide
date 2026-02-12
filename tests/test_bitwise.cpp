/// @file test_bitwise.cpp
/// @brief Tests for bitwise operations

#include <gtest/gtest.h>
#include "numhalide_all.h"

using namespace numhalide;

TEST(Bitwise, And) {
	shape_t s = { 1 };
	Halide::Func a("a"), b("b");
	Halide::Var x;
	a(x) = 0b1100;
	b(x) = 0b1010;

	Halide::Func result = bitwise_and(a, b, s);

	Halide::Runtime::Buffer<int32_t> out(1);
	result.realize(out);

	EXPECT_EQ(out(0), 0b1000);
}

TEST(Bitwise, Or) {
	shape_t s = { 1 };
	Halide::Func a("a"), b("b");
	Halide::Var x;
	a(x) = 0b1100;
	b(x) = 0b1010;

	Halide::Func result = bitwise_or(a, b, s);

	Halide::Runtime::Buffer<int32_t> out(1);
	result.realize(out);

	EXPECT_EQ(out(0), 0b1110);
}

TEST(Bitwise, Xor) {
	shape_t s = { 1 };
	Halide::Func a("a"), b("b");
	Halide::Var x;
	a(x) = 0b1100;
	b(x) = 0b1010;

	Halide::Func result = bitwise_xor(a, b, s);

	Halide::Runtime::Buffer<int32_t> out(1);
	result.realize(out);

	EXPECT_EQ(out(0), 0b0110);
}

TEST(Bitwise, Not) {
	shape_t s = { 1 };
	Halide::Func a("a");
	Halide::Var x;
	a(x) = 0;

	Halide::Func result = bitwise_not(a, s);

	Halide::Runtime::Buffer<int32_t> out(1);
	result.realize(out);

	EXPECT_EQ(out(0), ~0);  // -1 for int32
}

TEST(Bitwise, LeftShift) {
	shape_t s = { 1 };
	Halide::Func a("a");
	Halide::Var x;
	a(x) = 1;

	Halide::Func result = left_shift(a, s, Halide::Expr(3));

	Halide::Runtime::Buffer<int32_t> out(1);
	result.realize(out);

	EXPECT_EQ(out(0), 8);
}

TEST(Bitwise, RightShift) {
	shape_t s = { 1 };
	Halide::Func a("a");
	Halide::Var x;
	a(x) = 8;

	Halide::Func result = right_shift(a, s, Halide::Expr(2));

	Halide::Runtime::Buffer<int32_t> out(1);
	result.realize(out);

	EXPECT_EQ(out(0), 2);
}

TEST(Bitwise, Popcount) {
	shape_t s = { 3 };
	Halide::Func a("a");
	Halide::Var x;
	a(x) = Halide::select(x == 0, 0b1011, x == 1, 0b0000, 0b1111);

	Halide::Func result = popcount(a, s);

	Halide::Runtime::Buffer<int32_t> out(s.extents[0]);
	result.realize(out);

	EXPECT_EQ(out(0), 3);
	EXPECT_EQ(out(1), 0);
	EXPECT_EQ(out(2), 4);
}

TEST(Bitwise, XorSelf) {
	shape_t s = { 4 };
	Halide::Func a("a");
	Halide::Var x;
	a(x) = x * 17 + 42;

	Halide::Func result = bitwise_xor(a, a, s);

	Halide::Runtime::Buffer<int32_t> out(s.extents[0]);
	result.realize(out);

	for (int i = 0; i < 4; ++i) {
		EXPECT_EQ(out(i), 0);
	}
}

TEST(Bitwise, ShiftRoundtrip) {
	shape_t s = { 1 };
	Halide::Func a("a");
	Halide::Var x;
	a(x) = 42;

	Halide::Func shifted = left_shift(a, s, Halide::Expr(4));
	Halide::Func result = right_shift(shifted, s, Halide::Expr(4));

	Halide::Runtime::Buffer<int32_t> out(1);
	result.realize(out);

	EXPECT_EQ(out(0), 42);
}
