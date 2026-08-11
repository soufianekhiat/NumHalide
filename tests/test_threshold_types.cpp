/// @file test_threshold_types.cpp
/// @brief Type-genericity and boundary tests for thresholding
///
/// Pins the fix that made threshold_binary/trunc/tozero emit in the INPUT's
/// own type (they previously hardcoded f32, which broke f64/half combos):
/// an f64 input must flow through as f64, with binary emitting exact
/// 1.0/0.0 doubles. Also pins the strict '>' comparison at the boundary.

#include <gtest/gtest.h>
#include "numhalide_all.h"

using namespace numhalide;

namespace {

// Fixture [0.2, 0.45, 0.8, 1.0] as a Float(64) Func
Halide::Func make_f64_fixture(const std::string& n)
{
	Halide::Buffer<double> buf(4);
	buf(0) = 0.2;
	buf(1) = 0.45;
	buf(2) = 0.8;
	buf(3) = 1.0;
	Halide::Func f(n);
	Halide::Var x;
	f(x) = buf(x);
	return f;
}

} // namespace

TEST(ThresholdTypes, BinaryF64) {
	shape_t s = { 4 };
	auto f = make_f64_fixture("f_bin64");

	auto r = threshold_binary(f, s, Halide::Expr(0.5), "bin64");

	Halide::Runtime::Buffer<double> out(4);
	r.realize(out);

	// Exact 0.0/1.0 in f64 — cast(type, 1)/cast(type, 0) are exact constants.
	EXPECT_EQ(out(0), 0.0);
	EXPECT_EQ(out(1), 0.0);
	EXPECT_EQ(out(2), 1.0);
	EXPECT_EQ(out(3), 1.0);
}

TEST(ThresholdTypes, TruncF64) {
	shape_t s = { 4 };
	auto f = make_f64_fixture("f_tr64");

	auto r = threshold_trunc(f, s, Halide::Expr(0.5), "tr64");

	Halide::Runtime::Buffer<double> out(4);
	r.realize(out);

	// min(x, 0.5) passes below-threshold doubles through UNCHANGED
	// (bit-exact) and clamps the rest to exactly 0.5.
	EXPECT_EQ(out(0), 0.2);
	EXPECT_EQ(out(1), 0.45);
	EXPECT_EQ(out(2), 0.5);
	EXPECT_EQ(out(3), 0.5);
}

TEST(ThresholdTypes, ToZeroF64) {
	shape_t s = { 4 };
	auto f = make_f64_fixture("f_tz64");

	auto r = threshold_tozero(f, s, Halide::Expr(0.5), "tz64");

	Halide::Runtime::Buffer<double> out(4);
	r.realize(out);

	// x > 0.5 keeps the ORIGINAL double, else exact 0.0.
	EXPECT_EQ(out(0), 0.0);
	EXPECT_EQ(out(1), 0.0);
	EXPECT_EQ(out(2), 0.8);
	EXPECT_EQ(out(3), 1.0);
}

TEST(ThresholdTypes, BoundaryStrictGreater) {
	// The comparison is a STRICT '>': a value exactly equal to the threshold
	// is NOT above it. 0.5 and 0.5f are exactly representable, so this is an
	// exact test, not a rounding accident.
	shape_t s = { 3 };
	Halide::Buffer<float> buf(3);
	buf(0) = 0.4999f;
	buf(1) = 0.5f;     // exactly the threshold
	buf(2) = 0.5001f;
	Halide::Func f("f_bnd");
	Halide::Var x;
	f(x) = buf(x);

	auto rb = threshold_binary(f, s, 0.5f, "bnd_bin");
	Halide::Runtime::Buffer<float> out_b(3);
	rb.realize(out_b);
	EXPECT_EQ(out_b(0), 0.0f);
	EXPECT_EQ(out_b(1), 0.0f);  // == thresh -> below
	EXPECT_EQ(out_b(2), 1.0f);

	auto rz = threshold_tozero(f, s, 0.5f, "bnd_tz");
	Halide::Runtime::Buffer<float> out_z(3);
	rz.realize(out_z);
	EXPECT_EQ(out_z(0), 0.0f);
	EXPECT_EQ(out_z(1), 0.0f);  // == thresh -> zeroed
	EXPECT_EQ(out_z(2), 0.5001f);
}
