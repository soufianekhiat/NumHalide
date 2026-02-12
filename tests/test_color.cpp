/// @file test_color.cpp
/// @brief Tests for color space conversions

#include <gtest/gtest.h>
#include "numhalide_all.h"
#include <cmath>

using namespace numhalide;

TEST(Color, GrayFromWhite) {
	shape_t s = { 1 };
	Halide::Func input("input");
	Halide::Var x;
	input(x) = Halide::Tuple(1.0f, 1.0f, 1.0f);

	Halide::Func result = rgb_to_gray(input, s);

	Halide::Runtime::Buffer<float> out(1);
	result.realize(out);

	EXPECT_NEAR(out(0), 1.0f, 1e-4f);
}

TEST(Color, GrayFromRed) {
	shape_t s = { 1 };
	Halide::Func input("input");
	Halide::Var x;
	input(x) = Halide::Tuple(1.0f, 0.0f, 0.0f);

	Halide::Func result = rgb_to_gray(input, s);

	Halide::Runtime::Buffer<float> out(1);
	result.realize(out);

	EXPECT_NEAR(out(0), 0.299f, 1e-4f);
}

TEST(Color, GrayFromGreen) {
	shape_t s = { 1 };
	Halide::Func input("input");
	Halide::Var x;
	input(x) = Halide::Tuple(0.0f, 1.0f, 0.0f);

	Halide::Func result = rgb_to_gray(input, s);

	Halide::Runtime::Buffer<float> out(1);
	result.realize(out);

	EXPECT_NEAR(out(0), 0.587f, 1e-4f);
}

TEST(Color, GrayFromBlue) {
	shape_t s = { 1 };
	Halide::Func input("input");
	Halide::Var x;
	input(x) = Halide::Tuple(0.0f, 0.0f, 1.0f);

	Halide::Func result = rgb_to_gray(input, s);

	Halide::Runtime::Buffer<float> out(1);
	result.realize(out);

	EXPECT_NEAR(out(0), 0.114f, 1e-4f);
}

TEST(Color, GrayToRgb) {
	shape_t s = { 1 };
	Halide::Func input("input");
	Halide::Var x;
	input(x) = 0.5f;

	Halide::Func result = gray_to_rgb(input, s);

	Halide::Func r("r"), g("g"), b("b");
	r(x) = result(x)[0];
	g(x) = result(x)[1];
	b(x) = result(x)[2];

	Halide::Runtime::Buffer<float> r_out(1), g_out(1), b_out(1);
	r.realize(r_out);
	g.realize(g_out);
	b.realize(b_out);

	EXPECT_NEAR(r_out(0), 0.5f, 1e-5f);
	EXPECT_NEAR(g_out(0), 0.5f, 1e-5f);
	EXPECT_NEAR(b_out(0), 0.5f, 1e-5f);
}

TEST(Color, RgbToHsvRed) {
	shape_t s = { 1 };
	Halide::Func input("input");
	Halide::Var x;
	input(x) = Halide::Tuple(1.0f, 0.0f, 0.0f);  // pure red

	Halide::Func result = rgb_to_hsv(input, s);

	Halide::Func h("h"), sv("sv"), v("v");
	h(x) = result(x)[0];
	sv(x) = result(x)[1];
	v(x) = result(x)[2];

	Halide::Runtime::Buffer<float> h_out(1), s_out(1), v_out(1);
	h.realize(h_out);
	sv.realize(s_out);
	v.realize(v_out);

	EXPECT_NEAR(h_out(0), 0.0f, 1e-4f);   // H=0 for red
	EXPECT_NEAR(s_out(0), 1.0f, 1e-4f);   // S=1
	EXPECT_NEAR(v_out(0), 1.0f, 1e-4f);   // V=1
}

TEST(Color, RgbToYuvWhite) {
	shape_t s = { 1 };
	Halide::Func input("input");
	Halide::Var x;
	input(x) = Halide::Tuple(1.0f, 1.0f, 1.0f);

	Halide::Func result = rgb_to_yuv(input, s);

	Halide::Func yc("yc"), u("u"), v("v");
	yc(x) = result(x)[0];
	u(x) = result(x)[1];
	v(x) = result(x)[2];

	Halide::Runtime::Buffer<float> y_out(1), u_out(1), v_out(1);
	yc.realize(y_out);
	u.realize(u_out);
	v.realize(v_out);

	EXPECT_NEAR(y_out(0), 1.0f, 1e-3f);   // Y=1 for white
	EXPECT_NEAR(u_out(0), 0.0f, 1e-3f);   // U~0
	EXPECT_NEAR(v_out(0), 0.0f, 1e-3f);   // V~0
}

TEST(Color, HsvSaturationZero) {
	shape_t s = { 1 };
	Halide::Func input("input");
	Halide::Var x;
	// Gray: S=0, V=0.5 -> should map to (0.5, 0.5, 0.5)
	input(x) = Halide::Tuple(0.0f, 0.0f, 0.5f);

	Halide::Func result = hsv_to_rgb(input, s);

	Halide::Func r("r"), g("g"), b("b");
	r(x) = result(x)[0];
	g(x) = result(x)[1];
	b(x) = result(x)[2];

	Halide::Runtime::Buffer<float> r_out(1), g_out(1), b_out(1);
	r.realize(r_out);
	g.realize(g_out);
	b.realize(b_out);

	EXPECT_NEAR(r_out(0), 0.5f, 1e-4f);
	EXPECT_NEAR(g_out(0), 0.5f, 1e-4f);
	EXPECT_NEAR(b_out(0), 0.5f, 1e-4f);
}
