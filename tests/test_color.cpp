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

// -----------------------------------------------------------------------------
// ColorExt: extended round-trip and edge-case tests
// -----------------------------------------------------------------------------

TEST(ColorExt, RgbHsvRgbRoundTripRed) {
	// Red (1,0,0) -> HSV -> RGB should recover (1,0,0)
	shape_t s = { 1 };
	Halide::Func input("input");
	Halide::Var x;
	input(x) = Halide::Tuple(1.0f, 0.0f, 0.0f);

	Halide::Func hsv = rgb_to_hsv(input, s, "hsv_rt");
	Halide::Func rgb = hsv_to_rgb(hsv, s, "rgb_rt");

	Halide::Func r("r"), g("g"), b("b");
	r(x) = rgb(x)[0];
	g(x) = rgb(x)[1];
	b(x) = rgb(x)[2];

	Halide::Runtime::Buffer<float> r_out(1), g_out(1), b_out(1);
	r.realize(r_out);
	g.realize(g_out);
	b.realize(b_out);

	EXPECT_NEAR(r_out(0), 1.0f, 1e-4f);
	EXPECT_NEAR(g_out(0), 0.0f, 1e-4f);
	EXPECT_NEAR(b_out(0), 0.0f, 1e-4f);
}

TEST(ColorExt, RgbYuvRgbRoundTrip) {
	// (0.5, 0.3, 0.7) -> YUV -> RGB should recover the original
	shape_t s = { 1 };
	Halide::Func input("input");
	Halide::Var x;
	input(x) = Halide::Tuple(0.5f, 0.3f, 0.7f);

	Halide::Func yuv = rgb_to_yuv(input, s, "yuv_rt");
	Halide::Func rgb = yuv_to_rgb(yuv, s, "rgb_rt2");

	Halide::Func r("r"), g("g"), b("b");
	r(x) = rgb(x)[0];
	g(x) = rgb(x)[1];
	b(x) = rgb(x)[2];

	Halide::Runtime::Buffer<float> r_out(1), g_out(1), b_out(1);
	r.realize(r_out);
	g.realize(g_out);
	b.realize(b_out);

	EXPECT_NEAR(r_out(0), 0.5f, 1e-3f);
	EXPECT_NEAR(g_out(0), 0.3f, 1e-3f);
	EXPECT_NEAR(b_out(0), 0.7f, 1e-3f);
}

TEST(ColorExt, GrayToRgbRoundTrip) {
	// gray 0.6 -> gray_to_rgb -> rgb_to_gray should give back ~0.6
	shape_t s = { 1 };
	Halide::Func input("input");
	Halide::Var x;
	input(x) = 0.6f;

	Halide::Func rgb = gray_to_rgb(input, s, "gray_to_rgb_rt");
	Halide::Func gray = rgb_to_gray(rgb, s, "gray_rt");

	Halide::Runtime::Buffer<float> out(1);
	gray.realize(out);

	// gray_to_rgb replicates: rgb_to_gray(v,v,v) = (0.299+0.587+0.114)*v = 1.0*v
	EXPECT_NEAR(out(0), 0.6f, 1e-4f);
}

TEST(ColorExt, HsvWhiteHasSaturationZero) {
	// Pure white (1,1,1): R=G=B so delta=0, S should be 0
	shape_t s = { 1 };
	Halide::Func input("input");
	Halide::Var x;
	input(x) = Halide::Tuple(1.0f, 1.0f, 1.0f);

	Halide::Func result = rgb_to_hsv(input, s, "hsv_white");

	Halide::Func sv("sv");
	sv(x) = result(x)[1];

	Halide::Runtime::Buffer<float> s_out(1);
	sv.realize(s_out);

	EXPECT_NEAR(s_out(0), 0.0f, 1e-4f);
}

TEST(ColorExt, HsvGreenHue) {
	// Pure green (0,1,0): max_c=G, delta=1, hue_raw=(B-R)/delta+2 = 2, H=2/6=1/3
	shape_t s = { 1 };
	Halide::Func input("input");
	Halide::Var x;
	input(x) = Halide::Tuple(0.0f, 1.0f, 0.0f);

	Halide::Func result = rgb_to_hsv(input, s, "hsv_green");

	Halide::Func h("h");
	h(x) = result(x)[0];

	Halide::Runtime::Buffer<float> h_out(1);
	h.realize(h_out);

	EXPECT_NEAR(h_out(0), 1.0f / 3.0f, 1e-4f);
}

TEST(ColorExt, YuvBlackIsZero) {
	// Black (0,0,0): Y=0*coeff=0, U=0*coeff=0, V=0*coeff=0
	shape_t s = { 1 };
	Halide::Func input("input");
	Halide::Var x;
	input(x) = Halide::Tuple(0.0f, 0.0f, 0.0f);

	Halide::Func result = rgb_to_yuv(input, s, "yuv_black");

	Halide::Func yc("yc"), u("u"), v("v");
	yc(x) = result(x)[0];
	u(x) = result(x)[1];
	v(x) = result(x)[2];

	Halide::Runtime::Buffer<float> y_out(1), u_out(1), v_out(1);
	yc.realize(y_out);
	u.realize(u_out);
	v.realize(v_out);

	EXPECT_NEAR(y_out(0), 0.0f, 1e-5f);
	EXPECT_NEAR(u_out(0), 0.0f, 1e-5f);
	EXPECT_NEAR(v_out(0), 0.0f, 1e-5f);
}
