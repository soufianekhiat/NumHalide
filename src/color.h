/// @file color.h
/// @brief Color space conversions for 2D images
///
/// Provides: rgb_to_gray, gray_to_rgb, rgb_to_hsv, hsv_to_rgb, rgb_to_yuv, yuv_to_rgb
///
/// Color images are represented as 2D Tuple Funcs with 3 elements (R, G, B)
/// or (H, S, V), (Y, U, V) depending on color space.

#pragma once

#include "common.h"
#include "shape.h"

NS_NUM_HALIDE_BEGIN

// -----------------------------------------------------------------------------
// RGB <-> Grayscale
// -----------------------------------------------------------------------------

/// @brief Convert RGB to grayscale using ITU-R BT.601 coefficients
/// @param f Input Func returning Tuple(R, G, B)
/// @param shape Shape of input (any rank)
/// @param name Function name
/// @return Single-channel Func: 0.299*R + 0.587*G + 0.114*B
inline
Halide::Func rgb_to_gray(Halide::Func f, const shape_t& shape, std::string const& name = "gray")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) vars.push_back(Halide::Var());

	ret(vars) = 0.299f * f(vars)[0] + 0.587f * f(vars)[1] + 0.114f * f(vars)[2];

	return ret;
}

/// @brief Convert grayscale to RGB by replicating single channel to 3-tuple
/// @param f Input single-channel Func
/// @param shape Shape of input (any rank)
/// @param name Function name
/// @return Func returning Tuple(val, val, val)
inline
Halide::Func gray_to_rgb(Halide::Func f, const shape_t& shape, std::string const& name = "rgb")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) vars.push_back(Halide::Var());

	Halide::Expr val = f(vars);
	ret(vars) = Halide::Tuple(val, val, val);

	return ret;
}

// -----------------------------------------------------------------------------
// RGB <-> HSV
// -----------------------------------------------------------------------------

/// @brief Convert RGB to HSV color space
/// @param f Input Func returning Tuple(R, G, B), values in [0, 1]
/// @param shape Shape of input (any rank)
/// @param name Function name
/// @return Func returning Tuple(H, S, V) where H is in [0, 1] (representing 0-360 degrees)
///
/// Algorithm:
///   max_c = max(R, G, B)
///   min_c = min(R, G, B)
///   delta = max_c - min_c
///   V = max_c
///   S = delta / max_c  (0 if max_c == 0)
///   H = piecewise based on which channel is max, normalized to [0, 1]
inline
Halide::Func rgb_to_hsv(Halide::Func f, const shape_t& shape, std::string const& name = "hsv")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) vars.push_back(Halide::Var());

	Halide::Expr R = f(vars)[0];
	Halide::Expr G = f(vars)[1];
	Halide::Expr B = f(vars)[2];

	Halide::Expr max_c = Halide::max(R, Halide::max(G, B));
	Halide::Expr min_c = Halide::min(R, Halide::min(G, B));
	Halide::Expr delta = max_c - min_c;

	// Value
	Halide::Expr V = max_c;

	// Saturation
	Halide::Expr S = Halide::select(max_c == 0.0f, 0.0f, delta / max_c);

	// Hue (in [0, 6) range, then normalized to [0, 1))
	// When delta == 0: H = 0
	// When max == R: H = ((G - B) / delta) % 6
	// When max == G: H = (B - R) / delta + 2
	// When max == B: H = (R - G) / delta + 4
	Halide::Expr hue_raw = Halide::select(
		delta == 0.0f, 0.0f,
		max_c == R, ((G - B) / delta) % 6.0f,
		max_c == G, (B - R) / delta + 2.0f,
		             (R - G) / delta + 4.0f
	);

	// Normalize to [0, 1] and ensure non-negative
	Halide::Expr H = Halide::select(hue_raw < 0.0f, hue_raw + 6.0f, hue_raw) / 6.0f;

	ret(vars) = Halide::Tuple(H, S, V);

	return ret;
}

/// @brief Convert HSV to RGB color space
/// @param f Input Func returning Tuple(H, S, V), H in [0, 1], S in [0, 1], V in [0, 1]
/// @param shape Shape of input (any rank)
/// @param name Function name
/// @return Func returning Tuple(R, G, B) with values in [0, 1]
///
/// Standard HSV to RGB conversion using sector-based piecewise formula.
inline
Halide::Func hsv_to_rgb(Halide::Func f, const shape_t& shape, std::string const& name = "rgb")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) vars.push_back(Halide::Var());

	Halide::Expr H = f(vars)[0];
	Halide::Expr S = f(vars)[1];
	Halide::Expr V = f(vars)[2];

	// Convert H from [0,1] to [0,6)
	Halide::Expr h6 = H * 6.0f;

	Halide::Expr C = V * S;             // Chroma
	Halide::Expr X = C * (1.0f - Halide::abs(h6 % 2.0f - 1.0f));  // Intermediate
	Halide::Expr m = V - C;             // Match value

	// Sector-based selection
	// Sector 0: [0,1) -> (C, X, 0)
	// Sector 1: [1,2) -> (X, C, 0)
	// Sector 2: [2,3) -> (0, C, X)
	// Sector 3: [3,4) -> (0, X, C)
	// Sector 4: [4,5) -> (X, 0, C)
	// Sector 5: [5,6) -> (C, 0, X)
	Halide::Expr R = Halide::select(
		h6 < 1.0f, C,
		h6 < 2.0f, X,
		h6 < 3.0f, 0.0f,
		h6 < 4.0f, 0.0f,
		h6 < 5.0f, X,
		            C
	) + m;

	Halide::Expr G = Halide::select(
		h6 < 1.0f, X,
		h6 < 2.0f, C,
		h6 < 3.0f, C,
		h6 < 4.0f, X,
		h6 < 5.0f, 0.0f,
		            0.0f
	) + m;

	Halide::Expr B = Halide::select(
		h6 < 1.0f, 0.0f,
		h6 < 2.0f, 0.0f,
		h6 < 3.0f, X,
		h6 < 4.0f, C,
		h6 < 5.0f, C,
		            X
	) + m;

	ret(vars) = Halide::Tuple(R, G, B);

	return ret;
}

// -----------------------------------------------------------------------------
// RGB <-> YUV
// -----------------------------------------------------------------------------

/// @brief Convert RGB to YUV color space
/// @param f Input Func returning Tuple(R, G, B)
/// @param shape Shape of input (any rank)
/// @param name Function name
/// @return Func returning Tuple(Y, U, V)
///
/// Uses standard conversion:
///   Y =  0.299  * R + 0.587  * G + 0.114  * B
///   U = -0.14713* R - 0.28886* G + 0.436  * B
///   V =  0.615  * R - 0.51499* G - 0.10001* B
inline
Halide::Func rgb_to_yuv(Halide::Func f, const shape_t& shape, std::string const& name = "yuv")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) vars.push_back(Halide::Var());

	Halide::Expr R = f(vars)[0];
	Halide::Expr G = f(vars)[1];
	Halide::Expr B = f(vars)[2];

	Halide::Expr Y =  0.29900f * R + 0.58700f * G + 0.11400f * B;
	Halide::Expr U = -0.14713f * R - 0.28886f * G + 0.43600f * B;
	Halide::Expr V =  0.61500f * R - 0.51499f * G - 0.10001f * B;

	ret(vars) = Halide::Tuple(Y, U, V);

	return ret;
}

/// @brief Convert YUV to RGB color space
/// @param f Input Func returning Tuple(Y, U, V)
/// @param shape Shape of input (any rank)
/// @param name Function name
/// @return Func returning Tuple(R, G, B)
///
/// Inverse of rgb_to_yuv:
///   R = Y + 1.13983 * V
///   G = Y - 0.39465 * U - 0.58060 * V
///   B = Y + 2.03211 * U
inline
Halide::Func yuv_to_rgb(Halide::Func f, const shape_t& shape, std::string const& name = "rgb")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) vars.push_back(Halide::Var());

	Halide::Expr Y = f(vars)[0];
	Halide::Expr U = f(vars)[1];
	Halide::Expr V = f(vars)[2];

	Halide::Expr R = Y + 1.13983f * V;
	Halide::Expr G = Y - 0.39465f * U - 0.58060f * V;
	Halide::Expr B = Y + 2.03211f * U;

	ret(vars) = Halide::Tuple(R, G, B);

	return ret;
}

NS_NUM_HALIDE_END
