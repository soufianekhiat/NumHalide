/// @file window.h
/// @brief Window functions for signal processing
///
/// Provides: hanning, hamming, blackman, bartlett, kaiser

#pragma once

#include "common.h"
#include "shape.h"

NS_NUM_HALIDE_BEGIN

// -----------------------------------------------------------------------------
// Window Functions
// -----------------------------------------------------------------------------

/// @brief Hanning window: 0.5 * (1 - cos(2*pi*n/(N-1)))
/// @param size Window size
/// @param name Function name
/// @return 1D Halide::Func of float values
inline Halide::Func hanning(Halide::Expr size, std::string const& name = "hanning") {
	Halide::Func ret(name);
	Halide::Var x;
	float pi = 3.14159265358979323846f;
	ret(x) = 0.5f * (1.0f - Halide::cos(2.0f * pi * Halide::cast<float>(x) / Halide::cast<float>(size - 1)));
	return ret;
}

/// @brief Hamming window: 0.54 - 0.46 * cos(2*pi*n/(N-1))
/// @param size Window size
/// @param name Function name
/// @return 1D Halide::Func of float values
inline Halide::Func hamming(Halide::Expr size, std::string const& name = "hamming") {
	Halide::Func ret(name);
	Halide::Var x;
	float pi = 3.14159265358979323846f;
	ret(x) = 0.54f - 0.46f * Halide::cos(2.0f * pi * Halide::cast<float>(x) / Halide::cast<float>(size - 1));
	return ret;
}

/// @brief Blackman window: 0.42 - 0.5*cos(2*pi*n/(N-1)) + 0.08*cos(4*pi*n/(N-1))
/// @param size Window size
/// @param name Function name
/// @return 1D Halide::Func of float values
inline Halide::Func blackman(Halide::Expr size, std::string const& name = "blackman") {
	Halide::Func ret(name);
	Halide::Var x;
	float pi = 3.14159265358979323846f;
	Halide::Expr n = Halide::cast<float>(x);
	Halide::Expr N = Halide::cast<float>(size - 1);
	ret(x) = 0.42f - 0.5f * Halide::cos(2.0f * pi * n / N) + 0.08f * Halide::cos(4.0f * pi * n / N);
	return ret;
}

/// @brief Bartlett (triangular) window: 1 - |2*n/(N-1) - 1|
/// @param size Window size
/// @param name Function name
/// @return 1D Halide::Func of float values
inline Halide::Func bartlett(Halide::Expr size, std::string const& name = "bartlett") {
	Halide::Func ret(name);
	Halide::Var x;
	ret(x) = 1.0f - Halide::abs(2.0f * Halide::cast<float>(x) / Halide::cast<float>(size - 1) - 1.0f);
	return ret;
}

/// @brief Approximate modified Bessel function I0 using series expansion
/// @param x Input expression
/// @return Approximation of I0(x): 1 + (x/2)^2 + (x/2)^4/4 + (x/2)^6/36 + (x/2)^8/576 + (x/2)^10/14400
///
/// Uses the first 6 terms of the series: I0(x) = sum_{k=0}^{inf} ((x/2)^k / k!)^2
inline Halide::Expr bessel_i0_approx(Halide::Expr x) {
	Halide::Expr half_x = x / 2.0f;
	Halide::Expr h2 = half_x * half_x;
	Halide::Expr h4 = h2 * h2;
	Halide::Expr h6 = h4 * h2;
	Halide::Expr h8 = h4 * h4;
	Halide::Expr h10 = h8 * h2;
	return 1.0f + h2 + h4 / 4.0f + h6 / 36.0f + h8 / 576.0f + h10 / 14400.0f;
}

/// @brief Kaiser window using polynomial approximation of I0
/// @param size Window size
/// @param beta Shape parameter (default 12.0)
/// @param name Function name
/// @return 1D Halide::Func of float values
///
/// Kaiser window: w(n) = I0(beta * sqrt(1 - (2n/(N-1) - 1)^2)) / I0(beta)
/// where I0 is approximated by the first 6 terms of its series expansion.
inline Halide::Func kaiser(int size, float beta = 12.0f, std::string const& name = "kaiser") {
	Halide::Func ret(name);
	Halide::Var x;
	Halide::Expr n = Halide::cast<float>(x);
	Halide::Expr N = Halide::cast<float>(size - 1);
	// t = 2*n/(N-1) - 1, ranges from -1 to 1
	Halide::Expr t = 2.0f * n / N - 1.0f;
	// arg = beta * sqrt(1 - t^2)
	// Clamp (1 - t^2) to avoid sqrt of negative due to floating point
	Halide::Expr inner = Halide::max(1.0f - t * t, 0.0f);
	Halide::Expr arg = beta * Halide::sqrt(inner);
	// I0(arg) / I0(beta)
	Halide::Expr i0_beta = bessel_i0_approx(Halide::Expr(beta));
	ret(x) = bessel_i0_approx(arg) / i0_beta;
	return ret;
}

NS_NUM_HALIDE_END
