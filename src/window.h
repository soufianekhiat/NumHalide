/// @file window.h
/// @brief Window functions for signal processing
///
/// Provides: hanning, hamming, blackman, bartlett, kaiser

#pragma once

#include "common.h"
#include "shape.h"
#include "numeric.h" // i0_expr for the runtime kaiser overload

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
///
/// @note The 6-term truncated series drifts from the true Bessel I0 as beta
/// grows (the truncation error is beta-dependent and does not cancel in the
/// ratio). The runtime overload below uses the Abramowitz & Stegun i0_expr
/// (~1e-7 for all x) and is the A&S-accurate form; prefer it when Bessel
/// conformance matters. This overload keeps its historical behavior.
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

/// @brief Kaiser window, RUNTIME size and beta, computed in `type`
/// @param n Window size (runtime Expr)
/// @param beta Shape parameter (runtime Expr), cast to type
/// @param type Computation and output type (e.g. Float(32), Float(64))
/// @param name Function name (default carries the _rt suffix so a pipeline
///        can hold both the compile-time and runtime forms without a
///        Func-name clash)
/// @return 1D Halide::Func: w(i) = I0(beta * sqrt(1 - ((2i/(n-1)) - 1)^2)) / I0(beta)
///
/// numpy.kaiser symmetric form, with I0 the Abramowitz & Stegun i0_expr
/// (~1e-7 for all x) — unlike the compile-time overload above, whose
/// 6-term truncated series deviates from the true Bessel ratio at large
/// beta. Endpoints: w(0) = w(n-1) = 1 / I0(beta), center (odd n) = 1.
inline Halide::Func kaiser(Halide::Expr n, Halide::Expr beta, Halide::Type type,
                           std::string const& name = "kaiser_rt") {
	Halide::Func ret(name);
	Halide::Var x;
	Halide::Expr one = Halide::Internal::make_one(type);
	Halide::Expr two = Halide::Internal::make_const(type, 2);
	Halide::Expr fi = Halide::cast(type, x);
	Halide::Expr fn = Halide::cast(type, n - 1);
	Halide::Expr fb = Halide::cast(type, beta);
	// t = 2*i/(n-1) - 1, ranges over [-1, 1]
	Halide::Expr t = two * fi / fn - one;
	// 1 - t^2 can go epsilon-negative at the endpoints: when n-1 is not a
	// power of two, t = fl(fl(2i/(n-1)) - 1) lands just outside [-1, 1] and
	// the sqrt of the tiny negative residue would be NaN — clamp to >= 0.
	Halide::Expr inner = Halide::max(one - t * t, Halide::Internal::make_zero(type));
	Halide::Expr arg = fb * Halide::sqrt(inner);
	ret(x) = i0_expr(arg) / i0_expr(fb);
	return ret;
}

NS_NUM_HALIDE_END
