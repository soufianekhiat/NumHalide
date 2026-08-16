/// @file window.h
/// @brief Window functions for signal processing
///
/// Provides: hanning, hamming, blackman, bartlett, kaiser
///
/// Each window has a TYPED form (trailing Halide::Type parameter) that
/// computes and emits in that type (f32, f64, ...), plus the legacy form
/// without a Type, which delegates to the typed form at Float(32)
/// (value-identical to the historical f32 bodies — constants are folded
/// to the same f32 values).

#pragma once

#include "common.h"
#include "shape.h"
#include "numeric.h" // i0_expr for the runtime kaiser overload

NS_NUM_HALIDE_BEGIN

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// -----------------------------------------------------------------------------
// Window Functions
// -----------------------------------------------------------------------------

/// @brief Hanning window, computed in `type`: 0.5 * (1 - cos(2*pi*n/(N-1)))
/// @param size Window size
/// @param type Computation and output type (e.g. Float(32), Float(64))
/// @param name Function name
inline Halide::Func hanning(Halide::Expr size, Halide::Type type,
                            std::string const& name = "hanning") {
	Halide::Func ret(name);
	Halide::Var x;
	Halide::Expr one    = Halide::Internal::make_one(type);
	Halide::Expr half   = Halide::Internal::make_const(type, 0.5);
	Halide::Expr two_pi = Halide::Internal::make_const(type, 2.0 * M_PI);
	ret(x) = half * (one - Halide::cos(two_pi * Halide::cast(type, x) / Halide::cast(type, size - 1)));
	return ret;
}

/// @brief Hanning window: 0.5 * (1 - cos(2*pi*n/(N-1)))
/// @param size Window size
/// @param name Function name
/// @return 1D Halide::Func of float values (delegates to the typed form at
///         Float(32); value-identical to the historical f32 body)
inline Halide::Func hanning(Halide::Expr size, std::string const& name = "hanning") {
	return hanning(size, Halide::Float(32), name);
}

/// @brief Hamming window, computed in `type`: 0.54 - 0.46 * cos(2*pi*n/(N-1))
/// @param size Window size
/// @param type Computation and output type (e.g. Float(32), Float(64))
/// @param name Function name
inline Halide::Func hamming(Halide::Expr size, Halide::Type type,
                            std::string const& name = "hamming") {
	Halide::Func ret(name);
	Halide::Var x;
	Halide::Expr a0     = Halide::Internal::make_const(type, 0.54);
	Halide::Expr a1     = Halide::Internal::make_const(type, 0.46);
	Halide::Expr two_pi = Halide::Internal::make_const(type, 2.0 * M_PI);
	ret(x) = a0 - a1 * Halide::cos(two_pi * Halide::cast(type, x) / Halide::cast(type, size - 1));
	return ret;
}

/// @brief Hamming window: 0.54 - 0.46 * cos(2*pi*n/(N-1))
/// @param size Window size
/// @param name Function name
/// @return 1D Halide::Func of float values (delegates to the typed form at
///         Float(32); value-identical to the historical f32 body)
inline Halide::Func hamming(Halide::Expr size, std::string const& name = "hamming") {
	return hamming(size, Halide::Float(32), name);
}

/// @brief Blackman window, computed in `type`:
///        0.42 - 0.5*cos(2*pi*n/(N-1)) + 0.08*cos(4*pi*n/(N-1))
/// @param size Window size
/// @param type Computation and output type (e.g. Float(32), Float(64))
/// @param name Function name
inline Halide::Func blackman(Halide::Expr size, Halide::Type type,
                             std::string const& name = "blackman") {
	Halide::Func ret(name);
	Halide::Var x;
	Halide::Expr a0     = Halide::Internal::make_const(type, 0.42);
	Halide::Expr a1     = Halide::Internal::make_const(type, 0.5);
	Halide::Expr a2     = Halide::Internal::make_const(type, 0.08);
	Halide::Expr two_pi = Halide::Internal::make_const(type, 2.0 * M_PI);
	// four_pi = two_pi + two_pi: exact doubling, so at Float(32) it equals
	// the historical host-folded 4.0f * (float)M_PI constant bit-for-bit.
	Halide::Expr four_pi = two_pi + two_pi;
	Halide::Expr n = Halide::cast(type, x);
	Halide::Expr N = Halide::cast(type, size - 1);
	ret(x) = a0 - a1 * Halide::cos(two_pi * n / N) + a2 * Halide::cos(four_pi * n / N);
	return ret;
}

/// @brief Blackman window: 0.42 - 0.5*cos(2*pi*n/(N-1)) + 0.08*cos(4*pi*n/(N-1))
/// @param size Window size
/// @param name Function name
/// @return 1D Halide::Func of float values (delegates to the typed form at
///         Float(32); value-identical to the historical f32 body)
inline Halide::Func blackman(Halide::Expr size, std::string const& name = "blackman") {
	return blackman(size, Halide::Float(32), name);
}

/// @brief Bartlett (triangular) window, computed in `type`: 1 - |2*n/(N-1) - 1|
/// @param size Window size
/// @param type Computation and output type (e.g. Float(32), Float(64))
/// @param name Function name
inline Halide::Func bartlett(Halide::Expr size, Halide::Type type,
                             std::string const& name = "bartlett") {
	Halide::Func ret(name);
	Halide::Var x;
	Halide::Expr one = Halide::Internal::make_one(type);
	Halide::Expr two = Halide::Internal::make_const(type, 2.0);
	ret(x) = one - Halide::abs(two * Halide::cast(type, x) / Halide::cast(type, size - 1) - one);
	return ret;
}

/// @brief Bartlett (triangular) window: 1 - |2*n/(N-1) - 1|
/// @param size Window size
/// @param name Function name
/// @return 1D Halide::Func of float values (delegates to the typed form at
///         Float(32); value-identical to the historical f32 body)
inline Halide::Func bartlett(Halide::Expr size, std::string const& name = "bartlett") {
	return bartlett(size, Halide::Float(32), name);
}

/// @brief Approximate modified Bessel function I0 using series expansion
/// @param x Input expression
/// @return Approximation of I0(x): 1 + (x/2)^2 + (x/2)^4/4 + (x/2)^6/36 + (x/2)^8/576 + (x/2)^10/14400
///
/// Uses the first 6 terms of the series: I0(x) = sum_{k=0}^{inf} ((x/2)^k / k!)^2
///
/// @note LEGACY helper, kept for API stability. The truncation error grows
/// with |x|; kaiser() now uses the Abramowitz & Stegun i0_expr from
/// numeric.h (~1e-7 for all x) instead of this series.
inline Halide::Expr bessel_i0_approx(Halide::Expr x) {
	Halide::Expr half_x = x / 2.0f;
	Halide::Expr h2 = half_x * half_x;
	Halide::Expr h4 = h2 * h2;
	Halide::Expr h6 = h4 * h2;
	Halide::Expr h8 = h4 * h4;
	Halide::Expr h10 = h8 * h2;
	return 1.0f + h2 + h4 / 4.0f + h6 / 36.0f + h8 / 576.0f + h10 / 14400.0f;
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
/// (~1e-7 for all x). Endpoints: w(0) = w(n-1) = 1 / I0(beta), center
/// (odd n) = 1.
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

/// @brief Kaiser window (compile-time size, f32 values)
/// @param size Window size
/// @param beta Shape parameter (default 12.0)
/// @param name Function name
/// @return 1D Halide::Func of float values
///
/// Kaiser window: w(n) = I0(beta * sqrt(1 - (2n/(N-1) - 1)^2)) / I0(beta)
///
/// Delegates to the typed runtime overload above at Float(32), so I0 is
/// the Abramowitz & Stegun i0_expr (~1e-7 for all x). NOTE: this is an
/// ACCURACY IMPROVEMENT over the historical body, which used a 6-term
/// truncated series (bessel_i0_approx) whose beta-dependent truncation
/// error does not cancel in the I0 ratio and drifts from the true Bessel
/// ratio as beta grows. Values therefore differ from the pre-delegation
/// form (they now match numpy.kaiser to ~1e-6 at f32).
inline Halide::Func kaiser(int size, float beta = 12.0f, std::string const& name = "kaiser") {
	return kaiser(Halide::Expr(size), Halide::Expr(beta), Halide::Float(32), name);
}

NS_NUM_HALIDE_END
