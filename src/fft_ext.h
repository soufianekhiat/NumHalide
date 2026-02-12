/// @file fft_ext.h
/// @brief Extended FFT operations
///
/// Provides: cross_power_spectrum, spectral_centroid

#pragma once

#include "common.h"
#include "shape.h"
#include "fft.h"

NS_NUM_HALIDE_BEGIN

// -----------------------------------------------------------------------------
// Cross Power Spectrum
// -----------------------------------------------------------------------------

/// @brief Compute cross power spectrum for phase correlation
/// CPS = (A * conj(B)) / |A * conj(B)|
/// @param a Complex 2D Func (Tuple of real, imag)
/// @param b Complex 2D Func (Tuple of real, imag)
/// @param rows Number of rows
/// @param cols Number of columns
/// @param name Function name
/// @return Complex 2D Func with normalized cross power spectrum
inline
Halide::Func cross_power_spectrum(Halide::Func a, Halide::Func b, int rows, int cols, std::string const& name = "cps")
{
	Halide::Func ret(name);
	Halide::Var x, y;

	// A * conj(B) = (ar + ai*j)(br - bi*j) = (ar*br + ai*bi) + j*(ai*br - ar*bi)
	Halide::Expr ar = a(x, y)[0], ai = a(x, y)[1];
	Halide::Expr br = b(x, y)[0], bi = b(x, y)[1];

	Halide::Expr prod_re = ar * br + ai * bi;
	Halide::Expr prod_im = ai * br - ar * bi;

	Halide::Expr mag = Halide::sqrt(prod_re * prod_re + prod_im * prod_im);
	mag = Halide::max(mag, 1e-10f);  // avoid division by zero

	ret(x, y) = Halide::Tuple(prod_re / mag, prod_im / mag);
	return ret;
}

// -----------------------------------------------------------------------------
// Spectral Centroid
// -----------------------------------------------------------------------------

/// @brief Compute spectral centroid (weighted average frequency) for 1D power spectrum
/// centroid = sum(k * |X[k]|^2) / sum(|X[k]|^2)
/// @param f Complex 1D Func (Tuple of real, imag)
/// @param N Size of the spectrum
/// @param name Function name
/// @return Scalar Func with the spectral centroid value
inline
Halide::Func spectral_centroid(Halide::Func f, int N, std::string const& name = "centroid")
{
	Halide::Func ret(name);
	Halide::Var dummy;
	Halide::RDom r(0, N);

	Halide::Expr power = f(r)[0] * f(r)[0] + f(r)[1] * f(r)[1];
	Halide::Expr weighted = Halide::cast<float>(r) * power;

	ret(dummy) = Halide::sum(weighted) / Halide::max(Halide::sum(power), 1e-10f);
	return ret;
}

NS_NUM_HALIDE_END
