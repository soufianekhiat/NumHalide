/// @file fft_ext.h
/// @brief Extended FFT operations
///
/// Provides: cross_power_spectrum (2-D and 1-D typed forms),
/// spectral_centroid, spectral_centroid_magnitude

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
Halide::Func cross_power_spectrum(Halide::Func a, Halide::Func b, int /*rows*/, int /*cols*/, std::string const& name = "cps")
{
	Halide::Func ret(name);
	Halide::Var x, y;

	// A * conj(B) = (ar + ai*j)(br - bi*j) = (ar*br + ai*bi) + j*(ai*br - ar*bi)
	Halide::Expr ar = a(x, y)[0], ai = a(x, y)[1];
	Halide::Expr br = b(x, y)[0], bi = b(x, y)[1];

	Halide::Expr prod_re = ar * br + ai * bi;
	Halide::Expr prod_im = ai * br - ar * bi;

	// The magnitude floor is made in the input's own float type (same f32
	// value as the old 1e-10f literal; integer inputs keep f32).
	Halide::Type t = a.types()[0];
	if (!t.is_float()) t = Halide::Float(32);

	Halide::Expr mag = Halide::sqrt(prod_re * prod_re + prod_im * prod_im);
	mag = Halide::max(mag, Halide::Internal::make_const(t, 1e-10));  // avoid division by zero

	ret(x, y) = Halide::Tuple(prod_re / mag, prod_im / mag);
	return ret;
}

/// @brief 1-D typed cross power spectrum for phase correlation
/// CPS = (A * conj(B)) / |A * conj(B)|
/// @param a Complex 1D Func (Tuple of real, imag)
/// @param b Complex 1D Func (Tuple of real, imag)
/// @param type Floating-point type to compute in (e.g. Halide::Float(32))
/// @param name Function name
/// @return Complex 1D Func with the normalized cross power spectrum
///
/// Pointwise — no sizes. The magnitude floor is
/// sqrt(max(re^2 + im^2, 1e-20)) with typed constants — value-identical to
/// the 2-D form's max(mag, 1e-10) floor for mag >= 0. The normalized Tuple
/// is computed in the Func definition (division in the Tuple, NOT inside a
/// select).
inline
Halide::Func cross_power_spectrum(Halide::Func a, Halide::Func b, Halide::Type type,
                                  std::string const& name = "cps_1d")
{
	nh_require(type.is_float(), "cross_power_spectrum compute type must be a float type");

	Halide::Func ret(name);
	Halide::Var k("k");

	// A * conj(B) = (ar + ai*j)(br - bi*j) = (ar*br + ai*bi) + j*(ai*br - ar*bi)
	Halide::Expr ar = Halide::cast(type, a(k)[0]), ai = Halide::cast(type, a(k)[1]);
	Halide::Expr br = Halide::cast(type, b(k)[0]), bi = Halide::cast(type, b(k)[1]);

	Halide::Expr prod_re = ar * br + ai * bi;
	Halide::Expr prod_im = ai * br - ar * bi;

	Halide::Expr mag = Halide::sqrt(Halide::max(prod_re * prod_re + prod_im * prod_im,
	                                            Halide::Internal::make_const(type, 1e-20)));

	ret(k) = Halide::Tuple(prod_re / mag, prod_im / mag);
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

	// Weight index and guard in the input's own float type (f64 stays
	// f64); integer inputs keep the historical f32 path.
	Halide::Type t = f.types()[0];
	if (!t.is_float()) t = Halide::Float(32);

	Halide::Expr power = f(r)[0] * f(r)[0] + f(r)[1] * f(r)[1];
	Halide::Expr weighted = Halide::cast(t, r) * power;

	ret(dummy) = Halide::sum(weighted)
		/ Halide::max(Halide::sum(power), Halide::Internal::make_const(t, 1e-10));
	return ret;
}

/// @brief Magnitude-weighted spectral centroid with a RUNTIME size
/// centroid = sum(r * mag[r]) / sum(mag[r])   (0 when the total is 0)
/// @param magnitudes Real-valued 1D Func of magnitudes (NOT complex, NOT
///        power — the complex/power-weighted spectral_centroid above serves
///        a different contract and is kept)
/// @param n Number of bins as a runtime expression
/// @param type Floating-point type to compute in
/// @param name Function name
/// @return 0-D Func: ret() = weighted / select(total > 0, total, 1)
///
/// Both reductions are compute_root'd; the select guard substitutes 1 for
/// a zero total so an all-zero spectrum yields 0.
inline
Halide::Func spectral_centroid_magnitude(Halide::Func magnitudes, Halide::Expr n,
                                         Halide::Type type,
                                         std::string const& name = "centroid_mag")
{
	nh_require(type.is_float(), "spectral_centroid_magnitude compute type must be a float type");

	Halide::RDom r(0, n);
	Halide::Expr mag = Halide::cast(type, magnitudes(r));

	// Total magnitude
	Halide::Func total(name + "_total");
	total() = Halide::sum(mag);
	total.compute_root();

	// Frequency-weighted magnitude
	Halide::Func weighted(name + "_weighted");
	weighted() = Halide::sum(Halide::cast(type, r) * mag);
	weighted.compute_root();

	Halide::Func ret(name);
	Halide::Expr zero = Halide::Internal::make_const(type, 0.0);
	Halide::Expr one  = Halide::Internal::make_const(type, 1.0);
	ret() = weighted() / Halide::select(total() > zero, total(), one);
	return ret;
}

NS_NUM_HALIDE_END
