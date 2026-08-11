/// @file rfft.h
/// @brief Real FFT extensions and frequency utilities
///
/// Provides: rfft, irfft (int-N and runtime-Expr forms), rfft2d, irfft2d,
/// fftfreq, rfftfreq

#pragma once

#include "common.h"
#include "shape.h"
#include "fft.h"

NS_NUM_HALIDE_BEGIN

// -----------------------------------------------------------------------------
// 1D Real FFT
// -----------------------------------------------------------------------------

/// @brief Real-input 1D FFT returning complex output of size N/2+1
/// @param input Real-valued 1D Func (not Tuple)
/// @param N Size of transform (any size — the underlying c2c form is a
///        direct DFT)
/// @param name Function name
/// @return Complex Func (Tuple of real, imag) with N/2+1 meaningful bins
///
/// Takes a real 1D signal, converts to complex, runs full FFT, and returns
/// the result. For real input, only the first N/2+1 bins are unique due to
/// conjugate symmetry: X[k] = conj(X[N-k]).
///
/// Type-preserving (real_to_complex rule): float inputs keep their type
/// (f64 stays f64); integer inputs are promoted to f32.
inline
Halide::Func rfft(Halide::Func input, int N, std::string const& name = "rfft")
{
	nh_require(N > 0, "rfft requires N > 0, got %d", N);

	// Convert real input to complex (imaginary = 0), preserving float types
	Halide::Func complex_in = real_to_complex(input, name + "_cin");

	// Full FFT
	auto full = fft(complex_in, N, name + "_full");

	// Return the full result; user reads bins [0, N/2+1)
	Halide::Func ret(name);
	Halide::Var x;
	ret(x) = full(x);
	return ret;
}

/// @brief Real-input 1D DFT with a RUNTIME size
/// @param real_input Real-valued 1D Func (not Tuple)
/// @param n Size of transform as a runtime expression (any size)
/// @param type Floating-point type to compute in (e.g. Halide::Float(32))
/// @param name Function name
/// @return Complex Func (Tuple of real, imag); caller reads bins [0, n/2+1)
///
/// Direct sum over the real samples:
///   angle = -2*pi*k*r/n
///   re    = sum(sample * cos(angle))
///   im    = sum(sample * sin(angle))
/// The result is compute_root'd (reduction — chaining rule).
inline
Halide::Func rfft(Halide::Func real_input, Halide::Expr n, Halide::Type type,
                  std::string const& name = "rfft_rt")
{
	nh_require(type.is_float(), "rfft compute type must be a float type");

	Halide::Func result(name);
	Halide::Var k("k");
	Halide::RDom r(0, n);

	Halide::Expr angle = Halide::Internal::make_const(type, -2.0 * M_PI) *
	                     Halide::cast(type, k) * Halide::cast(type, r) /
	                     Halide::cast(type, n);
	Halide::Expr sample = Halide::cast(type, real_input(r));

	result(k) = Halide::Tuple(Halide::sum(sample * Halide::cos(angle)),
	                          Halide::sum(sample * Halide::sin(angle)));
	result.compute_root();

	return result;
}

/// @brief Inverse real FFT: takes N/2+1 complex bins, returns N real values
/// @param input Complex Func with N/2+1 bins (Tuple of real, imag)
/// @param N Original signal size (any size; the weighted form is exact for
///        the documented contract — a true Hermitian half-spectrum with
///        real DC and Nyquist, i.e. even N)
/// @param name Function name
/// @return Real-valued 1D Func of size N
///
/// Bounded Hermitian weighted sum — reads ONLY bins [0, K), K = N/2+1:
///   x[i] = (1/N) * sum_{r=0}^{K-1} weight(r) *
///          (re(r)*cos(2*pi*r*i/N) - im(r)*sin(2*pi*r*i/N))
///   weight = 1 for r==0 and r==K-1 (DC and Nyquist), else 2.
/// Value-identical to the old conjugate-symmetry reconstruction for the
/// documented contract. The old select body read input(N - x), so bounds
/// inference over both branches requested input over [0, N] — N+1 elements
/// from a buffer documented to hold K bins.
/// Float inputs keep their type; integer inputs promote to f32.
inline
Halide::Func irfft(Halide::Func input, int N, std::string const& name = "irfft")
{
	nh_require(N > 0, "irfft requires N > 0, got %d", N);

	int const K = N / 2 + 1;

	Halide::Type t = input.types()[0];
	if (!t.is_float()) t = Halide::Float(32);

	Halide::Func ret(name);
	Halide::Var x;
	Halide::RDom r(0, K);

	Halide::Expr angle = Halide::Internal::make_const(t, 2.0 * M_PI) *
	                     Halide::cast(t, r) * Halide::cast(t, x) /
	                     Halide::Internal::make_const(t, (double)N);
	Halide::Expr re = Halide::cast(t, input(r)[0]);
	Halide::Expr im = Halide::cast(t, input(r)[1]);
	// Weight: 1 for DC and Nyquist (r==0 and r==K-1), 2 for all others
	Halide::Expr weight = Halide::select(
		r == 0 || r == K - 1,
		Halide::Internal::make_const(t, 1.0),
		Halide::Internal::make_const(t, 2.0));

	ret(x) = Halide::sum(weight * (re * Halide::cos(angle) - im * Halide::sin(angle))) /
	         Halide::Internal::make_const(t, (double)N);
	ret.compute_root();  // reduction — chaining rule

	return ret;
}

/// @brief Inverse real FFT with a RUNTIME bin count
/// @param complex_input Complex Func with k_bins bins (Tuple of real, imag)
/// @param k_bins Number of half-spectrum bins as a runtime expression
///        (K = n/2+1; the reconstructed signal size is n = 2*(k_bins - 1))
/// @param type Floating-point type to compute in
/// @param name Function name
/// @return Real-valued 1D Func of size n = 2*(k_bins - 1)
///
/// Bounded Hermitian weighted sum — reads ONLY bins [0, k_bins):
///   x[i] = (1/n) * sum_{r=0}^{K-1} weight(r) *
///          (re(r)*cos(2*pi*r*i/n) - im(r)*sin(2*pi*r*i/n))
///   weight = 1 for r==0 and r==K-1 (DC and Nyquist), else 2.
/// The result is compute_root'd (reduction — chaining rule).
inline
Halide::Func irfft(Halide::Func complex_input, Halide::Expr k_bins, Halide::Type type,
                   std::string const& name = "irfft_rt")
{
	nh_require(type.is_float(), "irfft compute type must be a float type");

	Halide::Expr n = 2 * (k_bins - 1);

	Halide::Func result(name);
	Halide::Var i("i");
	Halide::RDom r(0, k_bins);

	Halide::Expr angle = Halide::Internal::make_const(type, 2.0 * M_PI) *
	                     Halide::cast(type, r) * Halide::cast(type, i) /
	                     Halide::cast(type, n);
	Halide::Expr re = Halide::cast(type, complex_input(r)[0]);
	Halide::Expr im = Halide::cast(type, complex_input(r)[1]);
	// Weight: 1 for DC and Nyquist (r==0 and r==k_bins-1), 2 for all others
	Halide::Expr weight = Halide::select(
		r == 0 || r == k_bins - 1,
		Halide::Internal::make_const(type, 1.0),
		Halide::Internal::make_const(type, 2.0));

	result(i) = Halide::sum(weight * (re * Halide::cos(angle) - im * Halide::sin(angle))) /
	            Halide::cast(type, n);
	result.compute_root();

	return result;
}

// -----------------------------------------------------------------------------
// 2D Real FFT
// -----------------------------------------------------------------------------

/// @brief Real-input 2D FFT
/// @param input Real-valued 2D Func (not Tuple)
/// @param rows Number of rows (any size — the underlying c2c form is a
///        direct DFT)
/// @param cols Number of columns (any size)
/// @param name Function name
/// @return Complex 2D Func (Tuple of real, imag)
///
/// Converts real input to complex and runs full 2D FFT.
/// Type-preserving (real_to_complex rule): float inputs keep their type;
/// integer inputs are promoted to f32.
inline
Halide::Func rfft2d(Halide::Func input, int rows, int cols, std::string const& name = "rfft2d")
{
	nh_require(rows > 0 && cols > 0, "rfft2d requires positive sizes, got %d x %d", rows, cols);

	// Convert real to complex, preserving float types
	Halide::Func complex_in = real_to_complex_2d(input, name + "_cin");

	// Full 2D FFT
	auto full = fft2d(complex_in, rows, cols, name + "_full");

	Halide::Func ret(name);
	Halide::Var x, y;
	ret(x, y) = full(x, y);
	return ret;
}

/// @brief Inverse real 2D FFT
/// @param input Complex 2D Func (FULL spectrum — no conjugate-symmetry
///        mirror reads, so no half-spectrum bounds concern here)
/// @param rows Number of rows (any size — the underlying c2c form is a
///        direct DFT)
/// @param cols Number of columns (any size)
/// @param name Function name
/// @return Real-valued 2D Func
///
/// Applies normalized inverse 2D FFT and returns real part.
inline
Halide::Func irfft2d(Halide::Func input, int rows, int cols, std::string const& name = "irfft2d")
{
	nh_require(rows > 0 && cols > 0, "irfft2d requires positive sizes, got %d x %d", rows, cols);

	auto result = ifft2d_normalized(input, rows, cols, name + "_ifft");

	// Return real part only
	Halide::Func ret(name);
	Halide::Var x, y;
	ret(x, y) = result(x, y)[0];
	return ret;
}

// -----------------------------------------------------------------------------
// Frequency Utilities
// -----------------------------------------------------------------------------

/// @brief Generate DFT sample frequencies
/// @param N Window length
/// @param d Sample spacing (default 1.0)
/// @param name Function name
/// @return 1D Func: even N -> [0, 1, ..., N/2-1, -N/2, ..., -1] / (N*d);
///         odd N -> [0, 1, ..., (N-1)/2, -(N-1)/2, ..., -1] / (N*d)
///
/// Equivalent to numpy.fft.fftfreq. Returns the frequency bin centers
/// in cycles per unit of the sample spacing. The positive branch runs
/// through (N-1)//2 — i.e. the select boundary is (N+1)/2, NOT N/2
/// (they differ for odd N; N/2 mis-signed bin k=(N-1)/2).
inline
Halide::Func fftfreq(int N, float d = 1.0f, std::string const& name = "fftfreq")
{
	nh_require(N > 0, "fftfreq requires N > 0, got %d", N);

	Halide::Func ret(name);
	Halide::Var x;

	int half = (N + 1) / 2;
	// For x in [0, half-1]: freq = x / (N*d)
	// For x in [half, N-1]: freq = (x - N) / (N*d)
	ret(x) = Halide::select(
		x < half,
		Halide::cast<float>(x),
		Halide::cast<float>(x - N)
	) / (N * d);

	return ret;
}

/// @brief fftfreq with RUNTIME size and spacing
/// @param n Window length as a runtime expression (integer)
/// @param d Sample spacing as a runtime expression
/// @param type Floating-point type to compute in (e.g. Halide::Float(32))
/// @param name Function name
/// @return Real-valued 1D Func of `type`
///
/// numpy.fft.fftfreq contract: positive frequencies through (n-1)//2,
/// i.e. the (n+1)/2 select boundary. The select picks the numerator; the
/// common division by n*d applies after it.
inline
Halide::Func fftfreq(Halide::Expr n, Halide::Expr d, Halide::Type type,
                     std::string const& name = "fftfreq_rt")
{
	nh_require(type.is_float(), "fftfreq compute type must be a float type");

	Halide::Func ret(name);
	Halide::Var k("k");

	ret(k) = Halide::select(
		k < (n + 1) / 2,
		Halide::cast(type, k),
		Halide::cast(type, k - n)
	) / (Halide::cast(type, n) * Halide::cast(type, d));

	return ret;
}

/// @brief Generate DFT sample frequencies for rfft
/// @param N Window length
/// @param d Sample spacing (default 1.0)
/// @param name Function name
/// @return 1D Func of size N/2+1: [0, 1, 2, ..., N/2] / (N*d)
///
/// Equivalent to numpy.fft.rfftfreq. Returns the frequency bin centers
/// for the positive half of the spectrum only.
inline
Halide::Func rfftfreq(int N, float d = 1.0f, std::string const& name = "rfftfreq")
{
	nh_require(N > 0, "rfftfreq requires N > 0, got %d", N);

	Halide::Func ret(name);
	Halide::Var x;

	// Simply: freq[k] = k / (N*d) for k in [0, N/2]
	ret(x) = Halide::cast<float>(x) / (N * d);

	return ret;
}

/// @brief rfftfreq with RUNTIME size and spacing
/// @param n Window length as a runtime expression (integer)
/// @param d Sample spacing as a runtime expression
/// @param type Floating-point type to compute in
/// @param name Function name
/// @return Real-valued 1D Func of `type`: freq[k] = k / (n*d)
///
/// Equivalent to numpy.fft.rfftfreq (all bins non-negative; the caller
/// realizes n/2+1 of them).
inline
Halide::Func rfftfreq(Halide::Expr n, Halide::Expr d, Halide::Type type,
                      std::string const& name = "rfftfreq_rt")
{
	nh_require(type.is_float(), "rfftfreq compute type must be a float type");

	Halide::Func ret(name);
	Halide::Var k("k");

	ret(k) = Halide::cast(type, k) / (Halide::cast(type, n) * Halide::cast(type, d));

	return ret;
}

NS_NUM_HALIDE_END
