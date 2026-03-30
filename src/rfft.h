/// @file rfft.h
/// @brief Real FFT extensions and frequency utilities
///
/// Provides: rfft, irfft, rfft2d, irfft2d, fftfreq, rfftfreq

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
/// @param N Size of transform (must be power of 2)
/// @param name Function name
/// @return Complex Func (Tuple of real, imag) with N/2+1 meaningful bins
///
/// Takes a real 1D signal, converts to complex, runs full FFT, and returns
/// the result. For real input, only the first N/2+1 bins are unique due to
/// conjugate symmetry: X[k] = conj(X[N-k]).
inline
Halide::Func rfft(Halide::Func input, int N, std::string const& name = "rfft")
{
	nh_require((N & (N - 1)) == 0, "rfft requires power of 2 size, got %d", N);

	// Convert real input to complex (imaginary = 0)
	Halide::Func complex_in("rfft_cin");
	Halide::Var x;
	complex_in(x) = Halide::Tuple(Halide::cast<float>(input(x)), 0.0f);

	// Full FFT
	auto full = fft(complex_in, N, name + "_full");

	// Return the full result; user reads bins [0, N/2+1)
	Halide::Func ret(name);
	ret(x) = full(x);
	return ret;
}

/// @brief Inverse real FFT: takes N/2+1 complex bins, returns N real values
/// @param input Complex Func with N/2+1 bins (Tuple of real, imag)
/// @param N Original signal size (must be power of 2)
/// @param name Function name
/// @return Real-valued 1D Func of size N
///
/// Reconstructs full spectrum using conjugate symmetry:
///   X[k] for k < N/2+1 is input(k)
///   X[k] for k >= N/2+1 is conj(X[N-k])
/// Then applies normalized inverse FFT and returns real part.
inline
Halide::Func irfft(Halide::Func input, int N, std::string const& name = "irfft")
{
	nh_require((N & (N - 1)) == 0, "irfft requires power of 2 size, got %d", N);

	int half = N / 2 + 1;

	Halide::Func full_spectrum("irfft_full");
	Halide::Var x;

	// Reconstruct full spectrum via conjugate symmetry
	// X[k] for k < half comes from input directly
	// X[k] for k >= half: X[k] = conj(X[N-k])
	Halide::Expr k = x;
	Halide::Expr mirror = N - x;

	full_spectrum(x) = Halide::select(
		k < half,
		Halide::Tuple(input(k)[0], input(k)[1]),
		Halide::Tuple(input(mirror)[0], -input(mirror)[1])
	);

	// Normalized inverse FFT
	auto result = ifft_normalized(full_spectrum, N, name + "_ifft");

	// Return real part only
	Halide::Func ret(name);
	ret(x) = result(x)[0];
	return ret;
}

// -----------------------------------------------------------------------------
// 2D Real FFT
// -----------------------------------------------------------------------------

/// @brief Real-input 2D FFT
/// @param input Real-valued 2D Func (not Tuple)
/// @param rows Number of rows (must be power of 2)
/// @param cols Number of columns (must be power of 2)
/// @param name Function name
/// @return Complex 2D Func (Tuple of real, imag)
///
/// Converts real input to complex and runs full 2D FFT.
inline
Halide::Func rfft2d(Halide::Func input, int rows, int cols, std::string const& name = "rfft2d")
{
	nh_require((rows & (rows - 1)) == 0, "rfft2d requires power of 2 rows, got %d", rows);
	nh_require((cols & (cols - 1)) == 0, "rfft2d requires power of 2 cols, got %d", cols);

	// Convert real to complex
	Halide::Func complex_in("rfft2d_cin");
	Halide::Var x, y;
	complex_in(x, y) = Halide::Tuple(Halide::cast<float>(input(x, y)), 0.0f);

	// Full 2D FFT
	auto full = fft2d(complex_in, rows, cols, name + "_full");

	Halide::Func ret(name);
	ret(x, y) = full(x, y);
	return ret;
}

/// @brief Inverse real 2D FFT
/// @param input Complex 2D Func
/// @param rows Number of rows (must be power of 2)
/// @param cols Number of columns (must be power of 2)
/// @param name Function name
/// @return Real-valued 2D Func
///
/// Applies normalized inverse 2D FFT and returns real part.
inline
Halide::Func irfft2d(Halide::Func input, int rows, int cols, std::string const& name = "irfft2d")
{
	nh_require((rows & (rows - 1)) == 0, "irfft2d requires power of 2 rows, got %d", rows);
	nh_require((cols & (cols - 1)) == 0, "irfft2d requires power of 2 cols, got %d", cols);

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
/// @return 1D Func: [0, 1, 2, ..., N/2-1, -N/2, ..., -1] / (N*d)
///
/// Equivalent to numpy.fft.fftfreq. Returns the frequency bin centers
/// in cycles per unit of the sample spacing.
inline
Halide::Func fftfreq(int N, float d = 1.0f, std::string const& name = "fftfreq")
{
	nh_require(N > 0, "fftfreq requires N > 0, got %d", N);

	Halide::Func ret(name);
	Halide::Var x;

	int half = N / 2;
	// For x in [0, half-1]: freq = x / (N*d)
	// For x in [half, N-1]: freq = (x - N) / (N*d)
	ret(x) = Halide::select(
		x <= half - 1,
		Halide::cast<float>(x),
		Halide::cast<float>(x - N)
	) / (N * d);

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

NS_NUM_HALIDE_END
