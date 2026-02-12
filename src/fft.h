/// @file fft.h
/// @brief Fast Fourier Transform operations
///
/// Provides: fft, ifft, fft2d, ifft2d, fftshift
///
/// Based on Halide's FFT implementation from apps/fft/
/// Uses Cooley-Tukey FFT algorithm for power-of-2 sizes.
///
/// Complex numbers are represented as Halide Tuples: (real, imaginary)

#pragma once

#include "common.h"
#include "numhalide.h"
#include "shape.h"

#include <cmath>
#include <vector>

NS_NUM_HALIDE_BEGIN

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// -----------------------------------------------------------------------------
// Complex Number Helpers
// -----------------------------------------------------------------------------

/// @brief Create a complex number expression from real and imaginary parts
inline Halide::Tuple complex(Halide::Expr real, Halide::Expr imag) {
    return Halide::Tuple(real, imag);
}

/// @brief Create a complex number from a real value (imaginary = 0)
inline Halide::Tuple complex_from_real(Halide::Expr real) {
    return Halide::Tuple(real, Halide::cast(real.type(), 0));
}

/// @brief Get real part of complex Tuple
inline Halide::Expr complex_re(Halide::Tuple z) {
    return z[0];
}

/// @brief Get imaginary part of complex Tuple
inline Halide::Expr complex_im(Halide::Tuple z) {
    return z[1];
}

/// @brief Complex addition
inline Halide::Tuple complex_add(Halide::Tuple a, Halide::Tuple b) {
    return Halide::Tuple(a[0] + b[0], a[1] + b[1]);
}

/// @brief Complex subtraction
inline Halide::Tuple complex_sub(Halide::Tuple a, Halide::Tuple b) {
    return Halide::Tuple(a[0] - b[0], a[1] - b[1]);
}

/// @brief Complex multiplication
inline Halide::Tuple complex_mul(Halide::Tuple a, Halide::Tuple b) {
    // (a + bi)(c + di) = (ac - bd) + (ad + bc)i
    return Halide::Tuple(
        a[0] * b[0] - a[1] * b[1],
        a[0] * b[1] + a[1] * b[0]
    );
}

/// @brief Complex conjugate
inline Halide::Tuple complex_conj(Halide::Tuple z) {
    return Halide::Tuple(z[0], -z[1]);
}

/// @brief Compute e^(j*x) = cos(x) + j*sin(x)
inline Halide::Tuple expj(Halide::Expr x) {
    return Halide::Tuple(Halide::cos(x), Halide::sin(x));
}

/// @brief Complex magnitude squared
inline Halide::Expr complex_abs2(Halide::Tuple z) {
    return z[0] * z[0] + z[1] * z[1];
}

/// @brief Complex magnitude
inline Halide::Expr complex_abs(Halide::Tuple z) {
    return Halide::sqrt(complex_abs2(z));
}

/// @brief Scale complex number by real value
inline Halide::Tuple complex_scale(Halide::Tuple z, Halide::Expr s) {
    return Halide::Tuple(z[0] * s, z[1] * s);
}

// -----------------------------------------------------------------------------
// 1D FFT (Cooley-Tukey, radix-2, decimation in time)
// -----------------------------------------------------------------------------

/// @brief Compute 1D FFT of a complex-valued function using DFT matrix
/// @param input Complex input Func (returns Tuple of real, imag)
/// @param N Size of transform (must be power of 2)
/// @param sign -1 for forward FFT, +1 for inverse FFT
/// @param name Function name
/// @return Complex Func with FFT result
///
/// Note: No normalization is applied. For inverse FFT, divide by N.
/// Uses direct DFT computation which is O(N^2) but avoids Halide scheduling issues.
inline
Halide::Func fft_1d_c2c(Halide::Func input, int N, int sign, std::string const& name = "fft1d")
{
    nh_require(nullptr, (N & (N - 1)) == 0, "FFT requires power of 2 size, got %d", N);
    nh_require(nullptr, sign == -1 || sign == 1, "FFT sign must be -1 or +1");

    const float pi = static_cast<float>(M_PI);

    Halide::Func result(name);
    Halide::Var k("k");
    Halide::RDom n(0, N);

    // DFT: X[k] = sum_{n=0}^{N-1} x[n] * exp(sign * j * 2 * pi * k * n / N)
    Halide::Expr angle = sign * 2.0f * pi * Halide::cast<float>(k) * Halide::cast<float>(n) / static_cast<float>(N);
    Halide::Expr tw_re = Halide::cos(angle);
    Halide::Expr tw_im = Halide::sin(angle);

    // x[n] * twiddle = (x_re + j*x_im) * (tw_re + j*tw_im)
    //                = (x_re*tw_re - x_im*tw_im) + j*(x_re*tw_im + x_im*tw_re)
    Halide::Expr x_re = input(n)[0];
    Halide::Expr x_im = input(n)[1];
    Halide::Expr prod_re = x_re * tw_re - x_im * tw_im;
    Halide::Expr prod_im = x_re * tw_im + x_im * tw_re;

    result(k) = Halide::Tuple(Halide::sum(prod_re), Halide::sum(prod_im));
    result.compute_root();

    return result;
}

/// @brief Compute forward 1D FFT
/// @param input Complex input (Tuple Func)
/// @param N Size of transform (power of 2)
/// @param name Function name
/// @return Complex FFT result
inline
Halide::Func fft(Halide::Func input, int N, std::string const& name = "fft")
{
    return fft_1d_c2c(input, N, -1, name);
}

/// @brief Compute inverse 1D FFT (unnormalized)
/// @param input Complex input (Tuple Func)
/// @param N Size of transform (power of 2)
/// @param name Function name
/// @return Complex IFFT result (divide by N for proper normalization)
inline
Halide::Func ifft(Halide::Func input, int N, std::string const& name = "ifft")
{
    return fft_1d_c2c(input, N, 1, name);
}

/// @brief Compute normalized inverse 1D FFT
/// @param input Complex input (Tuple Func)
/// @param N Size of transform (power of 2)
/// @param name Function name
/// @return Normalized complex IFFT result
inline
Halide::Func ifft_normalized(Halide::Func input, int N, std::string const& name = "ifft_norm")
{
    auto raw = fft_1d_c2c(input, N, 1, name + "_raw");

    Halide::Func result(name);
    Halide::Var x("x");
    Halide::Expr scale = 1.0f / static_cast<float>(N);

    result(x) = Halide::Tuple(raw(x)[0] * scale, raw(x)[1] * scale);

    return result;
}

// -----------------------------------------------------------------------------
// 2D FFT
// -----------------------------------------------------------------------------

/// @brief Compute 2D FFT of a complex-valued function using separable DFT
/// @param input Complex input Func (2D, returns Tuple)
/// @param rows Number of rows (must be power of 2)
/// @param cols Number of columns (must be power of 2)
/// @param sign -1 for forward, +1 for inverse
/// @param name Function name
/// @return Complex 2D FFT result
///
/// Uses separable DFT: first transform along x, then along y.
inline
Halide::Func fft_2d_c2c(Halide::Func input, int rows, int cols, int sign,
                         std::string const& name = "fft2d")
{
    nh_require(nullptr, (rows & (rows - 1)) == 0, "FFT requires power of 2 rows, got %d", rows);
    nh_require(nullptr, (cols & (cols - 1)) == 0, "FFT requires power of 2 cols, got %d", cols);

    const float pi = static_cast<float>(M_PI);

    Halide::Var kx("kx"), ky("ky");

    // Step 1: DFT along x (columns) for each row
    Halide::Func fft_x(name + "_x");
    Halide::RDom nx(0, cols);

    Halide::Expr angle_x = sign * 2.0f * pi * Halide::cast<float>(kx) * Halide::cast<float>(nx) / static_cast<float>(cols);
    Halide::Expr tw_x_re = Halide::cos(angle_x);
    Halide::Expr tw_x_im = Halide::sin(angle_x);

    Halide::Expr x_re = input(nx, ky)[0];
    Halide::Expr x_im = input(nx, ky)[1];
    Halide::Expr prod_x_re = x_re * tw_x_re - x_im * tw_x_im;
    Halide::Expr prod_x_im = x_re * tw_x_im + x_im * tw_x_re;

    fft_x(kx, ky) = Halide::Tuple(Halide::sum(prod_x_re), Halide::sum(prod_x_im));
    fft_x.compute_root();

    // Step 2: DFT along y (rows) for each column
    Halide::Func result(name);
    Halide::RDom ny(0, rows);

    Halide::Expr angle_y = sign * 2.0f * pi * Halide::cast<float>(ky) * Halide::cast<float>(ny) / static_cast<float>(rows);
    Halide::Expr tw_y_re = Halide::cos(angle_y);
    Halide::Expr tw_y_im = Halide::sin(angle_y);

    Halide::Expr y_re = fft_x(kx, ny)[0];
    Halide::Expr y_im = fft_x(kx, ny)[1];
    Halide::Expr prod_y_re = y_re * tw_y_re - y_im * tw_y_im;
    Halide::Expr prod_y_im = y_re * tw_y_im + y_im * tw_y_re;

    result(kx, ky) = Halide::Tuple(Halide::sum(prod_y_re), Halide::sum(prod_y_im));
    result.compute_root();

    return result;
}

/// @brief Compute forward 2D FFT
inline
Halide::Func fft2d(Halide::Func input, int rows, int cols, std::string const& name = "fft2d")
{
    return fft_2d_c2c(input, rows, cols, -1, name);
}

/// @brief Compute inverse 2D FFT (unnormalized)
inline
Halide::Func ifft2d(Halide::Func input, int rows, int cols, std::string const& name = "ifft2d")
{
    return fft_2d_c2c(input, rows, cols, 1, name);
}

/// @brief Compute normalized inverse 2D FFT
inline
Halide::Func ifft2d_normalized(Halide::Func input, int rows, int cols,
                                std::string const& name = "ifft2d_norm")
{
    auto raw = fft_2d_c2c(input, rows, cols, 1, name + "_raw");

    Halide::Func result(name);
    Halide::Var x("x"), y("y");
    Halide::Expr scale = 1.0f / static_cast<float>(rows * cols);

    result(x, y) = Halide::Tuple(raw(x, y)[0] * scale, raw(x, y)[1] * scale);

    return result;
}

// -----------------------------------------------------------------------------
// FFT Utilities
// -----------------------------------------------------------------------------

/// @brief Shift zero-frequency component to center of spectrum
/// @param input Complex input Func
/// @param N Size of 1D array
/// @param name Function name
/// @return Shifted spectrum
inline
Halide::Func fftshift_1d(Halide::Func input, int N, std::string const& name = "fftshift1d")
{
    Halide::Func result(name);
    Halide::Var x("x");

    int half = N / 2;
    Halide::Expr src_x = (x + half) % N;

    result(x) = Halide::Tuple(input(src_x)[0], input(src_x)[1]);

    return result;
}

/// @brief Shift zero-frequency component to center of 2D spectrum
/// @param input Complex input Func
/// @param rows Number of rows
/// @param cols Number of columns
/// @param name Function name
/// @return Shifted spectrum
inline
Halide::Func fftshift_2d(Halide::Func input, int rows, int cols, std::string const& name = "fftshift2d")
{
    Halide::Func result(name);
    Halide::Var x("x"), y("y");

    int half_cols = cols / 2;
    int half_rows = rows / 2;

    Halide::Expr src_x = (x + half_cols) % cols;
    Halide::Expr src_y = (y + half_rows) % rows;

    result(x, y) = Halide::Tuple(input(src_x, src_y)[0], input(src_x, src_y)[1]);

    return result;
}

/// @brief Inverse fftshift (same as fftshift for even sizes)
inline
Halide::Func ifftshift_1d(Halide::Func input, int N, std::string const& name = "ifftshift1d")
{
    return fftshift_1d(input, N, name);
}

/// @brief Inverse fftshift 2D
inline
Halide::Func ifftshift_2d(Halide::Func input, int rows, int cols, std::string const& name = "ifftshift2d")
{
    return fftshift_2d(input, rows, cols, name);
}

/// @brief Convert real array to complex (imaginary = 0)
/// @param input Real-valued input Func
/// @param name Function name
/// @return Complex Func
inline
Halide::Func real_to_complex(Halide::Func input, std::string const& name = "r2c")
{
    Halide::Func result(name);
    Halide::Var x("x");

    result(x) = Halide::Tuple(Halide::cast<float>(input(x)), 0.0f);

    return result;
}

/// @brief Convert real 2D array to complex
inline
Halide::Func real_to_complex_2d(Halide::Func input, std::string const& name = "r2c_2d")
{
    Halide::Func result(name);
    Halide::Var x("x"), y("y");

    result(x, y) = Halide::Tuple(Halide::cast<float>(input(x, y)), 0.0f);

    return result;
}

/// @brief Compute power spectrum (magnitude squared)
/// @param complex_input Complex input Func
/// @param name Function name
/// @return Real-valued power spectrum
inline
Halide::Func power_spectrum(Halide::Func complex_input, std::string const& name = "power_spectrum")
{
    Halide::Func result(name);
    Halide::Var x("x");

    result(x) = complex_input(x)[0] * complex_input(x)[0] +
                complex_input(x)[1] * complex_input(x)[1];

    return result;
}

/// @brief Compute 2D power spectrum
inline
Halide::Func power_spectrum_2d(Halide::Func complex_input, std::string const& name = "power_spectrum_2d")
{
    Halide::Func result(name);
    Halide::Var x("x"), y("y");

    result(x, y) = complex_input(x, y)[0] * complex_input(x, y)[0] +
                   complex_input(x, y)[1] * complex_input(x, y)[1];

    return result;
}

NS_NUM_HALIDE_END
