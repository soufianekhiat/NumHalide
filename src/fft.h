/// @file fft.h
/// @brief Fast Fourier Transform operations
///
/// Provides: fft, ifft, fft2d, ifft2d, fftshift, dft_1d, dft_2d (runtime sizes)
///
/// The compile-time-size forms compute the direct DFT matrix (O(N^2),
/// avoids Halide scheduling issues) and accept ANY size; the runtime-Expr
/// general-DFT overloads accept sizes as Halide::Exprs.
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
/// @param N Size of transform (any size — the body is a direct DFT)
/// @param sign -1 for forward FFT, +1 for inverse FFT
/// @param name Function name
/// @return Complex Func with FFT result
///
/// Note: No normalization is applied. For inverse FFT, divide by N.
/// Uses direct DFT computation which is O(N^2) but avoids Halide scheduling issues.
inline
Halide::Func fft_1d_c2c(Halide::Func input, int N, int sign, std::string const& name = "fft1d")
{
    nh_require(N > 0, "FFT size must be positive, got %d", N);
    nh_require(sign == -1 || sign == 1, "FFT sign must be -1 or +1");

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
/// @param rows Number of rows (any size — the body is a direct DFT)
/// @param cols Number of columns (any size — the body is a direct DFT)
/// @param sign -1 for forward, +1 for inverse
/// @param name Function name
/// @return Complex 2D FFT result
///
/// Uses separable DFT: first transform along x, then along y.
/// CAUTION on argument order: for input(x, y), `cols` is the x extent and
/// `rows` is the y extent — `cols` comes SECOND in the argument list.
inline
Halide::Func fft_2d_c2c(Halide::Func input, int rows, int cols, int sign,
                         std::string const& name = "fft2d")
{
    nh_require(rows > 0 && cols > 0, "FFT sizes must be positive, got %d x %d", rows, cols);

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
// General DFT (runtime-Expr sizes)
// -----------------------------------------------------------------------------

/// @brief Compute a general 1D DFT with a RUNTIME size
/// @param input Complex input Func (returns Tuple of real, imag)
/// @param n Size of transform as a runtime expression (any size, not just power of 2)
/// @param sign -1 for forward DFT, +1 for inverse DFT
/// @param type Floating-point type to compute in (e.g. Halide::Float(32))
/// @param name Function name
/// @return Complex Func with DFT result
///
/// X[k] = sum_{r=0}^{n-1} x[r] * exp(sign * j * 2 * pi * k * r / n)
///
/// A runtime size cannot drive the compile-time Cooley-Tukey recursion, so
/// this is the direct O(N^2) DFT. No normalization is applied
/// (dft_1d(+1) of dft_1d(-1) of x == n * x); use idft_1d_normalized for the
/// scaled inverse. The result is compute_root'd (same convention as
/// fft_1d_c2c) so calls chain safely.
inline
Halide::Func dft_1d(Halide::Func input, Halide::Expr n, int sign, Halide::Type type,
                    std::string const& name = "dft_1d")
{
    nh_require(sign == -1 || sign == 1, "DFT sign must be -1 or +1");
    nh_require(type.is_float(), "DFT compute type must be a float type");

    Halide::Func result(name);
    Halide::Var k("k");
    Halide::RDom r(0, n);

    // angle = sign * 2 * pi * k * r / n, computed in `type`.
    Halide::Expr two_pi = Halide::Internal::make_const(type, sign * 2.0 * M_PI);
    Halide::Expr angle = two_pi * Halide::cast(type, k) * Halide::cast(type, r) / Halide::cast(type, n);

    // x[r] * e^(j*angle) via the Tuple-complex helpers:
    // re = x_re*cos - x_im*sin, im = x_re*sin + x_im*cos
    Halide::Tuple xr = complex(Halide::cast(type, input(r)[0]),
                               Halide::cast(type, input(r)[1]));
    Halide::Tuple prod = complex_mul(xr, expj(angle));

    result(k) = Halide::Tuple(Halide::sum(prod[0]), Halide::sum(prod[1]));
    result.compute_root();

    return result;
}

/// @brief Compute the normalized inverse 1D DFT with a RUNTIME size
/// @param input Complex input Func (returns Tuple of real, imag)
/// @param n Size of transform as a runtime expression
/// @param type Floating-point type to compute in
/// @param name Function name
/// @return Normalized complex inverse-DFT result (unscaled inverse / n)
///
/// The unscaled inverse (dft_1d with sign +1, O(N^2)) is computed into a
/// compute_root'd raw stage, then divided by n — the raw-then-divide
/// staging keeps the pipeline staged for adjoint derivation.
inline
Halide::Func idft_1d_normalized(Halide::Func input, Halide::Expr n, Halide::Type type,
                                std::string const& name = "idft_1d_norm")
{
    Halide::Func raw = dft_1d(input, n, 1, type, name + "_raw");
    raw.compute_root();

    Halide::Func result(name);
    Halide::Var x("x");
    Halide::Expr nf = Halide::cast(type, n);

    result(x) = Halide::Tuple(raw(x)[0] / nf, raw(x)[1] / nf);

    return result;
}

/// @brief Compute a general 2D DFT with RUNTIME sizes
/// @param input Complex input 2D Func (returns Tuple of real, imag)
/// @param w Extent of dimension 0 (x) as a runtime expression
/// @param h Extent of dimension 1 (y) as a runtime expression
/// @param sign -1 for forward, +1 for inverse
/// @param type Floating-point type to compute in
/// @param name Function name
/// @return Complex 2D DFT result
///
/// Separable row-then-column DFT: first along x for each y into a
/// compute_root'd intermediate (name + "_row"), then along y. Runtime
/// sizes cannot drive the compile-time Cooley-Tukey recursion, so each
/// pass is the direct O(N^2) DFT (O(W*H*(W+H)) total). Unnormalized.
/// The result is compute_root'd (same convention as fft_2d_c2c) so calls
/// chain safely.
inline
Halide::Func dft_2d(Halide::Func input, Halide::Expr w, Halide::Expr h, int sign,
                    Halide::Type type, std::string const& name = "dft_2d")
{
    nh_require(sign == -1 || sign == 1, "DFT sign must be -1 or +1");
    nh_require(type.is_float(), "DFT compute type must be a float type");

    Halide::Expr two_pi = Halide::Internal::make_const(type, sign * 2.0 * M_PI);
    Halide::Var kx("kx"), ky("ky");

    // Step 1: DFT along x for each row y.
    Halide::Func row(name + "_row");
    {
        Halide::Var y("y");
        Halide::RDom rx(0, w);
        Halide::Expr angle = two_pi * Halide::cast(type, kx) * Halide::cast(type, rx) / Halide::cast(type, w);
        Halide::Tuple xr = complex(Halide::cast(type, input(rx, y)[0]),
                                   Halide::cast(type, input(rx, y)[1]));
        Halide::Tuple prod = complex_mul(xr, expj(angle));
        row(kx, y) = Halide::Tuple(Halide::sum(prod[0]), Halide::sum(prod[1]));
    }
    row.compute_root();

    // Step 2: DFT along y for each column kx.
    Halide::Func result(name);
    {
        Halide::RDom ry(0, h);
        Halide::Expr angle = two_pi * Halide::cast(type, ky) * Halide::cast(type, ry) / Halide::cast(type, h);
        Halide::Tuple xr = complex(Halide::cast(type, row(kx, ry)[0]),
                                   Halide::cast(type, row(kx, ry)[1]));
        Halide::Tuple prod = complex_mul(xr, expj(angle));
        result(kx, ky) = Halide::Tuple(Halide::sum(prod[0]), Halide::sum(prod[1]));
    }
    result.compute_root();

    return result;
}

/// @brief Compute the normalized inverse 2D DFT with RUNTIME sizes
/// @param input Complex input 2D Func (returns Tuple of real, imag)
/// @param w Extent of dimension 0 (x) as a runtime expression
/// @param h Extent of dimension 1 (y) as a runtime expression
/// @param type Floating-point type to compute in
/// @param name Function name
/// @return Normalized complex inverse-DFT result (unscaled inverse / (w*h))
///
/// The unscaled inverse (dft_2d with sign +1) is computed into a
/// compute_root'd raw stage, then divided by w*h — same raw-then-divide
/// staging as idft_1d_normalized.
inline
Halide::Func idft_2d_normalized(Halide::Func input, Halide::Expr w, Halide::Expr h,
                                Halide::Type type, std::string const& name = "idft_2d_norm")
{
    Halide::Func raw = dft_2d(input, w, h, 1, type, name + "_raw");
    raw.compute_root();

    Halide::Func result(name);
    Halide::Var x("x"), y("y");
    Halide::Expr scale = Halide::cast(type, w) * Halide::cast(type, h);

    result(x, y) = Halide::Tuple(raw(x, y)[0] / scale, raw(x, y)[1] / scale);

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
///
/// numpy conventions: fftshift gathers with offset ceil(N/2), ifftshift
/// with floor(N/2). The two coincide for even N; for odd N they differ,
/// and only the ceil/floor pair round-trips.
inline
Halide::Func fftshift_1d(Halide::Func input, int N, std::string const& name = "fftshift1d")
{
    Halide::Func result(name);
    Halide::Var x("x");

    int half = (N + 1) / 2;
    Halide::Expr src_x = (x + half) % N;

    result(x) = Halide::Tuple(input(src_x)[0], input(src_x)[1]);

    return result;
}

/// @brief fftshift with a RUNTIME size (same ceil(N/2) gather as the int
/// form; integer Expr arithmetic). Pure gather — no compute_root needed.
inline
Halide::Func fftshift_1d(Halide::Func input, Halide::Expr n, std::string const& name = "fftshift1d_rt")
{
    Halide::Func result(name);
    Halide::Var x("x");

    Halide::Expr half = (n + 1) / 2;
    Halide::Expr src_x = (x + half) % n;

    result(x) = Halide::Tuple(input(src_x)[0], input(src_x)[1]);

    return result;
}

/// @brief Shift zero-frequency component to center of 2D spectrum
/// @param input Complex input Func
/// @param rows Number of rows
/// @param cols Number of columns
/// @param name Function name
/// @return Shifted spectrum
///
/// Same ceil(N/2) gather offset per axis as fftshift_1d (numpy semantics).
inline
Halide::Func fftshift_2d(Halide::Func input, int rows, int cols, std::string const& name = "fftshift2d")
{
    Halide::Func result(name);
    Halide::Var x("x"), y("y");

    int half_cols = (cols + 1) / 2;
    int half_rows = (rows + 1) / 2;

    Halide::Expr src_x = (x + half_cols) % cols;
    Halide::Expr src_y = (y + half_rows) % rows;

    result(x, y) = Halide::Tuple(input(src_x, src_y)[0], input(src_x, src_y)[1]);

    return result;
}

/// @brief 2-D fftshift with RUNTIME sizes (same ceil per-axis gather as
/// the int form: x gathers with the cols offset, y with the rows offset).
/// Pure gather — no compute_root needed.
inline
Halide::Func fftshift_2d(Halide::Func input, Halide::Expr rows, Halide::Expr cols,
                         std::string const& name = "fftshift2d_rt")
{
    Halide::Func result(name);
    Halide::Var x("x"), y("y");

    Halide::Expr half_cols = (cols + 1) / 2;
    Halide::Expr half_rows = (rows + 1) / 2;

    Halide::Expr src_x = (x + half_cols) % cols;
    Halide::Expr src_y = (y + half_rows) % rows;

    result(x, y) = Halide::Tuple(input(src_x, src_y)[0], input(src_x, src_y)[1]);

    return result;
}

/// @brief Inverse fftshift (floor(N/2) gather offset — coincides with
/// fftshift for even N, differs for odd N; the pair round-trips)
inline
Halide::Func ifftshift_1d(Halide::Func input, int N, std::string const& name = "ifftshift1d")
{
    Halide::Func result(name);
    Halide::Var x("x");

    int half = N / 2;
    Halide::Expr src_x = (x + half) % N;

    result(x) = Halide::Tuple(input(src_x)[0], input(src_x)[1]);

    return result;
}

/// @brief Inverse fftshift with a RUNTIME size (same floor(N/2) gather as
/// the int form). Pure gather — no compute_root needed.
inline
Halide::Func ifftshift_1d(Halide::Func input, Halide::Expr n, std::string const& name = "ifftshift1d_rt")
{
    Halide::Func result(name);
    Halide::Var x("x");

    Halide::Expr half = n / 2;
    Halide::Expr src_x = (x + half) % n;

    result(x) = Halide::Tuple(input(src_x)[0], input(src_x)[1]);

    return result;
}

/// @brief Inverse fftshift 2D (floor offsets per axis)
inline
Halide::Func ifftshift_2d(Halide::Func input, int rows, int cols, std::string const& name = "ifftshift2d")
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

/// @brief Inverse 2-D fftshift with RUNTIME sizes (floor offsets per
/// axis, same as the int form). Pure gather — no compute_root needed.
inline
Halide::Func ifftshift_2d(Halide::Func input, Halide::Expr rows, Halide::Expr cols,
                          std::string const& name = "ifftshift2d_rt")
{
    Halide::Func result(name);
    Halide::Var x("x"), y("y");

    Halide::Expr half_cols = cols / 2;
    Halide::Expr half_rows = rows / 2;

    Halide::Expr src_x = (x + half_cols) % cols;
    Halide::Expr src_y = (y + half_rows) % rows;

    result(x, y) = Halide::Tuple(input(src_x, src_y)[0], input(src_x, src_y)[1]);

    return result;
}

/// @brief Convert real array to complex (imaginary = 0)
/// @param input Real-valued input Func
/// @param name Function name
/// @return Complex Func
///
/// Float inputs keep their type (f64 stays f64); integer inputs are
/// promoted to f32.
inline
Halide::Func real_to_complex(Halide::Func input, std::string const& name = "r2c")
{
    Halide::Type t = input.types()[0];
    if (!t.is_float()) t = Halide::Float(32);

    Halide::Func result(name);
    Halide::Var x("x");

    result(x) = Halide::Tuple(Halide::cast(t, input(x)), Halide::Internal::make_zero(t));

    return result;
}

/// @brief Convert real 2D array to complex
///
/// Same typing rule as real_to_complex.
inline
Halide::Func real_to_complex_2d(Halide::Func input, std::string const& name = "r2c_2d")
{
    Halide::Type t = input.types()[0];
    if (!t.is_float()) t = Halide::Float(32);

    Halide::Func result(name);
    Halide::Var x("x"), y("y");

    result(x, y) = Halide::Tuple(Halide::cast(t, input(x, y)), Halide::Internal::make_zero(t));

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
