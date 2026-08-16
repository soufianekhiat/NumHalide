/// @file fft_fast.h
/// @brief O(N log N) Cooley-Tukey radix-2 FFT for power-of-2 sizes
#pragma once
#include "fft.h"   // for complex helpers
#include "common.h"
#include <vector>
#include <cmath>
#include <cstdint>

NS_NUM_HALIDE_BEGIN

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/// @brief Forward 1D FFT, O(N log N) Cooley-Tukey DIT
/// Input: complex Func (Tuple of re, im), N = power of 2
/// Output: complex Func (Tuple of re, im), size N
inline Halide::Func fft_fast(Halide::Func input, int N,
                              std::string const& name = "fft_fast")
{
    nh_require(N > 0 && (N & (N-1)) == 0,
               "fft_fast: N must be power of 2, got %d", N);

    const int log2N = (int)std::round(std::log2((double)N));

    // Compute in the input's own float type (f64 stays f64); integer
    // inputs keep the historical f32 path. For f32 the folded twiddle
    // constants are bit-identical to the old -2.0f * (float)M_PI product.
    Halide::Type t = input.types()[0];
    if (!t.is_float()) t = Halide::Float(32);

    // Build compile-time bit-reversal table
    std::vector<int32_t> br(N);
    for (int i = 0; i < N; ++i) {
        int rev = 0, tmp = i;
        for (int b = 0; b < log2N; ++b) { rev = (rev << 1) | (tmp & 1); tmp >>= 1; }
        br[i] = rev;
    }
    Halide::Buffer<int32_t> br_buf(N, name + "_br_tbl");
    for (int i = 0; i < N; ++i) br_buf(i) = br[i];

    Halide::Var k("k");

    // Stage 0: bit-reversed permutation of input
    // Clamp br_idx to [0, N-1] so bounds inference knows the access range even
    // when 'input' is a compute_root() Func (e.g. inside ifft_fast).
    Halide::Func stage0(name + "_s0");
    Halide::Expr br_idx = Halide::clamp(br_buf(Halide::clamp(k, 0, N - 1)), 0, N - 1);
    stage0(k) = Halide::Tuple(
        Halide::cast(t, input(br_idx)[0]),
        Halide::cast(t, input(br_idx)[1])
    );
    stage0.compute_root();

    // DIT butterfly stages
    Halide::Func prev = stage0;
    for (int s = 0; s < log2N; ++s) {
        int stride = 1 << s;     // half-group size
        int group  = stride * 2; // full group size

        Halide::Func curr(name + "_s" + std::to_string(s + 1));

        Halide::Expr wg     = k % group;
        Halide::Expr is_top = wg < stride;

        // Twiddle index (0..stride-1)
        Halide::Expr tw_idx = Halide::select(is_top, wg, wg - stride);
        // Forward FFT: angle = -2*pi*tw_idx / group
        Halide::Expr angle  = Halide::Internal::make_const(t, -2.0 * M_PI)
                              * Halide::cast(t, tw_idx)
                              / Halide::Internal::make_const(t, (double)group);
        Halide::Expr tw_re  = Halide::cos(angle);
        Halide::Expr tw_im  = Halide::sin(angle);

        // Partner indices — CLAMP to fix Halide bounds inference
        Halide::Expr k_top = Halide::clamp(Halide::select(is_top, k, k - stride), 0, N - 1);
        Halide::Expr k_bot = Halide::clamp(Halide::select(is_top, k + stride, k), 0, N - 1);

        Halide::Expr a_re = prev(k_top)[0], a_im = prev(k_top)[1];
        Halide::Expr b_re = prev(k_bot)[0], b_im = prev(k_bot)[1];

        // W * b
        Halide::Expr wb_re = tw_re * b_re - tw_im * b_im;
        Halide::Expr wb_im = tw_re * b_im + tw_im * b_re;

        // top: a + W*b, bottom: a - W*b
        curr(k) = Halide::Tuple(
            Halide::select(is_top, a_re + wb_re, a_re - wb_re),
            Halide::select(is_top, a_im + wb_im, a_im - wb_im)
        );
        curr.compute_root();
        prev = curr;
    }
    return prev;
}

/// @brief Normalized inverse 1D FFT using IFFT(X) = conj(FFT(conj(X))) / N
inline Halide::Func ifft_fast(Halide::Func input, int N,
                               std::string const& name = "ifft_fast")
{
    nh_require(N > 0 && (N & (N-1)) == 0,
               "ifft_fast: N must be power of 2, got %d", N);

    Halide::Var k("k");
    // Conjugate input
    Halide::Func ci(name + "_ci");
    ci(k) = Halide::Tuple(input(k)[0], -input(k)[1]);

    auto fwd = fft_fast(ci, N, name + "_fwd");

    // Divide in the transform's own type (fwd follows the input's float
    // type); the folded (T)N constant is bit-identical for f32.
    Halide::Type t = fwd.types()[0];
    Halide::Expr nf = Halide::Internal::make_const(t, (double)N);

    Halide::Func ret(name);
    ret(k) = Halide::Tuple(fwd(k)[0] / nf, -fwd(k)[1] / nf);
    return ret;
}

/// @brief Forward 2D FFT (separable: x-FFT then y-FFT), O(N*M*log(N*M))
inline Halide::Func fft2d_fast(Halide::Func input, int rows, int cols,
                                std::string const& name = "fft2d_fast")
{
    nh_require((rows & (rows-1)) == 0,
               "fft2d_fast: rows must be power of 2, got %d", rows);
    nh_require((cols & (cols-1)) == 0,
               "fft2d_fast: cols must be power of 2, got %d", cols);

    const int logC = (int)std::round(std::log2((double)cols));
    const int logR = (int)std::round(std::log2((double)rows));

    // Compute in the input's own float type (f64 stays f64); integer
    // inputs keep the historical f32 path.
    Halide::Type t = input.types()[0];
    if (!t.is_float()) t = Halide::Float(32);

    // ---- build bit-reversal tables ----
    std::vector<int32_t> brx(cols), bry(rows);
    auto build_br = [](std::vector<int32_t>& tbl, int logN) {
        int N = (int)tbl.size();
        for (int i = 0; i < N; ++i) {
            int rev = 0, tmp = i;
            for (int b = 0; b < logN; ++b) { rev = (rev << 1) | (tmp & 1); tmp >>= 1; }
            tbl[i] = rev;
        }
    };
    build_br(brx, logC);
    build_br(bry, logR);

    Halide::Buffer<int32_t> brx_buf(cols, name + "_brx_tbl");
    Halide::Buffer<int32_t> bry_buf(rows, name + "_bry_tbl");
    for (int i = 0; i < cols; ++i) brx_buf(i) = brx[i];
    for (int i = 0; i < rows; ++i) bry_buf(i) = bry[i];

    Halide::Var x("x"), y("y");

    // ---- Stage x0: bit-reversed along x ----
    // Clamp bx/by to [0, extent-1] so bounds inference stays bounded when
    // 'input' is a compute_root() Func (e.g. inside ifft2d_fast).
    Halide::Func sx0(name + "_sx0");
    Halide::Expr bx = Halide::clamp(brx_buf(Halide::clamp(x, 0, cols - 1)), 0, cols - 1);
    sx0(x, y) = Halide::Tuple(
        Halide::cast(t, input(bx, y)[0]),
        Halide::cast(t, input(bx, y)[1])
    );
    sx0.compute_root();

    // ---- Butterfly stages along x ----
    Halide::Func prevx = sx0;
    for (int s = 0; s < logC; ++s) {
        int stride = 1 << s, group = stride * 2;
        Halide::Func curr(name + "_sx" + std::to_string(s + 1));
        Halide::Expr wg     = x % group;
        Halide::Expr is_top = wg < stride;
        Halide::Expr tw_idx = Halide::select(is_top, wg, wg - stride);
        Halide::Expr angle  = Halide::Internal::make_const(t, -2.0 * M_PI)
                              * Halide::cast(t, tw_idx)
                              / Halide::Internal::make_const(t, (double)group);
        Halide::Expr tw_re  = Halide::cos(angle), tw_im = Halide::sin(angle);
        Halide::Expr x_top  = Halide::clamp(Halide::select(is_top, x, x - stride), 0, cols - 1);
        Halide::Expr x_bot  = Halide::clamp(Halide::select(is_top, x + stride, x), 0, cols - 1);
        Halide::Expr a_re = prevx(x_top, y)[0], a_im = prevx(x_top, y)[1];
        Halide::Expr b_re = prevx(x_bot, y)[0], b_im = prevx(x_bot, y)[1];
        Halide::Expr wb_re = tw_re * b_re - tw_im * b_im;
        Halide::Expr wb_im = tw_re * b_im + tw_im * b_re;
        curr(x, y) = Halide::Tuple(
            Halide::select(is_top, a_re + wb_re, a_re - wb_re),
            Halide::select(is_top, a_im + wb_im, a_im - wb_im));
        curr.compute_root();
        prevx = curr;
    }

    // ---- Stage y0: bit-reversed along y ----
    Halide::Func sy0(name + "_sy0");
    Halide::Expr by = Halide::clamp(bry_buf(Halide::clamp(y, 0, rows - 1)), 0, rows - 1);
    sy0(x, y) = Halide::Tuple(prevx(x, by)[0], prevx(x, by)[1]);
    sy0.compute_root();

    // ---- Butterfly stages along y ----
    Halide::Func prevy = sy0;
    for (int s = 0; s < logR; ++s) {
        int stride = 1 << s, group = stride * 2;
        Halide::Func curr(name + "_sy" + std::to_string(s + 1));
        Halide::Expr wg     = y % group;
        Halide::Expr is_top = wg < stride;
        Halide::Expr tw_idx = Halide::select(is_top, wg, wg - stride);
        Halide::Expr angle  = Halide::Internal::make_const(t, -2.0 * M_PI)
                              * Halide::cast(t, tw_idx)
                              / Halide::Internal::make_const(t, (double)group);
        Halide::Expr tw_re  = Halide::cos(angle), tw_im = Halide::sin(angle);
        Halide::Expr y_top  = Halide::clamp(Halide::select(is_top, y, y - stride), 0, rows - 1);
        Halide::Expr y_bot  = Halide::clamp(Halide::select(is_top, y + stride, y), 0, rows - 1);
        Halide::Expr a_re = prevy(x, y_top)[0], a_im = prevy(x, y_top)[1];
        Halide::Expr b_re = prevy(x, y_bot)[0], b_im = prevy(x, y_bot)[1];
        Halide::Expr wb_re = tw_re * b_re - tw_im * b_im;
        Halide::Expr wb_im = tw_re * b_im + tw_im * b_re;
        curr(x, y) = Halide::Tuple(
            Halide::select(is_top, a_re + wb_re, a_re - wb_re),
            Halide::select(is_top, a_im + wb_im, a_im - wb_im));
        curr.compute_root();
        prevy = curr;
    }
    return prevy;
}

/// @brief Normalized inverse 2D FFT
inline Halide::Func ifft2d_fast(Halide::Func input, int rows, int cols,
                                 std::string const& name = "ifft2d_fast")
{
    Halide::Var x("x"), y("y");
    Halide::Func ci(name + "_ci");
    ci(x, y) = Halide::Tuple(input(x, y)[0], -input(x, y)[1]);
    auto fwd = fft2d_fast(ci, rows, cols, name + "_fwd");
    // Scale in the transform's own type (fwd follows the input's float
    // type); the folded 1/(rows*cols) constant is bit-identical for f32.
    Halide::Type t = fwd.types()[0];
    Halide::Expr scale = Halide::Internal::make_one(t)
                         / Halide::Internal::make_const(t, (double)rows * (double)cols);
    Halide::Func ret(name);
    ret(x, y) = Halide::Tuple(fwd(x, y)[0] * scale, -fwd(x, y)[1] * scale);
    return ret;
}

NS_NUM_HALIDE_END
