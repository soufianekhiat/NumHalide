/// @file 52_la_runtime.cpp
/// @brief Example 52: Runtime-parametrized linear algebra (la_large.h)
///
/// Demonstrates cholesky_large, qr_large and svd_large on a 16×16 matrix.
/// The 512×512 PNG output is divided into four 256×256 quadrants:
///
///   Top-left:     Cholesky factor L of a 16×16 PD matrix (heatmap).
///                 Adjacent to it: the original PD matrix A (two 128×128 tiles).
///   Top-right:    QR decomposition — Q matrix (left tile) and R matrix (right tile),
///                 each 16×16 scaled to 128×128 within the 256×256 quadrant.
///   Bottom-left:  SVD singular value bar chart — 16 bars showing σ_i.
///   Bottom-right: Reconstruction error bar chart — Frobenius ||A - L*L^T||_F for
///                 n = 4, 8, 12, 16 shown as horizontal bars.
///
/// Output: out/52_la_runtime.png

#include "numhalide_all.h"
#include "la_large.h"
#include "stbi_png.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <vector>

using namespace numhalide;

// =============================================================================
// Helpers
// =============================================================================

/// Build a symmetric positive-definite n×n matrix Buffer<float>(n_cols, n_rows).
/// Entry [row=i, col=j]: diagonal = n + diag_strength + 0.5*sin(2*i),
///                       off-diag  = 0.3*sin(min(i,j)*n + max(i,j))  [symmetric].
static Halide::Buffer<float> make_pd_buf(int n, float diag_strength)
{
    Halide::Buffer<float> data(n, n);  // (col, row)
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            int lo = std::min(i, j), hi = std::max(i, j);
            data(j, i) = (i == j)
                ? static_cast<float>(n) + diag_strength
                  + 0.5f * std::sin(static_cast<float>(2 * i))
                : 0.3f * std::sin(static_cast<float>(lo * n + hi));
        }
    return data;
}

/// Wrap a Buffer as a Halide Func (with clamp guard).
static Halide::Func buf_to_func(Halide::Buffer<float> data,
    int rows, int cols, const std::string& nm)
{
    Halide::Func f(nm);
    Halide::Var x("x"), y("y");
    f(x, y) = data(Halide::clamp(x, 0, cols - 1),
                   Halide::clamp(y, 0, rows - 1));
    return f;
}

/// Realize Func(col, row) into a row-major std::vector<float>.
static std::vector<float> realize_flat(Halide::Func f, int rows, int cols)
{
    Halide::Runtime::Buffer<float> tmp(cols, rows);
    f.realize(tmp);
    std::vector<float> out(static_cast<size_t>(rows * cols));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            out[static_cast<size_t>(i * cols + j)] = tmp(j, i);
    return out;
}

/// Map a float value in [lo, hi] to a uint8 intensity.
static uint8_t map_val(float v, float lo, float hi)
{
    if (hi <= lo) return 128;
    float t = (v - lo) / (hi - lo);
    t = std::max(0.0f, std::min(1.0f, t));
    return static_cast<uint8_t>(t * 255.0f + 0.5f);
}

/// Draw a 16×16 matrix as a scaled heatmap into img[0..255][0..255] grayscale.
/// dst_x, dst_y: top-left corner of the tile; tile_w, tile_h: tile size in pixels.
/// Uses bilinear-style nearest-neighbour scaling: pixel (px, py) -> matrix [py*n/h][px*n/w].
static void draw_matrix_tile(Halide::Runtime::Buffer<uint8_t>& img,
    const std::vector<float>& mat, int n_rows, int n_cols,
    int dst_x, int dst_y, int tile_w, int tile_h,
    float lo, float hi)
{
    for (int py = 0; py < tile_h; ++py) {
        int mi = py * n_rows / tile_h;
        if (mi >= n_rows) mi = n_rows - 1;
        for (int px = 0; px < tile_w; ++px) {
            int mj = px * n_cols / tile_w;
            if (mj >= n_cols) mj = n_cols - 1;
            float v = mat[static_cast<size_t>(mi * n_cols + mj)];
            img(dst_x + px, dst_y + py) = map_val(v, lo, hi);
        }
    }
}

/// Range [min, max] of a flat vector.
static std::pair<float,float> vec_range(const std::vector<float>& v)
{
    float lo = v[0], hi = v[0];
    for (float x : v) { lo = std::min(lo, x); hi = std::max(hi, x); }
    return {lo, hi};
}

/// Frobenius norm of the difference: A - B (both flat row-major, size n*n).
static float frob_diff(const std::vector<float>& A,
                       const std::vector<float>& B, int n)
{
    double sum = 0.0;
    for (int k = 0; k < n * n; ++k) {
        double d = static_cast<double>(A[static_cast<size_t>(k)])
                 - static_cast<double>(B[static_cast<size_t>(k)]);
        sum += d * d;
    }
    return static_cast<float>(std::sqrt(sum));
}

/// Multiply two dense matrices (row-major flat), result is flat row-major.
static std::vector<float> matmul_cpu(const std::vector<float>& A, int ra, int ca,
                                     const std::vector<float>& B, int rb, int cb)
{
    (void)rb;
    std::vector<float> C(static_cast<size_t>(ra * cb), 0.0f);
    for (int i = 0; i < ra; ++i)
        for (int k = 0; k < ca; ++k) {
            float aik = A[static_cast<size_t>(i * ca + k)];
            for (int j = 0; j < cb; ++j)
                C[static_cast<size_t>(i * cb + j)] += aik * B[static_cast<size_t>(k * cb + j)];
        }
    return C;
}

/// Transpose a flat row-major matrix.
static std::vector<float> transpose_cpu(const std::vector<float>& A, int rows, int cols)
{
    std::vector<float> At(static_cast<size_t>(rows * cols));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            At[static_cast<size_t>(j * rows + i)] = A[static_cast<size_t>(i * cols + j)];
    return At;
}

// =============================================================================
// main
// =============================================================================

int main(int /*argc*/, char** /*argv*/)
{
    try {
        std::cout << "NumHalide T3-B: Runtime-sized LA demonstration (n=16)\n\n";

        const int n = 16;
        const float DIAG = 20.0f;

        // --- Build PD matrix ---
        auto A_buf  = make_pd_buf(n, DIAG);
        auto A_func = buf_to_func(A_buf, n, n, "A16");

        // Flatten A for later comparisons
        std::vector<float> A_flat(static_cast<size_t>(n * n));
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                A_flat[static_cast<size_t>(i * n + j)] = A_buf(j, i);

        // ---- Cholesky --------------------------------------------------------
        std::cout << "Computing Cholesky (n=" << n << ")...\n";
        auto L_func = cholesky_large(A_func, n, "chol16");
        auto L_flat = realize_flat(L_func, n, n);

        // Verify: L * L^T ≈ A
        auto Lt_flat = transpose_cpu(L_flat, n, n);
        auto LLt_flat = matmul_cpu(L_flat, n, n, Lt_flat, n, n);
        float chol_err = frob_diff(LLt_flat, A_flat, n);
        std::cout << "  Cholesky ||L*L^T - A||_F = " << std::fixed
                  << std::setprecision(6) << chol_err << "\n";

        // ---- QR --------------------------------------------------------------
        std::cout << "Computing QR (m=" << n << ", n=" << n << ")...\n";
        auto [Q_func, R_func] = qr_large(A_func, n, n, "qr16");
        auto Q_flat = realize_flat(Q_func, n, n);
        auto R_flat = realize_flat(R_func, n, n);

        // Verify: Q^T Q ≈ I
        auto Qt_flat = transpose_cpu(Q_flat, n, n);
        auto QtQ_flat = matmul_cpu(Qt_flat, n, n, Q_flat, n, n);
        std::vector<float> I_flat(static_cast<size_t>(n * n), 0.0f);
        for (int i = 0; i < n; ++i) I_flat[static_cast<size_t>(i * n + i)] = 1.0f;
        float orth_err = frob_diff(QtQ_flat, I_flat, n);
        std::cout << "  QR orthogonality ||Q^T Q - I||_F = " << orth_err << "\n";

        auto QR_flat = matmul_cpu(Q_flat, n, n, R_flat, n, n);
        float qr_err = frob_diff(QR_flat, A_flat, n);
        std::cout << "  QR reconstruction ||A - Q*R||_F = " << qr_err << "\n";

        // ---- SVD (use n=8 — SVD Jacobi scales O(n^3) and n=16 takes hours) ----
        const int svd_n = 8;
        std::cout << "Computing SVD (m=" << svd_n << ", n=" << svd_n << ")...\n";
        auto A8_buf  = make_pd_buf(svd_n, DIAG);
        auto A8_func = buf_to_func(A8_buf, svd_n, svd_n, "A8svd");
        auto [U_func, S_func, Vt_func] = svd_large(A8_func, svd_n, svd_n, -1, "svd8");

        Halide::Runtime::Buffer<float> S_buf(svd_n);
        S_func.realize(S_buf);

        std::cout << "  Singular values: ";
        std::vector<float> sv(static_cast<size_t>(svd_n));
        for (int i = 0; i < svd_n; ++i) {
            sv[static_cast<size_t>(i)] = S_buf(i);
            std::cout << std::fixed << std::setprecision(3) << sv[static_cast<size_t>(i)];
            if (i < svd_n - 1) std::cout << ", ";
        }
        std::cout << "\n";

        // ---- Reconstruction errors for n=4,8,12,16 ---------------------------
        std::cout << "Computing Cholesky errors for n=4,8,12,16...\n";
        const int sizes[4] = {4, 8, 12, 16};
        float chol_errors[4];
        for (int si = 0; si < 4; ++si) {
            int sz = sizes[si];
            auto Ab = make_pd_buf(sz, DIAG);
            auto Af = buf_to_func(Ab, sz, sz, "Asub" + std::to_string(sz));
            auto Lf = cholesky_large(Af, sz, "cholsub" + std::to_string(sz));
            auto Lf_flat = realize_flat(Lf, sz, sz);
            auto Lft_flat = transpose_cpu(Lf_flat, sz, sz);
            auto LLtf_flat = matmul_cpu(Lf_flat, sz, sz, Lft_flat, sz, sz);

            std::vector<float> Af_flat(static_cast<size_t>(sz * sz));
            for (int i = 0; i < sz; ++i)
                for (int j = 0; j < sz; ++j)
                    Af_flat[static_cast<size_t>(i * sz + j)] = Ab(j, i);

            chol_errors[si] = frob_diff(LLtf_flat, Af_flat, sz);
            std::cout << "  n=" << sz << ": ||L*L^T - A||_F = "
                      << chol_errors[si] << "\n";
        }

        // =================================================================
        // Render 512×512 output image (grayscale)
        // =================================================================
        std::cout << "\nRendering output image...\n";

        Halide::Runtime::Buffer<uint8_t> img(512, 512);

        // Initialize to dark background
        for (int y = 0; y < 512; ++y)
            for (int x = 0; x < 512; ++x)
                img(x, y) = 20;

        // ---- TOP-LEFT quadrant (0..255, 0..255): Cholesky ----
        // Two 128×128 heatmap tiles: left = A, right = L
        {
            auto [lo_a, hi_a] = vec_range(A_flat);
            auto [lo_l, hi_l] = vec_range(L_flat);
            // A heatmap in left tile (x: 0..127, y: 0..255)
            draw_matrix_tile(img, A_flat,  n, n, 0,   0, 128, 256, lo_a, hi_a);
            // L heatmap in right tile (x: 128..255, y: 0..255)
            draw_matrix_tile(img, L_flat,  n, n, 128, 0, 128, 256, lo_l, hi_l);

            // Label separator: thin vertical line at x=128
            for (int y = 0; y < 256; ++y) img(128, y) = 180;
        }

        // ---- TOP-RIGHT quadrant (256..511, 0..255): QR ----
        // Q heatmap in left tile (x: 256..383), R in right tile (x: 384..511)
        {
            auto [lo_q, hi_q] = vec_range(Q_flat);
            auto [lo_r, hi_r] = vec_range(R_flat);
            draw_matrix_tile(img, Q_flat, n, n, 256, 0, 128, 256, lo_q, hi_q);
            draw_matrix_tile(img, R_flat, n, n, 384, 0, 128, 256, lo_r, hi_r);

            for (int y = 0; y < 256; ++y) img(384, y) = 180;
        }

        // ---- BOTTOM-LEFT quadrant (0..255, 256..511): SVD singular value bars ----
        {
            // Find max singular value for scaling
            float sv_max = *std::max_element(sv.begin(), sv.end());
            if (sv_max < 1e-6f) sv_max = 1.0f;

            const int bq_x = 0, bq_y = 256;
            const int bq_w = 256, bq_h = 256;
            const int bar_w = bq_w / svd_n;   // 32 px per bar for n=8
            const int pad   = 2;
            const int max_bar_h = bq_h - 20;

            for (int i = 0; i < svd_n; ++i) {
                float norm = sv[static_cast<size_t>(i)] / sv_max;
                int bar_h = static_cast<int>(norm * max_bar_h);
                int bx0 = bq_x + i * bar_w + pad;
                int bx1 = bq_x + (i + 1) * bar_w - pad;
                int by0 = bq_y + bq_h - 10 - bar_h;
                int by1 = bq_y + bq_h - 10;

                // Brightness encodes magnitude (brighter = larger)
                uint8_t col = static_cast<uint8_t>(80 + static_cast<int>(norm * 175));

                for (int py = by0; py < by1; ++py)
                    for (int px = bx0; px < bx1 && px < 256; ++px)
                        if (px >= 0 && py >= 0 && py < 512)
                            img(px, py) = col;
            }

            // Baseline
            for (int px = bq_x; px < bq_x + bq_w; ++px)
                img(px, bq_y + bq_h - 10) = 160;
        }

        // ---- BOTTOM-RIGHT quadrant (256..511, 256..511): Cholesky error bars ----
        {
            float max_err = 0.0f;
            for (int si = 0; si < 4; ++si)
                max_err = std::max(max_err, chol_errors[si]);
            if (max_err < 1e-8f) max_err = 1.0f;

            const int bq_x = 256, bq_y = 256;
            const int bq_w = 256, bq_h = 256;

            // 4 horizontal bars, evenly spaced
            const int bar_h_px = 30;
            const int gap      = (bq_h - 4 * bar_h_px) / 5;
            const int max_bar_w = bq_w - 40;

            for (int si = 0; si < 4; ++si) {
                float norm = chol_errors[si] / max_err;
                int bw = static_cast<int>(norm * max_bar_w);
                int by0 = bq_y + gap + si * (bar_h_px + gap);
                int by1 = by0 + bar_h_px;
                int bx0 = bq_x + 20;
                int bx1 = bx0 + bw;

                uint8_t col = static_cast<uint8_t>(60 + static_cast<int>(norm * 180));

                for (int py = by0; py < by1 && py < 512; ++py)
                    for (int px = bx0; px < bx1 && px < 512; ++px)
                        if (px >= 256 && py >= 256)
                            img(px, py) = col;
            }
        }

        // Quadrant dividers (cross at 256,256)
        for (int y = 0; y < 512; ++y) img(256, y) = 200;
        for (int x = 0; x < 512; ++x) img(x, 256) = 200;

        // ---- Save ----
        const char* path = "out/52_la_runtime.png";
        if (!save_png(img, path)) {
            std::cerr << "Failed to save " << path << "\n";
            return 1;
        }
        std::cout << "Saved " << path << "\n";
        std::cout << "\nLayout:\n";
        std::cout << "  Top-left:     A matrix (left) | Cholesky L (right)  [16x16 heatmaps]\n";
        std::cout << "  Top-right:    Q matrix (left) | R matrix   (right)  [16x16 heatmaps]\n";
        std::cout << "  Bottom-left:  SVD singular value bars (16 bars)\n";
        std::cout << "  Bottom-right: Cholesky ||L*L^T-A||_F for n=4,8,12,16\n";

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
