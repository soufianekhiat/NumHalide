/// @file 46_batched_la.cpp
/// @brief Example 46: Batched linear algebra operations
///
/// Demonstrates cholesky, qr_gs, and svd_jacobi applied to a batch of 4x4 matrices.
///
/// Batch: 4 different 4x4 positive-definite matrices:
///   A[b][i][j] = delta(i,j)*(4+b) + 0.5*sin(b+i+j)
///
/// Output: out/46_batched_la.png (512x512, four 256x256 quadrants)
///   Top-left:     Cholesky factors heat map (4 matrices tiled 2x2, each in 64x64 area)
///   Top-right:    QR reconstruction error ||A-QR||_F per batch as horizontal bars
///   Bottom-left:  SVD singular value spectrum: bars per batch
///   Bottom-right: Q orthogonality check ||Q^TQ - I||_F per batch as brightness bars

#include "numhalide_all.h"
#include "stbi_png.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>

using namespace numhalide;

static constexpr int N_BATCH = 4;
static constexpr int MAT_N   = 4;
static constexpr int MAT_M   = 6;   // for QR: 6x4 (m>=n)

/// Build a 2D Halide::Func from a flat row-major vector
static Halide::Func make_mat(int rows, int cols,
    const std::vector<float>& vals, const std::string& nm)
{
    Halide::Buffer<float> buf(cols, rows);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            buf(c, r) = vals[(size_t)(r * cols + c)];
    Halide::Func f(nm);
    Halide::Var x, y;
    f(x, y) = buf(x, y);
    return f;
}

/// Generate a diagonal-dominant symmetric PD matrix for batch b (n x n)
static std::vector<float> gen_pd_matrix(int b, int n)
{
    std::vector<float> data((size_t)(n * n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            float v = (i == j) ? float(4 + b) : 0.5f * std::sin(float(b + i + j));
            data[(size_t)(i * n + j)] = v;
        }
    return data;
}

/// Generate a tall matrix for QR (m x n)
static std::vector<float> gen_tall_matrix(int b, int m, int n)
{
    std::vector<float> data((size_t)(m * n));
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            data[(size_t)(i * n + j)] = float(1 + (i + b) % 3) * (float(j + 1) / float(n))
                                        + 0.1f * std::cos(float(b * m + i + j));
    return data;
}

int main() {
    try {
        const int OUT  = 512;
        const int QUAD = 256;
        const int n = MAT_N;
        const int m = MAT_M;

        // Realize all batch results into CPU buffers first
        // -----------------------------------------------------------------------
        // Cholesky: N_BATCH x n x n
        std::vector<Halide::Runtime::Buffer<float>> L_bufs;
        for (int b = 0; b < N_BATCH; ++b) {
            auto A_data = gen_pd_matrix(b, n);
            auto A = make_mat(n, n, A_data, "chol_A_b" + std::to_string(b));
            auto L = cholesky(A, n, "chol_L_b" + std::to_string(b));
            L.compute_root();
            Halide::Runtime::Buffer<float> L_buf(n, n);
            L.realize(L_buf);
            L_bufs.push_back(L_buf);
        }

        // QR: N_BATCH of m x n matrices, compute ||A - QR||_F
        std::vector<float> qr_errors(N_BATCH, 0.0f);
        float qr_err_max = 0.0f;
        for (int b = 0; b < N_BATCH; ++b) {
            auto A_data = gen_tall_matrix(b, m, n);
            auto A = make_mat(m, n, A_data, "qr_A_b" + std::to_string(b));
            auto qr = qr_gs(A, m, n, "qr_b" + std::to_string(b));
            qr.Q.compute_root();
            qr.R.compute_root();
            Halide::Runtime::Buffer<float> Q_buf(n, m);
            Halide::Runtime::Buffer<float> R_buf(n, n);
            qr.Q.realize(Q_buf);
            qr.R.realize(R_buf);

            // Compute ||A - QR||_F
            float err_sq = 0.0f;
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                    float qr_val = 0.0f;
                    for (int k = 0; k < n; ++k)
                        qr_val += Q_buf(k, i) * R_buf(j, k);
                    float diff = A_data[(size_t)(i * n + j)] - qr_val;
                    err_sq += diff * diff;
                }
            }
            qr_errors[(size_t)b] = std::sqrt(err_sq);
            qr_err_max = std::max(qr_err_max, qr_errors[(size_t)b]);
        }
        if (qr_err_max < 1e-8f) qr_err_max = 1e-8f;

        // SVD: N_BATCH of n x n, singular values
        std::vector<std::vector<float>> sv_all(N_BATCH);
        float sv_max = 0.0f;
        for (int b = 0; b < N_BATCH; ++b) {
            auto A_data = gen_pd_matrix(b, n);
            auto A = make_mat(n, n, A_data, "svd_A_b" + std::to_string(b));
            auto svd = svd_jacobi(A, n, n, 12, "svd_b" + std::to_string(b));
            svd.S.compute_root();
            Halide::Runtime::Buffer<float> S_buf(n);
            svd.S.realize(S_buf);
            sv_all[(size_t)b].resize((size_t)n);
            for (int k = 0; k < n; ++k) {
                sv_all[(size_t)b][(size_t)k] = S_buf(k);
                sv_max = std::max(sv_max, S_buf(k));
            }
        }
        if (sv_max < 1e-8f) sv_max = 1e-8f;

        // Q orthogonality: ||Q^TQ - I||_F per batch
        std::vector<float> orth_errs(N_BATCH, 0.0f);
        float orth_max = 0.0f;
        for (int b = 0; b < N_BATCH; ++b) {
            auto A_data = gen_tall_matrix(b, m, n);
            auto A = make_mat(m, n, A_data, "orth_A_b" + std::to_string(b));
            auto qr = qr_gs(A, m, n, "orth_qr_b" + std::to_string(b));
            qr.Q.compute_root();
            Halide::Runtime::Buffer<float> Q_buf(n, m);
            qr.Q.realize(Q_buf);

            float err_sq = 0.0f;
            for (int c1 = 0; c1 < n; ++c1) {
                for (int c2 = 0; c2 < n; ++c2) {
                    float dot = 0.0f;
                    for (int r = 0; r < m; ++r)
                        dot += Q_buf(c1, r) * Q_buf(c2, r);
                    float expected = (c1 == c2) ? 1.0f : 0.0f;
                    float diff = dot - expected;
                    err_sq += diff * diff;
                }
            }
            orth_errs[(size_t)b] = std::sqrt(err_sq);
            orth_max = std::max(orth_max, orth_errs[(size_t)b]);
        }
        if (orth_max < 1e-8f) orth_max = 1e-8f;

        // -----------------------------------------------------------------------
        // Build output image: 512x512 CPU-side
        // -----------------------------------------------------------------------
        Halide::Runtime::Buffer<uint8_t> result(OUT, OUT);
        result.fill(0);

        auto set_pixel = [&](int px, int py, float v) {
            if (px >= 0 && px < OUT && py >= 0 && py < OUT)
                result(px, py) = (uint8_t)(std::min(255.0f, std::max(0.0f, v * 255.0f)));
        };

        // Separator
        for (int i = 0; i < OUT; ++i) {
            set_pixel(QUAD, i, 0.5f);
            set_pixel(i, QUAD, 0.5f);
        }

        // -----------------------------------------------------------------------
        // Top-left (Q0): Cholesky factor heat maps
        //   4 matrices tiled in 2x2 grid, each rendered in 128x128 pixels
        //   within the 256x256 quadrant. Cell i is at offset (col_b*128, row_b*128).
        // -----------------------------------------------------------------------
        {
            const int CELL = QUAD / 2;   // 128 pixels per matrix
            const int PIX  = CELL / n;   // pixels per matrix element (~32 for n=4)
            for (int b = 0; b < N_BATCH; ++b) {
                int bx = (b % 2) * CELL;
                int by = (b / 2) * CELL;
                // Find max |L| for this batch for normalization
                float lmax = 1e-6f;
                for (int r = 0; r < n; ++r)
                    for (int c = 0; c < n; ++c)
                        lmax = std::max(lmax, std::abs(L_bufs[(size_t)b](c, r)));
                for (int r = 0; r < n; ++r) {
                    for (int c = 0; c < n; ++c) {
                        float v = (L_bufs[(size_t)b](c, r) / lmax) * 0.5f + 0.5f;
                        int px0 = bx + c * PIX;
                        int py0 = by + r * PIX;
                        for (int dy = 0; dy < PIX; ++dy)
                            for (int dx = 0; dx < PIX; ++dx)
                                set_pixel(px0 + dx, py0 + dy, v);
                    }
                }
            }
        }

        // -----------------------------------------------------------------------
        // Top-right (Q1): QR reconstruction error per batch as horizontal bars
        // -----------------------------------------------------------------------
        {
            const int BAR_H = QUAD / N_BATCH;  // 64 pixels tall per bar
            for (int b = 0; b < N_BATCH; ++b) {
                float width_frac = qr_errors[(size_t)b] / qr_err_max;
                int bar_width = (int)(width_frac * QUAD);
                int py0 = b * BAR_H;
                float brightness = 0.3f + 0.7f * (float)b / float(N_BATCH - 1);
                for (int dy = 2; dy < BAR_H - 2; ++dy) {
                    for (int dx = 0; dx < QUAD; ++dx) {
                        float v = (dx < bar_width) ? brightness : 0.1f;
                        set_pixel(QUAD + 1 + dx, py0 + dy, v);
                    }
                }
            }
        }

        // -----------------------------------------------------------------------
        // Bottom-left (Q2): SVD singular value spectrum — bars per batch
        //   4 batches, each has n=4 singular values
        //   Layout: 4 groups of 4 bars, each bar is 16 pixels wide
        // -----------------------------------------------------------------------
        {
            const int GROUP_W = QUAD / N_BATCH;   // 64 per batch group
            const int BAR_W   = GROUP_W / n;      // ~16 per SV bar
            for (int b = 0; b < N_BATCH; ++b) {
                for (int k = 0; k < n; ++k) {
                    float sv_frac = sv_all[(size_t)b][(size_t)k] / sv_max;
                    int bar_h = (int)(sv_frac * (QUAD - 4));
                    int px0 = b * GROUP_W + k * BAR_W;
                    float brightness = 0.4f + 0.6f * float(k) / float(n - 1);
                    for (int dx = 1; dx < BAR_W - 1; ++dx) {
                        for (int dh = 0; dh < bar_h; ++dh) {
                            int py = QUAD + (QUAD - 2 - dh);
                            set_pixel(px0 + dx, py, brightness);
                        }
                    }
                }
            }
        }

        // -----------------------------------------------------------------------
        // Bottom-right (Q3): Orthogonality error ||Q^TQ-I||_F per batch as bars
        // -----------------------------------------------------------------------
        {
            const int BAR_H = QUAD / N_BATCH;
            for (int b = 0; b < N_BATCH; ++b) {
                float width_frac = orth_errs[(size_t)b] / orth_max;
                int bar_width = std::max(1, (int)(width_frac * (QUAD - 2)));
                int py0 = QUAD + b * BAR_H;
                float brightness = 0.3f + 0.7f * float(b) / float(N_BATCH - 1);
                for (int dy = 2; dy < BAR_H - 2; ++dy) {
                    for (int dx = 0; dx < QUAD; ++dx) {
                        float v = (dx < bar_width) ? brightness : 0.08f;
                        set_pixel(QUAD + 1 + dx, py0 + dy, v);
                    }
                }
            }
        }

        if (save_png(result, "out/46_batched_la.png")) {
            std::cout << "Saved out/46_batched_la.png\n";
            // Print summary
            std::cout << std::fixed << std::setprecision(6);
            std::cout << "\nQR reconstruction errors per batch:\n";
            for (int b = 0; b < N_BATCH; ++b)
                std::cout << "  batch " << b << ": ||A-QR||_F = " << qr_errors[(size_t)b] << "\n";
            std::cout << "\nQ orthogonality errors per batch:\n";
            for (int b = 0; b < N_BATCH; ++b)
                std::cout << "  batch " << b << ": ||Q^TQ-I||_F = " << orth_errs[(size_t)b] << "\n";
        } else {
            std::cerr << "Failed to save PNG\n";
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
