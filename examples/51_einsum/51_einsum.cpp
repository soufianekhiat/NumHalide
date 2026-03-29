/// @file 51_einsum.cpp
/// @brief Example 51: Einstein summation (einsum) visualizations
///
/// Demonstrates four einsum forms as a 512x512 heatmap PNG:
///   Top-left:     Matrix multiply A(4x6) @ B(6x5) -> C(4x5)  ("ij,jk->ik")
///   Top-right:    Outer product of two length-16 vectors       ("i,j->ij")
///   Bottom-left:  Batch matmul A(4x8x8) @ B(4x8x8) -> slice 0 ("bij,bjk->bik")
///   Bottom-right: Hadamard product of two 16x16 sinusoidal     ("ij,ij->ij")
///
/// Output: out/51_einsum.png

#include "numhalide_all.h"
#include "stbi_png.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>

using namespace numhalide;

static const float PI = 3.14159265358979323846f;

int main(int /*argc*/, char** /*argv*/) {
    try {
        std::cout << "Computing einsum examples..." << std::endl;

        // ----------------------------------------------------------------
        // Top-left: matmul via einsum "ij,jk->ik"
        //   A: 4x6 (shape {4,6})  B: 6x5 (shape {6,5})  C: 4x5
        // ----------------------------------------------------------------
        const int TL_M = 4, TL_K = 6, TL_N = 5;

        // A[i,j] = sin(i * 1.3f + j * 0.7f) mapped to [0,1]
        // B[j,k] = cos(j * 0.5f + k * 1.1f) mapped to [0,1]
        Halide::Buffer<float> buf_A(TL_K, TL_M);
        Halide::Buffer<float> buf_B(TL_N, TL_K);
        for (int i = 0; i < TL_M; ++i)
            for (int j = 0; j < TL_K; ++j)
                buf_A(j, i) = 0.5f + 0.5f * std::sin(i * 1.3f + j * 0.7f);
        for (int j = 0; j < TL_K; ++j)
            for (int k = 0; k < TL_N; ++k)
                buf_B(k, j) = 0.5f + 0.5f * std::cos(j * 0.5f + k * 1.1f);

        Halide::Func fA("fA"), fB("fB");
        Halide::Var x, y, z;
        fA(x, y) = buf_A(Halide::clamp(x, 0, TL_K - 1),
                         Halide::clamp(y, 0, TL_M - 1));
        fB(x, y) = buf_B(Halide::clamp(x, 0, TL_N - 1),
                         Halide::clamp(y, 0, TL_K - 1));

        Halide::Func fC = einsum("ij,jk->ik", fA, {TL_M, TL_K}, fB, {TL_K, TL_N});

        Halide::Runtime::Buffer<float> matmul_out(TL_N, TL_M);
        fC.realize(matmul_out);

        // Normalize matmul result to [0,1]
        float mm_min = matmul_out(0, 0), mm_max = matmul_out(0, 0);
        for (int r = 0; r < TL_M; ++r)
            for (int c = 0; c < TL_N; ++c) {
                mm_min = std::min(mm_min, matmul_out(c, r));
                mm_max = std::max(mm_max, matmul_out(c, r));
            }
        float mm_range = mm_max - mm_min + 1e-6f;

        // ----------------------------------------------------------------
        // Top-right: outer product "i,j->ij"
        //   a: length 16, b: length 16  -> 16x16 matrix
        // ----------------------------------------------------------------
        const int OP_N = 16;
        Halide::Buffer<float> buf_a(OP_N), buf_b(OP_N);
        for (int i = 0; i < OP_N; ++i) {
            buf_a(i) = std::sin(2.0f * PI * i / OP_N);
            buf_b(i) = std::cos(2.0f * PI * i / OP_N * 1.5f);
        }

        Halide::Func fa("fa"), fb("fb");
        fa(x) = buf_a(Halide::clamp(x, 0, OP_N - 1));
        fb(x) = buf_b(Halide::clamp(x, 0, OP_N - 1));

        Halide::Func fOuter = einsum("i,j->ij", fa, {OP_N}, fb, {OP_N});

        Halide::Runtime::Buffer<float> outer_out(OP_N, OP_N);
        fOuter.realize(outer_out);

        // Normalize outer product to [0,1]
        float op_min = outer_out(0, 0), op_max = outer_out(0, 0);
        for (int r = 0; r < OP_N; ++r)
            for (int c = 0; c < OP_N; ++c) {
                op_min = std::min(op_min, outer_out(c, r));
                op_max = std::max(op_max, outer_out(c, r));
            }
        float op_range = op_max - op_min + 1e-6f;

        // ----------------------------------------------------------------
        // Bottom-left: batch matmul "bij,bjk->bik"
        //   A: 4x8x8, B: 4x8x8 -> C: 4x8x8, render slice 0
        // ----------------------------------------------------------------
        const int BM_BATCH = 4, BM_M = 8, BM_K = 8, BM_N = 8;

        Halide::Buffer<float> buf_bA(BM_K, BM_M, BM_BATCH);
        Halide::Buffer<float> buf_bB(BM_N, BM_K, BM_BATCH);
        for (int b = 0; b < BM_BATCH; ++b) {
            float phase = 2.0f * PI * b / BM_BATCH;
            for (int i = 0; i < BM_M; ++i)
                for (int j = 0; j < BM_K; ++j)
                    buf_bA(j, i, b) = std::sin(i * 0.8f + j * 0.5f + phase);
            for (int j = 0; j < BM_K; ++j)
                for (int k = 0; k < BM_N; ++k)
                    buf_bB(k, j, b) = std::cos(j * 0.6f + k * 0.9f + phase);
        }

        Halide::Func fbA("fbA"), fbB("fbB");
        fbA(x, y, z) = buf_bA(Halide::clamp(x, 0, BM_K - 1),
                               Halide::clamp(y, 0, BM_M - 1),
                               Halide::clamp(z, 0, BM_BATCH - 1));
        fbB(x, y, z) = buf_bB(Halide::clamp(x, 0, BM_N - 1),
                               Halide::clamp(y, 0, BM_K - 1),
                               Halide::clamp(z, 0, BM_BATCH - 1));

        Halide::Func fBatch = einsum("bij,bjk->bik",
                                     fbA, {BM_BATCH, BM_M, BM_K},
                                     fbB, {BM_BATCH, BM_K, BM_N});

        // Realize full batch result then pick slice 0
        Halide::Runtime::Buffer<float> batch_out(BM_N, BM_M, BM_BATCH);
        fBatch.realize(batch_out);

        // Normalize slice 0 to [0,1]
        float bm_min = batch_out(0, 0, 0), bm_max = batch_out(0, 0, 0);
        for (int r = 0; r < BM_M; ++r)
            for (int c = 0; c < BM_N; ++c) {
                bm_min = std::min(bm_min, batch_out(c, r, 0));
                bm_max = std::max(bm_max, batch_out(c, r, 0));
            }
        float bm_range = bm_max - bm_min + 1e-6f;

        // ----------------------------------------------------------------
        // Bottom-right: Hadamard "ij,ij->ij" of two 16x16 sinusoidal
        // ----------------------------------------------------------------
        const int HD_N = 16;

        Halide::Buffer<float> buf_hA(HD_N, HD_N), buf_hB(HD_N, HD_N);
        for (int r = 0; r < HD_N; ++r)
            for (int c = 0; c < HD_N; ++c) {
                buf_hA(c, r) = 0.5f + 0.5f * std::sin(2.0f * PI * c / HD_N + r * 0.4f);
                buf_hB(c, r) = 0.5f + 0.5f * std::cos(2.0f * PI * r / HD_N + c * 0.7f);
            }

        Halide::Func fhA("fhA"), fhB("fhB");
        fhA(x, y) = buf_hA(Halide::clamp(x, 0, HD_N - 1),
                            Halide::clamp(y, 0, HD_N - 1));
        fhB(x, y) = buf_hB(Halide::clamp(x, 0, HD_N - 1),
                            Halide::clamp(y, 0, HD_N - 1));

        Halide::Func fHad = einsum("ij,ij->ij", fhA, {HD_N, HD_N}, fhB, {HD_N, HD_N});

        Halide::Runtime::Buffer<float> had_out(HD_N, HD_N);
        fHad.realize(had_out);

        // Hadamard values are already in [0,1] (products of [0,1] values)
        float hd_min = had_out(0, 0), hd_max = had_out(0, 0);
        for (int r = 0; r < HD_N; ++r)
            for (int c = 0; c < HD_N; ++c) {
                hd_min = std::min(hd_min, had_out(c, r));
                hd_max = std::max(hd_max, had_out(c, r));
            }
        float hd_range = hd_max - hd_min + 1e-6f;

        // ----------------------------------------------------------------
        // Render 512x512 output (4 quadrants of 256x256 each)
        // ----------------------------------------------------------------
        const int HALF = 256;
        const int SIZE = HALF * 2;

        Halide::Runtime::Buffer<uint8_t> img(SIZE, SIZE, 3);

        // Helper: blue-white-red colormap for normalized [0,1] value
        auto colormap = [](float t, uint8_t& r_out, uint8_t& g_out, uint8_t& b_out) {
            t = std::max(0.0f, std::min(1.0f, t));
            float r_f = t > 0.5f ? 1.0f : t * 2.0f;
            float g_f = 1.0f - std::abs(t - 0.5f) * 2.0f;
            float b_f = t < 0.5f ? 1.0f : (1.0f - t) * 2.0f;
            r_out = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, r_f * 255.0f)));
            g_out = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, g_f * 255.0f)));
            b_out = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, b_f * 255.0f)));
        };

        std::cout << "Rendering quadrants..." << std::endl;

        for (int py = 0; py < SIZE; ++py) {
            for (int px = 0; px < SIZE; ++px) {
                int qx = px / HALF;  // 0 = left, 1 = right
                int qy = py / HALF;  // 0 = top,  1 = bottom
                int lx = px % HALF;  // local x within quadrant
                int ly = py % HALF;  // local y within quadrant

                float t = 0.0f;
                uint8_t r_pix = 0, g_pix = 0, b_pix = 0;

                if (qy == 0 && qx == 0) {
                    // Top-left: matmul C(4x5) — scale tile up to 256x256
                    int r = ly * TL_M / HALF;
                    int c = lx * TL_N / HALF;
                    r = std::min(r, TL_M - 1);
                    c = std::min(c, TL_N - 1);
                    t = (matmul_out(c, r) - mm_min) / mm_range;
                    colormap(t, r_pix, g_pix, b_pix);
                } else if (qy == 0 && qx == 1) {
                    // Top-right: outer product 16x16 — scale up to 256x256
                    int r = ly * OP_N / HALF;
                    int c = lx * OP_N / HALF;
                    r = std::min(r, OP_N - 1);
                    c = std::min(c, OP_N - 1);
                    t = (outer_out(c, r) - op_min) / op_range;
                    colormap(t, r_pix, g_pix, b_pix);
                } else if (qy == 1 && qx == 0) {
                    // Bottom-left: batch matmul slice 0, 8x8 — scale up to 256x256
                    int r = ly * BM_M / HALF;
                    int c = lx * BM_N / HALF;
                    r = std::min(r, BM_M - 1);
                    c = std::min(c, BM_N - 1);
                    t = (batch_out(c, r, 0) - bm_min) / bm_range;
                    colormap(t, r_pix, g_pix, b_pix);
                } else {
                    // Bottom-right: Hadamard 16x16 — scale up to 256x256
                    int r = ly * HD_N / HALF;
                    int c = lx * HD_N / HALF;
                    r = std::min(r, HD_N - 1);
                    c = std::min(c, HD_N - 1);
                    t = (had_out(c, r) - hd_min) / hd_range;
                    colormap(t, r_pix, g_pix, b_pix);
                }

                img(px, py, 0) = r_pix;
                img(px, py, 1) = g_pix;
                img(px, py, 2) = b_pix;
            }
        }

        // ----------------------------------------------------------------
        // Save PNG
        // ----------------------------------------------------------------
        const char* path = "out/51_einsum.png";
        if (!save_png(img, path)) {
            std::cerr << "Failed to save " << path << "\n";
            return 1;
        }
        std::cout << "Saved " << path << "\n";
        std::cout << "  Top-left:     matmul einsum('ij,jk->ik') A(4x6)@B(6x5)\n";
        std::cout << "  Top-right:    outer product einsum('i,j->ij') len-16 vectors\n";
        std::cout << "  Bottom-left:  batch matmul einsum('bij,bjk->bik') 4x8x8, slice 0\n";
        std::cout << "  Bottom-right: Hadamard einsum('ij,ij->ij') 16x16 sinusoids\n";
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
