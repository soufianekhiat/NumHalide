/// @file 50_views.cpp
/// @brief Example 50: Zero-copy strided views (transpose, slice, reshape)
///
/// Quadrant guide:
///   Top-left:     Original 256x256 matrix (gradient pattern)
///   Top-right:    Transposed view (same data, axes swapped)
///   Bottom-left:  Sliced view (rows 64..191 = middle 128 rows, all 256 cols)
///   Bottom-right: Reshaped view (flattened to 1D then viewed as 512x128)

#include "numhalide_all.h"
#include "stbi_png.h"
#include <iostream>
#include <cmath>

using namespace numhalide;

int main(int argc, char** argv) {
    try {
        const int half = 256;
        const int size = half * 2;
        const int MAT_W = half, MAT_H = half;  // 256x256

        // Generate source matrix: gradient + sine pattern
        Halide::Runtime::Buffer<float> mat(MAT_W, MAT_H);
        for (int r = 0; r < MAT_H; ++r)
            for (int c = 0; c < MAT_W; ++c) {
                float u = (float)c / (MAT_W - 1);
                float v = (float)r / (MAT_H - 1);
                mat(c, r) = u * v + 0.3f * std::sin(u * 8.0f * 3.14159f) * std::cos(v * 6.0f * 3.14159f);
            }

        // --- Create views ---
        // Transposed: view_transpose(mat) -- extents (H, W) = (256, 256), same here
        auto mat_T = view_transpose(mat);

        // Sliced: rows 64..191 (axis 1 = y = row) -> 256x128
        auto mat_S = view_slice(mat, 1, 64, 192);  // 128 rows

        // Reshaped: flatten 256x256 to 65536-element 1D, then view as 512x128
        auto mat_flat = view_reshape(mat, {65536});
        auto mat_R    = view_reshape(mat_flat, {512, 128});

        // Normalize mat to [0, 1] for display (all views share same data range)
        float mat_min = mat(0, 0), mat_max = mat(0, 0);
        for (int r = 0; r < MAT_H; ++r)
            for (int c = 0; c < MAT_W; ++c) {
                mat_min = std::min(mat_min, mat(c, r));
                mat_max = std::max(mat_max, mat(c, r));
            }
        float mat_range = std::max(mat_max - mat_min, 1e-6f);

        // Render 4 quadrants using CPU loops (Runtime::Buffer is O(1) view, no Halide JIT needed)
        Halide::Runtime::Buffer<uint8_t> img(size, size);
        img.fill(0);

        auto norm_byte = [&](float v) -> uint8_t {
            v = (v - mat_min) / mat_range;
            return (uint8_t)(std::min(255.0f, std::max(0.0f, v * 255.0f)));
        };

        for (int oy = 0; oy < size; ++oy) {
            for (int ox = 0; ox < size; ++ox) {
                int qx = ox / half, qy = oy / half;
                int lx = ox % half, ly = oy % half;
                float val;
                if (qy == 0 && qx == 0) {
                    // Top-left: original 256x256
                    val = mat(lx, ly);
                } else if (qy == 0 && qx == 1) {
                    // Top-right: transposed view — O(1) metadata swap
                    val = mat_T(ly, lx);
                } else if (qy == 1 && qx == 0) {
                    // Bottom-left: sliced (128 rows), scale ly [0,255] -> [0,127]
                    val = mat_S(lx, ly * 128 / 256);
                } else {
                    // Bottom-right: reshaped 512x128, scale lx [0,255]->[0,511], ly [0,255]->[0,127]
                    val = mat_R(lx * 512 / 256, ly * 128 / 256);
                }
                img(ox, oy) = norm_byte(val);
            }
        }

        const char* path = "out/50_views.png";
        if (!save_png(img, path)) {
            std::cerr << "Failed to save " << path << std::endl; return 1;
        }
        std::cout << "Saved to " << path << std::endl;
        std::cout << "\nQuadrant guide:" << std::endl;
        std::cout << "  Top-left:     Original 256x256 matrix" << std::endl;
        std::cout << "  Top-right:    Transposed view (O(1), zero-copy)" << std::endl;
        std::cout << "  Bottom-left:  Sliced view (middle 128 rows, scaled 2x)" << std::endl;
        std::cout << "  Bottom-right: Reshaped view (512x128 layout)" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl; return 1;
    }
}
