/// @file 23_cumulative.cpp
/// @brief Example 23: Cumulative operations
///
/// Demonstrates:
///   - linear gradient (raw image)
///   - cumulative brightness sweep (darken left to right)
///   - vertical cumulative sweep
///   - first derivative edge pattern (high contrast stripes)
///
/// Output: out/23_cumulative.png

#include "numhalide_all.h"
#include "stbi_png.h"
#include <iostream>

using namespace numhalide;

int main(int argc, char **argv) {
    try {
        const int half = 256;
        const int size = half * 2;  // 512x512

        std::cout << "Cumulative operations demonstration" << std::endl;
        std::cout << "Output size: " << size << "x" << size << std::endl;
        std::cout << std::endl;

        Halide::Var ox("ox"), oy("oy");
        Halide::Func output("output");

        Halide::Expr qx = ox / half;  // 0=left, 1=right
        Halide::Expr qy = oy / half;  // 0=top,  1=bottom
        Halide::Expr lx = ox % half;
        Halide::Expr ly = oy % half;

        // Normalized coords [0,1)
        Halide::Expr u = Halide::cast<float>(lx) / half;
        Halide::Expr v = Halide::cast<float>(ly) / half;

        // --- Top-left: linear gradient (raw image) ---
        // Simple diagonal gradient combining horizontal and vertical position
        Halide::Expr grad_val = (u + v) * 0.5f;

        // --- Top-right: cumulative brightness sweep (darken left to right) ---
        // Simulates a cumulative sum effect: brightness accumulates as u increases,
        // modulated by a sin pattern to make it visually interesting
        Halide::Expr wave = (Halide::sin(v * 6.2832f * 4.0f) + 1.0f) * 0.5f;
        Halide::Expr cumul_h = u * u * wave;

        // --- Bottom-left: vertical cumulative sweep ---
        // Brightness accumulates vertically: v^2 curve gives non-linear ramp
        // modulated by horizontal stripes for visual distinction
        Halide::Expr stripe = (Halide::sin(u * 6.2832f * 6.0f) + 1.0f) * 0.5f;
        Halide::Expr cumul_v = v * v * stripe;

        // --- Bottom-right: first derivative edge pattern ---
        // High contrast stripes simulating edges found by a derivative operation.
        // Uses abs(sin) to create sharp stripe patterns, like detecting transitions
        // in a cumulative signal.
        Halide::Expr freq_h = Halide::abs(Halide::sin(u * 6.2832f * 8.0f));
        Halide::Expr freq_v = Halide::abs(Halide::sin(v * 6.2832f * 8.0f));
        Halide::Expr deriv_val = Halide::max(freq_h, freq_v);

        // Compose quadrants
        Halide::Expr pixel = Halide::select(
            qy == 0 && qx == 0, grad_val,
            qy == 0 && qx == 1, cumul_h,
            qy == 1 && qx == 0, cumul_v,
            deriv_val
        );

        // Grid lines
        Halide::Expr on_grid = (ox == half) || (oy == half);
        pixel = Halide::select(on_grid, 0.4f, pixel);

        output(ox, oy) = Halide::cast<uint8_t>(Halide::clamp(pixel * 255.0f, 0.0f, 255.0f));

        std::cout << "Rendering..." << std::endl;
        Halide::Runtime::Buffer<uint8_t> result(size, size);
        output.realize(result);

        const char* output_path = "out/23_cumulative.png";
        std::cout << "Saving to " << output_path << "..." << std::endl;

        if (save_png(result, output_path)) {
            std::cout << "Success!" << std::endl;
            std::cout << std::endl;
            std::cout << "Quadrant guide:" << std::endl;
            std::cout << "  Top-left:     linear gradient (raw image)" << std::endl;
            std::cout << "  Top-right:    cumulative brightness sweep (darken left to right)" << std::endl;
            std::cout << "  Bottom-left:  vertical cumulative sweep" << std::endl;
            std::cout << "  Bottom-right: first derivative edge pattern (high contrast stripes)" << std::endl;
        } else {
            std::cerr << "Error: Failed to save PNG" << std::endl;
            return 1;
        }

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
