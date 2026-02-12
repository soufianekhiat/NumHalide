/// @file 24_splitting.cpp
/// @brief Example 24: Array splitting operations
///
/// Demonstrates:
///   - smooth gradient
///   - horizontal bands with alternating brightness
///   - vertical bands with alternating brightness
///   - checkerboard from split quadrants with alternating inversion
///
/// Output: out/24_splitting.png

#include "numhalide_all.h"
#include "stbi_png.h"
#include <iostream>

using namespace numhalide;

int main(int argc, char **argv) {
    try {
        const int half = 256;
        const int size = half * 2;  // 512x512

        std::cout << "Splitting operations demonstration" << std::endl;
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

        // --- Top-left: smooth gradient ---
        // A simple smooth gradient that would be the "original" array before splitting
        Halide::Expr smooth = (u + v) * 0.5f;

        // --- Top-right: horizontal bands with alternating brightness ---
        // Simulates splitting into horizontal strips with alternating inversion.
        // Every other band is inverted to visualize the split chunks.
        Halide::Expr num_bands_h = 8;
        Halide::Expr band_h = ly / (half / num_bands_h);
        Halide::Expr even_h = (band_h % 2) == 0;
        Halide::Expr base_h = (u + v) * 0.5f;
        Halide::Expr horiz_bands = Halide::select(even_h, base_h, 1.0f - base_h);

        // --- Bottom-left: vertical bands with alternating brightness ---
        // Simulates splitting into vertical strips with alternating brightness.
        Halide::Expr num_bands_v = 8;
        Halide::Expr band_v = lx / (half / num_bands_v);
        Halide::Expr even_v = (band_v % 2) == 0;
        Halide::Expr base_v = (u + v) * 0.5f;
        Halide::Expr vert_bands = Halide::select(even_v, base_v, 1.0f - base_v);

        // --- Bottom-right: checkerboard from split quadrants ---
        // Combines horizontal and vertical splitting to create a checkerboard
        // pattern where each cell alternates between normal and inverted gradient.
        Halide::Expr cell_x = lx / (half / num_bands_v);
        Halide::Expr cell_y = ly / (half / num_bands_h);
        Halide::Expr checker = ((cell_x + cell_y) % 2) == 0;
        Halide::Expr base_c = (u + v) * 0.5f;
        Halide::Expr checker_val = Halide::select(checker, base_c, 1.0f - base_c);

        // Compose quadrants
        Halide::Expr pixel = Halide::select(
            qy == 0 && qx == 0, smooth,
            qy == 0 && qx == 1, horiz_bands,
            qy == 1 && qx == 0, vert_bands,
            checker_val
        );

        // Grid lines
        Halide::Expr on_grid = (ox == half) || (oy == half);
        pixel = Halide::select(on_grid, 0.4f, pixel);

        output(ox, oy) = Halide::cast<uint8_t>(Halide::clamp(pixel * 255.0f, 0.0f, 255.0f));

        std::cout << "Rendering..." << std::endl;
        Halide::Runtime::Buffer<uint8_t> result(size, size);
        output.realize(result);

        const char* output_path = "out/24_splitting.png";
        std::cout << "Saving to " << output_path << "..." << std::endl;

        if (save_png(result, output_path)) {
            std::cout << "Success!" << std::endl;
            std::cout << std::endl;
            std::cout << "Quadrant guide:" << std::endl;
            std::cout << "  Top-left:     smooth gradient" << std::endl;
            std::cout << "  Top-right:    horizontal bands with alternating brightness" << std::endl;
            std::cout << "  Bottom-left:  vertical bands with alternating brightness" << std::endl;
            std::cout << "  Bottom-right: checkerboard from split quadrants" << std::endl;
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
