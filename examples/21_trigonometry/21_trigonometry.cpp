/// @file 21_trigonometry.cpp
/// @brief Example 21: Trigonometric functions
///
/// Demonstrates:
///   - sin wave horizontal sweep
///   - cos wave vertical sweep
///   - atan2 radial angle field
///   - hypot distance field
///
/// Output: out/21_trigonometry.png

#include "numhalide_all.h"
#include "stbi_png.h"
#include <iostream>

using namespace numhalide;

int main(int argc, char **argv) {
    try {
        const int half = 256;
        const int size = half * 2;  // 512x512

        std::cout << "Trigonometry demonstration" << std::endl;
        std::cout << "Output size: " << size << "x" << size << std::endl;
        std::cout << std::endl;

        constexpr float pi = 3.14159265358979323846f;

        Halide::Var ox("ox"), oy("oy");
        Halide::Func output("output");

        Halide::Expr qx = ox / half;  // 0=left, 1=right
        Halide::Expr qy = oy / half;  // 0=top,  1=bottom
        Halide::Expr lx = ox % half;
        Halide::Expr ly = oy % half;

        // Normalized coords [0,1)
        Halide::Expr u = Halide::cast<float>(lx) / half;
        Halide::Expr v = Halide::cast<float>(ly) / half;

        // --- Top-left: sin wave horizontal sweep ---
        // sin(u * 4pi) mapped to [0,1]
        Halide::Expr sin_val = (Halide::sin(u * 4.0f * pi) + 1.0f) * 0.5f;

        // --- Top-right: cos wave vertical sweep ---
        Halide::Expr cos_val = (Halide::cos(v * 4.0f * pi) + 1.0f) * 0.5f;

        // --- Bottom-left: atan2 radial angle field ---
        Halide::Expr cx = Halide::cast<float>(lx) - half / 2.0f;
        Halide::Expr cy = Halide::cast<float>(ly) - half / 2.0f;
        // atan2 in [-pi, pi], map to [0,1]
        Halide::Expr angle = Halide::atan2(cy, cx);
        Halide::Expr atan2_val = (angle + pi) / (2.0f * pi);

        // --- Bottom-right: hypot distance field ---
        Halide::Expr dist = Halide::sqrt(cx * cx + cy * cy);
        // Normalize by half-diagonal
        Halide::Expr max_dist = Halide::cast<float>(half) / 2.0f * 1.414f;
        Halide::Expr hypot_val = Halide::clamp(dist / max_dist, 0.0f, 1.0f);

        // Compose quadrants
        Halide::Expr pixel = Halide::select(
            qy == 0 && qx == 0, sin_val,
            qy == 0 && qx == 1, cos_val,
            qy == 1 && qx == 0, atan2_val,
            hypot_val
        );

        // Grid lines
        Halide::Expr on_grid = (ox == half) || (oy == half);
        pixel = Halide::select(on_grid, 0.4f, pixel);

        output(ox, oy) = Halide::cast<uint8_t>(Halide::clamp(pixel * 255.0f, 0.0f, 255.0f));

        std::cout << "Rendering..." << std::endl;
        Halide::Runtime::Buffer<uint8_t> result(size, size);
        output.realize(result);

        const char* output_path = "out/21_trigonometry.png";
        std::cout << "Saving to " << output_path << "..." << std::endl;

        if (save_png(result, output_path)) {
            std::cout << "Success!" << std::endl;
            std::cout << std::endl;
            std::cout << "Quadrant guide:" << std::endl;
            std::cout << "  Top-left:     sin wave horizontal sweep" << std::endl;
            std::cout << "  Top-right:    cos wave vertical sweep" << std::endl;
            std::cout << "  Bottom-left:  atan2 radial angle field" << std::endl;
            std::cout << "  Bottom-right: hypot distance field" << std::endl;
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
