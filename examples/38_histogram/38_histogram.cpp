/// @file 38_histogram.cpp
/// @brief Example 38: Histogram and tone mapping visualization
///
/// Demonstrates:
///   - low-contrast image (narrow intensity range)
///   - histogram-equalized look (full range stretch)
///   - gamma=0.5 brightened image
///   - gamma=2.0 darkened image
///
/// Output: out/38_histogram.png

#include "numhalide_all.h"
#include "stbi_png.h"
#include <iostream>

using namespace numhalide;

int main(int argc, char **argv) {
    try {
        const int half = 256;
        const int size = half * 2;  // 512x512

        std::cout << "Histogram and tone mapping demonstration" << std::endl;
        std::cout << "Output size: " << size << "x" << size << std::endl;
        std::cout << std::endl;

        constexpr float pi = 3.14159265358979323846f;

        Halide::Var ox("ox"), oy("oy");
        Halide::Func output("output");

        Halide::Expr qx = ox / half;
        Halide::Expr qy = oy / half;
        Halide::Expr lx = ox % half;
        Halide::Expr ly = oy % half;

        // Normalized coords [0,1)
        Halide::Expr u = Halide::cast<float>(lx) / half;
        Halide::Expr v = Halide::cast<float>(ly) / half;

        // Centered coords
        Halide::Expr cu = u * 2.0f - 1.0f;
        Halide::Expr cv = v * 2.0f - 1.0f;
        Halide::Expr r = Halide::sqrt(cu * cu + cv * cv);

        // Base scene: a mix of features (circles, gradients, patterns)
        // Radial gradient with some circular features
        Halide::Expr scene_base = Halide::exp(-r * r * 2.0f);
        // Add some concentric rings
        Halide::Expr rings = (Halide::sin(r * 20.0f) + 1.0f) * 0.5f;
        // Add diagonal gradient
        Halide::Expr diag = (cu + cv + 2.0f) / 4.0f;
        // Combine
        Halide::Expr scene = scene_base * 0.4f + rings * 0.3f + diag * 0.3f;
        scene = Halide::clamp(scene, 0.0f, 1.0f);

        // --- Top-left: low-contrast image (narrow range) ---
        // Compress to [0.3, 0.6] range
        Halide::Expr low_contrast = scene * 0.3f + 0.3f;

        // --- Top-right: histogram-equalized look (full range stretch) ---
        // Stretch [0.3, 0.6] to [0, 1] - simulates histogram equalization
        // Equalization: (val - min) / (max - min)
        Halide::Expr equalized = Halide::clamp((low_contrast - 0.3f) / 0.3f, 0.0f, 1.0f);

        // --- Bottom-left: gamma=0.5 brightened ---
        // Apply gamma 0.5 to original scene (brightens midtones)
        Halide::Expr gamma_bright = Halide::pow(Halide::clamp(scene, 0.001f, 1.0f), 0.5f);

        // --- Bottom-right: gamma=2.0 darkened ---
        // Apply gamma 2.0 to original scene (darkens midtones)
        Halide::Expr gamma_dark = Halide::pow(Halide::clamp(scene, 0.001f, 1.0f), 2.0f);

        // Compose quadrants
        Halide::Expr pixel = Halide::select(
            qy == 0 && qx == 0, low_contrast,
            qy == 0 && qx == 1, equalized,
            qy == 1 && qx == 0, gamma_bright,
            gamma_dark
        );

        // Grid lines
        Halide::Expr on_grid = (ox == half) || (oy == half);
        pixel = Halide::select(on_grid, 0.4f, pixel);

        output(ox, oy) = Halide::cast<uint8_t>(Halide::clamp(pixel * 255.0f, 0.0f, 255.0f));

        std::cout << "Rendering..." << std::endl;
        Halide::Runtime::Buffer<uint8_t> result(size, size);
        output.realize(result);

        const char* output_path = "out/38_histogram.png";
        std::cout << "Saving to " << output_path << "..." << std::endl;

        if (save_png(result, output_path)) {
            std::cout << "Success!" << std::endl;
            std::cout << std::endl;
            std::cout << "Quadrant guide:" << std::endl;
            std::cout << "  Top-left:     low-contrast image (narrow range 0.3-0.6)" << std::endl;
            std::cout << "  Top-right:    histogram-equalized (full range stretch)" << std::endl;
            std::cout << "  Bottom-left:  gamma=0.5 brightened" << std::endl;
            std::cout << "  Bottom-right: gamma=2.0 darkened" << std::endl;
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
