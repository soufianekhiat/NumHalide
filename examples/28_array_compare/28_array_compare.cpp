/// @file 28_array_compare.cpp
/// @brief Example 28: Array comparison operations
///
/// Demonstrates:
///   - smooth gradient A
///   - gradient B with perturbation
///   - per-pixel equality map (white where identical)
///   - per-pixel tolerance closeness (more white)
///
/// Output: out/28_array_compare.png

#include "numhalide_all.h"
#include "stbi_png.h"
#include <iostream>

using namespace numhalide;

int main(int argc, char **argv) {
    try {
        const int half = 256;
        const int size = half * 2;  // 512x512

        std::cout << "Array comparison demonstration" << std::endl;
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

        // --- Signal A: smooth diagonal gradient ---
        Halide::Expr signal_a = (u + v) * 0.5f;

        // --- Signal B: gradient with perturbation ---
        // Same base gradient plus a small sine-based perturbation and
        // quantization to make some pixels exactly equal, some slightly off
        Halide::Expr perturb = Halide::sin(Halide::cast<float>(lx) * 0.8f)
                             * Halide::sin(Halide::cast<float>(ly) * 0.6f)
                             * 0.04f;
        Halide::Expr signal_b = Halide::clamp(signal_a + perturb, 0.0f, 1.0f);

        // Quantize both signals to 8-bit to simulate stored pixel values.
        // This way some pixels will be exactly equal, others will differ by 1/255.
        Halide::Expr qa = Halide::floor(signal_a * 255.0f) / 255.0f;
        Halide::Expr qb = Halide::floor(signal_b * 255.0f) / 255.0f;

        // --- Top-left: gradient A ---
        Halide::Expr val_a = qa;

        // --- Top-right: gradient B with perturbation ---
        Halide::Expr val_b = qb;

        // --- Bottom-left: per-pixel exact equality map ---
        // White (1.0) where A == B exactly, black (0.0) where they differ
        Halide::Expr exact_diff = Halide::abs(qa - qb);
        Halide::Expr exact_eq = Halide::select(exact_diff < 0.001f, 1.0f, 0.0f);

        // --- Bottom-right: per-pixel tolerance closeness ---
        // White where |A - B| < tolerance (0.02), smooth falloff beyond
        // This produces more white than exact equality since it allows small differences
        Halide::Expr tol = 0.02f;
        Halide::Expr close_val = Halide::select(
            exact_diff < tol, 1.0f,
            Halide::clamp(1.0f - (exact_diff - tol) * 20.0f, 0.0f, 1.0f)
        );

        // Compose quadrants
        Halide::Expr pixel = Halide::select(
            qy == 0 && qx == 0, val_a,
            qy == 0 && qx == 1, val_b,
            qy == 1 && qx == 0, exact_eq,
            close_val
        );

        // Grid lines
        Halide::Expr on_grid = (ox == half) || (oy == half);
        pixel = Halide::select(on_grid, 0.4f, pixel);

        output(ox, oy) = Halide::cast<uint8_t>(Halide::clamp(pixel * 255.0f, 0.0f, 255.0f));

        std::cout << "Rendering..." << std::endl;
        Halide::Runtime::Buffer<uint8_t> result(size, size);
        output.realize(result);

        const char* output_path = "out/28_array_compare.png";
        std::cout << "Saving to " << output_path << "..." << std::endl;

        if (save_png(result, output_path)) {
            std::cout << "Success!" << std::endl;
            std::cout << std::endl;
            std::cout << "Quadrant guide:" << std::endl;
            std::cout << "  Top-left:     smooth gradient A" << std::endl;
            std::cout << "  Top-right:    gradient B with perturbation" << std::endl;
            std::cout << "  Bottom-left:  per-pixel equality map (white where identical)" << std::endl;
            std::cout << "  Bottom-right: per-pixel tolerance closeness (more white)" << std::endl;
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
