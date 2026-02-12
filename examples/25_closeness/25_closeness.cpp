/// @file 25_closeness.cpp
/// @brief Example 25: Closeness and approximate equality
///
/// Demonstrates:
///   - smooth Gaussian blob
///   - Gaussian blob with small noise pattern
///   - closeness map (white where similar, black where different)
///   - difference magnitude amplified 50x
///
/// Output: out/25_closeness.png

#include "numhalide_all.h"
#include "stbi_png.h"
#include <iostream>

using namespace numhalide;

int main(int argc, char **argv) {
    try {
        const int half = 256;
        const int size = half * 2;  // 512x512

        std::cout << "Closeness demonstration" << std::endl;
        std::cout << "Output size: " << size << "x" << size << std::endl;
        std::cout << std::endl;

        Halide::Var ox("ox"), oy("oy");
        Halide::Func output("output");

        Halide::Expr qx = ox / half;  // 0=left, 1=right
        Halide::Expr qy = oy / half;  // 0=top,  1=bottom
        Halide::Expr lx = ox % half;
        Halide::Expr ly = oy % half;

        // Centered coords [-1, 1)
        Halide::Expr cx = (Halide::cast<float>(lx) - half / 2.0f) / (half / 2.0f);
        Halide::Expr cy = (Halide::cast<float>(ly) - half / 2.0f) / (half / 2.0f);
        Halide::Expr r2 = cx * cx + cy * cy;

        // --- Top-left: smooth Gaussian blob ---
        // Clean Gaussian: exp(-3 * r^2)
        Halide::Expr gauss_clean = Halide::pow(2.718281828f, -3.0f * r2);

        // --- Top-right: Gaussian blob with small noise pattern ---
        // Add a high-frequency sine-based perturbation as "noise"
        // Using a hash-like combination of sin functions to simulate noise
        Halide::Expr noise_x = Halide::cast<float>(lx);
        Halide::Expr noise_y = Halide::cast<float>(ly);
        Halide::Expr noise = Halide::sin(noise_x * 127.1f + noise_y * 311.7f);
        noise = noise - Halide::floor(noise);  // fract
        noise = (noise - 0.5f) * 0.15f;  // small perturbation [-0.075, 0.075]
        Halide::Expr gauss_noisy = Halide::clamp(gauss_clean + noise, 0.0f, 1.0f);

        // --- Bottom-left: closeness map ---
        // White where the two signals are close (within tolerance 0.05), black otherwise
        Halide::Expr diff = Halide::abs(gauss_clean - gauss_noisy);
        Halide::Expr tolerance = 0.05f;
        Halide::Expr close_map = Halide::select(diff < tolerance, 1.0f, 0.0f);

        // --- Bottom-right: difference magnitude amplified 50x ---
        // Shows the actual difference between the two signals, magnified for visibility
        Halide::Expr diff_amplified = Halide::clamp(diff * 50.0f, 0.0f, 1.0f);

        // Compose quadrants
        Halide::Expr pixel = Halide::select(
            qy == 0 && qx == 0, gauss_clean,
            qy == 0 && qx == 1, gauss_noisy,
            qy == 1 && qx == 0, close_map,
            diff_amplified
        );

        // Grid lines
        Halide::Expr on_grid = (ox == half) || (oy == half);
        pixel = Halide::select(on_grid, 0.4f, pixel);

        output(ox, oy) = Halide::cast<uint8_t>(Halide::clamp(pixel * 255.0f, 0.0f, 255.0f));

        std::cout << "Rendering..." << std::endl;
        Halide::Runtime::Buffer<uint8_t> result(size, size);
        output.realize(result);

        const char* output_path = "out/25_closeness.png";
        std::cout << "Saving to " << output_path << "..." << std::endl;

        if (save_png(result, output_path)) {
            std::cout << "Success!" << std::endl;
            std::cout << std::endl;
            std::cout << "Quadrant guide:" << std::endl;
            std::cout << "  Top-left:     smooth Gaussian blob" << std::endl;
            std::cout << "  Top-right:    Gaussian blob with small noise pattern" << std::endl;
            std::cout << "  Bottom-left:  closeness map (white=similar, black=different)" << std::endl;
            std::cout << "  Bottom-right: difference magnitude amplified 50x" << std::endl;
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
