/// @file 37_stencil.cpp
/// @brief Example 37: Stencil / heat diffusion visualization
///
/// Demonstrates:
///   - initial heat distribution (hot circle in center)
///   - heat spreading (blurred, simulating diffusion steps)
///   - nearly uniform (more blur, further diffusion)
///   - convergence visualization (residual amplified)
///
/// Output: out/37_stencil.png

#include "numhalide_all.h"
#include "stbi_png.h"
#include <iostream>

using namespace numhalide;

int main(int argc, char **argv) {
    try {
        const int half = 256;
        const int size = half * 2;  // 512x512

        std::cout << "Stencil / heat diffusion demonstration" << std::endl;
        std::cout << "Output size: " << size << "x" << size << std::endl;
        std::cout << std::endl;

        constexpr float pi = 3.14159265358979323846f;

        Halide::Var ox("ox"), oy("oy");
        Halide::Func output("output");

        Halide::Expr qx = ox / half;
        Halide::Expr qy = oy / half;
        Halide::Expr lx = ox % half;
        Halide::Expr ly = oy % half;

        // Centered coords [-1,1)
        Halide::Expr cu = (Halide::cast<float>(lx) - half / 2.0f) / (half / 2.0f);
        Halide::Expr cv = (Halide::cast<float>(ly) - half / 2.0f) / (half / 2.0f);
        Halide::Expr r = Halide::sqrt(cu * cu + cv * cv);

        // --- Top-left: initial heat distribution (hot circle in center) ---
        // Sharp circle: 1.0 inside radius 0.3, 0.0 outside
        Halide::Expr init_heat = Halide::select(r < 0.3f, 1.0f, 0.0f);

        // --- Top-right: heat spreading (Gaussian-like blur of initial) ---
        // Simulate diffusion by using a wider Gaussian envelope
        // After some diffusion steps, sharp circle becomes Gaussian-ish
        Halide::Expr sigma1 = 0.35f;
        Halide::Expr spread1 = Halide::exp(-(r * r) / (2.0f * sigma1 * sigma1));
        // Scale so peak roughly matches initial amplitude
        Halide::Expr heat_spread = spread1 * 0.9f;

        // --- Bottom-left: nearly uniform (more blur, further diffusion) ---
        // Much wider Gaussian - nearly flat
        Halide::Expr sigma2 = 0.8f;
        Halide::Expr spread2 = Halide::exp(-(r * r) / (2.0f * sigma2 * sigma2));
        // Scale to show it's mostly uniform with slight variation
        Halide::Expr heat_uniform = spread2 * 0.5f + 0.2f;

        // --- Bottom-right: convergence visualization (residual amplified) ---
        // The residual is the difference between current state and equilibrium
        // In steady state the heat would be uniform; residual = current - mean
        // Amplify small differences to show convergence pattern
        Halide::Expr residual = (spread2 - 0.4f) * 5.0f;
        // Map to visible range: center at 0.5 gray
        Halide::Expr convergence_val = Halide::clamp(residual + 0.5f, 0.0f, 1.0f);
        // Add concentric ring indicators to show iso-residual contours
        Halide::Expr ring_phase = r * 8.0f;
        Halide::Expr ring_frac = ring_phase - Halide::floor(ring_phase);
        Halide::Expr on_ring = ring_frac < 0.08f;
        convergence_val = Halide::select(on_ring, Halide::clamp(convergence_val + 0.3f, 0.0f, 1.0f), convergence_val);

        // Compose quadrants
        Halide::Expr pixel = Halide::select(
            qy == 0 && qx == 0, init_heat,
            qy == 0 && qx == 1, heat_spread,
            qy == 1 && qx == 0, heat_uniform,
            convergence_val
        );

        // Grid lines
        Halide::Expr on_grid = (ox == half) || (oy == half);
        pixel = Halide::select(on_grid, 0.4f, pixel);

        output(ox, oy) = Halide::cast<uint8_t>(Halide::clamp(pixel * 255.0f, 0.0f, 255.0f));

        std::cout << "Rendering..." << std::endl;
        Halide::Runtime::Buffer<uint8_t> result(size, size);
        output.realize(result);

        const char* output_path = "out/37_stencil.png";
        std::cout << "Saving to " << output_path << "..." << std::endl;

        if (save_png(result, output_path)) {
            std::cout << "Success!" << std::endl;
            std::cout << std::endl;
            std::cout << "Quadrant guide:" << std::endl;
            std::cout << "  Top-left:     initial heat distribution (hot circle)" << std::endl;
            std::cout << "  Top-right:    heat spreading (Gaussian diffusion)" << std::endl;
            std::cout << "  Bottom-left:  nearly uniform (further diffusion)" << std::endl;
            std::cout << "  Bottom-right: convergence residual (amplified)" << std::endl;
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
