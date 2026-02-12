/// @file 39_spectral.cpp
/// @brief Example 39: Spectral / phase correlation visualization
///
/// Demonstrates:
///   - circles pattern (spatial domain)
///   - same circles shifted (translated version)
///   - cross power spectrum magnitude look
///   - phase correlation peak pattern
///
/// Output: out/39_spectral.png

#include "numhalide_all.h"
#include "stbi_png.h"
#include <iostream>

using namespace numhalide;

int main(int argc, char **argv) {
    try {
        const int half = 256;
        const int size = half * 2;  // 512x512

        std::cout << "Spectral / phase correlation demonstration" << std::endl;
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

        // Centered coords [-1,1)
        Halide::Expr cu = u * 2.0f - 1.0f;
        Halide::Expr cv = v * 2.0f - 1.0f;

        // --- Top-left: circles pattern ---
        // Multiple circles at different positions
        // Circle 1: center (-0.3, -0.2), radius 0.25
        Halide::Expr dx1 = cu - (-0.3f);
        Halide::Expr dy1 = cv - (-0.2f);
        Halide::Expr r1 = Halide::sqrt(dx1 * dx1 + dy1 * dy1);
        Halide::Expr c1 = Halide::select(r1 < 0.25f, 1.0f, 0.0f);

        // Circle 2: center (0.3, 0.3), radius 0.2
        Halide::Expr dx2 = cu - 0.3f;
        Halide::Expr dy2 = cv - 0.3f;
        Halide::Expr r2 = Halide::sqrt(dx2 * dx2 + dy2 * dy2);
        Halide::Expr c2 = Halide::select(r2 < 0.2f, 0.8f, 0.0f);

        // Circle 3: center (0.0, -0.4), radius 0.15
        Halide::Expr dx3 = cu - 0.0f;
        Halide::Expr dy3 = cv - (-0.4f);
        Halide::Expr r3 = Halide::sqrt(dx3 * dx3 + dy3 * dy3);
        Halide::Expr c3 = Halide::select(r3 < 0.15f, 0.6f, 0.0f);

        Halide::Expr circles_val = Halide::clamp(c1 + c2 + c3, 0.0f, 1.0f);

        // --- Top-right: same circles shifted (translated by +0.2, +0.15) ---
        Halide::Expr shift_x = 0.2f;
        Halide::Expr shift_y = 0.15f;
        Halide::Expr su = cu - shift_x;
        Halide::Expr sv = cv - shift_y;

        Halide::Expr sdx1 = su - (-0.3f);
        Halide::Expr sdy1 = sv - (-0.2f);
        Halide::Expr sr1 = Halide::sqrt(sdx1 * sdx1 + sdy1 * sdy1);
        Halide::Expr sc1 = Halide::select(sr1 < 0.25f, 1.0f, 0.0f);

        Halide::Expr sdx2 = su - 0.3f;
        Halide::Expr sdy2 = sv - 0.3f;
        Halide::Expr sr2 = Halide::sqrt(sdx2 * sdx2 + sdy2 * sdy2);
        Halide::Expr sc2 = Halide::select(sr2 < 0.2f, 0.8f, 0.0f);

        Halide::Expr sdx3 = su - 0.0f;
        Halide::Expr sdy3 = sv - (-0.4f);
        Halide::Expr sr3 = Halide::sqrt(sdx3 * sdx3 + sdy3 * sdy3);
        Halide::Expr sc3 = Halide::select(sr3 < 0.15f, 0.6f, 0.0f);

        Halide::Expr shifted_val = Halide::clamp(sc1 + sc2 + sc3, 0.0f, 1.0f);

        // --- Bottom-left: cross power spectrum magnitude look ---
        // The cross power spectrum of two shifted images has uniform magnitude
        // and linear phase. Visualize as a frequency-domain pattern.
        // Simulate the spectral magnitude: concentric rings with radial falloff
        Halide::Expr freq_r = Halide::sqrt(cu * cu + cv * cv);
        // Spectral magnitude: multiple frequency bands
        Halide::Expr freq_rings = (Halide::sin(freq_r * 30.0f) + 1.0f) * 0.5f;
        // Central DC component is bright, falloff with frequency
        Halide::Expr spectral_envelope = Halide::exp(-freq_r * 2.0f);
        Halide::Expr cross_power_val = Halide::clamp(freq_rings * spectral_envelope + 0.05f, 0.0f, 1.0f);

        // --- Bottom-right: phase correlation peak pattern ---
        // Phase correlation produces a sharp peak at the shift offset
        // The peak location encodes the translation between the two images
        // Simulate: sharp bright spot at the shift location, dark elsewhere
        Halide::Expr peak_x = cu - shift_x;
        Halide::Expr peak_y = cv - shift_y;
        Halide::Expr peak_r = Halide::sqrt(peak_x * peak_x + peak_y * peak_y);
        // Sharp Gaussian peak
        Halide::Expr peak = Halide::exp(-peak_r * peak_r / (2.0f * 0.02f * 0.02f));
        // Add very low-level noise-like pattern (sinusoidal hash)
        Halide::Expr noise_like = (Halide::sin(cu * 97.0f) * Halide::sin(cv * 131.0f) + 1.0f) * 0.02f;
        Halide::Expr phase_corr_val = Halide::clamp(peak + noise_like, 0.0f, 1.0f);

        // Compose quadrants
        Halide::Expr pixel = Halide::select(
            qy == 0 && qx == 0, circles_val,
            qy == 0 && qx == 1, shifted_val,
            qy == 1 && qx == 0, cross_power_val,
            phase_corr_val
        );

        // Grid lines
        Halide::Expr on_grid = (ox == half) || (oy == half);
        pixel = Halide::select(on_grid, 0.4f, pixel);

        output(ox, oy) = Halide::cast<uint8_t>(Halide::clamp(pixel * 255.0f, 0.0f, 255.0f));

        std::cout << "Rendering..." << std::endl;
        Halide::Runtime::Buffer<uint8_t> result(size, size);
        output.realize(result);

        const char* output_path = "out/39_spectral.png";
        std::cout << "Saving to " << output_path << "..." << std::endl;

        if (save_png(result, output_path)) {
            std::cout << "Success!" << std::endl;
            std::cout << std::endl;
            std::cout << "Quadrant guide:" << std::endl;
            std::cout << "  Top-left:     circles pattern (original)" << std::endl;
            std::cout << "  Top-right:    circles shifted (+0.2, +0.15)" << std::endl;
            std::cout << "  Bottom-left:  cross power spectrum magnitude" << std::endl;
            std::cout << "  Bottom-right: phase correlation peak" << std::endl;
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
