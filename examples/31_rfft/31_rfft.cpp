/// @file 31_rfft.cpp
/// @brief Example 31: Real FFT visualization
///
/// Demonstrates:
///   - real-valued sine pattern
///   - FFT-like power spectrum pattern (log scale)
///   - band-pass filtered look (ring mask)
///   - smoothed reconstruction
///
/// Output: out/31_rfft.png

#include "numhalide_all.h"
#include "stbi_png.h"
#include <iostream>

using namespace numhalide;

int main(int argc, char **argv) {
    try {
        const int half = 256;
        const int size = half * 2;  // 512x512

        std::cout << "Real FFT visualization demonstration" << std::endl;
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
        Halide::Expr cx = (Halide::cast<float>(lx) - half / 2.0f) / (half / 2.0f);
        Halide::Expr cy = (Halide::cast<float>(ly) - half / 2.0f) / (half / 2.0f);
        Halide::Expr r = Halide::sqrt(cx * cx + cy * cy);

        // --- Top-left: real-valued sine pattern ---
        // Superposition of sine waves at different frequencies and angles
        Halide::Expr freq1 = Halide::sin(u * 16.0f * pi);
        Halide::Expr freq2 = Halide::sin(v * 8.0f * pi);
        Halide::Expr freq3 = Halide::sin((u + v) * 12.0f * pi);
        Halide::Expr sine_val = (freq1 + freq2 + freq3 + 3.0f) / 6.0f;

        // --- Top-right: FFT-like power spectrum (log scale) ---
        // Simulate a centered power spectrum with bright dots at frequency locations
        // Use distance from center and angular patterns
        Halide::Expr spectrum_r = r;
        // Create spectral peaks at specific radii corresponding to our frequencies
        Halide::Expr peak1 = Halide::pow(2.0f, -80.0f * (spectrum_r - 0.25f) * (spectrum_r - 0.25f));
        Halide::Expr peak2 = Halide::pow(2.0f, -80.0f * (spectrum_r - 0.5f) * (spectrum_r - 0.5f));
        Halide::Expr peak3 = Halide::pow(2.0f, -80.0f * (spectrum_r - 0.375f) * (spectrum_r - 0.375f));
        // Angular modulation to create discrete peaks rather than rings
        Halide::Expr angle = Halide::atan2(cy, cx);
        Halide::Expr ang_mod1 = Halide::pow(Halide::cos(angle), 2.0f);           // horizontal
        Halide::Expr ang_mod2 = Halide::pow(Halide::cos(angle - pi / 2.0f), 2.0f); // vertical
        Halide::Expr ang_mod3 = Halide::pow(Halide::cos(angle - pi / 4.0f), 2.0f); // diagonal
        Halide::Expr spectrum = peak1 * ang_mod1 + peak2 * ang_mod2 + peak3 * ang_mod3;
        // DC component at center
        Halide::Expr dc = Halide::pow(2.0f, -200.0f * r * r);
        // Log scale
        Halide::Expr spectrum_val = Halide::clamp(Halide::log(1.0f + 20.0f * (spectrum + dc)) / 4.0f, 0.0f, 1.0f);

        // --- Bottom-left: band-pass filtered (ring mask) ---
        // Apply a ring-shaped mask to simulate band-pass filtering
        // Only keep middle frequencies - creates edge-enhanced look
        Halide::Expr ring_inner = 0.15f;
        Halide::Expr ring_outer = 0.45f;
        Halide::Expr ring_mask = Halide::select(
            r >= ring_inner && r <= ring_outer, 1.0f, 0.0f
        );
        // Simulate filtered result: the medium-frequency sine components
        Halide::Expr filtered = Halide::sin((u + v) * 12.0f * pi) * 0.5f + 0.5f;
        // Add edge-like sharpening from the band-pass
        Halide::Expr edge_enhance = Halide::abs(Halide::sin(u * 16.0f * pi) - Halide::sin(v * 8.0f * pi));
        Halide::Expr bandpass_val = Halide::clamp(filtered * 0.5f + edge_enhance * 0.5f, 0.0f, 1.0f);

        // --- Bottom-right: smoothed reconstruction ---
        // Low-pass filtered version: only low frequencies, creating a blurry image
        // Use Gaussian-weighted combination of low-frequency sinusoids
        Halide::Expr smooth = Halide::sin(u * 4.0f * pi) * 0.3f
                            + Halide::sin(v * 2.0f * pi) * 0.3f
                            + Halide::sin((u + v) * 3.0f * pi) * 0.2f
                            + 0.5f;
        // Gaussian envelope for extra smoothness
        Halide::Expr gauss_env = Halide::pow(2.0f, -2.0f * r * r);
        Halide::Expr smooth_val = Halide::clamp(smooth * (0.6f + 0.4f * gauss_env), 0.0f, 1.0f);

        // Compose quadrants
        Halide::Expr pixel = Halide::select(
            qy == 0 && qx == 0, sine_val,
            qy == 0 && qx == 1, spectrum_val,
            qy == 1 && qx == 0, bandpass_val,
            smooth_val
        );

        // Grid lines
        Halide::Expr on_grid = (ox == half) || (oy == half);
        pixel = Halide::select(on_grid, 0.4f, pixel);

        output(ox, oy) = Halide::cast<uint8_t>(Halide::clamp(pixel * 255.0f, 0.0f, 255.0f));

        std::cout << "Rendering..." << std::endl;
        Halide::Runtime::Buffer<uint8_t> result(size, size);
        output.realize(result);

        const char* output_path = "out/31_rfft.png";
        std::cout << "Saving to " << output_path << "..." << std::endl;

        if (save_png(result, output_path)) {
            std::cout << "Success!" << std::endl;
            std::cout << std::endl;
            std::cout << "Quadrant guide:" << std::endl;
            std::cout << "  Top-left:     real-valued sine pattern" << std::endl;
            std::cout << "  Top-right:    FFT-like power spectrum (log scale)" << std::endl;
            std::cout << "  Bottom-left:  band-pass filtered look (ring mask)" << std::endl;
            std::cout << "  Bottom-right: smoothed reconstruction" << std::endl;
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
