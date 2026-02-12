/// @file 18_fft.cpp
/// @brief Example 18: Fast Fourier Transform
///
/// Demonstrates:
///   - 1D FFT and inverse FFT
///   - 2D FFT for image processing
///   - Power spectrum visualization
///   - FFT shift
///
/// Output: out/18_fft.png

#include "numhalide_all.h"
#include "stbi_png.h"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace numhalide;

int main(int argc, char **argv) {
    try {
        const int size = 64;  // Power of 2 for FFT
        const int output_size = size * 2;

        std::cout << "FFT demonstration" << std::endl;
        std::cout << "Transform size: " << size << "x" << size << std::endl;
        std::cout << "Output size: " << output_size << "x" << output_size << std::endl;
        std::cout << std::endl;

        // Create a test image with some features
        Halide::Func input("input");
        Halide::Var x("x"), y("y");

        // Create a pattern: circle + horizontal stripes
        float cx = size / 2.0f;
        float cy = size / 2.0f;
        Halide::Expr dx = Halide::cast<float>(x) - cx;
        Halide::Expr dy = Halide::cast<float>(y) - cy;
        Halide::Expr dist = Halide::sqrt(dx * dx + dy * dy);

        // Circle
        Halide::Expr circle = dist < size / 4.0f;

        // Horizontal stripes (frequency ~4 cycles)
        Halide::Expr stripes = Halide::sin(Halide::cast<float>(y) * 4.0f * 3.14159f / size) > 0.0f;

        // Combine: circle with stripes
        Halide::Expr pattern = Halide::select(circle, 1.0f, 0.0f) +
                               Halide::select(stripes, 0.3f, 0.0f);
        pattern = Halide::clamp(pattern, 0.0f, 1.0f);

        // Convert to complex
        Halide::Func complex_input("complex_input");
        complex_input(x, y) = Halide::Tuple(pattern, 0.0f);

        std::cout << "Computing 2D FFT..." << std::endl;

        // Compute 2D FFT
        auto fft_result = fft2d(complex_input, size, size, "fft2d");

        // Compute power spectrum
        auto power = power_spectrum_2d(fft_result, "power");

        // Apply fftshift to center the spectrum
        auto shifted_fft = fftshift_2d(fft_result, size, size, "shifted");
        auto shifted_power = power_spectrum_2d(shifted_fft, "shifted_power");

        // Compute inverse FFT to verify roundtrip
        auto reconstructed = ifft2d_normalized(fft_result, size, size, "ifft2d");

        std::cout << "Building output visualization..." << std::endl;

        // Create output: 4 quadrants
        // Top-left: Original image
        // Top-right: Reconstructed (IFFT of FFT)
        // Bottom-left: Power spectrum (unshifted)
        // Bottom-right: Power spectrum (shifted, DC in center)

        Halide::Func output("output");
        Halide::Var ox("ox"), oy("oy");

        Halide::Expr qx = ox / size;
        Halide::Expr qy = oy / size;
        Halide::Expr lx = ox % size;
        Halide::Expr ly = oy % size;

        // Get values for each quadrant
        Halide::Expr orig_val = pattern;
        Halide::Expr recon_val = reconstructed(lx, ly)[0];  // Real part only

        // Log scale for power spectrum visualization
        Halide::Expr power_val = power(lx, ly);
        Halide::Expr shifted_power_val = shifted_power(lx, ly);

        // Normalize power spectrum with log scale
        // Add small epsilon to avoid log(0)
        Halide::Expr log_power = Halide::log(power_val + 1.0f) / Halide::log(static_cast<float>(size * size) + 1.0f);
        Halide::Expr log_shifted = Halide::log(shifted_power_val + 1.0f) / Halide::log(static_cast<float>(size * size) + 1.0f);

        // Define original pattern inline for top-left
        Halide::Expr dx_tl = Halide::cast<float>(lx) - cx;
        Halide::Expr dy_tl = Halide::cast<float>(ly) - cy;
        Halide::Expr dist_tl = Halide::sqrt(dx_tl * dx_tl + dy_tl * dy_tl);
        Halide::Expr circle_tl = dist_tl < size / 4.0f;
        Halide::Expr stripes_tl = Halide::sin(Halide::cast<float>(ly) * 4.0f * 3.14159f / size) > 0.0f;
        Halide::Expr pattern_tl = Halide::select(circle_tl, 1.0f, 0.0f) +
                                   Halide::select(stripes_tl, 0.3f, 0.0f);
        pattern_tl = Halide::clamp(pattern_tl, 0.0f, 1.0f);

        Halide::Expr pixel = Halide::select(
            qy == 0 && qx == 0, pattern_tl,
            Halide::select(
                qy == 0 && qx == 1, Halide::clamp(recon_val, 0.0f, 1.0f),
                Halide::select(
                    qy == 1 && qx == 0, Halide::clamp(log_power, 0.0f, 1.0f),
                    Halide::clamp(log_shifted, 0.0f, 1.0f)
                )
            )
        );

        // Add grid lines
        Halide::Expr on_grid = (ox == size) || (oy == size);
        pixel = Halide::select(on_grid, 0.4f, pixel);

        output(ox, oy) = Halide::cast<uint8_t>(Halide::clamp(pixel * 255.0f, 0.0f, 255.0f));

        // Demonstrate 1D FFT with simple example
        std::cout << std::endl << "1D FFT example:" << std::endl;
        Halide::Func signal_1d("signal_1d");
        // Simple sinusoid: cos(2*pi*k*n/N) for k=1
        float pi = 3.14159265f;
        signal_1d(x) = Halide::Tuple(
            Halide::cos(2.0f * pi * Halide::cast<float>(x) / 8.0f),
            0.0f
        );

        auto fft_1d = fft(signal_1d, 8, "fft1d");

        // Extract real and imaginary parts separately
        Halide::Func fft_re("fft_re"), fft_im("fft_im");
        fft_re(x) = fft_1d(x)[0];
        fft_im(x) = fft_1d(x)[1];

        Halide::Runtime::Buffer<float> re_out(8), im_out(8);
        fft_re.realize(re_out);
        fft_im.realize(im_out);

        std::cout << "  Input: cos(2*pi*n/8) for n=0..7" << std::endl;
        std::cout << "  FFT magnitude: [";
        for (int i = 0; i < 8; i++) {
            float mag = std::sqrt(re_out(i) * re_out(i) + im_out(i) * im_out(i));
            std::cout << std::fixed << std::setprecision(2) << mag;
            if (i < 7) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << "  (Peak at bin 1 and 7 = frequency +-1)" << std::endl;
        std::cout << std::endl;

        // Realize 2D visualization
        std::cout << "Rendering output..." << std::endl;
        Halide::Runtime::Buffer<uint8_t> result(output_size, output_size);
        output.realize(result);

        // Save
        const char* output_path = "out/18_fft.png";
        std::cout << "Saving to " << output_path << "..." << std::endl;

        if (save_png(result, output_path)) {
            std::cout << "Success! FFT visualization saved." << std::endl;
            std::cout << std::endl;
            std::cout << "Quadrant guide:" << std::endl;
            std::cout << "  Top-left:     Original image (circle + stripes)" << std::endl;
            std::cout << "  Top-right:    Reconstructed via IFFT(FFT(x))" << std::endl;
            std::cout << "  Bottom-left:  Power spectrum (unshifted)" << std::endl;
            std::cout << "  Bottom-right: Power spectrum (shifted, DC centered)" << std::endl;
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
