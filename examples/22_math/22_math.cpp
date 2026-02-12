/// @file 22_math.cpp
/// @brief Example 22: Extended math functions
///
/// Demonstrates:
///   - sinc 2D radial pattern
///   - exp2 falloff
///   - fmod checkerboard
///   - heaviside circular mask
///
/// Output: out/22_math.png

#include "numhalide_all.h"
#include "stbi_png.h"
#include <iostream>

using namespace numhalide;

int main(int argc, char **argv) {
    try {
        const int half = 256;
        const int size = half * 2;  // 512x512

        std::cout << "Extended math demonstration" << std::endl;
        std::cout << "Output size: " << size << "x" << size << std::endl;
        std::cout << std::endl;

        constexpr float pi = 3.14159265358979323846f;

        Halide::Var ox("ox"), oy("oy");
        Halide::Func output("output");

        Halide::Expr qx = ox / half;
        Halide::Expr qy = oy / half;
        Halide::Expr lx = ox % half;
        Halide::Expr ly = oy % half;

        // Centered coords [-1, 1)
        Halide::Expr u = (Halide::cast<float>(lx) - half / 2.0f) / (half / 2.0f);
        Halide::Expr v = (Halide::cast<float>(ly) - half / 2.0f) / (half / 2.0f);
        Halide::Expr r = Halide::sqrt(u * u + v * v);

        // --- Top-left: sinc 2D radial pattern ---
        // sinc(r * 4) to get visible rings
        Halide::Expr sinc_arg = r * 4.0f;
        Halide::Expr pix_sinc = pi * sinc_arg;
        Halide::Expr sinc_val = Halide::select(
            sinc_arg < 0.001f, 1.0f,
            Halide::sin(pix_sinc) / pix_sinc
        );
        // Map from [-0.2, 1] to [0, 1] for visibility
        sinc_val = Halide::clamp((sinc_val + 0.3f) / 1.3f, 0.0f, 1.0f);

        // --- Top-right: exp2 falloff ---
        // 2^(-r*5) gives a bright center that fades out
        Halide::Expr exp2_val = Halide::pow(2.0f, -r * 5.0f);

        // --- Bottom-left: fmod checkerboard ---
        // fmod creates repeating patterns
        Halide::Expr fu = Halide::cast<float>(lx) / (half / 8.0f);
        Halide::Expr fv = Halide::cast<float>(ly) / (half / 8.0f);
        Halide::Expr fmod_u = fu - Halide::floor(fu);
        Halide::Expr fmod_v = fv - Halide::floor(fv);
        // Checkerboard: XOR of half-cycle thresholds
        Halide::Expr check_a = fmod_u > 0.5f;
        Halide::Expr check_b = fmod_v > 0.5f;
        Halide::Expr fmod_val = Halide::select(
            (check_a && !check_b) || (!check_a && check_b), 1.0f, 0.0f
        );

        // --- Bottom-right: heaviside circular mask ---
        // Heaviside of (0.5 - r): 1 inside circle, 0.5 on boundary, 0 outside
        Halide::Expr h_input = 0.5f - r;
        Halide::Expr heaviside_val = Halide::select(
            h_input < 0.0f, 0.0f,
            h_input > 0.0f, 1.0f,
            0.5f
        );

        // Compose quadrants
        Halide::Expr pixel = Halide::select(
            qy == 0 && qx == 0, sinc_val,
            qy == 0 && qx == 1, exp2_val,
            qy == 1 && qx == 0, fmod_val,
            heaviside_val
        );

        // Grid lines
        Halide::Expr on_grid = (ox == half) || (oy == half);
        pixel = Halide::select(on_grid, 0.4f, pixel);

        output(ox, oy) = Halide::cast<uint8_t>(Halide::clamp(pixel * 255.0f, 0.0f, 255.0f));

        std::cout << "Rendering..." << std::endl;
        Halide::Runtime::Buffer<uint8_t> result(size, size);
        output.realize(result);

        const char* output_path = "out/22_math.png";
        std::cout << "Saving to " << output_path << "..." << std::endl;

        if (save_png(result, output_path)) {
            std::cout << "Success!" << std::endl;
            std::cout << std::endl;
            std::cout << "Quadrant guide:" << std::endl;
            std::cout << "  Top-left:     sinc 2D radial pattern" << std::endl;
            std::cout << "  Top-right:    exp2 falloff" << std::endl;
            std::cout << "  Bottom-left:  fmod checkerboard" << std::endl;
            std::cout << "  Bottom-right: heaviside circular mask" << std::endl;
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
