/// @file 30_windows.cpp
/// @brief Example 30: Window functions
///
/// Demonstrates:
///   - Hanning-like 2D radial window on gradient
///   - Hamming-like 2D radial window
///   - Blackman-like 2D radial window (tighter)
///   - Four 1D window curves as horizontal strips
///
/// Output: out/30_windows.png

#include "numhalide_all.h"
#include "stbi_png.h"
#include <iostream>

using namespace numhalide;

int main(int argc, char **argv) {
    try {
        const int half = 256;
        const int size = half * 2;  // 512x512

        std::cout << "Window functions demonstration" << std::endl;
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

        // Centered coords for radial windows
        Halide::Expr cx = (Halide::cast<float>(lx) - half / 2.0f) / (half / 2.0f);
        Halide::Expr cy = (Halide::cast<float>(ly) - half / 2.0f) / (half / 2.0f);
        Halide::Expr r = Halide::clamp(Halide::sqrt(cx * cx + cy * cy), 0.0f, 1.0f);

        // Background gradient to show window effect
        Halide::Expr gradient = (u + v) * 0.5f;

        // --- Top-left: Hanning-like 2D radial window ---
        // Hanning: 0.5 * (1 - cos(2*pi*n/(N-1))), radial version: 0.5*(1+cos(pi*r))
        Halide::Expr hanning = Halide::select(
            r <= 1.0f, 0.5f * (1.0f + Halide::cos(pi * r)),
            0.0f
        );
        Halide::Expr hanning_val = gradient * hanning;

        // --- Top-right: Hamming-like 2D radial window ---
        // Hamming: 0.54 + 0.46*cos(pi*r)
        Halide::Expr hamming = Halide::select(
            r <= 1.0f, 0.54f + 0.46f * Halide::cos(pi * r),
            0.0f
        );
        Halide::Expr hamming_val = gradient * hamming;

        // --- Bottom-left: Blackman-like 2D radial window ---
        // Blackman: 0.42 + 0.5*cos(pi*r) + 0.08*cos(2*pi*r)
        Halide::Expr blackman = Halide::select(
            r <= 1.0f, 0.42f + 0.5f * Halide::cos(pi * r) + 0.08f * Halide::cos(2.0f * pi * r),
            0.0f
        );
        Halide::Expr blackman_val = gradient * blackman;

        // --- Bottom-right: Four 1D window curves as horizontal strips ---
        // Divide the quadrant into 4 horizontal strips, each showing a different window
        Halide::Expr strip = ly / (half / 4);  // 0..3
        Halide::Expr t = u;  // horizontal position [0,1)

        // Rectangular window
        Halide::Expr rect_w = Halide::select(t > 0.1f && t < 0.9f, 1.0f, 0.0f);
        // Hanning 1D
        Halide::Expr hann_1d = 0.5f * (1.0f - Halide::cos(2.0f * pi * t));
        // Hamming 1D
        Halide::Expr hamm_1d = 0.54f - 0.46f * Halide::cos(2.0f * pi * t);
        // Blackman 1D
        Halide::Expr black_1d = 0.42f - 0.5f * Halide::cos(2.0f * pi * t) + 0.08f * Halide::cos(4.0f * pi * t);

        // For each strip, draw the curve: bright where v_in_strip is near the window value
        Halide::Expr v_in_strip = Halide::cast<float>(ly % (half / 4)) / (half / 4.0f);
        // Invert v so curve goes up
        Halide::Expr v_inv = 1.0f - v_in_strip;

        Halide::Expr curve_val_sel = Halide::select(
            strip == 0, rect_w,
            strip == 1, hann_1d,
            strip == 2, hamm_1d,
            black_1d
        );

        // Draw curve: bright pixel if v_inv is close to curve value
        Halide::Expr curve_dist = Halide::abs(v_inv - curve_val_sel);
        Halide::Expr curves_val = Halide::select(curve_dist < 0.02f, 1.0f, 0.08f);

        // Compose quadrants
        Halide::Expr pixel = Halide::select(
            qy == 0 && qx == 0, hanning_val,
            qy == 0 && qx == 1, hamming_val,
            qy == 1 && qx == 0, blackman_val,
            curves_val
        );

        // Grid lines
        Halide::Expr on_grid = (ox == half) || (oy == half);
        pixel = Halide::select(on_grid, 0.4f, pixel);

        output(ox, oy) = Halide::cast<uint8_t>(Halide::clamp(pixel * 255.0f, 0.0f, 255.0f));

        std::cout << "Rendering..." << std::endl;
        Halide::Runtime::Buffer<uint8_t> result(size, size);
        output.realize(result);

        const char* output_path = "out/30_windows.png";
        std::cout << "Saving to " << output_path << "..." << std::endl;

        if (save_png(result, output_path)) {
            std::cout << "Success!" << std::endl;
            std::cout << std::endl;
            std::cout << "Quadrant guide:" << std::endl;
            std::cout << "  Top-left:     Hanning-like 2D radial window on gradient" << std::endl;
            std::cout << "  Top-right:    Hamming-like 2D radial window" << std::endl;
            std::cout << "  Bottom-left:  Blackman-like 2D radial window (tighter)" << std::endl;
            std::cout << "  Bottom-right: four 1D window curves as horizontal strips" << std::endl;
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
