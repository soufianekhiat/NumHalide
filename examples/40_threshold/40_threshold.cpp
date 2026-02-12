/// @file 40_threshold.cpp
/// @brief Example 40: Thresholding techniques visualization
///
/// Demonstrates:
///   - grayscale with uneven illumination (gradient + circles)
///   - global binary threshold
///   - Otsu-like automatic threshold
///   - adaptive local threshold
///
/// Output: out/40_threshold.png

#include "numhalide_all.h"
#include "stbi_png.h"
#include <iostream>

using namespace numhalide;

int main(int argc, char **argv) {
    try {
        const int half = 256;
        const int size = half * 2;  // 512x512

        std::cout << "Thresholding techniques demonstration" << std::endl;
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

        // --- Build a scene with uneven illumination ---
        // Uneven illumination: bright top-left, dark bottom-right
        Halide::Expr illumination = 1.0f - (u + v) * 0.4f;

        // Scene features: several circles of varying size
        // Circle 1: center (-0.4, -0.3), radius 0.2
        Halide::Expr d1 = Halide::sqrt((cu + 0.4f) * (cu + 0.4f) + (cv + 0.3f) * (cv + 0.3f));
        Halide::Expr obj1 = Halide::select(d1 < 0.2f, 0.0f, 1.0f);

        // Circle 2: center (0.3, -0.4), radius 0.15
        Halide::Expr d2 = Halide::sqrt((cu - 0.3f) * (cu - 0.3f) + (cv + 0.4f) * (cv + 0.4f));
        Halide::Expr obj2 = Halide::select(d2 < 0.15f, 0.0f, 1.0f);

        // Circle 3: center (0.0, 0.2), radius 0.25
        Halide::Expr d3 = Halide::sqrt(cu * cu + (cv - 0.2f) * (cv - 0.2f));
        Halide::Expr obj3 = Halide::select(d3 < 0.25f, 0.0f, 1.0f);

        // Circle 4: center (0.5, 0.5), radius 0.18
        Halide::Expr d4 = Halide::sqrt((cu - 0.5f) * (cu - 0.5f) + (cv - 0.5f) * (cv - 0.5f));
        Halide::Expr obj4 = Halide::select(d4 < 0.18f, 0.0f, 1.0f);

        // Circle 5: center (-0.5, 0.4), radius 0.12
        Halide::Expr d5 = Halide::sqrt((cu + 0.5f) * (cu + 0.5f) + (cv - 0.4f) * (cv - 0.4f));
        Halide::Expr obj5 = Halide::select(d5 < 0.12f, 0.0f, 1.0f);

        // Combined objects (dark circles on bright background)
        Halide::Expr objects = obj1 * obj2 * obj3 * obj4 * obj5;

        // Final scene: objects modulated by uneven illumination
        Halide::Expr scene = objects * illumination;
        scene = Halide::clamp(scene, 0.0f, 1.0f);

        // --- Top-left: grayscale with uneven illumination ---
        Halide::Expr tl_val = scene;

        // --- Top-right: global binary threshold at 0.5 ---
        Halide::Expr global_thresh = 0.5f;
        Halide::Expr tr_val = Halide::select(scene > global_thresh, 1.0f, 0.0f);

        // --- Bottom-left: Otsu-like automatic threshold ---
        // Otsu's method finds optimal threshold to separate foreground/background.
        // For this synthetic image, the bimodal distribution has a valley around 0.4.
        // Simulate the Otsu threshold at approximately 0.4.
        Halide::Expr otsu_thresh = 0.4f;
        Halide::Expr bl_val = Halide::select(scene > otsu_thresh, 1.0f, 0.0f);

        // --- Bottom-right: adaptive local threshold ---
        // Compare each pixel to a local mean approximation.
        // Approximate local mean using a smooth version of illumination.
        // The adaptive threshold is: pixel > local_mean - C
        // Here we use the illumination gradient itself as local mean proxy,
        // with a small offset C to handle edge cases.
        Halide::Expr local_mean = illumination * 0.85f;
        Halide::Expr adapt_c = 0.08f;
        Halide::Expr br_val = Halide::select(scene > (local_mean - adapt_c), 1.0f, 0.0f);

        // Compose quadrants
        Halide::Expr pixel = Halide::select(
            qy == 0 && qx == 0, tl_val,
            qy == 0 && qx == 1, tr_val,
            qy == 1 && qx == 0, bl_val,
            br_val
        );

        // Grid lines
        Halide::Expr on_grid = (ox == half) || (oy == half);
        pixel = Halide::select(on_grid, 0.4f, pixel);

        output(ox, oy) = Halide::cast<uint8_t>(Halide::clamp(pixel * 255.0f, 0.0f, 255.0f));

        std::cout << "Rendering..." << std::endl;
        Halide::Runtime::Buffer<uint8_t> result(size, size);
        output.realize(result);

        const char* output_path = "out/40_threshold.png";
        std::cout << "Saving to " << output_path << "..." << std::endl;

        if (save_png(result, output_path)) {
            std::cout << "Success!" << std::endl;
            std::cout << std::endl;
            std::cout << "Quadrant guide:" << std::endl;
            std::cout << "  Top-left:     grayscale with uneven illumination" << std::endl;
            std::cout << "  Top-right:    global binary threshold (0.5)" << std::endl;
            std::cout << "  Bottom-left:  Otsu-like automatic threshold (0.4)" << std::endl;
            std::cout << "  Bottom-right: adaptive local threshold" << std::endl;
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
