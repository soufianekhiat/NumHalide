/// @file 33_morphology.cpp
/// @brief Example 33: Morphological operations
///
/// Demonstrates:
///   - binary image with circles and noise
///   - erosion-like effect (circles shrunk)
///   - dilation-like effect (circles grown)
///   - morphological gradient (outlines)
///
/// Output: out/33_morphology.png

#include "numhalide_all.h"
#include "stbi_png.h"
#include <iostream>

using namespace numhalide;

int main(int argc, char **argv) {
    try {
        const int half = 256;
        const int size = half * 2;  // 512x512

        std::cout << "Morphological operations demonstration" << std::endl;
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

        // --- Define a binary image with circles ---
        // Circle 1: center (0.25, 0.25), radius 0.15
        Halide::Expr d1 = Halide::sqrt((u - 0.25f) * (u - 0.25f) + (v - 0.25f) * (v - 0.25f));
        Halide::Expr c1 = d1 < 0.15f;

        // Circle 2: center (0.7, 0.3), radius 0.1
        Halide::Expr d2 = Halide::sqrt((u - 0.7f) * (u - 0.7f) + (v - 0.3f) * (v - 0.3f));
        Halide::Expr c2 = d2 < 0.1f;

        // Circle 3: center (0.5, 0.65), radius 0.18
        Halide::Expr d3 = Halide::sqrt((u - 0.5f) * (u - 0.5f) + (v - 0.65f) * (v - 0.65f));
        Halide::Expr c3 = d3 < 0.18f;

        // Circle 4: center (0.2, 0.75), radius 0.07
        Halide::Expr d4 = Halide::sqrt((u - 0.2f) * (u - 0.2f) + (v - 0.75f) * (v - 0.75f));
        Halide::Expr c4 = d4 < 0.07f;

        // Circle 5: center (0.8, 0.7), radius 0.12
        Halide::Expr d5 = Halide::sqrt((u - 0.8f) * (u - 0.8f) + (v - 0.7f) * (v - 0.7f));
        Halide::Expr c5 = d5 < 0.12f;

        // Add pseudo-random noise dots using a hash-like pattern
        // Use sin-based pseudo-random: sin(lx*127.1 + ly*311.7) fract
        Halide::Expr hash_val = Halide::sin(Halide::cast<float>(lx) * 127.1f + Halide::cast<float>(ly) * 311.7f) * 43758.5453f;
        Halide::Expr noise_fract = hash_val - Halide::floor(hash_val);
        Halide::Expr noise_dots = noise_fract > 0.97f;  // sparse noise

        // Binary image: circles OR noise
        Halide::Expr binary = c1 || c2 || c3 || c4 || c5 || noise_dots;
        Halide::Expr binary_val = Halide::select(binary, 1.0f, 0.0f);

        // --- Erosion-like effect ---
        // A pixel is ON only if it and all neighbors within a radius are ON
        // Approximate by requiring the distance to nearest circle boundary is > erosion_radius
        Halide::Expr erode_r = 0.03f;  // erosion radius
        Halide::Expr eroded_c1 = d1 < (0.15f - erode_r);
        Halide::Expr eroded_c2 = d2 < (0.1f - erode_r);
        Halide::Expr eroded_c3 = d3 < (0.18f - erode_r);
        Halide::Expr eroded_c4 = d4 < (0.07f - erode_r);
        Halide::Expr eroded_c5 = d5 < (0.12f - erode_r);
        // Noise dots are removed by erosion (too small to survive)
        Halide::Expr eroded = eroded_c1 || eroded_c2 || eroded_c3 || eroded_c4 || eroded_c5;
        Halide::Expr eroded_val = Halide::select(eroded, 1.0f, 0.0f);

        // --- Dilation-like effect ---
        // A pixel is ON if any pixel within a radius is ON
        // Approximate by expanding circle radii
        Halide::Expr dilate_r = 0.03f;  // dilation radius
        Halide::Expr dilated_c1 = d1 < (0.15f + dilate_r);
        Halide::Expr dilated_c2 = d2 < (0.1f + dilate_r);
        Halide::Expr dilated_c3 = d3 < (0.18f + dilate_r);
        Halide::Expr dilated_c4 = d4 < (0.07f + dilate_r);
        Halide::Expr dilated_c5 = d5 < (0.12f + dilate_r);
        // Noise dots grow slightly
        Halide::Expr dilated_noise = noise_fract > 0.96f;
        Halide::Expr dilated = dilated_c1 || dilated_c2 || dilated_c3 || dilated_c4 || dilated_c5 || dilated_noise;
        Halide::Expr dilated_val = Halide::select(dilated, 1.0f, 0.0f);

        // --- Morphological gradient (outlines) ---
        // Dilation - Erosion = outline of the shapes
        Halide::Expr morph_grad = Halide::select(dilated && !eroded, 1.0f, 0.0f);

        // Compose quadrants
        Halide::Expr pixel = Halide::select(
            qy == 0 && qx == 0, binary_val,
            qy == 0 && qx == 1, eroded_val,
            qy == 1 && qx == 0, dilated_val,
            morph_grad
        );

        // Grid lines
        Halide::Expr on_grid = (ox == half) || (oy == half);
        pixel = Halide::select(on_grid, 0.4f, pixel);

        output(ox, oy) = Halide::cast<uint8_t>(Halide::clamp(pixel * 255.0f, 0.0f, 255.0f));

        std::cout << "Rendering..." << std::endl;
        Halide::Runtime::Buffer<uint8_t> result(size, size);
        output.realize(result);

        const char* output_path = "out/33_morphology.png";
        std::cout << "Saving to " << output_path << "..." << std::endl;

        if (save_png(result, output_path)) {
            std::cout << "Success!" << std::endl;
            std::cout << std::endl;
            std::cout << "Quadrant guide:" << std::endl;
            std::cout << "  Top-left:     binary image with circles and noise" << std::endl;
            std::cout << "  Top-right:    erosion-like effect (circles shrunk)" << std::endl;
            std::cout << "  Bottom-left:  dilation-like effect (circles grown)" << std::endl;
            std::cout << "  Bottom-right: morphological gradient (outlines)" << std::endl;
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
