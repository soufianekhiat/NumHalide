/// @file 34_color.cpp
/// @brief Example 34: Color space operations
///
/// Demonstrates:
///   - synthetic RGB test pattern shown as luminance
///   - Hue-like channel (color wheel pattern)
///   - Saturation-like channel
///   - YUV Y-like channel (perceptual luminance)
///
/// Output: out/34_color.png

#include "numhalide_all.h"
#include "stbi_png.h"
#include <iostream>

using namespace numhalide;

int main(int argc, char **argv) {
    try {
        const int half = 256;
        const int size = half * 2;  // 512x512

        std::cout << "Color space operations demonstration" << std::endl;
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

        // Centered coords for polar
        Halide::Expr cx = (Halide::cast<float>(lx) - half / 2.0f) / (half / 2.0f);
        Halide::Expr cy = (Halide::cast<float>(ly) - half / 2.0f) / (half / 2.0f);
        Halide::Expr r = Halide::sqrt(cx * cx + cy * cy);
        Halide::Expr angle = Halide::atan2(cy, cx);

        // --- Synthetic RGB test pattern ---
        // Create three channels that vary differently across the image
        // R: horizontal gradient with vertical sine modulation
        Halide::Expr R = Halide::clamp(u * (0.5f + 0.5f * Halide::sin(v * 4.0f * pi)), 0.0f, 1.0f);
        // G: vertical gradient with horizontal cosine modulation
        Halide::Expr G = Halide::clamp(v * (0.5f + 0.5f * Halide::cos(u * 6.0f * pi)), 0.0f, 1.0f);
        // B: radial pattern (bright center fading out)
        Halide::Expr B = Halide::clamp(1.0f - r * 0.8f, 0.0f, 1.0f);

        // --- Top-left: RGB shown as simple average luminance ---
        Halide::Expr avg_lum = (R + G + B) / 3.0f;

        // --- Top-right: Hue-like channel (color wheel) ---
        // Compute hue from RGB: approximate using atan2 of color differences
        // Hue is the angle in the color hexagon, approximated here
        Halide::Expr rgb_max = Halide::max(R, Halide::max(G, B));
        Halide::Expr rgb_min = Halide::min(R, Halide::min(G, B));
        Halide::Expr chroma = rgb_max - rgb_min;

        // Hue computation (simplified):
        // When chroma ~= 0, hue is undefined (achromatic)
        // When max=R: hue = (G-B)/chroma mod 6
        // When max=G: hue = (B-R)/chroma + 2
        // When max=B: hue = (R-G)/chroma + 4
        Halide::Expr hue_raw = Halide::select(
            chroma < 0.001f, 0.0f,
            rgb_max == R, (G - B) / chroma,
            rgb_max == G, (B - R) / chroma + 2.0f,
            (R - G) / chroma + 4.0f
        );
        // Normalize to [0, 1] (hue_raw is in [0, 6) range, but can be negative)
        Halide::Expr hue_mod = hue_raw - Halide::floor(hue_raw / 6.0f) * 6.0f;
        Halide::Expr hue_val = hue_mod / 6.0f;

        // --- Bottom-left: Saturation-like channel ---
        // Saturation = chroma / max (HSV saturation)
        Halide::Expr saturation = Halide::select(
            rgb_max < 0.001f, 0.0f,
            chroma / rgb_max
        );

        // --- Bottom-right: YUV Y-like channel (perceptual luminance) ---
        // Y = 0.299*R + 0.587*G + 0.114*B (BT.601 luma)
        Halide::Expr yuv_y = 0.299f * R + 0.587f * G + 0.114f * B;

        // Compose quadrants
        Halide::Expr pixel = Halide::select(
            qy == 0 && qx == 0, avg_lum,
            qy == 0 && qx == 1, hue_val,
            qy == 1 && qx == 0, saturation,
            yuv_y
        );

        // Grid lines
        Halide::Expr on_grid = (ox == half) || (oy == half);
        pixel = Halide::select(on_grid, 0.4f, pixel);

        output(ox, oy) = Halide::cast<uint8_t>(Halide::clamp(pixel * 255.0f, 0.0f, 255.0f));

        std::cout << "Rendering..." << std::endl;
        Halide::Runtime::Buffer<uint8_t> result(size, size);
        output.realize(result);

        const char* output_path = "out/34_color.png";
        std::cout << "Saving to " << output_path << "..." << std::endl;

        if (save_png(result, output_path)) {
            std::cout << "Success!" << std::endl;
            std::cout << std::endl;
            std::cout << "Quadrant guide:" << std::endl;
            std::cout << "  Top-left:     synthetic RGB test pattern as luminance" << std::endl;
            std::cout << "  Top-right:    Hue-like channel (color wheel pattern)" << std::endl;
            std::cout << "  Bottom-left:  Saturation-like channel" << std::endl;
            std::cout << "  Bottom-right: YUV Y-like channel (perceptual luminance)" << std::endl;
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
