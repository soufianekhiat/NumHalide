/// @file 29_bitwise.cpp
/// @brief Example 29: Bitwise operations
///
/// Demonstrates:
///   - gradient image
///   - bit-plane extraction (highest bit as binary)
///   - XOR pattern (Sierpinski-like fractal)
///   - AND mask grid pattern
///
/// Output: out/29_bitwise.png

#include "numhalide_all.h"
#include "stbi_png.h"
#include <iostream>

using namespace numhalide;

int main(int argc, char **argv) {
    try {
        const int half = 256;
        const int size = half * 2;  // 512x512

        std::cout << "Bitwise operations demonstration" << std::endl;
        std::cout << "Output size: " << size << "x" << size << std::endl;
        std::cout << std::endl;

        Halide::Var ox("ox"), oy("oy");
        Halide::Func output("output");

        Halide::Expr qx = ox / half;
        Halide::Expr qy = oy / half;
        Halide::Expr lx = ox % half;
        Halide::Expr ly = oy % half;

        // Normalized coords [0,1)
        Halide::Expr u = Halide::cast<float>(lx) / half;
        Halide::Expr v = Halide::cast<float>(ly) / half;

        // --- Top-left: gradient image ---
        // Simple diagonal gradient combining x and y
        Halide::Expr grad_val = (u + v) * 0.5f;

        // --- Top-right: bit-plane extraction (highest bit as binary) ---
        // Map local coords to 0-255 range, extract bit 7 (highest bit)
        Halide::Expr intensity = Halide::cast<int>(u * 255.0f);
        Halide::Expr bit7 = (intensity >> 7) & 1;
        Halide::Expr bitplane_val = Halide::cast<float>(bit7);

        // --- Bottom-left: XOR pattern (Sierpinski-like fractal) ---
        // XOR of x and y coordinates produces a fractal pattern
        Halide::Expr xor_val = lx ^ ly;
        Halide::Expr xor_norm = Halide::cast<float>(xor_val % 256) / 255.0f;

        // --- Bottom-right: AND mask grid pattern ---
        // AND of coordinates with a mask creates regular grid patterns
        Halide::Expr and_mask = 0x1F;  // 31 - repeating blocks of 32
        Halide::Expr and_x = lx & and_mask;
        Halide::Expr and_y = ly & and_mask;
        Halide::Expr and_val = Halide::cast<float>(and_x * and_y) / Halide::cast<float>(and_mask * and_mask);

        // Compose quadrants
        Halide::Expr pixel = Halide::select(
            qy == 0 && qx == 0, grad_val,
            qy == 0 && qx == 1, bitplane_val,
            qy == 1 && qx == 0, xor_norm,
            and_val
        );

        // Grid lines
        Halide::Expr on_grid = (ox == half) || (oy == half);
        pixel = Halide::select(on_grid, 0.4f, pixel);

        output(ox, oy) = Halide::cast<uint8_t>(Halide::clamp(pixel * 255.0f, 0.0f, 255.0f));

        std::cout << "Rendering..." << std::endl;
        Halide::Runtime::Buffer<uint8_t> result(size, size);
        output.realize(result);

        const char* output_path = "out/29_bitwise.png";
        std::cout << "Saving to " << output_path << "..." << std::endl;

        if (save_png(result, output_path)) {
            std::cout << "Success!" << std::endl;
            std::cout << std::endl;
            std::cout << "Quadrant guide:" << std::endl;
            std::cout << "  Top-left:     gradient image" << std::endl;
            std::cout << "  Top-right:    bit-plane extraction (highest bit)" << std::endl;
            std::cout << "  Bottom-left:  XOR pattern (Sierpinski-like fractal)" << std::endl;
            std::cout << "  Bottom-right: AND mask grid pattern" << std::endl;
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
