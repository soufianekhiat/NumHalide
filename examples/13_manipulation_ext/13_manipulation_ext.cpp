/// @file 13_manipulation_ext.cpp
/// @brief Example 13: Extended manipulation operations
///
/// Demonstrates:
///   - flip, flipud, fliplr for reversing arrays
///   - rot90 for rotating images
///   - tile for repeating patterns
///   - pad for boundary handling
///
/// Output: out/13_manipulation_ext.png

#include "numhalide_all.h"
#include "stbi_png.h"
#include <iostream>

using namespace numhalide;

int main(int argc, char **argv) {
    try {
        const int size = 128;  // Size of each quadrant
        const int output_size = size * 2;

        std::cout << "Extended manipulation demonstration" << std::endl;
        std::cout << "Output size: " << output_size << "x" << output_size << std::endl;
        std::cout << std::endl;

        // Create a test pattern: gradient with a marker
        Halide::Func pattern("pattern");
        Halide::Var x("x"), y("y");

        // Create an asymmetric pattern to show transformations
        Halide::Expr cx = size / 4;
        Halide::Expr cy = size / 4;
        Halide::Expr dist = (x - cx) * (x - cx) + (y - cy) * (y - cy);
        Halide::Expr in_circle = dist < 20 * 20;

        // Gradient background with circle marker
        Halide::Expr grad = Halide::cast<float>(x + y) / (2 * size);
        pattern(x, y) = Halide::select(in_circle, 1.0f, grad * 0.7f);

        shape_t shape = {size, size};

        std::cout << "Transformations:" << std::endl;

        // Quadrant 1: Original
        std::cout << "  Top-left:     Original pattern" << std::endl;

        // Quadrant 2: Rot90
        auto rotated = rot90(pattern, shape, 1, "rotated");
        auto rot_shape = infer_rot90(shape, 1);
        std::cout << "  Top-right:    Rotated 90 degrees CCW" << std::endl;

        // Quadrant 3: Flipped
        auto flipped = flipud(pattern, shape, "flipped");
        std::cout << "  Bottom-left:  Flipped vertically" << std::endl;

        // Quadrant 4: Tiled (2x2 of smaller pattern)
        shape_t small_shape = {size / 2, size / 2};
        Halide::Func small_pattern("small_pattern");
        small_pattern(x, y) = pattern(x * 2, y * 2);  // Downsample

        auto tiled = tile(small_pattern, small_shape, {2, 2}, "tiled");
        std::cout << "  Bottom-right: Tiled 2x2" << std::endl;

        // Build output composite
        Halide::Func output("output");
        Halide::Var ox("ox"), oy("oy");

        Halide::Expr qx = ox / size;
        Halide::Expr qy = oy / size;
        Halide::Expr lx = ox % size;
        Halide::Expr ly = oy % size;

        Halide::Expr pixel = Halide::select(
            qy == 0 && qx == 0, pattern(lx, ly),
            Halide::select(
                qy == 0 && qx == 1, rotated(lx, ly),
                Halide::select(
                    qy == 1 && qx == 0, flipped(lx, ly),
                    tiled(lx, ly)
                )
            )
        );

        // Add grid lines
        Halide::Expr on_grid = (ox == size) || (oy == size);
        pixel = Halide::select(on_grid, 0.3f, pixel);

        output(ox, oy) = Halide::cast<uint8_t>(Halide::clamp(pixel * 255.0f, 0.0f, 255.0f));

        // Realize
        std::cout << std::endl << "Rendering output..." << std::endl;
        Halide::Runtime::Buffer<uint8_t> result(output_size, output_size);
        output.realize(result);

        // Save
        const char* output_path = "out/13_manipulation_ext.png";
        std::cout << "Saving to " << output_path << "..." << std::endl;

        if (save_png(result, output_path)) {
            std::cout << "Success! Manipulation visualization saved." << std::endl;
            std::cout << std::endl;
            std::cout << "Quadrant guide:" << std::endl;
            std::cout << "  Top-left:     Original (gradient + circle marker)" << std::endl;
            std::cout << "  Top-right:    rot90 (rotated 90 degrees CCW)" << std::endl;
            std::cout << "  Bottom-left:  flipud (flipped vertically)" << std::endl;
            std::cout << "  Bottom-right: tile (2x2 tiling)" << std::endl;
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
