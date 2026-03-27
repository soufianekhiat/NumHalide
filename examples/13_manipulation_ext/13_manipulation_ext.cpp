/// @file 13_manipulation_ext.cpp
/// @brief Example 13: Extended manipulation operations
///
/// Demonstrates:
///   - flip / flipud  for reversing arrays along an axis
///   - rot90          for 90-degree rotation
///   - roll           for circular shift along an axis
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

        std::cout << "Extended manipulation demonstration\n";
        std::cout << "Output size: " << output_size << "x" << output_size << "\n\n";

        // Create an asymmetric source pattern so transformations are obvious:
        // diagonal gradient background + bright circle in the top-left corner
        Halide::Func pattern("pattern");
        Halide::Var x("x"), y("y");

        Halide::Expr cx = size / 4;
        Halide::Expr cy = size / 4;
        Halide::Expr dist = (x - cx) * (x - cx) + (y - cy) * (y - cy);
        Halide::Expr in_circle = dist < 20 * 20;

        Halide::Expr grad = Halide::cast<float>(x + y) / (2.0f * size);
        pattern(x, y) = Halide::select(in_circle, 1.0f, grad * 0.7f);

        shape_t shape = {size, size};

        // ---- Quadrant 1 (TL): Original ----
        // (displayed as-is for reference)

        // ---- Quadrant 2 (TR): rot90 ----
        auto rotated  = rot90(pattern, shape, 1, "rotated");

        // ---- Quadrant 3 (BL): flipud ----
        auto flipped  = flipud(pattern, shape, "flipped");

        // ---- Quadrant 4 (BR): roll along axis=1 (columns) by 3/4 of width ----
        // Shift = 3*size/4 wraps the circle marker from x=32 to x=(32+96)%128=0.
        // The wrap-around is clearly visible at the left/right edges.
        const int roll_shift = (size * 3) / 4;
        auto rolled = roll(pattern, shape, roll_shift, 1, "rolled");

        std::cout << "Transformations applied:\n";
        std::cout << "  Top-left:     Original pattern\n";
        std::cout << "  Top-right:    rot90 (90 degrees CCW)\n";
        std::cout << "  Bottom-left:  flipud (vertical flip)\n";
        std::cout << "  Bottom-right: roll along columns by " << roll_shift << " px (wraps)\n\n";

        // Build output composite
        Halide::Func output("output");
        Halide::Var ox("ox"), oy("oy");

        Halide::Expr qx = ox / size;
        Halide::Expr qy = oy / size;
        Halide::Expr lx = ox % size;
        Halide::Expr ly = oy % size;

        Halide::Expr pixel = Halide::select(
            qy == 0 && qx == 0, pattern(lx, ly),
            qy == 0 && qx == 1, rotated(lx, ly),
            qy == 1 && qx == 0, flipped(lx, ly),
            rolled(lx, ly)
        );

        // Grid lines between quadrants
        Halide::Expr on_grid = (ox == size) || (oy == size);
        pixel = Halide::select(on_grid, 0.3f, pixel);

        output(ox, oy) = Halide::cast<uint8_t>(Halide::clamp(pixel * 255.0f, 0.0f, 255.0f));

        std::cout << "Rendering...\n";
        Halide::Runtime::Buffer<uint8_t> result(output_size, output_size);
        output.realize(result);

        const char* out_path = "out/13_manipulation_ext.png";
        if (save_png(result, out_path)) {
            std::cout << "Saved to " << out_path << "\n";
        } else {
            std::cerr << "Error: Failed to save PNG\n";
            return 1;
        }
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
