/// @file 14_comparisons.cpp
/// @brief Example 14: Comparison and logical operations
///
/// Demonstrates:
///   - greater, less, equal for threshold-based operations
///   - logical_and, logical_or for combining conditions
///   - Edge detection using comparisons
///
/// Output: out/14_comparisons.png

#include "numhalide_all.h"
#include "stbi_png.h"
#include <iostream>

using namespace numhalide;

int main(int argc, char **argv) {
    try {
        const int size = 256;

        std::cout << "Comparison and logical operations demonstration" << std::endl;
        std::cout << "Image size: " << size << "x" << size << std::endl;
        std::cout << std::endl;

        // Create a test image with gradients and shapes
        Halide::Func input("input");
        Halide::Var x("x"), y("y");

        // Create a pattern with circles at different intensities
        Halide::Expr cx1 = size / 3, cy1 = size / 3;
        Halide::Expr cx2 = 2 * size / 3, cy2 = 2 * size / 3;
        Halide::Expr d1 = (x - cx1) * (x - cx1) + (y - cy1) * (y - cy1);
        Halide::Expr d2 = (x - cx2) * (x - cx2) + (y - cy2) * (y - cy2);
        Halide::Expr r1 = 40, r2 = 50;

        Halide::Expr in_circle1 = d1 < r1 * r1;
        Halide::Expr in_circle2 = d2 < r2 * r2;

        // Background gradient + circles
        Halide::Expr grad = Halide::cast<float>(x + y) / (2 * size);
        input(x, y) = Halide::select(in_circle1, 0.9f,
                      Halide::select(in_circle2, 0.6f, grad));

        shape_t shape = {size, size};

        // Compute gradients for edge detection
        Halide::Func grad_x("grad_x"), grad_y("grad_y");
        grad_x(x, y) = input(Halide::clamp(x + 1, 0, size - 1), y) -
                       input(Halide::clamp(x - 1, 0, size - 1), y);
        grad_y(x, y) = input(x, Halide::clamp(y + 1, 0, size - 1)) -
                       input(x, Halide::clamp(y - 1, 0, size - 1));

        // Gradient magnitude
        Halide::Func grad_mag("grad_mag");
        grad_mag(x, y) = Halide::sqrt(grad_x(x, y) * grad_x(x, y) +
                                       grad_y(x, y) * grad_y(x, y));

        // Use comparisons for edge thresholding
        Halide::Expr threshold = 0.1f;
        auto edges = greater(grad_mag, threshold, shape, "edges");

        // Use comparisons for intensity ranges
        auto bright = greater(input, 0.7f, shape, "bright");
        auto mid = logical_and(
            greater(input, 0.3f, shape, "gt_low"),
            less(input, 0.7f, shape, "lt_high"),
            shape, "mid"
        );

        // Combine edge detection with intensity
        auto edge_on_bright = logical_and(edges, bright, shape, "edge_bright");

        std::cout << "Operations:" << std::endl;
        std::cout << "  Top-left:     Original image" << std::endl;
        std::cout << "  Top-right:    Edge detection (gradient > threshold)" << std::endl;
        std::cout << "  Bottom-left:  Mid-intensity mask (0.3 < val < 0.7)" << std::endl;
        std::cout << "  Bottom-right: Edges on bright regions" << std::endl;

        // Build output
        const int half = size / 2;
        Halide::Func output("output");
        Halide::Var ox("ox"), oy("oy");

        Halide::Expr qx = ox / half;
        Halide::Expr qy = oy / half;
        Halide::Expr lx = (ox % half) * 2;
        Halide::Expr ly = (oy % half) * 2;

        Halide::Expr pixel = Halide::select(
            qy == 0 && qx == 0, input(lx, ly),
            Halide::select(
                qy == 0 && qx == 1, Halide::cast<float>(edges(lx, ly)),
                Halide::select(
                    qy == 1 && qx == 0, Halide::cast<float>(mid(lx, ly)) * 0.7f,
                    Halide::cast<float>(edge_on_bright(lx, ly))
                )
            )
        );

        output(ox, oy) = Halide::cast<uint8_t>(Halide::clamp(pixel * 255.0f, 0.0f, 255.0f));

        // Realize
        std::cout << std::endl << "Rendering output..." << std::endl;
        Halide::Runtime::Buffer<uint8_t> result(size, size);
        output.realize(result);

        // Save
        const char* output_path = "out/14_comparisons.png";
        std::cout << "Saving to " << output_path << "..." << std::endl;

        if (save_png(result, output_path)) {
            std::cout << "Success! Comparison operations visualization saved." << std::endl;
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
