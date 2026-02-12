/// @file 19_convolution.cpp
/// @brief Example 19: Convolution and filter gallery
///
/// Demonstrates:
///   - convolve2d with various kernels
///   - box_kernel for blur
///   - sobel_x_kernel, sobel_y_kernel for edge detection
///   - laplacian_kernel for second derivative
///   - gaussian_kernel_1d for Gaussian blur
///
/// Output: out/19_convolution.png

#include "numhalide_all.h"
#include "stbi_png.h"
#include <iostream>
#include <cmath>

using namespace numhalide;

int main(int argc, char **argv) {
    try {
        const int tile_size = 128;
        const int output_size = tile_size * 2;

        std::cout << "Convolution filter gallery" << std::endl;
        std::cout << "Output size: " << output_size << "x" << output_size << std::endl;
        std::cout << std::endl;

        // Create a test image with various features
        Halide::Func input("input");
        Halide::Var x("x"), y("y");

        // Create a pattern with edges and gradients
        Halide::Expr cx = tile_size / 2;
        Halide::Expr cy = tile_size / 2;

        // Circle
        Halide::Expr dist_sq = (x - cx) * (x - cx) + (y - cy) * (y - cy);
        Halide::Expr in_circle = dist_sq < (tile_size / 4) * (tile_size / 4);

        // Rectangle
        Halide::Expr in_rect = (x > tile_size / 4) && (x < 3 * tile_size / 4) &&
                               (y > tile_size / 6) && (y < tile_size / 3);

        // Diagonal lines
        Halide::Expr diag1 = Halide::abs((x - y) % 16) < 2;

        // Combine features
        input(x, y) = Halide::cast<float>(
            Halide::select(in_circle, 0.9f,
            Halide::select(in_rect, 0.7f,
            Halide::select(diag1, 0.5f, 0.1f)))
        );

        shape_t shape = {tile_size, tile_size};

        std::cout << "Applying filters:" << std::endl;

        // Quadrant 1: Original
        std::cout << "  Top-left:     Original image" << std::endl;

        // Quadrant 2: Box blur
        auto box = box_kernel(5, "box");
        auto blurred = convolve2d(input, shape, box, 5, 5, "blurred");
        std::cout << "  Top-right:    5x5 box blur" << std::endl;

        // Quadrant 3: Sobel edge detection (magnitude)
        auto sobel_x = sobel_x_kernel("sx");
        auto sobel_y = sobel_y_kernel("sy");
        auto edge_x = convolve2d(input, shape, sobel_x, 3, 3, "edge_x");
        auto edge_y = convolve2d(input, shape, sobel_y, 3, 3, "edge_y");

        Halide::Func edge_mag("edge_mag");
        edge_mag(x, y) = Halide::sqrt(edge_x(x, y) * edge_x(x, y) +
                                       edge_y(x, y) * edge_y(x, y)) / 4.0f;
        std::cout << "  Bottom-left:  Sobel edge magnitude" << std::endl;

        // Quadrant 4: Laplacian (edge enhancement)
        auto lap = laplacian_kernel("lap");
        auto laplacian = convolve2d(input, shape, lap, 3, 3, "laplacian");

        // Enhance by subtracting Laplacian (sharpening)
        Halide::Func sharpened("sharpened");
        sharpened(x, y) = input(x, y) - 0.3f * laplacian(x, y);
        std::cout << "  Bottom-right: Laplacian sharpening" << std::endl;

        // Build output composite
        Halide::Func output("output");
        Halide::Var ox("ox"), oy("oy");

        Halide::Expr qx = ox / tile_size;
        Halide::Expr qy = oy / tile_size;
        Halide::Expr lx = ox % tile_size;
        Halide::Expr ly = oy % tile_size;

        Halide::Expr pixel = Halide::select(
            qy == 0 && qx == 0, input(lx, ly),
            Halide::select(
                qy == 0 && qx == 1, blurred(lx, ly),
                Halide::select(
                    qy == 1 && qx == 0, edge_mag(lx, ly),
                    Halide::clamp(sharpened(lx, ly), 0.0f, 1.0f)
                )
            )
        );

        // Add grid lines
        Halide::Expr on_grid = (ox == tile_size) || (oy == tile_size);
        pixel = Halide::select(on_grid, 0.3f, pixel);

        output(ox, oy) = Halide::cast<uint8_t>(Halide::clamp(pixel * 255.0f, 0.0f, 255.0f));

        // Realize
        std::cout << std::endl << "Rendering output..." << std::endl;
        Halide::Runtime::Buffer<uint8_t> result(output_size, output_size);
        output.realize(result);

        // Save
        const char* output_path = "out/19_convolution.png";
        std::cout << "Saving to " << output_path << "..." << std::endl;

        if (save_png(result, output_path)) {
            std::cout << "Success! Filter gallery saved." << std::endl;
            std::cout << std::endl;
            std::cout << "Quadrant guide:" << std::endl;
            std::cout << "  Top-left:     Original test pattern" << std::endl;
            std::cout << "  Top-right:    Box blur (smoothing)" << std::endl;
            std::cout << "  Bottom-left:  Sobel edge detection" << std::endl;
            std::cout << "  Bottom-right: Laplacian sharpening" << std::endl;
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
