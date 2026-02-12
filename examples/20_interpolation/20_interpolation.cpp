/// @file 20_interpolation.cpp
/// @brief Example 20: Interpolation and image warping
///
/// Demonstrates:
///   - resize_bilinear for smooth scaling
///   - resize_nearest for pixel-perfect scaling
///   - zoom for scaling by factor
///   - map_coordinates for arbitrary warping
///
/// Output: out/20_interpolation.png

#include "numhalide_all.h"
#include "stbi_png.h"
#include <iostream>
#include <cmath>

using namespace numhalide;

int main(int argc, char **argv) {
    try {
        const int tile_size = 128;
        const int output_size = tile_size * 2;

        std::cout << "Interpolation and warping demonstration" << std::endl;
        std::cout << "Output size: " << output_size << "x" << output_size << std::endl;
        std::cout << std::endl;

        // Create a small test pattern (checkerboard)
        const int small_size = 32;
        Halide::Func small_input("small_input");
        Halide::Var x("x"), y("y");

        // Checkerboard pattern
        Halide::Expr checker_size = 4;
        Halide::Expr checker = ((x / checker_size) + (y / checker_size)) % 2;
        small_input(x, y) = Halide::cast<float>(checker);

        shape_t small_shape = {small_size, small_size};

        std::cout << "Creating interpolation visualizations:" << std::endl;

        // Quadrant 1: Nearest neighbor resize (4x)
        auto nearest = resize_nearest(small_input, small_shape, tile_size, tile_size, "nearest");
        std::cout << "  Top-left:     Nearest neighbor 4x upscale" << std::endl;

        // Quadrant 2: Bilinear resize (4x)
        auto bilinear = resize_bilinear(small_input, small_shape, tile_size, tile_size, "bilinear");
        std::cout << "  Top-right:    Bilinear 4x upscale" << std::endl;

        // Create a larger test image for warping
        const int warp_size = 64;
        Halide::Func warp_input("warp_input");

        // Circular gradient with center marker
        Halide::Expr wcx = warp_size / 2.0f;
        Halide::Expr wcy = warp_size / 2.0f;
        Halide::Expr wdist = Halide::sqrt(
            Halide::cast<float>((x - wcx) * (x - wcx) + (y - wcy) * (y - wcy))
        );
        Halide::Expr radial = 1.0f - wdist / (warp_size * 0.7f);
        radial = Halide::clamp(radial, 0.0f, 1.0f);

        // Add crosshair
        Halide::Expr cross = (x == warp_size/2) || (y == warp_size/2);
        warp_input(x, y) = Halide::select(cross, 0.0f, radial);

        shape_t warp_shape = {warp_size, warp_size};

        // Quadrant 3: Zoom (2x) - centered crop effect
        auto zoomed = resize_bilinear(warp_input, warp_shape, tile_size, tile_size, "zoomed");
        std::cout << "  Bottom-left:  Bilinear resize of radial gradient" << std::endl;

        // Quadrant 4: Barrel distortion using map_coordinates
        Halide::Func coords_x("coords_x"), coords_y("coords_y");

        // Barrel distortion: map (x,y) -> (x', y') where
        // x' = cx + (x - cx) * (1 + k * r^2)
        // y' = cy + (y - cy) * (1 + k * r^2)
        float k = -0.0001f;  // Distortion coefficient
        float cx = tile_size / 2.0f;
        float cy = tile_size / 2.0f;

        // Map output coords to input coords (inverse distortion)
        Halide::Expr dx = Halide::cast<float>(x) - cx;
        Halide::Expr dy = Halide::cast<float>(y) - cy;
        Halide::Expr r_sq = dx * dx + dy * dy;

        // Apply inverse barrel distortion to find source coordinates
        // For inverse: we need to find where in the source this output pixel came from
        Halide::Expr scale = 1.0f / (1.0f + k * r_sq);

        // Scale coordinates back to warp_size range
        float coord_scale = static_cast<float>(warp_size) / tile_size;
        coords_x(x, y) = (cx + dx * scale) * coord_scale;
        coords_y(x, y) = (cy + dy * scale) * coord_scale;

        auto warped = map_coordinates(warp_input, warp_shape, coords_x, coords_y, "warped");
        std::cout << "  Bottom-right: Barrel distortion warp" << std::endl;

        // Build output composite
        Halide::Func output("output");
        Halide::Var ox("ox"), oy("oy");

        Halide::Expr qx = ox / tile_size;
        Halide::Expr qy = oy / tile_size;
        Halide::Expr lx = ox % tile_size;
        Halide::Expr ly = oy % tile_size;

        Halide::Expr pixel = Halide::select(
            qy == 0 && qx == 0, nearest(lx, ly),
            Halide::select(
                qy == 0 && qx == 1, bilinear(lx, ly),
                Halide::select(
                    qy == 1 && qx == 0, zoomed(lx, ly),
                    warped(lx, ly)
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
        const char* output_path = "out/20_interpolation.png";
        std::cout << "Saving to " << output_path << "..." << std::endl;

        if (save_png(result, output_path)) {
            std::cout << "Success! Interpolation visualization saved." << std::endl;
            std::cout << std::endl;
            std::cout << "Quadrant guide:" << std::endl;
            std::cout << "  Top-left:     Nearest neighbor (pixelated, sharp edges)" << std::endl;
            std::cout << "  Top-right:    Bilinear (smooth, anti-aliased)" << std::endl;
            std::cout << "  Bottom-left:  Bilinear resize of gradient" << std::endl;
            std::cout << "  Bottom-right: Barrel distortion (pincushion effect)" << std::endl;
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
