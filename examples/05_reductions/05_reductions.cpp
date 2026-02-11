/// @file 05_reductions.cpp
/// @brief Example 05: Reduction operations (sum, mean, min, max)
///
/// Demonstrates:
///   - reduce_sum() for summing along axes
///   - reduce_mean() for computing averages
///   - reduce_min() / reduce_max() for finding extrema
///   - keepdims parameter for preserving dimensions
///   - Visualizing per-row reductions as a heatmap
///
/// Output: out/05_reductions.png

#include "numhalide_all.h"
#include "stbi_png.h"
#include <iostream>

using namespace numhalide;

int main(int argc, char **argv) {
	try {
		const int width = 256;
		const int height = 256;

		std::cout << "Creating " << width << "x" << height << " input image..." << std::endl;

		// Create a 2D input with a gradient pattern
		// Values range from 0 to 255 based on position
		shape_t s_input = { height, width };
		Halide::Func input("input");
		Halide::Var x, y;
		// Create diagonal gradient: brighter toward bottom-right
		input(x, y) = Halide::cast<float>((x + y) * 255 / (width + height - 2));

		// Compute per-row mean (reduce along axis 1 = columns)
		std::cout << "Computing per-row mean..." << std::endl;
		Halide::Func row_mean = reduce_mean(input, s_input, {1}, true, "row_mean");

		// Broadcast back to full image size for visualization
		shape_t s_row = { height, 1 };
		Halide::Func row_mean_img = broadcast_to(row_mean, s_row, s_input, "row_mean_broadcast");

		// Compute per-column mean (reduce along axis 0 = rows)
		std::cout << "Computing per-column mean..." << std::endl;
		Halide::Func col_mean = reduce_mean(input, s_input, {0}, true, "col_mean");

		// Broadcast back for visualization
		shape_t s_col = { 1, width };
		Halide::Func col_mean_img = broadcast_to(col_mean, s_col, s_input, "col_mean_broadcast");

		// Create a composite image showing:
		// Top-left: original, Top-right: row means
		// Bottom-left: col means, Bottom-right: global mean
		std::cout << "Computing global statistics..." << std::endl;
		Halide::Func global_mean = reduce_mean(input, s_input, "global_mean");

		// Create output image (2x2 grid of 128x128 tiles)
		const int tile = 128;
		const int out_width = tile * 2;
		const int out_height = tile * 2;

		Halide::Func output("output");
		Halide::Var ox, oy;

		// Scale coordinates to sample from the input
		Halide::Expr sx = (ox % tile) * width / tile;
		Halide::Expr sy = (oy % tile) * height / tile;

		// Determine which quadrant we're in
		Halide::Expr quadrant_x = ox / tile;
		Halide::Expr quadrant_y = oy / tile;

		// Sample appropriate source for each quadrant
		Halide::Expr val = Halide::select(
			quadrant_y == 0 && quadrant_x == 0, input(sx, sy),           // Top-left: original
			quadrant_y == 0 && quadrant_x == 1, row_mean_img(sx, sy),    // Top-right: row means
			quadrant_y == 1 && quadrant_x == 0, col_mean_img(sx, sy),    // Bottom-left: col means
			global_mean(0)                                               // Bottom-right: global mean
		);

		output(ox, oy) = Halide::cast<uint8_t>(Halide::clamp(val, 0.0f, 255.0f));

		// Realize
		std::cout << "Rendering output..." << std::endl;
		Halide::Runtime::Buffer<uint8_t> result(out_width, out_height);
		output.realize(result);

		// Save
		const char* output_path = "out/05_reductions.png";
		std::cout << "Saving to " << output_path << "..." << std::endl;

		if (save_png(result, output_path)) {
			std::cout << "Success! Reduction visualization saved." << std::endl;
			std::cout << "  Top-left: Original gradient" << std::endl;
			std::cout << "  Top-right: Per-row mean (horizontal bands)" << std::endl;
			std::cout << "  Bottom-left: Per-column mean (vertical bands)" << std::endl;
			std::cout << "  Bottom-right: Global mean (solid color)" << std::endl;
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
