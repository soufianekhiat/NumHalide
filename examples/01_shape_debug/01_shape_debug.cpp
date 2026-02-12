/// @file 01_shape_debug.cpp
/// @brief Example 01: Visualize axis extents as colored bars
///
/// Demonstrates:
///   - shape_t structure for multi-dimensional shapes
///   - shape_to_string() for debugging
///   - Visualizing tensor shapes
///
/// Output: out/01_shape_debug.png

#include "numhalide_all.h"
#include "stbi_png.h"
#include <iostream>

using namespace numhalide;

int main(int argc, char **argv) {
	try {
		// Image dimensions
		const int width = 512;
		const int height = 256;
		const int channels = 3;

		std::cout << "Creating shape visualization " << width << "x" << height << "..." << std::endl;

		// Define some example shapes to visualize
		shape_t shapes[] = {
			{8},           // 1D: [8]
			{4, 8},        // 2D: [4, 8]
			{3, 4, 8},     // 3D: [3, 4, 8]
			{2, 3, 4, 8},  // 4D: [2, 3, 4, 8]
		};
		const int num_shapes = 4;

		// Print shape info
		for (int i = 0; i < num_shapes; ++i) {
			std::cout << "Shape " << i << ": " << shape_to_string(shapes[i]) << std::endl;
		}

		// Test shape operations
		shape_t a = {4, 8};
		shape_t b = {1, 8};
		shape_t broadcast_result = infer_broadcast(a, b);
		std::cout << "Broadcast " << shape_to_string(a) << " + " << shape_to_string(b)
		          << " = " << shape_to_string(broadcast_result) << std::endl;

		// Create visualization
		// Each row shows a shape, each column shows dimension extent as bar width
		Halide::Func vis("shape_vis");
		Halide::Var x("x"), y("y"), c("c");

		// Compute which shape we're in (row) and position within row
		int row_height = height / num_shapes;

		// Build expression for each shape's visualization
		// Color bars based on dimension: R=dim0, G=dim1, B=dim2
		Halide::Expr result = Halide::cast<uint8_t>(0);

		for (int s = num_shapes - 1; s >= 0; --s) {
			int y_start = s * row_height;
			int y_end = (s + 1) * row_height;

			// Create colored bars for each dimension
			Halide::Expr in_row = (y >= y_start) && (y < y_end);

			// Bar parameters
			int bar_width = width / 8;  // Max 8 dimensions
			int scale = 16;  // Pixels per unit extent

			// For each dimension, draw a bar
			for (int d = 0; d < shapes[s].rank; ++d) {
				int bar_x_start = d * bar_width;
				int bar_x_end = bar_x_start + shapes[s].extents[d] * scale;
				int bar_y_start = y_start + 10;
				int bar_y_end = y_end - 10;

				Halide::Expr in_bar = (x >= bar_x_start) && (x < bar_x_end) &&
				                      (y >= bar_y_start) && (y < bar_y_end);

				// Color based on dimension index
				uint8_t colors[8][3] = {
					{255, 100, 100},  // dim 0: red
					{100, 255, 100},  // dim 1: green
					{100, 100, 255},  // dim 2: blue
					{255, 255, 100},  // dim 3: yellow
					{255, 100, 255},  // dim 4: magenta
					{100, 255, 255},  // dim 5: cyan
					{200, 200, 200},  // dim 6: gray
					{255, 200, 100},  // dim 7: orange
				};

				Halide::Expr color_val = Halide::select(
					c == 0, Halide::cast<uint8_t>(colors[d][0]),
					Halide::select(c == 1, Halide::cast<uint8_t>(colors[d][1]),
					               Halide::cast<uint8_t>(colors[d][2])));

				result = Halide::select(in_row && in_bar, color_val, result);
			}
		}

		vis(x, y, c) = result;

		// Realize the result
		std::cout << "Computing visualization..." << std::endl;
		Halide::Runtime::Buffer<uint8_t> output(width, height, channels);
		vis.realize(output);

		// Save to PNG
		const char* output_path = "out/01_shape_debug.png";
		std::cout << "Saving to " << output_path << "..." << std::endl;

		if (save_png(output, output_path)) {
			std::cout << "Success! Shape visualization saved to " << output_path << std::endl;
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
