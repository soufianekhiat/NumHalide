/// @file 00_gradient.cpp
/// @brief Example 00: Generate a 2D gradient using meshgrid
///
/// Demonstrates:
///   - linspace() for creating coordinate arrays
///   - meshgrid() for creating 2D coordinate grids
///   - Basic Func operations
///   - Saving output to PNG using stb_image_write
///
/// Output: out/00_gradient.png

#include "numhalide_all.h"
#include "stbi_png.h"
#include <iostream>

using namespace numhalide;

int main(int argc, char **argv) {
	try {
		// Image dimensions
		const int width = 256;
		const int height = 256;

		std::cout << "Creating " << width << "x" << height << " gradient..." << std::endl;

		// Create coordinate arrays from 0 to 1
		Halide::Func xs = linspace(Halide::Float(32), 0.0f, 1.0f, width, true, 0, "xs");
		Halide::Func ys = linspace(Halide::Float(32), 0.0f, 1.0f, height, true, 0, "ys");

		// Create 2D meshgrid
		std::vector<Halide::Func> grids = meshgrid(Halide::Float(32), {xs, ys}, "meshgrid");
		Halide::Func x_grid = grids[0];
		Halide::Func y_grid = grids[1];

		// Create gradient: combine x and y coordinates
		// gradient(x,y) = sqrt(x^2 + y^2) normalized to [0,1]
		Halide::Func gradient("gradient");
		Halide::Var x, y;
		gradient(x, y) = Halide::sqrt(x_grid(x, y) * x_grid(x, y) +
		                               y_grid(x, y) * y_grid(x, y)) / Halide::sqrt(2.0f);

		// Convert to 8-bit for PNG output
		Halide::Func gradient_8bit("gradient_8bit");
		gradient_8bit(x, y) = Halide::cast<uint8_t>(gradient(x, y) * 255.0f);

		// Realize the result
		std::cout << "Computing gradient..." << std::endl;
		Halide::Runtime::Buffer<uint8_t> output(width, height);
		gradient_8bit.realize(output);

		// Save to PNG
		const char* output_path = "out/00_gradient.png";
		std::cout << "Saving to " << output_path << "..." << std::endl;

		if (save_png(output, output_path)) {
			std::cout << "Success! Gradient saved to " << output_path << std::endl;
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
