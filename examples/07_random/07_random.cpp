/// @file 07_random.cpp
/// @brief Example 07: Random number generation
///
/// Demonstrates:
///   - rand_uniform() for uniform random values
///   - rand_normal() for Gaussian noise
///   - rand_int() for random integers
///   - Seed-based reproducibility
///   - Creating procedural noise textures
///
/// Output: out/07_random.png

#include "numhalide_all.h"
#include "stbi_png.h"
#include <iostream>

using namespace numhalide;

int main(int argc, char **argv) {
	try {
		const int tile_size = 128;
		const int out_width = tile_size * 2;
		const int out_height = tile_size * 2;

		std::cout << "Generating random noise patterns..." << std::endl;

		// 1. Uniform noise [0, 1)
		std::cout << "  Creating uniform noise..." << std::endl;
		shape_t s_tile = { tile_size, tile_size };
		Halide::Func uniform_noise = rand_uniform(Halide::Float(32), s_tile, 42, "uniform");

		// 2. Normal (Gaussian) noise - centered at 0.5, stddev 0.15
		std::cout << "  Creating Gaussian noise..." << std::endl;
		Halide::Func normal_noise = rand_normal(Halide::Float(32), s_tile, 0.5f, 0.15f, 123, "normal");

		// 3. Binary noise (random 0 or 1)
		std::cout << "  Creating binary noise..." << std::endl;
		Halide::Func binary_noise = rand_int(Halide::Int(32), s_tile, 0, 2, 456, "binary");

		// 4. Colored noise - different seeds for R, G, B channels
		std::cout << "  Creating colored noise..." << std::endl;
		Halide::Func r_noise = rand_uniform(Halide::Float(32), s_tile, 100, "r_noise");
		Halide::Func g_noise = rand_uniform(Halide::Float(32), s_tile, 200, "g_noise");
		Halide::Func b_noise = rand_uniform(Halide::Float(32), s_tile, 300, "b_noise");

		// Compose output (2x2 grid)
		Halide::Func output("output");
		Halide::Var ox, oy, c;

		// Local coordinates within tile
		Halide::Expr tx = ox % tile_size;
		Halide::Expr ty = oy % tile_size;

		// Quadrant selection
		Halide::Expr qx = ox / tile_size;
		Halide::Expr qy = oy / tile_size;

		// Get values from each noise type
		Halide::Expr uniform_val = uniform_noise(tx, ty) * 255.0f;
		Halide::Expr normal_val = Halide::clamp(normal_noise(tx, ty), 0.0f, 1.0f) * 255.0f;
		Halide::Expr binary_val = Halide::cast<float>(binary_noise(tx, ty)) * 255.0f;

		// Select based on quadrant
		// Top-left: Uniform grayscale
		// Top-right: Normal/Gaussian grayscale
		// Bottom-left: Binary noise
		// Bottom-right: Colored RGB noise
		Halide::Expr is_color_quad = (qy == 1 && qx == 1);

		Halide::Expr gray_val = Halide::select(
			qy == 0 && qx == 0, uniform_val,
			qy == 0 && qx == 1, normal_val,
			binary_val
		);

		// For the color quadrant, use different noise per channel
		Halide::Expr r_val = r_noise(tx, ty) * 255.0f;
		Halide::Expr g_val = g_noise(tx, ty) * 255.0f;
		Halide::Expr b_val = b_noise(tx, ty) * 255.0f;

		Halide::Expr channel_val = Halide::select(
			c == 0, Halide::select(is_color_quad, r_val, gray_val),
			c == 1, Halide::select(is_color_quad, g_val, gray_val),
			Halide::select(is_color_quad, b_val, gray_val)
		);

		output(ox, oy, c) = Halide::cast<uint8_t>(Halide::clamp(channel_val, 0.0f, 255.0f));

		// Realize
		std::cout << "Rendering output..." << std::endl;
		Halide::Runtime::Buffer<uint8_t> result(out_width, out_height, 3);
		output.realize(result);

		// Save (need to handle RGB properly for stb)
		const char* output_path = "out/07_random.png";
		std::cout << "Saving to " << output_path << "..." << std::endl;

		// stb expects interleaved, but Halide buffer is planar
		// Create interleaved buffer
		std::vector<uint8_t> interleaved(out_width * out_height * 3);
		for (int y = 0; y < out_height; ++y) {
			for (int x = 0; x < out_width; ++x) {
				int idx = (y * out_width + x) * 3;
				interleaved[idx + 0] = result(x, y, 0);
				interleaved[idx + 1] = result(x, y, 1);
				interleaved[idx + 2] = result(x, y, 2);
			}
		}

		if (stbi_write_png(output_path, out_width, out_height, 3, interleaved.data(), out_width * 3)) {
			std::cout << "Success! Random noise patterns saved." << std::endl;
			std::cout << "  Top-left: Uniform noise" << std::endl;
			std::cout << "  Top-right: Gaussian noise" << std::endl;
			std::cout << "  Bottom-left: Binary noise" << std::endl;
			std::cout << "  Bottom-right: RGB colored noise" << std::endl;
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
