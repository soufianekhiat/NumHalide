/// @file 04_broadcasting.cpp
/// @brief Example 04: Demonstrate broadcasting and elementwise operations
///
/// Demonstrates:
///   - broadcast_to() for expanding dimensions
///   - add(), mul() with automatic broadcasting
///   - Per-channel color manipulation
///
/// Output: out/04_broadcasting.png

#include "numhalide_all.h"
#include "stbi_png.h"
#include <iostream>

using namespace numhalide;

int main(int argc, char **argv) {
	try {
		// Create an image and apply per-channel color bias using broadcasting
		const int width = 256;
		const int height = 256;
		const int channels = 3;

		std::cout << "Creating broadcasting visualization " << width << "x" << height << "..." << std::endl;

		Halide::Var x("x"), y("y"), c("c");

		// Create a base grayscale gradient (radial)
		Halide::Func base_gray("base_gray");
		Halide::Expr cx = (x - width / 2.0f) / (width / 2.0f);
		Halide::Expr cy = (y - height / 2.0f) / (height / 2.0f);
		Halide::Expr dist = Halide::sqrt(cx * cx + cy * cy);
		base_gray(x, y) = Halide::clamp(1.0f - dist, 0.0f, 1.0f);

		// Create RGB version of base (same value in all channels)
		// Shape: [height, width, 3]
		Halide::Func base_rgb("base_rgb");
		base_rgb(x, y, c) = base_gray(x, y);

		// Create per-channel bias: [R_bias, G_bias, B_bias]
		// This is a 1D array of shape [3] that will be broadcast to [height, width, 3]
		// R: boost center, G: neutral, B: reduce center
		Halide::Func channel_bias("channel_bias");
		channel_bias(c) = Halide::select(
			c == 0, 0.3f,   // R: add 0.3
			Halide::select(c == 1, 0.0f,   // G: no change
			               -0.2f));        // B: subtract 0.2

		// Create per-channel scale
		Halide::Func channel_scale("channel_scale");
		channel_scale(c) = Halide::select(
			c == 0, 1.2f,   // R: scale by 1.2
			Halide::select(c == 1, 0.8f,   // G: scale by 0.8
			               1.0f));         // B: scale by 1.0

		// Apply broadcasting manually:
		// result = base_rgb * channel_scale + channel_bias
		// The channel dimension broadcasts across width and height
		Halide::Func result("result");
		result(x, y, c) = Halide::clamp(
			base_rgb(x, y, c) * channel_scale(c) + channel_bias(c),
			0.0f, 1.0f
		);

		// Convert to uint8
		Halide::Func output_func("output_func");
		output_func(x, y, c) = Halide::cast<uint8_t>(result(x, y, c) * 255.0f);

		// Realize the result
		std::cout << "Computing..." << std::endl;
		Halide::Runtime::Buffer<uint8_t> output(width, height, channels);
		output_func.realize(output);

		// Save to PNG
		const char* output_path = "out/04_broadcasting.png";
		std::cout << "Saving to " << output_path << "..." << std::endl;

		if (save_png(output, output_path)) {
			std::cout << "Success! Broadcasting visualization saved to " << output_path << std::endl;
			std::cout << "\nDemonstrates per-channel operations:" << std::endl;
			std::cout << "  - Base: radial gradient (grayscale)" << std::endl;
			std::cout << "  - R channel: scaled by 1.2, bias +0.3" << std::endl;
			std::cout << "  - G channel: scaled by 0.8, no bias" << std::endl;
			std::cout << "  - B channel: scaled by 1.0, bias -0.2" << std::endl;
			std::cout << "\nResult: warm-tinted radial gradient (more red, less blue)" << std::endl;
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
