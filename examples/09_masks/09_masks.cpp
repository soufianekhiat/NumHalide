/// @file 09_masks.cpp
/// @brief Example 09: Masking, where(), clip(), and type conversion
///
/// Demonstrates:
///   - where() for conditional selection
///   - clip() for value clamping
///   - astype() for type conversion
///   - Creating masked blending effects
///
/// Output: out/09_masks.png

#include "numhalide_all.h"
#include "stbi_png.h"
#include <iostream>

using namespace numhalide;

int main(int argc, char **argv) {
	try {
		const int tile_size = 128;
		const int out_width = tile_size * 2;
		const int out_height = tile_size * 2;

		std::cout << "Creating masked images..." << std::endl;

		shape_t s_tile = { tile_size, tile_size };

		// Create a gradient image
		Halide::Func gradient("gradient");
		Halide::Var x, y;
		gradient(x, y) = Halide::cast<float>(x + y) / (2.0f * tile_size);

		// Create a checkerboard pattern
		Halide::Func checker("checker");
		int check_size = 16;
		checker(x, y) = Halide::cast<float>(
			Halide::select((x / check_size + y / check_size) % 2 == 0, 1.0f, 0.0f)
		);

		// Create a circular mask
		Halide::Func mask("mask");
		Halide::Expr cx = tile_size / 2;
		Halide::Expr cy = tile_size / 2;
		Halide::Expr radius = tile_size / 3;
		Halide::Expr dist_sq = (x - cx) * (x - cx) + (y - cy) * (y - cy);
		mask(x, y) = Halide::select(dist_sq < radius * radius, 1, 0);

		// 1. Masked blend: show checker inside circle, gradient outside
		std::cout << "  Creating masked blend..." << std::endl;
		Halide::Func masked = where(mask, checker, gradient, s_tile, "masked_blend");

		// 2. Clipped gradient: values clamped to [0.25, 0.75]
		std::cout << "  Creating clipped gradient..." << std::endl;
		Halide::Func clipped = clip(gradient, s_tile, 0.25f, 0.75f, "clipped");

		// 3. Inverted with abs: |0.5 - gradient| creates a V-shape pattern
		std::cout << "  Creating V-shape pattern..." << std::endl;
		Halide::Func centered("centered");
		centered(x, y) = Halide::cast<float>(0.5f) - gradient(x, y);
		Halide::Func v_pattern = numhalide::nh_abs(centered, s_tile, "v_pattern");

		// 4. Sign-based: shows -1, 0, +1 regions
		std::cout << "  Creating sign pattern..." << std::endl;
		Halide::Func shifted("shifted");
		shifted(x, y) = gradient(x, y) - 0.5f;  // Range: -0.5 to 0.5
		Halide::Func sign_pattern = sign(shifted, s_tile, "sign_pattern");
		// Normalize to [0, 1] for display
		Halide::Func sign_display("sign_display");
		sign_display(x, y) = (sign_pattern(x, y) + 1.0f) / 2.0f;

		// Composite output
		Halide::Func output("output");
		Halide::Var ox, oy;

		Halide::Expr tx = ox % tile_size;
		Halide::Expr ty = oy % tile_size;
		Halide::Expr qx = ox / tile_size;
		Halide::Expr qy = oy / tile_size;

		Halide::Expr val = Halide::select(
			qy == 0 && qx == 0, masked(tx, ty),
			qy == 0 && qx == 1, clipped(tx, ty),
			qy == 1 && qx == 0, v_pattern(tx, ty) * 2.0f,  // Scale for visibility
			sign_display(tx, ty)
		);

		output(ox, oy) = Halide::cast<uint8_t>(Halide::clamp(val * 255.0f, 0.0f, 255.0f));

		// Realize
		std::cout << "Rendering output..." << std::endl;
		Halide::Runtime::Buffer<uint8_t> result(out_width, out_height);
		output.realize(result);

		// Save
		const char* output_path = "out/09_masks.png";
		std::cout << "Saving to " << output_path << "..." << std::endl;

		if (save_png(result, output_path)) {
			std::cout << "Success! Mask operations visualization saved." << std::endl;
			std::cout << "  Top-left: Masked blend (checker inside circle)" << std::endl;
			std::cout << "  Top-right: Clipped gradient [0.25, 0.75]" << std::endl;
			std::cout << "  Bottom-left: Absolute value (V-pattern)" << std::endl;
			std::cout << "  Bottom-right: Sign function (-1/0/+1)" << std::endl;
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
