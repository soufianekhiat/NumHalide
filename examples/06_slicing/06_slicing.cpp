/// @file 06_slicing.cpp
/// @brief Example 06: Slicing, transpose, and axis manipulation
///
/// Demonstrates:
///   - slice() for extracting sub-regions
///   - transpose() for swapping dimensions
///   - moveaxis() for rearranging dimensions
///   - expand_dims() and squeeze() for adding/removing singleton dims
///
/// Output: out/06_slicing.png

#include "numhalide_all.h"
#include "stbi_png.h"
#include <iostream>

using namespace numhalide;

int main(int argc, char **argv) {
	try {
		const int size = 128;

		std::cout << "Creating " << size << "x" << size << " input images..." << std::endl;

		// Create a gradient image
		shape_t s_input = { size, size };
		Halide::Func gradient("gradient");
		Halide::Var x, y;
		// Diagonal gradient
		gradient(x, y) = Halide::cast<float>((x + y) * 255 / (2 * size - 2));

		// Create a checkerboard pattern
		Halide::Func checker("checker");
		checker(x, y) = Halide::cast<float>(
			Halide::select((x / 16 + y / 16) % 2 == 0, 200.0f, 55.0f)
		);

		// 1. Slice: Extract top-left quadrant
		std::cout << "Slicing top-left quadrant..." << std::endl;
		shape_t s_half = { size / 2, size / 2 };
		Halide::Func slice_y = slice(gradient, s_input, 0, 0, size / 2, 1, "slice_rows");
		Halide::Func top_left = slice(slice_y, { size / 2, size }, 1, 0, size / 2, 1, "slice_cols");

		// 2. Transpose: Swap x and y
		std::cout << "Transposing image..." << std::endl;
		Halide::Func transposed = transpose(checker, s_input, "transposed");
		shape_t s_transposed = infer_transpose(s_input, {1, 0});

		// 3. Stride slice: Every 2nd pixel (downscale)
		std::cout << "Downscaling with stride..." << std::endl;
		Halide::Func stride_y = slice(gradient, s_input, 0, 0, size, 2, "stride_rows");
		shape_t s_stride_y = infer_slice(s_input, 0, 0, size, 2);
		Halide::Func downscaled = slice(stride_y, s_stride_y, 1, 0, size, 2, "stride_cols");
		shape_t s_downscaled = infer_slice(s_stride_y, 1, 0, size, 2);

		// 4. Create a composite output (2x2 grid of 128x128 tiles)
		const int tile = 128;
		const int out_width = tile * 2;
		const int out_height = tile * 2;

		Halide::Func output("output");
		Halide::Var ox, oy;

		// Local coordinates within tile
		Halide::Expr tx = ox % tile;
		Halide::Expr ty = oy % tile;

		// Quadrant selection
		Halide::Expr qx = ox / tile;
		Halide::Expr qy = oy / tile;

		// Scale coordinates for the downscaled quadrant
		Halide::Expr half_x = tx / 2;
		Halide::Expr half_y = ty / 2;

		// Compose output:
		// Top-left: Original gradient
		// Top-right: Transposed checker
		// Bottom-left: Sliced top-left quadrant (scaled up 2x)
		// Bottom-right: Downscaled gradient (tiled 2x)
		Halide::Expr val = Halide::select(
			qy == 0 && qx == 0, gradient(tx, ty),           // Original
			qy == 0 && qx == 1, transposed(tx, ty),         // Transposed checker
			qy == 1 && qx == 0, top_left(tx / 2, ty / 2),   // Sliced (scaled up)
			downscaled(tx % s_downscaled.extents[1], ty % s_downscaled.extents[0])  // Downscaled (tiled)
		);

		output(ox, oy) = Halide::cast<uint8_t>(Halide::clamp(val, 0.0f, 255.0f));

		// Realize
		std::cout << "Rendering output..." << std::endl;
		Halide::Runtime::Buffer<uint8_t> result(out_width, out_height);
		output.realize(result);

		// Save
		const char* output_path = "out/06_slicing.png";
		std::cout << "Saving to " << output_path << "..." << std::endl;

		if (save_png(result, output_path)) {
			std::cout << "Success! Slicing demonstration saved." << std::endl;
			std::cout << "  Top-left: Original gradient" << std::endl;
			std::cout << "  Top-right: Transposed checkerboard" << std::endl;
			std::cout << "  Bottom-left: Sliced quadrant (scaled 2x)" << std::endl;
			std::cout << "  Bottom-right: Downscaled gradient (tiled)" << std::endl;
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
