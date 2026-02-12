/// @file 08_matmul.cpp
/// @brief Example 08: Matrix multiplication and linear algebra
///
/// Demonstrates:
///   - matmul() for matrix multiplication
///   - outer() for outer products
///   - matvec() for matrix-vector multiplication
///   - Visualizing matrices as heatmaps
///
/// Output: out/08_matmul.png

#include "numhalide_all.h"
#include "stbi_png.h"
#include <iostream>

using namespace numhalide;

int main(int argc, char **argv) {
	try {
		const int tile_size = 128;
		const int mat_size = 32;  // Size of matrices to multiply

		std::cout << "Computing matrix operations..." << std::endl;

		// Create two random-ish matrices for visualization
		// Matrix A: gradient pattern
		shape_t smat = { mat_size, mat_size };
		Halide::Func mat_a("mat_a");
		Halide::Var x, y;
		mat_a(x, y) = Halide::cast<float>(x + y) / (2.0f * mat_size);

		// Matrix B: another gradient pattern (perpendicular)
		Halide::Func mat_b("mat_b");
		mat_b(x, y) = Halide::cast<float>(mat_size - x + y) / (2.0f * mat_size);

		// Compute C = A @ B
		std::cout << "  Computing A @ B..." << std::endl;
		Halide::Func mat_c = matmul(mat_a, smat, mat_b, smat);

		// Compute outer product of two vectors
		std::cout << "  Computing outer product..." << std::endl;
		shape_t svec = { mat_size };
		Halide::Func vec_a("vec_a");
		Halide::Func vec_b("vec_b");
		vec_a(x) = Halide::cast<float>(x) / mat_size;
		vec_b(x) = Halide::cast<float>(mat_size - x) / mat_size;
		Halide::Func outer_prod = outer(vec_a, svec, vec_b, svec);

		// Create output: 2x2 grid showing A, B, C=A@B, outer(a,b)
		const int out_width = tile_size * 2;
		const int out_height = tile_size * 2;

		Halide::Func output("output");
		Halide::Var ox, oy;

		// Local coordinates within tile, scaled to matrix size
		Halide::Expr tx = (ox % tile_size) * mat_size / tile_size;
		Halide::Expr ty = (oy % tile_size) * mat_size / tile_size;

		// Quadrant selection
		Halide::Expr qx = ox / tile_size;
		Halide::Expr qy = oy / tile_size;

		// Get values from each matrix (normalized to [0,1])
		// Top-left: Matrix A
		// Top-right: Matrix B
		// Bottom-left: C = A @ B (need to normalize)
		// Bottom-right: outer(a, b)

		Halide::Expr a_val = mat_a(tx, ty);
		Halide::Expr b_val = mat_b(tx, ty);
		Halide::Expr c_val = mat_c(tx, ty) / mat_size;  // Normalize by matrix size
		Halide::Expr outer_val = outer_prod(tx, ty);

		Halide::Expr val = Halide::select(
			qy == 0 && qx == 0, a_val,
			qy == 0 && qx == 1, b_val,
			qy == 1 && qx == 0, c_val,
			outer_val
		);

		// Apply colormap (simple grayscale to blue-red gradient)
		Halide::Expr normalized = Halide::clamp(val, 0.0f, 1.0f);

		// Simple blue-white-red colormap
		Halide::Expr r = Halide::select(normalized > 0.5f, 1.0f, normalized * 2.0f);
		Halide::Expr g = 1.0f - Halide::abs(normalized - 0.5f) * 2.0f;
		Halide::Expr b_col = Halide::select(normalized < 0.5f, 1.0f, (1.0f - normalized) * 2.0f);

		Halide::Func colored("colored");
		Halide::Var c;
		colored(ox, oy, c) = Halide::cast<uint8_t>(Halide::clamp(
			Halide::select(c == 0, r, c == 1, g, b_col) * 255.0f, 0.0f, 255.0f));

		// Realize
		std::cout << "Rendering output..." << std::endl;
		Halide::Runtime::Buffer<uint8_t> result(out_width, out_height, 3);
		colored.realize(result);

		// Save
		const char* output_path = "out/08_matmul.png";
		std::cout << "Saving to " << output_path << "..." << std::endl;

		// stb expects interleaved, but Halide buffer is planar
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
			std::cout << "Success! Matrix visualization saved." << std::endl;
			std::cout << "  Top-left: Matrix A (gradient)" << std::endl;
			std::cout << "  Top-right: Matrix B (gradient)" << std::endl;
			std::cout << "  Bottom-left: C = A @ B (matrix product)" << std::endl;
			std::cout << "  Bottom-right: outer(a, b) (outer product)" << std::endl;
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
