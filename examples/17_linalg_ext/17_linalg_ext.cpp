/// @file 17_linalg_ext.cpp
/// @brief Example 17: Extended linear algebra operations
///
/// Demonstrates:
///   - norm() for vector and matrix norms
///   - triu/tril for triangular matrices
///   - det2x2, det3x3, inv2x2 for small matrix operations
///
/// Output: out/17_linalg_ext.png

#include "numhalide_all.h"
#include "stbi_png.h"
#include <iostream>
#include <cmath>

using namespace numhalide;

int main(int argc, char **argv) {
    try {
        const int size = 256;

        std::cout << "Extended linear algebra demonstration" << std::endl;
        std::cout << "Output size: " << size << "x" << size << std::endl;
        std::cout << std::endl;

        // Create a test matrix pattern
        Halide::Func matrix("matrix");
        Halide::Var x("x"), y("y");

        // Create a gradient pattern
        matrix(x, y) = Halide::cast<float>(x + y) / (2.0f * size);

        shape_t mat_shape = {size, size};

        // Apply triangular extraction
        auto upper = triu(matrix, mat_shape, 0, "upper");
        auto lower = tril(matrix, mat_shape, 0, "lower");

        // Compute Frobenius norm for demonstration
        auto mat_norm = frobenius_norm(matrix, mat_shape, "mat_norm");

        // Create a visualization showing:
        // Top-left: original matrix
        // Top-right: upper triangular
        // Bottom-left: lower triangular
        // Bottom-right: matrix with norm visualization

        const int half = size / 2;
        Halide::Func output("output");

        Halide::Expr qx = x / half;
        Halide::Expr qy = y / half;
        Halide::Expr lx = (x % half) * 2;  // Scale local coords
        Halide::Expr ly = (y % half) * 2;

        Halide::Expr pixel = Halide::select(
            qy == 0 && qx == 0, matrix(lx, ly),
            Halide::select(
                qy == 0 && qx == 1, upper(lx, ly),
                Halide::select(
                    qy == 1 && qx == 0, lower(lx, ly),
                    // Bottom-right: sharpening visualization using norm
                    // Apply edge enhancement
                    matrix(lx, ly) * 1.5f - 0.25f
                )
            )
        );

        // Add grid lines
        Halide::Expr on_grid = (x == half) || (y == half);
        pixel = Halide::select(on_grid, 0.5f, pixel);

        output(x, y) = Halide::cast<uint8_t>(Halide::clamp(pixel * 255.0f, 0.0f, 255.0f));

        // Demonstrate small matrix operations
        std::cout << "Small matrix operations:" << std::endl;

        // Create a 2x2 matrix
        Halide::Func mat2x2("mat2x2");
        // [[3, 1], [2, 4]] -> det = 3*4 - 1*2 = 10
        mat2x2(x, y) = Halide::select(
            x == 0 && y == 0, 3.0f,
            Halide::select(x == 1 && y == 0, 1.0f,
            Halide::select(x == 0 && y == 1, 2.0f, 4.0f))
        );

        auto det2 = det2x2(mat2x2, "det2");
        auto inv2 = inv2x2(mat2x2, "inv2");

        Halide::Runtime::Buffer<float> det_out(1);
        det2.realize(det_out);
        std::cout << "  2x2 matrix [[3,1],[2,4]]:" << std::endl;
        std::cout << "    Determinant: " << det_out(0) << " (expected: 10)" << std::endl;

        Halide::Runtime::Buffer<float> inv_out(2, 2);
        inv2.realize(inv_out);
        std::cout << "    Inverse: [[" << inv_out(0,0) << ", " << inv_out(1,0) << "], ["
                  << inv_out(0,1) << ", " << inv_out(1,1) << "]]" << std::endl;
        std::cout << std::endl;

        // Create a 3x3 matrix
        Halide::Func mat3x3("mat3x3");
        // [[1, 2, 3], [0, 1, 4], [5, 6, 0]] -> det = 1*(0-24) - 2*(0-20) + 3*(0-5) = -24 + 40 - 15 = 1
        mat3x3(x, y) = Halide::select(
            y == 0, Halide::select(x == 0, 1.0f, Halide::select(x == 1, 2.0f, 3.0f)),
            Halide::select(y == 1, Halide::select(x == 0, 0.0f, Halide::select(x == 1, 1.0f, 4.0f)),
            Halide::select(x == 0, 5.0f, Halide::select(x == 1, 6.0f, 0.0f)))
        );

        auto det3 = det3x3(mat3x3, "det3");
        Halide::Runtime::Buffer<float> det3_out(1);
        det3.realize(det3_out);
        std::cout << "  3x3 matrix [[1,2,3],[0,1,4],[5,6,0]]:" << std::endl;
        std::cout << "    Determinant: " << det3_out(0) << " (expected: 1)" << std::endl;
        std::cout << std::endl;

        // Compute and display Frobenius norm
        Halide::Runtime::Buffer<float> norm_out(1);
        mat_norm.realize(norm_out);
        std::cout << "  " << size << "x" << size << " gradient matrix Frobenius norm: "
                  << norm_out(0) << std::endl;
        std::cout << std::endl;

        // Realize visualization
        std::cout << "Rendering visualization..." << std::endl;
        Halide::Runtime::Buffer<uint8_t> result(size, size);
        output.realize(result);

        // Save
        const char* output_path = "out/17_linalg_ext.png";
        std::cout << "Saving to " << output_path << "..." << std::endl;

        if (save_png(result, output_path)) {
            std::cout << "Success! Linear algebra visualization saved." << std::endl;
            std::cout << std::endl;
            std::cout << "Quadrant guide:" << std::endl;
            std::cout << "  Top-left:     Original gradient matrix" << std::endl;
            std::cout << "  Top-right:    Upper triangular (triu)" << std::endl;
            std::cout << "  Bottom-left:  Lower triangular (tril)" << std::endl;
            std::cout << "  Bottom-right: Enhanced contrast" << std::endl;
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
