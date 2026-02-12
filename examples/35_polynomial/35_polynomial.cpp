/// @file 35_polynomial.cpp
/// @brief Example 35: Polynomial functions visualization
///
/// Demonstrates:
///   - polynomial curve y = x^3 - x as 2D height field
///   - Chebyshev T_5(x) oscillating wave pattern
///   - Legendre P_4(x) wave pattern
///   - polynomial lens distortion (barrel distortion on grid)
///
/// Output: out/35_polynomial.png

#include "numhalide_all.h"
#include "stbi_png.h"
#include <iostream>

using namespace numhalide;

int main(int argc, char **argv) {
    try {
        const int half = 256;
        const int size = half * 2;  // 512x512

        std::cout << "Polynomial functions demonstration" << std::endl;
        std::cout << "Output size: " << size << "x" << size << std::endl;
        std::cout << std::endl;

        Halide::Var ox("ox"), oy("oy");
        Halide::Func output("output");

        Halide::Expr qx = ox / half;
        Halide::Expr qy = oy / half;
        Halide::Expr lx = ox % half;
        Halide::Expr ly = oy % half;

        // Normalized coords [0,1)
        Halide::Expr u = Halide::cast<float>(lx) / half;
        Halide::Expr v = Halide::cast<float>(ly) / half;

        // --- Top-left: polynomial curve y = x^3 - x as 2D height field ---
        // Map u to [-1.5, 1.5], v to [-1.0, 1.0]
        Halide::Expr px = u * 3.0f - 1.5f;
        Halide::Expr py = v * 2.0f - 1.0f;
        // Evaluate y = x^3 - x
        Halide::Expr curve_y = px * px * px - px;
        // Distance of py from the curve value, rendered as a height field
        Halide::Expr curve_dist = Halide::abs(py - curve_y);
        Halide::Expr poly_val = Halide::clamp(1.0f - curve_dist * 3.0f, 0.0f, 1.0f);

        // --- Top-right: Chebyshev T_5(x) oscillating wave pattern ---
        // T_5(x) = 16x^5 - 20x^3 + 5x, map u to [-1,1]
        Halide::Expr cx = u * 2.0f - 1.0f;
        Halide::Expr cy = v * 2.0f - 1.0f;
        Halide::Expr cx2 = cx * cx;
        Halide::Expr cx3 = cx2 * cx;
        Halide::Expr cx5 = cx3 * cx2;
        Halide::Expr t5 = 16.0f * cx5 - 20.0f * cx3 + 5.0f * cx;
        // Draw the curve: distance of cy from T_5(cx)
        Halide::Expr cheb_dist = Halide::abs(cy - t5);
        Halide::Expr cheb_val = Halide::clamp(1.0f - cheb_dist * 4.0f, 0.0f, 1.0f);

        // --- Bottom-left: Legendre P_4(x) wave pattern ---
        // P_4(x) = (35x^4 - 30x^2 + 3) / 8
        Halide::Expr lxn = u * 2.0f - 1.0f;  // [-1, 1]
        Halide::Expr lyn = v * 2.0f - 1.0f;
        Halide::Expr lxn2 = lxn * lxn;
        Halide::Expr lxn4 = lxn2 * lxn2;
        Halide::Expr p4 = (35.0f * lxn4 - 30.0f * lxn2 + 3.0f) / 8.0f;
        Halide::Expr leg_dist = Halide::abs(lyn - p4);
        Halide::Expr leg_val = Halide::clamp(1.0f - leg_dist * 4.0f, 0.0f, 1.0f);

        // --- Bottom-right: polynomial lens distortion (barrel distortion on grid) ---
        // Centered coordinates
        Halide::Expr du = u * 2.0f - 1.0f;
        Halide::Expr dv = v * 2.0f - 1.0f;
        Halide::Expr r2 = du * du + dv * dv;
        // Barrel distortion: r' = r * (1 + k1*r^2 + k2*r^4)
        Halide::Expr k1 = -0.3f;
        Halide::Expr k2 = 0.1f;
        Halide::Expr distort = 1.0f + k1 * r2 + k2 * r2 * r2;
        Halide::Expr dist_u = du * distort;
        Halide::Expr dist_v = dv * distort;
        // Map back to [0, half) and create a grid pattern
        Halide::Expr grid_u = (dist_u + 1.0f) * 0.5f * half;
        Halide::Expr grid_v = (dist_v + 1.0f) * 0.5f * half;
        Halide::Expr grid_period = 16.0f;
        Halide::Expr grid_mu = grid_u - Halide::floor(grid_u / grid_period) * grid_period;
        Halide::Expr grid_mv = grid_v - Halide::floor(grid_v / grid_period) * grid_period;
        Halide::Expr on_line = (grid_mu < 1.5f) || (grid_mv < 1.5f);
        Halide::Expr barrel_val = Halide::select(on_line, 1.0f, 0.15f);
        // Fade out outside unit circle
        Halide::Expr r_dist = Halide::sqrt(r2);
        barrel_val = Halide::select(r_dist > 1.0f, 0.0f, barrel_val);

        // Compose quadrants
        Halide::Expr pixel = Halide::select(
            qy == 0 && qx == 0, poly_val,
            qy == 0 && qx == 1, cheb_val,
            qy == 1 && qx == 0, leg_val,
            barrel_val
        );

        // Grid lines
        Halide::Expr on_grid = (ox == half) || (oy == half);
        pixel = Halide::select(on_grid, 0.4f, pixel);

        output(ox, oy) = Halide::cast<uint8_t>(Halide::clamp(pixel * 255.0f, 0.0f, 255.0f));

        std::cout << "Rendering..." << std::endl;
        Halide::Runtime::Buffer<uint8_t> result(size, size);
        output.realize(result);

        const char* output_path = "out/35_polynomial.png";
        std::cout << "Saving to " << output_path << "..." << std::endl;

        if (save_png(result, output_path)) {
            std::cout << "Success!" << std::endl;
            std::cout << std::endl;
            std::cout << "Quadrant guide:" << std::endl;
            std::cout << "  Top-left:     polynomial curve y = x^3 - x height field" << std::endl;
            std::cout << "  Top-right:    Chebyshev T_5(x) oscillating wave" << std::endl;
            std::cout << "  Bottom-left:  Legendre P_4(x) wave pattern" << std::endl;
            std::cout << "  Bottom-right: barrel distortion on grid" << std::endl;
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
