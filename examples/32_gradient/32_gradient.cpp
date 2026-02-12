/// @file 32_gradient.cpp
/// @brief Example 32: Image gradient operations
///
/// Demonstrates:
///   - smooth image (sum of Gaussians)
///   - gradient magnitude (edge strength)
///   - Laplacian-like blob detector
///   - gradient direction as grayscale
///
/// Output: out/32_gradient.png

#include "numhalide_all.h"
#include "stbi_png.h"
#include <iostream>

using namespace numhalide;

int main(int argc, char **argv) {
    try {
        const int half = 256;
        const int size = half * 2;  // 512x512

        std::cout << "Image gradient operations demonstration" << std::endl;
        std::cout << "Output size: " << size << "x" << size << std::endl;
        std::cout << std::endl;

        constexpr float pi = 3.14159265358979323846f;

        Halide::Var ox("ox"), oy("oy");
        Halide::Func output("output");

        Halide::Expr qx = ox / half;
        Halide::Expr qy = oy / half;
        Halide::Expr lx = ox % half;
        Halide::Expr ly = oy % half;

        // Normalized coords [0,1)
        Halide::Expr u = Halide::cast<float>(lx) / half;
        Halide::Expr v = Halide::cast<float>(ly) / half;

        // --- Helper: sum of Gaussians as a smooth test image ---
        // Place several Gaussian blobs at different positions and sizes
        // Blob 1: center (0.3, 0.3), sigma 0.12
        Halide::Expr dx1 = u - 0.3f;
        Halide::Expr dy1 = v - 0.3f;
        Halide::Expr g1 = Halide::pow(2.0f, -(dx1 * dx1 + dy1 * dy1) / (2.0f * 0.12f * 0.12f));

        // Blob 2: center (0.7, 0.4), sigma 0.08
        Halide::Expr dx2 = u - 0.7f;
        Halide::Expr dy2 = v - 0.4f;
        Halide::Expr g2 = Halide::pow(2.0f, -(dx2 * dx2 + dy2 * dy2) / (2.0f * 0.08f * 0.08f));

        // Blob 3: center (0.5, 0.75), sigma 0.15
        Halide::Expr dx3 = u - 0.5f;
        Halide::Expr dy3 = v - 0.75f;
        Halide::Expr g3 = Halide::pow(2.0f, -(dx3 * dx3 + dy3 * dy3) / (2.0f * 0.15f * 0.15f));

        // Blob 4: center (0.2, 0.7), sigma 0.06
        Halide::Expr dx4 = u - 0.2f;
        Halide::Expr dy4 = v - 0.7f;
        Halide::Expr g4 = Halide::pow(2.0f, -(dx4 * dx4 + dy4 * dy4) / (2.0f * 0.06f * 0.06f));

        // Combined smooth image
        Halide::Expr smooth = Halide::clamp(g1 + g2 + g3 + g4, 0.0f, 1.0f);

        // --- Top-left: smooth image (sum of Gaussians) ---
        Halide::Expr smooth_val = smooth;

        // --- Analytical gradients ---
        // Derivative of pow(2, -d^2/(2*s^2)) with respect to u is:
        //   -ln(2) * (u-cx)/(s^2) * pow(2, -d^2/(2*s^2))
        // We approximate with finite difference-like approach using analytical derivatives
        Halide::Expr ln2 = 0.6931472f;

        Halide::Expr du_g1 = -ln2 * dx1 / (0.12f * 0.12f) * g1;
        Halide::Expr du_g2 = -ln2 * dx2 / (0.08f * 0.08f) * g2;
        Halide::Expr du_g3 = -ln2 * dx3 / (0.15f * 0.15f) * g3;
        Halide::Expr du_g4 = -ln2 * dx4 / (0.06f * 0.06f) * g4;
        Halide::Expr grad_u = du_g1 + du_g2 + du_g3 + du_g4;

        Halide::Expr dv_g1 = -ln2 * dy1 / (0.12f * 0.12f) * g1;
        Halide::Expr dv_g2 = -ln2 * dy2 / (0.08f * 0.08f) * g2;
        Halide::Expr dv_g3 = -ln2 * dy3 / (0.15f * 0.15f) * g3;
        Halide::Expr dv_g4 = -ln2 * dy4 / (0.06f * 0.06f) * g4;
        Halide::Expr grad_v = dv_g1 + dv_g2 + dv_g3 + dv_g4;

        // --- Top-right: gradient magnitude (edge strength) ---
        Halide::Expr grad_mag = Halide::sqrt(grad_u * grad_u + grad_v * grad_v);
        Halide::Expr grad_mag_val = Halide::clamp(grad_mag * 0.15f, 0.0f, 1.0f);

        // --- Bottom-left: Laplacian-like blob detector ---
        // Second derivatives (Laplacian = d2f/du2 + d2f/dv2)
        // d2/du2 of pow(2,-d^2/(2s^2)) = ln2/s^2 * ((ln2*(u-cx)^2/s^2) - 1) * g
        Halide::Expr d2u_g1 = ln2 / (0.12f * 0.12f) * (ln2 * dx1 * dx1 / (0.12f * 0.12f) - 1.0f) * g1;
        Halide::Expr d2v_g1 = ln2 / (0.12f * 0.12f) * (ln2 * dy1 * dy1 / (0.12f * 0.12f) - 1.0f) * g1;
        Halide::Expr d2u_g2 = ln2 / (0.08f * 0.08f) * (ln2 * dx2 * dx2 / (0.08f * 0.08f) - 1.0f) * g2;
        Halide::Expr d2v_g2 = ln2 / (0.08f * 0.08f) * (ln2 * dy2 * dy2 / (0.08f * 0.08f) - 1.0f) * g2;
        Halide::Expr d2u_g3 = ln2 / (0.15f * 0.15f) * (ln2 * dx3 * dx3 / (0.15f * 0.15f) - 1.0f) * g3;
        Halide::Expr d2v_g3 = ln2 / (0.15f * 0.15f) * (ln2 * dy3 * dy3 / (0.15f * 0.15f) - 1.0f) * g3;
        Halide::Expr d2u_g4 = ln2 / (0.06f * 0.06f) * (ln2 * dx4 * dx4 / (0.06f * 0.06f) - 1.0f) * g4;
        Halide::Expr d2v_g4 = ln2 / (0.06f * 0.06f) * (ln2 * dy4 * dy4 / (0.06f * 0.06f) - 1.0f) * g4;

        Halide::Expr laplacian = (d2u_g1 + d2v_g1) + (d2u_g2 + d2v_g2) + (d2u_g3 + d2v_g3) + (d2u_g4 + d2v_g4);
        // Map laplacian: negative at blob centers, positive at edges
        Halide::Expr laplacian_val = Halide::clamp((-laplacian) * 0.005f, 0.0f, 1.0f);

        // --- Bottom-right: gradient direction as grayscale ---
        Halide::Expr grad_dir = Halide::atan2(grad_v, grad_u);
        // Map from [-pi, pi] to [0, 1]
        Halide::Expr grad_dir_val = (grad_dir + pi) / (2.0f * pi);
        // Mask out regions with very low gradient (no meaningful direction)
        Halide::Expr has_grad = grad_mag > 0.5f;
        grad_dir_val = Halide::select(has_grad, grad_dir_val, 0.0f);

        // Compose quadrants
        Halide::Expr pixel = Halide::select(
            qy == 0 && qx == 0, smooth_val,
            qy == 0 && qx == 1, grad_mag_val,
            qy == 1 && qx == 0, laplacian_val,
            grad_dir_val
        );

        // Grid lines
        Halide::Expr on_grid = (ox == half) || (oy == half);
        pixel = Halide::select(on_grid, 0.4f, pixel);

        output(ox, oy) = Halide::cast<uint8_t>(Halide::clamp(pixel * 255.0f, 0.0f, 255.0f));

        std::cout << "Rendering..." << std::endl;
        Halide::Runtime::Buffer<uint8_t> result(size, size);
        output.realize(result);

        const char* output_path = "out/32_gradient.png";
        std::cout << "Saving to " << output_path << "..." << std::endl;

        if (save_png(result, output_path)) {
            std::cout << "Success!" << std::endl;
            std::cout << std::endl;
            std::cout << "Quadrant guide:" << std::endl;
            std::cout << "  Top-left:     smooth image (sum of Gaussians)" << std::endl;
            std::cout << "  Top-right:    gradient magnitude (edge strength)" << std::endl;
            std::cout << "  Bottom-left:  Laplacian-like blob detector" << std::endl;
            std::cout << "  Bottom-right: gradient direction as grayscale" << std::endl;
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
