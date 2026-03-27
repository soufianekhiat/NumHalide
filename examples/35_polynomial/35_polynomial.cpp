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

        // --- polyfit: fit y = 0.5*x^2 - 0.3*x at 11 sample points in [-1,1] ---
        // Realize the coefficients before building the render pipeline so they
        // can be embedded as float literals in the Halide expression below.
        const int n_pts = 11;
        Halide::Buffer<float> xd(n_pts), yd(n_pts);
        for (int k = 0; k < n_pts; ++k) {
            float xi = -1.0f + k * 0.2f;
            xd(k) = xi;
            yd(k) = 0.5f * xi * xi - 0.3f * xi;
        }
        Halide::Var kv("kv");
        Halide::Func xf("xf_pf"), yf("yf_pf");
        xf(kv) = xd(kv);
        yf(kv) = yd(kv);
        auto poly_coeffs = polyfit(xf, yf, n_pts, 2, "pf");
        Halide::Runtime::Buffer<float> cbuf(3);
        poly_coeffs.realize(cbuf);
        float c0 = cbuf(0), c1 = cbuf(1), c2 = cbuf(2);

        std::cout << "Polynomial functions demonstration" << std::endl;
        std::cout << "Output size: " << size << "x" << size << std::endl;
        std::cout << "polyfit(y=0.5x²-0.3x, deg=2): c0=" << c0
                  << "  c1=" << c1 << "  c2=" << c2 << std::endl;
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

        // --- Bottom-right: polyfit — curve fitting visualization ---
        // Map pixel (lx, ly) to data space (pf_u, pf_v) in [-1.3, 1.3]
        Halide::Expr pf_u = Halide::cast<float>(lx) / half * 2.6f - 1.3f;
        Halide::Expr pf_v = 1.3f - Halide::cast<float>(ly) / half * 2.6f;

        // Fitted curve y = c0 + c1*x + c2*x^2 (coefficients realized above)
        Halide::Expr pf_y = c0 + c1 * pf_u + c2 * pf_u * pf_u;

        // Thin curve band
        Halide::Expr on_curve = Halide::abs(pf_v - pf_y) < 0.06f;

        // Data point dots
        Halide::Expr on_pt = Halide::cast<float>(0);
        for (int pk = 0; pk < n_pts; ++pk) {
            float xi = xd(pk), yi = yd(pk);
            Halide::Expr dxi = pf_u - xi, dyi = pf_v - yi;
            on_pt = Halide::max(on_pt,
                Halide::select(dxi * dxi + dyi * dyi < 0.005f, 1.0f, 0.0f));
        }
        // Coordinate axes
        Halide::Expr pf_axis = (Halide::abs(pf_u) < 0.022f) || (Halide::abs(pf_v) < 0.022f);

        Halide::Expr polyfit_val = Halide::select(on_pt > 0.5f, 1.0f,
            Halide::select(on_curve, 0.75f,
            Halide::select(pf_axis, 0.35f, 0.08f)));

        // Compose quadrants
        Halide::Expr pixel = Halide::select(
            qy == 0 && qx == 0, poly_val,
            qy == 0 && qx == 1, cheb_val,
            qy == 1 && qx == 0, leg_val,
            polyfit_val
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
            std::cout << "  Bottom-right: polyfit — degree-2 fit of y=0.5x²-0.3x" << std::endl;
            std::cout << "                (dots = data points, line = fitted curve)" << std::endl;
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
