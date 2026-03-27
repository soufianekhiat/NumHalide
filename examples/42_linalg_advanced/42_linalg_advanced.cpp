/// @file 42_linalg_advanced.cpp
/// @brief Example 42: Advanced linear algebra — eigenvalues, rank, condition, log-determinant
///
/// Demonstrates:
///   - eig2x2:      exact eigenvalues of a 2x2 real matrix (may be complex)
///   - cond:        condition number via SVD (σ_max / σ_min)
///   - slogdet:     sign and log-|determinant| via LU decomposition
///   - matrix_rank: numerical rank via SVD with tolerance
///
/// Visualization: each of 4 quadrants shows a 2x2 matrix A as the image of the
/// unit ball under A (gray ellipse) overlaid with the unit circle (white disk).
/// The ellipse semi-axes equal the singular values of A.
///
/// Output: out/42_linalg_advanced.png

#include "numhalide_all.h"
#include "stbi_png.h"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace numhalide;

/// Wrap a 2x2 float array as a Halide Func(col, row).
/// Row-major convention: a00=top-left, a01=top-right, a10=bottom-left, a11=bottom-right.
static Halide::Func mat22(float a00, float a01, float a10, float a11, const std::string& nm)
{
    Halide::Buffer<float> buf(2, 2);
    buf(0, 0) = a00; buf(1, 0) = a01;
    buf(0, 1) = a10; buf(1, 1) = a11;
    Halide::Func f(nm);
    Halide::Var i("i"), j("j");
    f(i, j) = buf(i, j);
    return f;
}

/// Print eig2x2 / cond / slogdet / matrix_rank for one 2x2 matrix.
static void print_info(Halide::Func A, const std::string& label)
{
    auto eig = eig2x2(A, label + "_eig");
    Halide::Runtime::Buffer<float> er(2), ei(2);
    eig.real.realize(er); eig.imag.realize(ei);

    auto cb = Halide::Buffer<float>::make_scalar();
    cond(A, 2, 2, 10, label + "_cond").realize(cb);

    auto sd = slogdet(A, 2, label + "_sd");
    auto sb  = Halide::Buffer<float>::make_scalar();
    auto ldb = Halide::Buffer<float>::make_scalar();
    sd.sign.realize(sb); sd.logabsdet.realize(ldb);

    auto rkb = Halide::Buffer<float>::make_scalar();
    matrix_rank(A, 2, 2, 0.1f, 10, label + "_rk").realize(rkb);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << label << ":\n";
    if (std::abs(ei(0)) < 1e-4f && std::abs(ei(1)) < 1e-4f)
        std::cout << "  eig = (" << er(0) << ", " << er(1) << ")  [real]\n";
    else
        std::cout << "  eig = (" << er(0) << "+" << ei(0) << "i, "
                  << er(1) << "+" << ei(1) << "i)  [complex]\n";
    std::cout << "  cond = " << cb()
              << "  sign = " << sb() << "  logdet = " << ldb()
              << "  rank = " << (int)rkb() << "\n\n";
}

int main(int argc, char **argv) {
    try {
        const int Q   = 256;
        const int OUT = Q * 2;   // 512x512

        std::cout << "Advanced linear algebra demonstration\n\n";

        // Four 2x2 matrices
        auto A1 = mat22(2, 0,  0, 1, "A1");   // diag(2,1): λ=(2,1),   cond=2
        auto A2 = mat22(1, 0,  0, 3, "A2");   // diag(1,3): λ=(1,3),   cond=3
        auto A3 = mat22(2, 1,  1, 2, "A3");   // symmetric: λ=(3,1),   cond=3
        auto A4 = mat22(1,-2,  2, 1, "A4");   // rot-scale: λ=1±2i,    cond=1

        // Print numerical values (JIT-realized before building the render pipeline)
        print_info(A1, "A1=diag(2,1)");
        print_info(A2, "A2=diag(1,3)");
        print_info(A3, "A3=[[2,1],[1,2]]");
        print_info(A4, "A4=[[1,-2],[2,1]]");

        // ---- Visualization: image of unit ball under each A ----
        // Coordinate system: pixel (lx, ly) -> (u, v) in [-RANGE, RANGE]
        // A point (u,v) is inside the image of the unit ball iff ||A^{-1}(u,v)||_2 <= 1.
        Halide::Func output("output");
        Halide::Var ox("ox"), oy("oy");
        Halide::Expr qx = ox / Q, qy = oy / Q;
        Halide::Expr lx = ox % Q, ly = oy % Q;

        const float RANGE = 2.8f;
        Halide::Expr u = (Halide::cast<float>(lx) - (Q - 1) * 0.5f) / ((Q - 1) * 0.5f) * RANGE;
        Halide::Expr v = ((Q - 1) * 0.5f - Halide::cast<float>(ly)) / ((Q - 1) * 0.5f) * RANGE;

        // Unit circle: u^2 + v^2 <= 1
        Halide::Expr in_circle = u * u + v * v <= 1.0f;

        // Image of unit ball (|| A^{-1}(u,v) ||^2 <= 1) for each matrix:
        // A1=diag(2,1):  A^{-1}=diag(0.5,1)  =>  0.25u^2 + v^2 <= 1
        Halide::Expr in_A1 = 0.25f * u * u + v * v <= 1.0f;
        // A2=diag(1,3):  A^{-1}=diag(1,1/3)  =>  u^2 + v^2/9 <= 1
        Halide::Expr in_A2 = u * u + v * v / 9.0f <= 1.0f;
        // A3=[[2,1],[1,2]]:  det=3,  A^{-1}=(1/3)[[2,-1],[-1,2]]
        //   ||A^{-1}(u,v)||^2 = (5u^2 - 8uv + 5v^2) / 9
        Halide::Expr in_A3 = (5.0f * u * u - 8.0f * u * v + 5.0f * v * v) / 9.0f <= 1.0f;
        // A4=[[1,-2],[2,1]]:  det=5,  A^{-1}=(1/5)[[1,2],[-2,1]]
        //   ||A^{-1}(u,v)||^2 = (u^2+v^2)/5  =>  circle of radius sqrt(5)≈2.24
        Halide::Expr in_A4 = (u * u + v * v) / 5.0f <= 1.0f;

        // Grid at integer u/v values and coordinate axes
        Halide::Expr near_int_u = Halide::abs(u - Halide::round(u)) < 0.065f;
        Halide::Expr near_int_v = Halide::abs(v - Halide::round(v)) < 0.065f;
        Halide::Expr on_grid  = near_int_u || near_int_v;
        Halide::Expr on_axis  = (Halide::abs(u) < 0.065f) || (Halide::abs(v) < 0.065f);

        // Pixel value:  ellipse=gray, circle=lighter, overlap=medium, grid/axis=dim lines
        auto quadrant = [&](Halide::Expr in_ball) -> Halide::Expr {
            Halide::Expr bg  = Halide::select(in_ball, 0.40f, 0.05f);
            Halide::Expr val = Halide::select(in_circle, 0.75f, bg);
            // Grid only in background (outside both shapes)
            val = Halide::select(!in_ball && !in_circle,
                Halide::select(on_axis,  0.30f,
                Halide::select(on_grid,  0.18f, val)), val);
            return val;
        };

        Halide::Expr pix = Halide::select(
            qy == 0 && qx == 0, quadrant(in_A1),
            Halide::select(qy == 0 && qx == 1, quadrant(in_A2),
            Halide::select(qy == 1 && qx == 0, quadrant(in_A3),
                           quadrant(in_A4))));

        pix = Halide::select((ox == Q) || (oy == Q), 0.5f, pix);

        output(ox, oy) = Halide::cast<uint8_t>(Halide::clamp(pix * 255.0f, 0.0f, 255.0f));

        std::cout << "Rendering...\n";
        Halide::Runtime::Buffer<uint8_t> result(OUT, OUT);
        output.realize(result);

        const char* out_path = "out/42_linalg_advanced.png";
        if (save_png(result, out_path)) {
            std::cout << "Saved to " << out_path << "\n\n";
            std::cout << "Each quadrant: gray=image-of-unit-ball, white=unit-circle\n";
            std::cout << "  Top-left:     A=diag(2,1)      wider ellipse  (cond=2)\n";
            std::cout << "  Top-right:    A=diag(1,3)      taller ellipse (cond=3)\n";
            std::cout << "  Bottom-left:  A=[[2,1],[1,2]]  rotated ellipse(cond=3)\n";
            std::cout << "  Bottom-right: A=[[1,-2],[2,1]] circle×sqrt(5) (cond=1)\n";
        } else {
            std::cerr << "Error: Failed to save PNG\n";
            return 1;
        }
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
