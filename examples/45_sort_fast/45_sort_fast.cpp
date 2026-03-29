/// @file 45_sort_fast.cpp
/// @brief Example 45: sort_2d and argsort_2d visualization
///
/// Demonstrates:
///   - sort_2d axis=1 (sort each row)   → horizontal banded structure
///   - sort_2d axis=0 (sort each col)   → vertical gradient structure
///   - argsort_2d axis=1 (row argsort)  → normalized index image
///
/// Input: pseudo-random 64x64 image f(x,y) = frac(sin(x*37+y*17)*1000)
///
/// Output: out/45_sort_fast.png (512x512, four 256x256 quadrants)
///   Top-left:     Original random-like input
///   Top-right:    sort_2d axis=1 (each row sorted ascending)
///   Bottom-left:  sort_2d axis=0 (each column sorted ascending)
///   Bottom-right: argsort_2d axis=1 (sorted indices, normalized to [0,1])

#include "numhalide_all.h"
#include "stbi_png.h"
#include <iostream>
#include <cmath>

using namespace numhalide;

int main() {
    try {
        const int DATA  = 64;
        const int OUT   = 512;
        const int QUAD  = 256;

        Halide::Var x("x"), y("y"), px("px"), py("py");

        // -----------------------------------------------------------------------
        // Input: pseudo-random pattern in [0,1]
        //   f(x,y) = frac(sin(cast<float>(x*37 + y*17)) * 1000.0)
        // -----------------------------------------------------------------------
        Halide::Func f_input("f_input");
        f_input(x, y) = Halide::fract(
            Halide::sin(Halide::cast<float>(x * 37 + y * 17)) * 1000.0f);
        f_input.compute_root();

        // -----------------------------------------------------------------------
        // Q1 (top-right): sort each row ascending (axis=1)
        // -----------------------------------------------------------------------
        auto sorted_rows = sort_2d(f_input, DATA, DATA, 1, true, "sort_rows");
        sorted_rows.compute_root();

        // -----------------------------------------------------------------------
        // Q2 (bottom-left): sort each column ascending (axis=0)
        // -----------------------------------------------------------------------
        auto sorted_cols = sort_2d(f_input, DATA, DATA, 0, true, "sort_cols");
        sorted_cols.compute_root();

        // -----------------------------------------------------------------------
        // Q3 (bottom-right): argsort each row ascending (axis=1)
        //   Normalize indices [0, DATA-1] to [0.0, 1.0]
        // -----------------------------------------------------------------------
        auto argsorted = argsort_2d(f_input, DATA, DATA, 1, true, "argsort_rows");
        argsorted.compute_root();

        Halide::Func argsort_norm("argsort_norm");
        argsort_norm(x, y) = Halide::cast<float>(argsorted(x, y)) / float(DATA - 1);
        argsort_norm.compute_root();

        // -----------------------------------------------------------------------
        // Compositing: 512x512 with 4 quadrants, nearest-neighbor scaling
        // -----------------------------------------------------------------------
        Halide::Expr lx   = px % QUAD;
        Halide::Expr ly   = py % QUAD;
        Halide::Expr qx   = px / QUAD;
        Halide::Expr qy   = py / QUAD;
        Halide::Expr is_sep = (px == QUAD) || (py == QUAD);
        Halide::Expr dx   = (lx * DATA) / QUAD;
        Halide::Expr dy   = (ly * DATA) / QUAD;

        Halide::Func output("output_45");
        Halide::Expr val = Halide::select(
            is_sep, 0.5f,
            qx == 0 && qy == 0, f_input(dx, dy),
            qx == 1 && qy == 0, sorted_rows(dx, dy),
            qx == 0 && qy == 1, sorted_cols(dx, dy),
            argsort_norm(dx, dy)
        );
        output(px, py) = Halide::cast<uint8_t>(
            Halide::clamp(val * 255.0f, 0.0f, 255.0f));

        Halide::Runtime::Buffer<uint8_t> result(OUT, OUT);
        output.realize(result);

        if (save_png(result, "out/45_sort_fast.png")) {
            std::cout << "Saved out/45_sort_fast.png\n";
        } else {
            std::cerr << "Failed to save PNG\n";
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
