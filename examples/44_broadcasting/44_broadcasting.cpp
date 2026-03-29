/// @file 44_broadcasting.cpp
/// @brief Example 44: Broadcasting binary operations in NumHalide
///
/// Demonstrates:
///   - Row-mean centering via sub() broadcast
///   - Column normalization via div() broadcast
///   - Outer product via mul() broadcast
///   - Per-channel bias addition via add() broadcast
///
/// Output: out/44_broadcasting.png (512x512, four 256x256 quadrants)
///   Top-left:     Row-mean centering of 64x64 signal
///   Top-right:    Column normalization by column scale vector
///   Bottom-left:  Outer product of sin/cos vectors
///   Bottom-right: Per-channel bias addition (4 vertical bands)

#include "numhalide_all.h"
#include "stbi_png.h"
#include <iostream>
#include <cmath>

using namespace numhalide;

static constexpr int DATA_SIZE = 64;
static constexpr float PI = 3.14159265358979323846f;

int main() {
    try {
        Halide::Var x("x"), y("y"), ox("ox"), oy("oy");

        // -----------------------------------------------------------------------
        // Quadrant 0 (top-left): Row-mean centering
        //   signal f(x,y) = (x + y*64) / 4096.0f  shape {64,64}
        //   row_mean = reduce_mean(f, axis=1, keepdims=true)  shape {64,1}
        //   centered = f - row_mean                           shape {64,64}
        // -----------------------------------------------------------------------
        Halide::Func signal_q0("sig_q0");
        signal_q0(x, y) = (Halide::cast<float>(x) + Halide::cast<float>(y) * DATA_SIZE)
                          / float(DATA_SIZE * DATA_SIZE);
        signal_q0.compute_root();

        // Row mean: reduce over cols (axis=1), keepdims=true → shape {64,1}
        auto row_mean = reduce_mean(signal_q0, {DATA_SIZE, DATA_SIZE},
                                    {1}, true, "row_mean_q0");
        row_mean.compute_root();

        // sub() broadcasts {64,1} to {64,64}
        auto centered = sub(signal_q0, {DATA_SIZE, DATA_SIZE},
                            row_mean, {DATA_SIZE, 1}, "centered_q0");
        centered.compute_root();

        // Map centered values [-0.5, 0.5] -> [0, 1]
        Halide::Func q0("q0");
        q0(x, y) = Halide::clamp(centered(x, y) + 0.5f, 0.0f, 1.0f);
        q0.compute_root();

        // -----------------------------------------------------------------------
        // Quadrant 1 (top-right): Column normalization
        //   A(x,y) = cos(x * pi / 64) * 0.5 + 0.5  (columns have different base)
        //   col_scale(x,y) = (x + 1) / float(DATA_SIZE)  shape {1,64}
        //   normalized = div(A, {64,64}, col_scale, {1,64})
        // -----------------------------------------------------------------------
        Halide::Func sig_q1("sig_q1");
        sig_q1(x, y) = Halide::cos(Halide::cast<float>(x) * PI / float(DATA_SIZE))
                       * 0.5f + 0.5f;
        sig_q1.compute_root();

        // Column scale: shape {1, DATA_SIZE} — each column gets different scale
        // col_scale(x,y): x is col index (1 column, DATA_SIZE cols in output dim)
        // In shape {1,64}: 1 row, 64 cols => f(col, row) with row clamped to 0
        Halide::Func col_scale("col_scale_q1");
        col_scale(x, y) = (Halide::cast<float>(x) + 1.0f) / float(DATA_SIZE);

        // div broadcasts {1,64} to {64,64}
        auto normalized = div(sig_q1, {DATA_SIZE, DATA_SIZE},
                              col_scale, {1, DATA_SIZE}, "norm_q1");
        normalized.compute_root();

        Halide::Func q1("q1");
        q1(x, y) = Halide::clamp(normalized(x, y), 0.0f, 1.0f);
        q1.compute_root();

        // -----------------------------------------------------------------------
        // Quadrant 2 (bottom-left): Outer product
        //   row_vec(x,y) = sin(x*2*pi/64)*0.5+0.5  shape {1,64}
        //   col_vec(x,y) = cos(y*2*pi/64)*0.5+0.5  shape {64,1}
        //   outer = mul(col_vec, {64,1}, row_vec, {1,64}) → {64,64}
        // -----------------------------------------------------------------------
        Halide::Func row_vec("row_vec_q2"), col_vec("col_vec_q2");
        // row_vec: 1 row, 64 cols  → shape {1,64}
        row_vec(x, y) = Halide::sin(Halide::cast<float>(x) * 2.0f * PI / float(DATA_SIZE))
                        * 0.5f + 0.5f;
        // col_vec: 64 rows, 1 col → shape {64,1}
        col_vec(x, y) = Halide::cos(Halide::cast<float>(y) * 2.0f * PI / float(DATA_SIZE))
                        * 0.5f + 0.5f;

        // mul broadcasts both to {64,64}
        auto outer_prod = mul(col_vec, {DATA_SIZE, 1},
                              row_vec, {1, DATA_SIZE}, "outer_q2");
        outer_prod.compute_root();

        Halide::Func q2("q2");
        q2(x, y) = Halide::clamp(outer_prod(x, y), 0.0f, 1.0f);
        q2.compute_root();

        // -----------------------------------------------------------------------
        // Quadrant 3 (bottom-right): Per-channel bias addition
        //   A(x,y) = (x*y) / 256.0f  shape {64,4} (64 rows, 4 cols)
        //   bias(x,y) = x / 4.0f for x in [0,4)   shape {1,4}
        //   result = add(A, {64,4}, bias, {1,4})
        // -----------------------------------------------------------------------
        const int NCHAN = 4;
        Halide::Func act("act_q3");
        act(x, y) = Halide::cast<float>(x * y) / float(DATA_SIZE * NCHAN);
        act.compute_root();

        Halide::Func bias("bias_q3");
        // shape {1,4}: 1 row, 4 cols — bias per channel
        bias(x, y) = Halide::cast<float>(x) / float(NCHAN);

        // add broadcasts {1,4} to {64,4}
        auto biased = add(act, {DATA_SIZE, NCHAN}, bias, {1, NCHAN}, "biased_q3");
        biased.compute_root();

        Halide::Func q3("q3");
        q3(x, y) = Halide::clamp(biased(x, y), 0.0f, 1.0f);
        q3.compute_root();

        // -----------------------------------------------------------------------
        // Compositing: 512x512 with 4 quadrants (256x256 each)
        // Each quadrant scales from DATA_SIZE x DATA_SIZE (or DATA_SIZE x NCHAN)
        // using nearest-neighbor interpolation.
        // A 1-pixel gray separator is drawn at ox==256 and oy==256.
        // -----------------------------------------------------------------------
        const int OUT_SIZE  = 512;
        const int QUAD_SIZE = 256;
        const float SEP = 0.5f;

        Halide::Func output("output_44");
        Halide::Var px("px"), py("py");

        // Local coords within quadrant [0, QUAD_SIZE)
        Halide::Expr lx = px % QUAD_SIZE;
        Halide::Expr ly = py % QUAD_SIZE;
        // Which quadrant: qx=0 left, qx=1 right; qy=0 top, qy=1 bottom
        Halide::Expr qx = px / QUAD_SIZE;
        Halide::Expr qy = py / QUAD_SIZE;

        // Separator pixels
        Halide::Expr is_sep = (px == QUAD_SIZE) || (py == QUAD_SIZE);

        // Scale lx/ly to DATA_SIZE coords
        Halide::Expr dx = (lx * DATA_SIZE) / QUAD_SIZE;
        Halide::Expr dy = (ly * DATA_SIZE) / QUAD_SIZE;

        // For Q3, cols dimension is NCHAN (scale lx to [0,NCHAN))
        Halide::Expr dx_q3 = (lx * NCHAN) / QUAD_SIZE;

        Halide::Expr pixel_val = Halide::select(
            is_sep, SEP,
            // top-left: Q0
            qx == 0 && qy == 0, q0(dx, dy),
            // top-right: Q1
            qx == 1 && qy == 0, q1(dx, dy),
            // bottom-left: Q2
            qx == 0 && qy == 1, q2(dx, dy),
            // bottom-right: Q3 (64 rows x 4 cols, scale x to [0,4))
            q3(dx_q3, dy)
        );

        output(px, py) = Halide::cast<uint8_t>(
            Halide::clamp(pixel_val * 255.0f, 0.0f, 255.0f));

        Halide::Runtime::Buffer<uint8_t> result(OUT_SIZE, OUT_SIZE);
        output.realize(result);

        if (save_png(result, "out/44_broadcasting.png")) {
            std::cout << "Saved out/44_broadcasting.png\n";
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
