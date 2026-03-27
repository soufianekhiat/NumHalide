/// @file 03_stacking.cpp
/// @brief Example 03: Concatenation and stacking operations
///
/// Demonstrates:
///   - concat_2d axis=1: join two 2D arrays side by side (horizontal)
///   - concat_2d axis=0: join two 2D arrays top/bottom (vertical)
///   - stack:            build a 2D matrix from multiple 1D arrays
///   - concatenate:      join a list of 1D arrays end-to-end
///
/// Output: out/03_stacking.png

#include "numhalide_all.h"
#include "stbi_png.h"
#include <iostream>

using namespace numhalide;

int main(int argc, char **argv) {
    try {
        const int Q   = 256;   // each quadrant is Q x Q
        const int OUT = Q * 2; // 512x512 total output

        std::cout << "Concatenation and stacking demonstration\n";
        std::cout << "Output: " << OUT << "x" << OUT << "\n\n";

        Halide::Var x("x"), y("y");

        // Two source patterns (both defined on [0,Q)x[0,Q))
        Halide::Func gradH("gradH");   // horizontal ramp: bright on right
        gradH(x, y) = Halide::cast<float>(x) / (Q - 1.0f);

        Halide::Func gradV("gradV");   // vertical ramp: bright at bottom
        gradV(x, y) = Halide::cast<float>(y) / (Q - 1.0f);

        // ---- TL: concat_2d axis=1 (horizontal join: left | right) ----
        // Left Q×(Q/2): horizontal ramp   Right Q×(Q/2): vertical ramp
        shape_t s_hcol = {Q, Q / 2};   // rows=Q, cols=Q/2
        auto h_join = concat_2d(gradH, s_hcol, gradV, s_hcol, 1, "h_join");

        // ---- TR: concat_2d axis=0 (vertical join: top / bottom) ----
        // Top (Q/2)×Q: horizontal ramp   Bottom (Q/2)×Q: vertical ramp
        shape_t s_vrow = {Q / 2, Q};   // rows=Q/2, cols=Q
        auto v_join = concat_2d(gradH, s_vrow, gradV, s_vrow, 0, "v_join");

        // ---- BL: stack 4 distinct 1D patterns into a 4-row matrix, tiled ----
        Halide::Func b0("b0"); b0(x) = 0.1f;                                          // constant low
        Halide::Func b1("b1"); b1(x) = Halide::cast<float>(x) / (Q - 1.0f);          // rising ramp
        Halide::Func b2("b2"); b2(x) = 0.9f;                                          // constant high
        Halide::Func b3("b3"); b3(x) = 1.0f - Halide::cast<float>(x) / (Q - 1.0f);  // falling ramp

        auto stk = stack({b0, b1, b2, b3}, Q, 0, "stk");
        // stk(x, y): x=column [0,Q), y=band index [0,3]
        // Tile vertically: divide Q rows evenly among 4 bands
        Halide::Func stk_img("stk_img");
        stk_img(x, y) = stk(x, (y * 4) / Q);

        // ---- BR: concatenate 3 wave segments into one length-Q array ----
        const int seg  = Q / 3;       // first two segments
        const int seg2 = Q - 2 * seg; // third segment (remainder)

        Halide::Func rise("rise"), flat("flat"), fall("fall");
        rise(x) = Halide::cast<float>(x) / (seg  - 1.0f);          // 0 → 1
        flat(x) = 0.9f;                                              // constant
        fall(x) = 1.0f - Halide::cast<float>(x) / (seg2 - 1.0f);   // 1 → 0

        auto cat3 = concatenate({rise, flat, fall}, {seg, seg, seg2}, "cat3");
        // Render the 1D result as a stripe (same value every row)
        Halide::Func cat_img("cat_img");
        cat_img(x, y) = cat3(x);

        // ---- Compose into 2x2 output ----
        Halide::Func output("output");
        Halide::Var ox("ox"), oy("oy");
        Halide::Expr qx = ox / Q, qy = oy / Q;
        Halide::Expr lx = ox % Q, ly = oy % Q;

        Halide::Expr pix = Halide::select(
            qy == 0 && qx == 0, h_join(lx, ly),
            qy == 0 && qx == 1, v_join(lx, ly),
            qy == 1 && qx == 0, stk_img(lx, ly),
            cat_img(lx, ly)
        );
        // Quadrant separator line
        pix = Halide::select((ox == Q) || (oy == Q), 0.5f, pix);

        output(ox, oy) = Halide::cast<uint8_t>(Halide::clamp(pix * 255.0f, 0.0f, 255.0f));

        std::cout << "Rendering...\n";
        Halide::Runtime::Buffer<uint8_t> result(OUT, OUT);
        output.realize(result);

        const char* out_path = "out/03_stacking.png";
        if (save_png(result, out_path)) {
            std::cout << "Saved to " << out_path << "\n\n";
            std::cout << "Quadrant guide:\n";
            std::cout << "  Top-left:     concat_2d axis=1 — horizontal join (left|right)\n";
            std::cout << "  Top-right:    concat_2d axis=0 — vertical join   (top/bottom)\n";
            std::cout << "  Bottom-left:  stack of 4 gradient bands tiled vertically\n";
            std::cout << "  Bottom-right: concatenate of 3 wave segments (rise|flat|fall)\n";
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
