/// @file 41_padding.cpp
/// @brief Example 41: Array padding with four boundary modes
///
/// Demonstrates:
///   - pad_2d:         constant-value border (black)
///   - pad_2d_edge:    replicate the nearest edge pixel
///   - pad_2d_reflect: mirror at the boundary (boundary pixel not repeated)
///   - pad_2d_wrap:    circular / periodic wrap-around
///
/// Source: 128x128 gradient pattern with a bright circle marker.
/// Each mode adds a 64-pixel border, producing a 256x256 padded image.
/// The four modes are shown in a 512x512 quad layout.
///
/// Output: out/41_padding.png

#include "numhalide_all.h"
#include "stbi_png.h"
#include <iostream>

using namespace numhalide;

int main(int argc, char **argv) {
    try {
        const int SRC  = 128;   // source image size
        const int PAD  = 64;    // padding on each side
        const int Q    = SRC + 2 * PAD;   // = 256: padded quadrant size
        const int OUT  = Q * 2;           // = 512: total output size

        std::cout << "Padding modes demonstration\n";
        std::cout << "Source: " << SRC << "x" << SRC
                  << "  Padding: " << PAD << " px each side"
                  << "  Padded: " << Q << "x" << Q << "\n\n";

        Halide::Var x("x"), y("y");

        // Source pattern: diagonal gradient + bright circle in top-left
        Halide::Func src("src");
        Halide::Expr in_circle = (x - SRC / 5) * (x - SRC / 5) +
                                  (y - SRC / 5) * (y - SRC / 5) <
                                  (SRC / 6) * (SRC / 6);
        Halide::Expr grad = Halide::cast<float>(x + y) / (2.0f * (SRC - 1));
        src(x, y) = Halide::select(in_circle, 1.0f, grad);

        shape_t src_shape = {SRC, SRC};

        // Apply the four padding modes
        auto p_const   = pad_2d(src, src_shape, PAD, PAD, PAD, PAD, 0.0f,  "p_const");
        auto p_edge    = pad_2d_edge   (src, src_shape, PAD, PAD, PAD, PAD, "p_edge");
        auto p_reflect = pad_2d_reflect(src, src_shape, PAD, PAD, PAD, PAD, "p_reflect");
        auto p_wrap    = pad_2d_wrap   (src, src_shape, PAD, PAD, PAD, PAD, "p_wrap");

        // Compose into 2x2 output
        Halide::Func output("output");
        Halide::Var ox("ox"), oy("oy");
        Halide::Expr qx = ox / Q, qy = oy / Q;
        Halide::Expr lx = ox % Q, ly = oy % Q;

        Halide::Expr pix = Halide::select(
            qy == 0 && qx == 0, p_const(lx, ly),
            qy == 0 && qx == 1, p_edge(lx, ly),
            qy == 1 && qx == 0, p_reflect(lx, ly),
            p_wrap(lx, ly)
        );

        // Thin border marking the source region boundary
        Halide::Expr at_src_border =
            (lx == PAD || lx == PAD + SRC - 1 ||
             ly == PAD || ly == PAD + SRC - 1);
        pix = Halide::select(at_src_border, 0.55f, pix);

        // Grid line between quadrants
        pix = Halide::select((ox == Q) || (oy == Q), 0.4f, pix);

        output(ox, oy) = Halide::cast<uint8_t>(Halide::clamp(pix * 255.0f, 0.0f, 255.0f));

        std::cout << "Rendering...\n";
        Halide::Runtime::Buffer<uint8_t> result(OUT, OUT);
        output.realize(result);

        const char* out_path = "out/41_padding.png";
        if (save_png(result, out_path)) {
            std::cout << "Saved to " << out_path << "\n\n";
            std::cout << "Quadrant guide (white border = source region):\n";
            std::cout << "  Top-left:     pad_2d constant  (black border)\n";
            std::cout << "  Top-right:    pad_2d_edge      (replicated edge pixel)\n";
            std::cout << "  Bottom-left:  pad_2d_reflect   (mirror at boundary)\n";
            std::cout << "  Bottom-right: pad_2d_wrap      (circular wrap-around)\n";
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
