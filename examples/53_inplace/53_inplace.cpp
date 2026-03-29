/// @file 53_inplace.cpp
/// @brief Example 53: In-place element-wise operations
///
/// Demonstrates in-place operations that modify a Halide::Runtime::Buffer<float>
/// directly without allocating a new output buffer.
///
/// Output: 512x512 PNG divided into four 256x256 quadrants:
///
///   Top-left:     Original gradient image (val = (x*y) / (255*255))
///   Top-right:    After in-place chain: scale(8) -> clamp(0,1) -> gamma(0.5)
///                 (brightened / gamma-corrected version of the gradient)
///   Bottom-left:  Sinusoidal pattern after inplace_normalize then inplace_exp
///   Bottom-right: Correctness check — |in-place result - expected formula| < 1e-4
///                 white pixels = match, red pixels = mismatch (should be all white)
///
/// Output: out/53_inplace.png

#include "numhalide_all.h"
#include "stbi_png.h"
#include "inplace.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>

using namespace numhalide;

static const float PI = 3.14159265358979323846f;

// Build a 256x256 float buffer from a lambda f(x, y) -> float
// Halide convention: Buffer(width=256, height=256), accessed as b(x, y)
template<typename F>
static Halide::Runtime::Buffer<float> make_float_image(int w, int h, F func)
{
    Halide::Runtime::Buffer<float> b(w, h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            b(x, y) = func(x, y);
    return b;
}

// Copy a float buffer (same dimensions)
static Halide::Runtime::Buffer<float> copy_buf(const Halide::Runtime::Buffer<float>& src)
{
    Halide::Runtime::Buffer<float> dst(src.width(), src.height());
    for (int y = 0; y < src.height(); ++y)
        for (int x = 0; x < src.width(); ++x)
            dst(x, y) = src(x, y);
    return dst;
}

// Write a float image [0,1] as grayscale into an RGB uint8 output buffer
// at offset (ox, oy) spanning w x h pixels.
static void blit_gray(Halide::Runtime::Buffer<uint8_t>& img,
                      const Halide::Runtime::Buffer<float>& src,
                      int ox, int oy)
{
    int w = src.width();
    int h = src.height();
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float v = std::max(0.0f, std::min(1.0f, src(x, y)));
            uint8_t byte = static_cast<uint8_t>(v * 255.0f + 0.5f);
            img(ox + x, oy + y, 0) = byte;
            img(ox + x, oy + y, 1) = byte;
            img(ox + x, oy + y, 2) = byte;
        }
    }
}

// Write a correctness-check image: white if |a - b| < tol, red otherwise.
static void blit_diff(Halide::Runtime::Buffer<uint8_t>& img,
                      const Halide::Runtime::Buffer<float>& a,
                      const Halide::Runtime::Buffer<float>& b_ref,
                      int ox, int oy, float tol = 1e-4f)
{
    int w = a.width();
    int h = a.height();
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float diff = std::abs(a(x, y) - b_ref(x, y));
            if (diff < tol) {
                img(ox + x, oy + y, 0) = 255;
                img(ox + x, oy + y, 1) = 255;
                img(ox + x, oy + y, 2) = 255;
            } else {
                img(ox + x, oy + y, 0) = 255;
                img(ox + x, oy + y, 1) = 0;
                img(ox + x, oy + y, 2) = 0;
            }
        }
    }
}

// Draw a thin grid line (dark gray) separating quadrants
static void draw_grid(Halide::Runtime::Buffer<uint8_t>& img, int size)
{
    int half = size / 2;
    for (int i = 0; i < size; ++i) {
        // Vertical line at x = half
        img(half, i, 0) = 80; img(half, i, 1) = 80; img(half, i, 2) = 80;
        // Horizontal line at y = half
        img(i, half, 0) = 80; img(i, half, 1) = 80; img(i, half, 2) = 80;
    }
}

int main(int /*argc*/, char** /*argv*/)
{
    try {
        const int half = 256;
        const int size = 512;

        std::cout << "In-place operations example" << std::endl;

        // ---------------------------------------------------------------
        // Top-left: original gradient image
        //   val = (x * y) / (255 * 255)   — dark near origin, bright at (255,255)
        // ---------------------------------------------------------------
        auto orig = make_float_image(half, half, [](int x, int y) {
            return (float(x) * float(y)) / (255.0f * 255.0f);
        });

        // ---------------------------------------------------------------
        // Top-right: in-place chain on a copy of the gradient
        //   scale(8) -> clamp(0,1) -> gamma(0.5)
        //   scale(8) amplifies the dim quadratic ramp so mid-values become bright,
        //   clamp keeps it in [0,1], gamma(0.5)=sqrt brings up mid-tones further.
        // ---------------------------------------------------------------
        auto processed = copy_buf(orig);
        inplace_scale(processed, 8.0f,    "ex53_scale");
        inplace_clamp(processed, 0.0f, 1.0f, "ex53_clamp");
        inplace_gamma(processed, 0.5f,    "ex53_gamma");

        // Also compute the expected result with plain C++ to validate
        auto expected_tr = make_float_image(half, half, [](int x, int y) {
            float v = (float(x) * float(y)) / (255.0f * 255.0f);
            v = v * 8.0f;
            v = std::max(0.0f, std::min(1.0f, v));
            v = std::pow(v, 0.5f);
            return v;
        });

        // ---------------------------------------------------------------
        // Bottom-left: sinusoidal pattern, normalize then exp
        //   raw(x,y) = sin(2*pi*x/256) * cos(2*pi*y/256)  in [-1, 1]
        //   After normalize: in [0, 1]
        //   After exp:       in [1, e] ≈ [1, 2.718]; we then normalize again to [0,1] for display
        // ---------------------------------------------------------------
        auto sinus = make_float_image(half, half, [](int x, int y) {
            return std::sin(2.0f * PI * float(x) / 256.0f) *
                   std::cos(2.0f * PI * float(y) / 256.0f);
        });
        inplace_normalize(sinus, "ex53_norm1");
        inplace_exp(sinus, "ex53_exp");
        // The exp output is in [1, e]; normalize for display
        inplace_normalize(sinus, "ex53_norm2");

        // ---------------------------------------------------------------
        // Bottom-right: correctness check
        //   Compare `processed` (in-place result) against `expected_tr`
        // ---------------------------------------------------------------

        // ---------------------------------------------------------------
        // Compose 512x512 RGB output
        // ---------------------------------------------------------------
        Halide::Runtime::Buffer<uint8_t> img(size, size, 3);

        blit_gray(img, orig,      0,    0);        // top-left
        blit_gray(img, processed, half, 0);        // top-right
        blit_gray(img, sinus,     0,    half);     // bottom-left
        blit_diff(img, processed, expected_tr, half, half);  // bottom-right

        draw_grid(img, size);

        // ---------------------------------------------------------------
        // Save PNG
        // ---------------------------------------------------------------
        const char* path = "out/53_inplace.png";
        if (!save_png(img, path)) {
            std::cerr << "Failed to save " << path << "\n";
            return 1;
        }
        std::cout << "Saved " << path << "\n";
        std::cout << "\nQuadrant guide:\n"
                  << "  Top-left:     Original gradient image\n"
                  << "  Top-right:    scale(8) -> clamp(0,1) -> gamma(0.5)\n"
                  << "  Bottom-left:  sin*cos pattern after normalize -> exp\n"
                  << "  Bottom-right: Correctness check (white=match, red=mismatch)\n";

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
