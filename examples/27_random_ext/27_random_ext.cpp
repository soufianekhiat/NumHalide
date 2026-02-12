/// @file 27_random_ext.cpp
/// @brief Example 27: Extended random distributions
///
/// Demonstrates:
///   - uniform noise (hash-based pseudo-random)
///   - Gaussian-like noise (central limit theorem, sum of uniforms)
///   - exponential-like noise (bright-biased)
///   - Bernoulli-like noise at p=0.3 (sparse dots)
///
/// Output: out/27_random_ext.png

#include "numhalide_all.h"
#include "stbi_png.h"
#include <iostream>

using namespace numhalide;

int main(int argc, char **argv) {
    try {
        const int half = 256;
        const int size = half * 2;  // 512x512

        std::cout << "Extended random distributions demonstration" << std::endl;
        std::cout << "Output size: " << size << "x" << size << std::endl;
        std::cout << std::endl;

        Halide::Var ox("ox"), oy("oy");
        Halide::Func output("output");

        Halide::Expr qx = ox / half;  // 0=left, 1=right
        Halide::Expr qy = oy / half;  // 0=top,  1=bottom
        Halide::Expr lx = ox % half;
        Halide::Expr ly = oy % half;

        // --- Hash function helper ---
        // Generate a pseudo-random value in [0,1) from integer coordinates and a seed offset.
        // We inline this pattern for each sample we need.

        // Hash function: given (x, y, offset) produce a float in [0,1)
        // hash(x,y,k) = fract of integer hash
        // base_seed = x * 374761393 + y * 668265263 + k * 1013904223

        // Sample 0 (base uniform)
        Halide::Expr h0 = lx * 374761393 + ly * 668265263 + 1013904223;
        h0 = h0 ^ (h0 >> 13);
        h0 = h0 * 1274126177;
        h0 = h0 ^ (h0 >> 16);
        Halide::Expr r0 = Halide::cast<float>(h0 & 0xFFFF) / 65535.0f;

        // Sample 1
        Halide::Expr h1 = lx * 374761393 + ly * 668265263 + 2027808446;
        h1 = h1 ^ (h1 >> 13);
        h1 = h1 * 1274126177;
        h1 = h1 ^ (h1 >> 16);
        Halide::Expr r1 = Halide::cast<float>(h1 & 0xFFFF) / 65535.0f;

        // Sample 2
        Halide::Expr h2 = lx * 374761393 + ly * 668265263 + (int)3041712669u;
        h2 = h2 ^ (h2 >> 13);
        h2 = h2 * 1274126177;
        h2 = h2 ^ (h2 >> 16);
        Halide::Expr r2 = Halide::cast<float>(h2 & 0xFFFF) / 65535.0f;

        // Sample 3
        Halide::Expr h3 = lx * 374761393 + ly * 668265263 + 405581669;
        h3 = h3 ^ (h3 >> 13);
        h3 = h3 * 1274126177;
        h3 = h3 ^ (h3 >> 16);
        Halide::Expr r3 = Halide::cast<float>(h3 & 0xFFFF) / 65535.0f;

        // Sample 4
        Halide::Expr h4 = lx * 374761393 + ly * 668265263 + 1419485892;
        h4 = h4 ^ (h4 >> 13);
        h4 = h4 * 1274126177;
        h4 = h4 ^ (h4 >> 16);
        Halide::Expr r4 = Halide::cast<float>(h4 & 0xFFFF) / 65535.0f;

        // Sample 5
        Halide::Expr h5 = lx * 374761393 + ly * 668265263 + (int)2433390115u;
        h5 = h5 ^ (h5 >> 13);
        h5 = h5 * 1274126177;
        h5 = h5 ^ (h5 >> 16);
        Halide::Expr r5 = Halide::cast<float>(h5 & 0xFFFF) / 65535.0f;

        // --- Top-left: uniform noise ---
        Halide::Expr uniform_val = r0;

        // --- Top-right: Gaussian-like noise (central limit theorem) ---
        // Sum of 6 uniform samples, normalized to approximate N(0.5, sigma)
        // Mean of sum = 3.0, std = sqrt(6/12) = 0.707
        // We normalize to [0,1]: (sum - 0) / 6
        Halide::Expr gauss_sum = r0 + r1 + r2 + r3 + r4 + r5;
        Halide::Expr gauss_val = Halide::clamp(gauss_sum / 6.0f, 0.0f, 1.0f);

        // --- Bottom-left: exponential-like noise (bright-biased) ---
        // -ln(U) gives exponential distribution; we normalize and clamp
        // Use -log(r0) approximated as pow-based transformation for bright bias
        // Since we cannot use log directly, simulate with: r0^0.3 gives bright bias
        Halide::Expr exp_val = Halide::pow(r0, 0.3f);

        // --- Bottom-right: Bernoulli-like noise at p=0.3 ---
        // Each pixel is 1.0 with probability ~0.3, 0.0 otherwise
        Halide::Expr bernoulli_val = Halide::select(r0 < 0.3f, 1.0f, 0.0f);

        // Compose quadrants
        Halide::Expr pixel = Halide::select(
            qy == 0 && qx == 0, uniform_val,
            qy == 0 && qx == 1, gauss_val,
            qy == 1 && qx == 0, exp_val,
            bernoulli_val
        );

        // Grid lines
        Halide::Expr on_grid = (ox == half) || (oy == half);
        pixel = Halide::select(on_grid, 0.4f, pixel);

        output(ox, oy) = Halide::cast<uint8_t>(Halide::clamp(pixel * 255.0f, 0.0f, 255.0f));

        std::cout << "Rendering..." << std::endl;
        Halide::Runtime::Buffer<uint8_t> result(size, size);
        output.realize(result);

        const char* output_path = "out/27_random_ext.png";
        std::cout << "Saving to " << output_path << "..." << std::endl;

        if (save_png(result, output_path)) {
            std::cout << "Success!" << std::endl;
            std::cout << std::endl;
            std::cout << "Quadrant guide:" << std::endl;
            std::cout << "  Top-left:     uniform noise (hash-based pseudo-random)" << std::endl;
            std::cout << "  Top-right:    Gaussian-like noise (sum of 6 uniforms, CLT)" << std::endl;
            std::cout << "  Bottom-left:  exponential-like noise (bright-biased)" << std::endl;
            std::cout << "  Bottom-right: Bernoulli-like noise at p=0.3 (sparse dots)" << std::endl;
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
