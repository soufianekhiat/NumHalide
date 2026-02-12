/// @file 26_statistics_ext.cpp
/// @brief Example 26: Extended statistics operations
///
/// Demonstrates:
///   - random noise pattern (hash-based)
///   - histogram-like bar chart rendered as an image
///   - horizontal median-like smooth strip
///   - peak-to-peak intensity visualization
///
/// Output: out/26_statistics_ext.png

#include "numhalide_all.h"
#include "stbi_png.h"
#include <iostream>

using namespace numhalide;

int main(int argc, char **argv) {
    try {
        const int half = 256;
        const int size = half * 2;  // 512x512

        std::cout << "Extended statistics demonstration" << std::endl;
        std::cout << "Output size: " << size << "x" << size << std::endl;
        std::cout << std::endl;

        Halide::Var ox("ox"), oy("oy");
        Halide::Func output("output");

        Halide::Expr qx = ox / half;  // 0=left, 1=right
        Halide::Expr qy = oy / half;  // 0=top,  1=bottom
        Halide::Expr lx = ox % half;
        Halide::Expr ly = oy % half;

        // Normalized coords [0,1)
        Halide::Expr u = Halide::cast<float>(lx) / half;
        Halide::Expr v = Halide::cast<float>(ly) / half;

        // --- Hash-based pseudo-random function ---
        // Using integer hashing to generate deterministic noise
        Halide::Expr seed = lx * 374761393 + ly * 668265263 + 1013904223;
        seed = seed ^ (seed >> 13);
        seed = seed * 1274126177;
        seed = seed ^ (seed >> 16);
        Halide::Expr hash_val = Halide::cast<float>(seed & 0xFFFF) / 65535.0f;

        // --- Top-left: random noise pattern ---
        Halide::Expr noise_val = hash_val;

        // --- Top-right: histogram-like bar chart ---
        // Create vertical bars whose height represents a simulated histogram.
        // The bar height is based on a Gaussian-like distribution centered at mid-x.
        Halide::Expr bar_idx = lx / (half / 16);  // 16 bars
        Halide::Expr bar_center = Halide::cast<float>(bar_idx) / 16.0f;
        // Gaussian envelope centered at 0.5
        Halide::Expr bar_dist = (bar_center - 0.5f);
        Halide::Expr bar_height = Halide::pow(2.718281828f, -8.0f * bar_dist * bar_dist);
        // Draw the bar: pixel is lit if v > (1 - bar_height)
        // v=0 is top, v=1 is bottom, so we want bar to grow upward from bottom
        Halide::Expr bar_threshold = 1.0f - bar_height;
        Halide::Expr in_bar = v > bar_threshold;
        // Add bar edges for clarity
        Halide::Expr bar_local_x = lx % (half / 16);
        Halide::Expr bar_edge = (bar_local_x == 0) || (bar_local_x == (half / 16 - 1));
        Halide::Expr hist_val = Halide::select(
            bar_edge, 0.2f,
            in_bar, 0.85f,
            0.05f
        );

        // --- Bottom-left: horizontal median-like smooth strip ---
        // Simulates the effect of a median filter by averaging multiple hash samples
        // at nearby positions, creating smooth horizontal bands
        Halide::Expr s1_seed = lx * 374761393 + (ly / 4) * 668265263 + 1013904223;
        s1_seed = s1_seed ^ (s1_seed >> 13);
        s1_seed = s1_seed * 1274126177;
        s1_seed = s1_seed ^ (s1_seed >> 16);
        Halide::Expr s1 = Halide::cast<float>(s1_seed & 0xFFFF) / 65535.0f;

        Halide::Expr s2_seed = (lx + 1) * 374761393 + (ly / 4) * 668265263 + 1013904223;
        s2_seed = s2_seed ^ (s2_seed >> 13);
        s2_seed = s2_seed * 1274126177;
        s2_seed = s2_seed ^ (s2_seed >> 16);
        Halide::Expr s2 = Halide::cast<float>(s2_seed & 0xFFFF) / 65535.0f;

        Halide::Expr s3_seed = (lx - 1) * 374761393 + (ly / 4) * 668265263 + 1013904223;
        s3_seed = s3_seed ^ (s3_seed >> 13);
        s3_seed = s3_seed * 1274126177;
        s3_seed = s3_seed ^ (s3_seed >> 16);
        Halide::Expr s3 = Halide::cast<float>(s3_seed & 0xFFFF) / 65535.0f;

        // Approximate median of 3 values: clamp the middle
        Halide::Expr median_val = Halide::max(Halide::min(s1, s2), Halide::min(Halide::max(s1, s2), s3));

        // --- Bottom-right: peak-to-peak intensity visualization ---
        // Shows the range (max - min) of a local region as brightness.
        // Uses several hash samples to compute a local range estimate.
        Halide::Expr p1 = s1;
        Halide::Expr p2 = s2;
        Halide::Expr p3 = s3;
        Halide::Expr local_max = Halide::max(Halide::max(p1, p2), p3);
        Halide::Expr local_min = Halide::min(Halide::min(p1, p2), p3);
        Halide::Expr ptp_val = local_max - local_min;

        // Compose quadrants
        Halide::Expr pixel = Halide::select(
            qy == 0 && qx == 0, noise_val,
            qy == 0 && qx == 1, hist_val,
            qy == 1 && qx == 0, median_val,
            ptp_val
        );

        // Grid lines
        Halide::Expr on_grid = (ox == half) || (oy == half);
        pixel = Halide::select(on_grid, 0.4f, pixel);

        output(ox, oy) = Halide::cast<uint8_t>(Halide::clamp(pixel * 255.0f, 0.0f, 255.0f));

        std::cout << "Rendering..." << std::endl;
        Halide::Runtime::Buffer<uint8_t> result(size, size);
        output.realize(result);

        const char* output_path = "out/26_statistics_ext.png";
        std::cout << "Saving to " << output_path << "..." << std::endl;

        if (save_png(result, output_path)) {
            std::cout << "Success!" << std::endl;
            std::cout << std::endl;
            std::cout << "Quadrant guide:" << std::endl;
            std::cout << "  Top-left:     random noise pattern (hash-based)" << std::endl;
            std::cout << "  Top-right:    histogram-like bar chart" << std::endl;
            std::cout << "  Bottom-left:  horizontal median-like smooth strip" << std::endl;
            std::cout << "  Bottom-right: peak-to-peak intensity visualization" << std::endl;
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
