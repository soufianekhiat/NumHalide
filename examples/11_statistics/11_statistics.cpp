/// @file 11_statistics.cpp
/// @brief Example 11: Statistical operations - variance and standard deviation
///
/// Demonstrates:
///   - stats::var() for variance computation
///   - stats::std() for standard deviation
///   - Axis-wise statistics
///   - Visualizing statistical properties of image regions
///
/// Output: out/11_statistics.png

#include "numhalide_all.h"
#include "stbi_png.h"
#include <iostream>
#include <cmath>

using namespace numhalide;

int main(int argc, char **argv) {
    try {
        const int width = 512;
        const int height = 512;
        const int block_size = 32;  // Size of blocks for local statistics

        std::cout << "Statistics demonstration" << std::endl;
        std::cout << "Image size: " << width << "x" << height << std::endl;
        std::cout << "Block size: " << block_size << "x" << block_size << std::endl;
        std::cout << std::endl;

        // Create a test pattern with varying statistics
        // Left half: smooth gradient (low variance)
        // Right half: noisy pattern (high variance)
        Halide::Func pattern("pattern");
        Halide::Var x("x"), y("y");

        // Use a simple hash for pseudo-random noise
        Halide::Expr hash = (x * 73856093) ^ (y * 19349663);
        Halide::Expr noise = Halide::cast<float>((hash % 256)) / 255.0f;

        // Smooth gradient on left, noisy on right
        Halide::Expr smooth = Halide::cast<float>(y) / height;
        Halide::Expr is_left = x < width / 2;
        pattern(x, y) = Halide::select(is_left, smooth, noise);

        // Compute statistics of the entire image
        shape_t full_shape = {height, width};

        // Global mean
        auto global_mean = reduce_mean(pattern, full_shape, "global_mean");
        Halide::Runtime::Buffer<float> mean_buf(1);
        global_mean.realize(mean_buf);
        std::cout << "Global mean: " << mean_buf(0) << std::endl;

        // Global std
        auto global_std = stats::std(pattern, full_shape, "global_std");
        Halide::Runtime::Buffer<float> std_buf(1);
        global_std.realize(std_buf);
        std::cout << "Global std:  " << std_buf(0) << std::endl;

        // Compute statistics along axes
        // Variance along rows (axis 0) - gives variance for each column
        auto var_axis0 = stats::var(pattern, full_shape, 0, false, 0, "var_axis0");
        Halide::Runtime::Buffer<float> var_cols(width);
        var_axis0.realize(var_cols);

        // Find min/max variance across columns
        float min_var = var_cols(0), max_var = var_cols(0);
        for (int i = 1; i < width; ++i) {
            min_var = std::min(min_var, var_cols(i));
            max_var = std::max(max_var, var_cols(i));
        }
        std::cout << "Variance range (per column): [" << min_var << ", " << max_var << "]" << std::endl;
        std::cout << std::endl;

        // Create output visualization
        // Top-left: Original pattern
        // Top-right: Local mean (block-wise)
        // Bottom-left: Local std (block-wise)
        // Bottom-right: High variance mask

        const int out_width = width;
        const int out_height = height;
        const int half_w = out_width / 2;
        const int half_h = out_height / 2;

        // Compute local statistics using block-wise reduction
        int num_blocks_x = width / block_size;
        int num_blocks_y = height / block_size;

        Halide::Func local_mean("local_mean");
        Halide::Func local_std_f("local_std_f");
        Halide::Var bx("bx"), by("by");

        // Block indices
        Halide::RDom rb(0, block_size, 0, block_size);

        // Local mean per block
        local_mean(bx, by) = Halide::cast<float>(0);
        local_mean(bx, by) += pattern(bx * block_size + rb.x, by * block_size + rb.y);
        Halide::Func local_mean_norm("local_mean_norm");
        local_mean_norm(bx, by) = local_mean(bx, by) / (block_size * block_size);

        // Local variance per block: E[X^2] - E[X]^2
        Halide::Func local_sum_sq("local_sum_sq");
        local_sum_sq(bx, by) = Halide::cast<float>(0);
        local_sum_sq(bx, by) += pattern(bx * block_size + rb.x, by * block_size + rb.y) *
                                pattern(bx * block_size + rb.x, by * block_size + rb.y);

        Halide::Func local_var("local_var");
        Halide::Expr mean_sq = local_mean_norm(bx, by) * local_mean_norm(bx, by);
        local_var(bx, by) = local_sum_sq(bx, by) / (block_size * block_size) - mean_sq;

        local_std_f(bx, by) = Halide::sqrt(Halide::max(local_var(bx, by), 0.0f));

        // Build output image as a single Halide pipeline
        Halide::Func output("output");
        Halide::Var ox("ox"), oy("oy");

        // Quadrant selection
        Halide::Expr qx = ox / half_w;
        Halide::Expr qy = oy / half_h;
        Halide::Expr lx = ox % half_w;
        Halide::Expr ly = oy % half_h;

        // Scale local coordinates to original image
        Halide::Expr sx = lx * width / half_w;
        Halide::Expr sy = ly * height / half_h;

        // Block coordinates for local stats
        Halide::Expr block_x = Halide::clamp(sx / block_size, 0, num_blocks_x - 1);
        Halide::Expr block_y = Halide::clamp(sy / block_size, 0, num_blocks_y - 1);

        // Get values from Halide Funcs (not Runtime::Buffers)
        Halide::Expr orig_val = pattern(sx, sy);
        Halide::Expr mean_val = local_mean_norm(block_x, block_y);

        // Normalize std to [0, 1] range (assume max std around 0.3 for visualization)
        Halide::Expr std_val = Halide::clamp(local_std_f(block_x, block_y) / 0.3f, 0.0f, 1.0f);

        // High variance mask (std > 0.1)
        Halide::Expr high_var_mask = Halide::select(
            local_std_f(block_x, block_y) > 0.1f,
            1.0f,
            0.3f
        );

        // Combine quadrants
        Halide::Expr pixel_val = Halide::select(
            qy == 0 && qx == 0, orig_val,
            Halide::select(
                qy == 0 && qx == 1, mean_val,
                Halide::select(
                    qy == 1 && qx == 0, std_val,
                    high_var_mask
                )
            )
        );

        output(ox, oy) = Halide::cast<uint8_t>(Halide::clamp(pixel_val * 255.0f, 0.0f, 255.0f));

        // Schedule for performance
        local_mean.compute_root();
        local_mean_norm.compute_root();
        local_sum_sq.compute_root();
        local_var.compute_root();
        local_std_f.compute_root();

        // Realize output
        std::cout << "Rendering output..." << std::endl;
        Halide::Runtime::Buffer<uint8_t> result(out_width, out_height);
        output.realize(result);

        // Save
        const char* output_path = "out/11_statistics.png";
        std::cout << "Saving to " << output_path << "..." << std::endl;

        if (save_png(result, output_path)) {
            std::cout << "Success! Statistics visualization saved." << std::endl;
            std::cout << std::endl;
            std::cout << "Quadrant guide:" << std::endl;
            std::cout << "  Top-left:     Original pattern (gradient left, noise right)" << std::endl;
            std::cout << "  Top-right:    Local mean (block-wise average)" << std::endl;
            std::cout << "  Bottom-left:  Local std (brighter = higher variance)" << std::endl;
            std::cout << "  Bottom-right: High variance mask (bright = high std regions)" << std::endl;
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
