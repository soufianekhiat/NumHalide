/// @file 12_bool_reduce.cpp
/// @brief Example 12: Boolean reductions - any, all, count_nonzero
///
/// Demonstrates:
///   - reduce_any() to check if any element meets a condition
///   - reduce_all() to check if all elements meet a condition
///   - count_nonzero() to count elements meeting a condition
///   - Using boolean reductions for image segmentation
///
/// Output: out/12_bool_reduce.png

#include "numhalide_all.h"
#include "stbi_png.h"
#include <iostream>

using namespace numhalide;

int main(int argc, char **argv) {
    try {
        const int width = 512;
        const int height = 512;

        std::cout << "Boolean reductions demonstration" << std::endl;
        std::cout << "Image size: " << width << "x" << height << std::endl;
        std::cout << std::endl;

        // Create a test pattern with circles at different positions
        Halide::Func pattern("pattern");
        Halide::Var x("x"), y("y");

        // Create circles at different positions
        Halide::Expr cx1 = width / 4, cy1 = height / 4;      // Top-left circle
        Halide::Expr cx2 = 3 * width / 4, cy2 = height / 4;  // Top-right circle
        Halide::Expr cx3 = width / 2, cy3 = 3 * height / 4;  // Bottom center circle
        Halide::Expr radius = 80;

        Halide::Expr dist1 = (x - cx1) * (x - cx1) + (y - cy1) * (y - cy1);
        Halide::Expr dist2 = (x - cx2) * (x - cx2) + (y - cy2) * (y - cy2);
        Halide::Expr dist3 = (x - cx3) * (x - cx3) + (y - cy3) * (y - cy3);

        // Inside any circle = 1, outside = 0
        Halide::Expr in_circle = (dist1 < radius * radius) ||
                                  (dist2 < radius * radius) ||
                                  (dist3 < radius * radius);
        pattern(x, y) = Halide::select(in_circle, 1.0f, 0.0f);

        shape_t full_shape = {height, width};

        // Count total non-zero pixels (pixels inside circles)
        auto total_count = count_nonzero(pattern, full_shape, "total_count");
        Halide::Runtime::Buffer<int32_t> count_buf(1);
        total_count.realize(count_buf);
        std::cout << "Total pixels inside circles: " << count_buf(0) << std::endl;
        std::cout << "Percentage of image: " << (100.0f * count_buf(0) / (width * height)) << "%" << std::endl;

        // Check if any pixel is non-zero (should be true)
        auto has_any = reduce_any(pattern, full_shape, "has_any");
        Halide::Runtime::Buffer<int32_t> any_buf(1);
        has_any.realize(any_buf);
        std::cout << "Has any filled pixels: " << (any_buf(0) ? "yes" : "no") << std::endl;

        // Check if all pixels are non-zero (should be false)
        auto has_all = reduce_all(pattern, full_shape, "has_all");
        Halide::Runtime::Buffer<int32_t> all_buf(1);
        has_all.realize(all_buf);
        std::cout << "All pixels filled: " << (all_buf(0) ? "yes" : "no") << std::endl;

        // Count non-zero per row (axis 1)
        auto count_per_row = count_nonzero(pattern, full_shape, 1, false, "count_per_row");
        Halide::Runtime::Buffer<int32_t> row_counts(height);
        count_per_row.realize(row_counts);

        // Find rows with most filled pixels
        int max_row = 0, max_count = 0;
        for (int i = 0; i < height; ++i) {
            if (row_counts(i) > max_count) {
                max_count = row_counts(i);
                max_row = i;
            }
        }
        std::cout << "Row with most filled pixels: " << max_row << " (" << max_count << " pixels)" << std::endl;

        // Check which rows have any filled pixels
        auto any_per_row = reduce_any(pattern, full_shape, 1, false, "any_per_row");
        Halide::Runtime::Buffer<int32_t> row_any(height);
        any_per_row.realize(row_any);

        int first_row = -1, last_row = -1;
        for (int i = 0; i < height; ++i) {
            if (row_any(i) && first_row < 0) first_row = i;
            if (row_any(i)) last_row = i;
        }
        std::cout << "Filled rows span: " << first_row << " to " << last_row << std::endl;
        std::cout << std::endl;

        // Create output visualization
        // Top-left: Original pattern
        // Top-right: Row-wise any (shows which rows have content)
        // Bottom-left: Column-wise count (brightness = count)
        // Bottom-right: Threshold mask (rows with >50 pixels)

        const int half_w = width / 2;
        const int half_h = height / 2;

        // Row-wise any visualization (as vertical bars)
        Halide::Func row_any_viz("row_any_viz");
        row_any_viz(x, y) = Halide::cast<float>(any_per_row(y));

        // Column-wise count
        auto count_per_col = count_nonzero(pattern, full_shape, 0, false, "count_per_col");

        // Normalize count for visualization
        Halide::Func col_count_viz("col_count_viz");
        col_count_viz(x, y) = Halide::cast<float>(count_per_col(x)) / Halide::cast<float>(height);

        // Threshold mask: rows with more than threshold pixels
        const int threshold = 50;
        Halide::Func threshold_mask("threshold_mask");
        threshold_mask(x, y) = Halide::select(
            count_per_row(y) > threshold,
            pattern(x, y),
            0.3f * pattern(x, y)
        );

        // Build output
        Halide::Func output("output");
        Halide::Var ox("ox"), oy("oy");

        Halide::Expr qx = ox / half_w;
        Halide::Expr qy = oy / half_h;
        Halide::Expr lx = ox % half_w;
        Halide::Expr ly = oy % half_h;

        // Scale to full image coordinates
        Halide::Expr sx = lx * 2;
        Halide::Expr sy = ly * 2;

        Halide::Expr pixel_val = Halide::select(
            qy == 0 && qx == 0, pattern(sx, sy),
            Halide::select(
                qy == 0 && qx == 1, row_any_viz(sx, sy),
                Halide::select(
                    qy == 1 && qx == 0, col_count_viz(sx, sy),
                    threshold_mask(sx, sy)
                )
            )
        );

        output(ox, oy) = Halide::cast<uint8_t>(Halide::clamp(pixel_val * 255.0f, 0.0f, 255.0f));

        // Schedule
        count_per_row.compute_root();
        count_per_col.compute_root();
        any_per_row.compute_root();

        // Realize
        std::cout << "Rendering output..." << std::endl;
        Halide::Runtime::Buffer<uint8_t> result(width, height);
        output.realize(result);

        // Save
        const char* output_path = "out/12_bool_reduce.png";
        std::cout << "Saving to " << output_path << "..." << std::endl;

        if (save_png(result, output_path)) {
            std::cout << "Success! Boolean reduction visualization saved." << std::endl;
            std::cout << std::endl;
            std::cout << "Quadrant guide:" << std::endl;
            std::cout << "  Top-left:     Original circles pattern" << std::endl;
            std::cout << "  Top-right:    Row-wise any (white = row has content)" << std::endl;
            std::cout << "  Bottom-left:  Column-wise count (brighter = more pixels)" << std::endl;
            std::cout << "  Bottom-right: Threshold mask (bright = rows with >" << threshold << " pixels)" << std::endl;
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
