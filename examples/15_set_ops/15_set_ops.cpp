/// @file 15_set_ops.cpp
/// @brief Example 15: Set operations
///
/// Demonstrates:
///   - Bitonic sorting
///   - Unique element detection
///   - Unique element counting
///
/// Output: out/15_set_ops.png

#include "numhalide_all.h"
#include "stbi_png.h"
#include <iostream>
#include <vector>

using namespace numhalide;

int main(int argc, char **argv) {
    try {
        const int width = 256;
        const int height = 256;

        std::cout << "Set operations demonstration" << std::endl;
        std::cout << "Output size: " << width << "x" << height << std::endl;
        std::cout << std::endl;

        // Create two sets of values (power of 2 size for bitonic sort)
        const int set_size = 8;

        // Set A: [3, 1, 4, 1, 5, 9, 2, 6] (with duplicate 1)
        Halide::Func set_a("set_a");
        Halide::Var x("x");
        set_a(x) = Halide::select(x == 0, 3,
                   Halide::select(x == 1, 1,
                   Halide::select(x == 2, 4,
                   Halide::select(x == 3, 1,
                   Halide::select(x == 4, 5,
                   Halide::select(x == 5, 9,
                   Halide::select(x == 6, 2, 6)))))));

        // Set B: [5, 3, 5, 8, 9, 7, 9, 3] (with duplicates)
        Halide::Func set_b("set_b");
        set_b(x) = Halide::select(x == 0, 5,
                   Halide::select(x == 1, 3,
                   Halide::select(x == 2, 5,
                   Halide::select(x == 3, 8,
                   Halide::select(x == 4, 9,
                   Halide::select(x == 5, 7,
                   Halide::select(x == 6, 9, 3)))))));

        std::cout << "Set A (unsorted): [3, 1, 4, 1, 5, 9, 2, 6]" << std::endl;
        std::cout << "Set B (unsorted): [5, 3, 5, 8, 9, 7, 9, 3]" << std::endl;
        std::cout << std::endl;

        // Sort both sets
        auto sorted_a = bitonic_sort(set_a, set_size, "sorted_a");
        auto sorted_b = bitonic_sort(set_b, set_size, "sorted_b");

        // Display sorted sets
        Halide::Runtime::Buffer<int32_t> sa_out(set_size), sb_out(set_size);
        sorted_a.realize(sa_out);
        sorted_b.realize(sb_out);

        std::cout << "Set A (sorted): [";
        for (int i = 0; i < set_size; i++) {
            std::cout << sa_out(i);
            if (i < set_size - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        std::cout << "Set B (sorted): [";
        for (int i = 0; i < set_size; i++) {
            std::cout << sb_out(i);
            if (i < set_size - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << std::endl;

        // Count unique elements
        auto count_a = count_unique(sorted_a, set_size, "count_a");
        auto count_b = count_unique(sorted_b, set_size, "count_b");

        Halide::Runtime::Buffer<int32_t> ca_out(1), cb_out(1);
        count_a.realize(ca_out);
        count_b.realize(cb_out);

        std::cout << "Unique elements in A: " << ca_out(0) << std::endl;
        std::cout << "Unique elements in B: " << cb_out(0) << std::endl;
        std::cout << std::endl;

        // Create visualization
        // Top-left: Set A visualization (bar chart)
        // Top-right: Set B visualization (bar chart)
        // Bottom-left: Sorted A with unique markers
        // Bottom-right: Sorted B with unique markers

        const int half = width / 2;
        const int bar_width = half / set_size;
        const int max_val = 10;

        Halide::Func output("output");
        Halide::Var ox("ox"), oy("oy");

        Halide::Expr qx = ox / half;
        Halide::Expr qy = oy / half;
        Halide::Expr local_x = ox % half;
        Halide::Expr local_y = oy % half;

        Halide::Expr bar_idx = local_x / bar_width;
        bar_idx = Halide::clamp(bar_idx, 0, set_size - 1);

        // Get values for each quadrant
        Halide::Expr val_a = sorted_a(bar_idx);
        Halide::Expr val_b = sorted_b(bar_idx);

        // Calculate bar heights (inverted y)
        Halide::Expr bar_height_a = (val_a * (half - 20)) / max_val;
        Halide::Expr bar_height_b = (val_b * (half - 20)) / max_val;

        Halide::Expr in_bar_a = (half - 10 - local_y) < bar_height_a;
        Halide::Expr in_bar_b = (half - 10 - local_y) < bar_height_b;

        // Color based on quadrant
        Halide::Expr pixel = Halide::select(
            qy == 0 && qx == 0, Halide::select(in_bar_a, 0.8f, 0.15f),
            Halide::select(
                qy == 0 && qx == 1, Halide::select(in_bar_b, 0.8f, 0.15f),
                Halide::select(
                    qy == 1 && qx == 0,
                    // Bottom left: unique count for A
                    Halide::select(local_x < half / 2,
                        Halide::select((half - 10 - local_y) < (ca_out(0) * (half - 20) / max_val), 0.6f, 0.15f),
                        0.15f),
                    // Bottom right: unique count for B
                    Halide::select(local_x < half / 2,
                        Halide::select((half - 10 - local_y) < (cb_out(0) * (half - 20) / max_val), 0.7f, 0.15f),
                        0.15f)
                )
            )
        );

        // Add grid lines
        Halide::Expr on_grid = (ox == half) || (oy == half);
        pixel = Halide::select(on_grid, 0.4f, pixel);

        output(ox, oy) = Halide::cast<uint8_t>(Halide::clamp(pixel * 255.0f, 0.0f, 255.0f));

        // Realize
        std::cout << "Rendering visualization..." << std::endl;
        Halide::Runtime::Buffer<uint8_t> result(width, height);
        output.realize(result);

        // Save
        const char* output_path = "out/15_set_ops.png";
        std::cout << "Saving to " << output_path << "..." << std::endl;

        if (save_png(result, output_path)) {
            std::cout << "Success! Set operations visualization saved." << std::endl;
            std::cout << std::endl;
            std::cout << "Quadrant guide:" << std::endl;
            std::cout << "  Top-left:     Sorted Set A (bar heights = values)" << std::endl;
            std::cout << "  Top-right:    Sorted Set B (bar heights = values)" << std::endl;
            std::cout << "  Bottom-left:  Unique count in A (bar height = count)" << std::endl;
            std::cout << "  Bottom-right: Unique count in B (bar height = count)" << std::endl;
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
