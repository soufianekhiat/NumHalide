/// @file 16_sorting.cpp
/// @brief Example 16: Sorting and search operations
///
/// Demonstrates:
///   - argmin/argmax for finding extrema indices
///   - bitonic_sort for power-of-2 sorting
///   - bitonic_argsort for getting sorted indices
///
/// Output: out/16_sorting.png

#include "numhalide_all.h"
#include "stbi_png.h"
#include <iostream>

using namespace numhalide;

int main(int argc, char **argv) {
    try {
        const int width = 256;
        const int height = 256;

        std::cout << "Sorting operations demonstration" << std::endl;
        std::cout << "Output size: " << width << "x" << height << std::endl;
        std::cout << std::endl;

        // Create a noisy signal and sort segments of it
        Halide::Func signal("signal");
        Halide::Var x("x"), y("y");

        // Create a pattern showing sorting: top half unsorted, bottom half sorted
        // Each row segment of 8 pixels will be sorted

        // Generate pseudo-random values
        Halide::Expr noise = Halide::cast<float>(
            ((x * 7 + y * 13 + 17) * 31 % 256)
        ) / 255.0f;

        // Create 8-element segments that we'll visualize
        const int segment_size = 8;

        // Top half: show original values
        // Bottom half: show sorted values (using bitonic sort on segments)

        // Create a sorted version of each 8-pixel segment
        Halide::Func sorted_segment("sorted_segment");
        Halide::Func segment_input("segment_input");

        // For demonstration, create a simple 8-element sort visualization
        // We'll show: unsorted bar graph on top, sorted bar graph on bottom

        Halide::Func output("output");

        // Create bar graph visualization
        // x ranges over width (32 bars of 8 pixels each)
        // y ranges over height

        int num_bars = width / segment_size;
        Halide::Expr bar_idx = x / segment_size;
        Halide::Expr local_x = x % segment_size;

        // Generate values for each bar (deterministic pattern)
        Halide::Func bar_values("bar_values");
        bar_values(x) = Halide::cast<float>(
            ((x * 31 + 17) * 7 % 256)
        ) / 256.0f;

        // Create sorted version using bitonic sort
        auto sorted_values = bitonic_sort(bar_values, 256, "sorted");

        // Argmin/argmax demonstration
        shape_t values_shape = {256};
        auto min_idx = argmin(bar_values, values_shape, "min_idx");
        auto max_idx = argmax(bar_values, values_shape, "max_idx");

        // For each pixel, compute the bar height and check if we're below it
        Halide::Expr section = y / (height / 2);  // 0 = top (unsorted), 1 = bottom (sorted)
        Halide::Expr local_y = y % (height / 2);

        // Get bar heights (scaled to half height)
        Halide::Expr unsorted_height = bar_values(x) * (height / 2 - 20);
        Halide::Expr sorted_height = sorted_values(x) * (height / 2 - 20);

        // Draw bars: white if y is below bar height, dark otherwise
        Halide::Expr in_unsorted_bar = (height / 2 - 10 - local_y) < unsorted_height;
        Halide::Expr in_sorted_bar = (height / 2 - 10 - local_y) < sorted_height;

        // Color coding: highlight min (blue tint) and max (red tint) in original
        Halide::Expr is_min = (x == min_idx(0));
        Halide::Expr is_max = (x == max_idx(0));

        // RGB output
        Halide::Func output_r("output_r"), output_g("output_g"), output_b("output_b");

        // Top half: unsorted bars
        // Bottom half: sorted bars
        Halide::Expr in_bar = Halide::select(section == 0, in_unsorted_bar, in_sorted_bar);

        // Base gray level
        Halide::Expr gray = Halide::select(in_bar, 220, 40);

        // Add color for min/max in top section
        output_r(x, y) = Halide::cast<uint8_t>(Halide::select(
            section == 0 && is_max && in_bar, 255,
            Halide::select(section == 0 && is_min && in_bar, 100, gray)
        ));
        output_g(x, y) = Halide::cast<uint8_t>(Halide::select(
            section == 0 && (is_min || is_max) && in_bar, 100, gray
        ));
        output_b(x, y) = Halide::cast<uint8_t>(Halide::select(
            section == 0 && is_min && in_bar, 255,
            Halide::select(section == 0 && is_max && in_bar, 100, gray)
        ));

        // Add separator line
        Halide::Expr on_separator = (y == height / 2) || (y == height / 2 + 1);
        output_r(x, y) = Halide::select(on_separator, Halide::cast<uint8_t>(128), output_r(x, y));
        output_g(x, y) = Halide::select(on_separator, Halide::cast<uint8_t>(128), output_g(x, y));
        output_b(x, y) = Halide::select(on_separator, Halide::cast<uint8_t>(128), output_b(x, y));

        // Realize to RGB buffer
        std::cout << "Rendering..." << std::endl;
        Halide::Runtime::Buffer<uint8_t> result(width, height, 3);

        Halide::Func final_output("final_output");
        Halide::Var c("c");
        final_output(x, y, c) = Halide::select(
            c == 0, output_r(x, y),
            Halide::select(c == 1, output_g(x, y), output_b(x, y))
        );

        final_output.realize(result);

        // Print statistics
        Halide::Runtime::Buffer<int32_t> min_out(1), max_out(1);
        min_idx.realize(min_out);
        max_idx.realize(max_out);

        std::cout << "Array statistics:" << std::endl;
        std::cout << "  Minimum value at index: " << min_out(0) << std::endl;
        std::cout << "  Maximum value at index: " << max_out(0) << std::endl;
        std::cout << std::endl;

        // Save
        const char* output_path = "out/16_sorting.png";
        std::cout << "Saving to " << output_path << "..." << std::endl;

        if (save_png(result, output_path)) {
            std::cout << "Success! Sorting visualization saved." << std::endl;
            std::cout << std::endl;
            std::cout << "Image guide:" << std::endl;
            std::cout << "  Top half:    Unsorted bar values (red=max, blue=min)" << std::endl;
            std::cout << "  Bottom half: Sorted bar values (ascending order)" << std::endl;
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
