/// @file 36_distance.cpp
/// @brief Example 36: Distance fields and similarity
///
/// Demonstrates:
///   - binary mask (circle + rectangle shapes)
///   - Euclidean distance field (smooth gradient)
///   - Manhattan distance field (diamond contours)
///   - cosine similarity map (angular pattern)
///
/// Output: out/36_distance.png

#include "numhalide_all.h"
#include "stbi_png.h"
#include <iostream>

using namespace numhalide;

int main(int argc, char **argv) {
    try {
        const int half = 256;
        const int size = half * 2;  // 512x512

        std::cout << "Distance fields demonstration" << std::endl;
        std::cout << "Output size: " << size << "x" << size << std::endl;
        std::cout << std::endl;

        constexpr float pi = 3.14159265358979323846f;

        Halide::Var ox("ox"), oy("oy");
        Halide::Func output("output");

        Halide::Expr qx = ox / half;
        Halide::Expr qy = oy / half;
        Halide::Expr lx = ox % half;
        Halide::Expr ly = oy % half;

        // Normalized coords [0,1)
        Halide::Expr u = Halide::cast<float>(lx) / half;
        Halide::Expr v = Halide::cast<float>(ly) / half;

        // Centered coords [-1,1)
        Halide::Expr cu = u * 2.0f - 1.0f;
        Halide::Expr cv = v * 2.0f - 1.0f;

        // --- Top-left: binary mask (circle + rectangle shapes) ---
        // Circle at center with radius 0.4
        Halide::Expr r = Halide::sqrt(cu * cu + cv * cv);
        Halide::Expr in_circle = r < 0.4f;
        // Rectangle: |x| < 0.6 and |y| < 0.15
        Halide::Expr in_rect = (Halide::abs(cu) < 0.6f) && (Halide::abs(cv) < 0.15f);
        // Union of shapes
        Halide::Expr mask = in_circle || in_rect;
        Halide::Expr mask_val = Halide::select(mask, 1.0f, 0.0f);

        // --- Top-right: Euclidean distance field (smooth gradient) ---
        // Distance from the same shapes: approximate by distance from center circle
        // and distance from rectangle, take minimum
        Halide::Expr dist_circle = Halide::max(r - 0.4f, 0.0f);
        Halide::Expr dist_rect_x = Halide::max(Halide::abs(cu) - 0.6f, 0.0f);
        Halide::Expr dist_rect_y = Halide::max(Halide::abs(cv) - 0.15f, 0.0f);
        Halide::Expr dist_rect = Halide::sqrt(dist_rect_x * dist_rect_x + dist_rect_y * dist_rect_y);
        Halide::Expr euclid_dist = Halide::min(dist_circle, dist_rect);
        // Normalize for visualization (0 at shape, bright far away)
        Halide::Expr euclid_val = Halide::clamp(euclid_dist * 3.0f, 0.0f, 1.0f);

        // --- Bottom-left: Manhattan distance field (diamond contours) ---
        // Manhattan distance from center creates diamond-shaped contours
        Halide::Expr manhattan = Halide::abs(cu) + Halide::abs(cv);
        // Create contour lines with repeating pattern
        Halide::Expr mnh_frac = manhattan * 5.0f;
        Halide::Expr mnh_band = mnh_frac - Halide::floor(mnh_frac);
        Halide::Expr manhattan_val = Halide::select(mnh_band < 0.5f, 0.8f, 0.2f);
        // Fade with distance
        manhattan_val = manhattan_val * Halide::clamp(1.0f - manhattan * 0.5f, 0.1f, 1.0f);

        // --- Bottom-right: cosine similarity map (angular pattern) ---
        // Cosine similarity between pixel direction vector and a reference direction
        // Reference direction: (1, 0.5) normalized
        Halide::Expr ref_len = Halide::sqrt(1.0f + 0.25f);
        Halide::Expr ref_x = 1.0f / ref_len;
        Halide::Expr ref_y = 0.5f / ref_len;
        // Pixel direction
        Halide::Expr pix_len = Halide::sqrt(cu * cu + cv * cv + 0.001f);
        Halide::Expr pix_x = cu / pix_len;
        Halide::Expr pix_y = cv / pix_len;
        // cos(theta) = dot product (already normalized)
        Halide::Expr cos_sim = pix_x * ref_x + pix_y * ref_y;
        // Map from [-1,1] to [0,1]
        Halide::Expr cosine_val = (cos_sim + 1.0f) * 0.5f;

        // Compose quadrants
        Halide::Expr pixel = Halide::select(
            qy == 0 && qx == 0, mask_val,
            qy == 0 && qx == 1, euclid_val,
            qy == 1 && qx == 0, manhattan_val,
            cosine_val
        );

        // Grid lines
        Halide::Expr on_grid = (ox == half) || (oy == half);
        pixel = Halide::select(on_grid, 0.4f, pixel);

        output(ox, oy) = Halide::cast<uint8_t>(Halide::clamp(pixel * 255.0f, 0.0f, 255.0f));

        std::cout << "Rendering..." << std::endl;
        Halide::Runtime::Buffer<uint8_t> result(size, size);
        output.realize(result);

        const char* output_path = "out/36_distance.png";
        std::cout << "Saving to " << output_path << "..." << std::endl;

        if (save_png(result, output_path)) {
            std::cout << "Success!" << std::endl;
            std::cout << std::endl;
            std::cout << "Quadrant guide:" << std::endl;
            std::cout << "  Top-left:     binary mask (circle + rectangle)" << std::endl;
            std::cout << "  Top-right:    Euclidean distance field" << std::endl;
            std::cout << "  Bottom-left:  Manhattan distance field (diamond contours)" << std::endl;
            std::cout << "  Bottom-right: cosine similarity map" << std::endl;
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
