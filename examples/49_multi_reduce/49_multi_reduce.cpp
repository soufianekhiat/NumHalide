/// @file 49_multi_reduce.cpp
/// @brief Example 49: Multi-axis reductions on 3D data cube
///
/// Quadrant guide:
///   Top-left:     3D data cube max-projected along Z (depth) -> 64x64 image
///   Top-right:    Mean across rows (axis 1) -> per-column-depth means
///   Bottom-left:  Sum along axes {0,2} (D and W) -> per-row profile bars
///   Bottom-right: De-mean normalization (subtract per-depth mean, keepdims)

#include "numhalide_all.h"
#include "stbi_png.h"
#include <iostream>
#include <cmath>

using namespace numhalide;
static const float PI = 3.14159265358979323846f;

int main(int argc, char** argv) {
    try {
        const int half = 256;
        const int size = half * 2;

        // 3D data cube: D=4, H=64, W=64
        // (shape {4, 64, 64})
        const int D = 4, H = 64, W = 64;

        // Generate data: each slice d has a 2D Gaussian + sinusoidal pattern
        Halide::Buffer<float> cube(W, H, D);
        for (int d = 0; d < D; ++d) {
            float cx = W / 2.0f + 15.0f * std::cos(2.0f * PI * d / D);
            float cy = H / 2.0f + 15.0f * std::sin(2.0f * PI * d / D);
            for (int r = 0; r < H; ++r) {
                for (int c = 0; c < W; ++c) {
                    float dx = c - cx, dy = r - cy;
                    float gauss = std::exp(-(dx*dx + dy*dy) / (2.0f * 10.0f * 10.0f));
                    float wave  = 0.3f * std::sin(2.0f * PI * c / W * 3.0f + (float)d);
                    cube(c, r, d) = gauss + wave + 0.5f;
                }
            }
        }

        // Create 3D Func
        Halide::Func f3d("cube");
        Halide::Var x("x"), y("y"), z("z");
        f3d(x, y, z) = cube(Halide::clamp(x, 0, W-1),
                            Halide::clamp(y, 0, H-1),
                            Halide::clamp(z, 0, D-1));

        shape_t shape3d = {D, H, W};

        // --- T2-C: Max projection along axis 0 (depth D) -> shape {H, W}
        auto max_proj_f = reduce_max(f3d, shape3d, {0}, false, "max_proj");
        Halide::Runtime::Buffer<float> max_proj(W, H);
        max_proj_f.realize(max_proj);

        // --- Mean across rows axis 1 (H) -> shape {D, W} = {4, 64}
        auto row_mean_f = reduce_mean(f3d, shape3d, {1}, false, "row_mean");
        Halide::Runtime::Buffer<float> row_mean(W, D);  // (W, D)
        row_mean_f.realize(row_mean);

        // --- Sum along axes {0, 2} (D and W) -> shape {H} = {64}
        auto col_sum_f = reduce_sum(f3d, shape3d, {0, 2}, false, "col_sum");
        Halide::Runtime::Buffer<float> col_sum(H);
        col_sum_f.realize(col_sum);

        // --- De-mean along axes {1, 2} keepdims -> shape {D, 1, 1}
        // per-depth global mean
        auto depth_mean_f = reduce_mean(f3d, shape3d, {1, 2}, true, "depth_mean");
        // demean = f3d - depth_mean_f
        Halide::Func demean("demean");
        demean(x, y, z) = f3d(x, y, z) - depth_mean_f(0, 0, z);
        Halide::Runtime::Buffer<float> demean_buf(W, H);
        // Realize demean for depth z=0
        Halide::Func dm0("dm0");
        dm0(x, y) = demean(x, y, 0);
        dm0.realize(demean_buf);

        // ---- Compose 512x512 output ----
        Halide::Var ox("ox"), oy("oy");
        Halide::Expr qx = ox / half, qy = oy / half;
        Halide::Expr lx = ox % half, ly = oy % half;

        // Normalize max_proj to [0,1]
        float mp_min = max_proj(0, 0), mp_max = max_proj(0, 0);
        for (int r = 0; r < H; ++r)
            for (int c = 0; c < W; ++c) {
                mp_min = std::min(mp_min, max_proj(c, r));
                mp_max = std::max(mp_max, max_proj(c, r));
            }
        float mp_range = std::max(mp_max - mp_min, 1e-6f);
        Halide::Buffer<float> mp_norm(W, H);
        for (int r = 0; r < H; ++r)
            for (int c = 0; c < W; ++c)
                mp_norm(c, r) = (max_proj(c, r) - mp_min) / mp_range;

        // Normalize col_sum for bar chart
        float cs_max = 1.0f;
        for (int r = 0; r < H; ++r) cs_max = std::max(cs_max, col_sum(r));
        Halide::Buffer<float> cs_norm(H);
        for (int r = 0; r < H; ++r) cs_norm(r) = col_sum(r) / cs_max;

        // Normalize demean_buf for visualization
        float dm_min = demean_buf(0, 0), dm_max = demean_buf(0, 0);
        for (int r = 0; r < H; ++r)
            for (int c = 0; c < W; ++c) {
                dm_min = std::min(dm_min, demean_buf(c, r));
                dm_max = std::max(dm_max, demean_buf(c, r));
            }
        float dm_range = std::max(dm_max - dm_min, 1e-6f);
        Halide::Buffer<float> dm_norm(W, H);
        for (int r = 0; r < H; ++r)
            for (int c = 0; c < W; ++c)
                dm_norm(c, r) = (demean_buf(c, r) - dm_min) / dm_range;

        // Normalize row_mean for visualization
        float rm_min = row_mean(0, 0), rm_max = row_mean(0, 0);
        for (int d = 0; d < D; ++d)
            for (int c = 0; c < W; ++c) {
                rm_min = std::min(rm_min, row_mean(c, d));
                rm_max = std::max(rm_max, row_mean(c, d));
            }
        float rm_range = std::max(rm_max - rm_min, 1e-6f);
        Halide::Buffer<float> rm_norm(W, D);
        for (int d = 0; d < D; ++d)
            for (int c = 0; c < W; ++c)
                rm_norm(c, d) = (row_mean(c, d) - rm_min) / rm_range;

        // Create Halide Funcs for each quadrant's pixel rendering
        Halide::Func mp_f("mp_f"), rm_f("rm_f"), cs_f("cs_f"), dm_f("dm_f");

        // Top-left: max projection (64x64 scaled to 256x256)
        mp_f(ox, oy) = mp_norm(Halide::clamp(ox * W / half, 0, W-1),
                               Halide::clamp(oy * H / half, 0, H-1));

        // Top-right: normalized row mean (W x D scaled to 256x256)
        rm_f(ox, oy) = rm_norm(Halide::clamp(ox * W / half, 0, W-1),
                               Halide::clamp(oy * D / half, 0, D-1));

        // Bottom-left: col sum bar chart (H values, each row gets a brightness)
        cs_f(ox, oy) = cs_norm(Halide::clamp(oy * H / half, 0, H-1));

        // Bottom-right: de-mean slice 0 (W x H scaled to 256x256)
        dm_f(ox, oy) = dm_norm(Halide::clamp(ox * W / half, 0, W-1),
                               Halide::clamp(oy * H / half, 0, H-1));

        Halide::Func output("output");
        Halide::Expr pixel = Halide::select(
            qy == 0 && qx == 0, mp_f(lx, ly),
            qy == 0 && qx == 1, rm_f(lx, ly),
            qy == 1 && qx == 0, cs_f(lx, ly),
            dm_f(lx, ly)
        );
        output(ox, oy) = Halide::cast<uint8_t>(Halide::clamp(pixel * 255.0f, 0.0f, 255.0f));

        Halide::Runtime::Buffer<uint8_t> img(size, size);
        output.realize(img);

        const char* path = "out/49_multi_reduce.png";
        if (!save_png(img, path)) {
            std::cerr << "Failed to save " << path << std::endl; return 1;
        }
        std::cout << "Saved to " << path << std::endl;
        std::cout << "\nQuadrant guide:" << std::endl;
        std::cout << "  Top-left:     Max projection along Z (depth D=4) of 64x64x4 cube" << std::endl;
        std::cout << "  Top-right:    Row mean (axis 1) -> per-column-per-depth heat map" << std::endl;
        std::cout << "  Bottom-left:  Column sum (axes {0,2}) -> bar chart across rows" << std::endl;
        std::cout << "  Bottom-right: De-mean normalization of depth slice 0" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl; return 1;
    }
}
