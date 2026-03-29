/// @file 55_autodiff.cpp
/// @brief Example 55: Tape-based reverse-mode automatic differentiation
///
/// Output: 512x512 PNG divided into four 256x256 quadrants:
///
///   Top-left:     f(x) = x²-4x+3 and its gradient f'(x)=2x-4
///                 for x ∈ [-2, 5], 100 samples. Two overlapping line plots.
///   Top-right:    Gradient descent on f(x,y)=(x-2)²+(y-3)², starting from
///                 (0,0), 30 steps with lr=0.1.  Position history as dots.
///   Bottom-left:  Loss curve from the gradient descent above (f vs step).
///   Bottom-right: sin(x²) and its autodiff gradient vs analytical 2x·cos(x²)
///                 for x ∈ [0, π].
///
/// Output: out/55_autodiff.png

#include "numhalide_all.h"
#include "stbi_png.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>

using namespace numhalide;

static const float PI = 3.14159265358979323846f;

// =============================================================================
// Drawing helpers
// =============================================================================

/// Fill one 256x256 quadrant with a colour
static void fill_quad(Halide::Runtime::Buffer<uint8_t>& img,
                      int ox, int oy, uint8_t r, uint8_t g, uint8_t b)
{
    for (int y = 0; y < 256; ++y)
        for (int x = 0; x < 256; ++x) {
            img(ox + x, oy + y, 0) = r;
            img(ox + x, oy + y, 1) = g;
            img(ox + x, oy + y, 2) = b;
        }
}

/// Clamp a pixel coordinate into [0,255]
static int clamp_px(int v) { return v < 0 ? 0 : (v > 255 ? 255 : v); }

/// Draw a polyline from (x_pixels[], y_pixels[]) in given colour, within quadrant (ox,oy)
static void draw_line(Halide::Runtime::Buffer<uint8_t>& img,
                      const std::vector<int>& px, const std::vector<int>& py,
                      int ox, int oy,
                      uint8_t r, uint8_t g, uint8_t b)
{
    int n = static_cast<int>(px.size());
    for (int i = 1; i < n; ++i) {
        // Bresenham between (px[i-1],py[i-1]) and (px[i],py[i])
        int x0 = px[i-1], y0 = py[i-1];
        int x1 = px[i],   y1 = py[i];
        int dx = std::abs(x1 - x0), dy = std::abs(y1 - y0);
        int sx = (x0 < x1) ? 1 : -1;
        int sy = (y0 < y1) ? 1 : -1;
        int err = dx - dy;
        while (true) {
            int cx = clamp_px(x0), cy = clamp_px(y0);
            img(ox + cx, oy + cy, 0) = r;
            img(ox + cx, oy + cy, 1) = g;
            img(ox + cx, oy + cy, 2) = b;
            if (x0 == x1 && y0 == y1) break;
            int e2 = 2 * err;
            if (e2 > -dy) { err -= dy; x0 += sx; }
            if (e2 <  dx) { err += dx; y0 += sy; }
        }
    }
}

/// Draw a filled 3x3 dot
static void draw_dot(Halide::Runtime::Buffer<uint8_t>& img,
                     int cx, int cy, int ox, int oy,
                     uint8_t r, uint8_t g, uint8_t b)
{
    for (int dy = -2; dy <= 2; ++dy)
        for (int dx = -2; dx <= 2; ++dx) {
            int px = clamp_px(cx + dx);
            int py = clamp_px(cy + dy);
            img(ox + px, oy + py, 0) = r;
            img(ox + px, oy + py, 1) = g;
            img(ox + px, oy + py, 2) = b;
        }
}

/// Draw separator lines
static void draw_grid(Halide::Runtime::Buffer<uint8_t>& img)
{
    for (int i = 0; i < 512; ++i) {
        img(256, i, 0) = 80; img(256, i, 1) = 80; img(256, i, 2) = 80;
        img(i, 256, 0) = 80; img(i, 256, 1) = 80; img(i, 256, 2) = 80;
    }
}

// =============================================================================
// Map float arrays to pixel coordinates within a 256x256 panel
// =============================================================================
static std::vector<int> to_px(const std::vector<float>& xs,
                               float x_min, float x_max)
{
    std::vector<int> out(xs.size());
    float range = x_max - x_min;
    for (size_t i = 0; i < xs.size(); ++i)
        out[i] = clamp_px(static_cast<int>((xs[i] - x_min) / range * 255.0f + 0.5f));
    return out;
}

static std::vector<int> to_py(const std::vector<float>& ys,
                               float y_min, float y_max)
{
    // y axis: smaller value = higher pixel (flip)
    std::vector<int> out(ys.size());
    float range = y_max - y_min;
    for (size_t i = 0; i < ys.size(); ++i) {
        float norm = (ys[i] - y_min) / range;
        out[i] = clamp_px(static_cast<int>((1.0f - norm) * 255.0f + 0.5f));
    }
    return out;
}

// =============================================================================
// main
// =============================================================================
int main(int /*argc*/, char** /*argv*/)
{
    try {
        std::cout << "Autodiff example\n";

        Halide::Runtime::Buffer<uint8_t> img(512, 512, 3);

        // ===================================================================
        // TOP-LEFT: f(x) = x²-4x+3 and f'(x)=2x-4 for x ∈ [-2, 5]
        // ===================================================================
        {
            fill_quad(img, 0, 0, 15, 15, 25);

            const int S = 100;
            std::vector<float> x_vals(S), f_vals(S), g_vals(S);
            float x_min = -2.0f, x_max = 5.0f;

            for (int s = 0; s < S; ++s) {
                float xv = x_min + (x_max - x_min) * float(s) / float(S - 1);
                x_vals[s] = xv;

                dtape_reset();
                DVar x(xv);
                DVar f = x * x - x * 4.0f + DVar(3.0f);
                f.backward();

                f_vals[s] = f.val();
                g_vals[s] = x.grad();
            }

            // Find combined range for y axis
            float y_min = *std::min_element(g_vals.begin(), g_vals.end());
            float y_max = *std::max_element(f_vals.begin(), f_vals.end());
            y_min = std::min(y_min, *std::min_element(f_vals.begin(), f_vals.end()));
            y_max = std::max(y_max, *std::max_element(g_vals.begin(), g_vals.end()));
            // Add 10% padding
            float pad = (y_max - y_min) * 0.1f;
            y_min -= pad; y_max += pad;

            auto px_s = to_px(x_vals, x_min, x_max);
            auto py_f = to_py(f_vals, y_min, y_max);
            auto py_g = to_py(g_vals, y_min, y_max);

            // Draw f(x) in yellow
            draw_line(img, px_s, py_f, 0, 0, 230, 200, 40);
            // Draw f'(x) in cyan
            draw_line(img, px_s, py_g, 0, 0, 40, 200, 220);
            // Draw y=0 axis
            {
                int y0_px = clamp_px(static_cast<int>((1.0f - (0.0f - y_min)/(y_max - y_min)) * 255.0f));
                for (int x = 0; x < 256; ++x) {
                    img(x, y0_px, 0) = 80; img(x, y0_px, 1) = 80; img(x, y0_px, 2) = 80;
                }
            }
        }

        // ===================================================================
        // TOP-RIGHT: Gradient descent on (x-2)²+(y-3)²
        //   Start (0,0), lr=0.1, 30 steps
        // ===================================================================
        std::vector<float> gd_x, gd_y, gd_loss;
        {
            fill_quad(img, 256, 0, 15, 25, 15);

            float cx = 0.0f, cy_val = 0.0f;
            const float lr = 0.1f;
            const int steps = 30;

            gd_x.push_back(cx);
            gd_y.push_back(cy_val);

            for (int s = 0; s < steps; ++s) {
                dtape_reset();
                DVar xv(cx), yv(cy_val);
                DVar loss = (xv - DVar(2.0f)) * (xv - DVar(2.0f))
                          + (yv - DVar(3.0f)) * (yv - DVar(3.0f));
                loss.backward();

                gd_loss.push_back(loss.val());
                cx     -= lr * xv.grad();
                cy_val -= lr * yv.grad();
                gd_x.push_back(cx);
                gd_y.push_back(cy_val);
            }
            // final loss
            {
                float fx = (cx - 2.0f), fy = (cy_val - 3.0f);
                gd_loss.push_back(fx*fx + fy*fy);
            }

            // Map positions to pixels: x ∈ [-0.5, 2.5], y ∈ [-0.5, 3.5]
            float xmin = -0.3f, xmax = 2.3f;
            float ymin = -0.3f, ymax = 3.3f;

            for (size_t i = 0; i < gd_x.size(); ++i) {
                int px = clamp_px(static_cast<int>((gd_x[i]-xmin)/(xmax-xmin) * 255.0f));
                int py = clamp_px(static_cast<int>((1.0f-(gd_y[i]-ymin)/(ymax-ymin)) * 255.0f));
                // colour: step 0 = red, final = green
                float t = float(i) / float(gd_x.size() - 1);
                uint8_t r = static_cast<uint8_t>((1.0f - t) * 220);
                uint8_t g = static_cast<uint8_t>(t * 220);
                draw_dot(img, px, py, 256, 0, r, g, 60);
            }
            // Draw target position (2,3) as white cross
            int tx = clamp_px(static_cast<int>((2.0f-xmin)/(xmax-xmin) * 255.0f));
            int ty = clamp_px(static_cast<int>((1.0f-(3.0f-ymin)/(ymax-ymin)) * 255.0f));
            for (int d = -6; d <= 6; ++d) {
                int px = clamp_px(tx + d), py = clamp_px(ty + d);
                img(256 + px, ty,  0) = 255; img(256 + px, ty,  1) = 255; img(256 + px, ty,  2) = 255;
                img(256 + tx, py,  0) = 255; img(256 + tx, py,  1) = 255; img(256 + tx, py,  2) = 255;
            }
        }

        // ===================================================================
        // BOTTOM-LEFT: Loss curve from gradient descent
        // ===================================================================
        {
            fill_quad(img, 0, 256, 25, 15, 15);

            int n = static_cast<int>(gd_loss.size());
            std::vector<float> step_idx(n);
            for (int i = 0; i < n; ++i) step_idx[i] = float(i);

            float loss_min = *std::min_element(gd_loss.begin(), gd_loss.end());
            float loss_max = *std::max_element(gd_loss.begin(), gd_loss.end());
            float pad = (loss_max - loss_min) * 0.05f + 1e-6f;
            loss_min -= pad; loss_max += pad;

            auto px_s = to_px(step_idx, 0.0f, float(n - 1));
            auto py_l = to_py(gd_loss,  loss_min, loss_max);

            draw_line(img, px_s, py_l, 0, 256, 220, 80, 80);
        }

        // ===================================================================
        // BOTTOM-RIGHT: sin(x²) and its gradient vs analytical 2x*cos(x²)
        //   for x ∈ [0, π]
        // ===================================================================
        {
            fill_quad(img, 256, 256, 15, 15, 25);

            const int S = 100;
            std::vector<float> xv(S), fv(S), gv(S), av(S);

            for (int s = 0; s < S; ++s) {
                float xf = float(s) / float(S - 1) * PI;
                xv[s] = xf;

                dtape_reset();
                DVar x(xf);
                DVar f = dsin(x * x);
                f.backward();

                fv[s] = f.val();
                gv[s] = x.grad();                        // autodiff gradient
                av[s] = 2.0f * xf * std::cos(xf * xf);  // analytical
            }

            float y_min = -2.5f, y_max = 2.5f;

            auto px_s = to_px(xv, 0.0f, PI);
            auto py_f = to_py(fv, y_min, y_max);
            auto py_g = to_py(gv, y_min, y_max);
            auto py_a = to_py(av, y_min, y_max);

            // f(x)=sin(x²) in blue
            draw_line(img, px_s, py_f, 256, 256, 80, 80, 220);
            // Autodiff gradient in yellow
            draw_line(img, px_s, py_g, 256, 256, 220, 200, 40);
            // Analytical gradient in cyan (should overlay yellow closely)
            draw_line(img, px_s, py_a, 256, 256, 40, 220, 200);

            // Draw y=0 axis
            {
                int y0 = clamp_px(static_cast<int>((1.0f - (0.0f - y_min)/(y_max-y_min)) * 255.0f));
                for (int x = 0; x < 256; ++x) {
                    img(256 + x, 256 + y0, 0) = 80;
                    img(256 + x, 256 + y0, 1) = 80;
                    img(256 + x, 256 + y0, 2) = 80;
                }
            }
        }

        draw_grid(img);

        // ---------------------------------------------------------------
        // Save PNG
        // ---------------------------------------------------------------
        const char* path = "out/55_autodiff.png";
        if (!save_png(img, path)) {
            std::cerr << "Failed to save " << path << "\n";
            return 1;
        }
        std::cout << "Saved " << path << "\n";
        std::cout << "\nQuadrant guide:\n"
                  << "  Top-left:     f(x)=x²-4x+3 (yellow) and f'(x)=2x-4 (cyan)\n"
                  << "  Top-right:    Gradient descent on (x-2)²+(y-3)² (red->green)\n"
                  << "  Bottom-left:  Loss vs step during gradient descent\n"
                  << "  Bottom-right: sin(x²) (blue), autodiff grad (yellow), "
                     "analytical (cyan)\n";
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
