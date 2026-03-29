/// @file 54_complex_type.cpp
/// @brief Example 54: C++ complex float type and buffer operations
///
/// Output: 512x512 PNG divided into four 256x256 quadrants:
///
///   Top-left:     Magnitude of a 256-element complex chirp signal
///                 (frequency sweep: f(t) = exp(i*pi*k*t^2/N)), plotted as
///                 column bars mapped to brightness.
///   Top-right:    Phase of the same chirp signal, mapped to [0,255].
///   Bottom-left:  Magnitude of the element-wise product of two complex
///                 sinusoids (frequency mixing).
///   Bottom-right: Polar-form reconstruction check — construct signals from
///                 polar, multiply them, verify abs matches direct multiplication.
///                 White = match (|diff| < 1e-4), dark = deviation.
///
/// Output: out/54_complex_type.png

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

/// Fill one 256x256 quadrant with a solid colour
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

/// Draw a vertical bar chart of a 256-element float array (values in [0,1]).
/// Each column x maps to one sample; bar height = val[x]*256.
static void draw_bar_chart(Halide::Runtime::Buffer<uint8_t>& img,
                           const std::vector<float>& vals,
                           int ox, int oy,
                           uint8_t fr, uint8_t fg, uint8_t fb)
{
    // Black background
    fill_quad(img, ox, oy, 10, 10, 10);
    int n = static_cast<int>(vals.size());
    for (int x = 0; x < 256 && x < n; ++x) {
        int bar_h = static_cast<int>(vals[x] * 255.0f + 0.5f);
        bar_h = std::max(0, std::min(255, bar_h));
        for (int dy = 0; dy < bar_h; ++dy) {
            int py = oy + 255 - dy;
            img(ox + x, py, 0) = fr;
            img(ox + x, py, 1) = fg;
            img(ox + x, py, 2) = fb;
        }
    }
}

/// Draw a grid separator line
static void draw_grid(Halide::Runtime::Buffer<uint8_t>& img)
{
    for (int i = 0; i < 512; ++i) {
        img(256, i, 0) = 80; img(256, i, 1) = 80; img(256, i, 2) = 80;
        img(i, 256, 0) = 80; img(i, 256, 1) = 80; img(i, 256, 2) = 80;
    }
}

// =============================================================================
// Normalise a float vector to [0,1]
// =============================================================================
static std::vector<float> normalize_vec(const std::vector<float>& v)
{
    float mn = *std::min_element(v.begin(), v.end());
    float mx = *std::max_element(v.begin(), v.end());
    float range = mx - mn;
    std::vector<float> out(v.size());
    if (range < 1e-9f) { return out; }
    for (size_t i = 0; i < v.size(); ++i)
        out[i] = (v[i] - mn) / range;
    return out;
}

// =============================================================================
// main
// =============================================================================
int main(int /*argc*/, char** /*argv*/)
{
    try {
        const int N = 256;  // number of complex samples

        std::cout << "Complex type example\n";

        // ---------------------------------------------------------------
        // Build chirp signal: c[k] = exp(i * pi * k^2 / N)
        // This is a linearly-sweeping-frequency (chirp) complex sinusoid.
        // ---------------------------------------------------------------
        ComplexBuffer chirp(N);
        for (int k = 0; k < N; ++k) {
            float theta = PI * float(k) * float(k) / float(N);
            chirp.set(k, complex_exp_i(theta));
        }

        // ---------------------------------------------------------------
        // Top-left: magnitude of the chirp
        // All elements of a unit chirp have |z| = 1, so the bar chart is flat.
        // To make it visually interesting, add a Gaussian envelope.
        // ---------------------------------------------------------------
        ComplexBuffer chirp_env(N);
        for (int k = 0; k < N; ++k) {
            float t = (float(k) - N / 2.0f) / (N / 4.0f);
            float env = std::exp(-0.5f * t * t);
            float theta = PI * float(k) * float(k) / float(N);
            chirp_env.set(k, complex_f32(env * std::cos(theta),
                                         env * std::sin(theta)));
        }
        auto mag_chirp_raw = complex_buf_abs(chirp_env);
        std::vector<float> mag_chirp_v(N);
        for (int k = 0; k < N; ++k) mag_chirp_v[k] = mag_chirp_raw(k);
        auto mag_chirp_n = normalize_vec(mag_chirp_v);

        // ---------------------------------------------------------------
        // Top-right: phase of the chirp (unwrapped display: map [-π,π] -> [0,1])
        // ---------------------------------------------------------------
        auto phase_chirp_raw = complex_buf_phase(chirp_env);
        std::vector<float> phase_v(N);
        for (int k = 0; k < N; ++k)
            phase_v[k] = (phase_chirp_raw(k) + PI) / (2.0f * PI);  // [0,1]

        // ---------------------------------------------------------------
        // Bottom-left: product of two sinusoids (frequency mixing)
        //   s1[k] = exp(i * 2*pi * f1 * k / N),  f1 = 5
        //   s2[k] = exp(i * 2*pi * f2 * k / N),  f2 = 13
        //   product = exp(i * 2*pi * (f1+f2) * k / N)  -> single frequency 18
        // ---------------------------------------------------------------
        const float f1 = 5.0f, f2 = 13.0f;
        ComplexBuffer s1(N), s2(N);
        for (int k = 0; k < N; ++k) {
            s1.set(k, complex_exp_i(2.0f * PI * f1 * float(k) / float(N)));
            s2.set(k, complex_exp_i(2.0f * PI * f2 * float(k) / float(N)));
        }
        auto product = complex_buf_mul(s1, s2);
        // Visualise as real part of product to show oscillation
        std::vector<float> product_re(N);
        for (int k = 0; k < N; ++k) product_re[k] = product(k).re;
        auto product_n = normalize_vec(product_re);

        // ---------------------------------------------------------------
        // Bottom-right: polar reconstruction check
        //   Build chirp_env from polar, multiply with s1, compare abs with
        //   direct complex_buf_mul result abs.
        // ---------------------------------------------------------------
        auto mag_env_buf  = complex_buf_abs(chirp_env);
        auto phase_env_buf = complex_buf_phase(chirp_env);
        auto chirp_from_polar = complex_buf_from_polar(mag_env_buf, phase_env_buf);
        // Multiply both ways
        auto direct_mul  = complex_buf_mul(chirp_env, s1);
        auto polar_mul   = complex_buf_mul(chirp_from_polar, s1);
        auto abs_direct  = complex_buf_abs(direct_mul);
        auto abs_polar   = complex_buf_abs(polar_mul);
        // Compute per-element absolute error, normalised
        std::vector<float> err(N);
        float max_err = 0.0f;
        for (int k = 0; k < N; ++k) {
            err[k] = std::abs(abs_direct(k) - abs_polar(k));
            max_err = std::max(max_err, err[k]);
        }
        // Map: small error -> bright, large error -> dark
        // (expect all near-zero -> all bright)
        float scale_err = (max_err > 1e-9f) ? (1.0f / max_err) : 1.0f;
        std::vector<float> err_display(N);
        for (int k = 0; k < N; ++k)
            err_display[k] = 1.0f - std::min(1.0f, err[k] * scale_err * 100.0f);

        // ---------------------------------------------------------------
        // Compose 512x512 RGB output
        // ---------------------------------------------------------------
        Halide::Runtime::Buffer<uint8_t> img(512, 512, 3);

        // Top-left: magnitude bar chart (cyan bars)
        draw_bar_chart(img, mag_chirp_n, 0, 0, 0, 200, 200);

        // Top-right: phase bar chart (orange bars)
        draw_bar_chart(img, phase_v, 256, 0, 220, 140, 40);

        // Bottom-left: product real part (green bars)
        draw_bar_chart(img, product_n, 0, 256, 40, 200, 80);

        // Bottom-right: error display (white = low error)
        draw_bar_chart(img, err_display, 256, 256, 200, 200, 200);

        draw_grid(img);

        // ---------------------------------------------------------------
        // Save PNG
        // ---------------------------------------------------------------
        const char* path = "out/54_complex_type.png";
        if (!save_png(img, path)) {
            std::cerr << "Failed to save " << path << "\n";
            return 1;
        }
        std::cout << "Saved " << path << "\n";
        std::cout << "\nQuadrant guide:\n"
                  << "  Top-left:     Magnitude of Gaussian-enveloped chirp signal\n"
                  << "  Top-right:    Phase of chirp signal mapped to [0,1]\n"
                  << "  Bottom-left:  Real part of product of two sinusoids (freq mixing)\n"
                  << "  Bottom-right: Polar reconstruction error (bright = accurate)\n"
                  << "  Max polar reconstruction error: " << max_err << "\n";
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
