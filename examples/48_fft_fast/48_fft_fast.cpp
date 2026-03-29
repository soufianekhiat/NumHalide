/// @file 48_fft_fast.cpp
/// @brief Example 48: Cooley-Tukey O(N log N) FFT demonstration
///
/// Quadrant guide:
///   Top-left:     Original 256-sample multi-frequency signal (waveform as heatmap)
///   Top-right:    Power spectrum via fft_fast (magnitude squared, log scale)
///   Bottom-left:  Low-pass filtered reconstruction (first 8 bins kept)
///   Bottom-right: 128x128 2D FFT magnitude (log scale) of a checkerboard pattern

#include "numhalide_all.h"
#include "stbi_png.h"
#include <iostream>

using namespace numhalide;
static const float PI = 3.14159265358979323846f;

int main(int argc, char** argv) {
    try {
        const int half = 256;
        const int size = half * 2;
        const int N    = 256;   // 1D signal length
        const int N2   = 128;   // 2D FFT size

        std::cout << "FFT Fast (Cooley-Tukey O(N log N)) demo" << std::endl;

        // ----------------------------------------------------------------
        // Build the 1D multi-frequency cosine signal
        // signal(n) = 0.5 + 0.3*cos(2π*5*n/N) + 0.15*sin(2π*12*n/N)
        //                 + 0.1*cos(2π*23*n/N)
        // ----------------------------------------------------------------
        Halide::Buffer<float> sig_re(N), sig_im(N);
        for (int n = 0; n < N; ++n) {
            float v = 0.5f
                + 0.30f * std::cos(2.0f * PI * 5.0f  * n / N)
                + 0.15f * std::sin(2.0f * PI * 12.0f * n / N)
                + 0.10f * std::cos(2.0f * PI * 23.0f * n / N);
            sig_re(n) = v;
            sig_im(n) = 0.0f;
        }

        Halide::Var x("x"), y("y");
        Halide::Func sig_func("sig");
        sig_func(x) = Halide::Tuple(sig_re(Halide::clamp(x, 0, N-1)),
                                     sig_im(Halide::clamp(x, 0, N-1)));

        // ---- Compute fast FFT ----
        auto SPEC = fft_fast(sig_func, N, "spec");

        // Power spectrum (magnitude squared per bin)
        Halide::Buffer<float> pow_spec(N / 2 + 1);
        {
            Halide::Func mag_sq("mag_sq_tmp");
            Halide::Var k("k");
            mag_sq(k) = SPEC(k)[0] * SPEC(k)[0] + SPEC(k)[1] * SPEC(k)[1];
            mag_sq.realize(pow_spec);
        }

        // ---- Low-pass filter: keep only bins 0..7 ----
        const int LP_BINS = 8;
        Halide::Buffer<float> filt_re(N), filt_im(N);
        {
            // Realize full spectrum
            Halide::Runtime::Buffer<float> spec_re_buf(N), spec_im_buf(N);
            Halide::Var k("k");
            Halide::Func sre("sre"), sim("sim");
            sre(k) = SPEC(k)[0]; sim(k) = SPEC(k)[1];
            sre.realize(spec_re_buf); sim.realize(spec_im_buf);

            // Zero out high-frequency bins
            for (int i = 0; i < N; ++i) {
                int bin = (i <= N/2) ? i : N - i;  // unwrap to positive freq
                if (bin < LP_BINS) {
                    filt_re(i) = spec_re_buf(i);
                    filt_im(i) = spec_im_buf(i);
                } else {
                    filt_re(i) = 0.0f;
                    filt_im(i) = 0.0f;
                }
            }
        }
        // Reconstruct via ifft_fast
        Halide::Func filt_func("filt_spec");
        Halide::Var k2("k2");
        filt_func(k2) = Halide::Tuple(filt_re(Halide::clamp(k2, 0, N-1)),
                                       filt_im(Halide::clamp(k2, 0, N-1)));
        auto recon = ifft_fast(filt_func, N, "recon");
        Halide::Buffer<float> recon_buf(N);
        {
            Halide::Func rr("rr"); rr(k2) = recon(k2)[0]; rr.realize(recon_buf);
        }

        // ---- 2D FFT of a checkerboard pattern ----
        Halide::Buffer<float> cb_re(N2, N2), cb_im(N2, N2);
        for (int r = 0; r < N2; ++r)
            for (int c = 0; c < N2; ++c) {
                cb_re(c, r) = (float)(((r / 8) + (c / 8)) % 2);
                cb_im(c, r) = 0.0f;
            }
        Halide::Func cb2d("cb2d");
        cb2d(x, y) = Halide::Tuple(cb_re(Halide::clamp(x, 0, N2-1), Halide::clamp(y, 0, N2-1)),
                                    cb_im(Halide::clamp(x, 0, N2-1), Halide::clamp(y, 0, N2-1)));
        auto SPEC2D = fft2d_fast(cb2d, N2, N2, "spec2d");
        Halide::Buffer<float> pow2d(N2, N2);
        {
            Halide::Func mag2("mag2d_sq"); Halide::Var bx("bx"), by("by");
            mag2(bx, by) = SPEC2D(bx, by)[0] * SPEC2D(bx, by)[0]
                         + SPEC2D(bx, by)[1] * SPEC2D(bx, by)[1];
            mag2.realize(pow2d);
        }
        // Log scale for 2D spectrum
        float max2d = 1.0f;
        for (int r = 0; r < N2; ++r)
            for (int c = 0; c < N2; ++c)
                max2d = std::max(max2d, pow2d(c, r));
        Halide::Buffer<float> log2d(N2, N2);
        for (int r = 0; r < N2; ++r)
            for (int c = 0; c < N2; ++c)
                log2d(c, r) = std::log(1.0f + pow2d(c, r) / max2d * 100.0f) / std::log(101.0f);

        // ---- Compose 512x512 output ----
        Halide::Func output("output");
        // quad selectors
        Halide::Expr qx = x / half;  // 0=left, 1=right
        Halide::Expr qy = y / half;  // 0=top, 1=bottom
        Halide::Expr lx = x % half;  // local x [0..255]
        Halide::Expr ly = y % half;  // local y [0..255]

        // Normalize signal to [0,1]
        float sig_min = *std::min_element(sig_re.begin(), sig_re.end());
        float sig_max = *std::max_element(sig_re.begin(), sig_re.end());
        float sig_range = std::max(sig_max - sig_min, 1e-6f);

        // Normalize recon to [0,1]
        float rec_min = *std::min_element(recon_buf.begin(), recon_buf.end());
        float rec_max = *std::max_element(recon_buf.begin(), recon_buf.end());
        float rec_range = std::max(rec_max - rec_min, 1e-6f);

        // Normalize power spectrum (log) to [0,1]
        float ps_max = 1.0f;
        for (int i = 0; i <= N/2; ++i) ps_max = std::max(ps_max, pow_spec(i));

        Halide::Buffer<float> sig_norm_buf(N), ps_norm_buf(N/2 + 1), rec_norm_buf(N);
        for (int i = 0; i < N; ++i)
            sig_norm_buf(i) = (sig_re(i) - sig_min) / sig_range;
        for (int i = 0; i <= N/2; ++i)
            ps_norm_buf(i) = std::log(1.0f + pow_spec(i) / ps_max * 100.0f) / std::log(101.0f);
        for (int i = 0; i < N; ++i)
            rec_norm_buf(i) = (recon_buf(i) - rec_min) / rec_range;

        Halide::Func sig_n("sig_n"), ps_n("ps_n"), rec_n("rec_n"), log2d_f("log2d_f");
        Halide::Var ix("ix");
        sig_n(ix) = sig_norm_buf(Halide::clamp(ix, 0, N-1));
        ps_n(ix)  = ps_norm_buf(Halide::clamp(ix, 0, N/2));
        rec_n(ix) = rec_norm_buf(Halide::clamp(ix, 0, N-1));
        log2d_f(x, y) = log2d(Halide::clamp(x, 0, N2-1), Halide::clamp(y, 0, N2-1));

        // Waveform bar chart: bright where ly/half <= value (ly=0 is top)
        // To show bars from bottom, flip: bright where (half - 1 - ly)/half <= value
        Halide::Expr bar_frac_s = (Halide::cast<float>(half - 1 - ly)) / (float)(half - 1);
        Halide::Expr bar_frac_r = (Halide::cast<float>(half - 1 - ly)) / (float)(half - 1);

        // Top-left: signal waveform
        Halide::Expr sig_val = sig_n(lx);
        Halide::Expr tl = Halide::select(bar_frac_s <= sig_val, 1.0f, 0.15f);

        // Top-right: power spectrum (first N/2 bins mapped to 0..255 columns)
        // bin = lx * (N/2) / 255
        Halide::Expr ps_bin = Halide::clamp(lx * (N / 2) / (half - 1), 0, N / 2);
        Halide::Expr ps_val = ps_n(ps_bin);
        Halide::Expr tr = Halide::select(bar_frac_s <= ps_val, 1.0f, 0.15f);

        // Bottom-left: reconstructed signal (low-pass)
        Halide::Expr rec_val = rec_n(lx);
        Halide::Expr bl = Halide::select(bar_frac_r <= rec_val, 1.0f, 0.15f);

        // Bottom-right: 2D FFT heatmap (scaled 2x: 128 → 256 pixels)
        Halide::Expr x2 = Halide::clamp(lx * N2 / half, 0, N2 - 1);
        Halide::Expr y2 = Halide::clamp(ly * N2 / half, 0, N2 - 1);
        Halide::Expr br = log2d_f(x2, y2);

        Halide::Expr pixel = Halide::select(
            qy == 0 && qx == 0, tl,
            qy == 0 && qx == 1, tr,
            qy == 1 && qx == 0, bl,
            br
        );
        output(x, y) = Halide::cast<uint8_t>(Halide::clamp(pixel * 255.0f, 0.0f, 255.0f));

        Halide::Runtime::Buffer<uint8_t> img(size, size);
        output.realize(img);

        const char* path = "out/48_fft_fast.png";
        if (!save_png(img, path)) {
            std::cerr << "Failed to save " << path << std::endl;
            return 1;
        }

        std::cout << "Saved to " << path << std::endl;
        std::cout << "\nQuadrant guide:" << std::endl;
        std::cout << "  Top-left:     Multi-frequency signal waveform (N=" << N << ")" << std::endl;
        std::cout << "  Top-right:    Power spectrum via fft_fast (log scale)" << std::endl;
        std::cout << "  Bottom-left:  Low-pass reconstruction (first " << LP_BINS << " bins)" << std::endl;
        std::cout << "  Bottom-right: 2D FFT magnitude of " << N2 << "x" << N2 << " checkerboard (log)" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
