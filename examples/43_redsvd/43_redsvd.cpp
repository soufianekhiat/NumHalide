/// @file 43_redsvd.cpp
/// @brief Example 43: RPCA via Iterative SVT (Principal Component Pursuit)
///        using NumHalide matmul, qr_gs, svd_jacobi
///
/// Robust PCA / Principal Component Pursuit (Candes et al. 2009):
///   Given A = L + S  (low-rank signal + sparse outliers)
///   Minimize  ||L||_*  +  lambda * ||S||_1   s.t.  A = L + S
///
/// Solved by Inexact Augmented Lagrangian Method (Lin et al. 2010):
///   T     = A - S + Y/mu
///   L     = SVT(T, 1/mu)     [Singular Value Thresholding of T]
///   S     = shrink(A-L+Y/mu, lambda/mu)
///   Y    += mu * (A - L - S)
///   mu    = min(rho*mu, mu_max)
///
/// SVT at each iteration uses RedSVD (randomized rank-K SVD):
///   1.  O ~ Gaussian(M,K);  Y = T^T * O;   Q_Y = orth(Y)   via qr_gs
///   2.  B = T * Q_Y;        Z = B * P;      Q_Z = orth(Z)   via qr_gs
///   3.  C = Q_Z^T * B  [K x K sketch]
///   4.  SVD(C) = U_C diag(sigma) Vt_C                       via svd_jacobi
///   5.  Sort sigma desc; soft-threshold: sigma_k = max(sigma_k - 1/mu, 0)
///   6.  U = Q_Z * U_C_sorted;  V = Q_Y * V_C_sorted
///   7.  L = U * diag(sigma_svt) * V^T
///
/// JIT caching: all Halide pipelines are defined ONCE before the iteration
/// loop.  The first realize() JIT-compiles each pipeline; subsequent calls
/// reuse the cached binary -- only buffer DATA changes between iterations.
///
/// Output: out/43_redsvd.png (512x512, four 256x256 quadrants)
///   Top-left:     Original A (signal + ~15% sparse outliers)
///   Top-right:    Recovered L (low-rank component, same scale as A)
///   Bottom-left:  Recovered S (sparse component, gray=0)
///   Bottom-right: Convergence curve  ||A-L-S||_F / ||A||_F  per iteration

#include "numhalide_all.h"
#include "stbi_png.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
#include <vector>
#include <numeric>
#include <algorithm>

using namespace numhalide;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static Halide::Func as_func(Halide::Buffer<float>& buf, const std::string& nm)
{
    Halide::Func f(nm); Halide::Var c("c"), r("r");
    f(c, r) = buf(c, r); return f;
}

static void fill_gaussian(Halide::Buffer<float>& buf, std::mt19937& rng)
{
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (int row = 0; row < buf.height(); ++row)
        for (int col = 0; col < buf.width(); ++col)
            buf(col, row) = nd(rng);
}

// Element-wise soft-threshold: sign(x)*max(|x|-t, 0)
static inline float soft(float x, float t)
{
    return (x > t) ? (x - t) : (x < -t) ? (x + t) : 0.0f;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int /*argc*/, char** /*argv*/)
{
    try {
        // ---- Parameters ------------------------------------------------
        const int M         = 32;    // matrix rows
        const int N         = 32;    // matrix cols
        const int TRUE_RANK = 3;     // true signal rank
        const int K         = 4;     // RedSVD rank  (>= TRUE_RANK)
        const int MAX_ITER  = 20;    // RPCA-ALM max iterations
        const int Q         = 256;   // output image quadrant size (pixels)
        const int OUT       = Q * 2;

        const float PI     = 3.14159265f;
        const float lambda = 1.0f / std::sqrt((float)std::max(M, N));
        const float rho    = 1.5f;   // mu growth factor

        std::mt19937 rng(42);

        std::cout << "RPCA via Iterative SVT (Principal Component Pursuit)\n";
        std::cout << "Matrix: " << M << " x " << N
                  << "  true rank=" << TRUE_RANK
                  << "  RedSVD rank=" << K
                  << "  lambda=" << std::fixed << std::setprecision(4) << lambda
                  << "\n\n";

        // ---- Build A = L_true + S_true ---------------------------------
        // Low-rank signal: TRUE_RANK smooth sinusoidal rank-1 layers
        // (frequencies capped at 1.5 cycles/dim => clearly structured)
        std::vector<float> A_flat(M * N, 0.0f);
        for (int k = 0; k < TRUE_RANK; ++k) {
            float sigma = 10.0f / (k + 1.0f);
            float fr    = (k + 1) * PI / (M - 1.0f);
            float fc    = (k + 1) * PI / (N - 1.0f);
            float ph_r  = k * PI / 6.0f;
            float ph_c  = k * PI / 4.0f;
            for (int row = 0; row < M; ++row) {
                float u = std::sin(row * fr + ph_r);
                for (int col = 0; col < N; ++col)
                    A_flat[row * N + col] += sigma * u * std::cos(col * fc + ph_c);
            }
        }

        // Sparse corruption: ~15% of entries, amplitude 8-18
        {
            std::uniform_int_distribution<int> ri(0, M - 1), ci(0, N - 1);
            std::uniform_real_distribution<float> amp(8.0f, 18.0f);
            int n_corrupt = (M * N * 15) / 100;
            for (int i = 0; i < n_corrupt; ++i)
                A_flat[ri(rng) * N + ci(rng)] +=
                    (rng() & 1 ? 1.0f : -1.0f) * amp(rng);
        }

        // Frobenius norm of A (for convergence criterion and mu_0)
        float norm_A_fro = 0.0f;
        for (float v : A_flat) norm_A_fro += v * v;
        norm_A_fro = std::sqrt(norm_A_fro);

        // mu_0: approximates 1.25 / ||A||_2  using ||A||_F / sqrt(min(M,N))
        float mu     = 1.25f * std::sqrt((float)std::min(M, N)) / norm_A_fro;
        float mu_max = 1e6f * mu;

        std::cout << "||A||_F=" << norm_A_fro
                  << "  mu_0=" << mu
                  << "  lambda/mu_0=" << lambda / mu << "\n\n";

        // ---- Halide buffers (all pre-allocated; host pointers are fixed) --
        // buf(col, row) convention: Buffer<float>(width=cols, height=rows)

        Halide::Buffer<float> A_buf(N, M);      // M x N  (input, constant)
        for (int row = 0; row < M; ++row)
            for (int col = 0; col < N; ++col)
                A_buf(col, row) = A_flat[row * N + col];

        // Fixed random sketch matrices (generated once, never changed)
        Halide::Buffer<float> O_buf(K, M);      // M x K
        fill_gaussian(O_buf, rng);
        Halide::Buffer<float> P_buf(K, K);      // K x K
        fill_gaussian(P_buf, rng);

        // T_buf: updated every iteration  (T = A - S + Y/mu)
        Halide::Buffer<float> T_buf(N, M);      // M x N

        // Intermediate RedSVD buffers (overwritten each iteration)
        Halide::Buffer<float> Yh_buf(K, N);     // N x K   Y = T^T * O
        Halide::Buffer<float> QY_buf(K, N);     // N x K   orth(Y)
        Halide::Buffer<float> B_buf (K, M);     // M x K   B = T * QY
        Halide::Buffer<float> Z_buf (K, M);     // M x K   Z = B * P
        Halide::Buffer<float> QZ_buf(K, M);     // M x K   orth(Z)
        Halide::Buffer<float> C_buf (K, K);     // K x K   C = QZ^T * B
        Halide::Buffer<float> UC_buf(K, K);     // K x K   SVD left
        Halide::Buffer<float> Sg_buf(K);        // K       SVD singular vals
        Halide::Buffer<float> VtC_buf(K, K);    // K x K   SVD right^T
        Halide::Buffer<float> UC_s  (K, K);     // sorted+SVT left factor
        Halide::Buffer<float> VtC_s (K, K);     // sorted right factor
        Halide::Buffer<float> U_buf (K, M);     // M x K   U = QZ * UC_s
        Halide::Buffer<float> V_buf (K, N);     // N x K   V = QY * VC_s

        // ---- Define all Halide pipelines ONCE (JIT-compiled on first realize)
        Halide::Func T_T("T_T");   // T^T : N x M
        { Halide::Var c("c"), r("r"); T_T(c, r) = T_buf(r, c); }

        Halide::Func QZ_T("QZ_T");  // QZ^T : K x M
        { Halide::Var c("c"), r("r"); QZ_T(c, r) = QZ_buf(r, c); }

        auto T_as   = as_func(T_buf,   "Tf");
        auto O_as   = as_func(O_buf,   "O");
        auto P_as   = as_func(P_buf,   "P");
        auto Yh_as  = as_func(Yh_buf,  "Yhf");
        auto QY_f   = as_func(QY_buf,  "QY");
        auto B_as   = as_func(B_buf,   "Bf");
        auto Z_as   = as_func(Z_buf,   "Zf");
        auto QZ_f   = as_func(QZ_buf,  "QZ");
        auto C_as   = as_func(C_buf,   "Cf");
        auto UC_sf  = as_func(UC_s,    "UC_s");
        auto VCsf   = as_func(VtC_s,   "VtC_s");

        // Y  = T^T * O    [N x K]
        auto Y_func  = matmul(T_T,   {N, M}, O_as,  {M, K}, "Y_rp");
        // QY = orth(Y)    [N x K]
        auto qr_Y    = qr_gs (Yh_as, N,  K,         "qrY");
        // B  = T  * QY   [M x K]
        auto B_func  = matmul(T_as,  {M, N}, QY_f,  {N, K}, "B_rp");
        // Z  = B  * P    [M x K]
        auto Z_func  = matmul(B_as,  {M, K}, P_as,  {K, K}, "Z_rp");
        // QZ = orth(Z)   [M x K]
        auto qr_Z    = qr_gs (Z_as,  M,  K,         "qrZ");
        // C  = QZ^T * B  [K x K]
        auto C_func  = matmul(QZ_T,  {K, M}, B_as,  {M, K}, "C_rp");
        // SVD(C) -> U_C, sigma, Vt_C
        auto svd_C   = svd_jacobi(C_as, K, K, 10,   "svdC");
        // U  = QZ * UC_sorted  [M x K]
        auto U_func  = matmul(QZ_f,  {M, K}, UC_sf, {K, K}, "U_rp");
        // V  = QY * VC_sorted  [N x K]
        auto V_func  = matmul(QY_f,  {N, K}, VCsf,  {K, K}, "V_rp");

        // ---- RPCA-ALM state (CPU arrays) --------------------------------
        std::vector<float> L_data(M * N, 0.0f);   // current L
        std::vector<float> S_data(M * N, 0.0f);   // current S
        std::vector<float> Ylag  (M * N, 0.0f);   // dual variable

        std::vector<float> residuals;
        residuals.reserve(MAX_ITER);

        // ---- Main RPCA-ALM iteration loop --------------------------------
        std::cout << "Iterating (Halide pipelines compile on first pass):\n";
        for (int iter = 0; iter < MAX_ITER; ++iter) {

            // T = A - S + Y/mu  (CPU fill)
            for (int row = 0; row < M; ++row)
                for (int col = 0; col < N; ++col) {
                    int i = row * N + col;
                    T_buf(col, row) = A_flat[i] - S_data[i] + Ylag[i] / mu;
                }

            // RedSVD of T (Halide; first iter compiles, subsequent reuse cache)
            Y_func.realize(Yh_buf);        // Y  = T^T * O
            qr_Y.Q.realize(QY_buf);        // QY = orth(Y)
            B_func.realize(B_buf);         // B  = T * QY
            Z_func.realize(Z_buf);         // Z  = B * P
            qr_Z.Q.realize(QZ_buf);        // QZ = orth(Z)
            C_func.realize(C_buf);         // C  = QZ^T * B
            svd_C.U .realize(UC_buf);
            svd_C.S .realize(Sg_buf);
            svd_C.Vt.realize(VtC_buf);

            // Sort singular values descending, soft-threshold by 1/mu
            std::vector<int> order(K);
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(), [&](int a, int b) {
                return std::abs(Sg_buf(a)) > std::abs(Sg_buf(b));
            });

            float tau = 1.0f / mu;  // SVT soft-threshold
            for (int i = 0; i < K; ++i) {
                int   j  = order[i];
                float sv = std::max(std::abs(Sg_buf(j)) - tau, 0.0f);
                // UC_s[:,i] = UC[:,j] * sv  (sigma folded into U column)
                // VtC_s[:,i]= V_C[:,j]  where V_C[r,j] = VtC_buf(r, j)
                for (int r = 0; r < K; ++r) {
                    UC_s (i, r) = UC_buf(j, r) * sv;
                    VtC_s(i, r) = VtC_buf(r, j);
                }
            }

            // U = QZ * UC_s  [M x K],  V = QY * VC_s  [N x K]
            U_func.realize(U_buf);
            V_func.realize(V_buf);

            // L = U_scaled * V^T  (sigma already in U_buf)
            for (int i = 0; i < M * N; ++i) L_data[i] = 0.0f;
            for (int k = 0; k < K; ++k)
                for (int row = 0; row < M; ++row) {
                    float uk = U_buf(k, row);
                    if (std::abs(uk) < 1e-12f) continue;
                    for (int col = 0; col < N; ++col)
                        L_data[row * N + col] += uk * V_buf(k, col);
                }

            // S = shrink(A - L + Y/mu, lambda/mu)
            float lam_mu = lambda / mu;
            for (int i = 0; i < M * N; ++i)
                S_data[i] = soft(A_flat[i] - L_data[i] + Ylag[i] / mu, lam_mu);

            // Dual update + convergence check
            float res = 0.0f;
            for (int i = 0; i < M * N; ++i) {
                float r = A_flat[i] - L_data[i] - S_data[i];
                res    += r * r;
                Ylag[i] += mu * r;
            }
            res = std::sqrt(res);
            residuals.push_back(res);

            mu = std::min(mu * rho, mu_max);

            float rel = res / norm_A_fro;
            std::cout << "  iter " << std::setw(2) << iter + 1
                      << "  ||A-L-S||_F=" << std::setw(9) << res
                      << "  rel=" << rel << "\n";

            if (rel < 1e-3f) {
                std::cout << "  Converged.\n";
                break;
            }
        }
        std::cout << "\n";

        // ---- Report recovery quality ------------------------------------
        // Count |S| > threshold  (half of minimum outlier amplitude = 4)
        float s_thresh = 4.0f;
        int n_detected = 0;
        for (float v : S_data) if (std::abs(v) > s_thresh) ++n_detected;
        int n_actual = (M * N * 15) / 100;
        std::cout << "Sparse S recovery  thresh=" << s_thresh
                  << "  detected=" << n_detected
                  << "  injected~=" << n_actual << "\n\n";

        // Nuclear norm of recovered L
        float nuc = 0.0f;
        for (int k = 0; k < K; ++k) {
            // U_buf(k,row) = U[:,k]*sigma[k]; extract sigma[k] as column L2 norm
            float col_norm = 0.0f;
            for (int row = 0; row < M; ++row)
                col_norm += U_buf(k, row) * U_buf(k, row);
            nuc += std::sqrt(col_norm);
        }
        std::cout << "Approx nuclear norm of L: " << nuc << "\n\n";

        // ---- Normalization for display ----------------------------------
        float A_min = *std::min_element(A_flat.begin(), A_flat.end());
        float A_max = *std::max_element(A_flat.begin(), A_flat.end());
        float A_rng = (A_max - A_min < 1e-8f) ? 1.0f : (A_max - A_min);

        std::vector<float> abs_S(M * N);
        for (int i = 0; i < M * N; ++i) abs_S[i] = std::abs(S_data[i]);
        float S_peak  = *std::max_element(abs_S.begin(), abs_S.end());
        float S_scale = (S_peak < 1e-8f) ? 1.0f : S_peak;

        Halide::Buffer<float> A_hbuf(N, M), L_hbuf(N, M), S_hbuf(N, M);
        for (int row = 0; row < M; ++row)
            for (int col = 0; col < N; ++col) {
                int i = row * N + col;
                A_hbuf(col, row) = (A_flat[i]  - A_min) / A_rng;
                L_hbuf(col, row) = std::clamp((L_data[i] - A_min) / A_rng, 0.0f, 1.0f);
                S_hbuf(col, row) = std::clamp(S_data[i] / S_scale * 0.5f + 0.5f, 0.0f, 1.0f);
            }

        // ---- Convergence curve (CPU -> buffer) -------------------------
        Halide::Buffer<float> conv_buf(Q, Q);
        {
            float r0  = residuals.empty() ? 1.0f : residuals[0];
            int   n_r = (int)residuals.size();
            for (int px = 0; px < Q; ++px) {
                float t   = (float)px / (Q - 1) * (n_r - 1);
                int   i0  = std::min((int)t, n_r - 1);
                int   i1  = std::min(i0 + 1, n_r - 1);
                float frac = t - i0;
                float rv  = residuals[i0] * (1.0f - frac) + residuals[i1] * frac;
                int   cy  = std::clamp((int)((1.0f - rv / r0) * (Q - 1)), 0, Q - 1);
                for (int py = 0; py < Q; ++py) {
                    float val;
                    if      (std::abs(py - cy) <= 1) val = 0.95f;  // curve
                    else if (py > cy)                val = 0.18f;  // filled below
                    else                             val = 0.05f;  // dark above
                    conv_buf(px, py) = val;
                }
            }
        }

        // ---- Halide rendering pipeline ----------------------------------
        Halide::Func fa("fa"), fl("fl"), fs("fs"), fconv("fconv");
        {
            Halide::Var c("c"), r("r");
            fa   (c, r) = A_hbuf(c, r);
            fl   (c, r) = L_hbuf(c, r);
            fs   (c, r) = S_hbuf(c, r);
            fconv(c, r) = conv_buf(c, r);
        }

        Halide::Var ox("ox"), oy("oy");
        Halide::Expr qx = ox / Q,  qy = oy / Q;
        Halide::Expr lx = ox % Q,  ly = oy % Q;

        Halide::Expr mr = Halide::clamp(
            Halide::cast<int>(Halide::cast<float>(ly) * M / Q), 0, M - 1);
        Halide::Expr mc = Halide::clamp(
            Halide::cast<int>(Halide::cast<float>(lx) * N / Q), 0, N - 1);

        Halide::Expr pix = Halide::select(
            qy == 0 && qx == 0, fa   (mc, mr),
            Halide::select(qy == 0 && qx == 1, fl   (mc, mr),
            Halide::select(qy == 1 && qx == 0, fs   (mc, mr),
                           fconv(lx, ly))));

        pix = Halide::select((ox == Q) || (oy == Q), 0.5f, pix);

        Halide::Func output("output");
        output(ox, oy) = Halide::cast<uint8_t>(
            Halide::clamp(pix * 255.0f, 0.0f, 255.0f));

        std::cout << "Rendering...\n";
        Halide::Runtime::Buffer<uint8_t> result(OUT, OUT);
        output.realize(result);

        const char* out_path = "out/43_redsvd.png";
        if (save_png(result, out_path)) {
            std::cout << "Saved to " << out_path << "\n\n";
            std::cout << "Quadrant guide:\n";
            std::cout << "  Top-left:     Original A  (signal + ~15% sparse outliers)\n";
            std::cout << "  Top-right:    Recovered L  (low-rank, same scale as A)\n";
            std::cout << "  Bottom-left:  Recovered S  (sparse component, gray=0)\n";
            std::cout << "  Bottom-right: Convergence  ||A-L-S||_F / ||A||_F per iter\n";
        } else {
            std::cerr << "Error: failed to save PNG\n";
            return 1;
        }
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
