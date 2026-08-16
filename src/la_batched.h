/// @file la_batched.h
/// @brief Batched linear algebra: QR, Cholesky, and SVD for stacks of matrices
///
/// All functions accept A(col, row, batch_idx) and return results with the same
/// batch dimension. Internally, each 2D intermediate Func gains a batch Var b,
/// and all reductions add a batch dimension to operate independently per slice.
///
/// Batched matmul is in la.h as batched_matmul().

#pragma once

#include "la.h"

NS_NUM_HALIDE_BEGIN

// ============================================================================
// Batched Cholesky
// ============================================================================

/// @brief Cholesky decomposition for a batch of PD matrices
/// @param A     3D Func A(col, row, batch): stack of n×n symmetric PD matrices
/// @param n     Matrix size
/// @param batch Number of matrices
/// @param name  Base Func name
/// @return 3D Func L(col, row, batch): lower-triangular Cholesky factor per slice
///
/// Each slice satisfies A[b] = L[b] · L[b]^T.
/// Internally creates O(n) compute_root stages (same as non-batched).
inline
Halide::Func batched_cholesky(Halide::Func A, int n, int batch,
    std::string const& name = "batched_chol")
{
    Halide::Var col("col"), row("row"), b("b");

    // Compute in A's own float type (f64 stays f64); integer inputs keep
    // the historical f32 path. The sqrt floor keeps the old 1e-12 value,
    // made in that type.
    Halide::Type type = A.types()[0];
    if (!type.is_float()) type = Halide::Float(32);
    Halide::Expr zero = Halide::Internal::make_zero(type);

    Halide::Func L_cur(name + "_init");
    L_cur(col, row, b) = zero;
    L_cur.compute_root();

    for (int j = 0; j < n; ++j) {
        const std::string sj = std::to_string(j);

        // ss(b) = sum_{k<j} L[b, j, k]^2
        Halide::Func ss(name + "_ss" + sj);
        ss(b) = zero;
        if (j > 0) {
            Halide::RDom rk(0, j, "rkss_b" + sj);
            ss(b) += L_cur(rk, j, b) * L_cur(rk, j, b);
        }
        ss.compute_root();

        // diag_j(b) = sqrt(A[b,j,j] - ss(b))
        Halide::Func diag_j(name + "_dj" + sj);
        diag_j(b) = Halide::sqrt(Halide::max(Halide::cast(type, A(j, j, b)) - ss(b),
            Halide::Internal::make_const(type, 1e-12)));
        diag_j.compute_root();

        // dots(row, b) = sum_{k<j} L[b, row, k] * L[b, j, k]
        Halide::Func dots(name + "_dt" + sj);
        dots(row, b) = zero;
        if (j > 0) {
            Halide::RDom rk(0, j, "rkd_b" + sj);
            dots(row, b) += L_cur(rk, row, b) * L_cur(rk, j, b);
        }
        dots.compute_root();

        // Build next L
        Halide::Func L_next(name + "_L" + sj);
        L_next(col, row, b) = L_cur(col, row, b);
        // Diagonal: L[b, j, j] = diag_j(b)
        {
            Halide::RDom rb(0, batch, "rb_d" + sj);
            L_next(j, j, rb) = diag_j(rb);
        }
        // Off-diagonal: L[b, i, j] = (A[b, i, j] - dots(i, b)) / diag_j(b)
        if (j < n - 1) {
            Halide::RDom ri_b(j + 1, n - j - 1, 0, batch, "ri_b" + sj);
            L_next(j, ri_b.x, ri_b.y) =
                (A(j, ri_b.x, ri_b.y) - dots(ri_b.x, ri_b.y)) / diag_j(ri_b.y);
        }
        L_next.compute_root();
        L_cur = L_next;
    }

    Halide::Func ret(name);
    ret(col, row, b) = L_cur(col, row, b);
    return ret;
}

// ============================================================================
// Batched QR
// ============================================================================

/// @brief Result of batched QR decomposition
struct BatchedQRResult {
    Halide::Func Q;  ///< 3D m×n×batch: Q(col, row, batch)
    Halide::Func R;  ///< 3D n×n×batch: R(col, row, batch)
};

/// @brief QR decomposition (modified Gram-Schmidt) for a batch of matrices
/// @param A     3D Func A(col, row, batch): stack of m×n matrices (m ≥ n)
/// @param m     Number of rows
/// @param n     Number of columns (≤ m)
/// @param batch Number of matrices
/// @param name  Base Func name
/// @return BatchedQRResult {Q (m×n×batch), R (n×n×batch)}
///
/// Each slice satisfies A[b] = Q[b] · R[b],  Q[b]^T · Q[b] = I.
/// Internally creates O(n) compute_root stages.
inline
BatchedQRResult batched_qr_gs(Halide::Func A, int m, int n, int batch,
    std::string const& name = "batched_qr")
{
    nh_require(m >= n, "batched_qr_gs: requires m >= n, got m=%d n=%d", m, n);

    Halide::Var col("col"), row("row"), b("b");

    // Compute in A's own float type (f64 stays f64); integer inputs keep
    // the historical f32 path.
    Halide::Type type = A.types()[0];
    if (!type.is_float()) type = Halide::Float(32);
    Halide::Expr zero = Halide::Internal::make_zero(type);

    Halide::Func Aw(name + "_aw0");
    Aw(col, row, b) = Halide::cast(type, A(col, row, b));
    Aw.compute_root();

    Halide::Func Q(name + "_q0");
    Q(col, row, b) = zero;
    Q.compute_root();

    Halide::Func R(name + "_r0");
    R(col, row, b) = zero;
    R.compute_root();

    for (int k = 0; k < n; ++k) {
        const std::string sk = std::to_string(k);

        // ss_k(b) = ||Aw[:,k,b]||^2
        Halide::Func ss_k(name + "_ssk" + sk);
        ss_k(b) = zero;
        { Halide::RDom rn(0, m, "rn_" + name + sk);
          ss_k(b) += Aw(k, rn, b) * Aw(k, rn, b); }
        ss_k.compute_root();

        Halide::Func nk(name + "_nk" + sk);
        nk(b) = Halide::sqrt(ss_k(b));
        nk.compute_root();

        // Q[:,k,b] = Aw[:,k,b] / nk(b)
        {
            Halide::Func Q_new(name + "_q" + sk);
            Q_new(col, row, b) = Q(col, row, b);
            { Halide::RDom rr_b(0, m, 0, batch, "rq_" + name + sk);
              Q_new(k, rr_b.x, rr_b.y) = Aw(k, rr_b.x, rr_b.y) / nk(rr_b.y); }
            Q_new.compute_root();
            Q = Q_new;
        }
        // R[k,k,b] = nk(b)
        {
            Halide::Func R_new(name + "_rd" + sk);
            R_new(col, row, b) = R(col, row, b);
            { Halide::RDom rb(0, batch, "rb_d" + name + sk);
              R_new(k, k, rb) = nk(rb); }
            R_new.compute_root();
            R = R_new;
        }

        if (k < n - 1) {
            // dp(col, b) = Q[:,k,b] · Aw[:,col,b]
            Halide::Func dp(name + "_dp" + sk);
            dp(col, b) = zero;
            { Halide::RDom rd(0, m, "rd_" + name + sk);
              dp(col, b) += Q(k, rd, b) * Aw(col, rd, b); }
            dp.compute_root();

            // R[k, j, b] = dp(j, b) for j in [k+1, n)
            {
                Halide::Func R_new(name + "_ru" + sk);
                R_new(col, row, b) = R(col, row, b);
                { Halide::RDom rj_b(k + 1, n - k - 1, 0, batch, "rru_" + name + sk);
                  R_new(rj_b.x, k, rj_b.y) = dp(rj_b.x, rj_b.y); }
                R_new.compute_root();
                R = R_new;
            }

            // Aw[:,j,b] -= dp(j,b) * Q[:,k,b]  for j in [k+1,n)
            {
                Halide::Func Aw_new(name + "_aw" + sk);
                Aw_new(col, row, b) = Aw(col, row, b);
                { Halide::RDom r_up(k + 1, n - k - 1, 0, m, 0, batch, "rawu_" + name + sk);
                  Aw_new(r_up.x, r_up.y, r_up.z) =
                      Aw(r_up.x, r_up.y, r_up.z) - dp(r_up.x, r_up.z) * Q(k, r_up.y, r_up.z); }
                Aw_new.compute_root();
                Aw = Aw_new;
            }
        }
    }

    Halide::Func Q_out(name + "_Q");
    Q_out(col, row, b) = Q(col, row, b);
    Halide::Func R_out(name + "_R");
    R_out(col, row, b) = R(col, row, b);
    return {Q_out, R_out};
}

// ============================================================================
// Batched SVD
// ============================================================================

/// @brief Result of batched SVD
struct BatchedSVDResult {
    Halide::Func U;   ///< 3D m×n×batch: U(col, row, batch)
    Halide::Func S;   ///< 2D n×batch:   S(k, batch)
    Halide::Func Vt;  ///< 3D n×n×batch: Vt(col, row, batch)
};

/// @brief Full SVD via Jacobi sweeps for a batch of matrices
/// @param A        3D Func A(col, row, batch): stack of m×n matrices (m ≥ n)
/// @param m        Number of rows
/// @param n        Number of columns (≤ m); keep n ≤ 6 for build time
/// @param batch    Number of matrices
/// @param n_sweeps Number of Jacobi sweeps
/// @param name     Base Func name
/// @return BatchedSVDResult {U (m×n×batch), S (n×batch), Vt (n×n×batch)}
///
/// Creates O(n_sweeps × n²) compute_root stages per batch — same as non-batched
/// since the batch dimension is carried along all existing stages.
inline
BatchedSVDResult batched_svd_jacobi(Halide::Func A, int m, int n, int batch,
    int n_sweeps = 10, std::string const& name = "batched_svd")
{
    Halide::Var col("col"), row("row"), b("b"), k("k");

    // Compute in A's own float type (f64 stays f64); integer inputs keep
    // the historical f32 path.
    Halide::Type type = A.types()[0];
    if (!type.is_float()) type = Halide::Float(32);
    Halide::Expr zero = Halide::Internal::make_zero(type);
    Halide::Expr one  = Halide::Internal::make_one(type);

    // W(col, row, b) starts as A
    Halide::Func W(name + "_W0");
    W(col, row, b) = Halide::cast(type, A(col, row, b));
    W.compute_root();

    // V(col, row, b) starts as identity (per batch)
    Halide::Func V(name + "_V0");
    V(col, row, b) = Halide::select(col == row, one, zero);
    V.compute_root();

    for (int sweep = 0; sweep < n_sweeps; ++sweep) {
        for (int p = 0; p < n - 1; ++p) {
            for (int q = p + 1; q < n; ++q) {
                const std::string tag = name
                    + "_s" + std::to_string(sweep)
                    + "p" + std::to_string(p)
                    + "q" + std::to_string(q);

                // alpha(b) = W[:,p,b] · W[:,p,b]
                Halide::Func alpha(tag + "_a");
                alpha(b) = zero;
                { Halide::RDom ra(0, m, "ra_" + tag);
                  alpha(b) += W(p, ra, b) * W(p, ra, b); }
                alpha.compute_root();

                Halide::Func beta(tag + "_b");
                beta(b) = zero;
                { Halide::RDom rb2(0, m, "rb_" + tag);
                  beta(b) += W(q, rb2, b) * W(q, rb2, b); }
                beta.compute_root();

                Halide::Func gam(tag + "_g");
                gam(b) = zero;
                { Halide::RDom rg(0, m, "rg_" + tag);
                  gam(b) += W(p, rg, b) * W(q, rg, b); }
                gam.compute_root();

                // Jacobi rotation (per batch).
                //
                // SKIP GUARD via multiplied 0/1 indicators, NOT a select
                // around the division: reverse-mode AD of a select
                // synthesizes an f32 zero for the inactive arm, which
                // type-clashes with the f64 derivative of the ζ division
                // inside the arm (same form as svd_jacobi in la.h).
                // rot = 1 applies the rotation; rot = 0 yields the exact
                // identity (cs = 1, sn = 0), and the guarded denominator
                // keeps the skipped branch finite (|2γ + 1| ≥ 1 − 2e-10).
                Halide::Func cs_f(tag + "_cs");
                {
                    Halide::Expr g    = gam(b);
                    Halide::Expr rot  = Halide::cast(type,
                        Halide::abs(g) >= Halide::Internal::make_const(type, 1e-10));
                    Halide::Expr skp  = one - rot;
                    Halide::Expr zeta = (beta(b) - alpha(b))
                        / (Halide::Internal::make_const(type, 2.0) * g + skp);
                    Halide::Expr abz  = Halide::abs(zeta);
                    Halide::Expr t    = Halide::select(zeta >= zero, one,
                                            Halide::Internal::make_const(type, -1.0))
                                        / (abz + Halide::sqrt(one + zeta * zeta));
                    cs_f(b) = skp + rot * (one / Halide::sqrt(one + t * t));
                }
                cs_f.compute_root();

                Halide::Func sn_f(tag + "_sn");
                {
                    Halide::Expr g    = gam(b);
                    Halide::Expr rot  = Halide::cast(type,
                        Halide::abs(g) >= Halide::Internal::make_const(type, 1e-10));
                    Halide::Expr skp  = one - rot;
                    Halide::Expr zeta = (beta(b) - alpha(b))
                        / (Halide::Internal::make_const(type, 2.0) * g + skp);
                    Halide::Expr abz  = Halide::abs(zeta);
                    Halide::Expr t    = Halide::select(zeta >= zero, one,
                                            Halide::Internal::make_const(type, -1.0))
                                        / (abz + Halide::sqrt(one + zeta * zeta));
                    // rot·cs·t: 0 when skipping, cs·t when rotating.
                    sn_f(b) = rot * cs_f(b) * t;
                }
                sn_f.compute_root();

                // Update W[:,p,b] and W[:,q,b]. Rotation-sign convention: the
                // t from t^2 + 2*zeta*t - 1 = 0 pairs with wp' = c*wp - s*wq /
                // wq' = s*wp + c*wq — the annihilating convention svd_jacobi
                // was fixed to (commit 6ccebe2). The old opposite pairing
                // OSCILLATED: measured max|U^T U - I| of 0.27..0.49 at sweeps
                // 4/10/30, invisible to reconstruction since U*S*V^T = A holds
                // for any orthogonally-drifting pair.
                Halide::Func W_next(tag + "_W");
                W_next(col, row, b) = W(col, row, b);
                { Halide::RDom rp(0, m, 0, batch, "rWp_" + tag);
                  W_next(p, rp.x, rp.y) =  cs_f(rp.y) * W(p, rp.x, rp.y)
                                           - sn_f(rp.y) * W(q, rp.x, rp.y); }
                { Halide::RDom rq(0, m, 0, batch, "rWq_" + tag);
                  W_next(q, rq.x, rq.y) =  sn_f(rq.y) * W(p, rq.x, rq.y)
                                           + cs_f(rq.y) * W(q, rq.x, rq.y); }
                W_next.compute_root();
                W = W_next;

                // Update V[:,p,b] and V[:,q,b] with the same pairing.
                Halide::Func V_next(tag + "_V");
                V_next(col, row, b) = V(col, row, b);
                { Halide::RDom rvp(0, n, 0, batch, "rVp_" + tag);
                  V_next(p, rvp.x, rvp.y) =  cs_f(rvp.y) * V(p, rvp.x, rvp.y)
                                             - sn_f(rvp.y) * V(q, rvp.x, rvp.y); }
                { Halide::RDom rvq(0, n, 0, batch, "rVq_" + tag);
                  V_next(q, rvq.x, rvq.y) =  sn_f(rvq.y) * V(p, rvq.x, rvq.y)
                                             + cs_f(rvq.y) * V(q, rvq.x, rvq.y); }
                V_next.compute_root();
                V = V_next;
            }
        }
    }

    // S(k, b) = ||W[:,k,b]||
    Halide::Func ss_sv(name + "_ss");
    ss_sv(k, b) = zero;
    { Halide::RDom rsv(0, m, "rsv_" + name);
      ss_sv(k, b) += W(k, rsv, b) * W(k, rsv, b); }
    ss_sv.compute_root();

    Halide::Func S(name + "_S");
    S(k, b) = Halide::sqrt(ss_sv(k, b));

    // U(col, row, b) = W(col, row, b) / S(col, b)
    Halide::Func U(name + "_U");
    U(col, row, b) = W(col, row, b) / S(col, b);

    // Vt(col, row, b) = V(row, col, b)
    Halide::Func Vt(name + "_Vt");
    Vt(col, row, b) = V(row, col, b);

    return {U, S, Vt};
}

NS_NUM_HALIDE_END
