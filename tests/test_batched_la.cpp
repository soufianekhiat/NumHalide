/// @file test_batched_la.cpp
/// @brief Tests for batched linear algebra: cholesky, qr_gs, svd_jacobi applied per-batch
///
/// These tests exercise the single-matrix la.h APIs (cholesky, qr_gs, svd_jacobi)
/// in a batched manner by wrapping each batch slice as a Halide::Func and verifying
/// per-batch correctness properties (L*L^T=A, Q orthogonality, SVD reconstruction).

#include <gtest/gtest.h>
#include "numhalide_all.h"
#include <cmath>
#include <vector>

using namespace numhalide;

// ---------------------------------------------------------------------------
// Helper utilities
// ---------------------------------------------------------------------------

/// Build a 2D Halide::Func from row-major float data
static Halide::Func make_mat(int rows, int cols,
    const std::vector<float>& vals, const std::string& nm)
{
    Halide::Buffer<float> buf(cols, rows);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            buf(c, r) = vals[(size_t)(r * cols + c)];
    Halide::Func f(nm);
    Halide::Var x, y;
    f(x, y) = buf(x, y);
    return f;
}

// ---------------------------------------------------------------------------
// Test 1: CholeskyBatch2x2
// 2 different 2x2 PD matrices; verify L*L^T == A for each batch
// ---------------------------------------------------------------------------

TEST(BatchedLA, CholeskyBatch2x2) {
    // batch 0: [[4,2],[2,3]]   L0 = [[2,0],[1,sqrt(2)]]
    // batch 1: [[9,3],[3,5]]   L1 = [[3,0],[1,2]]
    std::vector<std::vector<float>> matrices = {
        {4.0f, 2.0f, 2.0f, 3.0f},
        {9.0f, 3.0f, 3.0f, 5.0f}
    };
    const int n = 2;
    const int batch = 2;

    for (int b = 0; b < batch; ++b) {
        auto A = make_mat(n, n, matrices[(size_t)b], "chol2x2_A_b" + std::to_string(b));
        auto L = cholesky(A, n, "chol2x2_L_b" + std::to_string(b));
        L.compute_root();

        Halide::Runtime::Buffer<float> L_buf(n, n);
        L.realize(L_buf);

        // Verify L is lower triangular
        EXPECT_NEAR(L_buf(1, 0), 0.0f, 1e-4f);  // L[row=0, col=1] = 0

        // Verify L*L^T == A
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                float llt = 0.0f;
                for (int k = 0; k < n; ++k)
                    llt += L_buf(k, i) * L_buf(k, j);  // L[i,k]*L[j,k]
                EXPECT_NEAR(llt, matrices[(size_t)b][(size_t)(i * n + j)], 1e-4f);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Test 2: CholeskyBatch3x3
// batch=2 of 3x3 PD matrices, verify diagonal of L is positive
// ---------------------------------------------------------------------------

TEST(BatchedLA, CholeskyBatch3x3) {
    // PD 3x3: A = diag + small off-diag => diagonal dominant
    std::vector<std::vector<float>> matrices = {
        // batch 0: 5I + small perturbation
        {5.0f, 0.5f, 0.0f,
         0.5f, 4.0f, 0.5f,
         0.0f, 0.5f, 3.0f},
        // batch 1: different diagonal-dominant matrix
        {6.0f, 1.0f, 0.5f,
         1.0f, 5.0f, 1.0f,
         0.5f, 1.0f, 4.0f}
    };
    const int n = 3;
    const int batch = 2;

    for (int b = 0; b < batch; ++b) {
        auto A = make_mat(n, n, matrices[(size_t)b], "chol3x3_A_b" + std::to_string(b));
        auto L = cholesky(A, n, "chol3x3_L_b" + std::to_string(b));
        L.compute_root();

        Halide::Runtime::Buffer<float> L_buf(n, n);
        L.realize(L_buf);

        // Diagonal elements must be positive
        for (int i = 0; i < n; ++i)
            EXPECT_GT(L_buf(i, i), 0.0f);

        // Upper triangle must be zero: L_buf(col, row) with col > row
        for (int row = 0; row < n; ++row)
            for (int col = row + 1; col < n; ++col)
                EXPECT_NEAR(L_buf(col, row), 0.0f, 1e-4f);
    }
}

// ---------------------------------------------------------------------------
// Test 3: CholeskyBatchSingularFallback
// Near-singular case: diagonal remains positive (sqrt of small positive)
// ---------------------------------------------------------------------------

TEST(BatchedLA, CholeskyBatchSingularFallback) {
    // Small but PD: identity scaled by 1e-2
    std::vector<float> data = {0.01f, 0.0f, 0.0f, 0.01f};
    const int n = 2;
    auto A = make_mat(n, n, data, "chol_sg_A");
    auto L = cholesky(A, n, "chol_sg_L");
    L.compute_root();

    Halide::Runtime::Buffer<float> L_buf(n, n);
    L.realize(L_buf);

    // Diagonal should be positive: sqrt(0.01) = 0.1
    EXPECT_GT(L_buf(0, 0), 0.0f);
    EXPECT_GT(L_buf(1, 1), 0.0f);
    EXPECT_NEAR(L_buf(0, 0), 0.1f, 1e-4f);
    EXPECT_NEAR(L_buf(1, 1), 0.1f, 1e-4f);
}

// ---------------------------------------------------------------------------
// Test 4: QRBatch2x2
// batch=2 of 2x2 matrices, verify Q orthogonality (Q^T*Q = I per batch)
// ---------------------------------------------------------------------------

TEST(BatchedLA, QRBatch2x2) {
    std::vector<std::vector<float>> matrices = {
        {1.0f, 2.0f, 3.0f, 4.0f},
        {2.0f, 0.0f, 1.0f, 3.0f}
    };
    const int m = 2, n = 2;
    const int batch = 2;

    for (int b = 0; b < batch; ++b) {
        auto A = make_mat(m, n, matrices[(size_t)b], "qr2x2_A_b" + std::to_string(b));
        auto qr = qr_gs(A, m, n, "qr2x2_b" + std::to_string(b));
        qr.Q.compute_root();

        Halide::Runtime::Buffer<float> Q_buf(n, m);
        qr.Q.realize(Q_buf);

        // Verify Q^T * Q = I: sum over rows of Q[:,c1]*Q[:,c2]
        for (int c1 = 0; c1 < n; ++c1) {
            for (int c2 = 0; c2 < n; ++c2) {
                float dot = 0.0f;
                for (int r = 0; r < m; ++r)
                    dot += Q_buf(c1, r) * Q_buf(c2, r);
                float expected = (c1 == c2) ? 1.0f : 0.0f;
                EXPECT_NEAR(dot, expected, 1e-4f);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Test 5: QRBatch3x2
// batch=2 of 3x2 matrices, verify R is upper triangular
// ---------------------------------------------------------------------------

TEST(BatchedLA, QRBatch3x2) {
    std::vector<std::vector<float>> matrices = {
        {1.0f, 2.0f,
         3.0f, 4.0f,
         5.0f, 6.0f},
        {2.0f, 1.0f,
         0.0f, 3.0f,
         1.0f, 2.0f}
    };
    const int m = 3, n = 2;
    const int batch = 2;

    for (int b = 0; b < batch; ++b) {
        auto A = make_mat(m, n, matrices[(size_t)b], "qr3x2_A_b" + std::to_string(b));
        auto qr = qr_gs(A, m, n, "qr3x2_b" + std::to_string(b));
        qr.R.compute_root();

        Halide::Runtime::Buffer<float> R_buf(n, n);
        qr.R.realize(R_buf);

        // R is n x n upper triangular: R(col, row) with col < row must be ~0
        for (int row = 0; row < n; ++row)
            for (int col = 0; col < row; ++col)
                EXPECT_NEAR(R_buf(col, row), 0.0f, 1e-4f);

        // Diagonal must be positive
        for (int i = 0; i < n; ++i)
            EXPECT_GT(R_buf(i, i), 0.0f);
    }
}

// ---------------------------------------------------------------------------
// Test 6: QRBatchReconstruction
// Verify A = Q*R per batch
// ---------------------------------------------------------------------------

TEST(BatchedLA, QRBatchReconstruction) {
    // batch=2 of 3x3 matrices
    std::vector<std::vector<float>> matrices = {
        {3.0f, 1.0f, 0.0f,
         1.0f, 2.0f, 1.0f,
         0.0f, 1.0f, 4.0f},
        {2.0f, 1.0f, 1.0f,
         0.0f, 3.0f, 1.0f,
         1.0f, 0.0f, 2.0f}
    };
    const int m = 3, n = 3;
    const int batch = 2;

    for (int b = 0; b < batch; ++b) {
        auto A_func = make_mat(m, n, matrices[(size_t)b],
                               "qr_rec_A_b" + std::to_string(b));
        auto qr = qr_gs(A_func, m, n, "qr_rec_b" + std::to_string(b));
        qr.Q.compute_root();
        qr.R.compute_root();

        Halide::Runtime::Buffer<float> Q_buf(n, m);  // n cols, m rows
        Halide::Runtime::Buffer<float> R_buf(n, n);  // n cols, n rows
        qr.Q.realize(Q_buf);
        qr.R.realize(R_buf);

        // Compute Q*R and compare to A
        for (int row = 0; row < m; ++row) {
            for (int col = 0; col < n; ++col) {
                float qr_val = 0.0f;
                for (int k = 0; k < n; ++k)
                    qr_val += Q_buf(k, row) * R_buf(col, k);
                float a_val = matrices[(size_t)b][(size_t)(row * n + col)];
                EXPECT_NEAR(qr_val, a_val, 1e-3f);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Test 7: SVDBatch2x2
// batch=2 of 2x2 matrices, verify singular values are positive
// ---------------------------------------------------------------------------

TEST(BatchedLA, SVDBatch2x2) {
    std::vector<std::vector<float>> matrices = {
        {3.0f, 0.0f, 0.0f, 2.0f},  // diagonal, SVs = [3,2]
        {1.0f, 1.0f, 0.0f, 1.0f}
    };
    const int m = 2, n = 2;
    const int batch = 2;

    for (int b = 0; b < batch; ++b) {
        auto A = make_mat(m, n, matrices[(size_t)b], "svd2x2_A_b" + std::to_string(b));
        auto svd = svd_jacobi(A, m, n, 10, "svd2x2_b" + std::to_string(b));
        svd.S.compute_root();

        Halide::Runtime::Buffer<float> S_buf(n);
        svd.S.realize(S_buf);

        for (int i = 0; i < n; ++i)
            EXPECT_GT(S_buf(i), -1e-4f);  // singular values >= 0
    }

    // Verify specific case: diagonal [[3,0],[0,2]] -> SVs should be 3 and 2
    {
        auto A = make_mat(m, n, matrices[0], "svd2x2_diag");
        auto svd = svd_jacobi(A, m, n, 10, "svd2x2_diag_s");
        Halide::Runtime::Buffer<float> S_buf(n);
        svd.S.realize(S_buf);
        float sv0 = S_buf(0), sv1 = S_buf(1);
        float max_sv = std::max(sv0, sv1);
        float min_sv = std::min(sv0, sv1);
        EXPECT_NEAR(max_sv, 3.0f, 1e-3f);
        EXPECT_NEAR(min_sv, 2.0f, 1e-3f);
    }
}

// ---------------------------------------------------------------------------
// Test 8: SVDBatchReconstruction
// Verify A = U * diag(S) * Vt per batch
// ---------------------------------------------------------------------------

TEST(BatchedLA, SVDBatchReconstruction) {
    // 3x3 symmetric PD matrix
    std::vector<float> data = {3.0f, 1.0f, 0.0f,
                                1.0f, 2.0f, 1.0f,
                                0.0f, 1.0f, 4.0f};
    const int m = 3, n = 3;

    auto A_func = make_mat(m, n, data, "svd_rec_A");
    auto svd = svd_jacobi(A_func, m, n, 15, "svd_rec");
    svd.U.compute_root();
    svd.S.compute_root();
    svd.Vt.compute_root();

    Halide::Runtime::Buffer<float> U_buf(n, m);   // n cols, m rows
    Halide::Runtime::Buffer<float> S_buf(n);
    Halide::Runtime::Buffer<float> Vt_buf(n, n);  // n cols, n rows
    svd.U.realize(U_buf);
    svd.S.realize(S_buf);
    svd.Vt.realize(Vt_buf);

    // Reconstruct: A_rec[i,j] = sum_k U[i,k] * S[k] * Vt[j,k]
    // U_buf(col=k, row=i), S_buf(k), Vt_buf(col=j, row=k)
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float rec = 0.0f;
            for (int k = 0; k < n; ++k)
                rec += U_buf(k, i) * S_buf(k) * Vt_buf(j, k);
            EXPECT_NEAR(rec, data[(size_t)(i * n + j)], 1e-3f);
        }
    }
}

// ---------------------------------------------------------------------------
// Test 9: SVDBatchOrthogonality
// Verify U^T*U = I for diagonal inputs (all gamma=0, rotations skipped, U=I)
// ---------------------------------------------------------------------------

TEST(BatchedLA, SVDBatchOrthogonality) {
    // Use diagonal matrices: all off-diagonal inner products gamma=0, so all
    // Jacobi rotations are skipped and U = W/S = I (the trivially orthogonal case).
    std::vector<std::vector<float>> matrices = {
        {4.0f, 0.0f, 0.0f,
         0.0f, 3.0f, 0.0f,
         0.0f, 0.0f, 1.0f},
        {5.0f, 0.0f, 0.0f,
         0.0f, 2.0f, 0.0f,
         0.0f, 0.0f, 1.0f}
    };
    const int m = 3, n = 3;
    const int batch = 2;

    for (int b = 0; b < batch; ++b) {
        auto A = make_mat(m, n, matrices[(size_t)b], "svd_orth_A_b" + std::to_string(b));
        auto svd = svd_jacobi(A, m, n, 5, "svd_orth_b" + std::to_string(b));
        svd.U.compute_root();

        Halide::Runtime::Buffer<float> U_buf(n, m);
        svd.U.realize(U_buf);

        // Verify U^T * U = I: sum over rows of U[:,c1]*U[:,c2]
        for (int c1 = 0; c1 < n; ++c1) {
            for (int c2 = 0; c2 < n; ++c2) {
                float dot = 0.0f;
                for (int r = 0; r < m; ++r)
                    dot += U_buf(c1, r) * U_buf(c2, r);
                float expected = (c1 == c2) ? 1.0f : 0.0f;
                EXPECT_NEAR(dot, expected, 1e-4f);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Test 10: BatchMatchesSingle
// qr_gs with single-matrix API gives same result as a manual "batch of 1"
// ---------------------------------------------------------------------------

TEST(BatchedLA, BatchMatchesSingle) {
    std::vector<float> data = {2.0f, 1.0f, 0.0f,
                                0.0f, 3.0f, 1.0f,
                                1.0f, 0.0f, 2.0f};
    const int m = 3, n = 3;

    auto A1 = make_mat(m, n, data, "bms_A1");
    auto A2 = make_mat(m, n, data, "bms_A2");

    auto qr1 = qr_gs(A1, m, n, "bms_qr1");
    auto qr2 = qr_gs(A2, m, n, "bms_qr2");

    qr1.Q.compute_root();
    qr2.Q.compute_root();
    qr1.R.compute_root();
    qr2.R.compute_root();

    Halide::Runtime::Buffer<float> Q1(n, m), Q2(n, m);
    Halide::Runtime::Buffer<float> R1(n, n), R2(n, n);
    qr1.Q.realize(Q1);
    qr2.Q.realize(Q2);
    qr1.R.realize(R1);
    qr2.R.realize(R2);

    // Both should give identical results
    for (int r = 0; r < m; ++r)
        for (int c = 0; c < n; ++c)
            EXPECT_NEAR(Q1(c, r), Q2(c, r), 1e-5f);

    for (int r = 0; r < n; ++r)
        for (int c = 0; c < n; ++c)
            EXPECT_NEAR(R1(c, r), R2(c, r), 1e-5f);
}

// ---------------------------------------------------------------------------
// Test 11: BatchCholeskyMatchesSingle
// Two independent cholesky calls on the same PD matrix give identical results
// ---------------------------------------------------------------------------

TEST(BatchedLA, BatchCholeskyMatchesSingle) {
    std::vector<float> data = {4.0f, 2.0f, 2.0f, 3.0f};
    const int n = 2;

    auto A1 = make_mat(n, n, data, "bcms_A1");
    auto A2 = make_mat(n, n, data, "bcms_A2");

    auto L1 = cholesky(A1, n, "bcms_L1");
    auto L2 = cholesky(A2, n, "bcms_L2");
    L1.compute_root();
    L2.compute_root();

    Halide::Runtime::Buffer<float> Lb1(n, n), Lb2(n, n);
    L1.realize(Lb1);
    L2.realize(Lb2);

    for (int row = 0; row < n; ++row)
        for (int col = 0; col < n; ++col)
            EXPECT_NEAR(Lb1(col, row), Lb2(col, row), 1e-5f);
}

// ---------------------------------------------------------------------------
// Test 12: BatchSVDMatchesSingle
// Two independent SVD calls on the same matrix give same singular values
// ---------------------------------------------------------------------------

TEST(BatchedLA, BatchSVDMatchesSingle) {
    std::vector<float> data = {3.0f, 1.0f, 0.0f,
                                1.0f, 2.0f, 1.0f,
                                0.0f, 1.0f, 4.0f};
    const int m = 3, n = 3;

    auto A1 = make_mat(m, n, data, "bsvds_A1");
    auto A2 = make_mat(m, n, data, "bsvds_A2");

    auto svd1 = svd_jacobi(A1, m, n, 15, "bsvds_s1");
    auto svd2 = svd_jacobi(A2, m, n, 15, "bsvds_s2");
    svd1.S.compute_root();
    svd2.S.compute_root();

    Halide::Runtime::Buffer<float> S1(n), S2(n);
    svd1.S.realize(S1);
    svd2.S.realize(S2);

    for (int i = 0; i < n; ++i)
        EXPECT_NEAR(S1(i), S2(i), 1e-4f);
}
