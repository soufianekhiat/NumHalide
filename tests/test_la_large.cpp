/// @file test_la_large.cpp
/// @brief Tests for la_large.h — runtime-sized LA wrappers up to 32x32

#include <gtest/gtest.h>
#include <Halide.h>
#include "../src/numhalide_all.h"
#include "../src/la_large.h"
#include <cmath>
#include <algorithm>

using namespace numhalide;

// =============================================================================
// Helpers
// =============================================================================

/// Build a positive-definite SYMMETRIC n×n matrix as a Halide Func.
/// Uses A[i,j] = (n + diag_strength)*delta(i,j) + 0.3*sin(min(i,j)*n+max(i,j))
/// The off-diagonal formula is symmetric: A[i,j] = A[j,i].
static Halide::Func make_pd_matrix(int n, float diag_strength, const std::string& nm)
{
    Halide::Buffer<float> data(n, n);  // data(col=j, row=i) = A[i,j]
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            // Symmetric off-diagonal: use sorted (lo,hi) so A[i,j]=A[j,i]
            int lo = std::min(i, j), hi = std::max(i, j);
            data(j, i) = (i == j)
                ? static_cast<float>(n) + diag_strength + 0.5f * std::sin(static_cast<float>(2 * i))
                : 0.3f * std::sin(static_cast<float>(lo * n + hi));
        }
    Halide::Func f(nm);
    Halide::Var x("x"), y("y");
    f(x, y) = data(Halide::clamp(x, 0, n - 1), Halide::clamp(y, 0, n - 1));
    return f;
}

/// Build a well-conditioned (diagonal-dominant) n×n matrix for QR testing.
/// A[i,j] = (n+4)*delta(i,j) + 0.5*sin(i+j)*cos(i-j)
/// Symmetric and well-conditioned — ensures good QR orthogonality.
static Halide::Func make_conditioned_matrix(int n, const std::string& nm)
{
    Halide::Buffer<float> data(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            data(j, i) = (i == j)
                ? static_cast<float>(n + 4)
                : 0.5f * std::sin(static_cast<float>(i + j)) * std::cos(static_cast<float>(i - j));
    Halide::Func f(nm);
    Halide::Var x("x"), y("y");
    f(x, y) = data(Halide::clamp(x, 0, n - 1), Halide::clamp(y, 0, n - 1));
    return f;
}

/// Build a general (non-symmetric) m×n matrix Func with controlled values.
/// Entry [i,j] = 1/(1 + i + j*m) + 0.1*sin(i*n + j)
static Halide::Func make_general_matrix(int m, int n, const std::string& nm)
{
    Halide::Buffer<float> data(n, m);  // data(col, row)
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            data(j, i) = 1.0f / (1.0f + static_cast<float>(i + j * m))
                       + 0.1f * std::sin(static_cast<float>(i * n + j));

    Halide::Func f(nm);
    Halide::Var x("x"), y("y");
    f(x, y) = data(Halide::clamp(x, 0, n - 1), Halide::clamp(y, 0, m - 1));
    return f;
}

/// Frobenius norm of a dense float matrix (row-major CPU buffer).
static float frobenius(const std::vector<std::vector<float>>& M, int rows, int cols)
{
    double sum = 0.0;
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            sum += static_cast<double>(M[i][j]) * M[i][j];
    return static_cast<float>(std::sqrt(sum));
}

/// Realize an n×n Func into a 2D vector [row][col].
static std::vector<std::vector<float>> realize_mat(Halide::Func f, int rows, int cols)
{
    Halide::Runtime::Buffer<float> tmp(cols, rows);
    f.realize(tmp);
    std::vector<std::vector<float>> out(rows, std::vector<float>(cols));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            out[i][j] = tmp(j, i);
    return out;
}

/// C = A * B  (dense, rows_a x cols_a @ cols_a x cols_b -> rows_a x cols_b)
static std::vector<std::vector<float>> matmul_cpu(
    const std::vector<std::vector<float>>& A, int ra, int ca,
    const std::vector<std::vector<float>>& B, int rb, int cb)
{
    (void)rb;
    std::vector<std::vector<float>> C(ra, std::vector<float>(cb, 0.0f));
    for (int i = 0; i < ra; ++i)
        for (int j = 0; j < cb; ++j)
            for (int k = 0; k < ca; ++k)
                C[i][j] += A[i][k] * B[k][j];
    return C;
}

/// Transpose a matrix stored as vector-of-vectors.
static std::vector<std::vector<float>> transpose_cpu(
    const std::vector<std::vector<float>>& A, int rows, int cols)
{
    std::vector<std::vector<float>> At(cols, std::vector<float>(rows));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            At[j][i] = A[i][j];
    return At;
}

// =============================================================================
// Cholesky tests
// =============================================================================

/// 1. Cholesky_n4: verify L*L^T = A within tolerance for a 4×4 PD matrix.
TEST(LALarge, Cholesky_n4)
{
    const int n = 4;
    auto A = make_pd_matrix(n, 4.0f, "pd4");
    auto L_func = cholesky_large(A, n, "chol4");

    auto Lm = realize_mat(L_func, n, n);

    // Read original A values
    Halide::Runtime::Buffer<float> ab(n, n);
    A.realize(ab);

    // Compute L * L^T in CPU
    auto Lt = transpose_cpu(Lm, n, n);
    auto LLt = matmul_cpu(Lm, n, n, Lt, n, n);

    // Compare element-wise
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            EXPECT_NEAR(LLt[i][j], ab(j, i), 1e-4f)
                << "Mismatch at (" << i << "," << j << ")";
}

/// 2. Cholesky_n8: diagonal-dominant 8×8, verify diagonal of L is positive.
TEST(LALarge, Cholesky_n8)
{
    const int n = 8;
    auto A = make_pd_matrix(n, 8.0f, "pd8");
    auto L_func = cholesky_large(A, n, "chol8");

    auto Lm = realize_mat(L_func, n, n);

    // All diagonal entries must be positive
    for (int i = 0; i < n; ++i)
        EXPECT_GT(Lm[i][i], 0.0f) << "L[" << i << "," << i << "] <= 0";
}

/// 3. Cholesky_n16: 16×16 identity-scaled matrix, verify L*L^T ≈ A.
TEST(LALarge, Cholesky_n16)
{
    const int n = 16;
    auto A = make_pd_matrix(n, 20.0f, "pd16");
    auto L_func = cholesky_large(A, n, "chol16");

    auto Lm = realize_mat(L_func, n, n);

    Halide::Runtime::Buffer<float> ab(n, n);
    A.realize(ab);

    auto Lt = transpose_cpu(Lm, n, n);
    auto LLt = matmul_cpu(Lm, n, n, Lt, n, n);

    // Check Frobenius norm of (L*L^T - A)
    std::vector<std::vector<float>> diff(n, std::vector<float>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            diff[i][j] = LLt[i][j] - ab(j, i);

    float err = frobenius(diff, n, n);
    EXPECT_LT(err, 0.1f) << "||L*L^T - A||_F = " << err;
}

// =============================================================================
// QR tests
// =============================================================================

/// 4. QR_n4: 4×4, verify Q orthogonality (Q^T Q ≈ I) and A = Q*R.
TEST(LALarge, QR_n4)
{
    const int m = 4, n = 4;
    auto A = make_general_matrix(m, n, "qr4a");
    auto [Q, R] = qr_large(A, m, n, "qr4");

    auto Qm = realize_mat(Q, m, n);
    auto Rm = realize_mat(R, n, n);

    // Q^T * Q should be identity
    auto Qt = transpose_cpu(Qm, m, n);  // n×m
    auto QtQ = matmul_cpu(Qt, n, m, Qm, m, n);  // n×n

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            float expected = (i == j) ? 1.0f : 0.0f;
            EXPECT_NEAR(QtQ[i][j], expected, 1e-4f)
                << "Q^T Q [" << i << "," << j << "]";
        }

    // Q * R should recover A
    auto QR = matmul_cpu(Qm, m, n, Rm, n, n);  // m×n
    Halide::Runtime::Buffer<float> ab(n, m);
    A.realize(ab);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            EXPECT_NEAR(QR[i][j], ab(j, i), 1e-4f)
                << "A vs Q*R at (" << i << "," << j << ")";
}

/// 5. QR_m8n4: 8×4 tall matrix, verify Q orthogonality.
TEST(LALarge, QR_m8n4)
{
    const int m = 8, n = 4;
    auto A = make_general_matrix(m, n, "qrta");
    auto [Q, R] = qr_large(A, m, n, "qrt");

    auto Qm = realize_mat(Q, m, n);

    // Q^T * Q should be identity (n×n)
    auto Qt = transpose_cpu(Qm, m, n);
    auto QtQ = matmul_cpu(Qt, n, m, Qm, m, n);

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            float expected = (i == j) ? 1.0f : 0.0f;
            EXPECT_NEAR(QtQ[i][j], expected, 1e-4f)
                << "Q^T Q [" << i << "," << j << "]";
        }
}

/// 6. QR_n16: 16×16, verify ||A - Q*R||_F < 0.01.
TEST(LALarge, QR_n16)
{
    const int m = 16, n = 16;
    auto A = make_general_matrix(m, n, "qr16a");
    auto [Q, R] = qr_large(A, m, n, "qr16");

    auto Qm = realize_mat(Q, m, n);
    auto Rm = realize_mat(R, n, n);
    auto QR = matmul_cpu(Qm, m, n, Rm, n, n);

    Halide::Runtime::Buffer<float> ab(n, m);
    A.realize(ab);

    std::vector<std::vector<float>> diff(m, std::vector<float>(n));
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            diff[i][j] = QR[i][j] - ab(j, i);

    float err = frobenius(diff, m, n);
    EXPECT_LT(err, 0.01f) << "||A - Q*R||_F = " << err;
}

// =============================================================================
// SVD tests
// =============================================================================

/// 7. SVD_n4_Reconstruction: 4×4, verify A ≈ U * diag(S) * Vt within 0.05.
TEST(LALarge, SVD_n4_Reconstruction)
{
    const int m = 4, n = 4;
    auto A = make_general_matrix(m, n, "svd4a");
    auto [U, S, Vt] = svd_large(A, m, n, -1, "svd4");

    auto Um  = realize_mat(U,  m, n);
    auto Vtm = realize_mat(Vt, n, n);

    Halide::Runtime::Buffer<float> sb(n);
    S.realize(sb);

    // Build diag(S) as n×n
    std::vector<std::vector<float>> Dm(n, std::vector<float>(n, 0.0f));
    for (int i = 0; i < n; ++i)
        Dm[i][i] = sb(i);

    // Reconstruct: U * D * Vt
    auto UD  = matmul_cpu(Um, m, n, Dm, n, n);
    auto UDVt = matmul_cpu(UD, m, n, Vtm, n, n);

    Halide::Runtime::Buffer<float> ab(n, m);
    A.realize(ab);

    std::vector<std::vector<float>> diff(m, std::vector<float>(n));
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            diff[i][j] = UDVt[i][j] - ab(j, i);

    float err = frobenius(diff, m, n);
    EXPECT_LT(err, 0.05f) << "||A - U*S*Vt||_F = " << err;
}

/// 8. SVD_n8_Reconstruction: 8×8, verify reconstruction within tolerance.
TEST(LALarge, SVD_n8_Reconstruction)
{
    const int m = 8, n = 8;
    auto A = make_general_matrix(m, n, "svd8a");
    auto [U, S, Vt] = svd_large(A, m, n, -1, "svd8");

    auto Um  = realize_mat(U,  m, n);
    auto Vtm = realize_mat(Vt, n, n);

    Halide::Runtime::Buffer<float> sb(n);
    S.realize(sb);

    std::vector<std::vector<float>> Dm(n, std::vector<float>(n, 0.0f));
    for (int i = 0; i < n; ++i)
        Dm[i][i] = sb(i);

    auto UD   = matmul_cpu(Um,  m, n, Dm,  n, n);
    auto UDVt = matmul_cpu(UD,  m, n, Vtm, n, n);

    Halide::Runtime::Buffer<float> ab(n, m);
    A.realize(ab);

    std::vector<std::vector<float>> diff(m, std::vector<float>(n));
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            diff[i][j] = UDVt[i][j] - ab(j, i);

    float err = frobenius(diff, m, n);
    EXPECT_LT(err, 0.1f) << "||A - U*S*Vt||_F = " << err;
}

/// 9. Cholesky_LargeN_Validation: n=33 must throw (exceeds LA_LARGE_MAX_N=32).
TEST(LALarge, Cholesky_LargeN_Validation)
{
    Halide::Func dummy("dummy_chol_val");
    Halide::Var x("x"), y("y");
    dummy(x, y) = Halide::cast<float>(x + y);

    EXPECT_THROW(
        cholesky_large(dummy, 33, "chol_oor"),
        std::runtime_error
    );
}

/// 10. QR_Orthogonality_n16: verify ||Q^T Q - I||_F < 0.01 for 16×16.
TEST(LALarge, QR_Orthogonality_n16)
{
    const int m = 16, n = 16;
    auto A = make_conditioned_matrix(n, "qr16oa");
    auto [Q, R] = qr_large(A, m, n, "qr16o");

    auto Qm = realize_mat(Q, m, n);
    auto Qt = transpose_cpu(Qm, m, n);
    auto QtQ = matmul_cpu(Qt, n, m, Qm, m, n);

    // Subtract identity
    std::vector<std::vector<float>> err_mat(n, std::vector<float>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            err_mat[i][j] = QtQ[i][j] - (i == j ? 1.0f : 0.0f);

    float err = frobenius(err_mat, n, n);
    EXPECT_LT(err, 0.01f) << "||Q^T Q - I||_F = " << err;
}

/// 11. SVD_SingularValues_n4: all singular values must be non-negative.
TEST(LALarge, SVD_SingularValues_n4)
{
    const int m = 4, n = 4;
    auto A = make_general_matrix(m, n, "svd4sv_a");
    auto [U, S, Vt] = svd_large(A, m, n, -1, "svd4sv");

    Halide::Runtime::Buffer<float> sb(n);
    S.realize(sb);

    for (int i = 0; i < n; ++i)
        EXPECT_GE(sb(i), 0.0f) << "Singular value S[" << i << "] < 0";
}

/// 12. Cholesky_TriangularCheck_n8: L must be lower-triangular (L[col > row] == 0).
TEST(LALarge, Cholesky_TriangularCheck_n8)
{
    const int n = 8;
    auto A = make_pd_matrix(n, 8.0f, "pd8tc");
    auto L_func = cholesky_large(A, n, "chol8tc");

    auto Lm = realize_mat(L_func, n, n);

    // For lower-triangular: L[i][j] == 0 when j > i  (col > row)
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            EXPECT_NEAR(Lm[i][j], 0.0f, 1e-5f)
                << "L[row=" << i << ", col=" << j << "] != 0 (upper triangle)";
}
