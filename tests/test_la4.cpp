#include <gtest/gtest.h>
#include <Halide.h>
#include "../src/numhalide_all.h"
#include <cmath>
using namespace numhalide;

// Helper: make 2D matrix from row-major data
static Halide::Func make_mat(int rows, int cols,
    std::initializer_list<float> vals, const std::string& n = "m")
{
    std::vector<float> v(vals);
    Halide::Buffer<float> buf(cols, rows);  // Halide: (width=cols, height=rows)
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            buf(c, r) = v[(size_t)(r * cols + c)];
    Halide::Func f(n);
    Halide::Var x, y;
    f(x, y) = buf(x, y);
    return f;
}

// --- inv3x3 ---

TEST(LA4, Inv3x3_Identity) {
    auto I = make_mat(3, 3, {1,0,0, 0,1,0, 0,0,1}, "id3");
    auto inv = inv3x3(I);
    Halide::Runtime::Buffer<float> out(3, 3);
    inv.realize(out);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            EXPECT_NEAR(out(c, r), (c == r ? 1.0f : 0.0f), 1e-5f);
}

TEST(LA4, Inv3x3_Triangular) {
    // A = [[1,2,0],[0,1,0],[0,0,1]], inv = [[1,-2,0],[0,1,0],[0,0,1]]
    auto A = make_mat(3, 3, {1,2,0, 0,1,0, 0,0,1}, "A3tri");
    auto inv = inv3x3(A);
    Halide::Runtime::Buffer<float> out(3, 3);
    inv.realize(out);
    // inv[row=0, col=0]=1, [row=0, col=1]=-2, [row=0, col=2]=0
    // In Halide: out(col, row)
    EXPECT_NEAR(out(0, 0),  1.0f, 1e-5f);
    EXPECT_NEAR(out(1, 0), -2.0f, 1e-5f);
    EXPECT_NEAR(out(2, 0),  0.0f, 1e-5f);
    EXPECT_NEAR(out(0, 1),  0.0f, 1e-5f);
    EXPECT_NEAR(out(1, 1),  1.0f, 1e-5f);
    EXPECT_NEAR(out(2, 1),  0.0f, 1e-5f);
    EXPECT_NEAR(out(0, 2),  0.0f, 1e-5f);
    EXPECT_NEAR(out(1, 2),  0.0f, 1e-5f);
    EXPECT_NEAR(out(2, 2),  1.0f, 1e-5f);
}

TEST(LA4, Inv3x3_RoundTrip) {
    // A @ inv(A) ≈ I
    auto A = make_mat(3, 3, {2,1,0, 1,3,1, 0,1,4}, "Artrt");
    auto inv = inv3x3(A, "inv_rt");
    inv.compute_root();

    // Multiply A @ inv
    shape_t sA = {3, 3};
    auto prod = matmul(A, sA, inv, sA, "prod_rt");
    Halide::Runtime::Buffer<float> out(3, 3);
    prod.realize(out);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            EXPECT_NEAR(out(c, r), (c == r ? 1.0f : 0.0f), 1e-4f);
}

// --- svd2x2 ---

TEST(LA4, SVD2x2_Identity) {
    auto I = make_mat(2, 2, {1,0, 0,1}, "I2");
    auto svd = svd2x2(I);
    Halide::Runtime::Buffer<float> S(2);
    svd.S.realize(S);
    // Singular values of identity are [1, 1]
    EXPECT_NEAR(S(0), 1.0f, 1e-4f);
    EXPECT_NEAR(S(1), 1.0f, 1e-4f);
}

TEST(LA4, SVD2x2_Reconstruction) {
    // M = [[3,1],[1,2]], reconstruct U @ diag(S) @ Vt = M
    auto M = make_mat(2, 2, {3,1, 1,2}, "M2sv");
    auto svd = svd2x2(M, "svd_rc");
    svd.U.compute_root();
    svd.S.compute_root();
    svd.Vt.compute_root();

    // Build diag(S) as 2×2
    Halide::Var x("x"), y("y"), k("k");
    Halide::Func diagS("diagS");
    diagS(x, y) = Halide::select(x == y, svd.S(x), 0.0f);
    diagS.compute_root();

    // US = U @ diag(S)
    shape_t s22 = {2, 2};
    auto US = matmul(svd.U, s22, diagS, s22, "US");
    US.compute_root();

    // Rec = US @ Vt
    auto Rec = matmul(US, s22, svd.Vt, s22, "Rec");
    Halide::Runtime::Buffer<float> out(2, 2);
    Rec.realize(out);

    // Original matrix values
    EXPECT_NEAR(out(0, 0), 3.0f, 1e-3f);
    EXPECT_NEAR(out(1, 0), 1.0f, 1e-3f);
    EXPECT_NEAR(out(0, 1), 1.0f, 1e-3f);
    EXPECT_NEAR(out(1, 1), 2.0f, 1e-3f);
}

TEST(LA4, SVD2x2_SingularValuesPositive) {
    auto M = make_mat(2, 2, {1,2, 3,4}, "M2sp");
    auto svd = svd2x2(M, "svd_sp");
    Halide::Runtime::Buffer<float> S(2);
    svd.S.realize(S);
    EXPECT_GT(S(0), 0.0f);
    EXPECT_GT(S(1), 0.0f);
}

// --- cholesky ---

TEST(LA4, Cholesky_Identity2x2) {
    // chol(I) = I (lower triangular I)
    auto I = make_mat(2, 2, {1,0, 0,1}, "I_ch2");
    auto L = cholesky(I, 2);
    Halide::Runtime::Buffer<float> out(2, 2);
    L.realize(out);
    // L should be identity
    EXPECT_NEAR(out(0, 0), 1.0f, 1e-5f);
    EXPECT_NEAR(out(0, 1), 0.0f, 1e-5f);  // upper part = 0
    EXPECT_NEAR(out(1, 0), 0.0f, 1e-5f);  // actually L[row=0,col=1] = 0 in Halide: out(1,0)=L[row=0,col=1]=0
    EXPECT_NEAR(out(1, 1), 1.0f, 1e-5f);
}

TEST(LA4, Cholesky_2x2) {
    // A = [[4,2],[2,3]], L = [[2,0],[1,sqrt(2)]]
    auto A = make_mat(2, 2, {4,2, 2,3}, "A_ch2");
    auto L = cholesky(A, 2);
    Halide::Runtime::Buffer<float> out(2, 2);
    L.realize(out);
    // out(col, row): L[row=0,col=0]=2, L[row=1,col=0]=1, L[row=0,col=1]=0, L[row=1,col=1]=sqrt(2)
    EXPECT_NEAR(out(0, 0), 2.0f,           1e-4f);  // L[0,0]
    EXPECT_NEAR(out(0, 1), 1.0f,           1e-4f);  // L[1,0]
    EXPECT_NEAR(out(1, 0), 0.0f,           1e-4f);  // L[0,1] = 0 (upper)
    EXPECT_NEAR(out(1, 1), std::sqrt(2.0f), 1e-4f); // L[1,1]
}

TEST(LA4, Cholesky_3x3_RoundTrip) {
    // A = [[4,2,2],[2,5,3],[2,3,6]], compute L, verify L @ L^T = A
    auto A = make_mat(3, 3, {4,2,2, 2,5,3, 2,3,6}, "A_ch3");
    auto L = cholesky(A, 3, "L_ch3");
    L.compute_root();

    // L^T
    Halide::Var x("x"), y("y");
    Halide::Func Lt("Lt_ch3");
    Lt(x, y) = L(y, x);  // transpose: Lt(col, row) = L(row, col)
    Lt.compute_root();

    shape_t s33 = {3, 3};
    auto prod = matmul(L, s33, Lt, s33, "LLt");
    Halide::Runtime::Buffer<float> out(3, 3);
    prod.realize(out);

    // Compare with original A
    float A_data[] = {4,2,2, 2,5,3, 2,3,6};
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            EXPECT_NEAR(out(c, r), A_data[r * 3 + c], 1e-3f);
}

// --- qr_gs ---

TEST(LA4, QR_2x2_Orthonormal) {
    // Q^T @ Q = I
    auto A = make_mat(2, 2, {1,2, 3,4}, "A_qr2");
    auto [Q, R] = qr_gs(A, 2, 2);
    Q.compute_root();

    // Q^T
    Halide::Var x("x"), y("y");
    Halide::Func Qt("Qt_qr2");
    Qt(x, y) = Q(y, x);
    Qt.compute_root();

    shape_t s22 = {2, 2};
    auto QtQ = matmul(Qt, s22, Q, s22, "QtQ");
    Halide::Runtime::Buffer<float> out(2, 2);
    QtQ.realize(out);
    EXPECT_NEAR(out(0, 0), 1.0f, 1e-4f);
    EXPECT_NEAR(out(1, 0), 0.0f, 1e-4f);
    EXPECT_NEAR(out(0, 1), 0.0f, 1e-4f);
    EXPECT_NEAR(out(1, 1), 1.0f, 1e-4f);
}

TEST(LA4, QR_2x2_Reconstruction) {
    // Q @ R = A
    auto A = make_mat(2, 2, {1,2, 3,4}, "A_qrr2");
    auto [Q, R] = qr_gs(A, 2, 2, "qr_rec");
    Q.compute_root();
    R.compute_root();

    shape_t s22 = {2, 2};
    auto QR = matmul(Q, s22, R, s22, "QR_rec");
    Halide::Runtime::Buffer<float> out(2, 2);
    QR.realize(out);
    EXPECT_NEAR(out(0, 0), 1.0f, 1e-3f);
    EXPECT_NEAR(out(1, 0), 2.0f, 1e-3f);
    EXPECT_NEAR(out(0, 1), 3.0f, 1e-3f);
    EXPECT_NEAR(out(1, 1), 4.0f, 1e-3f);
}

TEST(LA4, QR_3x3_Orthonormal) {
    auto A = make_mat(3, 3, {1,2,3, 4,5,6, 7,8,10}, "A_qr3o");
    auto [Q, R] = qr_gs(A, 3, 3, "qr3o");
    Q.compute_root();

    Halide::Var x("x"), y("y");
    Halide::Func Qt("Qt3o");
    Qt(x, y) = Q(y, x);
    Qt.compute_root();

    shape_t s33 = {3, 3};
    auto QtQ = matmul(Qt, s33, Q, s33, "QtQ3o");
    Halide::Runtime::Buffer<float> out(3, 3);
    QtQ.realize(out);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            EXPECT_NEAR(out(c, r), (c == r ? 1.0f : 0.0f), 1e-3f);
}

TEST(LA4, QR_3x3_Reconstruction) {
    auto A = make_mat(3, 3, {1,2,3, 4,5,6, 7,8,10}, "A_qr3r");
    auto [Q, R] = qr_gs(A, 3, 3, "qr3r");
    Q.compute_root();
    R.compute_root();

    shape_t s33 = {3, 3};
    auto QR = matmul(Q, s33, R, s33, "QR3r");
    Halide::Runtime::Buffer<float> out(3, 3);
    QR.realize(out);

    float A_data[] = {1,2,3, 4,5,6, 7,8,10};
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            EXPECT_NEAR(out(c, r), A_data[r * 3 + c], 1e-3f);
}
