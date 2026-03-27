#include <gtest/gtest.h>
#include <Halide.h>
#include "../src/numhalide_all.h"
#include <cmath>
using namespace numhalide;

static Halide::Func make_mat(int rows, int cols,
    std::initializer_list<float> vals, const std::string& n = "m")
{
    std::vector<float> v(vals);
    Halide::Buffer<float> buf(cols, rows);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            buf(c, r) = v[(size_t)(r * cols + c)];
    Halide::Func f(n); Halide::Var x, y; f(x, y) = buf(x, y);
    return f;
}
static Halide::Func make_vec(std::initializer_list<float> vals,
    const std::string& n = "v")
{
    std::vector<float> v(vals);
    Halide::Buffer<float> buf((int)v.size());
    for (int i = 0; i < (int)v.size(); ++i) buf(i) = v[i];
    Halide::Func f(n); Halide::Var x; f(x) = buf(x);
    return f;
}

// ---- back_sub ----

TEST(LA5, BackSub_3x3) {
    // R = [[2,1,3],[0,4,-1],[0,0,5]], y=[14,7,5] → x=[4.5,2,1]
    auto R = make_mat(3, 3, {2,1,3, 0,4,-1, 0,0,5}, "R_bs");
    auto y = make_vec({14.0f, 7.0f, 5.0f}, "y_bs");
    auto x = back_sub(R, y, 3);
    Halide::Runtime::Buffer<float> out(3);
    x.realize(out);
    EXPECT_NEAR(out(0), 4.5f, 1e-4f);
    EXPECT_NEAR(out(1), 2.0f, 1e-4f);
    EXPECT_NEAR(out(2), 1.0f, 1e-4f);
}

TEST(LA5, FwdSub_3x3) {
    // L = [[2,0,0],[1,3,0],[2,-1,4]], b=[4,5,6] → y=[2,1,1]
    auto L = make_mat(3, 3, {2,0,0, 1,3,0, 2,-1,4}, "L_fs");
    auto b = make_vec({4.0f, 5.0f, 6.0f}, "b_fs");
    auto y = fwd_sub(L, b, 3);
    Halide::Runtime::Buffer<float> out(3);
    y.realize(out);
    EXPECT_NEAR(out(0), 2.0f,  1e-4f);  // 4/2
    EXPECT_NEAR(out(1), 1.0f,  1e-4f);  // (5-1*2)/3
    EXPECT_NEAR(out(2), 0.75f, 1e-4f);  // (6-2*2-(-1)*1)/4 = 3/4
}

// ---- svd_jacobi ----

TEST(LA5, SVD_Identity3) {
    auto I = make_mat(3, 3, {1,0,0, 0,1,0, 0,0,1}, "I3sv");
    auto svd = svd_jacobi(I, 3, 3);
    Halide::Runtime::Buffer<float> S(3);
    svd.S.realize(S);
    // Singular values of identity ≈ 1
    EXPECT_NEAR(S(0), 1.0f, 1e-3f);
    EXPECT_NEAR(S(1), 1.0f, 1e-3f);
    EXPECT_NEAR(S(2), 1.0f, 1e-3f);
}

TEST(LA5, SVD_Reconstruction3x3) {
    // M = [[3,1,0],[1,2,1],[0,1,4]]
    auto M = make_mat(3, 3, {3,1,0, 1,2,1, 0,1,4}, "M3sv");
    auto svd = svd_jacobi(M, 3, 3, 15, "sv3");
    svd.U.compute_root();
    svd.S.compute_root();
    svd.Vt.compute_root();

    Halide::Var x("x"), y("y");
    Halide::Func diagS("diagS3");
    diagS(x, y) = Halide::select(x == y, svd.S(x), 0.0f);
    diagS.compute_root();

    shape_t s33 = {3, 3};
    auto US   = matmul(svd.U,   s33, diagS,   s33, "US3");  US.compute_root();
    auto Rec  = matmul(US,      s33, svd.Vt,  s33, "Rec3");
    Halide::Runtime::Buffer<float> out(3, 3);
    Rec.realize(out);

    float M_data[] = {3,1,0, 1,2,1, 0,1,4};
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            EXPECT_NEAR(out(c, r), M_data[r * 3 + c], 0.01f);
}

TEST(LA5, SVD_SingularValues2x3) {
    // Non-square: [[1,2,0],[0,0,3]]  (m=2, n=3 → but n>m, so use m=3, n=2)
    // Use [[1,0],[2,0],[0,3]]  m=3, n=2
    auto A = make_mat(3, 2, {1,0, 2,0, 0,3}, "A23sv");
    auto svd = svd_jacobi(A, 3, 2, 10, "sv23");
    Halide::Runtime::Buffer<float> S(2);
    svd.S.realize(S);
    EXPECT_GT(S(0), 0.0f);
    EXPECT_GT(S(1), 0.0f);
}

// ---- eigh_jacobi ----

TEST(LA5, Eigh_Identity2) {
    auto I = make_mat(2, 2, {1,0, 0,1}, "I2eg");
    auto eg = eigh_jacobi(I, 2, 5);
    Halide::Runtime::Buffer<float> ev(2);
    eg.eigenvalues.realize(ev);
    // Eigenvalues of identity = 1, 1
    EXPECT_NEAR(ev(0), 1.0f, 1e-4f);
    EXPECT_NEAR(ev(1), 1.0f, 1e-4f);
}

TEST(LA5, Eigh_Symmetric2x2) {
    // [[3,1],[1,2]]: eigenvalues = (5 ± sqrt(5)) / 2 ≈ 3.618, 1.382
    auto A = make_mat(2, 2, {3,1, 1,2}, "A2eg");
    auto eg = eigh_jacobi(A, 2, 20);
    Halide::Runtime::Buffer<float> ev(2);
    eg.eigenvalues.realize(ev);
    // Sort the two eigenvalues
    float e0 = ev(0), e1 = ev(1);
    if (e0 < e1) std::swap(e0, e1);
    EXPECT_NEAR(e0, (5.0f + std::sqrt(5.0f)) / 2.0f, 1e-3f);
    EXPECT_NEAR(e1, (5.0f - std::sqrt(5.0f)) / 2.0f, 1e-3f);
}

TEST(LA5, Eigh_Reconstruction3x3) {
    // A = V diag(λ) V^T — check A' = V diag(λ) V^T ≈ A
    auto A = make_mat(3, 3, {4,2,2, 2,5,3, 2,3,6}, "A3eg");
    auto eg = eigh_jacobi(A, 3, 20, "eg3");
    eg.eigenvalues.compute_root();
    eg.eigenvectors.compute_root();

    Halide::Var x("x"), y("y");
    Halide::Func diagL("diagL3");
    diagL(x, y) = Halide::select(x == y, eg.eigenvalues(x), 0.0f);
    diagL.compute_root();

    // V (eigenvectors) is columns: V(col=k, row=i)
    // A' = V @ diag(λ) @ V^T
    Halide::Func Vt_eg("Vt_eg3");
    Vt_eg(x, y) = eg.eigenvectors(y, x);  // transpose
    Vt_eg.compute_root();

    shape_t s33 = {3, 3};
    auto VD  = matmul(eg.eigenvectors, s33, diagL, s33, "VD3"); VD.compute_root();
    auto Rec = matmul(VD, s33, Vt_eg, s33, "VDVt3");
    Halide::Runtime::Buffer<float> out(3, 3);
    Rec.realize(out);

    float A_data[] = {4,2,2, 2,5,3, 2,3,6};
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            EXPECT_NEAR(out(c, r), A_data[r * 3 + c], 0.01f);
}

// ---- solve ----

TEST(LA5, Solve_2x2) {
    // A = [[2,1],[1,3]], b=[5,10] → x=[1,3]
    auto A = make_mat(2, 2, {2,1, 1,3}, "A_slv2");
    auto b = make_vec({5.0f, 10.0f}, "b_slv2");
    auto x = solve(A, b, 2);
    Halide::Runtime::Buffer<float> out(2);
    x.realize(out);
    EXPECT_NEAR(out(0), 1.0f, 1e-3f);
    EXPECT_NEAR(out(1), 3.0f, 1e-3f);
}

TEST(LA5, Solve_3x3) {
    // A = [[1,0,0],[0,2,0],[0,0,3]], b=[1,2,3] → x=[1,1,1]
    auto A = make_mat(3, 3, {1,0,0, 0,2,0, 0,0,3}, "A_slv3");
    auto b = make_vec({1.0f, 2.0f, 3.0f}, "b_slv3");
    auto x = solve(A, b, 3);
    Halide::Runtime::Buffer<float> out(3);
    x.realize(out);
    EXPECT_NEAR(out(0), 1.0f, 1e-3f);
    EXPECT_NEAR(out(1), 1.0f, 1e-3f);
    EXPECT_NEAR(out(2), 1.0f, 1e-3f);
}

TEST(LA5, Solve_Residual) {
    // Check A @ x ≈ b
    auto A = make_mat(3, 3, {2,1,0, 1,3,1, 0,1,4}, "A_slvr");
    auto b = make_vec({3.0f, 6.0f, 5.0f}, "b_slvr");
    auto x = solve(A, b, 3, "slvr");
    x.compute_root();

    // Compute A @ x manually
    Halide::Var i("i");
    Halide::RDom rj(0, 3, "rj_res");
    Halide::Func Ax_f("Ax_res");
    Ax_f(i) = 0.0f;
    Ax_f(i) += A(rj, i) * x(rj);  // A(col=rj, row=i) * x(rj)
    Halide::Runtime::Buffer<float> out(3);
    Ax_f.realize(out);
    EXPECT_NEAR(out(0), 3.0f, 1e-3f);
    EXPECT_NEAR(out(1), 6.0f, 1e-3f);
    EXPECT_NEAR(out(2), 5.0f, 1e-3f);
}

// ---- lstsq ----

TEST(LA5, Lstsq_SquareExact) {
    // Square system = exact solve
    auto A = make_mat(2, 2, {2,1, 1,3}, "A_lsq2");
    auto b = make_vec({5.0f, 10.0f}, "b_lsq2");
    auto x = lstsq(A, b, 2, 2);
    Halide::Runtime::Buffer<float> out(2);
    x.realize(out);
    EXPECT_NEAR(out(0), 1.0f, 1e-3f);
    EXPECT_NEAR(out(1), 3.0f, 1e-3f);
}

TEST(LA5, Lstsq_Overdetermined) {
    // A = [[1,1],[1,2],[1,3]], b=[1,2,2] (line-fitting)
    // Least-squares solution ≈ [0.333, 0.5]
    auto A = make_mat(3, 2, {1,1, 1,2, 1,3}, "A_lsq3");
    auto b = make_vec({1.0f, 2.0f, 2.0f}, "b_lsq3");
    auto x = lstsq(A, b, 3, 2);
    Halide::Runtime::Buffer<float> out(2);
    x.realize(out);
    // Normal equations: A^T A x = A^T b
    // A^T A = [[3,6],[6,14]], A^T b = [5,11]
    // x = [[3,6],[6,14]]^{-1} [5,11] = [1/6, 7/6] ≈ [0.1667, 0.5]
    // Let me verify: det = 42-36=6, inv = [[14,-6],[-6,3]]/6
    // x = (14*5 + (-6)*11)/6, ((-6)*5 + 3*11)/6 = (70-66)/6, (-30+33)/6 = 4/6, 3/6 = 2/3, 0.5
    EXPECT_NEAR(out(0), 2.0f/3.0f, 0.02f);
    EXPECT_NEAR(out(1), 0.5f,      0.02f);
}

// ---- pinv ----

TEST(LA5, Pinv_Identity2) {
    auto I = make_mat(2, 2, {1,0, 0,1}, "I2pv");
    auto Pi = pinv(I, 2, 2, 1e-6f, 10, "pv2");
    Halide::Runtime::Buffer<float> out(2, 2);
    Pi.realize(out);
    // pinv(I) = I
    EXPECT_NEAR(out(0,0), 1.0f, 0.01f);
    EXPECT_NEAR(out(1,0), 0.0f, 0.01f);
    EXPECT_NEAR(out(0,1), 0.0f, 0.01f);
    EXPECT_NEAR(out(1,1), 1.0f, 0.01f);
}

TEST(LA5, Pinv_Tall) {
    // A = [[1,0],[0,1],[0,0]]  (3×2), pinv = [[1,0,0],[0,1,0]]  (2×3)
    auto A = make_mat(3, 2, {1,0, 0,1, 0,0}, "A3x2pv");
    auto Pi = pinv(A, 3, 2, 1e-6f, 10, "pv32");
    // pinv(3×2) → 2×3 Func, realize as Buffer(cols=m=3, rows=n=2)
    // pinv(col=j, row=i) where j ∈ [0,m) and i ∈ [0,n)
    Halide::Runtime::Buffer<float> out(3, 2);
    Pi.realize(out);
    // Expected pinv: [[1,0,0],[0,1,0]]
    EXPECT_NEAR(out(0,0), 1.0f, 0.02f);  // pinv[row=0, col=0]
    EXPECT_NEAR(out(1,0), 0.0f, 0.02f);  // pinv[row=0, col=1]
    EXPECT_NEAR(out(2,0), 0.0f, 0.02f);  // pinv[row=0, col=2]
    EXPECT_NEAR(out(0,1), 0.0f, 0.02f);  // pinv[row=1, col=0]
    EXPECT_NEAR(out(1,1), 1.0f, 0.02f);  // pinv[row=1, col=1]
    EXPECT_NEAR(out(2,1), 0.0f, 0.02f);  // pinv[row=1, col=2]
}

// ---- det_lu ----

TEST(LA5, DetLU_2x2) {
    // det([[3,8],[4,6]]) = 3*6 - 8*4 = 18-32 = -14
    auto A = make_mat(2, 2, {3,8, 4,6}, "A_det2");
    auto d = det_lu(A, 2);
    auto out = Halide::Runtime::Buffer<float>::make_scalar();
    d.realize(out);
    EXPECT_NEAR(out(), -14.0f, 1e-4f);
}

TEST(LA5, DetLU_3x3) {
    // det([[1,2,3],[4,5,6],[7,8,10]]) = -3 (verified)
    auto A = make_mat(3, 3, {1,2,3, 4,5,6, 7,8,10}, "A_det3");
    auto d = det_lu(A, 3);
    auto out = Halide::Runtime::Buffer<float>::make_scalar();
    d.realize(out);
    EXPECT_NEAR(out(), -3.0f, 0.01f);
}

TEST(LA5, DetLU_Identity) {
    auto I = make_mat(3, 3, {1,0,0, 0,1,0, 0,0,1}, "I_det");
    auto d = det_lu(I, 3);
    auto out = Halide::Runtime::Buffer<float>::make_scalar();
    d.realize(out);
    EXPECT_NEAR(out(), 1.0f, 1e-4f);
}

// ---- matrix_rank ----

TEST(LA5, MatrixRank_Identity3) {
    auto I = make_mat(3, 3, {1,0,0, 0,1,0, 0,0,1}, "I_mr");
    auto r = matrix_rank(I, 3, 3, 1e-4f);
    auto out = Halide::Runtime::Buffer<float>::make_scalar();
    r.realize(out);
    EXPECT_NEAR(out(), 3.0f, 0.5f);
}

TEST(LA5, MatrixRank_Rank1) {
    // rank-1 2×2 matrix: [[2,0],[0,0]] → singular values [2, 0]
    auto A = make_mat(2, 2, {2,0, 0,0}, "A_mr1");
    auto r = matrix_rank(A, 2, 2, 0.5f);
    auto out = Halide::Runtime::Buffer<float>::make_scalar();
    r.realize(out);
    EXPECT_NEAR(out(), 1.0f, 0.5f);
}

// ---- cond ----

TEST(LA5, Cond_Identity3) {
    auto I = make_mat(3, 3, {1,0,0, 0,1,0, 0,0,1}, "I_cond");
    auto c = cond(I, 3, 3);
    auto out = Halide::Runtime::Buffer<float>::make_scalar();
    c.realize(out);
    EXPECT_NEAR(out(), 1.0f, 0.01f);
}

TEST(LA5, Cond_Diagonal) {
    // diag([1,2,4]) → cond = 4/1 = 4
    auto A = make_mat(3, 3, {1,0,0, 0,2,0, 0,0,4}, "A_cond");
    auto c = cond(A, 3, 3);
    auto out = Halide::Runtime::Buffer<float>::make_scalar();
    c.realize(out);
    EXPECT_NEAR(out(), 4.0f, 0.1f);
}

// ---- slogdet ----

TEST(LA5, Slogdet_Identity3) {
    auto I = make_mat(3, 3, {1,0,0, 0,1,0, 0,0,1}, "I_sld");
    auto res = slogdet(I, 3);
    auto s = Halide::Runtime::Buffer<float>::make_scalar();
    auto l = Halide::Runtime::Buffer<float>::make_scalar();
    res.sign.realize(s);
    res.logabsdet.realize(l);
    EXPECT_NEAR(s(), 1.0f, 0.01f);   // sign = +1
    EXPECT_NEAR(l(), 0.0f, 0.01f);   // log(1) = 0
}

TEST(LA5, Slogdet_NegDet) {
    // det([[3,8],[4,6]]) = -14  →  sign=-1, logabsdet=log(14)
    auto A = make_mat(2, 2, {3,8, 4,6}, "A_sld");
    auto res = slogdet(A, 2);
    auto s = Halide::Runtime::Buffer<float>::make_scalar();
    auto l = Halide::Runtime::Buffer<float>::make_scalar();
    res.sign.realize(s);
    res.logabsdet.realize(l);
    EXPECT_NEAR(s(), -1.0f, 0.01f);
    EXPECT_NEAR(l(), std::log(14.0f), 0.01f);
}
