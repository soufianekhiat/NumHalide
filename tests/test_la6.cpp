#include <gtest/gtest.h>
#include <Halide.h>
#include "../src/numhalide_all.h"
using namespace numhalide;

static Halide::Func make_1d(std::initializer_list<float> vals, const std::string& n = "f")
{
    std::vector<float> v(vals);
    Halide::Buffer<float> buf((int)v.size());
    for (int i = 0; i < (int)v.size(); ++i) buf(i) = v[i];
    Halide::Func f(n); Halide::Var x; f(x) = buf(x); return f;
}
static Halide::Func make_mat(int rows, int cols,
    std::initializer_list<float> vals, const std::string& n = "m")
{
    std::vector<float> v(vals);
    Halide::Buffer<float> buf(cols, rows);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            buf(c, r) = v[(size_t)(r * cols + c)];
    Halide::Func f(n); Halide::Var x, y; f(x, y) = buf(x, y); return f;
}

// ---- eig2x2 ----

TEST(La6Eig, Eig2x2_RealDistinct) {
    // [[3,0],[0,1]] → eigenvalues 3, 1
    auto A = make_mat(2, 2, {3.0f,0.0f, 0.0f,1.0f}, "A_e2rd");
    auto [re, im] = eig2x2(A, "e2rd");
    Halide::Runtime::Buffer<float> r(2), i(2);
    re.realize(r); im.realize(i);
    EXPECT_NEAR(r(0), 3.0f, 1e-5f);
    EXPECT_NEAR(r(1), 1.0f, 1e-5f);
    EXPECT_NEAR(i(0), 0.0f, 1e-5f);
    EXPECT_NEAR(i(1), 0.0f, 1e-5f);
}

TEST(La6Eig, Eig2x2_Symmetric) {
    // [[3,1],[1,3]] → eigenvalues 4, 2
    auto A = make_mat(2, 2, {3.0f,1.0f, 1.0f,3.0f}, "A_e2s");
    auto [re, im] = eig2x2(A, "e2s");
    Halide::Runtime::Buffer<float> r(2), i(2);
    re.realize(r); im.realize(i);
    EXPECT_NEAR(r(0), 4.0f, 1e-5f);
    EXPECT_NEAR(r(1), 2.0f, 1e-5f);
    EXPECT_NEAR(i(0), 0.0f, 1e-5f);
    EXPECT_NEAR(i(1), 0.0f, 1e-5f);
}

TEST(La6Eig, Eig2x2_Complex) {
    // [[0,-1],[1,0]] rotation matrix → eigenvalues +i, -i
    auto A = make_mat(2, 2, {0.0f,-1.0f, 1.0f,0.0f}, "A_e2c");
    auto [re, im] = eig2x2(A, "e2c");
    Halide::Runtime::Buffer<float> r(2), i(2);
    re.realize(r); im.realize(i);
    EXPECT_NEAR(r(0), 0.0f, 1e-5f);
    EXPECT_NEAR(r(1), 0.0f, 1e-5f);
    EXPECT_NEAR(i(0),  1.0f, 1e-5f);
    EXPECT_NEAR(i(1), -1.0f, 1e-5f);
}

TEST(La6Eig, Eig2x2_Identity) {
    // [[1,0],[0,1]] → eigenvalues 1, 1
    auto A = make_mat(2, 2, {1.0f,0.0f, 0.0f,1.0f}, "A_e2id");
    auto [re, im] = eig2x2(A, "e2id");
    Halide::Runtime::Buffer<float> r(2), i(2);
    re.realize(r); im.realize(i);
    EXPECT_NEAR(r(0), 1.0f, 1e-5f);
    EXPECT_NEAR(r(1), 1.0f, 1e-5f);
    EXPECT_NEAR(i(0), 0.0f, 1e-5f);
    EXPECT_NEAR(i(1), 0.0f, 1e-5f);
}

// ---- eig_qr ----

TEST(La6Eig, EigQR_2x2_Delegates) {
    // eig_qr with n=2 delegates to eig2x2
    auto A = make_mat(2, 2, {5.0f,0.0f, 0.0f,2.0f}, "A_eqr2");
    auto [re, im] = eig_qr(A, 2, 40, "eqr2");
    Halide::Runtime::Buffer<float> r(2), i(2);
    re.realize(r); im.realize(i);
    EXPECT_NEAR(r(0), 5.0f, 1e-4f);
    EXPECT_NEAR(r(1), 2.0f, 1e-4f);
    EXPECT_NEAR(i(0), 0.0f, 1e-4f);
    EXPECT_NEAR(i(1), 0.0f, 1e-4f);
}

TEST(La6Eig, EigQR_3x3_Diagonal_ZeroIter) {
    // Diagonal input already converged — 0 iterations reads diagonal directly
    auto A = make_mat(3, 3, {1.0f,0.0f,0.0f, 0.0f,2.0f,0.0f, 0.0f,0.0f,3.0f}, "A_eqr3d0");
    auto [re, im] = eig_qr(A, 3, 0, "eqr3d0");
    Halide::Runtime::Buffer<float> r(3);
    re.realize(r);
    EXPECT_NEAR(r(0), 1.0f, 1e-4f);
    EXPECT_NEAR(r(1), 2.0f, 1e-4f);
    EXPECT_NEAR(r(2), 3.0f, 1e-4f);
}

TEST(La6Eig, EigQR_3x3_Diagonal_Iterates) {
    // QR on diagonal should stay diagonal (Q=I, R=A → RQ=A)
    auto A = make_mat(3, 3, {4.0f,0.0f,0.0f, 0.0f,2.0f,0.0f, 0.0f,0.0f,1.0f}, "A_eqr3di");
    auto [re, im] = eig_qr(A, 3, 2, "eqr3di");
    Halide::Runtime::Buffer<float> r(3);
    re.realize(r);
    EXPECT_NEAR(r(0), 4.0f, 1e-3f);
    EXPECT_NEAR(r(1), 2.0f, 1e-3f);
    EXPECT_NEAR(r(2), 1.0f, 1e-3f);
}

// ---- polyfit ----

TEST(La6Polyfit, Linear_Exact) {
    // y = 2x + 1 at x={0,1,2}: exact solution
    auto xf = make_1d({0.0f, 1.0f, 2.0f}, "x_ple");
    auto yf = make_1d({1.0f, 3.0f, 5.0f}, "y_ple");
    auto c = polyfit(xf, yf, 3, 1, "ple");
    Halide::Runtime::Buffer<float> out(2);
    c.realize(out);
    EXPECT_NEAR(out(0), 1.0f, 1e-3f);  // constant term
    EXPECT_NEAR(out(1), 2.0f, 1e-3f);  // slope
}

TEST(La6Polyfit, Quadratic_Exact) {
    // y = x^2 at x={0,1,2,3}: exact solution
    auto xf = make_1d({0.0f, 1.0f, 2.0f, 3.0f}, "x_pqe");
    auto yf = make_1d({0.0f, 1.0f, 4.0f, 9.0f}, "y_pqe");
    auto c = polyfit(xf, yf, 4, 2, "pqe");
    Halide::Runtime::Buffer<float> out(3);
    c.realize(out);
    EXPECT_NEAR(out(0), 0.0f, 1e-3f);  // x^0 coefficient
    EXPECT_NEAR(out(1), 0.0f, 1e-3f);  // x^1 coefficient
    EXPECT_NEAR(out(2), 1.0f, 1e-3f);  // x^2 coefficient
}

TEST(La6Polyfit, Linear_Overdetermined) {
    // y = 3x + 1 at 5 points: overdetermined but exact solution exists
    auto xf = make_1d({0.0f, 1.0f, 2.0f, 3.0f, 4.0f}, "x_plo");
    auto yf = make_1d({1.0f, 4.0f, 7.0f, 10.0f, 13.0f}, "y_plo");
    auto c = polyfit(xf, yf, 5, 1, "plo");
    Halide::Runtime::Buffer<float> out(2);
    c.realize(out);
    EXPECT_NEAR(out(0), 1.0f, 1e-2f);  // constant term
    EXPECT_NEAR(out(1), 3.0f, 1e-2f);  // slope
}
