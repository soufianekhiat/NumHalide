/// @file test_autodiff_tensor.cpp
/// @brief Tests for autodiff_tensor.h — Tensor class and TVar reverse-mode AD

#include <gtest/gtest.h>
#include "numhalide_all.h"
#include <cmath>

using namespace numhalide;

// =============================================================================
// Tensor class tests
// =============================================================================

TEST(TensorAD, Tensor_Scalar) {
    Tensor s(3.0f);
    EXPECT_TRUE(s.is_scalar());
    EXPECT_EQ(s.rank(), 0);
    EXPECT_NEAR(s.item(), 3.0f, 1e-6f);
}

TEST(TensorAD, Tensor_Vector) {
    Tensor v({1.0f, 2.0f, 3.0f});
    EXPECT_EQ(v.rank(), 1);
    EXPECT_EQ(v.size(), 3);
    EXPECT_NEAR(v(0), 1.0f, 1e-6f);
    EXPECT_NEAR(v(2), 3.0f, 1e-6f);
}

TEST(TensorAD, Tensor_Matrix) {
    Tensor M({{1.0f, 2.0f}, {3.0f, 4.0f}});
    EXPECT_EQ(M.rank(), 2);
    EXPECT_EQ(M.dim(0), 2);
    EXPECT_EQ(M.dim(1), 2);
    EXPECT_NEAR(M(0, 0), 1.0f, 1e-6f);
    EXPECT_NEAR(M(1, 1), 4.0f, 1e-6f);
}

TEST(TensorAD, Tensor_Zeros_Ones) {
    Tensor z = Tensor::zeros({2, 3});
    EXPECT_EQ(z.rank(), 2);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
            EXPECT_NEAR(z(i, j), 0.0f, 1e-6f);

    Tensor o = Tensor::ones({3});
    for (int i = 0; i < 3; ++i)
        EXPECT_NEAR(o(i), 1.0f, 1e-6f);
}

TEST(TensorAD, Tensor_Eye) {
    Tensor I = Tensor::eye(3);
    EXPECT_EQ(I.rank(), 2);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            EXPECT_NEAR(I(i, j), i == j ? 1.0f : 0.0f, 1e-6f);
}

TEST(TensorAD, Tensor_Elementwise_Add) {
    Tensor a({1.0f, 2.0f, 3.0f});
    Tensor b({4.0f, 5.0f, 6.0f});
    Tensor c = a + b;
    EXPECT_NEAR(c(0), 5.0f, 1e-6f);
    EXPECT_NEAR(c(2), 9.0f, 1e-6f);
}

TEST(TensorAD, Tensor_ScalarBroadcast) {
    Tensor v({1.0f, 2.0f, 3.0f});
    Tensor c = v * Tensor(2.0f);
    EXPECT_NEAR(c(0), 2.0f, 1e-6f);
    EXPECT_NEAR(c(2), 6.0f, 1e-6f);
}

TEST(TensorAD, Tensor_Matmul) {
    // [[1,2],[3,4]] @ [[1,0],[0,1]] = [[1,2],[3,4]]
    Tensor A({{1.0f, 2.0f}, {3.0f, 4.0f}});
    Tensor I = Tensor::eye(2);
    Tensor C = A.matmul(I);
    EXPECT_NEAR(C(0, 0), 1.0f, 1e-5f);
    EXPECT_NEAR(C(0, 1), 2.0f, 1e-5f);
    EXPECT_NEAR(C(1, 0), 3.0f, 1e-5f);
    EXPECT_NEAR(C(1, 1), 4.0f, 1e-5f);
}

TEST(TensorAD, Tensor_Transpose) {
    Tensor A({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
    Tensor At = A.transpose();
    EXPECT_EQ(At.dim(0), 3);
    EXPECT_EQ(At.dim(1), 2);
    EXPECT_NEAR(At(0, 1), 4.0f, 1e-6f);
    EXPECT_NEAR(At(2, 0), 3.0f, 1e-6f);
}

TEST(TensorAD, Tensor_DotNorm) {
    Tensor a({3.0f, 4.0f});
    EXPECT_NEAR(a.norm(), 5.0f, 1e-5f);
    Tensor b({1.0f, 0.0f});
    EXPECT_NEAR(a.dot(b), 3.0f, 1e-6f);
}

TEST(TensorAD, Tensor_Trace) {
    Tensor A({{1.0f, 2.0f}, {3.0f, 4.0f}});
    EXPECT_NEAR(A.trace(), 5.0f, 1e-6f);
}

// =============================================================================
// TVar — scalar autodiff
// =============================================================================

TEST(TensorAD, TVar_Scalar_Add) {
    ttape_reset();
    TVar a(2.0f), b(3.0f);
    TVar c = a + b;
    c.backward();
    EXPECT_NEAR(a.grad().item(), 1.0f, 1e-5f);
    EXPECT_NEAR(b.grad().item(), 1.0f, 1e-5f);
}

TEST(TensorAD, TVar_Scalar_Mul) {
    // z = x * y, dz/dx = y = 3, dz/dy = x = 2
    ttape_reset();
    TVar x(2.0f), y(3.0f);
    TVar z = x * y;
    z.backward();
    EXPECT_NEAR(x.grad().item(), 3.0f, 1e-5f);
    EXPECT_NEAR(y.grad().item(), 2.0f, 1e-5f);
}

TEST(TensorAD, TVar_Scalar_Exp) {
    // z = exp(x), dz/dx = exp(x) = e^2
    ttape_reset();
    TVar x(2.0f);
    TVar z = texp(x);
    z.backward();
    EXPECT_NEAR(x.grad().item(), std::exp(2.0f), 1e-4f);
}

TEST(TensorAD, TVar_Scalar_Chain) {
    // z = exp(x^2), dz/dx = 2x*exp(x^2)
    float xv = 1.5f;
    ttape_reset();
    TVar x(xv);
    TVar z = texp(x * x);
    z.backward();
    float expected = 2.0f * xv * std::exp(xv * xv);
    EXPECT_NEAR(x.grad().item(), expected, 1e-4f);
}

// =============================================================================
// TVar — vector autodiff
// =============================================================================

TEST(TensorAD, TVar_Vector_Sum) {
    // loss = sum(v), dloss/dv = [1,1,1]
    ttape_reset();
    TVar v(std::vector<float>{1.0f, 2.0f, 3.0f});
    TVar loss = tsum(v);
    loss.backward();
    Tensor g = v.grad();
    EXPECT_EQ(g.size(), 3);
    for (int i = 0; i < 3; ++i)
        EXPECT_NEAR(g(i), 1.0f, 1e-5f);
}

TEST(TensorAD, TVar_Vector_Norm) {
    // loss = ||v||, dloss/dv_i = v_i / ||v||
    float v0 = 3.0f, v1 = 4.0f;
    float nrm = 5.0f;
    ttape_reset();
    TVar v(std::vector<float>{v0, v1});
    TVar loss = tnorm(v);
    loss.backward();
    Tensor g = v.grad();
    EXPECT_NEAR(g(0), v0 / nrm, 1e-4f);
    EXPECT_NEAR(g(1), v1 / nrm, 1e-4f);
}

TEST(TensorAD, TVar_Dot) {
    // loss = dot(a, b), dloss/da = b, dloss/db = a
    ttape_reset();
    TVar a(std::vector<float>{1.0f, 2.0f, 3.0f});
    TVar b(std::vector<float>{4.0f, 5.0f, 6.0f});
    TVar loss = tdot(a, b);
    loss.backward();
    // dloss/da_i = b_i, dloss/db_i = a_i
    EXPECT_NEAR(a.grad()(0), 4.0f, 1e-5f);
    EXPECT_NEAR(a.grad()(2), 6.0f, 1e-5f);
    EXPECT_NEAR(b.grad()(0), 1.0f, 1e-5f);
    EXPECT_NEAR(b.grad()(2), 3.0f, 1e-5f);
}

// =============================================================================
// TVar — matrix autodiff
// =============================================================================

TEST(TensorAD, TVar_Matmul_Grad) {
    // C = A @ B, loss = sum(C)
    // dL/dA = dL/dC @ B^T = ones(m,n) @ B^T
    // dL/dB = A^T @ dL/dC = A^T @ ones(m,n)
    ttape_reset();
    TVar A({{1.0f, 2.0f}, {3.0f, 4.0f}});  // 2x2
    TVar B({{1.0f, 0.0f}, {0.0f, 1.0f}});  // 2x2 identity
    TVar C = tmatmul(A, B);
    TVar loss = tsum(C);
    loss.backward();

    // dL/dA = ones @ I^T = ones
    Tensor dA = A.grad();
    EXPECT_NEAR(dA(0, 0), 1.0f, 1e-5f);
    EXPECT_NEAR(dA(1, 1), 1.0f, 1e-5f);

    // dL/dB = A^T @ ones = column sums of A
    Tensor dB = B.grad();
    EXPECT_NEAR(dB(0, 0), 1.0f + 3.0f, 1e-5f);  // col 0 sum of A
    EXPECT_NEAR(dB(1, 1), 2.0f + 4.0f, 1e-5f);  // col 1 sum of A
}

TEST(TensorAD, TVar_Transpose_Grad) {
    // loss = sum(A^T), dL/dA = (dL/dA^T)^T = ones^T = ones
    ttape_reset();
    TVar A({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});  // 2x3
    TVar At = ttranspose(A);
    TVar loss = tsum(At);
    loss.backward();
    Tensor dA = A.grad();
    EXPECT_EQ(dA.dim(0), 2);
    EXPECT_EQ(dA.dim(1), 3);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
            EXPECT_NEAR(dA(i, j), 1.0f, 1e-5f);
}

TEST(TensorAD, TVar_Frobenius_Grad) {
    // loss = ||A||_F^2 = sum(A^2), dloss/dA = 2*A
    ttape_reset();
    TVar A({{1.0f, 2.0f}, {3.0f, 4.0f}});
    TVar loss = tfrobenius_sq(A);
    loss.backward();
    Tensor dA = A.grad();
    EXPECT_NEAR(dA(0, 0), 2.0f, 1e-5f);
    EXPECT_NEAR(dA(0, 1), 4.0f, 1e-5f);
    EXPECT_NEAR(dA(1, 0), 6.0f, 1e-5f);
    EXPECT_NEAR(dA(1, 1), 8.0f, 1e-5f);
}

TEST(TensorAD, TVar_Trace_Grad) {
    // loss = trace(A), dloss/dA[i,j] = 1 if i==j else 0
    ttape_reset();
    TVar A({{1.0f, 2.0f}, {3.0f, 4.0f}});
    TVar loss = ttrace(A);
    loss.backward();
    Tensor dA = A.grad();
    EXPECT_NEAR(dA(0, 0), 1.0f, 1e-5f);
    EXPECT_NEAR(dA(0, 1), 0.0f, 1e-5f);
    EXPECT_NEAR(dA(1, 0), 0.0f, 1e-5f);
    EXPECT_NEAR(dA(1, 1), 1.0f, 1e-5f);
}

// =============================================================================
// Finite-difference gradient verification for TVar
// =============================================================================

static float fd_scalar(float x0, float eps = 1e-3f) {
    // f(x) = tanh(x) * sin(x) + sqrt(|x|+1)
    auto f = [](float x) {
        return std::tanh(x) * std::sin(x) + std::sqrt(std::abs(x) + 1.0f);
    };
    return (f(x0 + eps) - f(x0 - eps)) / (2.0f * eps);
}

TEST(TensorAD, TVar_FD_ScalarComposed) {
    float xv = 0.8f;
    ttape_reset();
    TVar x(xv);
    TVar y = ttanh(x) * tsin(x) + tsqrt(tabs(x) + TVar(1.0f));
    y.backward();
    float ad_grad  = x.grad().item();
    float fd_grad  = fd_scalar(xv);
    EXPECT_NEAR(ad_grad, fd_grad, 1e-3f);
}

TEST(TensorAD, TVar_FD_VectorElementwise) {
    // f(v) = sum(v^2 + exp(v)), df/dv_i = 2*v_i + exp(v_i)
    std::vector<float> vv = {0.5f, -0.3f, 1.2f};
    ttape_reset();
    TVar v(vv);
    TVar loss = tsum(v * v + texp(v));
    loss.backward();
    Tensor g = v.grad();
    for (int i = 0; i < 3; ++i) {
        float expected = 2.0f * vv[i] + std::exp(vv[i]);
        EXPECT_NEAR(g(i), expected, 1e-4f) << "i=" << i;
    }
}
