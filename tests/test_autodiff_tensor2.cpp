/// @file test_autodiff_tensor2.cpp
/// @brief Extended tests: axis-sum ND, least-squares, ADMM, proximal, rank-3/4 tensors

#include <gtest/gtest.h>
#include "numhalide_all.h"
#include <cmath>

using namespace numhalide;

// =============================================================================
// Axis-wise tsum (ND)
// =============================================================================

TEST(TensorAD2, AxisSum_2D_Axis0) {
    // A: 3x2, sum axis 0 → {2}; grad back = ones(3,2)
    ttape_reset();
    TVar A(Tensor({{1.f,2.f},{3.f,4.f},{5.f,6.f}}));
    TVar loss = tsum(tsum(A, 0));
    loss.backward();
    Tensor g = A.grad();
    EXPECT_EQ(g.dim(0), 3); EXPECT_EQ(g.dim(1), 2);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 2; ++j)
            EXPECT_NEAR(g(i,j), 1.f, 1e-5f);
}

TEST(TensorAD2, AxisSum_2D_Axis1) {
    // B: 2x3, sum axis 1 → {2}; grad back = ones(2,3)
    ttape_reset();
    TVar B(Tensor({{1.f,2.f,3.f},{4.f,5.f,6.f}}));
    TVar loss = tsum(tsum(B, 1));
    loss.backward();
    Tensor g = B.grad();
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
            EXPECT_NEAR(g(i,j), 1.f, 1e-5f);
}

TEST(TensorAD2, AxisSum_3D_Axis0) {
    // Shape {2,3,4}, sum axis 0 → {3,4}; all grads = 1
    ttape_reset();
    std::vector<float> d(24); for (int i=0;i<24;++i) d[i]=(float)(i+1)*0.1f;
    TVar T(Tensor({2,3,4}, d));
    TVar loss = tsum(tsum(T, 0));
    loss.backward();
    Tensor g = T.grad();
    EXPECT_EQ(g.size(), 24);
    for (int k=0; k<24; ++k) EXPECT_NEAR(g.flat(k), 1.f, 1e-5f);
}

TEST(TensorAD2, AxisSum_3D_Axis1) {
    // Shape {2,4,3}, sum axis 1 → {2,3}; all grads = 1
    ttape_reset();
    std::vector<float> d(24, 1.f);
    TVar T(Tensor({2,4,3}, d));
    TVar s = tsum(T, 1);
    EXPECT_EQ(s.val().dim(0), 2); EXPECT_EQ(s.val().dim(1), 3);
    TVar loss = tsum(s);
    loss.backward();
    Tensor g = T.grad();
    for (int k=0; k<24; ++k) EXPECT_NEAR(g.flat(k), 1.f, 1e-5f);
}

TEST(TensorAD2, AxisSum_3D_Axis2) {
    // Shape {2,3,4}, sum axis 2 → {2,3}; all grads = 1
    ttape_reset();
    std::vector<float> d(24, 1.f);
    TVar T(Tensor({2,3,4}, d));
    TVar s = tsum(T, 2);
    EXPECT_EQ(s.val().rank(), 2);
    EXPECT_EQ(s.val().dim(0), 2); EXPECT_EQ(s.val().dim(1), 3);
    tsum(s).backward();
    Tensor g = T.grad();
    for (int k=0; k<24; ++k) EXPECT_NEAR(g.flat(k), 1.f, 1e-5f);
}

// =============================================================================
// Least squares: f(x) = ||Ax - b||^2_2,  df/dx = 2 A^T (Ax - b)
// =============================================================================

TEST(TensorAD2, LeastSquares_Gradient) {
    Tensor Av({{2.f,0.f},{0.f,3.f},{1.f,1.f}});  // 3x2
    Tensor xv({2,1}, {1.f,2.f});                   // 2x1
    Tensor bv({3,1}, {2.f,6.f,3.f});               // 3x1

    // Analytical: Ax-b = {0,0,0} → grad = 0
    // (with xv = [1,2]: A[0]*1+A[1]*2 = 2, 6, 3 = b → residual = 0)
    ttape_reset();
    TVar A(Av), x(xv), b(bv);
    TVar res  = tmatmul(A, x) - b;
    TVar loss = tsum(res * res);
    loss.backward();
    Tensor dx = x.grad();
    EXPECT_NEAR(dx(0,0), 0.f, 1e-4f);
    EXPECT_NEAR(dx(1,0), 0.f, 1e-4f);
}

TEST(TensorAD2, LeastSquares_Nonzero_Residual) {
    // x not at optimum → nonzero gradient
    Tensor Av({{1.f,0.f},{0.f,1.f},{1.f,1.f}});  // 3x2
    Tensor bv({3,1}, {0.f,0.f,0.f});
    Tensor xv({2,1}, {1.f,1.f});

    // residual = Ax - b = {1,1,2}
    // grad = 2*A^T*{1,1,2} = 2*{1*1+0*1+1*2, 0*1+1*1+1*2} = 2*{3,3} = {6,6}
    ttape_reset();
    TVar A(Av), x(xv), b(bv);
    tsum((tmatmul(A,x) - b) * (tmatmul(A,x) - b)).backward();
    Tensor dx = x.grad();
    EXPECT_NEAR(dx(0,0), 6.f, 1e-4f);
    EXPECT_NEAR(dx(1,0), 6.f, 1e-4f);
}

TEST(TensorAD2, LeastSquares_FD_Verify) {
    // Finite-difference gradient check for least squares
    Tensor Av({{1.f,2.f},{3.f,1.f},{0.f,1.f}});  // 3x2
    Tensor bv({3,1}, {1.f,2.f,1.f});
    std::vector<float> x0 = {1.f, 1.f};
    float eps = 1e-3f;

    auto scalar_loss = [&](std::vector<float> xd) -> float {
        float s = 0;
        for (int i=0;i<3;++i) {
            float r = 0;
            for (int j=0;j<2;++j) r += Av(i,j)*xd[j];
            r -= bv.flat(i);
            s += r*r;
        }
        return s;
    };

    ttape_reset();
    TVar A(Av), x(Tensor({2,1}, x0)), b(bv);
    tsum((tmatmul(A,x)-b)*(tmatmul(A,x)-b)).backward();
    Tensor dx = x.grad();

    for (int comp=0; comp<2; ++comp) {
        std::vector<float> xp=x0, xm=x0;
        xp[comp]+=eps; xm[comp]-=eps;
        float fd = (scalar_loss(xp) - scalar_loss(xm)) / (2.f*eps);
        EXPECT_NEAR(dx.flat(comp), fd, 1e-3f) << "comp=" << comp;
    }
}

// =============================================================================
// Ridge regression: f(x) = ||Ax - b||^2 + lambda*||x||^2
//   df/dx = 2*A^T*(Ax-b) + 2*lambda*x
// =============================================================================

TEST(TensorAD2, Ridge_Gradient) {
    float lam = 0.5f;
    Tensor Av({{1.f,0.f},{0.f,2.f}});
    Tensor bv({2,1}, {1.f,1.f});
    Tensor xv({2,1}, {2.f,1.f});
    // Ax = {2,2}, residual = {1,1}
    // 2*A^T*{1,1} = 2*{1,2} = {2,4}
    // 2*lam*x = 1*{2,1} = {2,1}
    // total = {4,5}
    ttape_reset();
    TVar A(Av), x(xv), b(bv);
    TVar res  = tmatmul(A,x) - b;
    TVar loss = tsum(res*res) + TVar(Tensor(lam)) * tsum(x*x);
    loss.backward();
    Tensor dx = x.grad();
    EXPECT_NEAR(dx(0,0), 4.f, 1e-4f);
    EXPECT_NEAR(dx(1,0), 5.f, 1e-4f);
}

// =============================================================================
// ADMM: gradient of augmented Lagrangian w.r.t. x
//   L(x) = ||Ax-b||^2 + y^T(x-z) + (rho/2)||x-z||^2
//   dL/dx = 2*A^T*(Ax-b) + y + rho*(x-z)
// =============================================================================

TEST(TensorAD2, ADMM_AugLag_Gradient) {
    float rho = 1.f;
    Tensor Av({{1.f,0.f},{0.f,1.f}});  // 2x2 identity
    Tensor bv({2,1}, {1.f,2.f});
    Tensor xv({2,1}, {3.f,4.f});
    Tensor zv({2,1}, {2.f,3.f});
    Tensor yv(std::vector<float>{0.5f,-0.5f});  // 1D dual variable

    // Ax-b = {2,2}  →  2*A^T*{2,2} = {4,4}
    // y^T(x-z): dy/dx = y = {0.5,-0.5}
    // rho*(x-z) = {1,1}
    // Total: {4,4}+{0.5,-0.5}+{1,1} = {5.5, 4.5}

    ttape_reset();
    TVar A(Av), x(xv), b(bv), z(zv), y(yv);
    TVar diff    = x - z;                    // 2x1
    TVar res     = tmatmul(A,x) - b;
    TVar fid     = tsum(res*res);
    TVar dual    = tdot(y, treshape(diff,{2}));
    TVar proxpen = TVar(Tensor(rho/2.f)) * tsum(diff*diff);
    TVar L       = fid + dual + proxpen;
    L.backward();

    Tensor dx = x.grad();
    EXPECT_NEAR(dx(0,0), 5.5f, 1e-4f);
    EXPECT_NEAR(dx(1,0), 4.5f, 1e-4f);
}

TEST(TensorAD2, ADMM_ConsensusUpdate) {
    // x-update in consensus ADMM: minimize f(x) + (rho/2)||x - (z - u)||^2
    // gradient w.r.t. x = df/dx + rho*(x - v)  where v = z - u
    // for f(x) = ||x||^2: df/dx = 2x, gradient = 2x + rho*(x-v) = (2+rho)*x - rho*v
    float rho = 2.f;
    Tensor zv(std::vector<float>{1.f,2.f,3.f});
    Tensor uv(std::vector<float>{0.5f,0.5f,0.5f});
    Tensor xv(std::vector<float>{1.f,1.f,1.f});

    // v = z - u = {0.5, 1.5, 2.5}
    // gradient = (2+2)*x - 2*v = 4*{1,1,1} - 2*{0.5,1.5,2.5} = {3,1,-1}
    ttape_reset();
    TVar x(xv), z(zv), u(uv);
    TVar v    = z - u;
    TVar f    = tsum(x*x);
    TVar pen  = TVar(Tensor(rho/2.f)) * tsum((x-v)*(x-v));
    TVar loss = f + pen;
    loss.backward();

    Tensor dx = x.grad();
    EXPECT_NEAR(dx(0), 3.f, 1e-4f);
    EXPECT_NEAR(dx(1), 1.f, 1e-4f);
    EXPECT_NEAR(dx(2),-1.f, 1e-4f);
}

// =============================================================================
// Proximal operators
// =============================================================================

TEST(TensorAD2, Proximal_L2_OptimumIsZeroGrad) {
    // prox_{lam/2 ||.||^2}(v) = v/(1+lam)  — optimum of: ||x-v||^2 + lam*||x||^2
    // gradient at x* = v/(1+lam) is zero
    float lam = 2.f;
    Tensor vv(std::vector<float>{3.f,-6.f,1.5f});
    // x* = {1,-2,0.5}
    ttape_reset();
    Tensor _tx_opt(std::vector<float>{1.f,-2.f,0.5f});
    TVar x(_tx_opt);
    TVar v(vv);
    TVar obj = tsum((x-v)*(x-v)) + TVar(Tensor(lam)) * tsum(x*x);
    obj.backward();
    Tensor dx = x.grad();
    for (int i=0; i<3; ++i) EXPECT_NEAR(dx(i), 0.f, 1e-4f) << "i=" << i;
}

TEST(TensorAD2, Proximal_SoftThresh_SmoothApprox) {
    // Pseudo-Huber smooth L1: f(x) = sum( sqrt(x^2 + eps) )
    // df/dx_i = x_i / sqrt(x_i^2 + eps)
    float eps = 0.01f;
    std::vector<float> xd = {1.5f, -0.5f, 0.1f, -1.2f};
    ttape_reset();
    Tensor _txh(xd);
    TVar x(_txh);
    Tensor _eps_t(eps);
    tsum(tsqrt(x*x + TVar(_eps_t))).backward();
    Tensor dx = x.grad();
    for (int i=0; i<(int)xd.size(); ++i) {
        float xi = xd[i];
        EXPECT_NEAR(dx(i), xi / std::sqrt(xi*xi + eps), 1e-4f) << "i=" << i;
    }
}

TEST(TensorAD2, Proximal_LogBarrier_Gradient) {
    // f(x) = -sum(log(x))  [log barrier for x > 0]
    // df/dx_i = -1/x_i
    std::vector<float> xd = {2.f, 0.5f, 1.f, 4.f};
    ttape_reset();
    Tensor _txl(xd);
    TVar x(_txl);
    TVar loss = TVar(Tensor(-1.f)) * tsum(tlog(x));
    loss.backward();
    Tensor dx = x.grad();
    for (int i=0; i<(int)xd.size(); ++i)
        EXPECT_NEAR(dx(i), -1.f/xd[i], 1e-4f) << "i=" << i;
}

// =============================================================================
// Rank-3 tensor operations
// =============================================================================

TEST(TensorAD2, Rank3_Elementwise_Tanh) {
    // shape {2,3,2}, loss = sum(tanh(T)), d/dT = 1-tanh^2
    std::vector<float> d = {0.f,1.f, 0.5f,-0.5f, 1.f,0.f,
                            -1.f,0.5f, 0.f,0.2f, 0.3f,-0.3f};
    ttape_reset();
    TVar T(Tensor({2,3,2}, d));
    tsum(ttanh(T)).backward();
    Tensor g = T.grad();
    EXPECT_EQ(g.size(), 12);
    for (int i=0; i<12; ++i) {
        float th = std::tanh(d[i]);
        EXPECT_NEAR(g.flat(i), 1.f-th*th, 1e-4f) << "i=" << i;
    }
}

TEST(TensorAD2, Rank3_ScaledSum) {
    // loss = 3 * sum(T), d/dT = 3 everywhere
    ttape_reset();
    std::vector<float> d(24, 1.f);
    TVar T(Tensor({2,3,4}, d));
    TVar loss = TVar(Tensor(3.f)) * tsum(T);
    loss.backward();
    Tensor g = T.grad();
    for (int k=0; k<24; ++k) EXPECT_NEAR(g.flat(k), 3.f, 1e-5f);
}

TEST(TensorAD2, BatchMatmul_Identity_B) {
    // A: {3,2,4}, B: {3,4,2} identity-like
    // loss = sum(A@B) → dA[b,i,k] = sum_j B[b,k,j], dB[b,k,j] = sum_i A[b,i,k]
    ttape_reset();
    int BS=3, M=2, K=4, N=2;
    std::vector<float> ad(BS*M*K), bd(BS*K*N, 0.f);
    for (int i=0;i<BS*M*K;++i) ad[i]=(float)(i+1)*0.1f;
    // B[b, k, j] = 1 if k%N == j else 0  (first N rows of identity repeated)
    for (int b=0;b<BS;++b) for (int k=0;k<K;++k) for (int j=0;j<N;++j)
        bd[b*K*N + k*N + j] = (k%N == j) ? 1.f : 0.f;

    TVar A(Tensor({BS,M,K}, ad)), B(Tensor({BS,K,N}, bd));
    TVar C = tbatch_matmul(A, B);
    TVar loss = tsum(C);
    loss.backward();

    // dA[b,i,k] = sum_j B[b,k,j] — check a few
    Tensor dA = A.grad();
    for (int b=0;b<BS;++b) for (int i=0;i<M;++i)
        for (int k=0;k<K;++k) {
            float expected = 0;
            for (int j=0;j<N;++j) expected += bd[b*K*N+k*N+j];
            EXPECT_NEAR(dA(b,i,k), expected, 1e-4f)
                << "b="<<b<<" i="<<i<<" k="<<k;
        }
}

TEST(TensorAD2, BatchMatmul_FD_Verify) {
    // Cross-check gradients of batch matmul with finite differences
    int BS=2, M=2, K=3, N=2;
    std::vector<float> ad(BS*M*K), bd(BS*K*N);
    for (int i=0;i<(int)ad.size();++i) ad[i]=(float)(i%5-2)*0.3f;
    for (int i=0;i<(int)bd.size();++i) bd[i]=(float)(i%4-1)*0.2f;
    float eps = 1e-3f;

    auto scalar_loss = [&](std::vector<float> A_data, std::vector<float> B_data) {
        Tensor Av({BS,M,K}, A_data), Bv({BS,K,N}, B_data);
        Tensor C = Av.batch_matmul(Bv);
        float s=0; for (int i=0;i<C.size();++i) s+=C.flat(i);
        return s;
    };

    // AD gradients
    ttape_reset();
    TVar A(Tensor({BS,M,K},ad)), B(Tensor({BS,K,N},bd));
    tsum(tbatch_matmul(A,B)).backward();
    Tensor dA = A.grad(), dB = B.grad();

    // FD check for a few elements of A
    for (int fi=0; fi<(int)ad.size(); fi+=3) {
        std::vector<float> ap=ad, am=ad;
        ap[fi]+=eps; am[fi]-=eps;
        float fd = (scalar_loss(ap,bd) - scalar_loss(am,bd)) / (2.f*eps);
        EXPECT_NEAR(dA.flat(fi), fd, 1e-3f) << "A fi=" << fi;
    }
}

// =============================================================================
// Rank-4 tensor operations
// =============================================================================

TEST(TensorAD2, Rank4_Elementwise_Sigmoid) {
    // shape {2,3,4,2}, sigmoid, loss = sum
    // d/dT = sigmoid(T) * (1 - sigmoid(T))
    std::vector<float> d(48);
    for (int i=0;i<48;++i) d[i]=(float)(i%5-2)*0.4f;
    ttape_reset();
    TVar T(Tensor({2,3,4,2}, d));
    tsum(tsigmoid(T)).backward();
    Tensor g = T.grad();
    EXPECT_EQ(g.size(), 48);
    for (int i=0;i<48;++i) {
        float s = 1.f/(1.f+std::exp(-d[i]));
        EXPECT_NEAR(g.flat(i), s*(1.f-s), 1e-4f) << "i=" << i;
    }
}

TEST(TensorAD2, Rank4_AxisSum_Axis2) {
    // shape {2,3,4,2}, sum axis 2 → {2,3,2}; all grads = 1
    ttape_reset();
    std::vector<float> d(48, 1.f);
    TVar T(Tensor({2,3,4,2}, d));
    TVar s = tsum(T, 2);
    EXPECT_EQ(s.val().rank(), 3);
    EXPECT_EQ(s.val().dim(0), 2); EXPECT_EQ(s.val().dim(1), 3); EXPECT_EQ(s.val().dim(2), 2);
    tsum(s).backward();
    Tensor g = T.grad();
    for (int k=0;k<48;++k) EXPECT_NEAR(g.flat(k), 1.f, 1e-5f);
}

TEST(TensorAD2, Rank4_AxisSum_Axis3) {
    // shape {2,3,4,5}, sum axis 3 → {2,3,4}; all grads = 1
    ttape_reset();
    std::vector<float> d(120, 1.f);
    TVar T(Tensor({2,3,4,5}, d));
    TVar s = tsum(T, 3);
    EXPECT_EQ(s.val().rank(), 3);
    EXPECT_EQ(s.val().dim(2), 4);
    tsum(s).backward();
    Tensor g = T.grad();
    for (int k=0;k<120;++k) EXPECT_NEAR(g.flat(k), 1.f, 1e-5f);
}

TEST(TensorAD2, Rank4_Reshape_To_2D_Matmul) {
    // T: {2,2,2,2} reshape→ {2,8}; W: {8,3}; loss = sum(T_flat @ W)
    // dL/dW = T_flat^T @ ones(2,3), shape {8,3}
    // dL/dT = (dL/dTflat) reshaped back to {2,2,2,2}
    //       = (ones(2,3) @ W^T) reshaped; each row of T_flat gets sum_j W[k,j]
    std::vector<float> td(16), wd(24);
    for (int i=0;i<16;++i) td[i]=(float)(i+1)*0.1f;
    for (int i=0;i<24;++i) wd[i]=(float)(i%3+1)*0.5f;  // W[k,j] = {0.5,1,1.5,...}

    ttape_reset();
    TVar T(Tensor({2,2,2,2}, td));
    TVar W(Tensor({8,3}, wd));
    TVar loss = tsum(tmatmul(treshape(T,{2,8}), W));
    loss.backward();

    // dL/dTflat[i,k] = sum_j W[k,j]  (upstream = ones(2,3))
    Tensor dT = T.grad();
    EXPECT_EQ(dT.rank(), 4);
    for (int i=0;i<2;++i) for (int k=0;k<8;++k) {
        float sum_wkj = 0; for (int j=0;j<3;++j) sum_wkj += wd[k*3+j];
        EXPECT_NEAR(dT.flat(i*8+k), sum_wkj, 1e-4f)
            << "i=" << i << " k=" << k;
    }
}

// =============================================================================
// Cross-entropy loss: loss = log(sum(exp(z))) - z[y]
//   dloss/dz_k = softmax(z)_k - 1_{k==y}
// =============================================================================

TEST(TensorAD2, CrossEntropy_Gradient) {
    std::vector<float> zd = {1.f, 2.f, 0.5f};
    float ez0=std::exp(1.f), ez1=std::exp(2.f), ez2=std::exp(0.5f);
    float S = ez0+ez1+ez2;
    float sm0=ez0/S, sm1=ez1/S, sm2=ez2/S;

    ttape_reset();
    Tensor _tz(zd);
    TVar z(_tz);
    Tensor _ty({0.f, 1.f, 0.f});
    TVar y_onehot(_ty);
    TVar loss = tlog(tsum(texp(z))) - tdot(y_onehot, z);
    loss.backward();

    Tensor dz = z.grad();
    EXPECT_NEAR(dz(0), sm0,     1e-4f);
    EXPECT_NEAR(dz(1), sm1-1.f, 1e-4f);
    EXPECT_NEAR(dz(2), sm2,     1e-4f);
}

TEST(TensorAD2, CrossEntropy_BatchGradient) {
    // Batch of 2 samples, 3 classes each — sum losses
    std::vector<float> z0 = {1.f,2.f,0.f}, z1 = {0.f,1.f,3.f};
    // Sample 0: true class 1; Sample 1: true class 2
    // Concatenated: z=[z0;z1] as a 2-element sum
    auto softmax = [](std::vector<float> v) {
        float s=0; for (auto x:v) s+=std::exp(x);
        std::vector<float> r; for (auto x:v) r.push_back(std::exp(x)/s);
        return r;
    };
    auto sm0 = softmax(z0), sm1s = softmax(z1);

    ttape_reset();
    Tensor _tza(z0), _tzb(z1);
    TVar za(_tza), zb(_tzb);
    Tensor _tya({0.f,1.f,0.f}), _tyb({0.f,0.f,1.f});
    TVar ya(_tya), yb(_tyb);
    TVar la = tlog(tsum(texp(za))) - tdot(ya, za);
    TVar lb = tlog(tsum(texp(zb))) - tdot(yb, zb);
    TVar loss = la + lb;
    loss.backward();

    Tensor da = za.grad(), db = zb.grad();
    EXPECT_NEAR(da(0), sm0[0],     1e-4f);
    EXPECT_NEAR(da(1), sm0[1]-1.f, 1e-4f);
    EXPECT_NEAR(da(2), sm0[2],     1e-4f);
    EXPECT_NEAR(db(0), sm1s[0],     1e-4f);
    EXPECT_NEAR(db(1), sm1s[1],     1e-4f);
    EXPECT_NEAR(db(2), sm1s[2]-1.f, 1e-4f);
}
