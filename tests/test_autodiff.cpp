/// @file test_autodiff.cpp
/// @brief Tests for autodiff.h (tape-based reverse-mode autodiff)

#include <gtest/gtest.h>
#include "numhalide_all.h"

using namespace numhalide;

static const float PI = 3.14159265358979323846f;

// =============================================================================
// 1. Const_NoGrad
// =============================================================================
TEST(Autodiff, Const_NoGrad)
{
    dtape_reset();
    DVar x(3.0f);
    // backward from x itself: dx/dx = 1
    x.backward();
    EXPECT_NEAR(x.val(),  3.0f, 1e-5f);
    EXPECT_NEAR(x.grad(), 1.0f, 1e-5f);
}

// =============================================================================
// 2. Add_Grad
// =============================================================================
TEST(Autodiff, Add_Grad)
{
    dtape_reset();
    DVar x(2.0f), y(5.0f);
    DVar z = x + y;
    z.backward();
    EXPECT_NEAR(x.grad(), 1.0f, 1e-5f);
    EXPECT_NEAR(y.grad(), 1.0f, 1e-5f);
}

// =============================================================================
// 3. Mul_Grad
// =============================================================================
TEST(Autodiff, Mul_Grad)
{
    dtape_reset();
    DVar x(3.0f), y(4.0f);
    DVar z = x * y;
    z.backward();
    EXPECT_NEAR(z.val(),  12.0f, 1e-5f);
    EXPECT_NEAR(x.grad(),  4.0f, 1e-5f);  // dz/dx = y = 4
    EXPECT_NEAR(y.grad(),  3.0f, 1e-5f);  // dz/dy = x = 3
}

// =============================================================================
// 4. Square_Grad
// =============================================================================
TEST(Autodiff, Square_Grad)
{
    dtape_reset();
    DVar x(5.0f);
    DVar z = x * x;
    z.backward();
    EXPECT_NEAR(z.val(),  25.0f, 1e-5f);
    EXPECT_NEAR(x.grad(), 10.0f, 1e-5f);  // dz/dx = 2x = 10
}

// =============================================================================
// 5. Polynomial
// =============================================================================
TEST(Autodiff, Polynomial)
{
    // z = x*x + 3*x + 2  at x=2
    // dz/dx = 2x + 3 = 7
    dtape_reset();
    DVar x(2.0f);
    DVar z = x * x + x * 3.0f + DVar(2.0f);
    z.backward();
    EXPECT_NEAR(z.val(),   12.0f, 1e-5f);  // 4 + 6 + 2
    EXPECT_NEAR(x.grad(),   7.0f, 1e-5f);
}

// =============================================================================
// 6. Exp_Grad
// =============================================================================
TEST(Autodiff, Exp_Grad)
{
    // z = exp(x) at x=0; dz/dx = exp(0) = 1
    dtape_reset();
    DVar x(0.0f);
    DVar z = dexp(x);
    z.backward();
    EXPECT_NEAR(z.val(),  1.0f, 1e-5f);
    EXPECT_NEAR(x.grad(), 1.0f, 1e-5f);
}

// =============================================================================
// 7. Log_Grad
// =============================================================================
TEST(Autodiff, Log_Grad)
{
    // z = log(x) at x=2; dz/dx = 1/2 = 0.5
    dtape_reset();
    DVar x(2.0f);
    DVar z = dlog(x);
    z.backward();
    EXPECT_NEAR(x.grad(), 0.5f, 1e-5f);
}

// =============================================================================
// 8. Sin_Grad
// =============================================================================
TEST(Autodiff, Sin_Grad)
{
    // z = sin(x) at x=0; dz/dx = cos(0) = 1
    dtape_reset();
    DVar x(0.0f);
    DVar z = dsin(x);
    z.backward();
    EXPECT_NEAR(x.grad(), 1.0f, 1e-5f);
}

// =============================================================================
// 9. Chain_ExpSin
// =============================================================================
TEST(Autodiff, Chain_ExpSin)
{
    // z = exp(sin(x)) at x=0
    // dz/dx = cos(x) * exp(sin(x)) = cos(0)*exp(0) = 1
    dtape_reset();
    DVar x(0.0f);
    DVar z = dexp(dsin(x));
    z.backward();
    EXPECT_NEAR(x.grad(), 1.0f, 1e-5f);
}

// =============================================================================
// 10. MultiVar_Independent
// =============================================================================
TEST(Autodiff, MultiVar_Independent)
{
    // f = x + 2*y + 3*z at any values
    // df/dx=1, df/dy=2, df/dz=3
    dtape_reset();
    DVar x(1.0f), y(2.0f), z(3.0f);
    DVar f = x + DVar(2.0f) * y + DVar(3.0f) * z;
    f.backward();
    EXPECT_NEAR(x.grad(), 1.0f, 1e-5f);
    EXPECT_NEAR(y.grad(), 2.0f, 1e-5f);
    EXPECT_NEAR(z.grad(), 3.0f, 1e-5f);
}

// =============================================================================
// 11. Tanh_Grad
// =============================================================================
TEST(Autodiff, Tanh_Grad)
{
    // z = tanh(x) at x=0; dz/dx = 1 - tanh²(0) = 1
    dtape_reset();
    DVar x(0.0f);
    DVar z = dtanh(x);
    z.backward();
    EXPECT_NEAR(z.val(),  0.0f, 1e-5f);
    EXPECT_NEAR(x.grad(), 1.0f, 1e-5f);
}

// =============================================================================
// 12. Sqrt_Grad
// =============================================================================
TEST(Autodiff, Sqrt_Grad)
{
    // z = sqrt(x) at x=4; dz/dx = 1/(2*sqrt(4)) = 1/4 = 0.25
    dtape_reset();
    DVar x(4.0f);
    DVar z = dsqrt(x);
    z.backward();
    EXPECT_NEAR(z.val(),  2.0f,  1e-5f);
    EXPECT_NEAR(x.grad(), 0.25f, 1e-5f);
}

// =============================================================================
// AutodiffFiniteDiff — gradient checks via central finite differences
//
// For each test:
//   analytic gradient: DVar + backward()
//   numerical gradient: (f(x+eps) - f(x-eps)) / (2*eps), eps = 1e-3
// =============================================================================

static const float FD_EPS = 1e-3f;

// 1. FD_Square: f(x) = x*x at x=3; df/dx = 2x = 6
TEST(AutodiffFiniteDiff, FD_Square)
{
    const float x0 = 3.0f;

    dtape_reset();
    DVar x(x0);
    DVar z = x * x;
    z.backward();
    float analytic = x.grad();

    auto f = [](float t) { return t * t; };
    float fd = (f(x0 + FD_EPS) - f(x0 - FD_EPS)) / (2.0f * FD_EPS);

    EXPECT_NEAR(analytic, fd,    1e-3f);
    EXPECT_NEAR(analytic, 6.0f, 1e-3f);
}

// 2. FD_Cube: f(x) = x*x*x at x=2; df/dx = 3x^2 = 12
TEST(AutodiffFiniteDiff, FD_Cube)
{
    const float x0 = 2.0f;

    dtape_reset();
    DVar x(x0);
    DVar z = x * x * x;
    z.backward();
    float analytic = x.grad();

    auto f = [](float t) { return t * t * t; };
    float fd = (f(x0 + FD_EPS) - f(x0 - FD_EPS)) / (2.0f * FD_EPS);

    EXPECT_NEAR(analytic, fd,     1e-3f);
    EXPECT_NEAR(analytic, 12.0f, 1e-3f);
}

// 3. FD_ExpX: f(x) = exp(x) at x=1; df/dx = exp(1) = e
TEST(AutodiffFiniteDiff, FD_ExpX)
{
    const float x0 = 1.0f;

    dtape_reset();
    DVar x(x0);
    DVar z = dexp(x);
    z.backward();
    float analytic = x.grad();

    auto f = [](float t) { return std::exp(t); };
    float fd = (f(x0 + FD_EPS) - f(x0 - FD_EPS)) / (2.0f * FD_EPS);

    EXPECT_NEAR(analytic, fd,                        1e-3f);
    EXPECT_NEAR(analytic, std::exp(1.0f),            1e-3f);
}

// 4. FD_LogX: f(x) = log(x) at x=2; df/dx = 1/2 = 0.5
TEST(AutodiffFiniteDiff, FD_LogX)
{
    const float x0 = 2.0f;

    dtape_reset();
    DVar x(x0);
    DVar z = dlog(x);
    z.backward();
    float analytic = x.grad();

    auto f = [](float t) { return std::log(t); };
    float fd = (f(x0 + FD_EPS) - f(x0 - FD_EPS)) / (2.0f * FD_EPS);

    EXPECT_NEAR(analytic, fd,    1e-3f);
    EXPECT_NEAR(analytic, 0.5f, 1e-3f);
}

// 5. FD_SinX: f(x) = sin(x) at x=0.5; df/dx = cos(0.5)
TEST(AutodiffFiniteDiff, FD_SinX)
{
    const float x0 = 0.5f;

    dtape_reset();
    DVar x(x0);
    DVar z = dsin(x);
    z.backward();
    float analytic = x.grad();

    auto f = [](float t) { return std::sin(t); };
    float fd = (f(x0 + FD_EPS) - f(x0 - FD_EPS)) / (2.0f * FD_EPS);

    EXPECT_NEAR(analytic, fd,                    1e-3f);
    EXPECT_NEAR(analytic, std::cos(x0),          1e-3f);
}

// 6. FD_TanhX: f(x) = tanh(x) at x=1; df/dx = 1 - tanh^2(1)
TEST(AutodiffFiniteDiff, FD_TanhX)
{
    const float x0 = 1.0f;

    dtape_reset();
    DVar x(x0);
    DVar z = dtanh(x);
    z.backward();
    float analytic = x.grad();

    auto f = [](float t) { return std::tanh(t); };
    float fd = (f(x0 + FD_EPS) - f(x0 - FD_EPS)) / (2.0f * FD_EPS);

    float expected = 1.0f - std::tanh(x0) * std::tanh(x0);
    EXPECT_NEAR(analytic, fd,       1e-3f);
    EXPECT_NEAR(analytic, expected, 1e-3f);
}

// 7. FD_Chain: f(x) = exp(x*x) at x=1; df/dx = 2x*exp(x^2) = 2*exp(1)
TEST(AutodiffFiniteDiff, FD_Chain)
{
    const float x0 = 1.0f;

    dtape_reset();
    DVar x(x0);
    DVar z = dexp(x * x);
    z.backward();
    float analytic = x.grad();

    auto f = [](float t) { return std::exp(t * t); };
    float fd = (f(x0 + FD_EPS) - f(x0 - FD_EPS)) / (2.0f * FD_EPS);

    float expected = 2.0f * std::exp(1.0f);
    EXPECT_NEAR(analytic, fd,       1e-3f);
    EXPECT_NEAR(analytic, expected, 1e-3f);
}

// 8. FD_MultiVar: f(x,y) = x*x + x*y at x=2, y=3
//    df/dx = 2x + y = 7;  df/dy = x = 2
TEST(AutodiffFiniteDiff, FD_MultiVar)
{
    const float x0 = 2.0f, y0 = 3.0f;

    // --- analytic ---
    dtape_reset();
    DVar x(x0), y(y0);
    DVar z = x * x + x * y;
    z.backward();
    float analytic_dx = x.grad();
    float analytic_dy = y.grad();

    // --- finite diff for df/dx (perturb x, hold y fixed) ---
    auto fx = [&](float tx) { return tx * tx + tx * y0; };
    float fd_dx = (fx(x0 + FD_EPS) - fx(x0 - FD_EPS)) / (2.0f * FD_EPS);

    // --- finite diff for df/dy (perturb y, hold x fixed) ---
    auto fy = [&](float ty) { return x0 * x0 + x0 * ty; };
    float fd_dy = (fy(y0 + FD_EPS) - fy(y0 - FD_EPS)) / (2.0f * FD_EPS);

    EXPECT_NEAR(analytic_dx, fd_dx, 1e-3f);
    EXPECT_NEAR(analytic_dx, 7.0f,  1e-3f);  // 2*2 + 3 = 7

    EXPECT_NEAR(analytic_dy, fd_dy, 1e-3f);
    EXPECT_NEAR(analytic_dy, 2.0f,  1e-3f);  // x = 2
}
