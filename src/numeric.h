/// @file numeric.h
/// @brief Numeric and special math functions
///
/// Provides: logaddexp, logaddexp2, copysign, signbit, trapz_1d,
///           i0 (modified Bessel), correlate1d

#pragma once

#include "common.h"
#include "shape.h"

NS_NUM_HALIDE_BEGIN

// -----------------------------------------------------------------------------
// Numerically Stable Log-Sum-Exp
// -----------------------------------------------------------------------------

/// @brief Compute log(exp(a) + exp(b)) element-wise, numerically stable
inline
Halide::Func logaddexp(Halide::Func a, Halide::Func b, const shape_t& shape,
    std::string const& name = "logaddexp")
{
    Halide::Func ret(name);
    std::vector<Halide::Var> vars;
    for (int i = 0; i < shape.rank; ++i) vars.push_back(Halide::Var());
    Halide::Expr va = a(vars), vb = b(vars);
    // log(exp(a) + exp(b)) = max(a,b) + log(1 + exp(-|a-b|))
    ret(vars) = Halide::max(va, vb) + Halide::log(1.0f + Halide::exp(-Halide::abs(va - vb)));
    return ret;
}

/// @brief Compute log2(2^a + 2^b) element-wise, numerically stable
inline
Halide::Func logaddexp2(Halide::Func a, Halide::Func b, const shape_t& shape,
    std::string const& name = "logaddexp2")
{
    Halide::Func ret(name);
    std::vector<Halide::Var> vars;
    for (int i = 0; i < shape.rank; ++i) vars.push_back(Halide::Var());
    Halide::Expr va = a(vars), vb = b(vars);
    // log2(2^a + 2^b) = max(a,b) + log2(1 + 2^(-|a-b|))
    // ln(2) is taken of a constant made in the input's own float type —
    // for f32 this folds to the same value as the old log(2.0f); for f64
    // it is the full-precision ln(2) instead of a widened f32 constant.
    Halide::Type t = a.types()[0];
    if (!t.is_float()) t = Halide::Float(32);
    Halide::Expr ln2 = Halide::log(Halide::Internal::make_const(t, 2.0));
    ret(vars) = Halide::max(va, vb) +
        Halide::log(1.0f + Halide::pow(2.0f, -Halide::abs(va - vb))) / ln2;
    return ret;
}

// -----------------------------------------------------------------------------
// Sign Operations
// -----------------------------------------------------------------------------

/// @brief Copy sign of sign_src into magnitude element-wise
inline
Halide::Func copysign(Halide::Func magnitude, Halide::Func sign_src, const shape_t& shape,
    std::string const& name = "copysign")
{
    Halide::Func ret(name);
    std::vector<Halide::Var> vars;
    for (int i = 0; i < shape.rank; ++i) vars.push_back(Halide::Var());
    ret(vars) = Halide::select(sign_src(vars) >= 0.0f,
        Halide::abs(magnitude(vars)), -Halide::abs(magnitude(vars)));
    return ret;
}

/// @brief Return 1 (uint8) where f is negative, 0 elsewhere
inline
Halide::Func signbit(Halide::Func f, const shape_t& shape,
    std::string const& name = "signbit")
{
    Halide::Func ret(name);
    std::vector<Halide::Var> vars;
    for (int i = 0; i < shape.rank; ++i) vars.push_back(Halide::Var());
    ret(vars) = Halide::cast<uint8_t>(f(vars) < 0.0f);
    return ret;
}

// -----------------------------------------------------------------------------
// Trapezoidal Integration
// -----------------------------------------------------------------------------

/// @brief 1D trapezoidal integration with uniform spacing dx
/// @param f Input Func (1D, n elements)
/// @param n Number of elements
/// @param dx Spacing between samples
/// @return 0D (scalar) Func with integral value
inline
Halide::Func trapz_1d(Halide::Func f, int n, float dx = 1.0f,
    std::string const& name = "trapz")
{
    nh_require(n >= 2, "trapz_1d: need at least 2 points");
    // Accumulate in the input's own float type (f64 stays f64); integer
    // inputs keep the historical f32 accumulation. The half-spacing is the
    // host-float product dx * 0.5f cast into that type — value-identical
    // for f32; dx itself is f32-quantized by the float API parameter.
    Halide::Type t = f.types()[0];
    if (!t.is_float()) t = Halide::Float(32);
    Halide::Func ret(name);
    Halide::RDom r(0, n - 1, "r_trapz");
    ret() = Halide::cast(t, 0);
    ret() += (f(r) + f(r + 1)) * Halide::Internal::make_const(t, static_cast<double>(dx) * 0.5);
    return ret;
}

/// @brief 1D trapezoidal integration with non-uniform x coordinates
/// @param f  y-values (1D, n elements)
/// @param x  x-coordinates (1D, n elements, must be sorted)
/// @param n  Number of elements
inline
Halide::Func trapz_1d(Halide::Func f, Halide::Func x, int n,
    std::string const& name = "trapz_xu")
{
    nh_require(n >= 2, "trapz_1d: need at least 2 points");
    // Accumulate in the input's own float type (f64 stays f64); the exact
    // 0.5 literal widens exactly, so only the seed needs typing.
    Halide::Type t = f.types()[0];
    if (!t.is_float()) t = Halide::Float(32);
    Halide::Func ret(name);
    Halide::RDom r(0, n - 1, "r_trapz_xu");
    ret() = Halide::cast(t, 0);
    ret() += (f(r) + f(r + 1)) * (x(r + 1) - x(r)) * 0.5f;
    return ret;
}

// -----------------------------------------------------------------------------
// Modified Bessel Function of the First Kind, Order 0
// -----------------------------------------------------------------------------

/// @brief I0(x) as a scalar Expr, Abramowitz & Stegun polynomial approximation
/// @note Accurate to ~1e-7 for all x
///
/// Expr-level form so window code (kaiser) can use the A&S approximation
/// without the shape_t Func wrapper; the Func form i0() below delegates
/// here. Computed in the type of v — the float-literal coefficients are
/// constants, so Halide's match_types promotes them to the operand type
/// (e.g. Float(64) input stays Float(64) throughout).
inline
Halide::Expr i0_expr(Halide::Expr v)
{
    Halide::Expr x = Halide::abs(v);

    // For |x| <= 3.75: Horner form in t2 = (x/3.75)^2
    Halide::Expr t2 = (x / 3.75f) * (x / 3.75f);
    Halide::Expr small = 1.0f + t2 * (3.5156229f + t2 * (3.0899424f + t2 * (1.2067492f
        + t2 * (0.2659732f + t2 * (0.0360768f + t2 * 0.0045813f)))));

    // For |x| > 3.75: exp(x)/sqrt(x) * poly(3.75/x)
    Halide::Expr y = 3.75f / x;
    Halide::Expr poly = 0.39894228f + y * (0.01328592f + y * (0.00225319f
        + y * (-0.00157565f + y * (0.00916281f + y * (-0.02057706f
        + y * (0.02635537f + y * (-0.01647633f + y * 0.00392377f)))))));
    Halide::Expr large = (Halide::exp(x) / Halide::sqrt(x)) * poly;

    return Halide::select(x <= 3.75f, small, large);
}

/// @brief Compute I0(x) element-wise using Abramowitz & Stegun polynomial approximation
/// @note Accurate to ~1e-7 for all x; delegates to i0_expr (value-identical)
inline
Halide::Func i0(Halide::Func f, const shape_t& shape,
    std::string const& name = "i0")
{
    Halide::Func ret(name);
    std::vector<Halide::Var> vars;
    for (int i = 0; i < shape.rank; ++i) vars.push_back(Halide::Var());
    ret(vars) = i0_expr(f(vars));
    return ret;
}

// -----------------------------------------------------------------------------
// 1D Cross-Correlation
// -----------------------------------------------------------------------------

/// @brief 1D cross-correlation (without kernel flip)
/// @param a Input signal (size na)
/// @param v Correlation kernel (size nv)
/// @param na Length of a
/// @param nv Length of v
/// @param mode "full" (na+nv-1), "same" (na), or "valid" (na-nv+1)
/// @return Correlation result
///
/// c[m] = sum_n a[n + m - pad] * v[n]  (with zero padding outside a)
/// Compare to convolve1d which flips v before correlating.
inline
Halide::Func correlate1d(Halide::Func a, Halide::Func v, int na, int nv,
    const std::string& mode = "full", std::string const& name = "corr1d")
{
    int pad, out_size;
    if (mode == "full") {
        pad = nv - 1;
        out_size = na + nv - 1;
    } else if (mode == "same") {
        pad = (nv - 1) / 2;
        out_size = na;
    } else { // "valid"
        nh_require(na >= nv, "correlate1d valid mode: na must be >= nv");
        pad = 0;
        out_size = na - nv + 1;
    }

    // Accumulate in the input's own float type (f64 stays f64); integer
    // inputs keep the historical f32 validity factor.
    Halide::Type t = a.types()[0];
    if (!t.is_float()) t = Halide::Float(32);

    Halide::Func ret(name);
    Halide::Var k("k");
    Halide::RDom rn(0, nv, "rn_corr1d");
    ret(k) = Halide::cast(t, 0);
    Halide::Expr a_idx = rn + k - pad;
    // Guard as multiplied 0/1 factor + unconditional clamp — a select whose
    // condition proves the clamp redundant lets the simplifier strip it and
    // CMOV reads OOB (see polymul in polynomial.h).
    Halide::Expr valid = a_idx >= 0 && a_idx < na;
    ret(k) += a(Halide::clamp(a_idx, 0, na - 1)) * v(rn) * Halide::cast(t, valid);
    return ret;
}

NS_NUM_HALIDE_END
