/// @file polynomial.h
/// @brief Polynomial evaluation operations
///
/// Provides: polyval, chebyshev_t, legendre_p, polyfit

#pragma once

#include "common.h"
#include "shape.h"
#include "la.h"

NS_NUM_HALIDE_BEGIN

// -----------------------------------------------------------------------------
// Polynomial Evaluation
// -----------------------------------------------------------------------------

/// @brief Evaluate polynomial using Horner's method
/// @param coeffs 1D Func of coefficients (highest degree first, NumPy convention)
///        coeffs[0]*x^(n-1) + coeffs[1]*x^(n-2) + ... + coeffs[n-1]
///        Horner form: ((coeffs[0]*x + coeffs[1])*x + coeffs[2])*x + ...
/// @param n_coeffs Number of coefficients
/// @param x_func Input values Func
/// @param shape Shape of x_func
/// @param name Function name
/// @return Func with polynomial evaluated at each element of x_func
inline
Halide::Func polyval(Halide::Func coeffs, int n_coeffs, Halide::Func x_func, const shape_t& shape, std::string const& name = "polyval")
{
	nh_require(n_coeffs >= 1, "polyval requires at least 1 coefficient");

	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	// Build Horner's method expression in C++ loop
	Halide::Expr result = Halide::cast<float>(coeffs(0));
	Halide::Expr xval = x_func(vars);
	for (int i = 1; i < n_coeffs; ++i) {
		result = result * xval + coeffs(i);
	}

	ret(vars) = result;
	return ret;
}

// -----------------------------------------------------------------------------
// Chebyshev Polynomials
// -----------------------------------------------------------------------------

/// @brief Chebyshev polynomial of the first kind, degree n
/// @param n Degree of the polynomial
/// @param x_func Input values Func
/// @param shape Shape of x_func
/// @param name Function name
/// @return Func with T_n(x) evaluated at each element
///
/// Recurrence: T_0 = 1, T_1 = x, T_n = 2*x*T_{n-1} - T_{n-2}
inline
Halide::Func chebyshev_t(int n, Halide::Func x_func, const shape_t& shape, std::string const& name = "chebyshev")
{
	nh_require(n >= 0, "Chebyshev degree must be non-negative, got %d", n);

	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	Halide::Expr xval = x_func(vars);

	if (n == 0) {
		ret(vars) = 1.0f;
	} else if (n == 1) {
		ret(vars) = xval;
	} else {
		// Build iteratively using recurrence relation
		Halide::Expr t_prev2 = 1.0f;   // T_0
		Halide::Expr t_prev1 = xval;   // T_1
		Halide::Expr t_curr = t_prev1;
		for (int i = 2; i <= n; ++i) {
			t_curr = 2.0f * xval * t_prev1 - t_prev2;
			t_prev2 = t_prev1;
			t_prev1 = t_curr;
		}
		ret(vars) = t_curr;
	}

	return ret;
}

// -----------------------------------------------------------------------------
// Legendre Polynomials
// -----------------------------------------------------------------------------

/// @brief Legendre polynomial of degree n
/// @param n Degree of the polynomial
/// @param x_func Input values Func
/// @param shape Shape of x_func
/// @param name Function name
/// @return Func with P_n(x) evaluated at each element
///
/// Recurrence: P_0 = 1, P_1 = x, P_n = ((2n-1)*x*P_{n-1} - (n-1)*P_{n-2}) / n
inline
Halide::Func legendre_p(int n, Halide::Func x_func, const shape_t& shape, std::string const& name = "legendre")
{
	nh_require(n >= 0, "Legendre degree must be non-negative, got %d", n);

	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	Halide::Expr xval = x_func(vars);

	if (n == 0) {
		ret(vars) = 1.0f;
	} else if (n == 1) {
		ret(vars) = xval;
	} else {
		// Build iteratively using Bonnet's recursion formula
		Halide::Expr p_prev2 = 1.0f;   // P_0
		Halide::Expr p_prev1 = xval;   // P_1
		Halide::Expr p_curr = p_prev1;
		for (int i = 2; i <= n; ++i) {
			float fi = static_cast<float>(i);
			p_curr = ((2.0f * fi - 1.0f) * xval * p_prev1 - (fi - 1.0f) * p_prev2) / fi;
			p_prev2 = p_prev1;
			p_prev1 = p_curr;
		}
		ret(vars) = p_curr;
	}

	return ret;
}

// -----------------------------------------------------------------------------
// Polynomial Arithmetic
// -----------------------------------------------------------------------------

/// @brief Add two polynomials (coefficient arrays)
/// @param a Coefficients of first polynomial (1D Func of size na, highest power first)
/// @param na Degree of a (na coefficients: power na-1 down to 0)
/// @param b Coefficients of second polynomial (1D Func of size nb)
/// @param nb Degree of b
/// @param name Function name
/// @return Coefficient array of size max(na, nb), highest power first
///
/// Pads the shorter array with zeros on the left (high-degree side).
inline
Halide::Func polyadd(Halide::Func a, int na, Halide::Func b, int nb,
    std::string const& name = "polyadd")
{
    int nout = std::max(na, nb);
    Halide::Func ret(name);
    Halide::Var i("i");

    // a has coefficients for powers na-1..0 at indices 0..na-1
    // b has coefficients for powers nb-1..0 at indices 0..nb-1
    // result: powers nout-1..0 at indices 0..nout-1
    // a[i] corresponds to power (na-1-i), which in output is at index (nout-1-(na-1-i)) = nout-na+i
    // So a[k] → result[nout - na + k]
    // Equivalently, result[i] = a[i - (nout - na)] + b[i - (nout - nb)]
    //                           (with 0 if index out of range)

    Halide::Expr offset_a = nout - na;
    Halide::Expr offset_b = nout - nb;

    Halide::Expr ai = i - offset_a;
    Halide::Expr bi = i - offset_b;

    Halide::Type type = a.types()[0];

    Halide::Expr aval = Halide::select(ai >= 0 && ai < na, a(Halide::clamp(ai, 0, na-1)), Halide::Internal::make_const(type, 0));
    Halide::Expr bval = Halide::select(bi >= 0 && bi < nb, b(Halide::clamp(bi, 0, nb-1)), Halide::Internal::make_const(type, 0));

    ret(i) = aval + bval;
    return ret;
}

/// @brief Subtract two polynomials: a - b
inline
Halide::Func polysub(Halide::Func a, int na, Halide::Func b, int nb,
    std::string const& name = "polysub")
{
    int nout = std::max(na, nb);
    Halide::Func ret(name);
    Halide::Var i("i");

    Halide::Expr offset_a = nout - na;
    Halide::Expr offset_b = nout - nb;
    Halide::Expr ai = i - offset_a;
    Halide::Expr bi = i - offset_b;
    Halide::Type type = a.types()[0];

    Halide::Expr aval = Halide::select(ai >= 0 && ai < na, a(Halide::clamp(ai, 0, na-1)), Halide::Internal::make_const(type, 0));
    Halide::Expr bval = Halide::select(bi >= 0 && bi < nb, b(Halide::clamp(bi, 0, nb-1)), Halide::Internal::make_const(type, 0));

    ret(i) = aval - bval;
    return ret;
}

/// @brief Multiply two polynomials (convolution of coefficients)
/// @param a Coefficients of first polynomial (size na, highest power first)
/// @param na Size of a
/// @param b Coefficients of second polynomial (size nb)
/// @param nb Size of b
/// @param name Function name
/// @return Coefficient array of size na+nb-1
///
/// If a has degree (na-1) and b has degree (nb-1), result has degree (na+nb-2).
inline
Halide::Func polymul(Halide::Func a, int na, Halide::Func b, int nb,
    std::string const& name = "polymul")
{
    Halide::Func ret(name);
    Halide::Var i("i");
    Halide::Type type = a.types()[0];

    // result[i] = sum_{k} a[k] * b[i - k]  (standard polynomial multiplication)
    // But since highest power is first, we need to think carefully.
    // a represents polynomial: a[0]*x^(na-1) + a[1]*x^(na-2) + ... + a[na-1]*x^0
    // b represents polynomial: b[0]*x^(nb-1) + b[1]*x^(nb-2) + ... + b[nb-1]*x^0
    // result has degree na-1 + nb-1 = na+nb-2, size nout = na+nb-1
    // result[0]*x^(nout-1) + ...
    // result[i] corresponds to power (nout-1-i) = (na+nb-2-i)
    // Contribution of a[ka]*b[kb] (powers (na-1-ka) + (nb-1-kb) = na+nb-2-ka-kb) goes to index ka+kb
    // So result[i] = sum_{k=0}^{na-1} a[k] * b[i-k]  where b[j] = 0 if j<0 or j>=nb

    ret(i) = Halide::Internal::make_const(type, 0);
    Halide::RDom rk(0, na, "rk_polymul");
    Halide::Expr bj = i - rk;
    ret(i) += Halide::select(bj >= 0 && bj < nb,
        a(rk) * b(Halide::clamp(bj, 0, nb-1)),
        Halide::Internal::make_const(type, 0));

    return ret;
}

/// @brief Derivative of a polynomial
/// @param a Coefficients (size na, highest power first: a[0]*x^(na-1) + ...)
/// @param na Number of coefficients
/// @param m Order of derivative (default 1)
/// @param name Function name
/// @return Coefficients of derivative (size na-m, or size 1 with 0 if overdifferentiated)
///
/// d/dx (c * x^n) = c*n * x^(n-1)
/// a[i] has power (na-1-i), so derivative coefficient = a[i] * (na-1-i)
/// output has size na-1 (for m=1)
inline
Halide::Func polyder(Halide::Func a, int na, int m = 1,
    std::string const& name = "polyder")
{
    nh_require(m >= 0, "polyder: order m must be non-negative");

    if (m == 0) return a;
    if (m >= na) {
        // Overdifferentiated: return zero polynomial of degree 0
        Halide::Func ret(name);
        Halide::Var i("i");
        ret(i) = Halide::Internal::make_const(a.types()[0], 0);
        return ret;
    }

    // Apply m-th derivative: each application reduces degree by 1
    Halide::Func cur = a;
    int cur_size = na;
    for (int pass = 0; pass < m; ++pass) {
        int new_size = cur_size - 1;
        Halide::Func next(name + "_der" + std::to_string(pass));
        Halide::Var i("i");
        Halide::Type type = a.types()[0];
        // cur[i] has power (cur_size - 1 - i)
        // derivative: cur[i] * (cur_size - 1 - i), placed at index i in result of size new_size
        next(i) = cur(i) * Halide::cast(type, cur_size - 1 - i);
        next.compute_root();
        cur = next;
        cur_size = new_size;
    }

    Halide::Func ret(name);
    Halide::Var i("i");
    ret(i) = cur(i);
    return ret;
}

/// @brief Antiderivative (integral) of a polynomial
/// @param a Coefficients (size na, highest power first)
/// @param na Number of coefficients
/// @param k Integration constant (coefficient for x^0 in result)
/// @param name Function name
/// @return Coefficients of antiderivative (size na+1, highest power first)
///
/// integral(c * x^n) = c/(n+1) * x^(n+1)
/// a[i] has power (na-1-i), integral coeff = a[i]/(na-i) at power (na-i)
/// result has size na+1: result[i] = a[i] / (na - i) for i<na, result[na] = k
inline
Halide::Func polyint(Halide::Func a, int na, Halide::Expr k = 0.0f,
    std::string const& name = "polyint")
{
    Halide::Func ret(name);
    Halide::Var i("i");
    Halide::Type type = a.types()[0];

    // result[i] for i in 0..na-1: a[i] / (na - i)  [integration of power (na-1-i) → power (na-i)]
    // result[na]: integration constant k
    ret(i) = Halide::select(
        i < na,
        a(Halide::clamp(i, 0, na-1)) / Halide::cast(type, na - i),
        Halide::cast(type, k)
    );

    return ret;
}

// -----------------------------------------------------------------------------
// Polynomial Division
// -----------------------------------------------------------------------------

/// @brief Polynomial long division: a / b → (quotient, remainder)
/// @param a  Dividend coefficients (size na, highest power first)
/// @param na Degree+1 of dividend
/// @param b  Divisor coefficients (size nb, highest power first)
/// @param nb Degree+1 of divisor; must satisfy na >= nb >= 1
/// @return {quotient Func (size na-nb+1), remainder Func (size nb-1)}
///
/// Example: polydiv([1,-3,2], [1,-1]) → ([1,-2], [0])
///   (x²-3x+2) / (x-1) = (x-2) remainder 0
inline
std::pair<Halide::Func, Halide::Func> polydiv(
    Halide::Func a, int na,
    Halide::Func b, int nb,
    std::string const& name = "polydiv")
{
    nh_require(na >= nb, "polydiv: degree of dividend must be >= divisor");
    nh_require(nb >= 1,  "polydiv: divisor must have >= 1 coefficient");

    int q_len = na - nb + 1; // number of quotient coefficients
    Halide::Var i("i");

    // Initialize work array = a
    Halide::Func work(name + "_w0");
    work(i) = a(i);
    work.compute_root();

    // Collect scalar quotient values (0D Funcs)
    std::vector<Halide::Func> q_funcs;

    for (int k = 0; k < q_len; ++k) {
        // q[k] = work[k] / b[0]
        Halide::Func qk(name + "_qk" + std::to_string(k));
        qk() = work(k) / b(0);
        qk.compute_root();
        q_funcs.push_back(qk);

        // Subtract q[k] * b from work at positions [k, k+nb-1]
        Halide::Func nw(name + "_w" + std::to_string(k + 1));
        nw(i) = work(i);
        Halide::RDom rj(k, nb, "rj_pd_" + std::to_string(k));
        nw(rj) = work(rj) - qk() * b(rj - k);
        nw.compute_root();
        work = nw;
    }

    // Build quotient Func from collected scalar values
    Halide::Func q_ret(name + "_q");
    if (q_len == 1) {
        q_ret(i) = q_funcs[0]();
    } else {
        Halide::Expr q_expr = q_funcs[q_len - 1]();
        for (int k = q_len - 2; k >= 0; --k) {
            q_expr = Halide::select(i == k, q_funcs[k](), q_expr);
        }
        q_ret(i) = q_expr;
    }

    // Build remainder Func: last nb-1 elements of the work array
    Halide::Func r_ret(name + "_r");
    int r_len = nb - 1;
    if (r_len > 0) {
        r_ret(i) = work(i + q_len);
    } else {
        r_ret(i) = Halide::cast(a.types()[0], 0);
    }

    return {q_ret, r_ret};
}

// -----------------------------------------------------------------------------
// polyfit — least-squares polynomial fit
// -----------------------------------------------------------------------------

/// @brief Fit a polynomial of degree `deg` to (x[i], y[i]) data points
/// @param x    1D Func of x coordinates, size n
/// @param y    1D Func of y values,       size n
/// @param n    Number of data points (n ≥ deg+1)
/// @param deg  Polynomial degree
/// @param name Base name
/// @return 1D Func of deg+1 coefficients, lowest degree first:
///         coeffs[0] + coeffs[1]*x + ... + coeffs[deg]*x^deg
///
/// Uses QR-based least squares (numerically stable — solves Vandermonde system
/// directly, not the ill-conditioned normal equations).
/// Note: convention differs from numpy.polyfit (which returns highest-first).
///       Use polyval() with the same lowest-first convention.
inline Halide::Func polyfit(Halide::Func x, Halide::Func y, int n, int deg,
    std::string const& name = "polyfit")
{
    int nc = deg + 1;   // number of coefficients

    // Vandermonde matrix: V(col=j, row=i) = x[i]^j  for j = 0..deg
    Halide::Var col("col_pf"), row("row_pf");
    Halide::Func V(name + "_V");
    V(col, row) = Halide::pow(x(row), Halide::cast<float>(col));
    V.compute_root();

    // Solve V @ c = y in the least-squares sense via QR
    return lstsq(V, y, n, nc, name + "_c");
}

NS_NUM_HALIDE_END
