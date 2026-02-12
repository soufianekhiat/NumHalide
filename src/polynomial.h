/// @file polynomial.h
/// @brief Polynomial evaluation operations
///
/// Provides: polyval, chebyshev_t, legendre_p

#pragma once

#include "common.h"
#include "shape.h"

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
	nh_require(nullptr, n_coeffs >= 1, "polyval requires at least 1 coefficient");

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
	nh_require(nullptr, n >= 0, "Chebyshev degree must be non-negative, got %d", n);

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
	nh_require(nullptr, n >= 0, "Legendre degree must be non-negative, got %d", n);

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

NS_NUM_HALIDE_END
