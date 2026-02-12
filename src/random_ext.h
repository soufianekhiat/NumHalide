/// @file random_ext.h
/// @brief Extended random distributions for Halide::Func objects
///
/// Provides: rand_exponential, rand_bernoulli, rand_choice

#pragma once

#include "common.h"
#include "shape.h"
#include "factory_func.h"

NS_NUM_HALIDE_BEGIN

// -----------------------------------------------------------------------------
// Extended Random Distributions
// -----------------------------------------------------------------------------

/// @brief Generate exponential random values via inverse CDF: -ln(u)/lambda
/// @param type Data type (should be float type)
/// @param shape Shape of the output
/// @param lambda Rate parameter (default 1.0)
/// @param seed Random seed for reproducibility
/// @param name Function name
/// @return Halide::Func with exponential random values
inline Halide::Func rand_exponential(Halide::Type type, const shape_t& shape, Halide::Expr lambda = 1.0f, int seed = 0, std::string const& name = "rand_exp") {
	Halide::Func u = rand_uniform(type, shape, seed, name + "_u");
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) vars.push_back(Halide::Var());
	Halide::Expr u_val = Halide::max(u(vars), Halide::cast(type, 1e-10f));
	ret(vars) = -Halide::log(u_val) / lambda;
	return ret;
}

/// @brief Generate Bernoulli random values: 1 with probability p, 0 otherwise
/// @param type Data type
/// @param shape Shape of the output
/// @param p Probability of 1 (default 0.5)
/// @param seed Random seed for reproducibility
/// @param name Function name
/// @return Halide::Func with Bernoulli random values
inline Halide::Func rand_bernoulli(Halide::Type type, const shape_t& shape, Halide::Expr p = 0.5f, int seed = 0, std::string const& name = "rand_bernoulli") {
	Halide::Func u = rand_uniform(Halide::Float(32), shape, seed, name + "_u");
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) vars.push_back(Halide::Var());
	ret(vars) = Halide::cast(type, Halide::select(u(vars) < p, Halide::cast(type, 1), Halide::cast(type, 0)));
	return ret;
}

/// @brief Generate uniform random integers in [0, n)
/// @param type Data type (should be integer type)
/// @param shape Shape of the output
/// @param n Upper bound (exclusive)
/// @param seed Random seed for reproducibility
/// @param name Function name
/// @return Halide::Func with random integers in [0, n)
inline Halide::Func rand_choice(Halide::Type type, const shape_t& shape, Halide::Expr n, int seed = 0, std::string const& name = "rand_choice") {
	Halide::Func u = rand_uniform(Halide::Float(32), shape, seed, name + "_u");
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) vars.push_back(Halide::Var());
	ret(vars) = Halide::cast(type, Halide::floor(u(vars) * Halide::cast<float>(n)));
	return ret;
}

NS_NUM_HALIDE_END
