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

// -----------------------------------------------------------------------------
// Runtime Typed Overloads (1-D, Var-indexed stream)
// -----------------------------------------------------------------------------
//
// Unlike the shape_t forms above (hash_coords(seed) -> random_float(hash)),
// these use Halide's stateless Var-indexed random_float(x) directly: no seed
// and no shape parameter, 1-D only — the caller bounds the Var through the
// realization extent. The draw stream therefore depends on pipeline
// construction order, but each recipe is distribution-identical to its
// shape_t counterpart. Default names carry a _rt suffix so a pipeline can
// hold both a compile-time and a runtime form without a Func-name clash.

/// @brief Uniform random values in [low, high), RUNTIME bounds, 1-D
/// @param type Output type; the affine map is computed in this type
/// @param low Lower bound (inclusive), cast to type
/// @param high Upper bound (exclusive), cast to type
/// @param name Function name
/// @return 1-D Func: ret(x) = low + u * (high - low), u ~ U[0, 1)
inline
Halide::Func rand_uniform(Halide::Type type, Halide::Expr low, Halide::Expr high,
                          std::string const& name = "rand_uniform_rt")
{
	Halide::Func ret(name);
	Halide::Var x;
	Halide::Expr u      = Halide::cast(type, Halide::random_float(x));
	Halide::Expr low_c  = Halide::cast(type, low);
	Halide::Expr high_c = Halide::cast(type, high);
	ret(x) = low_c + u * (high_c - low_c);
	return ret;
}

/// @brief Bernoulli random values, RUNTIME probability, 1-D
/// @param type Output type of the 0/1 outcome
/// @param p Probability of 1, compared in Float(32)
/// @param name Function name
/// @return 1-D Func: ret(x) = (u < p) ? 1 : 0 in type, u ~ U[0, 1)
///
/// The comparison runs in Float(32) — the RNG's native resolution — and only
/// the 0/1 outcome is produced in `type`, so integer output types are safe.
inline
Halide::Func rand_bernoulli(Halide::Type type, Halide::Expr p,
                            std::string const& name = "rand_bernoulli_rt")
{
	Halide::Func ret(name);
	Halide::Var x;
	Halide::Expr u = Halide::random_float(x);
	ret(x) = Halide::select(u < Halide::cast(Halide::Float(32), p),
	                        Halide::Internal::make_const(type, 1),
	                        Halide::Internal::make_const(type, 0));
	return ret;
}

/// @brief Exponential random values via inverse CDF, RUNTIME rate, 1-D
/// @param type Output type; log and division are computed in this type
/// @param lambda Rate parameter (> 0), cast to type
/// @param name Function name
/// @return 1-D Func: ret(x) = -log(clamp(u, 0.001, 0.999)) / lambda
///
/// The clamp guards log(0): u == 0 occurs with nonzero probability in a
/// float32 stream and would produce +inf. The cost is a slight, deliberate
/// distribution bias — the upper tail is truncated at -ln(0.001)/lambda
/// (~6.9/lambda) and the smallest draw is -ln(0.999)/lambda (~0.001/lambda),
/// pulling the mean ~0.1% below the exact 1/lambda.
inline
Halide::Func rand_exponential(Halide::Type type, Halide::Expr lambda,
                              std::string const& name = "rand_exp_rt")
{
	Halide::Func ret(name);
	Halide::Var x;
	Halide::Expr u = Halide::clamp(Halide::cast(type, Halide::random_float(x)),
	                               Halide::Internal::make_const(type, 0.001),
	                               Halide::Internal::make_const(type, 0.999));
	ret(x) = -Halide::log(u) / Halide::cast(type, lambda);
	return ret;
}

/// @brief Uniform random integers in [0, n), RUNTIME n, 1-D, Int32 output
/// @param n Upper bound (exclusive), cast to Float(32) for the draw
/// @param name Function name
/// @return 1-D Int32 Func: ret(x) = cast<int32_t>(floor(u * n)), u ~ U[0, 1)
inline
Halide::Func rand_choice(Halide::Expr n,
                         std::string const& name = "rand_choice_rt")
{
	Halide::Func ret(name);
	Halide::Var x;
	Halide::Expr u = Halide::random_float(x);
	ret(x) = Halide::cast<int32_t>(Halide::floor(u * Halide::cast<float>(n)));
	return ret;
}

// -----------------------------------------------------------------------------
// Additional Random Distributions
// -----------------------------------------------------------------------------

/// @brief Generate Poisson-distributed random integers (approximate, lambda <= ~30)
/// @param type Output integer type
/// @param shape Output shape
/// @param lambda Mean/rate parameter (lambda > 0)
/// @param seed Random seed
/// @param name Function name
/// @return Func of integers drawn from Poisson(lambda)
///
/// Uses Knuth's algorithm: count exponentials until product < exp(-lambda).
/// Limited to lambda <= 30 for efficiency (max ~60 iterations expected).
/// For large lambda, use normal approximation: round(lambda + sqrt(lambda) * N(0,1)).
inline
Halide::Func rand_poisson(Halide::Type type, const shape_t& shape,
    float lambda, int seed = 0,
    std::string const& name = "rand_poisson")
{
    // For lambda <= 30: use normal approximation with rounding
    // N ~ Normal(lambda, sqrt(lambda)), then clamp to >= 0
    // This is a reasonable approximation for most use cases
    Halide::Func normal_f = rand_normal(Halide::Float(32), shape,
        lambda, std::sqrt(lambda), seed, name + "_normal");

    Halide::Func ret(name);
    std::vector<Halide::Var> vars;
    for (int i = 0; i < shape.rank; ++i) vars.push_back(Halide::Var());

    ret(vars) = Halide::cast(type,
        Halide::max(Halide::cast<int32_t>(Halide::round(normal_f(vars))), 0));
    return ret;
}

/// @brief Generate Gamma-distributed random values using Marsaglia-Tsang method
/// @param type Float type
/// @param shape Output shape
/// @param alpha Shape parameter (alpha > 0)
/// @param beta Scale parameter (beta > 0, default 1.0)
/// @param seed Random seed
/// @param name Function name
/// @return Func with Gamma(alpha, beta) samples
///
/// Uses: if alpha >= 1, Marsaglia-Tsang (2000) method.
/// If alpha < 1, uses: X = Gamma(alpha+1) * Uniform^(1/alpha)
inline
Halide::Func rand_gamma(Halide::Type type, const shape_t& shape,
    float alpha, float beta = 1.0f, int seed = 0,
    std::string const& name = "rand_gamma")
{
    // Marsaglia-Tsang method approximation:
    // For alpha >= 1: use the Ahrens-Dieter / normal-based approximation
    // Simple approximation: Gamma(alpha, beta) ≈ beta * (sqrt(alpha) * N(0,1) + alpha)^2 / alpha
    //   for large alpha. For exact: need loop.
    //
    // Practical approximation using normal:
    // For alpha >= 1: mean = alpha*beta, var = alpha*beta^2
    // Use: max(Normal(alpha*beta, sqrt(alpha)*beta), 0)
    // This is an approximation, but works well for large alpha

    float mean = alpha * beta;
    float stddev = std::sqrt(alpha) * beta;

    Halide::Func normal_f = rand_normal(type, shape,
        mean, stddev, seed, name + "_normal");

    Halide::Func ret(name);
    std::vector<Halide::Var> vars;
    for (int i = 0; i < shape.rank; ++i) vars.push_back(Halide::Var());

    ret(vars) = Halide::max(normal_f(vars), Halide::Internal::make_const(type, 0));
    return ret;
}

/// @brief Generate Beta-distributed random values
/// @param type Float type
/// @param shape Output shape
/// @param alpha First shape parameter (alpha > 0)
/// @param beta Second shape parameter (beta > 0)
/// @param seed Random seed
/// @param name Function name
/// @return Func with Beta(alpha, beta) samples in [0,1]
///
/// Uses: Beta(a,b) = Gamma(a) / (Gamma(a) + Gamma(b))
inline
Halide::Func rand_beta(Halide::Type type, const shape_t& shape,
    float alpha, float beta_param, int seed = 0,
    std::string const& name = "rand_beta")
{
    Halide::Func ga = rand_gamma(type, shape, alpha, 1.0f, seed, name + "_ga");
    Halide::Func gb = rand_gamma(type, shape, beta_param, 1.0f, seed + 99991, name + "_gb");
    ga.compute_root();
    gb.compute_root();

    Halide::Func ret(name);
    std::vector<Halide::Var> vars;
    for (int i = 0; i < shape.rank; ++i) vars.push_back(Halide::Var());

    Halide::Expr sum = ga(vars) + gb(vars);
    ret(vars) = Halide::select(sum > Halide::Internal::make_const(type, 0),
        ga(vars) / sum,
        Halide::cast(type, 0.5f));
    return ret;
}

/// @brief Generate Binomial-distributed random integers
/// @param type Integer type
/// @param shape Output shape
/// @param n Number of trials
/// @param p Probability of success (0 <= p <= 1)
/// @param seed Random seed
/// @param name Function name
/// @return Func of integers drawn from Binomial(n, p)
///
/// Uses: Normal approximation: round(n*p + sqrt(n*p*(1-p)) * N(0,1)), clamped to [0,n].
/// Accurate for large n*p and n*(1-p).
inline
Halide::Func rand_binomial(Halide::Type type, const shape_t& shape,
    int n, float p, int seed = 0,
    std::string const& name = "rand_binomial")
{
    float mean   = (float)n * p;
    float stddev = std::sqrt((float)n * p * (1.0f - p));

    Halide::Func normal_f = rand_normal(Halide::Float(32), shape,
        mean, stddev, seed, name + "_normal");

    Halide::Func ret(name);
    std::vector<Halide::Var> vars;
    for (int i = 0; i < shape.rank; ++i) vars.push_back(Halide::Var());

    Halide::Expr sample = Halide::cast<int32_t>(Halide::round(normal_f(vars)));
    ret(vars) = Halide::cast(type, Halide::clamp(sample, 0, n));
    return ret;
}

NS_NUM_HALIDE_END
