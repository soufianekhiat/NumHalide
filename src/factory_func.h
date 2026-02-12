/// @file factory_func.h
/// @brief Factory functions for creating Halide::Func objects with NumPy-like semantics
///
/// Provides: linspace, arange, zeros, ones, full, identity, meshgrid

#pragma once

#include "numhalide.h"
#include "shape.h"

NS_NUM_HALIDE_BEGIN

/// @brief Check if expression is a constant with specific value
template < typename ValueType >
bool is_const_value(Halide::Expr x, ValueType v)
{
	if (int64_t const* pValue = Halide::Internal::as_const_int(x))
	{
		return static_cast<ValueType>(*pValue) == v;
	}
	else if (uint64_t const* pValue = Halide::Internal::as_const_uint(x))
	{
		return static_cast<ValueType>(*pValue) == v;
	}
	else if (f64 const* pValue = Halide::Internal::as_const_float(x))
	{
		return static_cast<ValueType>(*pValue) == v;
	}
	return Halide::Internal::is_const(x, static_cast<Int64>(v));
}

/// @brief Create evenly spaced values over a specified interval
/// @param type Data type for the output
/// @param _start Starting value
/// @param _stop End value
/// @param num Number of samples (default 50)
/// @param endpoint If true, stop is the last sample (default true)
/// @param axis Axis index (currently only 0 supported)
/// @param name Function name
/// @return Halide::Func representing the linspace
///
/// Usage:
///   Func xs = linspace(Float(32), 0.0f, 1.0f, 3);
///   // xs(0) == 0.0f, xs(1) == 0.5f, xs(2) == 1.0f
inline
Halide::Func	linspace( Halide::Type type, Halide::Expr _start, Halide::Expr _stop, Halide::Expr num = 50, Bool endpoint = true, Int32 axis = 0, std::string const& name = "linspace" )
{
	Halide::Func ret( name );
	Halide::Var x;//{ "x" };

	Halide::Expr start = Halide::cast(type, _start);
	Halide::Expr stop = Halide::cast(type, _stop);

	// TODO: Re-enable these requires once we figure out why they crash
	//Halide::require(start < stop, {start, stop});
	//Halide::require(num > 0, { num });

	if (Halide::Internal::is_const_zero(num))
	{
		ret(x) = Halide::Internal::make_const(type, 0);
	}

	if (Halide::Internal::is_const_one(num))
	{
		ret(x) = start;
	}

	if (endpoint)
	{
		if (is_const_value(num, 2))
		{
			ret(x) = Halide::mux(x, { start, stop });
		}
		else
		{
			Halide::Expr step = (stop - start) / Halide::cast(type, num - 1);
			ret(x) = start + Halide::cast(type, x) * step;
		}
	}
	else
	{
		Halide::Expr step = (stop - start) / Halide::cast(type, num);
		ret(x) = start + Halide::cast(type, x) * step;
	}

	if (axis > 0)
	{
		NH_ASSERT(false && "Multi-axis linspace not yet supported");
	}

	return ret;
}

/// @brief Return evenly spaced values within interval [0, stop)
/// @param type Data type
/// @param _stop End value (exclusive)
/// @param name Function name
/// @return Halide::Func representing the range
///
/// Usage:
///   arange(Float(32), 5) == [0, 1, 2, 3, 4]
inline
Halide::Func	arange(Halide::Type type, Halide::Expr _stop, std::string const& name = "arange")
{
	Halide::Func ret(name);
	Halide::Var x("x");

	Halide::Expr stop = Halide::cast(type, _stop);
	// TODO: Re-enable once require issue is fixed
	//Halide::require(stop > 0, stop);

	ret(x) = Halide::cast(type, x);

	return ret;
}

/// @brief Return evenly spaced values within interval [start, stop) with step
/// @param type Data type
/// @param _start Start value
/// @param _stop End value (exclusive)
/// @param _step Step size (default 1)
/// @param name Function name
/// @return Halide::Func representing the range
///
/// Usage:
///   arange(Float(32), 0.0f, 5.0f, 0.5f) == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
inline
Halide::Func	arange(Halide::Type type, Halide::Expr _start, Halide::Expr _stop, Halide::Expr _step = 1, std::string const& name = "arange")
{
	Halide::Func ret(name);
	Halide::Var x("x");

	Halide::Expr start = Halide::cast(type, _start);
	Halide::Expr stop = Halide::cast(type, _stop);
	Halide::Expr step = Halide::cast(type, _step);

	// TODO: Re-enable once require issue is fixed
	//Halide::require(
	//	(step > Halide::Internal::make_const(type, 0) && stop > start) ||
	//	(step < Halide::Internal::make_const(type, 0) && start > stop),
	//	{start, stop, step}
	//);

	ret(x) = Halide::cast(type, start + step * Halide::cast(type, x));

	return ret;
}

/// @brief Return a Func filled with a constant value
/// @param type Data type
/// @param value Fill value
/// @param shape Shape of the array
/// @param name Function name
/// @return Halide::Func filled with value
inline
Halide::Func	full(Halide::Type type, Halide::Expr value, const shape_t& shape, std::string const& name = "full")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;

	// Create uniquely named variables
	for (Int32 i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	ret(vars) = Halide::cast(type, value);

	return ret;
}

/// @brief Return a Func filled with a constant value (legacy int dim)
/// @param type Data type
/// @param value Fill value
/// @param dim Number of dimensions
/// @param name Function name
/// @return Halide::Func filled with value
inline
Halide::Func	full(Halide::Type type, Halide::Expr value, Int32 dim, std::string const& name = "full")
{
	shape_t s;
	s.rank = dim;
	// Extents don't matter for full() as it's infinite, but rank does for Var creation
	return full(type, value, s, name);
}

/// @brief Return an empty (uninitialized) Func
inline
Halide::Func	empty(Halide::Type type, const shape_t& shape, std::string const& name = "empty")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;

	// Create uniquely named variables
	for (Int32 i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	return ret;
}

/// @brief Return an empty (uninitialized) Func (legacy int dim)
inline
Halide::Func	empty(Halide::Type type, Int32 dim, std::string const& name = "empty")
{
	shape_t s;
	s.rank = dim;
	return empty(type, s, name);
}

/// @brief Return a Func filled with ones
inline
Halide::Func	ones(Halide::Type type, const shape_t& shape, std::string const& name = "ones")
{
	return full(type, 1, shape, name);
}

/// @brief Return a Func filled with ones (legacy)
inline
Halide::Func	ones(Halide::Type type, Int32 const dim, std::string const& name = "ones")
{
	return full(type, 1, dim, name);
}

/// @brief Return a Func filled with zeros
inline
Halide::Func	zeros(Halide::Type type, const shape_t& shape, std::string const& name = "zeros")
{
	return full(type, 0, shape, name);
}

/// @brief Return a Func filled with zeros (legacy)
inline
Halide::Func	zeros(Halide::Type type, Int32 const dim, std::string const& name = "zeros")
{
	return full(type, 0, dim, name);
}

/// @brief Create identity matrix
/// @param type Data type
/// @param dim Matrix dimension (creates dim x dim identity)
/// @param name Function name
/// @return Halide::Func representing identity matrix
///
/// Usage:
///   identity(Float(32), 3) == [[1,0,0], [0,1,0], [0,0,1]]
inline
Halide::Func	identity(Halide::Type type, Int32 const dim, std::string const& name = "identity")
{
	Halide::Func ret(name);
	Halide::Var x("x"), y("y");

	ret(x, y) = Halide::select(x == y, Halide::cast(type, 1), Halide::cast(type, 0));

	return ret;
}

/// @brief Create coordinate matrices from coordinate vectors (meshgrid)
/// @param type Data type
/// @param xis Vector of 1D coordinate Funcs
/// @param name Base function name
/// @return Vector of Funcs representing coordinate grids
///
/// Usage:
///   Func xs = linspace(Float(32), 0.0f, 1.0f, 8);
///   Func ys = linspace(Float(32), 0.0f, 1.0f, 4);
///   auto mg = meshgrid(Float(32), {xs, ys});
///   // mg[0](x,y) = x coordinates, mg[1](x,y) = y coordinates
inline
std::vector<Halide::Func>	meshgrid(Halide::Type type, std::vector<Halide::Func> xis, std::string const& name = "meshgrid")
{
	Int32 rank = (Int32)xis.size();

	std::vector<Halide::Var> vars;

	// Create uniquely named variables
	for (Int32 i = 0; i < rank; ++i) {
		vars.push_back(Halide::Var());
	}

	std::vector<Halide::Func> values;
	values.reserve(rank);
	for (Int32 k = 0; k < rank; ++k)
	{
		Halide::Func base = empty(type, rank, name + '_' + std::to_string(k));
		base(vars) = xis[k](vars[k]);

		values.push_back(base);
	}

	return values;
}

// -----------------------------------------------------------------------------
// Random Number Generation
// -----------------------------------------------------------------------------

/// @brief Hash function for deterministic pseudo-random number generation
/// @param seed Base seed value
/// @param coords Coordinate expressions to hash
/// @return Hashed integer expression
///
/// Uses a simple but effective hash combining technique for reproducibility.
inline
Halide::Expr hash_coords(int seed, const std::vector<Halide::Expr>& coords)
{
	// Use a simple multiplicative hash
	// Constants from: https://stackoverflow.com/questions/664014/
	Halide::Expr hash = Halide::cast<int32_t>(seed);
	const int32_t prime1 = 0x45d9f3b;
	const int32_t prime2 = 0x119de1f3;

	for (size_t i = 0; i < coords.size(); ++i) {
		Halide::Expr coord = Halide::cast<int32_t>(coords[i]);
		hash = hash ^ (coord * prime1);
		hash = hash * prime2;
	}
	return hash;
}

/// @brief Generate uniform random values in [0, 1)
/// @param type Data type (should be float type)
/// @param shape Shape of the output
/// @param seed Random seed for reproducibility
/// @param name Function name
/// @return Halide::Func with uniform random values
///
/// Usage:
///   Func noise = rand_uniform(Float(32), {256, 256}, 42);
inline
Halide::Func rand_uniform(Halide::Type type, const shape_t& shape, int seed = 0, std::string const& name = "rand_uniform")
{
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	std::vector<Halide::Expr> var_exprs;

	for (Int32 i = 0; i < shape.rank; ++i) {
		Halide::Var v;
		vars.push_back(v);
		var_exprs.push_back(v);
	}

	// Generate hash and convert to float in [0, 1)
	Halide::Expr hash = hash_coords(seed, var_exprs);
	// Use the hash to seed Halide's random_float for better distribution
	Halide::Expr random_val = Halide::random_float(hash);

	ret(vars) = Halide::cast(type, random_val);
	return ret;
}

/// @brief Generate uniform random values in [low, high)
/// @param type Data type
/// @param shape Shape of the output
/// @param low Lower bound (inclusive)
/// @param high Upper bound (exclusive)
/// @param seed Random seed for reproducibility
/// @param name Function name
/// @return Halide::Func with uniform random values in [low, high)
inline
Halide::Func rand_uniform(Halide::Type type, const shape_t& shape, Halide::Expr low, Halide::Expr high, int seed = 0, std::string const& name = "rand_uniform")
{
	Halide::Func base = rand_uniform(type, shape, seed, name + "_base");

	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (Int32 i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	Halide::Expr low_cast = Halide::cast(type, low);
	Halide::Expr high_cast = Halide::cast(type, high);
	ret(vars) = low_cast + base(vars) * (high_cast - low_cast);
	return ret;
}

/// @brief Generate normal (Gaussian) random values using Box-Muller transform
/// @param type Data type (should be float type)
/// @param shape Shape of the output
/// @param mean Mean of the distribution (default 0)
/// @param stddev Standard deviation (default 1)
/// @param seed Random seed for reproducibility
/// @param name Function name
/// @return Halide::Func with normal random values
///
/// Usage:
///   Func noise = rand_normal(Float(32), {256, 256}, 0.0f, 1.0f, 42);
inline
Halide::Func rand_normal(Halide::Type type, const shape_t& shape, Halide::Expr mean = 0.0f, Halide::Expr stddev = 1.0f, int seed = 0, std::string const& name = "rand_normal")
{
	// Box-Muller transform: generate two uniform randoms, produce one normal
	Halide::Func u1 = rand_uniform(type, shape, seed, name + "_u1");
	Halide::Func u2 = rand_uniform(type, shape, seed + 12345, name + "_u2");

	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (Int32 i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	// Box-Muller: z = sqrt(-2 * ln(u1)) * cos(2 * pi * u2)
	const float pi = 3.14159265358979323846f;
	Halide::Expr u1_val = u1(vars);
	Halide::Expr u2_val = u2(vars);

	// Clamp u1 to avoid log(0)
	u1_val = Halide::max(u1_val, Halide::cast(type, 1e-10f));

	Halide::Expr z = Halide::sqrt(-2.0f * Halide::log(u1_val)) * Halide::cos(2.0f * pi * u2_val);
	ret(vars) = Halide::cast(type, mean) + Halide::cast(type, stddev) * z;
	return ret;
}

/// @brief Generate random integers in [low, high)
/// @param type Data type (should be integer type)
/// @param shape Shape of the output
/// @param low Lower bound (inclusive)
/// @param high Upper bound (exclusive)
/// @param seed Random seed for reproducibility
/// @param name Function name
/// @return Halide::Func with random integers
inline
Halide::Func rand_int(Halide::Type type, const shape_t& shape, Halide::Expr low, Halide::Expr high, int seed = 0, std::string const& name = "rand_int")
{
	Halide::Func base = rand_uniform(Halide::Float(32), shape, seed, name + "_base");

	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (Int32 i = 0; i < shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	Halide::Expr range = Halide::cast<float>(high - low);
	ret(vars) = Halide::cast(type, low + Halide::cast<int32_t>(Halide::floor(base(vars) * range)));
	return ret;
}

// -----------------------------------------------------------------------------
// *_like Functions
// -----------------------------------------------------------------------------

/// @brief Return a Func filled with zeros, same shape as input
/// @param f Reference Func (used only for shape)
/// @param shape Shape of the reference
/// @param name Function name
/// @return Halide::Func filled with zeros
inline
Halide::Func zeros_like(Halide::Func f, const shape_t& shape, std::string const& name = "zeros_like")
{
	Halide::Type type = f.types()[0];
	return zeros(type, shape, name);
}

/// @brief Return a Func filled with ones, same shape as input
/// @param f Reference Func (used only for shape)
/// @param shape Shape of the reference
/// @param name Function name
/// @return Halide::Func filled with ones
inline
Halide::Func ones_like(Halide::Func f, const shape_t& shape, std::string const& name = "ones_like")
{
	Halide::Type type = f.types()[0];
	return ones(type, shape, name);
}

/// @brief Return a Func filled with a constant value, same shape/type as input
/// @param f Reference Func (used only for type)
/// @param shape Shape of the reference
/// @param value Fill value
/// @param name Function name
/// @return Halide::Func filled with value
inline
Halide::Func full_like(Halide::Func f, const shape_t& shape, Halide::Expr value, std::string const& name = "full_like")
{
	Halide::Type type = f.types()[0];
	return full(type, value, shape, name);
}

/// @brief Return a Func filled with zeros, same type as input
/// @param f Reference Func (used only for type)
/// @param shape Shape of the output
/// @param name Function name
/// @return Halide::Func filled with zeros
inline
Halide::Func zeros_like(Halide::Type type, const shape_t& shape, std::string const& name = "zeros_like")
{
	return zeros(type, shape, name);
}

/// @brief Return a Func filled with ones, same type as input
/// @param f Reference Func (used only for type)
/// @param shape Shape of the output
/// @param name Function name
/// @return Halide::Func filled with ones
inline
Halide::Func ones_like(Halide::Type type, const shape_t& shape, std::string const& name = "ones_like")
{
	return ones(type, shape, name);
}

NS_NUM_HALIDE_END
