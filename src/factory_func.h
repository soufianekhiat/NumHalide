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
	std::vector<Halide::Var> vars;

	// Create uniquely named variables
	for (Int32 i = 0; i < dim; ++i) {
		vars.push_back(Halide::Var());
	}

	ret(vars) = Halide::cast(type, 0);
	std::vector<Halide::Var> args(dim, vars[0]);
	ret(args) = Halide::cast(type, 1);

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

NS_NUM_HALIDE_END
