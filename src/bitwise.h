/// @file bitwise.h
/// @brief Bitwise operations for integer Halide::Func objects
///
/// Provides: bitwise_and, bitwise_or, bitwise_xor, bitwise_not, left_shift, right_shift, popcount

#pragma once

#include "common.h"
#include "shape.h"

NS_NUM_HALIDE_BEGIN

// -----------------------------------------------------------------------------
// Bitwise Operations
// -----------------------------------------------------------------------------

/// @brief Element-wise bitwise AND of two integer arrays
/// @param a First input Func (integer type)
/// @param b Second input Func (integer type)
/// @param shape Shape of both arrays
/// @param name Function name
/// @return Halide::Func with a & b
inline Halide::Func bitwise_and(Halide::Func a, Halide::Func b, const shape_t& shape, std::string const& name = "bitwise_and") {
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) vars.push_back(Halide::Var());
	ret(vars) = a(vars) & b(vars);
	return ret;
}

/// @brief Element-wise bitwise OR of two integer arrays
/// @param a First input Func (integer type)
/// @param b Second input Func (integer type)
/// @param shape Shape of both arrays
/// @param name Function name
/// @return Halide::Func with a | b
inline Halide::Func bitwise_or(Halide::Func a, Halide::Func b, const shape_t& shape, std::string const& name = "bitwise_or") {
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) vars.push_back(Halide::Var());
	ret(vars) = a(vars) | b(vars);
	return ret;
}

/// @brief Element-wise bitwise XOR of two integer arrays
/// @param a First input Func (integer type)
/// @param b Second input Func (integer type)
/// @param shape Shape of both arrays
/// @param name Function name
/// @return Halide::Func with a ^ b
inline Halide::Func bitwise_xor(Halide::Func a, Halide::Func b, const shape_t& shape, std::string const& name = "bitwise_xor") {
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) vars.push_back(Halide::Var());
	ret(vars) = a(vars) ^ b(vars);
	return ret;
}

/// @brief Element-wise bitwise NOT of an integer array
/// @param a Input Func (integer type)
/// @param shape Shape of the array
/// @param name Function name
/// @return Halide::Func with ~a
inline Halide::Func bitwise_not(Halide::Func a, const shape_t& shape, std::string const& name = "bitwise_not") {
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) vars.push_back(Halide::Var());
	ret(vars) = ~a(vars);
	return ret;
}

/// @brief Element-wise left shift of an integer array
/// @param a Input Func (integer type)
/// @param shape Shape of the array
/// @param n Number of bits to shift
/// @param name Function name
/// @return Halide::Func with a << n
inline Halide::Func left_shift(Halide::Func a, const shape_t& shape, Halide::Expr n, std::string const& name = "left_shift") {
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) vars.push_back(Halide::Var());
	ret(vars) = a(vars) << n;
	return ret;
}

/// @brief Element-wise right shift of an integer array
/// @param a Input Func (integer type)
/// @param shape Shape of the array
/// @param n Number of bits to shift
/// @param name Function name
/// @return Halide::Func with a >> n
inline Halide::Func right_shift(Halide::Func a, const shape_t& shape, Halide::Expr n, std::string const& name = "right_shift") {
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) vars.push_back(Halide::Var());
	ret(vars) = a(vars) >> n;
	return ret;
}

/// @brief Element-wise population count (number of set bits) of an integer array
/// @param a Input Func (integer type)
/// @param shape Shape of the array
/// @param name Function name
/// @return Halide::Func with popcount of each element
inline Halide::Func popcount(Halide::Func a, const shape_t& shape, std::string const& name = "popcount") {
	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < shape.rank; ++i) vars.push_back(Halide::Var());
	ret(vars) = Halide::popcount(a(vars));
	return ret;
}

NS_NUM_HALIDE_END
