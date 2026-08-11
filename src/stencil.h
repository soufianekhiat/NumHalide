/// @file stencil.h
/// @brief Stencil operations for PDE computations
///
/// Provides: stencil_apply, jacobi_step, heat_diffusion_step

#pragma once

#include "common.h"
#include "shape.h"

NS_NUM_HALIDE_BEGIN

// -----------------------------------------------------------------------------
// Generic Stencil
// -----------------------------------------------------------------------------

/// @brief Apply a generic weighted stencil to a 2D function
/// @param f Input 2D Func
/// @param shape Input shape (must be 2D)
/// @param weights 1D Func of stencil weights (n_points elements)
/// @param offsets_x 1D Func of x offsets (n_points elements)
/// @param offsets_y 1D Func of y offsets (n_points elements)
/// @param n_points Number of stencil points
/// @param name Function name
/// @return Func: ret(x,y) = sum_i weights(i) * f(clamp(x+offsets_x(i)), clamp(y+offsets_y(i)))
inline
Halide::Func stencil_apply(Halide::Func f, const shape_t& shape,
                            Halide::Func weights, Halide::Func offsets_x, Halide::Func offsets_y,
                            int n_points, std::string const& name = "stencil")
{
	nh_require(shape.rank == 2, "stencil_apply requires 2D input");
	int rows = shape.extents[0];
	int cols = shape.extents[1];

	Halide::Func ret(name);
	Halide::Var x, y;
	Halide::RDom r(0, n_points);

	Halide::Expr ix = Halide::clamp(x + Halide::cast<int32_t>(offsets_x(r)), 0, cols - 1);
	Halide::Expr iy = Halide::clamp(y + Halide::cast<int32_t>(offsets_y(r)), 0, rows - 1);

	ret(x, y) = Halide::cast<float>(0);
	ret(x, y) += weights(r) * f(ix, iy);
	return ret;
}

// -----------------------------------------------------------------------------
// Jacobi Iteration
// -----------------------------------------------------------------------------

/// @brief One Jacobi iteration step (average of 4 neighbors)
/// @param f Input 2D Func
/// @param shape Input shape (must be 2D)
/// @param name Function name
/// @return Func: ret(x,y) = (f(x+1,y) + f(x-1,y) + f(x,y+1) + f(x,y-1)) / 4
inline
Halide::Func jacobi_step(Halide::Func f, const shape_t& shape, std::string const& name = "jacobi")
{
	nh_require(shape.rank == 2, "jacobi requires 2D");
	int rows = shape.extents[0];
	int cols = shape.extents[1];

	Halide::Func ret(name);
	Halide::Var x, y;

	Halide::Expr xp = Halide::clamp(x + 1, 0, cols - 1);
	Halide::Expr xm = Halide::clamp(x - 1, 0, cols - 1);
	Halide::Expr yp = Halide::clamp(y + 1, 0, rows - 1);
	Halide::Expr ym = Halide::clamp(y - 1, 0, rows - 1);

	ret(x, y) = (f(xp, y) + f(xm, y) + f(x, yp) + f(x, ym)) / 4.0f;
	return ret;
}

/// @brief One Jacobi iteration step with RUNTIME extents
/// @param f Input 2D Func
/// @param rows Number of rows (y extent) as a runtime expression
/// @param cols Number of columns (x extent) as a runtime expression
/// @param name Function name
/// @return Func: ret(x,y) = 0.25 * (f(x-1,y) + f(x+1,y) + f(x,y-1) + f(x,y+1))
///
/// Type-generic: computes in f.types()[0]. Clamp indexing = repeat-edge
/// boundary, same as the compile-time overload above.
inline
Halide::Func jacobi_step(Halide::Func f, Halide::Expr rows, Halide::Expr cols,
	std::string const& name = "jacobi_rt")
{
	Halide::Func ret(name);
	Halide::Var x("x"), y("y");
	Halide::Type type = f.types()[0];

	Halide::Expr xp = Halide::clamp(x + 1, 0, cols - 1);
	Halide::Expr xm = Halide::clamp(x - 1, 0, cols - 1);
	Halide::Expr yp = Halide::clamp(y + 1, 0, rows - 1);
	Halide::Expr ym = Halide::clamp(y - 1, 0, rows - 1);

	ret(x, y) = Halide::Internal::make_const(type, 0.25) *
		(f(xm, y) + f(xp, y) + f(x, ym) + f(x, yp));
	return ret;
}

// -----------------------------------------------------------------------------
// Heat Diffusion
// -----------------------------------------------------------------------------

/// @brief One explicit Euler step for heat diffusion: f + dt*alpha*laplacian(f)
/// @param f Input 2D Func (temperature field)
/// @param shape Input shape (must be 2D)
/// @param dt Time step size
/// @param alpha Thermal diffusivity
/// @param name Function name
/// @return Func with updated temperature field
inline
Halide::Func heat_diffusion_step(Halide::Func f, const shape_t& shape, float dt = 0.1f, float alpha = 1.0f, std::string const& name = "heat")
{
	nh_require(shape.rank == 2, "heat requires 2D");
	int rows = shape.extents[0];
	int cols = shape.extents[1];

	Halide::Func ret(name);
	Halide::Var x, y;

	Halide::Expr xp = Halide::clamp(x + 1, 0, cols - 1);
	Halide::Expr xm = Halide::clamp(x - 1, 0, cols - 1);
	Halide::Expr yp = Halide::clamp(y + 1, 0, rows - 1);
	Halide::Expr ym = Halide::clamp(y - 1, 0, rows - 1);

	// Discrete Laplacian: sum of neighbors minus 4 times center
	Halide::Expr lap = f(xp, y) + f(xm, y) + f(x, yp) + f(x, ym) - 4.0f * f(x, y);

	ret(x, y) = f(x, y) + dt * alpha * lap;
	return ret;
}

/// @brief Heat diffusion step with RUNTIME extents and RUNTIME scalars
/// @param f Input 2D Func (temperature field)
/// @param rows Number of rows (y extent) as a runtime expression
/// @param cols Number of columns (x extent) as a runtime expression
/// @param dt Time step size as a runtime expression
/// @param alpha Thermal diffusivity as a runtime expression
/// @param name Function name
/// @return Func: ret(x,y) = f(x,y) + alpha*dt*laplacian(f)(x,y)
///
/// Type-generic: computes in f.types()[0]; dt and alpha are cast into that
/// type. Clamp indexing = repeat-edge boundary, same as the compile-time
/// overload above.
inline
Halide::Func heat_diffusion_step(Halide::Func f, Halide::Expr rows, Halide::Expr cols,
	Halide::Expr dt, Halide::Expr alpha, std::string const& name = "heat_rt")
{
	Halide::Func ret(name);
	Halide::Var x("x"), y("y");
	Halide::Type type = f.types()[0];

	Halide::Expr xp = Halide::clamp(x + 1, 0, cols - 1);
	Halide::Expr xm = Halide::clamp(x - 1, 0, cols - 1);
	Halide::Expr yp = Halide::clamp(y + 1, 0, rows - 1);
	Halide::Expr ym = Halide::clamp(y - 1, 0, rows - 1);

	// Discrete Laplacian: sum of neighbors minus 4 times center
	Halide::Expr lap = f(xm, y) + f(xp, y) + f(x, ym) + f(x, yp)
	                 - Halide::Internal::make_const(type, 4.0) * f(x, y);

	ret(x, y) = f(x, y) + Halide::cast(type, alpha) * Halide::cast(type, dt) * lap;
	return ret;
}

NS_NUM_HALIDE_END
