/// @file schedule.h
/// @brief Scheduling helpers for Halide::Func optimization
///
/// Provides: auto_tile, vectorize, parallel, compute_root, compute_inline

#pragma once

#include "common.h"
#include "shape.h"

NS_NUM_HALIDE_BEGIN

namespace schedule {

/// @brief Apply tiling to a 2D Func for cache-friendly access
/// @param f Func to tile
/// @param tx Tile size in x dimension
/// @param ty Tile size in y dimension
///
/// Usage:
///   auto_tile(my_func, 32, 32);
inline
void auto_tile(Halide::Func& f, int tx, int ty)
{
	Halide::Var x = f.args()[0];
	Halide::Var y = f.args()[1];
	Halide::Var xo("xo"), yo("yo"), xi("xi"), yi("yi");

	f.tile(x, y, xo, yo, xi, yi, tx, ty);
}

/// @brief Apply tiling and reorder for 3D Func
/// @param f Func to tile
/// @param tx Tile size in x dimension
/// @param ty Tile size in y dimension
/// @param tz Tile size in z dimension (if 0, no tiling in z)
inline
void auto_tile_3d(Halide::Func& f, int tx, int ty, int tz = 0)
{
	Halide::Var x = f.args()[0];
	Halide::Var y = f.args()[1];
	Halide::Var z = f.args()[2];
	Halide::Var xo("xo"), yo("yo"), xi("xi"), yi("yi");

	f.tile(x, y, xo, yo, xi, yi, tx, ty);

	if (tz > 0) {
		Halide::Var zo("zo"), zi("zi");
		f.split(z, zo, zi, tz);
		f.reorder(xi, yi, zi, xo, yo, zo);
	} else {
		f.reorder(xi, yi, xo, yo, z);
	}
}

/// @brief Apply vectorization to the innermost dimension
/// @param f Func to vectorize
/// @param width Vector width (typically 4, 8, or 16)
///
/// Usage:
///   vectorize(my_func, 8);
inline
void vectorize(Halide::Func& f, int width)
{
	Halide::Var x = f.args()[0];
	Halide::Var xo("xo"), xi("xi");

	f.split(x, xo, xi, width);
	f.vectorize(xi);
}

/// @brief Apply parallelization to an axis
/// @param f Func to parallelize
/// @param axis Axis index to parallelize (0 = innermost, higher = outer)
///
/// Usage:
///   parallel(my_func, 1);  // Parallelize y (outer loop)
inline
void parallel(Halide::Func& f, int axis = 1)
{
	if (axis < static_cast<int>(f.args().size())) {
		Halide::Var v = f.args()[axis];
		f.parallel(v);
	}
}

/// @brief Apply both vectorization and parallelization
/// @param f Func to optimize
/// @param vec_width Vector width
/// @param parallel_axis Axis to parallelize
///
/// Usage:
///   vectorize_parallel(my_func, 8, 1);
inline
void vectorize_parallel(Halide::Func& f, int vec_width, int parallel_axis = 1)
{
	vectorize(f, vec_width);
	parallel(f, parallel_axis);
}

/// @brief Apply tiling, vectorization, and parallelization
/// @param f Func to optimize
/// @param tx Tile size in x
/// @param ty Tile size in y
/// @param vec_width Vector width
///
/// Usage:
///   full_optimize_2d(my_func, 32, 32, 8);
inline
void full_optimize_2d(Halide::Func& f, int tx, int ty, int vec_width)
{
	Halide::Var x = f.args()[0];
	Halide::Var y = f.args()[1];
	Halide::Var xo("xo"), yo("yo"), xi("xi"), yi("yi");

	f.tile(x, y, xo, yo, xi, yi, tx, ty);
	f.vectorize(xi, vec_width);
	f.parallel(yo);
}

/// @brief Force computation at root (fully computed before use)
/// @param f Func to compute at root
inline
void compute_root(Halide::Func& f)
{
	f.compute_root();
}

/// @brief Force inline computation (no intermediate storage)
/// @param f Func to inline
inline
void compute_inline(Halide::Func& f)
{
	f.compute_inline();
}

/// @brief Apply unrolling to the innermost dimension
/// @param f Func to unroll
/// @param factor Unroll factor
inline
void unroll(Halide::Func& f, int factor)
{
	Halide::Var x = f.args()[0];
	Halide::Var xo("xo"), xi("xi");

	f.split(x, xo, xi, factor);
	f.unroll(xi);
}

/// @brief Reorder dimensions for better memory access patterns
/// @param f Func to reorder
/// @param order New dimension order (indices into original args)
///
/// Usage:
///   reorder(my_func, {1, 0});  // Swap x and y
inline
void reorder(Halide::Func& f, const std::vector<int>& order)
{
	std::vector<Halide::VarOrRVar> new_order;
	for (int idx : order) {
		new_order.push_back(f.args()[idx]);
	}
	f.reorder(new_order);
}

/// @brief Store computation results in local memory (for GPU)
/// @param f Func to store locally
/// @param inner_var Variable to store at
inline
void store_at(Halide::Func& f, Halide::Func& consumer, Halide::Var at_var)
{
	f.store_at(consumer, at_var);
}

/// @brief Compute intermediate at a specific location
/// @param f Func to compute
/// @param consumer Consumer Func
/// @param at_var Variable to compute at
inline
void compute_at(Halide::Func& f, Halide::Func& consumer, Halide::Var at_var)
{
	f.compute_at(consumer, at_var);
}

/// @brief Fuse two dimensions into one
/// @param f Func to fuse
/// @param outer Outer dimension index
/// @param inner Inner dimension index
/// @return The fused variable
inline
Halide::Var fuse(Halide::Func& f, int outer, int inner)
{
	Halide::Var outer_v = f.args()[outer];
	Halide::Var inner_v = f.args()[inner];
	Halide::Var fused("fused");
	f.fuse(outer_v, inner_v, fused);
	return fused;
}

/// @brief Get recommended vector width for the current target
/// @return Vector width in bytes / 4 (for float32)
inline
int get_vector_width()
{
	Halide::Target target = Halide::get_host_target();
	if (target.has_feature(Halide::Target::AVX512)) {
		return 16;  // 512 bits / 32 bits = 16 floats
	} else if (target.has_feature(Halide::Target::AVX2) ||
	           target.has_feature(Halide::Target::AVX)) {
		return 8;   // 256 bits / 32 bits = 8 floats
	} else if (target.has_feature(Halide::Target::SSE41)) {
		return 4;   // 128 bits / 32 bits = 4 floats
	}
	return 4;  // Default
}

/// @brief Auto-schedule a 2D Func with sensible defaults
/// @param f Func to schedule
///
/// Applies tiling (64x64), vectorization, and parallelization
inline
void auto_schedule_2d(Halide::Func& f)
{
	int vec_width = get_vector_width();
	full_optimize_2d(f, 64, 64, vec_width);
}

#ifdef HALIDE_WITH_D3D12
/// @brief Apply GPU scheduling for D3D12 backend
/// @param f Func to schedule for GPU
/// @param block_x GPU block size in x
/// @param block_y GPU block size in y
inline
void gpu_d3d12(Halide::Func& f, int block_x = 16, int block_y = 16)
{
	Halide::Var x = f.args()[0];
	Halide::Var y = f.args()[1];
	Halide::Var xo("xo"), yo("yo"), xi("xi"), yi("yi");

	f.tile(x, y, xo, yo, xi, yi, block_x, block_y);
	f.gpu_blocks(xo, yo).gpu_threads(xi, yi);
}
#endif

#ifdef HALIDE_WITH_CUDA
/// @brief Apply GPU scheduling for CUDA backend
/// @param f Func to schedule for GPU
/// @param block_x GPU block size in x
/// @param block_y GPU block size in y
inline
void gpu_cuda(Halide::Func& f, int block_x = 16, int block_y = 16)
{
	Halide::Var x = f.args()[0];
	Halide::Var y = f.args()[1];
	Halide::Var xo("xo"), yo("yo"), xi("xi"), yi("yi");

	f.tile(x, y, xo, yo, xi, yi, block_x, block_y);
	f.gpu_blocks(xo, yo).gpu_threads(xi, yi);
}
#endif

} // namespace schedule

NS_NUM_HALIDE_END
