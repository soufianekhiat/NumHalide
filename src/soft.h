#pragma once

/// @file soft.h
/// @brief Differentiable variants of the primitives that reverse mode refuses.
///
/// Halide's propagate_adjoints cannot differentiate max/min ("Can't take the
/// gradients of maximum"), and abs is a kink. A function built out of those
/// therefore has no usable derivative, however well-defined its value is --
/// and a delegated kernel cannot fix that from the caller's side, because the
/// primitive is in HERE.
///
/// So each op has two forms, chosen by a build mode:
///   exact          -> the bit-exact Halide primitive. Production output is
///                     unchanged, bit for bit.
///   differentiable -> a smooth algebraic surrogate, so the derived adjoint
///                     and tangent exist by construction.
///
/// The smooth forms use the sqrt((a-b)^2 + e^2) family rather than log-sum-exp:
/// fully differentiable, containing no internal max (which would re-break AD),
/// and free of exp overflow. `eps` is the softness knob -- smaller is sharper
/// and closer to the hard op.
///
/// THE MODE IS AMBIENT, not a parameter. Threading a mode argument through
/// every public entry point would change the whole API and every call site;
/// the caller instead opens a scope around the pipeline construction it wants
/// smoothed. It is thread-local because kernels are built on a worker pool.
///
///   {
///       numhalide::diff_mode_scope smooth;   // this thread, until it closes
///       f = numhalide::threshold_trunc( ... );
///   }
///
/// Mirrors flow::priv::kd on the caller's side, deliberately: a kernel that
/// keeps its body local and one that delegates here should get the same
/// treatment, or delegation silently costs the kernel its derivative.

#include "numhalide.h"

NS_NUM_HALIDE_BEGIN

enum class diff_build_mode
{
	exact,		   ///< bit-identical to the Halide primitive
	differentiable ///< smooth surrogate, for AD
};

/// The current thread's build mode. Function-local so this stays header-only.
inline diff_build_mode&
current_diff_mode()
{
	static thread_local diff_build_mode mode = diff_build_mode::exact;
	return mode;
}

inline Bool
is_differentiable_build()
{
	return current_diff_mode() == diff_build_mode::differentiable;
}

/// Selects the smooth forms for as long as it is alive, on this thread only.
/// Restores the previous mode, so nesting and early returns behave.
struct diff_mode_scope
{
	diff_build_mode previous;

	diff_mode_scope()
	: previous( current_diff_mode() )
	{
		current_diff_mode() = diff_build_mode::differentiable;
	}

	explicit diff_mode_scope( diff_build_mode m )
	: previous( current_diff_mode() )
	{
		current_diff_mode() = m;
	}

	~diff_mode_scope()
	{
		current_diff_mode() = previous;
	}

	diff_mode_scope( diff_mode_scope const& )			 = delete;
	diff_mode_scope& operator=( diff_mode_scope const& ) = delete;
};

// |x|  ~  sqrt(x^2 + e^2)
inline Halide::Expr
soft_abs( Halide::Expr const& x, f32 eps = 1e-4f )
{
	if ( !is_differentiable_build() )
		return Halide::abs( x );

	return Halide::sqrt( x * x + eps * eps );
}

// max(a,b) ~ ((a+b) + sqrt((a-b)^2 + e^2)) / 2
inline Halide::Expr
soft_max( Halide::Expr const& a, Halide::Expr const& b, f32 eps = 1e-4f )
{
	if ( !is_differentiable_build() )
		return Halide::max( a, b );

	Halide::Expr const d = a - b;
	return ( ( a + b ) + Halide::sqrt( d * d + eps * eps ) ) * 0.5f;
}

// min(a,b) ~ ((a+b) - sqrt((a-b)^2 + e^2)) / 2
inline Halide::Expr
soft_min( Halide::Expr const& a, Halide::Expr const& b, f32 eps = 1e-4f )
{
	if ( !is_differentiable_build() )
		return Halide::min( a, b );

	Halide::Expr const d = a - b;
	return ( ( a + b ) - Halide::sqrt( d * d + eps * eps ) ) * 0.5f;
}

// clamp(x,lo,hi) built from the two above, so it inherits their smoothness.
inline Halide::Expr
soft_clamp( Halide::Expr const& x,
			Halide::Expr const& lo,
			Halide::Expr const& hi,
			f32					eps = 1e-4f )
{
	if ( !is_differentiable_build() )
		return Halide::clamp( x, lo, hi );

	return soft_min( soft_max( x, lo, eps ), hi, eps );
}

NS_NUM_HALIDE_END
