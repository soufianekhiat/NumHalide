/// @file soft_sort.h
/// @brief Differentiable (soft) sorting and ranking operations
///
/// Provides: soft_rank, soft_sort, soft_argsort
///
/// These implement smooth approximations to hard sorting via sigmoid-based
/// pairwise comparisons and Gaussian position assignment.  The temperature
/// parameter tau controls the sharpness: tau→0 recovers hard sorting.

#pragma once

#include "common.h"
#include "shape.h"

NS_NUM_HALIDE_BEGIN

/// @brief Differentiable soft rank of a 1D array (ascending order)
/// @param f    Input 1D Func
/// @param n    Number of elements
/// @param tau  Temperature: smaller → harder, approaches hard argsort
/// @param name Func name
/// @return 1D Func where result(i) ≈ position of element i in sorted order
///
/// soft_rank[i] = Σ_j sigmoid((f(i) − f(j)) / tau)
/// Range: [~0.5, ~n−0.5].  Minimum element → ~0.5, maximum → ~n−0.5.
inline Halide::Func soft_rank(
    Halide::Func f, int n, float tau = 1.0f,
    std::string const& name = "soft_rank")
{
    Halide::Var  i("i");
    Halide::RDom rj(0, n, "rj_sr");

    Halide::Func ret(name);
    ret(i) = 0.0f;
    // sigmoid((f(i) - f(j)) / tau) = 1 / (1 + exp(-(f(i)-f(j))/tau))
    ret(i) += 1.0f / (1.0f + Halide::exp(-(f(i) - f(rj)) / tau));
    ret.compute_root();
    return ret;
}

/// @brief Differentiable soft sort of a 1D array (ascending)
/// @param f      Input 1D Func
/// @param n      Number of elements
/// @param tau    Temperature for soft rank  (smaller = closer to hard sort)
/// @param spread Gaussian spread for position assignment (smaller = sharper)
/// @param name   Func name
/// @return 1D Func where result(p) is the soft-sorted value at position p
///
/// Algorithm:
///   1. sr[i]  = soft_rank[i]                              (real-valued rank)
///   2. w[p,i] = exp(-(sr[i] − (p+0.5))² / (2·spread²))  (Gaussian kernel)
///   3. normalise column-wise: w_norm[p,i] = w[p,i] / Σ_k w[p,k]
///   4. out[p] = Σ_i w_norm[p,i] · f[i]
inline Halide::Func soft_sort(
    Halide::Func f, int n, float tau = 1.0f, float spread = 1.0f,
    std::string const& name = "soft_sort")
{
    Halide::Func sr = soft_rank(f, n, tau, name + "_sr");

    Halide::Var p("p"), i("i");
    float two_s2 = 2.0f * spread * spread;

    // Weight matrix: w(p, i) where p = output position, i = input element
    Halide::Func w(name + "_w");
    {
        Halide::Expr diff = sr(i) - (Halide::cast<float>(p) + 0.5f);
        w(p, i) = Halide::exp(-diff * diff / two_s2);
    }
    w.compute_root();

    // Column sums for normalisation (one sum per output position)
    Halide::RDom ri1(0, n, "ri1_ss");
    Halide::Func wsum(name + "_wsum");
    wsum(p) = 0.0f;
    wsum(p) += w(p, ri1);
    wsum.compute_root();

    // Weighted output
    Halide::RDom ri2(0, n, "ri2_wo");
    Halide::Func ret(name);
    ret(p) = 0.0f;
    ret(p) += (w(p, ri2) / wsum(p)) * f(ri2);
    return ret;
}

/// @brief Soft permutation matrix: P[p,i] = weight of element i at sorted position p
/// @param f      Input 1D Func
/// @param n      Number of elements
/// @param tau    Temperature for soft rank
/// @param spread Gaussian spread for position assignment
/// @param name   Func name
/// @return 2D Func P(p, i) in [0,1]; rows sum to 1 (column-normalised)
///
/// Usage: soft_sorted_value(p) = Σ_i P(p,i) * f(i)
inline Halide::Func soft_argsort(
    Halide::Func f, int n, float tau = 1.0f, float spread = 1.0f,
    std::string const& name = "soft_argsort")
{
    Halide::Func sr = soft_rank(f, n, tau, name + "_sr");

    Halide::Var p("p"), i("i");
    float two_s2 = 2.0f * spread * spread;

    Halide::Func w(name + "_w");
    {
        Halide::Expr diff = sr(i) - (Halide::cast<float>(p) + 0.5f);
        w(p, i) = Halide::exp(-diff * diff / two_s2);
    }
    w.compute_root();

    Halide::RDom ri(0, n, "ri_sa");
    Halide::Func wsum(name + "_wsum");
    wsum(p) = 0.0f;
    wsum(p) += w(p, ri);
    wsum.compute_root();

    Halide::Func ret(name);
    ret(p, i) = w(p, i) / wsum(p);
    return ret;
}

NS_NUM_HALIDE_END
