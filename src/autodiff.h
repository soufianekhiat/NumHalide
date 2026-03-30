/// @file autodiff.h
/// @brief Tape-based reverse-mode automatic differentiation (pure C++)
///
/// Provides:
///   DVar       — differentiable variable / expression node
///   dtape_reset() — clear the global thread-local tape before a new computation
///   Arithmetic operators: +, -, *, /, unary-
///   Math functions: dexp, dlog, dsin, dcos, dsqrt, dpow, dtanh, dabs
///
/// Usage:
///   dtape_reset();
///   DVar x(3.0f), y(4.0f);
///   DVar z = x * x + y;
///   z.backward();          // populates gradients for all nodes
///   float dx = x.grad();   // dz/dx
///   float dy = y.grad();   // dz/dy
///
/// No Halide headers are used; this is pure C++.

#pragma once

#include <vector>
#include <cmath>
#include <stdexcept>
#include <utility>

NS_NUM_HALIDE_BEGIN

// =============================================================================
// Internal tape storage
// =============================================================================

/// @brief A single node on the Wengert tape.
struct _DNode {
    float val;  ///< Computed value at this node
    /// Parent edges: (parent_node_index, local_partial d_this/d_parent)
    std::vector<std::pair<int, float>> parents;
};

/// Thread-local tape (Wengert list)
inline thread_local std::vector<_DNode> _dtape;
/// Thread-local gradient accumulation array (populated by backward())
inline thread_local std::vector<float>  _dgrads;

/// @brief Reset the tape — call before starting a new computation graph.
inline void dtape_reset() {
    _dtape.clear();
    _dgrads.clear();
}

// Note: for tensor (vector/matrix) autodiff, use autodiff_tensor.h (TVar, Tensor, tmatmul, etc.)

// =============================================================================
// DVar — differentiable variable
// =============================================================================

class DVar {
public:
    int id_;  ///< Index into _dtape

    /// @brief Create a leaf node with value v.
    explicit DVar(float v) {
        _DNode node;
        node.val = v;
        id_ = static_cast<int>(_dtape.size());
        _dtape.push_back(std::move(node));
    }

    /// @brief The computed value at this node.
    float val() const { return _dtape[id_].val; }

    /// @brief Gradient d(output) / d(this).  Valid only after backward() has been called.
    float grad() const {
        if (id_ < static_cast<int>(_dgrads.size()))
            return _dgrads[id_];
        return 0.0f;
    }

    /// @brief Reverse-mode backward pass from this node.
    /// After calling, grad() returns dthis/d(leaf) for every leaf.
    void backward() const {
        _dgrads.assign(_dtape.size(), 0.0f);
        _dgrads[id_] = 1.0f;
        for (int i = id_; i >= 0; --i) {
            if (_dgrads[i] == 0.0f) continue;
            for (auto& [pid, pd] : _dtape[i].parents) {
                _dgrads[pid] += _dgrads[i] * pd;
            }
        }
    }

    // -------------------------------------------------------------------------
    // Arithmetic operators — DVar op DVar
    // -------------------------------------------------------------------------

    DVar operator+(const DVar& b) const {
        DVar out(val() + b.val());
        _dtape[out.id_].parents = {{id_, 1.0f}, {b.id_, 1.0f}};
        return out;
    }
    DVar operator-(const DVar& b) const {
        DVar out(val() - b.val());
        _dtape[out.id_].parents = {{id_, 1.0f}, {b.id_, -1.0f}};
        return out;
    }
    DVar operator*(const DVar& b) const {
        DVar out(val() * b.val());
        _dtape[out.id_].parents = {{id_, b.val()}, {b.id_, val()}};
        return out;
    }
    DVar operator/(const DVar& b) const {
        float bv = b.val();
        if (bv == 0.0f) throw std::domain_error("DVar: division by zero");
        float av = val();
        DVar out(av / bv);
        // d(a/b)/da = 1/b,  d(a/b)/db = -a/b²
        _dtape[out.id_].parents = {{id_, 1.0f / bv},
                                   {b.id_, -av / (bv * bv)}};
        return out;
    }
    DVar operator-() const {
        DVar out(-val());
        _dtape[out.id_].parents = {{id_, -1.0f}};
        return out;
    }

    // -------------------------------------------------------------------------
    // Arithmetic operators — DVar op float (convenience)
    // -------------------------------------------------------------------------

    DVar operator+(float b) const { return *this + DVar(b); }
    DVar operator-(float b) const { return *this - DVar(b); }
    DVar operator*(float b) const { return *this * DVar(b); }
    DVar operator/(float b) const { return *this / DVar(b); }

    // -------------------------------------------------------------------------
    // Symmetric float op DVar (defined as free functions below)
    // -------------------------------------------------------------------------
};

// Allow float op DVar
inline DVar operator+(float a, const DVar& b) { return DVar(a) + b; }
inline DVar operator-(float a, const DVar& b) { return DVar(a) - b; }
inline DVar operator*(float a, const DVar& b) { return DVar(a) * b; }
inline DVar operator/(float a, const DVar& b) { return DVar(a) / b; }

// =============================================================================
// Math functions
// =============================================================================

/// @brief exp(x) — derivative: exp(x)
inline DVar dexp(const DVar& x) {
    float v = std::exp(x.val());
    DVar out(v);
    _dtape[out.id_].parents = {{x.id_, v}};
    return out;
}

/// @brief log(x) — derivative: 1/x
inline DVar dlog(const DVar& x) {
    if (x.val() <= 0.0f) throw std::domain_error("DVar: log of non-positive value");
    DVar out(std::log(x.val()));
    _dtape[out.id_].parents = {{x.id_, 1.0f / x.val()}};
    return out;
}

/// @brief sin(x) — derivative: cos(x)
inline DVar dsin(const DVar& x) {
    DVar out(std::sin(x.val()));
    _dtape[out.id_].parents = {{x.id_, std::cos(x.val())}};
    return out;
}

/// @brief cos(x) — derivative: -sin(x)
inline DVar dcos(const DVar& x) {
    DVar out(std::cos(x.val()));
    _dtape[out.id_].parents = {{x.id_, -std::sin(x.val())}};
    return out;
}

/// @brief sqrt(x) — derivative: 1 / (2*sqrt(x))
inline DVar dsqrt(const DVar& x) {
    if (x.val() < 0.0f) throw std::domain_error("DVar: sqrt of negative value");
    float v = std::sqrt(x.val());
    DVar out(v);
    float deriv = (v > 0.0f) ? (0.5f / v) : 0.0f;
    _dtape[out.id_].parents = {{x.id_, deriv}};
    return out;
}

/// @brief x^n for constant n — derivative: n * x^(n-1)
inline DVar dpow(const DVar& x, float n) {
    float v = std::pow(x.val(), n);
    DVar out(v);
    float deriv = n * std::pow(x.val(), n - 1.0f);
    _dtape[out.id_].parents = {{x.id_, deriv}};
    return out;
}

/// @brief tanh(x) — derivative: 1 - tanh²(x)
inline DVar dtanh(const DVar& x) {
    float v = std::tanh(x.val());
    DVar out(v);
    _dtape[out.id_].parents = {{x.id_, 1.0f - v * v}};
    return out;
}

/// @brief |x| — derivative: sign(x); 0 at x=0
inline DVar dabs(const DVar& x) {
    float v = std::abs(x.val());
    DVar out(v);
    float deriv = (x.val() > 0.0f) ? 1.0f : (x.val() < 0.0f) ? -1.0f : 0.0f;
    _dtape[out.id_].parents = {{x.id_, deriv}};
    return out;
}

NS_NUM_HALIDE_END
