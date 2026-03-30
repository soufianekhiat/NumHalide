/// @file autodiff_tensor.h
/// @brief Tensor-based reverse-mode automatic differentiation (pure C++20)
///
/// Provides:
///   Tensor        — N-dimensional float tensor (plain data, no autodiff)
///   TVar          — differentiable tensor variable / expression node
///   ttape_reset() — clear the thread-local tensor tape before a new computation
///   Arithmetic operators: +, -, *, /, unary-  (with scalar broadcasting)
///   Tensor ops: tmatmul, tdot, tsum, tmean, ttranspose, treshape
///   Elementwise: texp, tlog, tsin, tcos, tsqrt, tpow, ttanh, tabs, trelu, tsigmoid
///   Vector ops: tnorm, tnormalize
///   Matrix ops: ttrace, tfrobenius_sq, tfrobenius
///
/// Usage:
///   ttape_reset();
///   TVar W({ {1,2}, {3,4} });        // 2x2 matrix leaf
///   TVar x({ 1.0f, 1.0f });          // vector leaf
///   TVar y = tmatmul(W, treshape(x, {2,1}));
///   TVar loss = tsum(y);
///   loss.backward();
///   Tensor dW = W.grad();   // dloss/dW
///
/// Note: for scalar-only autodiff, use autodiff.h (DVar).
/// No Halide headers are used; this is pure C++20.

#pragma once

#include "common.h"
#include <vector>
#include <functional>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <initializer_list>

NS_NUM_HALIDE_BEGIN

// =============================================================================
// Tensor — plain N-dimensional float tensor (no autodiff)
// =============================================================================

/// @brief N-dimensional float tensor (plain data, no autodiff).
class Tensor {
    std::vector<int>   shape_;  // empty = scalar
    std::vector<float> data_;

    static int product(const std::vector<int>& v) {
        int p = 1;
        for (int x : v) p *= x;
        return p;
    }

    static int flat_index(const std::vector<int>& shape,
                          const std::vector<int>& idx) {
        int fi = 0, stride = 1;
        for (int i = (int)shape.size() - 1; i >= 0; --i) {
            fi += idx[i] * stride;
            stride *= shape[i];
        }
        return fi;
    }

public:
    /// @brief C-order (row-major) strides for the given shape.
    static std::vector<int> make_strides(const std::vector<int>& shape) {
        int nd = (int)shape.size();
        std::vector<int> st(nd, 1);
        for (int i = nd - 2; i >= 0; --i) st[i] = st[i + 1] * shape[i + 1];
        return st;
    }

    // -------------------------------------------------------------------------
    // Constructors
    // -------------------------------------------------------------------------

    /// @brief Default: scalar 0
    Tensor() : shape_{}, data_{0.0f} {}

    /// @brief Scalar from float value
    explicit Tensor(float v) : shape_{}, data_{v} {}

    /// @brief 1D vector from std::vector<float>
    explicit Tensor(std::vector<float> v)
        : shape_{ (int)v.size() }, data_(std::move(v)) {}

    /// @brief 1D from initializer_list<float>
    Tensor(std::initializer_list<float> v)
        : shape_{ (int)v.size() }, data_(v) {}

    /// @brief General ND tensor from shape + flat data
    Tensor(std::vector<int> shape, std::vector<float> data)
        : shape_(std::move(shape)), data_(std::move(data)) {
        if (!shape_.empty()) {
            int expected = product(shape_);
            if (expected != (int)data_.size())
                throw std::invalid_argument("Tensor: shape/data size mismatch");
        }
    }

    /// @brief 2D from nested initializer_list
    Tensor(std::initializer_list<std::initializer_list<float>> rows) {
        int nrows = (int)rows.size();
        int ncols = (rows.size() > 0) ? (int)rows.begin()->size() : 0;
        shape_ = { nrows, ncols };
        data_.reserve(nrows * ncols);
        for (const auto& row : rows)
            for (float v : row)
                data_.push_back(v);
    }

    // -------------------------------------------------------------------------
    // Shape / size accessors
    // -------------------------------------------------------------------------

    int rank() const { return (int)shape_.size(); }
    bool is_scalar() const { return shape_.empty(); }
    int size() const { return (int)data_.size(); }
    const std::vector<int>& shape() const { return shape_; }
    int dim(int i) const { return shape_[i]; }

    // -------------------------------------------------------------------------
    // Element access
    // -------------------------------------------------------------------------

    float& flat(int i) { return data_[i]; }
    float  flat(int i) const { return data_[i]; }

    /// @brief Extract scalar value (works for scalar tensor or single-element)
    float item() const { return data_[0]; }

    /// @brief General N-dimensional index access
    float operator()(const std::vector<int>& idx) const {
        return data_[flat_index(shape_, idx)];
    }
    float& operator()(const std::vector<int>& idx) {
        return data_[flat_index(shape_, idx)];
    }

    /// @brief 1D convenience access
    float operator()(int i) const { return data_[i]; }
    float& operator()(int i) { return data_[i]; }

    /// @brief 2D convenience access (row, col)
    float operator()(int r, int c) const { return data_[r * shape_[1] + c]; }
    float& operator()(int r, int c) { return data_[r * shape_[1] + c]; }

    /// @brief 3D convenience access (batch, row, col)
    float operator()(int b, int r, int c) const {
        return data_[b * shape_[1] * shape_[2] + r * shape_[2] + c];
    }
    float& operator()(int b, int r, int c) {
        return data_[b * shape_[1] * shape_[2] + r * shape_[2] + c];
    }

    /// @brief 4D convenience access (n, c, h, w)
    float operator()(int n, int c, int h, int w) const {
        return data_[n * shape_[1] * shape_[2] * shape_[3]
                   + c * shape_[2] * shape_[3]
                   + h * shape_[3] + w];
    }
    float& operator()(int n, int c, int h, int w) {
        return data_[n * shape_[1] * shape_[2] * shape_[3]
                   + c * shape_[2] * shape_[3]
                   + h * shape_[3] + w];
    }

    // -------------------------------------------------------------------------
    // Static factories
    // -------------------------------------------------------------------------

    static Tensor zeros(std::vector<int> shape) {
        if (shape.empty()) return Tensor(0.0f);
        return Tensor(shape, std::vector<float>(product(shape), 0.0f));
    }

    static Tensor ones(std::vector<int> shape) {
        if (shape.empty()) return Tensor(1.0f);
        return Tensor(shape, std::vector<float>(product(shape), 1.0f));
    }

    static Tensor eye(int n) {
        Tensor t = zeros({ n, n });
        for (int i = 0; i < n; ++i) t(i, i) = 1.0f;
        return t;
    }

    // -------------------------------------------------------------------------
    // Elementwise arithmetic with scalar broadcasting
    // -------------------------------------------------------------------------

    template<typename Fn>
    Tensor broadcast_op(const Tensor& b, Fn fn) const {
        if (is_scalar()) {
            // broadcast this scalar across b
            std::vector<float> out(b.data_.size());
            for (int i = 0; i < (int)b.data_.size(); ++i)
                out[i] = fn(data_[0], b.data_[i]);
            return Tensor(b.shape_, out);
        }
        if (b.is_scalar()) {
            // broadcast b scalar across this
            std::vector<float> out(data_.size());
            for (int i = 0; i < (int)data_.size(); ++i)
                out[i] = fn(data_[i], b.data_[0]);
            return Tensor(shape_, out);
        }
        // Same shape required
        std::vector<float> out(data_.size());
        for (int i = 0; i < (int)data_.size(); ++i)
            out[i] = fn(data_[i], b.data_[i]);
        return Tensor(shape_, out);
    }

    Tensor operator+(const Tensor& b) const {
        return broadcast_op(b, [](float x, float y) { return x + y; });
    }
    Tensor operator-(const Tensor& b) const {
        return broadcast_op(b, [](float x, float y) { return x - y; });
    }
    Tensor operator*(const Tensor& b) const {
        return broadcast_op(b, [](float x, float y) { return x * y; });
    }
    Tensor operator/(const Tensor& b) const {
        return broadcast_op(b, [](float x, float y) { return x / y; });
    }
    Tensor operator-() const {
        std::vector<float> out(data_.size());
        for (int i = 0; i < (int)data_.size(); ++i) out[i] = -data_[i];
        return Tensor(shape_, out);
    }

    // -------------------------------------------------------------------------
    // Reductions
    // -------------------------------------------------------------------------

    /// @brief Sum all elements → scalar Tensor
    Tensor sum_all() const {
        float s = 0.0f;
        for (float x : data_) s += x;
        return Tensor(s);
    }

    /// @brief Sum along one axis — works for any rank tensor.
    /// @param axis Axis to reduce (negative indices supported).
    /// @return Tensor with rank()-1 dimensions; scalar if rank==1.
    Tensor sum_axis(int axis) const {
        int nd = rank();
        if (axis < 0) axis += nd;

        // Build output shape (remove reduced axis)
        std::vector<int> out_shape;
        for (int i = 0; i < nd; ++i)
            if (i != axis) out_shape.push_back(shape_[i]);

        Tensor r = out_shape.empty() ? Tensor(0.0f) : zeros(out_shape);

        auto in_st  = make_strides(shape_);
        auto out_st = out_shape.empty() ? std::vector<int>{} : make_strides(out_shape);

        for (int fi = 0; fi < size(); ++fi) {
            // Decompose flat index into per-dim indices
            int tmp = fi;
            std::vector<int> midx(nd);
            for (int i = 0; i < nd; ++i) {
                midx[i] = tmp / in_st[i];
                tmp     %= in_st[i];
            }
            // Map to output flat index (skip reduced axis)
            int out_fi = 0, oi = 0;
            for (int i = 0; i < nd; ++i) {
                if (i == axis) continue;
                if (oi < (int)out_st.size()) out_fi += midx[i] * out_st[oi];
                ++oi;
            }
            r.flat(out_fi) += data_[fi];
        }
        return r;
    }

    /// @brief Mean of all elements → scalar Tensor
    Tensor mean_all() const {
        return Tensor(sum_all().item() / (float)size());
    }

    // -------------------------------------------------------------------------
    // Vector specializations (1D)
    // -------------------------------------------------------------------------

    float dot(const Tensor& b) const {
        float s = 0.0f;
        for (int i = 0; i < (int)data_.size(); ++i) s += data_[i] * b.data_[i];
        return s;
    }

    float norm() const { return std::sqrt(dot(*this)); }

    // -------------------------------------------------------------------------
    // Matrix specializations (2D)
    // -------------------------------------------------------------------------

    Tensor matmul(const Tensor& B) const {
        int m = shape_[0], k = shape_[1], n = B.shape_[1];
        Tensor C = zeros({ m, n });
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                for (int kk = 0; kk < k; ++kk)
                    C(i, j) += (*this)(i, kk) * B(kk, j);
        return C;
    }

    /// @brief Batched matrix multiply.
    /// @param B Tensor of shape {batch, K, N}
    /// @return Tensor of shape {batch, M, N} where this has shape {batch, M, K}
    Tensor batch_matmul(const Tensor& B) const {
        int batch = shape_[0], M = shape_[1], K = shape_[2], N = B.shape_[2];
        Tensor C = zeros({batch, M, N});
        for (int b = 0; b < batch; ++b)
            for (int i = 0; i < M; ++i)
                for (int j = 0; j < N; ++j)
                    for (int k = 0; k < K; ++k)
                        C(b, i, j) += (*this)(b, i, k) * B(b, k, j);
        return C;
    }

    Tensor transpose() const {
        if (rank() != 2)
            throw std::invalid_argument("Tensor::transpose: only 2D supported");
        int m = shape_[0], n = shape_[1];
        Tensor T = zeros({ n, m });
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                T(j, i) = (*this)(i, j);
        return T;
    }

    float trace() const {
        int n = std::min(shape_[0], shape_[1]);
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += (*this)(i, i);
        return s;
    }

    Tensor reshape(std::vector<int> new_shape) const {
        return Tensor(new_shape, data_);
    }
};

// Free-function scalar * Tensor convenience overloads
inline Tensor operator*(float s, const Tensor& t) { return Tensor(s) * t; }
inline Tensor operator+(float s, const Tensor& t) { return Tensor(s) + t; }
inline Tensor operator-(float s, const Tensor& t) { return Tensor(s) - t; }

// =============================================================================
// Internal tape for tensor autodiff
// =============================================================================

/// @brief A single node on the tensor tape.
struct _TTNode {
    Tensor val;                                      ///< Forward value
    mutable Tensor grad;                             ///< Accumulated gradient
    mutable bool grad_initialized = false;           ///< Has grad been set yet?
    std::function<void(const Tensor&)> backward_fn;  ///< Backward propagation function
};

/// @brief Thread-local tensor tape (Wengert list for tensors).
inline thread_local std::vector<_TTNode> _ttape;

/// @brief Reset the tensor autodiff tape. Call before each new computation.
inline void ttape_reset() { _ttape.clear(); }

// =============================================================================
// Internal gradient accumulation helper (anonymous namespace = internal linkage)
// =============================================================================

namespace {

/// @brief Accumulate gradient g into tape node node_id.
/// Sets node.grad on first call; adds elementwise on subsequent calls.
/// Handles scalar↔tensor broadcasting.
inline void _taccum(int node_id, const Tensor& g) {
    auto& node = _ttape[node_id];
    if (!node.grad_initialized) {
        node.grad = g;
        node.grad_initialized = true;
    } else {
        auto& ng = node.grad;
        if (ng.is_scalar() && g.is_scalar()) {
            ng.flat(0) += g.flat(0);
        } else if (ng.is_scalar() && !g.is_scalar()) {
            // Reduce g to scalar by summing
            float s = 0.0f;
            for (int i = 0; i < g.size(); ++i) s += g.flat(i);
            ng.flat(0) += s;
        } else if (!ng.is_scalar() && g.is_scalar()) {
            // Broadcast g scalar across all elements of ng
            for (int i = 0; i < ng.size(); ++i) ng.flat(i) += g.flat(0);
        } else {
            for (int i = 0; i < g.size(); ++i) ng.flat(i) += g.flat(i);
        }
    }
}

} // anonymous namespace

// =============================================================================
// TVar — differentiable tensor variable
// =============================================================================

class TVar {
    int id_;

    explicit TVar(int id) : id_(id) {}

public:
    // -------------------------------------------------------------------------
    // Internal: create a new node on the tape
    // -------------------------------------------------------------------------
    static TVar make(Tensor val, std::function<void(const Tensor&)> bwd) {
        _ttape.push_back(_TTNode{ std::move(val), Tensor(), false, std::move(bwd) });
        return TVar((int)_ttape.size() - 1);
    }

    // -------------------------------------------------------------------------
    // Leaf constructors (no backward function)
    // -------------------------------------------------------------------------

    /// @brief Scalar leaf
    explicit TVar(float v)
        : TVar(make(Tensor(v), nullptr)) {}

    /// @brief 1D vector leaf from std::vector<float>
    explicit TVar(std::vector<float> v)
        : TVar(make(Tensor(std::move(v)), nullptr)) {}

    /// @brief ND tensor leaf from flat data + shape
    TVar(std::vector<float> data, std::vector<int> shape)
        : TVar(make(Tensor(std::move(shape), std::move(data)), nullptr)) {}

    /// @brief 2D leaf from nested initializer_list
    TVar(std::initializer_list<std::initializer_list<float>> rows)
        : TVar(make(Tensor(rows), nullptr)) {}

    /// @brief General tensor leaf (from existing Tensor)
    explicit TVar(Tensor val)
        : TVar(make(std::move(val), nullptr)) {}

    // -------------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------------

    int id() const { return id_; }

    const Tensor& val() const { return _ttape[id_].val; }

    /// @brief Gradient of the output w.r.t. this node.
    /// Valid only after backward() has been called from a descendant node.
    Tensor grad() const {
        const auto& node = _ttape[id_];
        if (!node.grad_initialized)
            return Tensor::zeros(node.val.shape());
        return node.grad;
    }

    // -------------------------------------------------------------------------
    // Backward pass
    // -------------------------------------------------------------------------

    /// @brief Reverse-mode backward pass from this node (should be scalar).
    /// After calling, grad() returns d(this)/d(leaf) for every leaf.
    void backward() const {
        _ttape[id_].grad = Tensor(1.0f);
        _ttape[id_].grad_initialized = true;
        for (int i = id_; i >= 0; --i) {
            auto& node = _ttape[i];
            if (!node.grad_initialized) continue;
            if (node.backward_fn) node.backward_fn(node.grad);
        }
    }

    // -------------------------------------------------------------------------
    // Arithmetic operators — TVar op TVar  (with scalar broadcasting)
    // -------------------------------------------------------------------------

    TVar operator+(const TVar& b) const {
        int a_id = id_, b_id = b.id_;
        Tensor out = val() + b.val();
        return TVar::make(out, [a_id, b_id](const Tensor& g) {
            _taccum(a_id, g);
            _taccum(b_id, g);
        });
    }

    TVar operator-(const TVar& b) const {
        int a_id = id_, b_id = b.id_;
        Tensor out = val() - b.val();
        return TVar::make(out, [a_id, b_id](const Tensor& g) {
            _taccum(a_id, g);
            _taccum(b_id, -g);
        });
    }

    TVar operator*(const TVar& b) const {
        int a_id = id_, b_id = b.id_;
        Tensor a_val = val();
        Tensor b_val = b.val();
        Tensor out = a_val * b_val;
        return TVar::make(out, [a_id, b_id, a_val, b_val](const Tensor& g) {
            // d(a*b)/da = g * b,  d(a*b)/db = g * a
            _taccum(a_id, g * b_val);
            _taccum(b_id, g * a_val);
        });
    }

    TVar operator/(const TVar& b) const {
        int a_id = id_, b_id = b.id_;
        Tensor a_val = val();
        Tensor b_val = b.val();
        Tensor out = a_val / b_val;
        return TVar::make(out, [a_id, b_id, a_val, b_val](const Tensor& g) {
            // d(a/b)/da = g/b,  d(a/b)/db = -g*a/b^2
            _taccum(a_id, g / b_val);
            _taccum(b_id, -(g * a_val) / (b_val * b_val));
        });
    }

    TVar operator-() const {
        int a_id = id_;
        Tensor out = -val();
        return TVar::make(out, [a_id](const Tensor& g) {
            _taccum(a_id, -g);
        });
    }

    // -------------------------------------------------------------------------
    // Arithmetic operators — TVar op float convenience overloads
    // -------------------------------------------------------------------------

    TVar operator+(float s) const { return *this + TVar(Tensor(s)); }
    TVar operator-(float s) const { return *this - TVar(Tensor(s)); }
    TVar operator*(float s) const { return *this * TVar(Tensor(s)); }
    TVar operator/(float s) const { return *this / TVar(Tensor(s)); }
};

// Allow float op TVar (symmetric convenience overloads)
inline TVar operator+(float s, const TVar& t) { return TVar(Tensor(s)) + t; }
inline TVar operator-(float s, const TVar& t) { return TVar(Tensor(s)) - t; }
inline TVar operator*(float s, const TVar& t) { return TVar(Tensor(s)) * t; }
inline TVar operator/(float s, const TVar& t) { return TVar(Tensor(s)) / t; }

// =============================================================================
// Free-function tensor operations
// =============================================================================

// -------------------------------------------------------------------------
// Matrix operations
// -------------------------------------------------------------------------

/// @brief Matrix multiply A(m,k) @ B(k,n) → C(m,n)
/// Backward: dA += g @ B.T,  dB += A.T @ g
inline TVar tmatmul(const TVar& A, const TVar& B) {
    int a_id = A.id(), b_id = B.id();
    Tensor a_val = A.val();
    Tensor b_val = B.val();
    Tensor out = a_val.matmul(b_val);
    return TVar::make(out, [a_id, b_id, a_val, b_val](const Tensor& g) {
        _taccum(a_id, g.matmul(b_val.transpose()));
        _taccum(b_id, a_val.transpose().matmul(g));
    });
}

/// @brief Transpose a 2D matrix.
/// Backward: dA = g.transpose()
inline TVar ttranspose(const TVar& A) {
    int a_id = A.id();
    Tensor out = A.val().transpose();
    return TVar::make(out, [a_id](const Tensor& g) {
        _taccum(a_id, g.transpose());
    });
}

/// @brief Trace of a square (or rectangular) matrix → scalar TVar.
/// Backward: dA[i,i] += g_scalar
inline TVar ttrace(const TVar& A) {
    int a_id = A.id();
    std::vector<int> a_shape = A.val().shape();
    int n = std::min(a_shape[0], a_shape[1]);
    float tr = A.val().trace();
    return TVar::make(Tensor(tr), [a_id, a_shape, n](const Tensor& g) {
        Tensor dA = Tensor::zeros(a_shape);
        float gs = g.item();
        for (int i = 0; i < n; ++i) dA(i, i) = gs;
        _taccum(a_id, dA);
    });
}

// -------------------------------------------------------------------------
// Reduction operations
// -------------------------------------------------------------------------

/// @brief Sum all elements of a tensor → scalar TVar.
/// Backward: d(input)[i] = g (broadcast g to all elements)
inline TVar tsum(const TVar& a) {
    int a_id = a.id();
    std::vector<int> a_shape = a.val().shape();
    int a_sz = a.val().size();
    Tensor out = a.val().sum_all();
    return TVar::make(out, [a_id, a_shape, a_sz](const Tensor& g) {
        float gs = g.item();
        std::vector<float> da_data(a_sz, gs);
        _taccum(a_id, Tensor(a_shape, da_data));
    });
}

/// @brief Mean of all elements → scalar TVar.
/// Backward: d(input)[i] = g / n
inline TVar tmean(const TVar& a) {
    int a_id = a.id();
    std::vector<int> a_shape = a.val().shape();
    int n = a.val().size();
    float mean_val = a.val().mean_all().item();
    return TVar::make(Tensor(mean_val), [a_id, a_shape, n](const Tensor& g) {
        float gs = g.item() / (float)n;
        std::vector<float> da_data(n, gs);
        _taccum(a_id, Tensor(a_shape, da_data));
    });
}

/// @brief Reshape a tensor (no-op on data, changes shape).
/// Backward: grad reshaped back to original shape
inline TVar treshape(const TVar& a, std::vector<int> new_shape) {
    int a_id = a.id();
    std::vector<int> orig_shape = a.val().shape();
    Tensor out = a.val().reshape(new_shape);
    return TVar::make(out, [a_id, orig_shape](const Tensor& g) {
        _taccum(a_id, g.reshape(orig_shape));
    });
}

/// @brief Sum along one axis of any-rank tensor.
/// @param a    Input TVar
/// @param axis Axis to reduce (negative indices supported)
/// Backward: broadcast upstream gradient back along the reduced axis.
inline TVar tsum(const TVar& a, int axis) {
    int a_id = a.id();
    std::vector<int> orig_shape = a.val().shape();
    int nd = (int)orig_shape.size();
    if (axis < 0) axis += nd;

    Tensor out = a.val().sum_axis(axis);

    // Build output shape (for use inside backward)
    std::vector<int> out_shape;
    for (int i = 0; i < nd; ++i)
        if (i != axis) out_shape.push_back(orig_shape[i]);

    return TVar::make(out, [a_id, axis, orig_shape, out_shape](const Tensor& g) {
        // Expand g back to orig_shape: each element in input gets gradient
        // equal to the corresponding element in g (with axis dim re-inserted).
        Tensor da = Tensor::zeros(orig_shape);
        auto in_st  = Tensor::make_strides(orig_shape);
        auto out_st = out_shape.empty() ? std::vector<int>{} : Tensor::make_strides(out_shape);

        for (int fi = 0; fi < da.size(); ++fi) {
            int tmp = fi;
            int nd_ = (int)orig_shape.size();
            std::vector<int> midx(nd_);
            for (int i = 0; i < nd_; ++i) { midx[i] = tmp / in_st[i]; tmp %= in_st[i]; }
            int out_fi = 0, oi = 0;
            for (int i = 0; i < nd_; ++i) {
                if (i == axis) continue;
                if (oi < (int)out_st.size()) out_fi += midx[i] * out_st[oi];
                ++oi;
            }
            da.flat(fi) = out_shape.empty() ? g.item() : g.flat(out_fi);
        }
        _taccum(a_id, da);
    });
}

/// @brief Batched matrix multiply: A[batch,M,K] @ B[batch,K,N] → C[batch,M,N].
/// Backward:
///   dA[b,i,k] = sum_j dC[b,i,j] * B[b,k,j]   (dC @ B^T per batch)
///   dB[b,k,j] = sum_i dC[b,i,j] * A[b,i,k]   (A^T @ dC per batch)
inline TVar tbatch_matmul(const TVar& A, const TVar& B) {
    int a_id = A.id(), b_id = B.id();
    const Tensor& av = A.val(), &bv = B.val();
    int batch = av.shape()[0], M = av.shape()[1], K = av.shape()[2], N = bv.shape()[2];

    Tensor out = av.batch_matmul(bv);

    return TVar::make(out, [a_id, b_id, batch, M, K, N](const Tensor& g) {
        const Tensor& av_ = _ttape[a_id].val;
        const Tensor& bv_ = _ttape[b_id].val;

        // dA
        Tensor da = Tensor::zeros({batch, M, K});
        for (int b = 0; b < batch; ++b)
            for (int i = 0; i < M; ++i)
                for (int k = 0; k < K; ++k)
                    for (int j = 0; j < N; ++j)
                        da(b, i, k) += g(b, i, j) * bv_(b, k, j);
        _taccum(a_id, da);

        // dB
        Tensor db = Tensor::zeros({batch, K, N});
        for (int b = 0; b < batch; ++b)
            for (int k = 0; k < K; ++k)
                for (int j = 0; j < N; ++j)
                    for (int i = 0; i < M; ++i)
                        db(b, k, j) += g(b, i, j) * av_(b, i, k);
        _taccum(b_id, db);
    });
}

// -------------------------------------------------------------------------
// Vector (1D) dot product
// -------------------------------------------------------------------------

/// @brief Dot product of two 1D TVar tensors → scalar TVar.
/// Backward: da_i = g * b_i,  db_i = g * a_i
inline TVar tdot(const TVar& a, const TVar& b) {
    int a_id = a.id(), b_id = b.id();
    Tensor a_val = a.val();
    Tensor b_val = b.val();
    float dot_val = a_val.dot(b_val);
    return TVar::make(Tensor(dot_val), [a_id, b_id, a_val, b_val](const Tensor& g) {
        float gs = g.item();
        // da = g * b_val  (elementwise scale)
        std::vector<float> da_data(b_val.size());
        for (int i = 0; i < b_val.size(); ++i) da_data[i] = gs * b_val.flat(i);
        _taccum(a_id, Tensor(b_val.shape(), da_data));
        // db = g * a_val
        std::vector<float> db_data(a_val.size());
        for (int i = 0; i < a_val.size(); ++i) db_data[i] = gs * a_val.flat(i);
        _taccum(b_id, Tensor(a_val.shape(), db_data));
    });
}

// -------------------------------------------------------------------------
// Elementwise math functions
// -------------------------------------------------------------------------

/// @brief exp(x) elementwise — backward: g * exp(x)
inline TVar texp(const TVar& a) {
    int a_id = a.id();
    Tensor a_val = a.val();
    std::vector<float> out_data(a_val.size());
    for (int i = 0; i < a_val.size(); ++i) out_data[i] = std::exp(a_val.flat(i));
    Tensor out_val(a_val.shape(), out_data);
    return TVar::make(out_val, [a_id, out_val](const Tensor& g) {
        // d exp(x)/dx = exp(x) = out_val
        _taccum(a_id, g * out_val);
    });
}

/// @brief log(x) elementwise — backward: g / x
inline TVar tlog(const TVar& a) {
    int a_id = a.id();
    Tensor a_val = a.val();
    std::vector<float> out_data(a_val.size());
    for (int i = 0; i < a_val.size(); ++i) out_data[i] = std::log(a_val.flat(i));
    Tensor out_val(a_val.shape(), out_data);
    return TVar::make(out_val, [a_id, a_val](const Tensor& g) {
        _taccum(a_id, g / a_val);
    });
}

/// @brief sin(x) elementwise — backward: g * cos(x)
inline TVar tsin(const TVar& a) {
    int a_id = a.id();
    Tensor a_val = a.val();
    std::vector<float> out_data(a_val.size());
    std::vector<float> cos_data(a_val.size());
    for (int i = 0; i < a_val.size(); ++i) {
        out_data[i] = std::sin(a_val.flat(i));
        cos_data[i] = std::cos(a_val.flat(i));
    }
    Tensor out_val(a_val.shape(), out_data);
    Tensor cos_val(a_val.shape(), cos_data);
    return TVar::make(out_val, [a_id, cos_val](const Tensor& g) {
        _taccum(a_id, g * cos_val);
    });
}

/// @brief cos(x) elementwise — backward: g * (-sin(x))
inline TVar tcos(const TVar& a) {
    int a_id = a.id();
    Tensor a_val = a.val();
    std::vector<float> out_data(a_val.size());
    std::vector<float> neg_sin_data(a_val.size());
    for (int i = 0; i < a_val.size(); ++i) {
        out_data[i] = std::cos(a_val.flat(i));
        neg_sin_data[i] = -std::sin(a_val.flat(i));
    }
    Tensor out_val(a_val.shape(), out_data);
    Tensor neg_sin_val(a_val.shape(), neg_sin_data);
    return TVar::make(out_val, [a_id, neg_sin_val](const Tensor& g) {
        _taccum(a_id, g * neg_sin_val);
    });
}

/// @brief sqrt(x) elementwise — backward: g / (2 * sqrt(x))
inline TVar tsqrt(const TVar& a) {
    int a_id = a.id();
    Tensor a_val = a.val();
    std::vector<float> out_data(a_val.size());
    for (int i = 0; i < a_val.size(); ++i) out_data[i] = std::sqrt(a_val.flat(i));
    Tensor out_val(a_val.shape(), out_data);
    return TVar::make(out_val, [a_id, out_val](const Tensor& g) {
        // d sqrt(x)/dx = 1 / (2*sqrt(x)) = 1 / (2 * out)
        std::vector<float> deriv_data(out_val.size());
        for (int i = 0; i < out_val.size(); ++i) {
            float sv = out_val.flat(i);
            deriv_data[i] = (sv > 0.0f) ? (0.5f / sv) : 0.0f;
        }
        Tensor deriv(out_val.shape(), deriv_data);
        _taccum(a_id, g * deriv);
    });
}

/// @brief x^n for constant float n — backward: g * n * x^(n-1)
inline TVar tpow(const TVar& a, float n) {
    int a_id = a.id();
    Tensor a_val = a.val();
    std::vector<float> out_data(a_val.size());
    std::vector<float> deriv_data(a_val.size());
    for (int i = 0; i < a_val.size(); ++i) {
        float x = a_val.flat(i);
        out_data[i] = std::pow(x, n);
        deriv_data[i] = n * std::pow(x, n - 1.0f);
    }
    Tensor out_val(a_val.shape(), out_data);
    Tensor deriv_val(a_val.shape(), deriv_data);
    return TVar::make(out_val, [a_id, deriv_val](const Tensor& g) {
        _taccum(a_id, g * deriv_val);
    });
}

/// @brief tanh(x) elementwise — backward: g * (1 - tanh^2(x))
inline TVar ttanh(const TVar& a) {
    int a_id = a.id();
    Tensor a_val = a.val();
    std::vector<float> out_data(a_val.size());
    for (int i = 0; i < a_val.size(); ++i) out_data[i] = std::tanh(a_val.flat(i));
    Tensor out_val(a_val.shape(), out_data);
    return TVar::make(out_val, [a_id, out_val](const Tensor& g) {
        // d tanh(x)/dx = 1 - tanh^2(x)
        std::vector<float> deriv_data(out_val.size());
        for (int i = 0; i < out_val.size(); ++i) {
            float tv = out_val.flat(i);
            deriv_data[i] = 1.0f - tv * tv;
        }
        Tensor deriv(out_val.shape(), deriv_data);
        _taccum(a_id, g * deriv);
    });
}

/// @brief |x| elementwise — backward: g * sign(x); 0 at x=0
inline TVar tabs(const TVar& a) {
    int a_id = a.id();
    Tensor a_val = a.val();
    std::vector<float> out_data(a_val.size());
    std::vector<float> sign_data(a_val.size());
    for (int i = 0; i < a_val.size(); ++i) {
        float x = a_val.flat(i);
        out_data[i] = std::abs(x);
        sign_data[i] = (x > 0.0f) ? 1.0f : (x < 0.0f) ? -1.0f : 0.0f;
    }
    Tensor out_val(a_val.shape(), out_data);
    Tensor sign_val(a_val.shape(), sign_data);
    return TVar::make(out_val, [a_id, sign_val](const Tensor& g) {
        _taccum(a_id, g * sign_val);
    });
}

/// @brief relu(x) = max(x, 0) elementwise — backward: g * (x > 0 ? 1 : 0)
inline TVar trelu(const TVar& a) {
    int a_id = a.id();
    Tensor a_val = a.val();
    std::vector<float> out_data(a_val.size());
    std::vector<float> mask_data(a_val.size());
    for (int i = 0; i < a_val.size(); ++i) {
        float x = a_val.flat(i);
        out_data[i] = (x > 0.0f) ? x : 0.0f;
        mask_data[i] = (x > 0.0f) ? 1.0f : 0.0f;
    }
    Tensor out_val(a_val.shape(), out_data);
    Tensor mask_val(a_val.shape(), mask_data);
    return TVar::make(out_val, [a_id, mask_val](const Tensor& g) {
        _taccum(a_id, g * mask_val);
    });
}

/// @brief sigmoid(x) = 1 / (1 + exp(-x)) elementwise
/// Backward: g * sigmoid(x) * (1 - sigmoid(x))
inline TVar tsigmoid(const TVar& a) {
    int a_id = a.id();
    Tensor a_val = a.val();
    std::vector<float> out_data(a_val.size());
    for (int i = 0; i < a_val.size(); ++i) {
        float x = a_val.flat(i);
        out_data[i] = 1.0f / (1.0f + std::exp(-x));
    }
    Tensor out_val(a_val.shape(), out_data);
    return TVar::make(out_val, [a_id, out_val](const Tensor& g) {
        // d sigmoid/dx = sigmoid * (1 - sigmoid)
        std::vector<float> deriv_data(out_val.size());
        for (int i = 0; i < out_val.size(); ++i) {
            float sv = out_val.flat(i);
            deriv_data[i] = sv * (1.0f - sv);
        }
        Tensor deriv(out_val.shape(), deriv_data);
        _taccum(a_id, g * deriv);
    });
}

// -------------------------------------------------------------------------
// Vector specialization helpers (built from primitives)
// -------------------------------------------------------------------------

/// @brief L2 norm of a tensor: sqrt(sum(x_i^2))
inline TVar tnorm(const TVar& a) {
    return tsqrt(tsum(a * a));
}

/// @brief L2 normalize a vector: x / ||x||
inline TVar tnormalize(const TVar& a) {
    return a / tnorm(a);
}

// -------------------------------------------------------------------------
// Matrix specialization helpers
// -------------------------------------------------------------------------

/// @brief Frobenius norm squared: sum of squared elements
inline TVar tfrobenius_sq(const TVar& A) {
    return tsum(A * A);
}

/// @brief Frobenius norm: sqrt(sum of squared elements)
inline TVar tfrobenius(const TVar& A) {
    return tsqrt(tfrobenius_sq(A));
}

NS_NUM_HALIDE_END
