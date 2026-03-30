/// @file view.h
/// @brief Zero-copy strided views of Halide::Runtime::Buffer<float>
///
/// All operations are O(1) — they adjust stride metadata without copying data.
/// Provides: view_transpose, view_slice, view_reshape

#pragma once
#include "common.h"
#include <vector>
#include <stdexcept>

NS_NUM_HALIDE_BEGIN

/// @brief Transpose a 2D buffer: returns view with x and y axes swapped.
/// O(1) — no data copy. The view shares the same host pointer.
/// @param src  Source 2D float buffer
/// @return Transposed view: view(x,y) = src(y,x)
inline Halide::Runtime::Buffer<float> view_transpose(
    const Halide::Runtime::Buffer<float>& src)
{
    nh_require(src.dimensions() == 2,
               "view_transpose requires 2D buffer, got %d dims", src.dimensions());

    halide_dimension_t dims[2];
    // New dim 0 (x) = old dim 1 (y): swap extents and strides
    dims[0].min    = 0;
    dims[0].extent = src.dim(1).extent();
    dims[0].stride = src.dim(1).stride();
    dims[0].flags  = 0;
    // New dim 1 (y) = old dim 0 (x)
    dims[1].min    = 0;
    dims[1].extent = src.dim(0).extent();
    dims[1].stride = src.dim(0).stride();
    dims[1].flags  = 0;

    return Halide::Runtime::Buffer<float>(src.data(), 2, dims);
}

/// @brief Slice a buffer along one axis (O(1), no data copy).
/// @param src   Source buffer (any dimensionality)
/// @param axis  Dimension to slice (0-based)
/// @param start First index (inclusive)
/// @param stop  Last index (exclusive)
/// @return Sliced view: same data pointer + start * stride[axis], reduced extent
inline Halide::Runtime::Buffer<float> view_slice(
    const Halide::Runtime::Buffer<float>& src, int axis, int start, int stop)
{
    int ndim = src.dimensions();
    nh_require(axis >= 0 && axis < ndim,
               "view_slice: axis %d out of [0, %d)", axis, ndim);
    nh_require(start >= 0 && stop <= src.dim(axis).extent() && start < stop,
               "view_slice: [%d, %d) invalid for axis %d extent %d",
               start, stop, axis, src.dim(axis).extent());

    // Offset data pointer by start * stride[axis]
    float* new_data = src.data() + (ptrdiff_t)start * src.dim(axis).stride();

    // Copy all dimension descriptors, then update the sliced axis
    std::vector<halide_dimension_t> dims(ndim);
    for (int i = 0; i < ndim; ++i) {
        dims[i].min    = 0;
        dims[i].extent = src.dim(i).extent();
        dims[i].stride = src.dim(i).stride();
        dims[i].flags  = 0;
    }
    dims[axis].extent = stop - start;

    return Halide::Runtime::Buffer<float>(new_data, ndim, dims.data());
}

/// @brief Reshape a contiguous (row-major) buffer to a new shape (O(1), no data copy).
/// @param buf         Source buffer (must be row-major / contiguous)
/// @param new_extents New dimensions (innermost first: x, y, z, ...)
/// @return Reshaped view with strides recomputed for new_extents
///
/// Requires: product(new_extents) == total elements of buf.
/// The source buffer must be contiguous (stride[0]=1, stride[i]=prod(extent[0..i-1])).
inline Halide::Runtime::Buffer<float> view_reshape(
    const Halide::Runtime::Buffer<float>& src,
    const std::vector<int>& new_extents)
{
    int old_total = 1;
    for (int d = 0; d < src.dimensions(); ++d) old_total *= src.dim(d).extent();

    int new_total = 1;
    for (int e : new_extents) new_total *= e;

    nh_require(old_total == new_total,
               "view_reshape: element count mismatch %d vs %d", old_total, new_total);

    int ndim = (int)new_extents.size();
    std::vector<halide_dimension_t> dims(ndim);
    int stride = 1;
    for (int i = 0; i < ndim; ++i) {
        dims[i].min    = 0;
        dims[i].extent = new_extents[i];
        dims[i].stride = stride;
        dims[i].flags  = 0;
        stride *= new_extents[i];
    }

    return Halide::Runtime::Buffer<float>(src.data(), ndim, dims.data());
}

// =============================================================================
// Bridge: Runtime::Buffer -> Func
// =============================================================================

/// @brief Wrap a Halide::Runtime::Buffer<float> in a Func.
/// The buffer must outlive all JIT compilations that use the returned Func.
/// @param src  Source buffer (any dimensionality 1-3D)
/// @param name Func name
/// @return Func that reads from src; handles 1D, 2D, and 3D buffers.
inline Halide::Func func_from_buffer(const Halide::Runtime::Buffer<float>& src,
                                      const std::string& name = "buf_func")
{
    int ndim = src.dimensions();
    nh_require(ndim >= 1 && ndim <= 3,
        "func_from_buffer: only 1-3D buffers supported, got %d dims", ndim);

    // Build a Halide::Buffer (non-Runtime) sharing the same host pointer
    // so that Halide can use it as an image argument.
    Halide::Func f(name);
    if (ndim == 1) {
        int n = src.dim(0).extent();
        Halide::Buffer<float> hbuf(const_cast<float*>(src.data()), n);
        Halide::Var x;
        f(x) = hbuf(x);
    } else if (ndim == 2) {
        int w = src.dim(0).extent(), h = src.dim(1).extent();
        Halide::Buffer<float> hbuf(const_cast<float*>(src.data()), w, h);
        Halide::Var x, y;
        f(x, y) = hbuf(x, y);
    } else {
        int w = src.dim(0).extent(), h = src.dim(1).extent(), d = src.dim(2).extent();
        Halide::Buffer<float> hbuf(const_cast<float*>(src.data()), w, h, d);
        Halide::Var x, y, z;
        f(x, y, z) = hbuf(x, y, z);
    }
    return f;
}

/// @brief Compute the shape_t for a Runtime::Buffer.
inline shape_t shape_from_buffer(const Halide::Runtime::Buffer<float>& src)
{
    shape_t s;
    s.rank = src.dimensions();
    for (int i = 0; i < s.rank; ++i)
        s.extents[i] = src.dim(i).extent();
    return s;
}

NS_NUM_HALIDE_END
