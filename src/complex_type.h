/// @file complex_type.h
/// @brief C++ complex float value type and buffer wrapper
///
/// Provides:
///   complex_f32        — C++ struct for a single complex float value
///   ComplexBuffer      — pair of Halide::Runtime::Buffer<float> (re + im)
///   complex_buf_*      — buffer-level arithmetic and conversion helpers
///
/// These work at the C++ / buffer level (not at the Halide Func/Expr level).
/// For Expr-level complex, see fft.h which provides Halide::Tuple-based helpers.

#pragma once

#include "shape.h"

#include <cmath>
#include <algorithm>

NS_NUM_HALIDE_BEGIN

// =============================================================================
// complex_f32 — scalar complex value
// =============================================================================

/// @brief Single-precision complex floating-point value
struct complex_f32 {
    float re, im;

    complex_f32(float r = 0.0f, float i = 0.0f) : re(r), im(i) {}

    // -------------------------------------------------------------------------
    // Arithmetic operators
    // -------------------------------------------------------------------------

    complex_f32 operator+(const complex_f32& b) const {
        return complex_f32(re + b.re, im + b.im);
    }
    complex_f32 operator-(const complex_f32& b) const {
        return complex_f32(re - b.re, im - b.im);
    }
    /// (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    complex_f32 operator*(const complex_f32& b) const {
        return complex_f32(re * b.re - im * b.im,
                           re * b.im + im * b.re);
    }
    /// (a+bi)/(c+di) = [(a+bi)(c-di)] / (c²+d²)
    complex_f32 operator/(const complex_f32& b) const {
        float denom = b.re * b.re + b.im * b.im;
        nh_require(nullptr, denom != 0.0f, "complex_f32::operator/: division by zero");
        return complex_f32((re * b.re + im * b.im) / denom,
                           (im * b.re - re * b.im) / denom);
    }
    complex_f32 operator-() const {
        return complex_f32(-re, -im);
    }

    bool operator==(const complex_f32& b) const {
        return re == b.re && im == b.im;
    }
    bool operator!=(const complex_f32& b) const { return !(*this == b); }

    // -------------------------------------------------------------------------
    // Properties
    // -------------------------------------------------------------------------

    /// Magnitude: sqrt(re² + im²)
    float abs() const { return std::sqrt(re * re + im * im); }
    /// Squared magnitude: re² + im²
    float abs2() const { return re * re + im * im; }
    /// Phase angle in radians: atan2(im, re)
    float phase() const { return std::atan2(im, re); }
    /// Complex conjugate: (re, -im)
    complex_f32 conj() const { return complex_f32(re, -im); }
    /// Unit vector in same direction
    complex_f32 normalized() const {
        float a = abs();
        nh_require(nullptr, a > 0.0f, "complex_f32::normalized: zero magnitude");
        return complex_f32(re / a, im / a);
    }
};

// =============================================================================
// Polar construction helpers
// =============================================================================

/// @brief Construct complex from polar form: mag * exp(i * phase)
inline complex_f32 complex_from_polar(float mag, float phase_rad) {
    return complex_f32(mag * std::cos(phase_rad), mag * std::sin(phase_rad));
}

/// @brief Unit complex from angle: cos(theta) + i*sin(theta)
inline complex_f32 complex_exp_i(float theta) {
    return complex_f32(std::cos(theta), std::sin(theta));
}

// =============================================================================
// ComplexBuffer — 1D complex signal stored as separate re/im float buffers
// =============================================================================

/// @brief 1D complex buffer: pair of Halide::Runtime::Buffer<float>
struct ComplexBuffer {
    Halide::Runtime::Buffer<float> re;
    Halide::Runtime::Buffer<float> im;

    /// @brief Allocate an n-element complex buffer, initialised to zero.
    explicit ComplexBuffer(int n)
        : re(n), im(n)
    {
        nh_require(nullptr, n > 0, "ComplexBuffer: n must be > 0, got %d", n);
        for (int i = 0; i < n; ++i) { re(i) = 0.0f; im(i) = 0.0f; }
    }

    /// @brief Number of complex elements
    int size() const { return re.width(); }

    /// @brief Read element i
    complex_f32 operator()(int i) const {
        return complex_f32(re(i), im(i));
    }

    /// @brief Write element i
    void set(int i, complex_f32 v) {
        re(i) = v.re;
        im(i) = v.im;
    }
};

// =============================================================================
// Buffer-level operations (all return new ComplexBuffer, no aliasing)
// =============================================================================

/// @brief Element-wise addition
inline ComplexBuffer complex_buf_add(const ComplexBuffer& a, const ComplexBuffer& b) {
    nh_require(nullptr, a.size() == b.size(),
               "complex_buf_add: size mismatch %d vs %d", a.size(), b.size());
    int n = a.size();
    ComplexBuffer out(n);
    for (int i = 0; i < n; ++i)
        out.set(i, a(i) + b(i));
    return out;
}

/// @brief Element-wise subtraction
inline ComplexBuffer complex_buf_sub(const ComplexBuffer& a, const ComplexBuffer& b) {
    nh_require(nullptr, a.size() == b.size(),
               "complex_buf_sub: size mismatch %d vs %d", a.size(), b.size());
    int n = a.size();
    ComplexBuffer out(n);
    for (int i = 0; i < n; ++i)
        out.set(i, a(i) - b(i));
    return out;
}

/// @brief Element-wise complex multiplication
inline ComplexBuffer complex_buf_mul(const ComplexBuffer& a, const ComplexBuffer& b) {
    nh_require(nullptr, a.size() == b.size(),
               "complex_buf_mul: size mismatch %d vs %d", a.size(), b.size());
    int n = a.size();
    ComplexBuffer out(n);
    for (int i = 0; i < n; ++i)
        out.set(i, a(i) * b(i));
    return out;
}

/// @brief Element-wise complex conjugate
inline ComplexBuffer complex_buf_conj(const ComplexBuffer& a) {
    int n = a.size();
    ComplexBuffer out(n);
    for (int i = 0; i < n; ++i)
        out.set(i, a(i).conj());
    return out;
}

/// @brief Scale every element by a real scalar
inline ComplexBuffer complex_buf_scale(const ComplexBuffer& a, float s) {
    int n = a.size();
    ComplexBuffer out(n);
    for (int i = 0; i < n; ++i)
        out.set(i, complex_f32(a.re(i) * s, a.im(i) * s));
    return out;
}

/// @brief Element-wise magnitude (returns real buffer)
inline Halide::Runtime::Buffer<float> complex_buf_abs(const ComplexBuffer& a) {
    int n = a.size();
    Halide::Runtime::Buffer<float> out(n);
    for (int i = 0; i < n; ++i)
        out(i) = a(i).abs();
    return out;
}

/// @brief Element-wise phase angle in radians (returns real buffer)
inline Halide::Runtime::Buffer<float> complex_buf_phase(const ComplexBuffer& a) {
    int n = a.size();
    Halide::Runtime::Buffer<float> out(n);
    for (int i = 0; i < n; ++i)
        out(i) = a(i).phase();
    return out;
}

/// @brief Construct ComplexBuffer from real values (imaginary part = 0)
inline ComplexBuffer complex_buf_from_real(const Halide::Runtime::Buffer<float>& re_buf) {
    int n = re_buf.width();
    ComplexBuffer out(n);
    for (int i = 0; i < n; ++i)
        out.set(i, complex_f32(re_buf(i), 0.0f));
    return out;
}

/// @brief Construct ComplexBuffer from magnitude and phase buffers
inline ComplexBuffer complex_buf_from_polar(const Halide::Runtime::Buffer<float>& mag,
                                            const Halide::Runtime::Buffer<float>& phase_buf) {
    nh_require(nullptr, mag.width() == phase_buf.width(),
               "complex_buf_from_polar: size mismatch %d vs %d",
               mag.width(), phase_buf.width());
    int n = mag.width();
    ComplexBuffer out(n);
    for (int i = 0; i < n; ++i)
        out.set(i, complex_from_polar(mag(i), phase_buf(i)));
    return out;
}

NS_NUM_HALIDE_END
