/// @file test_complex_type.cpp
/// @brief Tests for complex_type.h (complex_f32 and ComplexBuffer)

#include <gtest/gtest.h>
#include "numhalide_all.h"

using namespace numhalide;

static const float PI = 3.14159265358979323846f;

// =============================================================================
// 1. Value_Construction
// =============================================================================
TEST(ComplexType, Value_Construction)
{
    complex_f32 z(3.0f, 4.0f);
    EXPECT_NEAR(z.re, 3.0f, 1e-5f);
    EXPECT_NEAR(z.im, 4.0f, 1e-5f);
}

// =============================================================================
// 2. Value_Add
// =============================================================================
TEST(ComplexType, Value_Add)
{
    complex_f32 a(1.0f, 2.0f);
    complex_f32 b(3.0f, 4.0f);
    complex_f32 c = a + b;
    EXPECT_NEAR(c.re, 4.0f, 1e-5f);
    EXPECT_NEAR(c.im, 6.0f, 1e-5f);
}

// =============================================================================
// 3. Value_Mul
// =============================================================================
TEST(ComplexType, Value_Mul)
{
    // (1+2i)*(3+4i) = (3-8) + (4+6)i = -5 + 10i
    complex_f32 a(1.0f, 2.0f);
    complex_f32 b(3.0f, 4.0f);
    complex_f32 c = a * b;
    EXPECT_NEAR(c.re, -5.0f, 1e-5f);
    EXPECT_NEAR(c.im, 10.0f, 1e-5f);
}

// =============================================================================
// 4. Value_Div
// =============================================================================
TEST(ComplexType, Value_Div)
{
    // (2+4i) / (1+2i)
    // numerator * conj(denom) = (2+4i)*(1-2i) = (2+8) + (-4+4)i = 10 + 0i
    // |denom|² = 1 + 4 = 5
    // result = 10/5 + 0/5*i = 2 + 0i
    complex_f32 a(2.0f, 4.0f);
    complex_f32 b(1.0f, 2.0f);
    complex_f32 c = a / b;
    EXPECT_NEAR(c.re, 2.0f, 1e-5f);
    EXPECT_NEAR(c.im, 0.0f, 1e-5f);
}

// =============================================================================
// 5. Value_Abs
// =============================================================================
TEST(ComplexType, Value_Abs)
{
    complex_f32 z(3.0f, 4.0f);
    EXPECT_NEAR(z.abs(), 5.0f, 1e-5f);
}

// =============================================================================
// 6. Value_Phase
// =============================================================================
TEST(ComplexType, Value_Phase)
{
    // phase of (1 + i) should be pi/4
    complex_f32 z(1.0f, 1.0f);
    EXPECT_NEAR(z.phase(), PI / 4.0f, 1e-5f);
}

// =============================================================================
// 7. Value_Conj
// =============================================================================
TEST(ComplexType, Value_Conj)
{
    complex_f32 z(3.0f, -4.0f);
    complex_f32 c = z.conj();
    EXPECT_NEAR(c.re,  3.0f, 1e-5f);
    EXPECT_NEAR(c.im,  4.0f, 1e-5f);
}

// =============================================================================
// 8. Value_PolarRoundtrip
// =============================================================================
TEST(ComplexType, Value_PolarRoundtrip)
{
    float mag   = 5.0f;
    float phase = PI / 3.0f;  // 60 degrees
    complex_f32 z = complex_from_polar(mag, phase);

    // re = 5*cos(π/3) = 5*0.5 = 2.5
    // im = 5*sin(π/3) = 5*sqrt(3)/2 ≈ 4.330
    EXPECT_NEAR(z.re, mag * std::cos(phase), 1e-5f);
    EXPECT_NEAR(z.im, mag * std::sin(phase), 1e-5f);

    // Round-trip: abs and phase should recover mag and phase
    EXPECT_NEAR(z.abs(),   mag,   1e-5f);
    EXPECT_NEAR(z.phase(), phase, 1e-5f);
}

// =============================================================================
// 9. Buffer_Construction
// =============================================================================
TEST(ComplexType, Buffer_Construction)
{
    ComplexBuffer cb(8);
    EXPECT_EQ(cb.size(), 8);
    for (int i = 0; i < 8; ++i) {
        EXPECT_NEAR(cb(i).re, 0.0f, 1e-5f);
        EXPECT_NEAR(cb(i).im, 0.0f, 1e-5f);
    }
}

// =============================================================================
// 10. Buffer_Mul
// =============================================================================
TEST(ComplexType, Buffer_Mul)
{
    ComplexBuffer a(4);
    ComplexBuffer b(4);
    // a[0] = 1+2i, b[0] = 3+4i -> product = -5+10i
    a.set(0, complex_f32(1.0f, 2.0f));
    b.set(0, complex_f32(3.0f, 4.0f));
    a.set(1, complex_f32(1.0f, 0.0f));
    b.set(1, complex_f32(2.0f, 0.0f));

    ComplexBuffer c = complex_buf_mul(a, b);
    EXPECT_NEAR(c(0).re, -5.0f, 1e-5f);
    EXPECT_NEAR(c(0).im, 10.0f, 1e-5f);
    EXPECT_NEAR(c(1).re,  2.0f, 1e-5f);
    EXPECT_NEAR(c(1).im,  0.0f, 1e-5f);
}

// =============================================================================
// 11. Buffer_Abs
// =============================================================================
TEST(ComplexType, Buffer_Abs)
{
    ComplexBuffer cb(4);
    cb.set(0, complex_f32(3.0f, 4.0f));
    cb.set(1, complex_f32(0.0f, 1.0f));
    cb.set(2, complex_f32(1.0f, 0.0f));
    cb.set(3, complex_f32(5.0f, 12.0f));

    auto mag = complex_buf_abs(cb);
    EXPECT_NEAR(mag(0),  5.0f, 1e-5f);
    EXPECT_NEAR(mag(1),  1.0f, 1e-5f);
    EXPECT_NEAR(mag(2),  1.0f, 1e-5f);
    EXPECT_NEAR(mag(3), 13.0f, 1e-5f);
}

// =============================================================================
// 12. Buffer_Phase
// =============================================================================
TEST(ComplexType, Buffer_Phase)
{
    ComplexBuffer cb(2);
    cb.set(0, complex_f32(1.0f, 1.0f));  // phase = pi/4
    cb.set(1, complex_f32(0.0f, 1.0f));  // phase = pi/2

    auto phases = complex_buf_phase(cb);
    EXPECT_NEAR(phases(0), PI / 4.0f, 1e-5f);
    EXPECT_NEAR(phases(1), PI / 2.0f, 1e-5f);
}
