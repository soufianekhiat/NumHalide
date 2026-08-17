/// @file test_fft.cpp
/// @brief Tests for FFT operations

#include <gtest/gtest.h>
#include "numhalide_all.h"
#include <cmath>

using namespace numhalide;

// Helper to extract real part from complex Func
Halide::Func extract_real(Halide::Func complex_func, const std::string& name = "real") {
    Halide::Func result(name);
    Halide::Var x("x");
    result(x) = complex_func(x)[0];
    return result;
}

Halide::Func extract_imag(Halide::Func complex_func, const std::string& name = "imag") {
    Halide::Func result(name);
    Halide::Var x("x");
    result(x) = complex_func(x)[1];
    return result;
}

Halide::Func extract_real_2d(Halide::Func complex_func, const std::string& name = "real") {
    Halide::Func result(name);
    Halide::Var x("x"), y("y");
    result(x, y) = complex_func(x, y)[0];
    return result;
}

Halide::Func extract_imag_2d(Halide::Func complex_func, const std::string& name = "imag") {
    Halide::Func result(name);
    Halide::Var x("x"), y("y");
    result(x, y) = complex_func(x, y)[1];
    return result;
}

// -----------------------------------------------------------------------------
// Complex Number Tests
// -----------------------------------------------------------------------------

TEST(FFT, ComplexAdd) {
    Halide::Func result("result");
    Halide::Var x;
    result(x) = complex_add(Halide::Tuple(1.0f, 2.0f), Halide::Tuple(3.0f, 4.0f));

    auto re = extract_real(result, "re");
    auto im = extract_imag(result, "im");

    Halide::Runtime::Buffer<float> re_out(1), im_out(1);
    re.realize(re_out);
    im.realize(im_out);

    EXPECT_NEAR(re_out(0), 4.0f, 1e-5f);  // 1 + 3
    EXPECT_NEAR(im_out(0), 6.0f, 1e-5f);  // 2 + 4
}

TEST(FFT, ComplexMul) {
    // (1 + 2i) * (3 + 4i) = 3 + 4i + 6i + 8i^2 = 3 + 10i - 8 = -5 + 10i
    Halide::Func result("result");
    Halide::Var x;
    result(x) = complex_mul(Halide::Tuple(1.0f, 2.0f), Halide::Tuple(3.0f, 4.0f));

    auto re = extract_real(result, "re");
    auto im = extract_imag(result, "im");

    Halide::Runtime::Buffer<float> re_out(1), im_out(1);
    re.realize(re_out);
    im.realize(im_out);

    EXPECT_NEAR(re_out(0), -5.0f, 1e-5f);
    EXPECT_NEAR(im_out(0), 10.0f, 1e-5f);
}

TEST(FFT, Expj) {
    // e^(j*0) = 1 + 0i
    // e^(j*pi/2) = 0 + 1i
    // e^(j*pi) = -1 + 0i
    Halide::Func result("result");
    Halide::Var x;

    float pi = static_cast<float>(M_PI);
    result(x) = expj(Halide::select(x == 0, 0.0f,
                     Halide::select(x == 1, pi / 2.0f, pi)));

    auto re = extract_real(result, "re");
    auto im = extract_imag(result, "im");

    Halide::Runtime::Buffer<float> re_out(3), im_out(3);
    re.realize(re_out);
    im.realize(im_out);

    EXPECT_NEAR(re_out(0), 1.0f, 1e-5f);
    EXPECT_NEAR(im_out(0), 0.0f, 1e-5f);

    EXPECT_NEAR(re_out(1), 0.0f, 1e-5f);
    EXPECT_NEAR(im_out(1), 1.0f, 1e-5f);

    EXPECT_NEAR(re_out(2), -1.0f, 1e-5f);
    EXPECT_NEAR(im_out(2), 0.0f, 1e-5f);
}

// -----------------------------------------------------------------------------
// 1D FFT Tests
// -----------------------------------------------------------------------------

TEST(FFT, FFT1D_DC) {
    // FFT of constant signal: all energy at DC
    // Input: [1, 1, 1, 1]
    // Output: [4, 0, 0, 0]
    Halide::Func input("input");
    Halide::Var x;
    input(x) = Halide::Tuple(1.0f, 0.0f);

    auto result = fft(input, 4, "fft");
    auto re = extract_real(result, "re");
    auto im = extract_imag(result, "im");

    Halide::Runtime::Buffer<float> re_out(4), im_out(4);
    re.realize(re_out);
    im.realize(im_out);

    EXPECT_NEAR(re_out(0), 4.0f, 1e-4f);  // DC component
    EXPECT_NEAR(im_out(0), 0.0f, 1e-4f);

    // Other components should be ~0
    for (int i = 1; i < 4; i++) {
        EXPECT_NEAR(re_out(i), 0.0f, 1e-4f);
        EXPECT_NEAR(im_out(i), 0.0f, 1e-4f);
    }
}

TEST(FFT, FFT1D_Impulse) {
    // FFT of impulse: flat spectrum
    // Input: [1, 0, 0, 0]
    // Output: [1, 1, 1, 1]
    Halide::Func input("input");
    Halide::Var x;
    input(x) = Halide::Tuple(
        Halide::select(x == 0, 1.0f, 0.0f),
        0.0f
    );

    auto result = fft(input, 4, "fft");
    auto re = extract_real(result, "re");
    auto im = extract_imag(result, "im");

    Halide::Runtime::Buffer<float> re_out(4), im_out(4);
    re.realize(re_out);
    im.realize(im_out);

    // All components should be 1
    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(re_out(i), 1.0f, 1e-4f);
        EXPECT_NEAR(im_out(i), 0.0f, 1e-4f);
    }
}

TEST(FFT, FFT1D_Roundtrip) {
    // Test FFT -> IFFT roundtrip
    // Input: [1, 2, 3, 4]
    Halide::Func input("input");
    Halide::Var x;
    input(x) = Halide::Tuple(Halide::cast<float>(x + 1), 0.0f);

    auto fft_result = fft(input, 4, "fft");
    auto ifft_result = ifft_normalized(fft_result, 4, "ifft");

    auto re = extract_real(ifft_result, "re");
    auto im = extract_imag(ifft_result, "im");

    Halide::Runtime::Buffer<float> re_out(4), im_out(4);
    re.realize(re_out);
    im.realize(im_out);

    // Should recover original signal
    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(re_out(i), static_cast<float>(i + 1), 1e-4f);
        EXPECT_NEAR(im_out(i), 0.0f, 1e-4f);
    }
}

TEST(FFT, FFT1D_Size8) {
    // Test larger FFT size
    Halide::Func input("input");
    Halide::Var x;
    input(x) = Halide::Tuple(1.0f, 0.0f);

    auto result = fft(input, 8, "fft8");
    auto re = extract_real(result, "re");
    auto im = extract_imag(result, "im");

    Halide::Runtime::Buffer<float> re_out(8), im_out(8);
    re.realize(re_out);
    im.realize(im_out);

    // DC should be 8, others should be 0
    EXPECT_NEAR(re_out(0), 8.0f, 1e-4f);
    for (int i = 1; i < 8; i++) {
        EXPECT_NEAR(re_out(i), 0.0f, 1e-4f);
        EXPECT_NEAR(im_out(i), 0.0f, 1e-4f);
    }
}

// -----------------------------------------------------------------------------
// 2D FFT Tests
// -----------------------------------------------------------------------------

TEST(FFT, FFT2D_DC) {
    // 2D FFT of constant: all energy at DC
    Halide::Func input("input");
    Halide::Var x, y;
    input(x, y) = Halide::Tuple(1.0f, 0.0f);

    auto result = fft2d(input, 4, 4, "fft2d");
    auto re = extract_real_2d(result, "re");
    auto im = extract_imag_2d(result, "im");

    Halide::Runtime::Buffer<float> re_out(4, 4), im_out(4, 4);
    re.realize(re_out);
    im.realize(im_out);

    // DC should be 16, others should be ~0
    EXPECT_NEAR(re_out(0, 0), 16.0f, 1e-3f);

    for (int j = 0; j < 4; j++) {
        for (int i = 0; i < 4; i++) {
            if (i != 0 || j != 0) {
                EXPECT_NEAR(re_out(i, j), 0.0f, 1e-3f);
                EXPECT_NEAR(im_out(i, j), 0.0f, 1e-3f);
            }
        }
    }
}

TEST(FFT, FFT2D_Roundtrip) {
    // Test 2D FFT -> IFFT roundtrip
    Halide::Func input("input");
    Halide::Var x, y;
    input(x, y) = Halide::Tuple(Halide::cast<float>(y * 4 + x + 1), 0.0f);

    auto fft_result = fft2d(input, 4, 4, "fft2d");
    auto ifft_result = ifft2d_normalized(fft_result, 4, 4, "ifft2d");

    auto re = extract_real_2d(ifft_result, "re");
    auto im = extract_imag_2d(ifft_result, "im");

    Halide::Runtime::Buffer<float> re_out(4, 4), im_out(4, 4);
    re.realize(re_out);
    im.realize(im_out);

    // Should recover original
    for (int j = 0; j < 4; j++) {
        for (int i = 0; i < 4; i++) {
            float expected = static_cast<float>(j * 4 + i + 1);
            EXPECT_NEAR(re_out(i, j), expected, 1e-3f);
            EXPECT_NEAR(im_out(i, j), 0.0f, 1e-3f);
        }
    }
}

// -----------------------------------------------------------------------------
// FFT Utility Tests
// -----------------------------------------------------------------------------

TEST(FFT, FFTShift1D) {
    // Input: [0, 1, 2, 3]
    // After shift: [2, 3, 0, 1]
    Halide::Func input("input");
    Halide::Var x;
    input(x) = Halide::Tuple(Halide::cast<float>(x), 0.0f);

    auto result = fftshift_1d(input, 4, "shift");
    auto re = extract_real(result, "re");

    Halide::Runtime::Buffer<float> re_out(4);
    re.realize(re_out);

    EXPECT_NEAR(re_out(0), 2.0f, 1e-5f);
    EXPECT_NEAR(re_out(1), 3.0f, 1e-5f);
    EXPECT_NEAR(re_out(2), 0.0f, 1e-5f);
    EXPECT_NEAR(re_out(3), 1.0f, 1e-5f);
}

TEST(FFT, PowerSpectrum) {
    // Input: 3 + 4i -> |z|^2 = 9 + 16 = 25
    Halide::Func input("input");
    Halide::Var x;
    input(x) = Halide::Tuple(3.0f, 4.0f);

    auto result = power_spectrum(input, "power");

    Halide::Runtime::Buffer<float> out(1);
    result.realize(out);

    EXPECT_NEAR(out(0), 25.0f, 1e-5f);
}

TEST(FFT, RealToComplex) {
    Halide::Func input("input");
    Halide::Var x;
    input(x) = Halide::cast<float>(x + 1);  // [1, 2, 3, 4]

    auto result = real_to_complex(input, "r2c");
    auto re = extract_real(result, "re");
    auto im = extract_imag(result, "im");

    Halide::Runtime::Buffer<float> re_out(4), im_out(4);
    re.realize(re_out);
    im.realize(im_out);

    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(re_out(i), static_cast<float>(i + 1), 1e-5f);
        EXPECT_NEAR(im_out(i), 0.0f, 1e-5f);
    }
}
