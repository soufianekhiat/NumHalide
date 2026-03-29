/// @file test_fft_fast.cpp
/// @brief Tests for O(N log N) Cooley-Tukey fft_fast
#include <gtest/gtest.h>
#include "numhalide_all.h"
#include <cmath>
#include <vector>
#include <complex>
using namespace numhalide;

static const float PI_F = 3.14159265358979323846f;

// Helper: create complex Func from vectors
static Halide::Func make_complex(const std::vector<float>& re,
                                  const std::vector<float>& im,
                                  const std::string& nm)
{
    int N = (int)re.size();
    Halide::Buffer<float> re_b(N), im_b(N);
    for (int i = 0; i < N; ++i) { re_b(i) = re[i]; im_b(i) = im[i]; }
    Halide::Func f(nm);
    Halide::Var k;
    f(k) = Halide::Tuple(re_b(Halide::clamp(k, 0, N-1)),
                         im_b(Halide::clamp(k, 0, N-1)));
    return f;
}

// Helper: realize complex Func into re/im vectors
static void realize_1d(Halide::Func f, int N, std::vector<float>& re, std::vector<float>& im)
{
    Halide::Runtime::Buffer<float> rb(N), ib(N);
    Halide::Var k;
    Halide::Func rf("r_" + f.name()), imf("i_" + f.name());
    rf(k)  = f(k)[0];
    imf(k) = f(k)[1];
    rf.realize(rb);  imf.realize(ib);
    re.resize(N); im.resize(N);
    for (int i = 0; i < N; ++i) { re[i] = rb(i); im[i] = ib(i); }
}

// 1. Impulse: x[0]=1, rest 0 → all bins (1, 0)
TEST(FFTFast, Impulse_N8) {
    const int N = 8;
    std::vector<float> re(N, 0.0f), im(N, 0.0f); re[0] = 1.0f;
    auto f = make_complex(re, im, "imp8");
    auto F = fft_fast(f, N, "imp8_out");
    std::vector<float> ore, oim;
    realize_1d(F, N, ore, oim);
    for (int k = 0; k < N; ++k) {
        EXPECT_NEAR(ore[k], 1.0f, 1e-5f) << "re[" << k << "]";
        EXPECT_NEAR(oim[k], 0.0f, 1e-5f) << "im[" << k << "]";
    }
}

// 2. DC: x[n]=1 → bin 0 = N, rest 0
TEST(FFTFast, DC_N8) {
    const int N = 8;
    std::vector<float> re(N, 1.0f), im(N, 0.0f);
    auto f = make_complex(re, im, "dc8");
    auto F = fft_fast(f, N, "dc8_out");
    std::vector<float> ore, oim;
    realize_1d(F, N, ore, oim);
    EXPECT_NEAR(ore[0], (float)N, 1e-4f);
    EXPECT_NEAR(oim[0], 0.0f, 1e-4f);
    for (int k = 1; k < N; ++k) {
        EXPECT_NEAR(ore[k], 0.0f, 1e-4f) << "re[" << k << "]";
        EXPECT_NEAR(oim[k], 0.0f, 1e-4f) << "im[" << k << "]";
    }
}

// 3. N=4 known values: x = [1, 2, 3, 4]
// DFT: X[0]=10, X[1]=-2+2j, X[2]=-2, X[3]=-2-2j
TEST(FFTFast, KnownN4) {
    const int N = 4;
    std::vector<float> re = {1.0f, 2.0f, 3.0f, 4.0f}, im(N, 0.0f);
    auto f = make_complex(re, im, "kn4");
    auto F = fft_fast(f, N, "kn4_out");
    std::vector<float> ore, oim;
    realize_1d(F, N, ore, oim);
    EXPECT_NEAR(ore[0], 10.0f, 1e-4f);
    EXPECT_NEAR(oim[0], 0.0f,  1e-4f);
    EXPECT_NEAR(ore[1], -2.0f, 1e-4f);
    EXPECT_NEAR(oim[1],  2.0f, 1e-4f);
    EXPECT_NEAR(ore[2], -2.0f, 1e-4f);
    EXPECT_NEAR(oim[2],  0.0f, 1e-4f);
    EXPECT_NEAR(ore[3], -2.0f, 1e-4f);
    EXPECT_NEAR(oim[3], -2.0f, 1e-4f);
}

// 4. Roundtrip N=16: fft_fast → ifft_fast → original
TEST(FFTFast, Roundtrip_N16) {
    const int N = 16;
    std::vector<float> re(N), im(N, 0.0f);
    for (int i = 0; i < N; ++i)
        re[i] = std::cos(2.0f * PI_F * 3.0f * i / N) + 0.3f * std::sin(2.0f * PI_F * 7.0f * i / N);
    auto f  = make_complex(re, im, "rt16");
    auto F  = fft_fast(f, N, "rt16_f");
    auto xr = ifft_fast(F, N, "rt16_i");
    std::vector<float> ore, oim;
    realize_1d(xr, N, ore, oim);
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(ore[i], re[i], 1e-4f) << "re[" << i << "]";
        EXPECT_NEAR(oim[i], im[i], 1e-4f) << "im[" << i << "]";
    }
}

// 5. Roundtrip N=64
TEST(FFTFast, Roundtrip_N64) {
    const int N = 64;
    std::vector<float> re(N), im(N);
    for (int i = 0; i < N; ++i) {
        re[i] = std::sin(2.0f * PI_F * 5.0f * i / N);
        im[i] = std::cos(2.0f * PI_F * 11.0f * i / N) * 0.5f;
    }
    auto f  = make_complex(re, im, "rt64");
    auto F  = fft_fast(f, N, "rt64_f");
    auto xr = ifft_fast(F, N, "rt64_i");
    std::vector<float> ore, oim;
    realize_1d(xr, N, ore, oim);
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(ore[i], re[i], 1e-4f) << "re[" << i << "]";
        EXPECT_NEAR(oim[i], im[i], 1e-4f) << "im[" << i << "]";
    }
}

// 6. Golden comparison vs slow fft, N=32
TEST(FFTFast, GoldenVsSlow_N32) {
    const int N = 32;
    std::vector<float> re(N), im(N);
    for (int i = 0; i < N; ++i) {
        re[i] = std::cos(2.0f * PI_F * i / N * 4.0f) + 0.3f * std::sin(2.0f * PI_F * i / N * 9.0f);
        im[i] = 0.0f;
    }
    auto f = make_complex(re, im, "gv32");

    // Slow fft (existing DFT)
    auto F_slow = fft(f, N, "gv32_slow");
    auto F_fast = fft_fast(f, N, "gv32_fast");

    std::vector<float> rs, is_v, rf, i_f;
    realize_1d(F_slow, N, rs, is_v);
    realize_1d(F_fast, N, rf, i_f);

    for (int k = 0; k < N; ++k) {
        EXPECT_NEAR(rf[k], rs[k], 1e-3f) << "re[" << k << "]";
        EXPECT_NEAR(i_f[k], is_v[k], 1e-3f) << "im[" << k << "]";
    }
}

// 7. Roundtrip N=256
TEST(FFTFast, Roundtrip_N256) {
    const int N = 256;
    std::vector<float> re(N), im(N, 0.0f);
    for (int i = 0; i < N; ++i)
        re[i] = std::sin(2.0f * PI_F * 17.0f * i / N) * 0.7f + std::cos(2.0f * PI_F * 41.0f * i / N) * 0.3f;
    auto f  = make_complex(re, im, "rt256");
    auto F  = fft_fast(f, N, "rt256_f");
    auto xr = ifft_fast(F, N, "rt256_i");
    std::vector<float> ore, oim;
    realize_1d(xr, N, ore, oim);
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(ore[i], re[i], 1e-3f) << "[" << i << "]";
        EXPECT_NEAR(oim[i], 0.0f,  1e-3f) << "im[" << i << "]";
    }
}

// 8. Golden comparison vs slow fft, N=64
TEST(FFTFast, GoldenVsSlow_N64) {
    const int N = 64;
    std::vector<float> re(N), im(N);
    for (int i = 0; i < N; ++i) {
        re[i] = 0.5f * std::sin(2.0f * PI_F * 5.0f * i / N);
        im[i] = 0.3f * std::cos(2.0f * PI_F * 11.0f * i / N);
    }
    auto f = make_complex(re, im, "gv64");
    auto F_slow = fft(f, N, "gv64_slow");
    auto F_fast = fft_fast(f, N, "gv64_fast");
    std::vector<float> rs, is_v, rf, i_f;
    realize_1d(F_slow, N, rs, is_v);
    realize_1d(F_fast, N, rf, i_f);
    for (int k = 0; k < N; ++k) {
        EXPECT_NEAR(rf[k], rs[k], 1e-3f) << "re[" << k << "]";
        EXPECT_NEAR(i_f[k], is_v[k], 1e-3f) << "im[" << k << "]";
    }
}

// 9. 2D Roundtrip 8x8
TEST(FFTFast, Roundtrip2D_8x8) {
    const int rows = 8, cols = 8;
    Halide::Buffer<float> re_b(cols, rows), im_b(cols, rows);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            re_b(c, r) = std::cos(2.0f * PI_F * c / cols * 2.0f) * std::cos(2.0f * PI_F * r / rows * 3.0f);
            im_b(c, r) = 0.0f;
        }
    Halide::Func f2("rt2d8"); Halide::Var x, y;
    f2(x, y) = Halide::Tuple(re_b(x, y), im_b(x, y));
    auto F2  = fft2d_fast(f2, rows, cols, "rt2d8_f");
    auto xr2 = ifft2d_fast(F2, rows, cols, "rt2d8_i");
    Halide::Runtime::Buffer<float> ore(cols, rows), oim(cols, rows);
    Halide::Func rf("r2d8"), imf("i2d8");
    rf(x, y) = xr2(x, y)[0]; imf(x, y) = xr2(x, y)[1];
    rf.realize(ore); imf.realize(oim);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            EXPECT_NEAR(ore(c, r), re_b(c, r), 1e-4f) << "(" << c << "," << r << ")";
        }
}

// 10. 2D Golden comparison vs slow fft2d, 8x8
TEST(FFTFast, Golden2D_8x8) {
    const int rows = 8, cols = 8;
    Halide::Buffer<float> re_b(cols, rows), im_b(cols, rows);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            re_b(c, r) = (float)((r * cols + c) % 5) - 2.0f;
            im_b(c, r) = 0.0f;
        }
    Halide::Func f2("g2d8"); Halide::Var x, y;
    f2(x, y) = Halide::Tuple(re_b(x, y), im_b(x, y));
    auto F_slow = fft2d(f2, rows, cols, "g2d8_slow");
    auto F_fast = fft2d_fast(f2, rows, cols, "g2d8_fast");
    Halide::Runtime::Buffer<float> rs(cols, rows), is_b(cols, rows), rf(cols, rows), i_f(cols, rows);
    Halide::Func rfs("rs2d"), ifs("is2d"), rff("rf2d"), iff("if2d");
    rfs(x, y) = F_slow(x, y)[0]; ifs(x, y) = F_slow(x, y)[1];
    rff(x, y) = F_fast(x, y)[0]; iff(x, y) = F_fast(x, y)[1];
    rfs.realize(rs); ifs.realize(is_b);
    rff.realize(rf); iff.realize(i_f);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            EXPECT_NEAR(rf(c, r), rs(c, r), 1e-3f) << "re(" << c << "," << r << ")";
            EXPECT_NEAR(i_f(c, r), is_b(c, r), 1e-3f) << "im(" << c << "," << r << ")";
        }
}

// 11. 2D Roundtrip 16x16
TEST(FFTFast, Roundtrip2D_16x16) {
    const int rows = 16, cols = 16;
    Halide::Buffer<float> re_b(cols, rows), im_b(cols, rows);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            re_b(c, r) = std::sin(2.0f * PI_F * c * 3.0f / cols + 2.0f * PI_F * r * 2.0f / rows);
            im_b(c, r) = 0.0f;
        }
    Halide::Func f2("rt2d16"); Halide::Var x, y;
    f2(x, y) = Halide::Tuple(re_b(x, y), im_b(x, y));
    auto F2  = fft2d_fast(f2, rows, cols, "rt2d16_f");
    auto xr2 = ifft2d_fast(F2, rows, cols, "rt2d16_i");
    Halide::Runtime::Buffer<float> ore(cols, rows);
    Halide::Func rf("r16"); rf(x, y) = xr2(x, y)[0]; rf.realize(ore);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            EXPECT_NEAR(ore(c, r), re_b(c, r), 1e-4f) << "(" << c << "," << r << ")";
}

// 12. Parseval's theorem: sum|x|^2 = (1/N) * sum|X|^2
TEST(FFTFast, Parseval_N32) {
    const int N = 32;
    std::vector<float> re(N), im(N);
    for (int i = 0; i < N; ++i) {
        re[i] = std::sin(2.0f * PI_F * 3.0f * i / N);
        im[i] = std::cos(2.0f * PI_F * 7.0f * i / N) * 0.5f;
    }
    // Time-domain energy
    float E_time = 0.0f;
    for (int i = 0; i < N; ++i) E_time += re[i] * re[i] + im[i] * im[i];

    auto f = make_complex(re, im, "pars32");
    auto F = fft_fast(f, N, "pars32_f");
    std::vector<float> orf, oif;
    realize_1d(F, N, orf, oif);

    float E_freq = 0.0f;
    for (int k = 0; k < N; ++k) E_freq += orf[k] * orf[k] + oif[k] * oif[k];
    E_freq /= (float)N;

    EXPECT_NEAR(E_freq, E_time, E_time * 1e-3f);
}
