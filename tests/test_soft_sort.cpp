#include <gtest/gtest.h>
#include <Halide.h>
#include "../src/numhalide_all.h"
using namespace numhalide;

static Halide::Func make_1d(std::initializer_list<float> vals,
    const std::string& n = "f")
{
    std::vector<float> v(vals);
    Halide::Buffer<float> buf((int)v.size());
    for (int i = 0; i < (int)v.size(); ++i) buf(i) = v[i];
    Halide::Func f(n);
    Halide::Var x;
    f(x) = buf(x);
    return f;
}

// --- soft_rank ---

TEST(SoftSort, SoftRank_AscendingOrder) {
    // [1, 3, 2]: min→rank~0.5, max→rank~2.5
    auto f = make_1d({1.0f, 3.0f, 2.0f}, "f_sr");
    auto r = soft_rank(f, 3, 0.1f);  // small tau → near-hard
    Halide::Runtime::Buffer<float> out(3);
    r.realize(out);
    // element 0 (value 1) should have smallest rank
    // element 1 (value 3) should have largest rank
    // element 2 (value 2) should be middle
    EXPECT_LT(out(0), out(2));
    EXPECT_LT(out(2), out(1));
}

TEST(SoftSort, SoftRank_MonotonicWithTemperature) {
    // Identical values → all ranks equal (~n/2)
    auto f = make_1d({5.0f, 5.0f, 5.0f}, "f_eq");
    auto r = soft_rank(f, 3, 1.0f);
    Halide::Runtime::Buffer<float> out(3);
    r.realize(out);
    // All equal → each rank ≈ 0.5+0.5+0.5 = 1.5
    EXPECT_NEAR(out(0), out(1), 1e-4f);
    EXPECT_NEAR(out(1), out(2), 1e-4f);
}

TEST(SoftSort, SoftRank_Range) {
    // Ranks should lie in [0, n]
    auto f = make_1d({3.0f, 1.0f, 4.0f, 1.0f, 5.0f}, "f_rng");
    auto r = soft_rank(f, 5, 0.5f);
    Halide::Runtime::Buffer<float> out(5);
    r.realize(out);
    for (int i = 0; i < 5; ++i) {
        EXPECT_GE(out(i), 0.0f);
        EXPECT_LE(out(i), 5.0f);
    }
}

// --- soft_sort ---

TEST(SoftSort, SoftSort_ApproximatesHardSort) {
    // [4, 1, 3, 2] → hard sort ascending: [1, 2, 3, 4]
    // With small tau, soft_sort should be close
    auto f = make_1d({4.0f, 1.0f, 3.0f, 2.0f}, "f_ss");
    auto r = soft_sort(f, 4, 0.05f, 0.5f);
    Halide::Runtime::Buffer<float> out(4);
    r.realize(out);
    EXPECT_NEAR(out(0), 1.0f, 0.3f);
    EXPECT_NEAR(out(1), 2.0f, 0.3f);
    EXPECT_NEAR(out(2), 3.0f, 0.3f);
    EXPECT_NEAR(out(3), 4.0f, 0.3f);
}

TEST(SoftSort, SoftSort_Monotone) {
    // Output should be non-decreasing
    auto f = make_1d({5.0f, 2.0f, 8.0f, 1.0f, 9.0f}, "f_mon");
    auto r = soft_sort(f, 5, 0.2f, 0.5f);
    Halide::Runtime::Buffer<float> out(5);
    r.realize(out);
    for (int i = 1; i < 5; ++i) {
        EXPECT_LE(out(i - 1), out(i) + 1e-4f);
    }
}

TEST(SoftSort, SoftSort_SumPreserved) {
    // Sum of soft-sorted values should equal sum of original values
    auto f = make_1d({3.0f, 1.0f, 4.0f, 1.0f, 5.0f}, "f_sum");
    auto r = soft_sort(f, 5, 0.5f, 1.0f);
    Halide::Runtime::Buffer<float> out(5);
    r.realize(out);
    float total = 0.0f;
    for (int i = 0; i < 5; ++i) total += out(i);
    EXPECT_NEAR(total, 14.0f, 0.05f);  // 3+1+4+1+5 = 14
}

// --- soft_argsort ---

TEST(SoftSort, SoftArgsort_RowsSumToOne) {
    // Each row of P (each output position) should sum to ~1
    auto f = make_1d({2.0f, 0.0f, 1.0f}, "f_perm");
    auto r = soft_argsort(f, 3, 0.3f, 0.5f);
    // out(p, i): p = position (0..2), i = element (0..2)
    Halide::Runtime::Buffer<float> out(3, 3);  // (cols=p, rows=i) but soft_argsort returns (p, i)
    // In Halide: out(p, i) means x=p, y=i
    // soft_argsort returns ret(p, i) so realize on Buffer(n_positions, n_elements)
    Halide::Runtime::Buffer<float> pmat(3, 3);  // pmat(x=p, y=i)
    r.realize(pmat);
    for (int p = 0; p < 3; ++p) {
        float row_sum = 0.0f;
        for (int i = 0; i < 3; ++i) row_sum += pmat(p, i);
        EXPECT_NEAR(row_sum, 1.0f, 1e-4f);
    }
}

TEST(SoftSort, SoftArgsort_NonNegative) {
    auto f = make_1d({1.0f, 5.0f, 3.0f}, "f_nn");
    auto r = soft_argsort(f, 3, 0.5f, 1.0f);
    Halide::Runtime::Buffer<float> pmat(3, 3);
    r.realize(pmat);
    for (int p = 0; p < 3; ++p)
        for (int i = 0; i < 3; ++i)
            EXPECT_GE(pmat(p, i), 0.0f);
}
