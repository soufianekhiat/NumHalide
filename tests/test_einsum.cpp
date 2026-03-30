/// @file test_einsum.cpp
/// @brief Tests for einsum operations

#include <gtest/gtest.h>
#include "numhalide_all.h"

using namespace numhalide;

// =============================================================================
// Test helpers
// =============================================================================

/// Make a 1D Func backed by data, clamped for OOB safety.
static Halide::Func make_buf_1d(const std::vector<float>& data, int n,
                                 const std::string& nm)
{
    Halide::Buffer<float> b(n);
    for (int i = 0; i < n; ++i)
        b(i) = data[i];

    Halide::Func f(nm);
    Halide::Var x;
    f(x) = b(Halide::clamp(x, 0, n - 1));
    return f;
}

/// Make a 2D Func backed by row-major data, clamped for OOB safety.
/// shape: {rows, cols}  → Halide buffer (cols, rows) → f(x=col, y=row)
static Halide::Func make_buf_2d(const std::vector<float>& data, int rows, int cols,
                                 const std::string& nm)
{
    Halide::Buffer<float> b(cols, rows);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            b(c, r) = data[r * cols + c];

    Halide::Func f(nm);
    Halide::Var x, y;
    f(x, y) = b(Halide::clamp(x, 0, cols - 1),
                Halide::clamp(y, 0, rows - 1));
    return f;
}

/// Make a 3D Func backed by data (layout d,r,c = outermost to innermost).
/// shape: {D, rows, cols}  → Halide buffer (cols, rows, D) → f(x=col, y=row, z=D)
static Halide::Func make_buf_3d(const std::vector<float>& data,
                                 int D, int rows, int cols,
                                 const std::string& nm)
{
    Halide::Buffer<float> b(cols, rows, D);
    for (int d = 0; d < D; ++d)
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c)
                b(c, r, d) = data[d * rows * cols + r * cols + c];

    Halide::Func f(nm);
    Halide::Var x, y, z;
    f(x, y, z) = b(Halide::clamp(x, 0, cols - 1),
                   Halide::clamp(y, 0, rows - 1),
                   Halide::clamp(z, 0, D - 1));
    return f;
}

// =============================================================================
// Test 1: MatMul — einsum("ij,jk->ik", A{2,3}, B{3,4})
// =============================================================================

TEST(EinsumTest, MatMul_ij_jk_ik) {
    // A = [[1,2,3],[4,5,6]]  (2x3)
    // B = [[1,0,1,0],[0,1,0,1],[1,1,0,0]]  (3x4)
    // C[i,k] = sum_j A[i,j]*B[j,k]
    // C[0,0]=1+0+3=4  C[0,1]=2+2+3=7  ... wait let's be explicit
    // C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    // C[0,:] = 1*B[0,:]+2*B[1,:]+3*B[2,:]
    //        = [1,0,1,0]+[0,2,0,2]+[3,3,0,0] = [4,5,1,2]
    // C[1,:] = 4*B[0,:]+5*B[1,:]+6*B[2,:]
    //        = [4,0,4,0]+[0,5,0,5]+[6,6,0,0] = [10,11,4,5]

    const int M = 2, K = 3, N = 4;
    Halide::Func A = make_buf_2d({1,2,3,4,5,6}, M, K, "A");
    Halide::Func B = make_buf_2d({1,0,1,0, 0,1,0,1, 1,1,0,0}, K, N, "B");

    Halide::Func C = einsum("ij,jk->ik", A, {M,K}, B, {K,N});
    shape_t sc = infer_einsum("ij,jk->ik", {M,K}, {K,N});
    ASSERT_EQ(sc.rank, 2);
    ASSERT_EQ(sc.extents[0], M);
    ASSERT_EQ(sc.extents[1], N);

    // Halide buffer: (N, M) since innermost = cols = N
    Halide::Runtime::Buffer<float> out(N, M);
    C.realize(out);

    // out(col, row) = C[row, col]
    EXPECT_NEAR(out(0, 0), 4.0f,  1e-3f);
    EXPECT_NEAR(out(1, 0), 5.0f,  1e-3f);
    EXPECT_NEAR(out(2, 0), 1.0f,  1e-3f);
    EXPECT_NEAR(out(3, 0), 2.0f,  1e-3f);
    EXPECT_NEAR(out(0, 1), 10.0f, 1e-3f);
    EXPECT_NEAR(out(1, 1), 11.0f, 1e-3f);
    EXPECT_NEAR(out(2, 1), 4.0f,  1e-3f);
    EXPECT_NEAR(out(3, 1), 5.0f,  1e-3f);

    // Also verify it matches matmul()
    Halide::Func C2 = matmul(A, {M,K}, B, {K,N});
    Halide::Runtime::Buffer<float> out2(N, M);
    C2.realize(out2);
    for (int r = 0; r < M; ++r)
        for (int c = 0; c < N; ++c)
            EXPECT_NEAR(out(c, r), out2(c, r), 1e-3f);
}

// =============================================================================
// Test 2: InnerProduct — einsum("ij,ij->", A{3,4}, B{3,4})
// =============================================================================

TEST(EinsumTest, InnerProduct_ij_ij_scalar) {
    const int M = 3, N = 4;
    // A[i,j] = i*N + j + 1  (1..12)
    // B[i,j] = 1 for all
    // result = sum of A = 78
    std::vector<float> data_A, data_B;
    float expected = 0.0f;
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float a = static_cast<float>(i * N + j + 1);
            data_A.push_back(a);
            data_B.push_back(1.0f);
            expected += a * 1.0f;
        }

    Halide::Func A = make_buf_2d(data_A, M, N, "A");
    Halide::Func B = make_buf_2d(data_B, M, N, "B");

    Halide::Func result = einsum("ij,ij->", A, {M,N}, B, {M,N});
    shape_t sr = infer_einsum("ij,ij->", {M,N}, {M,N});
    ASSERT_EQ(sr.rank, 1);
    ASSERT_EQ(sr.extents[0], 1);

    Halide::Runtime::Buffer<float> out(1);
    result.realize(out);
    EXPECT_NEAR(out(0), expected, 1e-3f);
}

// =============================================================================
// Test 3: OuterProduct — einsum("i,j->ij", a{3}, b{4})
// =============================================================================

TEST(EinsumTest, OuterProduct_i_j_ij) {
    const int M = 3, N = 4;
    // a = [1,2,3], b = [10,20,30,40]
    Halide::Func a = make_buf_1d({1,2,3}, M, "a");
    Halide::Func b = make_buf_1d({10,20,30,40}, N, "b");

    Halide::Func result = einsum("i,j->ij", a, {M}, b, {N});
    shape_t sr = infer_einsum("i,j->ij", {M}, {N});
    ASSERT_EQ(sr.rank, 2);
    ASSERT_EQ(sr.extents[0], M);
    ASSERT_EQ(sr.extents[1], N);

    // Halide buffer: (N, M)
    Halide::Runtime::Buffer<float> out(N, M);
    result.realize(out);

    // out(j, i) = a[i] * b[j]
    float a_vals[] = {1,2,3};
    float b_vals[] = {10,20,30,40};
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            EXPECT_NEAR(out(j, i), a_vals[i] * b_vals[j], 1e-3f);
}

// =============================================================================
// Test 4: Transpose — einsum("ij->ji", A{3,4})
// =============================================================================

TEST(EinsumTest, Transpose_ij_ji) {
    const int M = 3, N = 4;
    // A[i,j] = i*N + j + 1
    std::vector<float> data;
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            data.push_back(static_cast<float>(i * N + j + 1));

    Halide::Func A = make_buf_2d(data, M, N, "A");
    Halide::Func T = einsum("ij->ji", A, {M,N});

    shape_t st = infer_einsum1("ij->ji", {M,N});
    ASSERT_EQ(st.rank, 2);
    ASSERT_EQ(st.extents[0], N);  // output[0] = j extent = N
    ASSERT_EQ(st.extents[1], M);  // output[1] = i extent = M

    // T has shape {N, M}: Halide buffer (M, N)
    Halide::Runtime::Buffer<float> out(M, N);
    T.realize(out);

    // out(i, j) = T[j,i] = A[i,j]
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            EXPECT_NEAR(out(i, j), data[i * N + j], 1e-3f);
}

// =============================================================================
// Test 5: Trace — einsum("ii->", A{4,4})
// =============================================================================

TEST(EinsumTest, Trace_ii_scalar) {
    const int N = 4;
    // A[i,j] = i*N + j + 1 → diag = A[0,0]+A[1,1]+A[2,2]+A[3,3] = 1+6+11+16 = 34
    std::vector<float> data;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            data.push_back(static_cast<float>(i * N + j + 1));

    Halide::Func A = make_buf_2d(data, N, N, "A");
    Halide::Func result = einsum("ii->", A, {N,N});

    Halide::Runtime::Buffer<float> out(1);
    result.realize(out);
    EXPECT_NEAR(out(0), 34.0f, 1e-3f);
}

// =============================================================================
// Test 6: RowSum — einsum("ij->i", A{3,4})
// =============================================================================

TEST(EinsumTest, RowSum_ij_i) {
    const int M = 3, N = 4;
    // A[i,j] = i*N + j + 1
    // row sums: [1+2+3+4, 5+6+7+8, 9+10+11+12] = [10, 26, 42]
    std::vector<float> data;
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            data.push_back(static_cast<float>(i * N + j + 1));

    Halide::Func A = make_buf_2d(data, M, N, "A");
    Halide::Func result = einsum("ij->i", A, {M,N});

    shape_t sr = infer_einsum1("ij->i", {M,N});
    ASSERT_EQ(sr.rank, 1);
    ASSERT_EQ(sr.extents[0], M);

    Halide::Runtime::Buffer<float> out(M);
    result.realize(out);
    EXPECT_NEAR(out(0), 10.0f, 1e-3f);
    EXPECT_NEAR(out(1), 26.0f, 1e-3f);
    EXPECT_NEAR(out(2), 42.0f, 1e-3f);
}

// =============================================================================
// Test 7: ColSum — einsum("ij->j", A{3,4})
// =============================================================================

TEST(EinsumTest, ColSum_ij_j) {
    const int M = 3, N = 4;
    // A[i,j] = i*N + j + 1
    // col sums: [1+5+9, 2+6+10, 3+7+11, 4+8+12] = [15, 18, 21, 24]
    std::vector<float> data;
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            data.push_back(static_cast<float>(i * N + j + 1));

    Halide::Func A = make_buf_2d(data, M, N, "A");
    Halide::Func result = einsum("ij->j", A, {M,N});

    shape_t sr = infer_einsum1("ij->j", {M,N});
    ASSERT_EQ(sr.rank, 1);
    ASSERT_EQ(sr.extents[0], N);

    Halide::Runtime::Buffer<float> out(N);
    result.realize(out);
    EXPECT_NEAR(out(0), 15.0f, 1e-3f);
    EXPECT_NEAR(out(1), 18.0f, 1e-3f);
    EXPECT_NEAR(out(2), 21.0f, 1e-3f);
    EXPECT_NEAR(out(3), 24.0f, 1e-3f);
}

// =============================================================================
// Test 8: BatchMatMul — einsum("bij,bjk->bik", A{2,3,4}, B{2,4,5})
// =============================================================================

TEST(EinsumTest, BatchMatMul_bij_bjk_bik) {
    const int BATCH = 2, M = 3, K = 4, N = 5;
    // Fill A and B with simple values and compare to batched_matmul
    // A[b,i,j] = b*M*K + i*K + j + 1
    // B[b,j,k] = b*K*N + j*N + k + 1
    std::vector<float> data_A, data_B;
    for (int b = 0; b < BATCH; ++b)
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < K; ++j)
                data_A.push_back(static_cast<float>(b * M * K + i * K + j + 1));
    for (int b = 0; b < BATCH; ++b)
        for (int j = 0; j < K; ++j)
            for (int k = 0; k < N; ++k)
                data_B.push_back(static_cast<float>(b * K * N + j * N + k + 1));

    Halide::Func A = make_buf_3d(data_A, BATCH, M, K, "A");
    Halide::Func B = make_buf_3d(data_B, BATCH, K, N, "B");

    Halide::Func result = einsum("bij,bjk->bik", A, {BATCH,M,K}, B, {BATCH,K,N});
    shape_t sr = infer_einsum("bij,bjk->bik", {BATCH,M,K}, {BATCH,K,N});
    ASSERT_EQ(sr.rank, 3);
    ASSERT_EQ(sr.extents[0], BATCH);
    ASSERT_EQ(sr.extents[1], M);
    ASSERT_EQ(sr.extents[2], N);

    // Halide buffer: (N, M, BATCH)
    Halide::Runtime::Buffer<float> out(N, M, BATCH);
    result.realize(out);

    // Verify by computing expected manually
    for (int b = 0; b < BATCH; ++b) {
        for (int i = 0; i < M; ++i) {
            for (int k = 0; k < N; ++k) {
                float expected = 0.0f;
                for (int j = 0; j < K; ++j) {
                    float a_val = static_cast<float>(b * M * K + i * K + j + 1);
                    float b_val = static_cast<float>(b * K * N + j * N + k + 1);
                    expected += a_val * b_val;
                }
                EXPECT_NEAR(out(k, i, b), expected, 0.1f);
            }
        }
    }
}

// =============================================================================
// Test 9: Hadamard — einsum("ij,ij->ij", A{3,4}, B{3,4})
// =============================================================================

TEST(EinsumTest, Hadamard_ij_ij_ij) {
    const int M = 3, N = 4;
    std::vector<float> data_A, data_B;
    for (int i = 0; i < M * N; ++i) {
        data_A.push_back(static_cast<float>(i + 1));
        data_B.push_back(static_cast<float>(2 * (i + 1)));
    }

    Halide::Func A = make_buf_2d(data_A, M, N, "A");
    Halide::Func B = make_buf_2d(data_B, M, N, "B");

    Halide::Func result = einsum("ij,ij->ij", A, {M,N}, B, {M,N});
    shape_t sr = infer_einsum("ij,ij->ij", {M,N}, {M,N});
    ASSERT_EQ(sr.rank, 2);
    ASSERT_EQ(sr.extents[0], M);
    ASSERT_EQ(sr.extents[1], N);

    Halide::Runtime::Buffer<float> out(N, M);
    result.realize(out);

    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float a_val = data_A[i * N + j];
            float b_val = data_B[i * N + j];
            EXPECT_NEAR(out(j, i), a_val * b_val, 1e-3f);
        }
}

// =============================================================================
// Test 10: VectorDot — einsum("i,i->", a{5}, b{5})
// =============================================================================

TEST(EinsumTest, VectorDot_i_i) {
    const int N = 5;
    // a = [1,2,3,4,5], b = [5,4,3,2,1]
    // dot = 1*5+2*4+3*3+4*2+5*1 = 5+8+9+8+5 = 35
    Halide::Func a = make_buf_1d({1,2,3,4,5}, N, "a");
    Halide::Func b = make_buf_1d({5,4,3,2,1}, N, "b");

    Halide::Func result = einsum("i,i->", a, {N}, b, {N});

    Halide::Runtime::Buffer<float> out(1);
    result.realize(out);
    EXPECT_NEAR(out(0), 35.0f, 1e-3f);
}

// =============================================================================
// Test 11: MatVec — einsum("ij,j->i", A{3,4}, v{4})
// =============================================================================

TEST(EinsumTest, MatVec_ij_j_i) {
    const int M = 3, N = 4;
    // A[i,j] = i*N + j + 1
    // v = [1,1,1,1]
    // result[i] = sum_j A[i,j] * 1 = row sums = [10, 26, 42]
    std::vector<float> data_A;
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            data_A.push_back(static_cast<float>(i * N + j + 1));

    Halide::Func A = make_buf_2d(data_A, M, N, "A");
    Halide::Func v = make_buf_1d({1,1,1,1}, N, "v");

    Halide::Func result = einsum("ij,j->i", A, {M,N}, v, {N});
    shape_t sr = infer_einsum("ij,j->i", {M,N}, {N});
    ASSERT_EQ(sr.rank, 1);
    ASSERT_EQ(sr.extents[0], M);

    Halide::Runtime::Buffer<float> out(M);
    result.realize(out);
    EXPECT_NEAR(out(0), 10.0f, 1e-3f);
    EXPECT_NEAR(out(1), 26.0f, 1e-3f);
    EXPECT_NEAR(out(2), 42.0f, 1e-3f);
}

// =============================================================================
// Test 12: DiagExtract — einsum("ii->i", A{4,4})
// =============================================================================

TEST(EinsumTest, DiagExtract_ii_i) {
    const int N = 4;
    // A[i,j] = i*N + j + 1 → diag = [1, 6, 11, 16]
    std::vector<float> data;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            data.push_back(static_cast<float>(i * N + j + 1));

    Halide::Func A = make_buf_2d(data, N, N, "A");
    Halide::Func result = einsum("ii->i", A, {N,N});

    shape_t sr = infer_einsum1("ii->i", {N,N});
    ASSERT_EQ(sr.rank, 1);
    ASSERT_EQ(sr.extents[0], N);

    Halide::Runtime::Buffer<float> out(N);
    result.realize(out);
    EXPECT_NEAR(out(0), 1.0f,  1e-3f);
    EXPECT_NEAR(out(1), 6.0f,  1e-3f);
    EXPECT_NEAR(out(2), 11.0f, 1e-3f);
    EXPECT_NEAR(out(3), 16.0f, 1e-3f);
}

// =============================================================================
// Test 13: Identity — einsum("ij->ij", A{3,4})
// =============================================================================

TEST(EinsumTest, Identity_ij_ij) {
    const int M = 3, N = 4;
    std::vector<float> data;
    for (int i = 0; i < M * N; ++i)
        data.push_back(static_cast<float>(i + 1));

    Halide::Func A = make_buf_2d(data, M, N, "A");
    Halide::Func result = einsum("ij->ij", A, {M,N});

    Halide::Runtime::Buffer<float> out(N, M);
    result.realize(out);

    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            EXPECT_NEAR(out(j, i), data[i * N + j], 1e-3f);
}

// =============================================================================
// Test 14: MatMul variant (3x3 @ 3x3 = 3x3), confirms matmul agreement
// =============================================================================

TEST(EinsumTest, MatMul_3x3_matches_matmul) {
    const int M = 3, K = 3, N = 3;
    // A = identity, B = [[1,2,3],[4,5,6],[7,8,9]]
    // A @ B = B
    std::vector<float> data_A = {1,0,0, 0,1,0, 0,0,1};
    std::vector<float> data_B = {1,2,3, 4,5,6, 7,8,9};

    Halide::Func A = make_buf_2d(data_A, M, K, "A");
    Halide::Func B = make_buf_2d(data_B, K, N, "B");

    Halide::Func C_einsum = einsum("ij,jk->ik", A, {M,K}, B, {K,N});
    Halide::Func C_matmul = matmul(A, {M,K}, B, {K,N});

    Halide::Runtime::Buffer<float> out_e(N, M);
    Halide::Runtime::Buffer<float> out_m(N, M);
    C_einsum.realize(out_e);
    C_matmul.realize(out_m);

    for (int r = 0; r < M; ++r)
        for (int c = 0; c < N; ++c)
            EXPECT_NEAR(out_e(c, r), out_m(c, r), 1e-3f);

    // Also verify result == B (since A is identity)
    for (int r = 0; r < M; ++r)
        for (int c = 0; c < N; ++c)
            EXPECT_NEAR(out_e(c, r), data_B[r * N + c], 1e-3f);
}

// =============================================================================
// EinsumExtended — additional coverage: 3D contractions, edge shapes, errors
// =============================================================================

// Test E1: Trace3x3_matches_sum — einsum1("ii->", A{3,3}) matches manual diagonal sum
TEST(EinsumExtended, Trace3x3_matches_sum)
{
    const int N = 3;
    // A[i,j] = i*N + j + 1  → diag = A[0,0]+A[1,1]+A[2,2] = 1 + 5 + 9 = 15
    std::vector<float> data;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            data.push_back(static_cast<float>(i * N + j + 1));

    float manual_trace = 0.0f;
    for (int i = 0; i < N; ++i)
        manual_trace += data[i * N + i];

    Halide::Func A = make_buf_2d(data, N, N, "trace3_A");
    Halide::Func result = einsum("ii->", A, {N, N}, "trace3");

    Halide::Runtime::Buffer<float> out(1);
    result.realize(out);

    EXPECT_NEAR(out(0), manual_trace, 1e-3f);
    EXPECT_NEAR(out(0), 15.0f,        1e-3f);
}

// Test E2: ColSum_4x3 — einsum1("ij->j", A{4,3}) computes column sums
TEST(EinsumExtended, ColSum_4x3)
{
    const int rows = 4, cols = 3;
    // A[i,j] = i*cols + j + 1
    // col sums: col0=1+4+7+10=22, col1=2+5+8+11=26, col2=3+6+9+12=30
    std::vector<float> data;
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            data.push_back(static_cast<float>(i * cols + j + 1));

    Halide::Func A = make_buf_2d(data, rows, cols, "cs4x3_A");
    // shape {rows, cols} = {4, 3}; subscript "ij->j": i has extent rows, j has extent cols
    Halide::Func result = einsum("ij->j", A, {rows, cols}, "colsum");

    shape_t sr = infer_einsum1("ij->j", {rows, cols});
    ASSERT_EQ(sr.rank, 1);
    ASSERT_EQ(sr.extents[0], cols);

    // Halide buffer has extent cols (j extent)
    Halide::Runtime::Buffer<float> out(cols);
    result.realize(out);

    // out(j) = sum_i A[i,j]
    for (int j = 0; j < cols; ++j) {
        float expected = 0.0f;
        for (int i = 0; i < rows; ++i)
            expected += data[i * cols + j];
        EXPECT_NEAR(out(j), expected, 1e-3f);
    }
}

// Test E3: Contraction_3D_ijk_jkl_il — einsum("ijk,jkl->il", A{2,3,4}, B{3,4,2})
// A has axes i=2, j=3, k=4; B has axes j=3, k=4, l=2
// Output "il": i=2, l=2  → Halide buffer (2, 2)
// result[i,l] = sum_{j,k} A[i,j,k] * B[j,k,l]
TEST(EinsumExtended, Contraction_3D_ijk_jkl_il)
{
    const int I = 2, J = 3, K = 4, L = 2;

    // A[i,j,k] = i*J*K + j*K + k + 1
    std::vector<float> data_A;
    for (int i = 0; i < I; ++i)
        for (int j = 0; j < J; ++j)
            for (int k = 0; k < K; ++k)
                data_A.push_back(static_cast<float>(i * J * K + j * K + k + 1));

    // B[j,k,l] = (j*K*L + k*L + l + 1) * 0.1f
    std::vector<float> data_B;
    for (int j = 0; j < J; ++j)
        for (int k = 0; k < K; ++k)
            for (int l = 0; l < L; ++l)
                data_B.push_back(static_cast<float>(j * K * L + k * L + l + 1) * 0.1f);

    // shape_A {I,J,K} = {2,3,4}; make_buf_3d(data, D=I, rows=J, cols=K)
    Halide::Func A = make_buf_3d(data_A, I, J, K, "c3d_A");
    // shape_B {J,K,L} = {3,4,2}; make_buf_3d(data, D=J, rows=K, cols=L)
    Halide::Func B = make_buf_3d(data_B, J, K, L, "c3d_B");

    // shape_t for A: extents in subscript order "ijk" → {I, J, K}
    // shape_t for B: extents in subscript order "jkl" → {J, K, L}
    Halide::Func result = einsum("ijk,jkl->il", A, {I, J, K}, B, {J, K, L}, "c3d_res");

    shape_t sr = infer_einsum("ijk,jkl->il", {I, J, K}, {J, K, L});
    ASSERT_EQ(sr.rank, 2);
    ASSERT_EQ(sr.extents[0], I);
    ASSERT_EQ(sr.extents[1], L);

    // Halide buffer: (L, I) since innermost = last output letter 'l' → extent L
    Halide::Runtime::Buffer<float> out(L, I);
    result.realize(out);

    // Verify result[i,l] = sum_{j,k} A[i,j,k] * B[j,k,l]
    for (int i = 0; i < I; ++i) {
        for (int l = 0; l < L; ++l) {
            float expected = 0.0f;
            for (int j = 0; j < J; ++j)
                for (int k = 0; k < K; ++k) {
                    float a_val = data_A[i * J * K + j * K + k];
                    float b_val = data_B[j * K * L + k * L + l];
                    expected += a_val * b_val;
                }
            // out(l, i) since Halide buffer innermost dim = l
            EXPECT_NEAR(out(l, i), expected, 0.1f)
                << "Mismatch at i=" << i << " l=" << l;
        }
    }
}

// Test E4: Hadamard_NonSquare — einsum("ij,ij->ij", A{3,5}, B{3,5})
TEST(EinsumExtended, Hadamard_NonSquare)
{
    const int rows = 3, cols = 5;
    std::vector<float> data_A, data_B;
    for (int k = 0; k < rows * cols; ++k) {
        data_A.push_back(static_cast<float>(k + 1));
        data_B.push_back(static_cast<float>(k + 2));
    }

    Halide::Func A = make_buf_2d(data_A, rows, cols, "had_A");
    Halide::Func B = make_buf_2d(data_B, rows, cols, "had_B");

    Halide::Func result = einsum("ij,ij->ij", A, {rows, cols}, B, {rows, cols}, "had_res");

    shape_t sr = infer_einsum("ij,ij->ij", {rows, cols}, {rows, cols});
    ASSERT_EQ(sr.rank, 2);
    ASSERT_EQ(sr.extents[0], rows);
    ASSERT_EQ(sr.extents[1], cols);

    // Halide buffer (cols, rows)
    Halide::Runtime::Buffer<float> out(cols, rows);
    result.realize(out);

    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            EXPECT_NEAR(out(j, i), data_A[i * cols + j] * data_B[i * cols + j], 1e-3f);
}

// Test E5: InnerProduct_Vector8 — einsum("i,i->", a{8}, b{8}) matches dot product
TEST(EinsumExtended, InnerProduct_Vector8)
{
    const int N = 8;
    std::vector<float> data_a, data_b;
    float expected = 0.0f;
    for (int i = 0; i < N; ++i) {
        float a = static_cast<float>(i + 1);
        float b = static_cast<float>(N - i);
        data_a.push_back(a);
        data_b.push_back(b);
        expected += a * b;
    }

    Halide::Func fa = make_buf_1d(data_a, N, "ip8_a");
    Halide::Func fb = make_buf_1d(data_b, N, "ip8_b");

    Halide::Func result = einsum("i,i->", fa, {N}, fb, {N}, "ip8_res");

    Halide::Runtime::Buffer<float> out(1);
    result.realize(out);

    EXPECT_NEAR(out(0), expected, 1e-3f);
}

// Test E6: BatchedTrace — einsum1("bii->b", A{3,3,3}) computes trace per batch element
// A[b,i,i] is the diagonal element for batch b.
// Data layout (outermost b, then i-row, then i-col):
//   A[b][r][c] = b*9 + r*3 + c + 1
// Trace[b] = A[b,0,0] + A[b,1,1] + A[b,2,2] = (b*9+1) + (b*9+5) + (b*9+9)
//           = 3*b*9 + 15
TEST(EinsumExtended, BatchedTrace)
{
    const int B = 3, N = 3;

    // A[b,i,j] stored as (b outermost, i next, j innermost)
    std::vector<float> data;
    for (int b = 0; b < B; ++b)
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                data.push_back(static_cast<float>(b * N * N + i * N + j + 1));

    // make_buf_3d(data, D=B, rows=N, cols=N) → f(x=j, y=i, z=b)
    Halide::Func A = make_buf_3d(data, B, N, N, "btrace_A");

    // subscript "bii->b": shape {B, N, N} in subscript order b,i,i
    // sub[0]='b'→extent B, sub[1]='i'→extent N, sub[2]='i'→extent N
    Halide::Func result = einsum("bii->b", A, {B, N, N}, "btrace");

    shape_t sr = infer_einsum1("bii->b", {B, N, N});
    ASSERT_EQ(sr.rank, 1);
    ASSERT_EQ(sr.extents[0], B);

    Halide::Runtime::Buffer<float> out(B);
    result.realize(out);

    // out(b) = trace of slice b = sum_i A[b,i,i]
    for (int b = 0; b < B; ++b) {
        float expected = 0.0f;
        for (int i = 0; i < N; ++i)
            expected += data[b * N * N + i * N + i];
        EXPECT_NEAR(out(b), expected, 1e-3f);
    }
}
