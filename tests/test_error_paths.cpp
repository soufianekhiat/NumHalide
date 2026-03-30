/// @file test_error_paths.cpp
/// @brief Tests for nh_require error paths — each test triggers a std::runtime_error
///        from a deliberate invalid argument.
#include <gtest/gtest.h>
#include "numhalide_all.h"
#include <stdexcept>
#include <vector>
using namespace numhalide;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static Halide::Func make_func_2d(const std::string& nm) {
    Halide::Func f(nm);
    Halide::Var x("x"), y("y");
    f(x, y) = Halide::cast<float>(x + y);
    return f;
}

static Halide::Func make_func_1d(const std::string& nm) {
    Halide::Func f(nm);
    Halide::Var x("x");
    f(x) = Halide::cast<float>(x);
    return f;
}

// ---------------------------------------------------------------------------
// view_slice error paths
// ---------------------------------------------------------------------------

// axis out of range (axis >= ndim)
TEST(ErrorPaths, ViewSlice_AxisOutOfRange) {
    Halide::Runtime::Buffer<float> buf(4, 4);
    buf.fill(0.0f);
    // 2D buffer has dims 0 and 1; axis=2 is out of range
    EXPECT_THROW(view_slice(buf, 2, 0, 2), std::runtime_error);
}

// start >= stop
TEST(ErrorPaths, ViewSlice_StartGeStop) {
    Halide::Runtime::Buffer<float> buf(4, 4);
    buf.fill(0.0f);
    // start=3 >= stop=2 is invalid
    EXPECT_THROW(view_slice(buf, 0, 3, 2), std::runtime_error);
}

// start == stop (empty slice)
TEST(ErrorPaths, ViewSlice_StartEqStop) {
    Halide::Runtime::Buffer<float> buf(4, 4);
    buf.fill(0.0f);
    EXPECT_THROW(view_slice(buf, 0, 2, 2), std::runtime_error);
}

// stop exceeds extent
TEST(ErrorPaths, ViewSlice_StopExceedsExtent) {
    Halide::Runtime::Buffer<float> buf(4, 4);
    buf.fill(0.0f);
    // extent along axis 0 is 4; stop=5 is beyond it
    EXPECT_THROW(view_slice(buf, 0, 0, 5), std::runtime_error);
}

// ---------------------------------------------------------------------------
// view_reshape error paths
// ---------------------------------------------------------------------------

// total element count mismatch
TEST(ErrorPaths, ViewReshape_ElementCountMismatch) {
    Halide::Runtime::Buffer<float> buf(4, 4);  // 16 elements
    buf.fill(0.0f);
    // requesting reshape to 3*5=15 elements — mismatch
    EXPECT_THROW(view_reshape(buf, {3, 5}), std::runtime_error);
}

// ---------------------------------------------------------------------------
// inplace error paths — 4D buffer (ndim=4 is unsupported)
// ---------------------------------------------------------------------------

TEST(ErrorPaths, InplaceThreshold_4D) {
    // Construct a 4D Runtime::Buffer manually
    halide_dimension_t dims[4] = {
        {0, 2, 1,  0},
        {0, 2, 2,  0},
        {0, 2, 4,  0},
        {0, 2, 8,  0},
    };
    std::vector<float> storage(16, 0.0f);
    Halide::Runtime::Buffer<float> buf(storage.data(), 4, dims);
    EXPECT_THROW(inplace_threshold(buf, 0.5f), std::runtime_error);
}

TEST(ErrorPaths, InplaceClamp_4D) {
    halide_dimension_t dims[4] = {
        {0, 2, 1,  0},
        {0, 2, 2,  0},
        {0, 2, 4,  0},
        {0, 2, 8,  0},
    };
    std::vector<float> storage(16, 0.0f);
    Halide::Runtime::Buffer<float> buf(storage.data(), 4, dims);
    EXPECT_THROW(inplace_clamp(buf, 0.0f, 1.0f), std::runtime_error);
}

TEST(ErrorPaths, InplaceScale_4D) {
    halide_dimension_t dims[4] = {
        {0, 2, 1,  0},
        {0, 2, 2,  0},
        {0, 2, 4,  0},
        {0, 2, 8,  0},
    };
    std::vector<float> storage(16, 0.0f);
    Halide::Runtime::Buffer<float> buf(storage.data(), 4, dims);
    EXPECT_THROW(inplace_scale(buf, 2.0f), std::runtime_error);
}

// ---------------------------------------------------------------------------
// la_large error paths — n > LA_LARGE_MAX_N (32)
// ---------------------------------------------------------------------------

TEST(ErrorPaths, CholeskyLarge_NTooLarge) {
    Halide::Func A = make_func_2d("chol_large_err");
    EXPECT_THROW(cholesky_large(A, 33), std::runtime_error);
}

TEST(ErrorPaths, QrLarge_NTooLarge) {
    Halide::Func A = make_func_2d("qr_large_err");
    EXPECT_THROW(qr_large(A, 33, 33), std::runtime_error);
}

TEST(ErrorPaths, SvdLarge_NTooLarge) {
    Halide::Func A = make_func_2d("svd_large_err");
    EXPECT_THROW(svd_large(A, 33, 33), std::runtime_error);
}

// qr_large: m < n is also invalid
TEST(ErrorPaths, QrLarge_MlessThanN) {
    Halide::Func A = make_func_2d("qr_large_mn_err");
    // m=3, n=4: m < n should throw
    EXPECT_THROW(qr_large(A, 3, 4), std::runtime_error);
}

// ---------------------------------------------------------------------------
// einsum error paths
// ---------------------------------------------------------------------------

// Contracted dimension mismatch: "ij,jk->ik" where j-dimension differs in A and B
TEST(ErrorPaths, Einsum_ContractedDimMismatch) {
    Halide::Func A = make_func_2d("ein_A_mismatch");  // shape {4, 3}
    Halide::Func B = make_func_2d("ein_B_mismatch");  // shape {5, 6}
    // A is shape {4, 3}: sub "ij" -> i=4, j=3
    // B is shape {5, 6}: sub "jk" -> j=5, k=6
    // j=3 vs j=5: conflicting extent -> should throw
    EXPECT_THROW(
        einsum("ij,jk->ik", A, shape_t{4, 3}, B, shape_t{5, 6}),
        std::runtime_error);
}

// Subscript letter count mismatch: subscript "ij" for a rank-1 tensor
TEST(ErrorPaths, Einsum_SubscriptLengthMismatch_A) {
    Halide::Func A = make_func_1d("ein_A_rank1");
    Halide::Func B = make_func_2d("ein_B_ok");
    // A has rank 1 but sub_A "ij" has length 2 -> should throw
    EXPECT_THROW(
        einsum("ij,jk->ik", A, shape_t{5}, B, shape_t{5, 5}),
        std::runtime_error);
}

// Subscript without "->" arrow
TEST(ErrorPaths, Einsum_ImplicitOutput_Valid) {
    // Implicit output "ij,jk" == "ij,jk->ik" — should NOT throw, produces valid Func
    Halide::Func A("ein_impl_A"), B("ein_impl_B");
    Halide::Var x, y;
    A(x, y) = Halide::cast<float>(x + 1);
    B(x, y) = Halide::cast<float>(y + 1);
    EXPECT_NO_THROW(einsum("ij,jk", A, shape_t{2, 3}, B, shape_t{3, 4}));
}

// ---------------------------------------------------------------------------
// matmul error paths
// ---------------------------------------------------------------------------

// Inner dimension mismatch: (3x4) @ (5x6) — inner dims 4 != 5
TEST(ErrorPaths, Matmul_InnerDimMismatch) {
    Halide::Func A = make_func_2d("mm_A");  // shape {3, 4} (rows=3, cols=4)
    Halide::Func B = make_func_2d("mm_B");  // shape {5, 6}
    // shape_a={3,4}: extents[1]=4, shape_b={5,6}: extents[0]=5 -> 4 != 5
    EXPECT_THROW(
        matmul(A, shape_t{3, 4}, B, shape_t{5, 6}),
        std::runtime_error);
}

// matmul with non-2D input
TEST(ErrorPaths, Matmul_NonTwoDInput) {
    Halide::Func A("mm_3d_a"), B("mm_3d_b");
    Halide::Var x, y, z;
    A(x, y, z) = Halide::cast<float>(x + y + z);
    B(x, y, z) = Halide::cast<float>(x + y + z);
    // rank 3 inputs should throw
    EXPECT_THROW(
        matmul(A, shape_t{2, 3, 4}, B, shape_t{4, 3, 2}),
        std::runtime_error);
}

// ---------------------------------------------------------------------------
// matvec error paths
// ---------------------------------------------------------------------------

// Column count of matrix != vector length
TEST(ErrorPaths, Matvec_DimMismatch) {
    Halide::Func M = make_func_2d("mv_M");  // shape {3, 4}: 3 rows, 4 cols
    Halide::Func v = make_func_1d("mv_v");  // shape {5}
    // matrix cols = 4, vector len = 5: mismatch
    EXPECT_THROW(
        matvec(M, shape_t{3, 4}, v, shape_t{5}),
        std::runtime_error);
}

// matvec with 2D vector (should require 1D)
TEST(ErrorPaths, Matvec_NonOneDVector) {
    Halide::Func M = make_func_2d("mv_M2");
    Halide::Func V = make_func_2d("mv_V2d");
    // vector rank = 2 should throw
    EXPECT_THROW(
        matvec(M, shape_t{3, 4}, V, shape_t{4, 1}),
        std::runtime_error);
}

// ---------------------------------------------------------------------------
// view_transpose error path
// ---------------------------------------------------------------------------

// view_transpose requires exactly 2D; 3D should throw
TEST(ErrorPaths, ViewTranspose_NonTwoD) {
    Halide::Runtime::Buffer<float> buf(3, 3, 3);
    buf.fill(0.0f);
    EXPECT_THROW(view_transpose(buf), std::runtime_error);
}

// ---------------------------------------------------------------------------
// infer_slice — step == 0 throws
// ---------------------------------------------------------------------------

TEST(ErrorPaths, InferSlice_ZeroStep) {
    shape_t s{4, 4};
    // step=0 should throw
    EXPECT_THROW(infer_slice(s, 0, 0, 4, 0), std::runtime_error);
}
