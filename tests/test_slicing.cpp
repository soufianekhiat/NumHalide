/// @file test_slicing.cpp
/// @brief Tests for slice, take, transpose, moveaxis, expand_dims, squeeze

#include <gtest/gtest.h>
#include "numhalide_all.h"

using namespace numhalide;

// -----------------------------------------------------------------------------
// Slice Tests
// -----------------------------------------------------------------------------

TEST(Slice, Basic2DAxis0) {
	// 4x3 matrix, slice rows 1-3
	shape_t s = { 4, 3 };
	Halide::Func a("input");
	Halide::Var x, y;
	// a[row, col] = row * 10 + col
	a(x, y) = Halide::cast<float>(y * 10 + x);

	Halide::Func result = slice(a, s, 0, 1, 3, 1);
	shape_t out_shape = infer_slice(s, 0, 1, 3, 1);

	EXPECT_EQ(out_shape.rank, 2);
	EXPECT_EQ(out_shape.extents[0], 2);  // rows 1,2
	EXPECT_EQ(out_shape.extents[1], 3);  // columns unchanged

	Halide::Runtime::Buffer<float> out(out_shape.extents[1], out_shape.extents[0]);
	result.realize(out);

	// Row 0 of output = row 1 of input
	EXPECT_NEAR(out(0, 0), 10.0f, 1e-5f);  // [1,0]
	EXPECT_NEAR(out(1, 0), 11.0f, 1e-5f);  // [1,1]
	EXPECT_NEAR(out(2, 0), 12.0f, 1e-5f);  // [1,2]

	// Row 1 of output = row 2 of input
	EXPECT_NEAR(out(0, 1), 20.0f, 1e-5f);  // [2,0]
	EXPECT_NEAR(out(1, 1), 21.0f, 1e-5f);  // [2,1]
	EXPECT_NEAR(out(2, 1), 22.0f, 1e-5f);  // [2,2]
}

TEST(Slice, Basic2DAxis1) {
	// 3x4 matrix, slice columns 1-3
	shape_t s = { 3, 4 };
	Halide::Func a("input");
	Halide::Var x, y;
	a(x, y) = Halide::cast<float>(y * 10 + x);

	Halide::Func result = slice(a, s, 1, 1, 3, 1);
	shape_t out_shape = infer_slice(s, 1, 1, 3, 1);

	EXPECT_EQ(out_shape.extents[0], 3);  // rows unchanged
	EXPECT_EQ(out_shape.extents[1], 2);  // cols 1,2

	Halide::Runtime::Buffer<float> out(out_shape.extents[1], out_shape.extents[0]);
	result.realize(out);

	// Column 0 of output = column 1 of input
	EXPECT_NEAR(out(0, 0), 1.0f, 1e-5f);   // [0,1]
	EXPECT_NEAR(out(0, 1), 11.0f, 1e-5f);  // [1,1]
	EXPECT_NEAR(out(0, 2), 21.0f, 1e-5f);  // [2,1]
}

TEST(Slice, WithStep) {
	// 6x1 vector, take every 2nd element
	shape_t s = { 6, 1 };
	Halide::Func a("input");
	Halide::Var x, y;
	a(x, y) = Halide::cast<float>(y * 10);  // 0, 10, 20, 30, 40, 50

	Halide::Func result = slice(a, s, 0, 0, 6, 2);
	shape_t out_shape = infer_slice(s, 0, 0, 6, 2);

	EXPECT_EQ(out_shape.extents[0], 3);  // 3 elements

	Halide::Runtime::Buffer<float> out(out_shape.extents[1], out_shape.extents[0]);
	result.realize(out);

	EXPECT_NEAR(out(0, 0), 0.0f, 1e-5f);   // index 0
	EXPECT_NEAR(out(0, 1), 20.0f, 1e-5f);  // index 2
	EXPECT_NEAR(out(0, 2), 40.0f, 1e-5f);  // index 4
}

TEST(Slice, NegativeIndices) {
	// 5 element array, slice [-3:-1]
	shape_t s = { 5 };
	Halide::Func a("input");
	Halide::Var x;
	a(x) = Halide::cast<float>(x);  // 0, 1, 2, 3, 4

	Halide::Func result = slice(a, s, 0, -3, -1, 1);
	shape_t out_shape = infer_slice(s, 0, -3, -1, 1);

	EXPECT_EQ(out_shape.extents[0], 2);  // indices 2, 3

	Halide::Runtime::Buffer<float> out(out_shape.extents[0]);
	result.realize(out);

	EXPECT_NEAR(out(0), 2.0f, 1e-5f);
	EXPECT_NEAR(out(1), 3.0f, 1e-5f);
}

// -----------------------------------------------------------------------------
// Take Tests
// -----------------------------------------------------------------------------

TEST(Take, Basic) {
	// 4x3 matrix, take rows 0 and 2
	shape_t s = { 4, 3 };
	Halide::Func a("input");
	Halide::Var x, y;
	a(x, y) = Halide::cast<float>(y * 10 + x);

	Halide::Func result = take(a, s, {0, 2}, 0);
	shape_t out_shape = infer_take(s, 0, 2);

	EXPECT_EQ(out_shape.extents[0], 2);
	EXPECT_EQ(out_shape.extents[1], 3);

	Halide::Runtime::Buffer<float> out(out_shape.extents[1], out_shape.extents[0]);
	result.realize(out);

	// Row 0 of output = row 0 of input
	EXPECT_NEAR(out(0, 0), 0.0f, 1e-5f);
	EXPECT_NEAR(out(1, 0), 1.0f, 1e-5f);
	EXPECT_NEAR(out(2, 0), 2.0f, 1e-5f);

	// Row 1 of output = row 2 of input
	EXPECT_NEAR(out(0, 1), 20.0f, 1e-5f);
	EXPECT_NEAR(out(1, 1), 21.0f, 1e-5f);
	EXPECT_NEAR(out(2, 1), 22.0f, 1e-5f);
}

TEST(Take, SingleIndex) {
	// 3x2 matrix, take column 1
	shape_t s = { 3, 2 };
	Halide::Func a("input");
	Halide::Var x, y;
	a(x, y) = Halide::cast<float>(y * 10 + x);

	Halide::Func result = take(a, s, {1}, 1);
	shape_t out_shape = infer_take(s, 1, 1);

	EXPECT_EQ(out_shape.extents[0], 3);
	EXPECT_EQ(out_shape.extents[1], 1);

	Halide::Runtime::Buffer<float> out(out_shape.extents[1], out_shape.extents[0]);
	result.realize(out);

	EXPECT_NEAR(out(0, 0), 1.0f, 1e-5f);
	EXPECT_NEAR(out(0, 1), 11.0f, 1e-5f);
	EXPECT_NEAR(out(0, 2), 21.0f, 1e-5f);
}

// -----------------------------------------------------------------------------
// Transpose Tests
// -----------------------------------------------------------------------------

TEST(Transpose, 2D) {
	// 2x3 matrix -> 3x2
	shape_t s = { 2, 3 };
	Halide::Func a("input");
	Halide::Var x, y;
	// a[row, col] = row * 10 + col
	// [[0,1,2], [10,11,12]]
	a(x, y) = Halide::cast<float>(y * 10 + x);

	std::vector<int> axes_swap = {1, 0};
	Halide::Func result = transpose(a, s, axes_swap);
	shape_t out_shape = infer_transpose(s, axes_swap);

	EXPECT_EQ(out_shape.extents[0], 3);
	EXPECT_EQ(out_shape.extents[1], 2);

	Halide::Runtime::Buffer<float> out(out_shape.extents[1], out_shape.extents[0]);
	result.realize(out);

	// Transposed: [[0,10], [1,11], [2,12]]
	EXPECT_NEAR(out(0, 0), 0.0f, 1e-5f);   // [0,0] -> [0,0]
	EXPECT_NEAR(out(1, 0), 10.0f, 1e-5f);  // [0,1] came from [1,0]
	EXPECT_NEAR(out(0, 1), 1.0f, 1e-5f);   // [1,0] came from [0,1]
	EXPECT_NEAR(out(1, 1), 11.0f, 1e-5f);  // [1,1] -> [1,1]
	EXPECT_NEAR(out(0, 2), 2.0f, 1e-5f);   // [2,0] came from [0,2]
	EXPECT_NEAR(out(1, 2), 12.0f, 1e-5f);  // [2,1] came from [1,2]
}

TEST(Transpose, 2DSimple) {
	// Test the 2-arg transpose overload
	shape_t s = { 2, 3 };
	Halide::Func a("input");
	Halide::Var x, y;
	a(x, y) = Halide::cast<float>(y * 10 + x);

	Halide::Func result = transpose(a, s);
	shape_t out_shape = infer_transpose(s, {1, 0});

	Halide::Runtime::Buffer<float> out(out_shape.extents[1], out_shape.extents[0]);
	result.realize(out);

	EXPECT_NEAR(out(0, 0), 0.0f, 1e-5f);
	EXPECT_NEAR(out(1, 0), 10.0f, 1e-5f);
}

TEST(Transpose, 3D) {
	// 2x3x4 -> permute to 4x2x3
	shape_t s = { 2, 3, 4 };
	Halide::Func a("input");
	Halide::Var x, y, z;
	// a[dim0, dim1, dim2] = dim0*100 + dim1*10 + dim2
	a(x, y, z) = Halide::cast<float>(z * 100 + y * 10 + x);

	std::vector<int> axes_perm = {2, 0, 1};
	Halide::Func result = transpose(a, s, axes_perm);
	shape_t out_shape = infer_transpose(s, axes_perm);

	EXPECT_EQ(out_shape.extents[0], 4);  // was dim2
	EXPECT_EQ(out_shape.extents[1], 2);  // was dim0
	EXPECT_EQ(out_shape.extents[2], 3);  // was dim1

	Halide::Runtime::Buffer<float> out(out_shape.extents[2], out_shape.extents[1], out_shape.extents[0]);
	result.realize(out);

	// out[0,0,0] should be in[0,0,0] = 0
	EXPECT_NEAR(out(0, 0, 0), 0.0f, 1e-5f);
	// out[1,0,0] should be in[0,1,0] (axes{2,0,1} means out_dim0=in_dim2)
	// Actually: out[out_dim0, out_dim1, out_dim2] = in[in_dim0, in_dim1, in_dim2]
	// where out_dim0=in_dim2, out_dim1=in_dim0, out_dim2=in_dim1
	// So out[1,0,0] = in[0,0,1] = 0*100 + 0*10 + 1 = 1
	EXPECT_NEAR(out(0, 0, 1), 1.0f, 1e-5f);
}

// -----------------------------------------------------------------------------
// Moveaxis Tests
// -----------------------------------------------------------------------------

TEST(Moveaxis, Basic) {
	// 2x3x4 tensor, move axis 0 to position 2
	shape_t s = { 2, 3, 4 };
	Halide::Func a("input");
	Halide::Var x, y, z;
	a(x, y, z) = Halide::cast<float>(z * 100 + y * 10 + x);

	Halide::Func result = moveaxis(a, s, 0, 2);
	shape_t out_shape = infer_moveaxis(s, 0, 2);

	// Original: [2, 3, 4] -> axis 0 moves to position 2
	// Result: [3, 4, 2]
	EXPECT_EQ(out_shape.extents[0], 3);
	EXPECT_EQ(out_shape.extents[1], 4);
	EXPECT_EQ(out_shape.extents[2], 2);
}

TEST(Moveaxis, Negative) {
	// 2x3x4 tensor, move axis -1 to position 0
	shape_t s = { 2, 3, 4 };
	shape_t out_shape = infer_moveaxis(s, -1, 0);

	// axis -1 = axis 2 (size 4), move to position 0
	// Result: [4, 2, 3]
	EXPECT_EQ(out_shape.extents[0], 4);
	EXPECT_EQ(out_shape.extents[1], 2);
	EXPECT_EQ(out_shape.extents[2], 3);
}

// -----------------------------------------------------------------------------
// Expand dims Tests
// -----------------------------------------------------------------------------

TEST(ExpandDims, Basic) {
	// 3x4 matrix -> 3x1x4
	shape_t s = { 3, 4 };
	Halide::Func a("input");
	Halide::Var x, y;
	a(x, y) = Halide::cast<float>(y * 10 + x);

	Halide::Func result = expand_dims(a, s, 1);
	shape_t out_shape = infer_expand_dims(s, 1);

	EXPECT_EQ(out_shape.rank, 3);
	EXPECT_EQ(out_shape.extents[0], 3);
	EXPECT_EQ(out_shape.extents[1], 1);
	EXPECT_EQ(out_shape.extents[2], 4);

	Halide::Runtime::Buffer<float> out(out_shape.extents[2], out_shape.extents[1], out_shape.extents[0]);
	result.realize(out);

	// Values should be preserved
	EXPECT_NEAR(out(0, 0, 0), 0.0f, 1e-5f);
	EXPECT_NEAR(out(1, 0, 0), 1.0f, 1e-5f);
	EXPECT_NEAR(out(0, 0, 1), 10.0f, 1e-5f);
}

TEST(ExpandDims, AtEnd) {
	// 3x4 matrix -> 3x4x1
	shape_t s = { 3, 4 };
	shape_t out_shape = infer_expand_dims(s, 2);

	EXPECT_EQ(out_shape.rank, 3);
	EXPECT_EQ(out_shape.extents[0], 3);
	EXPECT_EQ(out_shape.extents[1], 4);
	EXPECT_EQ(out_shape.extents[2], 1);
}

TEST(ExpandDims, AtStart) {
	// 3x4 matrix -> 1x3x4
	shape_t s = { 3, 4 };
	shape_t out_shape = infer_expand_dims(s, 0);

	EXPECT_EQ(out_shape.rank, 3);
	EXPECT_EQ(out_shape.extents[0], 1);
	EXPECT_EQ(out_shape.extents[1], 3);
	EXPECT_EQ(out_shape.extents[2], 4);
}

// -----------------------------------------------------------------------------
// Squeeze Tests
// -----------------------------------------------------------------------------

TEST(Squeeze, Basic) {
	// 3x1x4 -> 3x4
	shape_t s = { 3, 1, 4 };
	Halide::Func a("input");
	Halide::Var x, y, z;
	a(x, y, z) = Halide::cast<float>(z * 100 + x);

	Halide::Func result = squeeze(a, s);
	shape_t out_shape = infer_squeeze(s);

	EXPECT_EQ(out_shape.rank, 2);
	EXPECT_EQ(out_shape.extents[0], 3);
	EXPECT_EQ(out_shape.extents[1], 4);

	Halide::Runtime::Buffer<float> out(out_shape.extents[1], out_shape.extents[0]);
	result.realize(out);

	EXPECT_NEAR(out(0, 0), 0.0f, 1e-5f);
	EXPECT_NEAR(out(1, 0), 1.0f, 1e-5f);
	EXPECT_NEAR(out(0, 1), 100.0f, 1e-5f);
}

TEST(Squeeze, SpecificAxis) {
	// 1x3x1x4 -> squeeze axis 0 -> 3x1x4
	shape_t s = { 1, 3, 1, 4 };
	shape_t out_shape = infer_squeeze(s, 0);

	EXPECT_EQ(out_shape.rank, 3);
	EXPECT_EQ(out_shape.extents[0], 3);
	EXPECT_EQ(out_shape.extents[1], 1);
	EXPECT_EQ(out_shape.extents[2], 4);
}

TEST(Squeeze, AllOnes) {
	// 1x1x1 -> 1 (scalar as 1D)
	shape_t s = { 1, 1, 1 };
	shape_t out_shape = infer_squeeze(s);

	EXPECT_EQ(out_shape.rank, 1);
	EXPECT_EQ(out_shape.extents[0], 1);
}
