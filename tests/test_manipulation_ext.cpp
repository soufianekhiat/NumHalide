/// @file test_manipulation_ext.cpp
/// @brief Tests for extended manipulation operations

#include <gtest/gtest.h>
#include "numhalide_all.h"

using namespace numhalide;

// -----------------------------------------------------------------------------
// Flip Tests
// -----------------------------------------------------------------------------

TEST(ManipExt, Flip2DAxis0) {
    // Flip rows: [[1,2],[3,4],[5,6]] -> [[5,6],[3,4],[1,2]]
    shape_t shape = {3, 2};
    Halide::Func f("input");
    Halide::Var x, y;
    f(x, y) = y * 2 + x + 1;  // Row-major values 1-6

    auto result = flip(f, shape, 0, "flipped");

    Halide::Runtime::Buffer<int32_t> out(2, 3);
    result.realize(out);

    // Row 0 should now be old row 2: [5, 6]
    EXPECT_EQ(out(0, 0), 5);
    EXPECT_EQ(out(1, 0), 6);
    // Row 1 should be old row 1: [3, 4]
    EXPECT_EQ(out(0, 1), 3);
    EXPECT_EQ(out(1, 1), 4);
    // Row 2 should be old row 0: [1, 2]
    EXPECT_EQ(out(0, 2), 1);
    EXPECT_EQ(out(1, 2), 2);
}

TEST(ManipExt, Flip2DAxis1) {
    // Flip columns: [[1,2,3],[4,5,6]] -> [[3,2,1],[6,5,4]]
    shape_t shape = {2, 3};
    Halide::Func f("input");
    Halide::Var x, y;
    f(x, y) = y * 3 + x + 1;

    auto result = flip(f, shape, 1, "flipped");

    Halide::Runtime::Buffer<int32_t> out(3, 2);
    result.realize(out);

    // Row 0: [3, 2, 1]
    EXPECT_EQ(out(0, 0), 3);
    EXPECT_EQ(out(1, 0), 2);
    EXPECT_EQ(out(2, 0), 1);
    // Row 1: [6, 5, 4]
    EXPECT_EQ(out(0, 1), 6);
    EXPECT_EQ(out(1, 1), 5);
    EXPECT_EQ(out(2, 1), 4);
}

TEST(ManipExt, Flipud) {
    shape_t shape = {3, 2};
    Halide::Func f("input");
    Halide::Var x, y;
    f(x, y) = y * 10 + x;

    auto result = flipud(f, shape, "flipud_result");

    Halide::Runtime::Buffer<int32_t> out(2, 3);
    result.realize(out);

    EXPECT_EQ(out(0, 0), 20);  // Was row 2
    EXPECT_EQ(out(0, 2), 0);   // Was row 0
}

TEST(ManipExt, Fliplr) {
    shape_t shape = {2, 4};
    Halide::Func f("input");
    Halide::Var x, y;
    f(x, y) = x;  // [0,1,2,3] for each row

    auto result = fliplr(f, shape, "fliplr_result");

    Halide::Runtime::Buffer<int32_t> out(4, 2);
    result.realize(out);

    EXPECT_EQ(out(0, 0), 3);  // Was col 3
    EXPECT_EQ(out(1, 0), 2);  // Was col 2
    EXPECT_EQ(out(2, 0), 1);  // Was col 1
    EXPECT_EQ(out(3, 0), 0);  // Was col 0
}

// -----------------------------------------------------------------------------
// Rot90 Tests
// -----------------------------------------------------------------------------

TEST(ManipExt, Rot90Once) {
    // 2x3 matrix rotated 90 CCW becomes 3x2
    // [[1,2,3],[4,5,6]] -> [[3,6],[2,5],[1,4]]
    shape_t shape = {2, 3};
    Halide::Func f("input");
    Halide::Var x, y;
    f(x, y) = y * 3 + x + 1;

    auto result = rot90(f, shape, 1, "rot90_1");
    auto out_shape = infer_rot90(shape, 1);

    EXPECT_EQ(out_shape.extents[0], 3);  // New rows = old cols
    EXPECT_EQ(out_shape.extents[1], 2);  // New cols = old rows

    Halide::Runtime::Buffer<int32_t> out(2, 3);
    result.realize(out);

    // New row 0: [3, 6]
    EXPECT_EQ(out(0, 0), 3);
    EXPECT_EQ(out(1, 0), 6);
    // New row 1: [2, 5]
    EXPECT_EQ(out(0, 1), 2);
    EXPECT_EQ(out(1, 1), 5);
    // New row 2: [1, 4]
    EXPECT_EQ(out(0, 2), 1);
    EXPECT_EQ(out(1, 2), 4);
}

TEST(ManipExt, Rot90Twice) {
    // 180 degree rotation
    shape_t shape = {2, 3};
    Halide::Func f("input");
    Halide::Var x, y;
    f(x, y) = y * 3 + x + 1;

    auto result = rot90(f, shape, 2, "rot90_2");

    Halide::Runtime::Buffer<int32_t> out(3, 2);
    result.realize(out);

    // [[1,2,3],[4,5,6]] -> [[6,5,4],[3,2,1]]
    EXPECT_EQ(out(0, 0), 6);
    EXPECT_EQ(out(1, 0), 5);
    EXPECT_EQ(out(2, 0), 4);
    EXPECT_EQ(out(0, 1), 3);
    EXPECT_EQ(out(1, 1), 2);
    EXPECT_EQ(out(2, 1), 1);
}

TEST(ManipExt, Rot90Thrice) {
    // 270 CCW = 90 CW
    shape_t shape = {2, 3};
    Halide::Func f("input");
    Halide::Var x, y;
    f(x, y) = y * 3 + x + 1;

    auto result = rot90(f, shape, 3, "rot90_3");

    Halide::Runtime::Buffer<int32_t> out(2, 3);
    result.realize(out);

    // [[1,2,3],[4,5,6]] -> [[4,1],[5,2],[6,3]]
    EXPECT_EQ(out(0, 0), 4);
    EXPECT_EQ(out(1, 0), 1);
    EXPECT_EQ(out(0, 1), 5);
    EXPECT_EQ(out(1, 1), 2);
    EXPECT_EQ(out(0, 2), 6);
    EXPECT_EQ(out(1, 2), 3);
}

// -----------------------------------------------------------------------------
// Roll Tests
// -----------------------------------------------------------------------------

TEST(ManipExt, Roll1D) {
    // [0,1,2,3,4] rolled by 2 -> [3,4,0,1,2]
    shape_t shape = {5};
    Halide::Func f("input");
    Halide::Var x;
    f(x) = x;

    auto result = roll(f, shape, 2, 0, "rolled");

    Halide::Runtime::Buffer<int32_t> out(5);
    result.realize(out);

    EXPECT_EQ(out(0), 3);
    EXPECT_EQ(out(1), 4);
    EXPECT_EQ(out(2), 0);
    EXPECT_EQ(out(3), 1);
    EXPECT_EQ(out(4), 2);
}

TEST(ManipExt, Roll2DAxis) {
    // Roll along axis 0 (rows)
    shape_t shape = {3, 2};
    Halide::Func f("input");
    Halide::Var x, y;
    f(x, y) = y * 10;  // Rows: 0, 10, 20

    auto result = roll(f, shape, 1, 0, "rolled");

    Halide::Runtime::Buffer<int32_t> out(2, 3);
    result.realize(out);

    // Row 0 should now have value 20 (old row 2)
    EXPECT_EQ(out(0, 0), 20);
    // Row 1 should have value 0 (old row 0)
    EXPECT_EQ(out(0, 1), 0);
    // Row 2 should have value 10 (old row 1)
    EXPECT_EQ(out(0, 2), 10);
}

TEST(ManipExt, RollNegative) {
    // Roll by -1 = roll by 4 for size 5
    shape_t shape = {5};
    Halide::Func f("input");
    Halide::Var x;
    f(x) = x;

    auto result = roll(f, shape, -1, 0, "rolled");

    Halide::Runtime::Buffer<int32_t> out(5);
    result.realize(out);

    // [0,1,2,3,4] rolled by -1 -> [1,2,3,4,0]
    EXPECT_EQ(out(0), 1);
    EXPECT_EQ(out(4), 0);
}

// -----------------------------------------------------------------------------
// Tile Tests
// -----------------------------------------------------------------------------

TEST(ManipExt, Tile2D) {
    // Tile 2x2 array by (2, 3)
    shape_t shape = {2, 2};
    Halide::Func f("input");
    Halide::Var x, y;
    f(x, y) = y * 2 + x;  // [[0,1],[2,3]]

    auto result = tile(f, shape, {2, 3}, "tiled");
    auto out_shape = infer_tile(shape, {2, 3});

    EXPECT_EQ(out_shape.extents[0], 4);
    EXPECT_EQ(out_shape.extents[1], 6);

    Halide::Runtime::Buffer<int32_t> out(6, 4);
    result.realize(out);

    // Check tiling pattern
    EXPECT_EQ(out(0, 0), 0);
    EXPECT_EQ(out(2, 0), 0);  // Tiled col
    EXPECT_EQ(out(4, 0), 0);  // Tiled col
    EXPECT_EQ(out(0, 2), 0);  // Tiled row
    EXPECT_EQ(out(1, 1), 3);  // (1,1) in original
    EXPECT_EQ(out(3, 3), 3);  // Tiled (1,1)
}

// -----------------------------------------------------------------------------
// Repeat Tests
// -----------------------------------------------------------------------------

TEST(ManipExt, Repeat1D) {
    // [0,1,2] with repeat=3 -> [0,0,0,1,1,1,2,2,2]
    shape_t shape = {3};
    Halide::Func f("input");
    Halide::Var x;
    f(x) = x;

    auto result = repeat(f, shape, 3, 0, "repeated");
    auto out_shape = infer_repeat(shape, 3, 0);

    EXPECT_EQ(out_shape.extents[0], 9);

    Halide::Runtime::Buffer<int32_t> out(9);
    result.realize(out);

    EXPECT_EQ(out(0), 0);
    EXPECT_EQ(out(1), 0);
    EXPECT_EQ(out(2), 0);
    EXPECT_EQ(out(3), 1);
    EXPECT_EQ(out(4), 1);
    EXPECT_EQ(out(5), 1);
    EXPECT_EQ(out(6), 2);
    EXPECT_EQ(out(7), 2);
    EXPECT_EQ(out(8), 2);
}

TEST(ManipExt, Repeat2DAxis1) {
    // [[0,1],[2,3]] with repeat=2 on axis 1 -> [[0,0,1,1],[2,2,3,3]]
    shape_t shape = {2, 2};
    Halide::Func f("input");
    Halide::Var x, y;
    f(x, y) = y * 2 + x;

    auto result = repeat(f, shape, 2, 1, "repeated");

    Halide::Runtime::Buffer<int32_t> out(4, 2);
    result.realize(out);

    EXPECT_EQ(out(0, 0), 0);
    EXPECT_EQ(out(1, 0), 0);
    EXPECT_EQ(out(2, 0), 1);
    EXPECT_EQ(out(3, 0), 1);
}

// -----------------------------------------------------------------------------
// Pad Tests
// -----------------------------------------------------------------------------

TEST(ManipExt, PadConstant) {
    // Pad 2x3 with 1 on all sides
    shape_t shape = {2, 3};
    Halide::Func f("input");
    Halide::Var x, y;
    f(x, y) = Halide::cast<float>(y * 3 + x + 1);

    auto result = pad(f, shape, {{1, 1}, {1, 1}}, PadMode::Constant, 0.0f, "padded");
    auto out_shape = infer_pad(shape, {{1, 1}, {1, 1}});

    EXPECT_EQ(out_shape.extents[0], 4);
    EXPECT_EQ(out_shape.extents[1], 5);

    Halide::Runtime::Buffer<float> out(5, 4);
    result.realize(out);

    // Check padding (should be 0)
    EXPECT_FLOAT_EQ(out(0, 0), 0.0f);  // Padded
    EXPECT_FLOAT_EQ(out(4, 3), 0.0f);  // Padded

    // Check original values shifted
    EXPECT_FLOAT_EQ(out(1, 1), 1.0f);  // Original (0,0)
    EXPECT_FLOAT_EQ(out(3, 2), 6.0f);  // Original (2,1)
}

TEST(ManipExt, PadEdge) {
    // Pad with edge values
    shape_t shape = {2, 3};
    Halide::Func f("input");
    Halide::Var x, y;
    f(x, y) = Halide::cast<float>(y * 3 + x + 1);

    auto result = pad(f, shape, {{1, 1}, {1, 1}}, PadMode::Edge, 0.0f, "padded");

    Halide::Runtime::Buffer<float> out(5, 4);
    result.realize(out);

    // Top-left corner should repeat value at (0,0) = 1
    EXPECT_FLOAT_EQ(out(0, 0), 1.0f);
    // Bottom-right should repeat value at (2,1) = 6
    EXPECT_FLOAT_EQ(out(4, 3), 6.0f);
}

TEST(ManipExt, PadReflect) {
    // Pad with reflection (not including edge)
    shape_t shape = {3, 4};
    Halide::Func f("input");
    Halide::Var x, y;
    f(x, y) = Halide::cast<float>(y * 4 + x);

    auto result = pad(f, shape, {{1, 1}, {1, 1}}, PadMode::Reflect, 0.0f, "padded");

    Halide::Runtime::Buffer<float> out(6, 5);
    result.realize(out);

    // Original value at (0,0) should be at (1,1) in output
    EXPECT_FLOAT_EQ(out(1, 1), 0.0f);
    // Reflection: (0,0) in output should reflect from (1,0) in original = 1
    EXPECT_FLOAT_EQ(out(0, 1), 1.0f);
}

TEST(ManipExt, PadSymmetric) {
    // Pad with symmetric reflection (including edge)
    shape_t shape = {2, 3};
    Halide::Func f("input");
    Halide::Var x, y;
    f(x, y) = Halide::cast<float>(y * 3 + x);

    auto result = pad(f, shape, {{1, 0}, {1, 0}}, PadMode::Symmetric, 0.0f, "padded");

    Halide::Runtime::Buffer<float> out(4, 3);
    result.realize(out);

    // Symmetric reflection includes edge, so index -1 reflects to 0
    EXPECT_FLOAT_EQ(out(0, 1), 0.0f);  // Reflects (0,0)
    EXPECT_FLOAT_EQ(out(1, 1), 0.0f);  // Original (0,0)
}
