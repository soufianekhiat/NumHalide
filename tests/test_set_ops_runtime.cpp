/// @file test_set_ops_runtime.cpp
/// @brief Tests for the FIXED-SIZE positional / zero-fill set-op variants
/// with RUNTIME (Halide::Expr) sizes: count_unique (0-D), mark_unique
/// (Int32 marks), unique_zerofill, in1d, intersect1d_zerofill,
/// setdiff1d_zerofill, union1d_positional. Each is pinned against known
/// values, an integer-element pin proves type-genericity, and the
/// count_unique n==0 / n==1 edge behavior is pinned to its documented
/// contract (result is 1 in both cases — the encoding cannot signal empty).

#include <gtest/gtest.h>
#include "numhalide_all.h"

using namespace numhalide;

namespace {

Halide::Func make_coeffs(std::initializer_list<float> vals, char const* nm) {
	Halide::Buffer<float> buf((int)vals.size());
	int i = 0;
	for (float v : vals) buf(i++) = v;
	Halide::Func f(nm);
	Halide::Var x;
	f(x) = buf(x);
	return f;
}

Halide::Func make_icoeffs(std::initializer_list<int> vals, char const* nm) {
	Halide::Buffer<int> buf((int)vals.size());
	int i = 0;
	for (int v : vals) buf(i++) = v;
	Halide::Func f(nm);
	Halide::Var x;
	f(x) = buf(x);
	return f;
}

} // namespace

TEST(SetOpsRuntime, CountUniquePin) {
	// Sorted [1,1,2,3,3] -> 3 unique elements. 0-D output.
	Halide::Func f = make_coeffs({1.0f, 1.0f, 2.0f, 3.0f, 3.0f}, "cu_f");

	Halide::Func rt = count_unique(f, Halide::Expr(5), "cu_rt");
	auto out = Halide::Runtime::Buffer<int>::make_scalar();
	rt.realize(out);
	EXPECT_EQ(out(), 3);
}

TEST(SetOpsRuntime, CountUniqueEdgePins) {
	// n == 1: single element -> 1 (the reduction domain [1, n) is empty).
	Halide::Func f1 = make_coeffs({7.0f}, "cue_f1");
	Halide::Func rt1 = count_unique(f1, Halide::Expr(1), "cue_rt1");
	auto out1 = Halide::Runtime::Buffer<int>::make_scalar();
	rt1.realize(out1);
	EXPECT_EQ(out1(), 1);

	// n == 0: documented contract — the reduction domain is still empty
	// and the result is STILL 1. This fixed-size encoding has no way to
	// signal an empty input; n == 0 is out of contract but must not read
	// the input (RDom extent is max(n-1, 0) = 0).
	Halide::Func f0 = make_coeffs({42.0f}, "cue_f0");
	Halide::Func rt0 = count_unique(f0, Halide::Expr(0), "cue_rt0");
	auto out0 = Halide::Runtime::Buffer<int>::make_scalar();
	rt0.realize(out0);
	EXPECT_EQ(out0(), 1);
}

TEST(SetOpsRuntime, MarkUniquePin) {
	// Sorted [1,1,2,3,3] -> marks [1,0,1,1,0].
	Halide::Func f = make_coeffs({1.0f, 1.0f, 2.0f, 3.0f, 3.0f}, "mu_f");

	Halide::Func rt = mark_unique(f, Halide::Expr(5), "mu_rt");
	Halide::Runtime::Buffer<int> out(5);
	rt.realize(out);

	int const expected[] = {1, 0, 1, 1, 0};
	for (int i = 0; i < 5; ++i)
		EXPECT_EQ(out(i), expected[i]) << "mark[" << i << "]";
}

TEST(SetOpsRuntime, UniqueZerofillPin) {
	// Sorted [1,1,2,3,3] -> first occurrences kept, duplicates zeroed:
	// [1,0,2,3,0].
	Halide::Func f = make_coeffs({1.0f, 1.0f, 2.0f, 3.0f, 3.0f}, "uz_f");

	Halide::Func rt = unique_zerofill(f, Halide::Expr(5), "uz_rt");
	Halide::Runtime::Buffer<float> out(5);
	rt.realize(out);

	float const expected[] = {1.0f, 0.0f, 2.0f, 3.0f, 0.0f};
	for (int i = 0; i < 5; ++i)
		EXPECT_NEAR(out(i), expected[i], 1e-6f) << "uz[" << i << "]";
}

TEST(SetOpsRuntime, In1dPinSortedAndUnsortedSet) {
	// values [1,4,3] in test set {1,2,3} -> [1,0,1].
	Halide::Func vals = make_coeffs({1.0f, 4.0f, 3.0f}, "in_vals");
	Halide::Func set_sorted = make_coeffs({1.0f, 2.0f, 3.0f}, "in_set_s");

	Halide::Func rt = in1d(vals, set_sorted, Halide::Expr(3), "in_rt");
	Halide::Runtime::Buffer<int> out(3);
	rt.realize(out);

	int const expected[] = {1, 0, 1};
	for (int i = 0; i < 3; ++i)
		EXPECT_EQ(out(i), expected[i]) << "in1d[" << i << "]";

	// The linear scan never exploits ordering: an unsorted permutation of
	// the same set gives identical results.
	Halide::Func set_unsorted = make_coeffs({3.0f, 1.0f, 2.0f}, "in_set_u");
	Halide::Func rt_u = in1d(vals, set_unsorted, Halide::Expr(3), "in_rt_u");
	Halide::Runtime::Buffer<int> out_u(3);
	rt_u.realize(out_u);
	for (int i = 0; i < 3; ++i)
		EXPECT_EQ(out_u(i), expected[i]) << "in1d_unsorted[" << i << "]";
}

TEST(SetOpsRuntime, IntersectZerofillPin) {
	// Unsorted fixtures: a = [5,2,9,2], b = [2,9,7].
	// Keep a[i] where present in b, else 0 -> [0,2,9,2].
	Halide::Func a = make_coeffs({5.0f, 2.0f, 9.0f, 2.0f}, "ix_a");
	Halide::Func b = make_coeffs({2.0f, 9.0f, 7.0f}, "ix_b");

	Halide::Func rt = intersect1d_zerofill(a, b, Halide::Expr(3), "ix_rt");
	Halide::Runtime::Buffer<float> out(4);
	rt.realize(out);

	float const expected[] = {0.0f, 2.0f, 9.0f, 2.0f};
	for (int i = 0; i < 4; ++i)
		EXPECT_NEAR(out(i), expected[i], 1e-6f) << "ix[" << i << "]";
}

TEST(SetOpsRuntime, SetdiffZerofillPin) {
	// Same unsorted fixtures: a = [5,2,9,2], b = [2,9,7].
	// Zero a[i] where present in b, keep otherwise -> [5,0,0,0].
	Halide::Func a = make_coeffs({5.0f, 2.0f, 9.0f, 2.0f}, "sd_a");
	Halide::Func b = make_coeffs({2.0f, 9.0f, 7.0f}, "sd_b");

	Halide::Func rt = setdiff1d_zerofill(a, b, Halide::Expr(3), "sd_rt");
	Halide::Runtime::Buffer<float> out(4);
	rt.realize(out);

	float const expected[] = {5.0f, 0.0f, 0.0f, 0.0f};
	for (int i = 0; i < 4; ++i)
		EXPECT_NEAR(out(i), expected[i], 1e-6f) << "sd[" << i << "]";
}

TEST(SetOpsRuntime, UnionPositionalPin) {
	// Pointwise maximum: a = [1,5,2], b = [3,4,1] -> [3,5,2].
	Halide::Func a = make_coeffs({1.0f, 5.0f, 2.0f}, "up_a");
	Halide::Func b = make_coeffs({3.0f, 4.0f, 1.0f}, "up_b");

	Halide::Func rt = union1d_positional(a, b, "up_rt");
	Halide::Runtime::Buffer<float> out(3);
	rt.realize(out);

	float const expected[] = {3.0f, 5.0f, 2.0f};
	for (int i = 0; i < 3; ++i)
		EXPECT_NEAR(out(i), expected[i], 1e-6f) << "up[" << i << "]";
}

TEST(SetOpsRuntime, IntegerElementPins) {
	// i32 elements throughout — comparisons and zeros stay in the element
	// type, proving the runtime forms are type-generic (no float
	// promotion).

	// count_unique on sorted ints [4,4,6] -> 2.
	Halide::Func cf = make_icoeffs({4, 4, 6}, "int_cu_f");
	Halide::Func cu = count_unique(cf, Halide::Expr(3), "int_cu_rt");
	auto cu_out = Halide::Runtime::Buffer<int>::make_scalar();
	cu.realize(cu_out);
	EXPECT_EQ(cu_out(), 2);

	// unique_zerofill on sorted ints [4,4,6] -> [4,0,6], output type i32.
	Halide::Func uz = unique_zerofill(cf, Halide::Expr(3), "int_uz_rt");
	Halide::Runtime::Buffer<int> uz_out(3);
	uz.realize(uz_out);
	EXPECT_EQ(uz_out(0), 4);
	EXPECT_EQ(uz_out(1), 0);
	EXPECT_EQ(uz_out(2), 6);

	// in1d: values [4,9,6] in set {9,6} -> [0,1,1].
	Halide::Func iv = make_icoeffs({4, 9, 6}, "int_in_vals");
	Halide::Func is = make_icoeffs({9, 6}, "int_in_set");
	Halide::Func in = in1d(iv, is, Halide::Expr(2), "int_in_rt");
	Halide::Runtime::Buffer<int> in_out(3);
	in.realize(in_out);
	EXPECT_EQ(in_out(0), 0);
	EXPECT_EQ(in_out(1), 1);
	EXPECT_EQ(in_out(2), 1);

	// setdiff1d_zerofill on ints: a = [3,7], b = [7] -> [3,0], output i32.
	Halide::Func da = make_icoeffs({3, 7}, "int_sd_a");
	Halide::Func db = make_icoeffs({7}, "int_sd_b");
	Halide::Func sd = setdiff1d_zerofill(da, db, Halide::Expr(1), "int_sd_rt");
	Halide::Runtime::Buffer<int> sd_out(2);
	sd.realize(sd_out);
	EXPECT_EQ(sd_out(0), 3);
	EXPECT_EQ(sd_out(1), 0);

	// union1d_positional on ints: a = [1,9], b = [2,3] -> [2,9].
	Halide::Func ua = make_icoeffs({1, 9}, "int_up_a");
	Halide::Func ub = make_icoeffs({2, 3}, "int_up_b");
	Halide::Func up = union1d_positional(ua, ub, "int_up_rt");
	Halide::Runtime::Buffer<int> up_out(2);
	up.realize(up_out);
	EXPECT_EQ(up_out(0), 2);
	EXPECT_EQ(up_out(1), 9);
}
