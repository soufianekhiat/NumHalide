/// @file einsum.h
/// @brief Einstein summation (einsum) for Halide::Func objects
///
/// Provides: einsum (1-input, 2-input), infer_einsum, infer_einsum1
///
/// Supported subscript forms (always with "->"):
///   "ij,jk->ik"   — matrix multiply
///   "ij->i"        — row sum (reduce axis 1)
///   "ii->"         — trace (diagonal sum → scalar wrapped in 1D)
///   "bij,bjk->bik" — batch matmul
///   "ij->ji"       — transpose / permute
///   "ij,ij->"      — inner product (sum of elementwise products)
///   "i,j->ij"      — outer product
///   "ij,ij->ij"    — Hadamard (elementwise) product

#pragma once

#include "common.h"
#include "shape.h"

#include <map>
#include <set>
#include <string>
#include <vector>
#include <stdexcept>

NS_NUM_HALIDE_BEGIN

// =============================================================================
// Internal helpers
// =============================================================================

namespace einsum_detail {

/// Split a string by a single-character delimiter, returning all parts.
inline std::vector<std::string> split_str(const std::string& s, char delim) {
    std::vector<std::string> parts;
    std::string cur;
    for (char ch : s) {
        if (ch == delim) {
            parts.push_back(cur);
            cur.clear();
        } else {
            cur += ch;
        }
    }
    parts.push_back(cur);
    return parts;
}

/// Parse subscript string "lhs->output" or implicit "lhs" (no "->").
/// Returns {input_subs, output_sub}.
/// input_subs has 1 or 2 elements (split by ',').
/// If "->" is absent, the output is inferred:
///   - 1-input: sorted unique chars appearing an odd number of times
///   - 2-input: sorted unique chars that appear in exactly one of the two input subscripts (XOR)
inline void parse_subscripts(const std::string& subscripts,
                              std::vector<std::string>& input_subs,
                              std::string& output_sub)
{
    auto arrow_pos = subscripts.find("->");
    if (arrow_pos == std::string::npos) {
        // Implicit output notation: infer output subscript
        input_subs = split_str(subscripts, ',');
        nh_require(input_subs.size() == 1u || input_subs.size() == 2u,
            "einsum: only 1- or 2-input forms supported, got %d inputs (subscript: %s)",
            (int)input_subs.size(), subscripts.c_str());

        if (input_subs.size() == 1u) {
            // 1-input: output = sorted unique chars appearing an odd number of times
            std::map<char, int> counts;
            for (char c : input_subs[0]) counts[c]++;
            output_sub.clear();
            for (auto& kv : counts) {
                if (kv.second % 2 != 0) output_sub += kv.first;
            }
            // already sorted (map iterates in order)
        } else {
            // 2-input: output = sorted unique chars in exactly one of the two inputs (XOR)
            std::set<char> in_a(input_subs[0].begin(), input_subs[0].end());
            std::set<char> in_b(input_subs[1].begin(), input_subs[1].end());
            output_sub.clear();
            for (char c : in_a) { if (!in_b.count(c)) output_sub += c; }
            for (char c : in_b) { if (!in_a.count(c)) output_sub += c; }
            std::sort(output_sub.begin(), output_sub.end());
        }
        return;
    }

    std::string lhs = subscripts.substr(0, arrow_pos);
    output_sub = subscripts.substr(arrow_pos + 2);
    input_subs = split_str(lhs, ',');

    nh_require(input_subs.size() == 1u || input_subs.size() == 2u,
        "einsum: only 1- or 2-input forms supported, got %d inputs (subscript: %s)",
        (int)input_subs.size(), subscripts.c_str());
}

/// Build extent map from input subscript + shape.
/// subscript letter at position p → shape.extents[p]
inline void build_extent_map(const std::string& sub, const shape_t& shape,
                              std::map<char, int>& extent_map)
{
    nh_require((int)sub.size() == shape.rank,
        "einsum: subscript '%s' length %d does not match shape rank %d",
        sub.c_str(), (int)sub.size(), shape.rank);
    for (int p = 0; p < (int)sub.size(); ++p) {
        char letter = sub[p];
        int ext = shape.extents[p];
        auto it = extent_map.find(letter);
        if (it != extent_map.end()) {
            nh_require(it->second == ext,
                "einsum: conflicting extents for letter '%c': %d vs %d",
                letter, it->second, ext);
        } else {
            extent_map[letter] = ext;
        }
    }
}

} // namespace einsum_detail

// =============================================================================
// infer_einsum1 — output shape for 1-input einsum
// =============================================================================

/// @brief Infer output shape for a 1-input einsum.
inline shape_t infer_einsum1(const std::string& subscripts, const shape_t& shape_A)
{
    std::vector<std::string> input_subs;
    std::string output_sub;
    einsum_detail::parse_subscripts(subscripts, input_subs, output_sub);

    nh_require(input_subs.size() == 1u,
        "infer_einsum1: expected 1-input subscript, got %d", (int)input_subs.size());

    std::map<char, int> extent_map;
    einsum_detail::build_extent_map(input_subs[0], shape_A, extent_map);

    if (output_sub.empty()) {
        // Full reduction → scalar wrapped in 1D
        return shape_t{1};
    }

    std::vector<int> out_extents;
    for (char c : output_sub) {
        nh_require(extent_map.count(c),
            "infer_einsum1: output letter '%c' not found in input subscripts", c);
        out_extents.push_back(extent_map.at(c));
    }
    return shape_t(out_extents);
}

// =============================================================================
// infer_einsum — output shape for 2-input einsum
// =============================================================================

/// @brief Infer output shape for a 2-input einsum.
inline shape_t infer_einsum(const std::string& subscripts,
                             const shape_t& shape_A, const shape_t& shape_B)
{
    std::vector<std::string> input_subs;
    std::string output_sub;
    einsum_detail::parse_subscripts(subscripts, input_subs, output_sub);

    nh_require(input_subs.size() == 2u,
        "infer_einsum: expected 2-input subscript, got %d", (int)input_subs.size());

    std::map<char, int> extent_map;
    einsum_detail::build_extent_map(input_subs[0], shape_A, extent_map);
    einsum_detail::build_extent_map(input_subs[1], shape_B, extent_map);

    if (output_sub.empty()) {
        return shape_t{1};
    }

    std::vector<int> out_extents;
    for (char c : output_sub) {
        nh_require(extent_map.count(c),
            "infer_einsum: output letter '%c' not found in input subscripts", c);
        out_extents.push_back(extent_map.at(c));
    }
    return shape_t(out_extents);
}

// =============================================================================
// einsum — 1-input
// =============================================================================

/// @brief 1-input einsum.
///
/// Examples:
///   einsum("ij->i",  A, {M,N})          — row sum
///   einsum("ij->ji", A, {M,N})          — transpose
///   einsum("ii->",   A, {N,N})          — trace → scalar in 1D buffer
///   einsum("ii->i",  A, {N,N})          — diagonal extraction
inline Halide::Func einsum(const std::string& subscripts,
                            Halide::Func A, const shape_t& shape_A,
                            const std::string& name = "einsum1")
{
    using namespace einsum_detail;

    std::vector<std::string> input_subs;
    std::string output_sub;
    parse_subscripts(subscripts, input_subs, output_sub);

    nh_require(input_subs.size() == 1u,
        "einsum (1-input): subscript must have exactly 1 input, got %d",
        (int)input_subs.size());

    const std::string& sub_A = input_subs[0];
    nh_require((int)sub_A.size() == shape_A.rank,
        "einsum (1-input): subscript A length %d != shape rank %d",
        (int)sub_A.size(), shape_A.rank);

    // Build extent map
    std::map<char, int> extent_map;
    build_extent_map(sub_A, shape_A, extent_map);

    // Determine contracted letters:
    // A letter is contracted if it appears in the input but NOT in the output.
    // However, a repeated letter in the input (e.g. "ii") with output "i" is free,
    // not contracted.  A repeated letter NOT in the output IS contracted (trace).
    std::set<char> output_letters(output_sub.begin(), output_sub.end());

    // Collect contracted letters (appear in sub_A but not in output_sub), deduped
    std::vector<char> contracted_letters;
    {
        std::set<char> seen;
        for (char c : sub_A) {
            if (!output_letters.count(c) && !seen.count(c)) {
                contracted_letters.push_back(c);
                seen.insert(c);
            }
        }
    }

    // Create Halide::Var for each output letter
    std::map<char, Halide::Var> out_vars;
    for (char c : output_sub) {
        if (!out_vars.count(c)) {
            out_vars[c] = Halide::Var(std::string(1, c));
        }
    }

    // Create RDom for contracted letters
    // Map contracted letter → rdom dimension index
    std::map<char, int> contracted_idx;
    Halide::RDom rdom;
    bool has_rdom = !contracted_letters.empty();
    if (has_rdom) {
        std::vector<Halide::Range> rdom_ranges;
        for (int i = 0; i < (int)contracted_letters.size(); ++i) {
            char c = contracted_letters[i];
            rdom_ranges.push_back(Halide::Range(0, extent_map.at(c)));
            contracted_idx[c] = i;
        }
        rdom = Halide::RDom(rdom_ranges, "rdom");
    }

    // Helper: get the Halide::Expr for a given letter (var or rdom)
    // Returns an Expr (either Var or rdom[i])
    auto letter_expr = [&](char c) -> Halide::Expr {
        if (output_letters.count(c)) {
            return out_vars.at(c);
        } else {
            return rdom[contracted_idx.at(c)];
        }
    };

    // Build args for A in Halide dimension order (innermost = last subscript pos)
    // For rank r, subscript sub_A:
    //   A(letter_expr[sub_A[r-1]], letter_expr[sub_A[r-2]], ..., letter_expr[sub_A[0]])
    int rank_A = shape_A.rank;
    std::vector<Halide::Expr> a_args;
    for (int p = rank_A - 1; p >= 0; --p) {
        a_args.push_back(letter_expr(sub_A[p]));
    }

    // Accumulator seed type: the input's own float type (f64 stays f64);
    // integer inputs keep the historical f32 accumulation.
    Halide::Type acc_t = A.types()[0];
    if (!acc_t.is_float()) acc_t = Halide::Float(32);

    // Full reduction (empty output)?
    if (output_sub.empty()) {
        // Wrap scalar in 1D Func
        Halide::Func raw(name + "_raw");
        raw() = Halide::cast(acc_t, 0);
        if (has_rdom) {
            raw() += A(a_args);
        } else {
            raw() = A(a_args);
        }
        Halide::Func ret(name);
        Halide::Var _x("_x");
        ret(_x) = raw();
        return ret;
    }

    // Build output args (Halide order: innermost = last output letter)
    int out_rank = (int)output_sub.size();
    std::vector<Halide::Expr> result_args;
    for (int p = out_rank - 1; p >= 0; --p) {
        result_args.push_back(out_vars.at(output_sub[p]));
    }

    Halide::Func ret(name);
    if (has_rdom) {
        // Initialize to zero, then accumulate
        ret(result_args) = Halide::cast(acc_t, 0);
        ret(result_args) += A(a_args);
    } else {
        // Pure mapping (e.g. transpose, diagonal extract)
        ret(result_args) = A(a_args);
    }

    return ret;
}

// =============================================================================
// einsum — 2-input
// =============================================================================

/// @brief 2-input einsum.
///
/// Examples:
///   einsum("ij,jk->ik",   A, {M,K}, B, {K,N})   — matmul
///   einsum("ij,ij->",     A, {M,N}, B, {M,N})   — inner product
///   einsum("i,j->ij",     a, {M},   b, {N})      — outer product
///   einsum("bij,bjk->bik",A, {B,M,K}, B, {B,K,N}) — batch matmul
///   einsum("ij,ij->ij",   A, {M,N}, B, {M,N})   — Hadamard product
///   einsum("ij,j->i",     A, {M,N}, v, {N})      — matrix-vector
inline Halide::Func einsum(const std::string& subscripts,
                            Halide::Func A, const shape_t& shape_A,
                            Halide::Func B, const shape_t& shape_B,
                            const std::string& name = "einsum")
{
    using namespace einsum_detail;

    std::vector<std::string> input_subs;
    std::string output_sub;
    parse_subscripts(subscripts, input_subs, output_sub);

    nh_require(input_subs.size() == 2u,
        "einsum (2-input): subscript must have exactly 2 inputs, got %d",
        (int)input_subs.size());

    const std::string& sub_A = input_subs[0];
    const std::string& sub_B = input_subs[1];

    nh_require((int)sub_A.size() == shape_A.rank,
        "einsum: subscript A length %d != shape_A rank %d",
        (int)sub_A.size(), shape_A.rank);
    nh_require((int)sub_B.size() == shape_B.rank,
        "einsum: subscript B length %d != shape_B rank %d",
        (int)sub_B.size(), shape_B.rank);

    // Build extent map (validates conflicting extents automatically)
    std::map<char, int> extent_map;
    build_extent_map(sub_A, shape_A, extent_map);
    build_extent_map(sub_B, shape_B, extent_map);

    // Determine contracted letters:
    // A letter that appears in lhs (sub_A or sub_B) but NOT in output_sub
    std::set<char> output_letters(output_sub.begin(), output_sub.end());
    std::set<char> lhs_letters;
    for (char c : sub_A) lhs_letters.insert(c);
    for (char c : sub_B) lhs_letters.insert(c);

    std::vector<char> contracted_letters;
    for (char c : lhs_letters) {
        if (!output_letters.count(c)) {
            contracted_letters.push_back(c);
        }
    }
    // Sort for determinism
    std::sort(contracted_letters.begin(), contracted_letters.end());

    // Create Halide::Var for each unique output letter
    std::map<char, Halide::Var> out_vars;
    for (char c : output_sub) {
        if (!out_vars.count(c)) {
            out_vars[c] = Halide::Var(std::string(1, c));
        }
    }

    // Create RDom for contracted letters
    std::map<char, int> contracted_idx;
    Halide::RDom rdom;
    bool has_rdom = !contracted_letters.empty();
    if (has_rdom) {
        std::vector<Halide::Range> rdom_ranges;
        for (int i = 0; i < (int)contracted_letters.size(); ++i) {
            char c = contracted_letters[i];
            rdom_ranges.push_back(Halide::Range(0, extent_map.at(c)));
            contracted_idx[c] = i;
        }
        rdom = Halide::RDom(rdom_ranges, "rdom");
    }

    // Helper: Halide::Expr for a given letter
    auto letter_expr = [&](char c) -> Halide::Expr {
        if (output_letters.count(c)) {
            return out_vars.at(c);
        } else {
            return rdom[contracted_idx.at(c)];
        }
    };

    // Build args for A (Halide order: innermost = last subscript pos)
    int rank_A = shape_A.rank;
    std::vector<Halide::Expr> a_args;
    for (int p = rank_A - 1; p >= 0; --p) {
        a_args.push_back(letter_expr(sub_A[p]));
    }

    // Build args for B
    int rank_B = shape_B.rank;
    std::vector<Halide::Expr> b_args;
    for (int p = rank_B - 1; p >= 0; --p) {
        b_args.push_back(letter_expr(sub_B[p]));
    }

    // Accumulator seed type: the type of the A*B product (so mixed and f64
    // inputs accumulate without truncation); integer products keep the
    // historical f32 accumulation.
    Halide::Type acc_t = (A(a_args) * B(b_args)).type();
    if (!acc_t.is_float()) acc_t = Halide::Float(32);

    // Full reduction (empty output)?
    if (output_sub.empty()) {
        Halide::Func raw(name + "_raw");
        raw() = Halide::cast(acc_t, 0);
        if (has_rdom) {
            raw() += A(a_args) * B(b_args);
        } else {
            raw() = A(a_args) * B(b_args);
        }
        Halide::Func ret(name);
        Halide::Var _x("_x");
        ret(_x) = raw();
        return ret;
    }

    // Build output args (innermost = last output letter)
    int out_rank = (int)output_sub.size();
    std::vector<Halide::Expr> result_args;
    for (int p = out_rank - 1; p >= 0; --p) {
        result_args.push_back(out_vars.at(output_sub[p]));
    }

    Halide::Func ret(name);
    if (has_rdom) {
        ret(result_args) = Halide::cast(acc_t, 0);
        ret(result_args) += A(a_args) * B(b_args);
    } else {
        // Pure (no contraction): elementwise product or permutation
        ret(result_args) = A(a_args) * B(b_args);
    }

    return ret;
}

NS_NUM_HALIDE_END
