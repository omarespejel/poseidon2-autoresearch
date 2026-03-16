#!/usr/bin/env python3
"""Karpathy-compatible autonomous optimization loop for AutoPoseidon targets.

This is the canonical loop entrypoint in compatibility mode:
- prepare.py: fixed evaluator + utilities
- train.py: editable optimization loop
- program.md: human-authored instructions
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import prepare

ROOT = Path(__file__).resolve().parent
PROMPT_TEMPLATE = ROOT / "config" / "prompt_template.md"
ARTIFACTS_DIR = ROOT / "artifacts"
ALLOWED_TARGET_OVERRIDE_KEYS = {
    "min_improvement_abs",
    "min_improvement_rel",
    "min_improvement_rel_mode",
    "min_improvement_rel_sigma",
    "min_improvement_rel_min",
    "min_improvement_rel_max",
    "confirm_repeats",
    "max_rel_stdev",
    "min_effect_sigma",
    "ci_z",
    "require_ci_separation",
    "min_series_samples",
    "ab_repeats",
    "blocked_mutation_ttl",
}


def load_prompt_template() -> str:
    return PROMPT_TEMPLATE.read_text()


def build_prompt(
    template: str, *, target_name: str, metric_name: str, language: str, source_code: str
) -> str:
    return (
        template.replace("${target_name}", target_name)
        .replace("${metric_name}", metric_name)
        .replace("${language}", language)
        .replace("${source_code}", source_code)
    )


def remove_first_marked_line(text: str, marker: str) -> tuple[str, bool]:
    out: list[str] = []
    removed = False
    for line in text.splitlines():
        if (not removed) and marker in line:
            removed = True
            continue
        out.append(line)
    return "\n".join(out) + "\n", removed


def regex_replace_once(source: str, pattern: str, repl: str, *, flags: int = 0) -> tuple[str, bool]:
    candidate = re.sub(pattern, repl, source, count=1, flags=flags)
    return candidate, candidate != source


def rust_mutator_const_collects(source: str) -> tuple[str, str, bool]:
    rules = [
        (
            r"trace\[POSEIDON_16_COL_FLAG\]\s*=\s*\(0\.\.n_rows\)\.map\(\|_\|\s*F::ONE\)\.collect\(\);",
            "trace[POSEIDON_16_COL_FLAG] = vec![F::ONE; n_rows];",
            "rust_const_vec:flag",
        ),
        (
            r"trace\[POSEIDON_16_COL_RES\]\s*=\s*\(0\.\.n_rows\)\.map\(\|_\|\s*F::from_usize\(POSEIDON_16_NULL_HASH_PTR\)\)\.collect\(\);",
            "trace[POSEIDON_16_COL_RES] = vec![F::from_usize(POSEIDON_16_NULL_HASH_PTR); n_rows];",
            "rust_const_vec:res",
        ),
        (
            r"trace\[POSEIDON_16_COL_A\]\s*=\s*\(0\.\.n_rows\)\.map\(\|_\|\s*F::from_usize\(ZERO_VEC_PTR\)\)\.collect\(\);",
            "trace[POSEIDON_16_COL_A] = vec![F::from_usize(ZERO_VEC_PTR); n_rows];",
            "rust_const_vec:a",
        ),
        (
            r"trace\[POSEIDON_16_COL_B\]\s*=\s*\(0\.\.n_rows\)\.map\(\|_\|\s*F::from_usize\(ZERO_VEC_PTR\)\)\.collect\(\);",
            "trace[POSEIDON_16_COL_B] = vec![F::from_usize(ZERO_VEC_PTR); n_rows];",
            "rust_const_vec:b",
        ),
    ]
    for pattern, repl, name in rules:
        candidate, changed = regex_replace_once(source, pattern, repl, flags=re.MULTILINE)
        if changed:
            return candidate, name, True
    return source, "rust_const_vec:no_change", False


def rust_mutator_hoist_num_cols(source: str) -> tuple[str, str, bool]:
    marker = "let num_cols = num_cols_poseidon_16();"
    if marker in source:
        return source, "rust_hoist_num_cols:already_present", False

    original = "let mut trace = vec![vec![F::ZERO; n_rows]; num_cols_poseidon_16()];"
    if original not in source:
        return source, "rust_hoist_num_cols:pattern_missing", False

    candidate = source.replace(original, f"{marker}\n    let mut trace = vec![vec![F::ZERO; n_rows]; num_cols];", 1)
    head, sep, tail = candidate.partition(marker)
    if not sep:
        return source, "rust_hoist_num_cols:partition_failed", False
    tail = tail.replace("num_cols_poseidon_16()", "num_cols")
    return head + sep + tail, "rust_hoist_num_cols", True


def rust_mutator_cache_trace_refs(source: str) -> tuple[str, str, bool]:
    marker = "let trace_refs = collect_refs(&trace);"
    if marker in source:
        return source, "rust_cache_trace_refs:already_present", False

    old_check = "check_air_validity::<_, EF>(&air, &ExtraDataForBuses::default(), &collect_refs(&trace)).unwrap();"
    old_prove = "let air_claims = prove_air::<EF, _>(&mut prover_state, &air, extra_data, &collect_refs(&trace), None, true);"
    if old_check not in source or old_prove not in source:
        return source, "rust_cache_trace_refs:pattern_missing", False

    candidate = source.replace(
        old_check,
        "let trace_refs = collect_refs(&trace);\n\n    check_air_validity::<_, EF>(&air, &ExtraDataForBuses::default(), &trace_refs).unwrap();",
        1,
    )
    candidate = candidate.replace(
        old_prove,
        "let air_claims = prove_air::<EF, _>(&mut prover_state, &air, extra_data, &trace_refs, None, true);",
        1,
    )
    return candidate, "rust_cache_trace_refs", True


def rust_mutator_hoist_packed_cols(source: str) -> tuple[str, str, bool]:
    marker = "let packed_cols = num_cols << log_n_rows;"
    if marker in source:
        return source, "rust_hoist_packed_cols:already_present", False

    original = "let packed_n_vars = log2_ceil_usize(num_cols << log_n_rows);"
    if original not in source:
        return source, "rust_hoist_packed_cols:pattern_missing", False

    candidate = source.replace(original, f"{marker}\n    {original.replace('num_cols << log_n_rows', 'packed_cols')}", 1)
    head, sep, tail = candidate.partition(marker)
    if not sep:
        return source, "rust_hoist_packed_cols:partition_failed", False
    tail = tail.replace("num_cols << log_n_rows", "packed_cols")
    return head + sep + tail, "rust_hoist_packed_cols", True


def rust_mutator_pack_point_buffer(source: str) -> tuple[str, str, bool]:
    old = "let packed_point = MultilinearPoint([betas.clone(), air_claims.point.0].concat());"
    if old not in source:
        return source, "rust_pack_point_buffer:pattern_missing", False
    replacement = (
        "let mut packed_point_coords = betas.clone();\n"
        "        packed_point_coords.extend_from_slice(&air_claims.point.0);\n"
        "        let packed_point = MultilinearPoint(packed_point_coords);"
    )
    candidate = source.replace(old, replacement, 1)
    return candidate, "rust_pack_point_buffer", True


def rust_mutator_poseidon_table_res_copy(source: str) -> tuple[str, str, bool]:
    old = "let res_a: [F; DIGEST_LEN] = output[..DIGEST_LEN].try_into().unwrap();"
    if old not in source:
        return source, "rust_table_res_copy:pattern_missing", False
    new = (
        "let mut res_a = [F::ZERO; DIGEST_LEN];\n"
        "        res_a.copy_from_slice(&output[..DIGEST_LEN]);"
    )
    return source.replace(old, new, 1), "rust_table_res_copy", True


def rust_mutator_poseidon_table_split_copy(source: str) -> tuple[str, str, bool]:
    old = (
        "let mut input = [F::ZERO; DIGEST_LEN * 2];\n"
        "        input[..DIGEST_LEN].copy_from_slice(&arg0);\n"
        "        input[DIGEST_LEN..].copy_from_slice(&arg1);"
    )
    if old not in source:
        return source, "rust_table_split_copy:pattern_missing", False
    new = (
        "let mut input = [F::ZERO; DIGEST_LEN * 2];\n"
        "        let (left, right) = input.split_at_mut(DIGEST_LEN);\n"
        "        left.copy_from_slice(&arg0);\n"
        "        right.copy_from_slice(&arg1);"
    )
    return source.replace(old, new, 1), "rust_table_split_copy", True


def rust_mutator_poseidon_table_input_index_loop(source: str) -> tuple[str, str, bool]:
    old = (
        "for (i, value) in input.iter().enumerate() {\n"
        "            trace.base[POSEIDON_16_COL_INPUT_START + i].push(*value);\n"
        "        }"
    )
    if old not in source:
        return source, "rust_table_input_index_loop:pattern_missing", False
    new = (
        "for i in 0..WIDTH {\n"
        "            trace.base[POSEIDON_16_COL_INPUT_START + i].push(input[i]);\n"
        "        }"
    )
    return source.replace(old, new, 1), "rust_table_input_index_loop", True


def rust_mutator_poseidon_table_trace_base_alias(source: str) -> tuple[str, str, bool]:
    old_enumerate = (
        "trace.base[POSEIDON_16_COL_FLAG].push(F::ONE);\n"
        "        trace.base[POSEIDON_16_COL_A].push(arg_a);\n"
        "        trace.base[POSEIDON_16_COL_B].push(arg_b);\n"
        "        trace.base[POSEIDON_16_COL_RES].push(index_res_a);\n"
        "        for (i, value) in input.iter().enumerate() {\n"
        "            trace.base[POSEIDON_16_COL_INPUT_START + i].push(*value);\n"
        "        }"
    )
    old_indexed = (
        "trace.base[POSEIDON_16_COL_FLAG].push(F::ONE);\n"
        "        trace.base[POSEIDON_16_COL_A].push(arg_a);\n"
        "        trace.base[POSEIDON_16_COL_B].push(arg_b);\n"
        "        trace.base[POSEIDON_16_COL_RES].push(index_res_a);\n"
        "        for i in 0..WIDTH {\n"
        "            trace.base[POSEIDON_16_COL_INPUT_START + i].push(input[i]);\n"
        "        }"
    )
    replacement = (
        "let base = &mut trace.base;\n"
        "        base[POSEIDON_16_COL_FLAG].push(F::ONE);\n"
        "        base[POSEIDON_16_COL_A].push(arg_a);\n"
        "        base[POSEIDON_16_COL_B].push(arg_b);\n"
        "        base[POSEIDON_16_COL_RES].push(index_res_a);\n"
        "        for i in 0..WIDTH {\n"
        "            base[POSEIDON_16_COL_INPUT_START + i].push(input[i]);\n"
        "        }"
    )
    if old_enumerate in source:
        return source.replace(old_enumerate, replacement, 1), "rust_table_trace_base_alias", True
    if old_indexed in source:
        return source.replace(old_indexed, replacement, 1), "rust_table_trace_base_alias", True
    return source, "rust_table_trace_base_alias:pattern_missing", False


def rust_mutator_neon_internal_for_loop(source: str) -> tuple[str, str, bool]:
    marker = "self.packed_internal_constants.iter().for_each(|&rc| {"
    idx = source.find(marker)
    if idx == -1:
        return source, "rust_neon_internal_for_loop:pattern_missing", False

    tail = source[idx:].replace(marker, "for &rc in &self.packed_internal_constants {", 1)
    close = tail.find("\n            });")
    if close == -1:
        return source, "rust_neon_internal_for_loop:closing_missing", False
    tail = tail[:close] + "\n            }" + tail[close + len("\n            });") :]
    return source[:idx] + tail, "rust_neon_internal_for_loop", True


def rust_mutator_neon_internal_for_loop_w24(source: str) -> tuple[str, str, bool]:
    marker = "self.packed_internal_constants.iter().for_each(|&rc| {"
    first = source.find(marker)
    if first == -1:
        return source, "rust_neon_internal_for_loop_w24:pattern_missing", False
    second = source.find(marker, first + 1)
    if second == -1:
        return source, "rust_neon_internal_for_loop_w24:second_missing", False

    candidate = source[:second] + "for &rc in &self.packed_internal_constants {" + source[second + len(marker) :]

    close_marker = "\n            });"
    close_idx = candidate.find(close_marker, second)
    if close_idx == -1:
        return source, "rust_neon_internal_for_loop_w24:closing_missing", False
    candidate = candidate[:close_idx] + "\n            }" + candidate[close_idx + len(close_marker) :]
    return candidate, "rust_neon_internal_for_loop_w24", True


def rust_mutator_neon_internal_for_loop_both(source: str) -> tuple[str, str, bool]:
    marker = "self.packed_internal_constants.iter().for_each(|&rc| {"
    if source.count(marker) < 2:
        return source, "rust_neon_internal_for_loop_both:pattern_missing", False

    candidate = source.replace(marker, "for &rc in &self.packed_internal_constants {")
    close_marker = "\n            });"
    if candidate.count(close_marker) < 2:
        return source, "rust_neon_internal_for_loop_both:closing_missing", False
    candidate = candidate.replace(close_marker, "\n            }", 2)
    return candidate, "rust_neon_internal_for_loop_both", True


def rust_mutator_neon_add_sum_loops(source: str) -> tuple[str, str, bool]:
    rules = [
        (
            "input.as_mut()[..5]\n            .iter_mut()\n            .for_each(|x| *x = uint32x4_mod_add(sum, *x, PMP::PACKED_P));",
            "for x in input.as_mut()[..5].iter_mut() {\n            *x = uint32x4_mod_add(sum, *x, PMP::PACKED_P);\n        }",
            "rust_neon_add_sum_pos_head",
        ),
        (
            "input.as_mut()[5..8]\n            .iter_mut()\n            .for_each(|x| *x = uint32x4_mod_sub(sum, *x, PMP::PACKED_P));",
            "for x in input.as_mut()[5..8].iter_mut() {\n            *x = uint32x4_mod_sub(sum, *x, PMP::PACKED_P);\n        }",
            "rust_neon_add_sum_neg_head",
        ),
        (
            "input.as_mut()[8..(8 + Self::NUM_POS)]\n            .iter_mut()\n            .for_each(|x| *x = uint32x4_mod_add(sum, *x, PMP::PACKED_P));",
            "for x in input.as_mut()[8..(8 + Self::NUM_POS)].iter_mut() {\n            *x = uint32x4_mod_add(sum, *x, PMP::PACKED_P);\n        }",
            "rust_neon_add_sum_pos_tail",
        ),
        (
            "input.as_mut()[8 + Self::NUM_POS..]\n            .iter_mut()\n            .for_each(|x| *x = uint32x4_mod_sub(sum, *x, PMP::PACKED_P));",
            "for x in input.as_mut()[8 + Self::NUM_POS..].iter_mut() {\n            *x = uint32x4_mod_sub(sum, *x, PMP::PACKED_P);\n        }",
            "rust_neon_add_sum_neg_tail",
        ),
    ]
    for old, new, name in rules:
        if old in source:
            return source.replace(old, new, 1), name, True
    return source, "rust_neon_add_sum:pattern_missing", False


def rust_mutator_neon_add_sum_as_mut_hoist(source: str) -> tuple[str, str, bool]:
    old = (
        "    unsafe fn add_sum(input: &mut Self::ArrayLike, sum: uint32x4_t) {\n"
        "        // For the first 5 elements (s_1 to s_5), the diagonal coefficients are positive, so we add the sum.\n"
        "        input.as_mut()[..5]\n"
    )
    if old not in source:
        return source, "rust_neon_add_sum_as_mut_hoist:pattern_missing", False

    candidate = source.replace(
        old,
        "    unsafe fn add_sum(input: &mut Self::ArrayLike, sum: uint32x4_t) {\n"
        "        let input = input.as_mut();\n"
        "        // For the first 5 elements (s_1 to s_5), the diagonal coefficients are positive, so we add the sum.\n"
        "        input[..5]\n",
        1,
    )
    candidate = candidate.replace("input.as_mut()[5..8]", "input[5..8]", 1)
    candidate = candidate.replace("input.as_mut()[8..(8 + Self::NUM_POS)]", "input[8..(8 + Self::NUM_POS)]", 1)
    candidate = candidate.replace("input.as_mut()[8 + Self::NUM_POS..]", "input[8 + Self::NUM_POS..]", 1)
    return candidate, "rust_neon_add_sum_as_mut_hoist", True


def rust_mutator_neon_sum_vec_hoist(source: str) -> tuple[str, str, bool]:
    old = (
        "                ILP::add_sum(\n"
        "                    &mut internal_state.s_hi,\n"
        "                    transmute::<PackedMontyField31Neon<FP>, uint32x4_t>(sum),\n"
        "                );"
    )
    if old not in source:
        return source, "rust_neon_sum_vec_hoist:pattern_missing", False
    new = (
        "                let sum_vec = transmute::<PackedMontyField31Neon<FP>, uint32x4_t>(sum);\n"
        "                ILP::add_sum(&mut internal_state.s_hi, sum_vec);"
    )
    return source.replace(old, new, 1), "rust_neon_sum_vec_hoist", True


def rust_mutator_neon_sum_vec_hoist_w24(source: str) -> tuple[str, str, bool]:
    old = (
        "                ILP::add_sum(\n"
        "                    &mut internal_state.s_hi,\n"
        "                    transmute::<PackedMontyField31Neon<FP>, uint32x4_t>(sum),\n"
        "                );"
    )
    first = source.find(old)
    if first == -1:
        return source, "rust_neon_sum_vec_hoist_w24:pattern_missing", False
    second = source.find(old, first + 1)
    if second == -1:
        return source, "rust_neon_sum_vec_hoist_w24:second_missing", False
    replacement = (
        "                let sum_vec = transmute::<PackedMontyField31Neon<FP>, uint32x4_t>(sum);\n"
        "                ILP::add_sum(&mut internal_state.s_hi, sum_vec);"
    )
    candidate = source[:second] + replacement + source[second + len(old) :]
    return candidate, "rust_neon_sum_vec_hoist_w24", True


def rust_mutator_neon_sum_vec_hoist_both(source: str) -> tuple[str, str, bool]:
    old = (
        "                ILP::add_sum(\n"
        "                    &mut internal_state.s_hi,\n"
        "                    transmute::<PackedMontyField31Neon<FP>, uint32x4_t>(sum),\n"
        "                );"
    )
    if source.count(old) < 2:
        return source, "rust_neon_sum_vec_hoist_both:pattern_missing", False
    replacement = (
        "                let sum_vec = transmute::<PackedMontyField31Neon<FP>, uint32x4_t>(sum);\n"
        "                ILP::add_sum(&mut internal_state.s_hi, sum_vec);"
    )
    candidate = source.replace(old, replacement, 2)
    return candidate, "rust_neon_sum_vec_hoist_both", True


def rust_mutator_neon_sum_array_inline_w16(source: str) -> tuple[str, str, bool]:
    old = (
        "                let s_hi_transmuted: &[PackedMontyField31Neon<FP>; 15] = transmute(&internal_state.s_hi);\n"
        "                let sum_tail = PackedMontyField31Neon::<FP>::sum_array::<15>(s_hi_transmuted);"
    )
    if old not in source:
        return source, "rust_neon_sum_array_inline_w16:pattern_missing", False
    new = (
        "                let sum_tail = PackedMontyField31Neon::<FP>::sum_array::<15>(\n"
        "                    transmute::<&[uint32x4_t; 15], &[PackedMontyField31Neon<FP>; 15]>(&internal_state.s_hi),\n"
        "                );"
    )
    return source.replace(old, new, 1), "rust_neon_sum_array_inline_w16", True


def rust_mutator_neon_sum_array_inline_w24(source: str) -> tuple[str, str, bool]:
    old = (
        "                let s_hi_transmuted: &[PackedMontyField31Neon<FP>; 23] = transmute(&internal_state.s_hi);\n"
        "                let sum_tail = PackedMontyField31Neon::<FP>::sum_array::<23>(s_hi_transmuted);"
    )
    if old not in source:
        return source, "rust_neon_sum_array_inline_w24:pattern_missing", False
    new = (
        "                let sum_tail = PackedMontyField31Neon::<FP>::sum_array::<23>(\n"
        "                    transmute::<&[uint32x4_t; 23], &[PackedMontyField31Neon<FP>; 23]>(&internal_state.s_hi),\n"
        "                );"
    )
    return source.replace(old, new, 1), "rust_neon_sum_array_inline_w24", True


def rust_mutator_neon_sum_array_inline_both(source: str) -> tuple[str, str, bool]:
    old_w16 = (
        "                let s_hi_transmuted: &[PackedMontyField31Neon<FP>; 15] = transmute(&internal_state.s_hi);\n"
        "                let sum_tail = PackedMontyField31Neon::<FP>::sum_array::<15>(s_hi_transmuted);"
    )
    old_w24 = (
        "                let s_hi_transmuted: &[PackedMontyField31Neon<FP>; 23] = transmute(&internal_state.s_hi);\n"
        "                let sum_tail = PackedMontyField31Neon::<FP>::sum_array::<23>(s_hi_transmuted);"
    )
    if old_w16 not in source or old_w24 not in source:
        return source, "rust_neon_sum_array_inline_both:pattern_missing", False

    new_w16 = (
        "                let sum_tail = PackedMontyField31Neon::<FP>::sum_array::<15>(\n"
        "                    transmute::<&[uint32x4_t; 15], &[PackedMontyField31Neon<FP>; 15]>(&internal_state.s_hi),\n"
        "                );"
    )
    new_w24 = (
        "                let sum_tail = PackedMontyField31Neon::<FP>::sum_array::<23>(\n"
        "                    transmute::<&[uint32x4_t; 23], &[PackedMontyField31Neon<FP>; 23]>(&internal_state.s_hi),\n"
        "                );"
    )
    candidate = source.replace(old_w16, new_w16, 1)
    candidate = candidate.replace(old_w24, new_w24, 1)
    return candidate, "rust_neon_sum_array_inline_both", True


def rust_mutator_hoist_log_num_cols(source: str) -> tuple[str, str, bool]:
    marker = "let log_num_cols = log2_ceil_usize(num_cols);"
    if marker in source:
        return source, "rust_hoist_log_num_cols:already_present", False

    anchor = "let packed_n_vars = log2_ceil_usize(num_cols << log_n_rows);"
    needle = "sample_vec(log2_ceil_usize(num_cols))"
    if anchor not in source or source.count(needle) < 2:
        return source, "rust_hoist_log_num_cols:pattern_missing", False

    candidate = source.replace(anchor, f"{anchor}\n    {marker}", 1)
    candidate = candidate.replace(needle, "sample_vec(log_num_cols)")
    return candidate, "rust_hoist_log_num_cols", True


def generic_heuristic_candidate(source: str, iteration: int) -> tuple[str, str, bool]:
    markers = [
        "AUTOCIRCUIT_NOP_ASSERT",
        "AUTOCIRCUIT_NOP_FINAL",
        "AUTOCIRCUIT_WASTE_DIV",
        "AUTOCIRCUIT_NOP_ADD",
        "AUTOCIRCUIT_WASTE_BALANCE",
        "AUTOCIRCUIT_WASTE_HASH",
    ]
    shift = (iteration - 1) % len(markers)
    ordered = markers[shift:] + markers[:shift]
    for marker in ordered:
        candidate, changed = remove_first_marked_line(source, marker)
        if changed:
            return candidate, f"heuristic_remove:{marker}", True

    patterns = [
        (r"\+ 0", ""),
        (r"\* 1", ""),
    ]
    for pattern, repl in patterns:
        candidate2 = re.sub(pattern, repl, source, count=1)
        if candidate2 != source:
            return candidate2, f"heuristic_regex:{pattern}", True

    return source, "heuristic_no_change", False


def rust_heuristic_candidate(
    source: str,
    iteration: int,
    source_path: Path,
    blocked_mutations: set[str] | None = None,
    mutation_attempts: dict[str, int] | None = None,
    target_config: dict[str, Any] | None = None,
) -> tuple[str, str, bool]:
    path = str(source_path).replace("\\", "/").lower()

    operators: list[Any] = []
    if path.endswith("crates/lean_vm/src/tables/poseidon_16/mod.rs"):
        operators.extend(
            [
                rust_mutator_poseidon_table_res_copy,
                rust_mutator_poseidon_table_split_copy,
                rust_mutator_poseidon_table_input_index_loop,
                rust_mutator_poseidon_table_trace_base_alias,
            ]
        )
    if path.endswith("crates/backend/koala-bear/src/monty_31/aarch64_neon/poseidon2.rs"):
        operators.extend(
            [
                rust_mutator_neon_internal_for_loop,
                rust_mutator_neon_internal_for_loop_w24,
                rust_mutator_neon_internal_for_loop_both,
                rust_mutator_neon_add_sum_loops,
                rust_mutator_neon_add_sum_as_mut_hoist,
                rust_mutator_neon_sum_vec_hoist,
                rust_mutator_neon_sum_vec_hoist_w24,
                rust_mutator_neon_sum_vec_hoist_both,
                rust_mutator_neon_sum_array_inline_w16,
                rust_mutator_neon_sum_array_inline_w24,
                rust_mutator_neon_sum_array_inline_both,
            ]
        )

    operators.extend(
        [
            rust_mutator_const_collects,
            rust_mutator_hoist_num_cols,
            rust_mutator_cache_trace_refs,
            rust_mutator_hoist_packed_cols,
            rust_mutator_pack_point_buffer,
            rust_mutator_hoist_log_num_cols,
        ]
    )
    if not operators:
        return source, "rust_no_change", False

    shift = (iteration - 1) % len(operators)
    ordered = operators[shift:] + operators[:shift]
    candidates: list[tuple[int, int, int, str, str]] = []
    single_candidates: list[tuple[int, str, str]] = []
    attempts = mutation_attempts or {}

    compound_every = int((target_config or {}).get("compound_every", 0))
    compound_limit = max(0, int((target_config or {}).get("compound_limit", 0)))
    compound_second_window = max(1, int((target_config or {}).get("compound_second_window", len(ordered))))
    prefer_compound = compound_every > 0 and (iteration % compound_every == 0)
    single_tier = 1 if prefer_compound else 0
    compound_tier = 0 if prefer_compound else 1

    for idx, operator in enumerate(ordered):
        candidate, mutation, changed = operator(source)
        if blocked_mutations and mutation in blocked_mutations:
            continue
        if not changed:
            continue
        single_candidates.append((idx, candidate, mutation))
        candidates.append((single_tier, attempts.get(mutation, 0), idx, candidate, mutation))

    if compound_every > 0 and compound_limit > 0 and single_candidates:
        second_ops = ordered[: min(len(ordered), compound_second_window)]
        seen_labels: set[str] = set()
        added = 0
        for idx_a, candidate_a, mutation_a in single_candidates:
            if blocked_mutations and mutation_a in blocked_mutations:
                continue
            for idx_b, operator_b in enumerate(second_ops):
                candidate_b, mutation_b, changed_b = operator_b(candidate_a)
                if not changed_b:
                    continue
                if blocked_mutations and mutation_b in blocked_mutations:
                    continue
                if mutation_b == mutation_a:
                    continue
                combo_mutation = f"{mutation_a}+{mutation_b}"
                if combo_mutation in seen_labels:
                    continue
                seen_labels.add(combo_mutation)
                combo_attempts = attempts.get(combo_mutation, attempts.get(mutation_a, 0) + attempts.get(mutation_b, 0))
                combo_idx = idx_a * 1000 + idx_b
                candidates.append((compound_tier, combo_attempts, combo_idx, candidate_b, combo_mutation))
                added += 1
                if added >= compound_limit:
                    break
            if added >= compound_limit:
                break

    if candidates:
        candidates.sort(key=lambda x: (x[0], x[1], x[2]))
        _, _, _, chosen_candidate, chosen_mutation = candidates[0]
        return chosen_candidate, chosen_mutation, True
    return source, "rust_no_change", False


def heuristic_candidate(
    source: str,
    iteration: int,
    language: str,
    source_path: Path,
    blocked_mutations: set[str] | None = None,
    mutation_attempts: dict[str, int] | None = None,
    target_config: dict[str, Any] | None = None,
) -> tuple[str, str, bool]:
    if language.lower() == "rust":
        rust_candidate, mutation, changed = rust_heuristic_candidate(
            source,
            iteration,
            source_path,
            blocked_mutations=blocked_mutations,
            mutation_attempts=mutation_attempts,
            target_config=target_config,
        )
        if changed:
            return rust_candidate, mutation, True
    return generic_heuristic_candidate(source, iteration)


def source_extension(language: str, source_path: Path) -> str:
    explicit = source_path.suffix.lstrip(".")
    if explicit:
        return explicit
    mapping = {"rust": "rs", "cairo": "cairo", "noir": "nr"}
    return mapping.get(language.lower(), "txt")


def run_capture(argv: list[str], *, cwd: Path | None = None) -> str | None:
    try:
        proc = subprocess.run(
            argv,
            cwd=str(cwd) if cwd is not None else None,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    text = (proc.stdout or proc.stderr).strip()
    return text or None


def git_fingerprint(path: Path) -> dict[str, Any] | None:
    top = run_capture(["git", "-C", str(path), "rev-parse", "--show-toplevel"])
    if not top:
        return None
    root = Path(top.strip())
    commit = run_capture(["git", "-C", str(root), "rev-parse", "HEAD"])
    branch = run_capture(["git", "-C", str(root), "rev-parse", "--abbrev-ref", "HEAD"])
    remote = run_capture(["git", "-C", str(root), "config", "--get", "remote.origin.url"])
    dirty_raw = run_capture(["git", "-C", str(root), "status", "--porcelain"])
    return {
        "root": str(root),
        "commit": commit.strip() if commit else None,
        "branch": branch.strip() if branch else None,
        "remote": remote.strip() if remote else None,
        "dirty": bool(dirty_raw),
    }


def runtime_fingerprint(*, target_name: str, target: dict[str, Any], source_path: Path) -> dict[str, Any]:
    project_rel = target.get("project_dir")
    project_dir = ROOT / project_rel if isinstance(project_rel, str) else source_path.parent

    tools: dict[str, str] = {}
    tool_checks = {
        "python": [sys.executable, "--version"],
        "cargo": ["cargo", "--version"],
        "rustc": ["rustc", "--version"],
        "scarb": ["scarb", "--version"],
        "nargo": ["nargo", "--version"],
    }
    for name, argv in tool_checks.items():
        if name != "python" and shutil.which(argv[0]) is None:
            continue
        out = run_capture(argv)
        if out:
            tools[name] = out.splitlines()[0]

    git_snapshots: list[dict[str, Any]] = []
    seen_roots: set[str] = set()
    for label, path in (
        ("autoposeidon_root", ROOT),
        ("target_project_dir", project_dir),
        ("target_source_parent", source_path.parent),
    ):
        snap = git_fingerprint(path)
        if not snap:
            continue
        root = str(snap.get("root", ""))
        if root in seen_roots:
            continue
        seen_roots.add(root)
        snap["label"] = label
        git_snapshots.append(snap)

    return {
        "target": target_name,
        "timestamp": prepare.now_iso(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "node": platform.node(),
        "project_dir": str(project_dir),
        "source_file": str(source_path),
        "runtime_controls": {
            "AUTORESEARCH_NICE": os.getenv("AUTORESEARCH_NICE"),
            "AUTORESEARCH_CPU_AFFINITY": os.getenv("AUTORESEARCH_CPU_AFFINITY"),
        },
        "tools": tools,
        "git": git_snapshots,
    }


def load_target_overrides(path_value: str | None, *, target_name: str) -> dict[str, Any]:
    if not path_value:
        return {}

    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"target overrides file not found: {path}")

    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise TypeError("target overrides payload must be a JSON object")

    selected: Any
    if any(key in ALLOWED_TARGET_OVERRIDE_KEYS for key in payload):
        selected = payload
    else:
        selected = payload.get(target_name, {})
    if not isinstance(selected, dict):
        raise TypeError(f"target overrides entry for '{target_name}' must be a JSON object")

    return {key: value for key, value in selected.items() if key in ALLOWED_TARGET_OVERRIDE_KEYS}


def extract_mutation_label_from_notes(notes: str) -> str | None:
    if not notes:
        return None
    token = notes.split(";", 1)[0].strip()
    if not token:
        return None

    # Strip wrapper prefixes while preserving mutation sublabels like `rust_const_vec:a`.
    while True:
        changed = False
        if token.startswith("accepted:"):
            token = token[len("accepted:") :].strip()
            changed = True
        if token.startswith("fallback_"):
            token = token[len("fallback_") :].strip()
            changed = True
        if token.startswith("rejected_") and ":" in token:
            token = token.split(":", 1)[1].strip()
            changed = True
        if not changed:
            break

    if token.endswith(":no_change"):
        token = token[: -len(":no_change")].strip()

    if not token:
        return None
    if token in {"no_change", "n/a", "rust_no_change", "heuristic_no_change"}:
        return None
    return token


def load_mutation_attempt_counts(target_name: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    results_file = ROOT / "results.tsv"
    if not results_file.exists():
        return counts
    rows = results_file.read_text().splitlines()
    if len(rows) <= 1:
        return counts
    for row in rows[1:]:
        cols = row.split("\t")
        if len(cols) < 12:
            continue
        if cols[1] != target_name:
            continue
        try:
            iteration = int(cols[2])
        except Exception:  # noqa: BLE001
            iteration = 0
        if iteration <= 0:
            continue
        mutation = extract_mutation_label_from_notes(cols[11])
        if mutation is None:
            continue
        if "baseline" in mutation:
            continue
        counts[mutation] = counts.get(mutation, 0) + 1
    return counts


def make_run_label() -> str:
    stamp = re.sub(r"[^0-9A-Za-z]+", "", prepare.now_iso())
    return f"run_{stamp}_{os.getpid()}"


def confirm_improvement(
    *,
    target_name: str,
    best_metric: float,
    higher_is_better: bool,
    min_improvement_abs: float,
    min_improvement_rel: float,
    repeats: int,
) -> tuple[bool, float | None, list[float], str]:
    if repeats < 1:
        return True, None, [], "confirm_skipped"

    values: list[float] = []
    for _ in range(repeats):
        confirm_result = prepare.evaluate_target(target_name)
        metric_value = confirm_result.get("metric_value")
        if confirm_result.get("status") != "success" or metric_value is None:
            return False, None, values, "confirm_eval_failed"
        values.append(float(metric_value))

    confirmed = float(statistics.median(values))
    ok = is_better(
        confirmed,
        best_metric,
        higher_is_better=higher_is_better,
        min_improvement_abs=min_improvement_abs,
        min_improvement_rel=min_improvement_rel,
    )
    return ok, confirmed, values, "confirm_ok" if ok else "confirm_not_better"


def ab_validate_candidate(
    *,
    source_path: Path,
    target_name: str,
    candidate_source: str,
    original_source: str,
    higher_is_better: bool,
    min_improvement_abs: float,
    min_improvement_rel: float,
    repeats: int,
) -> tuple[bool, dict[str, Any], str]:
    if repeats < 1:
        return True, {}, "ab_skipped"

    candidate_values: list[float] = []
    original_values: list[float] = []
    restore_source = original_source

    try:
        source_path.write_text(candidate_source)
        for _ in range(repeats):
            candidate_result = prepare.evaluate_target(target_name)
            metric_value = candidate_result.get("metric_value")
            if candidate_result.get("status") != "success" or metric_value is None:
                return False, {"candidate_values": candidate_values}, "ab_candidate_eval_failed"
            candidate_values.append(float(metric_value))

        source_path.write_text(original_source)
        for _ in range(repeats):
            original_result = prepare.evaluate_target(target_name)
            metric_value = original_result.get("metric_value")
            if original_result.get("status") != "success" or metric_value is None:
                return False, {"candidate_values": candidate_values, "original_values": original_values}, "ab_original_eval_failed"
            original_values.append(float(metric_value))

        candidate_median = float(statistics.median(candidate_values))
        original_median = float(statistics.median(original_values))
        ok = is_better(
            candidate_median,
            original_median,
            higher_is_better=higher_is_better,
            min_improvement_abs=min_improvement_abs,
            min_improvement_rel=min_improvement_rel,
        )
        if ok:
            restore_source = candidate_source

        details = {
            "repeats": repeats,
            "candidate_values": candidate_values,
            "original_values": original_values,
            "candidate_median": candidate_median,
            "original_median": original_median,
        }
        return ok, details, "ab_ok" if ok else "ab_not_better"
    finally:
        source_path.write_text(restore_source)


def write_iteration_artifact(
    *,
    target_name: str,
    run_label: str,
    iteration: int,
    source_path: Path,
    language: str,
    previous_source: str,
    candidate_source: str,
    mutation: str,
    accepted: bool,
    best_before: float,
    best_after: float,
    diagnostics: dict[str, Any],
    result: dict[str, Any],
    runtime_env: dict[str, Any],
) -> None:
    run_dir = ARTIFACTS_DIR / target_name / run_label / f"iter_{iteration:05d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    ext = source_extension(language, source_path)
    before_file = run_dir / f"before.{ext}"
    after_file = run_dir / f"after.{ext}"
    diff_file = run_dir / "patch.diff"
    meta_file = run_dir / "metadata.json"
    debug_file = run_dir / "evaluation_debug.json"
    env_file = run_dir / "environment.json"

    before_file.write_text(previous_source)
    after_file.write_text(candidate_source)

    diff_text = "".join(
        difflib.unified_diff(
            previous_source.splitlines(keepends=True),
            candidate_source.splitlines(keepends=True),
            fromfile=before_file.name,
            tofile=after_file.name,
        )
    )
    diff_file.write_text(diff_text)

    metadata = {
        "timestamp": prepare.now_iso(),
        "target": target_name,
        "run_label": run_label,
        "iteration": iteration,
        "mutation": mutation,
        "accepted": accepted,
        "best_before": best_before,
        "best_after": best_after,
        "metric_name": result.get("metric_name"),
        "metric_value": result.get("metric_value"),
        "result_status": result.get("status"),
        "result_notes": result.get("notes"),
        "diagnostics": diagnostics,
    }
    meta_file.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    debug_file.write_text(json.dumps(result.get("debug", {}), indent=2, sort_keys=True))
    env_file.write_text(json.dumps(runtime_env, indent=2, sort_keys=True))


def request_openai_candidate(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
) -> tuple[str | None, dict[str, Any]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None, {"reason": "OPENAI_API_KEY not set"}

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    parsed_base_url = urllib.parse.urlparse(base_url)
    if parsed_base_url.scheme not in {"https", "http"}:
        return None, {"reason": "invalid_base_url_scheme", "url": base_url[:200]}
    if parsed_base_url.scheme == "http" and parsed_base_url.hostname not in {"127.0.0.1", "localhost"}:
        return None, {"reason": "invalid_base_url_host", "url": base_url[:200]}
    url = f"{base_url}/chat/completions"
    parsed_url = urllib.parse.urlparse(url)
    if parsed_url.scheme not in {"https", "http"} or not parsed_url.netloc:
        return None, {"reason": "invalid_request_url", "url": url[:200]}

    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    req = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as err:
        body = err.read().decode("utf-8", errors="replace")
        return None, {"reason": f"http_error:{err.code}", "body": body[:2000]}
    except Exception as exc:  # noqa: BLE001
        return None, {"reason": f"request_error:{type(exc).__name__}", "error": str(exc)}

    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return None, {"reason": "invalid_json", "body": body[:2000]}

    try:
        content = data["choices"][0]["message"]["content"]
    except Exception as exc:  # noqa: BLE001
        return None, {"reason": f"missing_content:{type(exc).__name__}", "response": data}

    usage = data.get("usage", {})
    return content, {"reason": "ok", "usage": usage}


def is_better(
    value: float,
    best: float,
    *,
    higher_is_better: bool,
    min_improvement_abs: float = 0.0,
    min_improvement_rel: float = 0.0,
) -> bool:
    min_improvement_abs = max(0.0, min_improvement_abs)
    min_improvement_rel = max(0.0, min_improvement_rel)

    if higher_is_better:
        abs_ok = value > (best + min_improvement_abs)
        rel_ok = value > (best * (1.0 + min_improvement_rel))
        return abs_ok and rel_ok

    abs_ok = value < (best - min_improvement_abs)
    rel_ok = value < (best * (1.0 - min_improvement_rel))
    return abs_ok and rel_ok


def finite_or_zero(value: float) -> float:
    if value != value:  # NaN
        return 0.0
    if value == float("inf") or value == float("-inf"):
        return 0.0
    return value


def relative_stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = statistics.fmean(values)
    if mean == 0.0:
        return float("inf")
    return statistics.pstdev(values) / abs(mean)


def parse_metric_series(raw: Any) -> list[float] | None:
    if not isinstance(raw, list):
        return None
    parsed: list[float] = []
    for value in raw:
        try:
            parsed.append(float(value))
        except Exception:  # noqa: BLE001
            return None
    return parsed


def median_abs_deviation(values: list[float]) -> float:
    if not values:
        return 0.0
    center = statistics.median(values)
    deviations = [abs(v - center) for v in values]
    return float(statistics.median(deviations))


def robust_sigma(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    # Normal-consistent MAD scale factor.
    sigma = 1.4826 * median_abs_deviation(values)
    if sigma == 0.0:
        sigma = statistics.pstdev(values)
    return float(sigma)


def mean_confidence_interval(values: list[float], z: float) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = float(statistics.fmean(values))
    if len(values) < 2:
        return mean, mean
    stdev = float(statistics.pstdev(values))
    se = stdev / (len(values) ** 0.5)
    margin = max(0.0, z) * se
    return mean - margin, mean + margin


def distribution_guard(
    *,
    candidate_values: list[float],
    baseline_values: list[float],
    higher_is_better: bool,
    min_effect_sigma: float,
    ci_z: float,
    require_ci_separation: bool,
) -> tuple[bool, str, dict[str, Any]]:
    cand_median = float(statistics.median(candidate_values))
    base_median = float(statistics.median(baseline_values))
    delta = cand_median - base_median
    direction_ok = delta > 0.0 if higher_is_better else delta < 0.0

    cand_sigma = robust_sigma(candidate_values)
    base_sigma = robust_sigma(baseline_values)
    pooled_sigma = ((cand_sigma**2 + base_sigma**2) / 2.0) ** 0.5
    if pooled_sigma == 0.0:
        effect_sigma = float("inf") if delta != 0.0 else 0.0
    else:
        effect_sigma = abs(delta) / pooled_sigma

    effect_ok = effect_sigma >= max(0.0, min_effect_sigma)
    cand_ci = mean_confidence_interval(candidate_values, ci_z)
    base_ci = mean_confidence_interval(baseline_values, ci_z)

    if require_ci_separation:
        ci_ok = cand_ci[0] > base_ci[1] if higher_is_better else cand_ci[1] < base_ci[0]
    else:
        ci_ok = True

    stats = {
        "candidate_n": len(candidate_values),
        "baseline_n": len(baseline_values),
        "candidate_median": cand_median,
        "baseline_median": base_median,
        "delta": delta,
        "candidate_sigma": cand_sigma,
        "baseline_sigma": base_sigma,
        "pooled_sigma": pooled_sigma,
        "effect_sigma": effect_sigma,
        "min_effect_sigma": min_effect_sigma,
        "ci_z": ci_z,
        "candidate_ci": [cand_ci[0], cand_ci[1]],
        "baseline_ci": [base_ci[0], base_ci[1]],
        "require_ci_separation": require_ci_separation,
    }

    if not direction_ok:
        return False, "direction", stats
    if not effect_ok:
        return False, "effect", stats
    if not ci_ok:
        return False, "ci_overlap", stats
    return True, "ok", stats


def resolve_min_improvement_rel(
    *,
    target: dict[str, Any],
    configured_rel: float,
    baseline_series: list[float] | None,
) -> tuple[float, dict[str, Any]]:
    mode = str(target.get("min_improvement_rel_mode", "fixed")).strip().lower()
    sigma = float(target.get("min_improvement_rel_sigma", 2.0))
    floor_rel = float(target.get("min_improvement_rel_min", 0.0))
    cap_raw = target.get("min_improvement_rel_max")
    cap_rel = float(cap_raw) if cap_raw is not None else None

    baseline_rel_stdev = 0.0
    if baseline_series is not None and len(baseline_series) >= 2:
        baseline_rel_stdev = finite_or_zero(relative_stdev(baseline_series))

    adaptive_rel = max(0.0, baseline_rel_stdev * max(0.0, sigma))
    configured_rel = max(0.0, configured_rel)
    effective_rel = configured_rel

    if mode == "floor":
        effective_rel = max(configured_rel, adaptive_rel)
    elif mode == "adaptive":
        effective_rel = adaptive_rel if adaptive_rel > 0.0 else configured_rel
    elif mode != "fixed":
        mode = "fixed"

    effective_rel = max(effective_rel, floor_rel)
    if cap_rel is not None:
        effective_rel = min(effective_rel, max(0.0, cap_rel))

    return effective_rel, {
        "mode": mode,
        "configured_rel": configured_rel,
        "baseline_rel_stdev": baseline_rel_stdev,
        "sigma": sigma,
        "adaptive_rel": adaptive_rel,
        "floor_rel": floor_rel,
        "cap_rel": cap_rel,
        "effective_rel": effective_rel,
    }


def run_loop(args: argparse.Namespace) -> int:
    targets = prepare.load_targets()
    if args.target not in targets:
        print(f"Unknown target: {args.target}", file=sys.stderr)
        return 2

    target = dict(targets[args.target])
    try:
        target_overrides = load_target_overrides(args.target_overrides_json, target_name=args.target)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to load target overrides: {exc}", file=sys.stderr)
        return 2
    if target_overrides:
        target.update(target_overrides)

    if "source_file" not in target:
        print("target missing source_file (required for optimization loop)", file=sys.stderr)
        return 2

    source_path = ROOT / target["source_file"]
    if not source_path.exists():
        print(f"Source file not found: {source_path}", file=sys.stderr)
        return 2

    template = load_prompt_template()
    if "language" in target:
        language = str(target["language"])
    elif target["type"] == "noir":
        language = "Noir"
    elif target["type"] == "cairo":
        language = "Cairo"
    else:
        language = "Source"

    baseline = prepare.evaluate_target(args.target)
    if baseline["status"] != "success" or baseline["metric_value"] is None:
        print("Baseline evaluation failed; aborting optimization loop", file=sys.stderr)
        print(json.dumps(baseline, indent=2, sort_keys=True), file=sys.stderr)
        return 1

    run_label = make_run_label()
    run_env = runtime_fingerprint(target_name=args.target, target=target, source_path=source_path)

    prepare.append_result_row(
        target=args.target,
        iteration=0,
        status="success",
        metric_name=baseline["metric_name"],
        metric_value=baseline["metric_value"],
        higher_is_better=bool(target["higher_is_better"]),
        check_s=baseline["check_s"],
        info_or_bench_s=baseline["info_or_bench_s"],
        execute_s=baseline["execute_s"],
        notes=f"loop_baseline;run={run_label}",
    )

    best_metric = float(baseline["metric_value"])
    baseline_metric_series = parse_metric_series(baseline.get("debug", {}).get("metric_values"))
    best_metric_series = baseline_metric_series if baseline_metric_series else [best_metric]
    accepted = 0
    min_improvement_abs = float(target.get("min_improvement_abs", 0.0))
    min_improvement_rel = float(target.get("min_improvement_rel", 0.0))
    confirm_repeats = int(target.get("confirm_repeats", args.confirm_repeats))
    min_effect_sigma = float(target.get("min_effect_sigma", 0.0))
    ci_z = float(target.get("ci_z", 1.0))
    require_ci_separation = bool(target.get("require_ci_separation", False))
    min_series_samples = int(target.get("min_series_samples", 2))
    ab_repeats = int(target.get("ab_repeats", 0))
    max_rel_stdev = target.get("max_rel_stdev")
    max_rel_stdev_f = float(max_rel_stdev) if max_rel_stdev is not None else None

    prepare.append_log(
        {
            "event": "loop_start",
            "timestamp": prepare.now_iso(),
            "target": args.target,
            "best_metric": best_metric,
            "run_label": run_label,
            "runtime_env": run_env,
            "target_overrides": target_overrides,
            "iterations": args.iterations,
            "mode": "openai" if os.getenv("OPENAI_API_KEY") else "heuristic",
            "stats_gate": {
                "min_effect_sigma": min_effect_sigma,
                "ci_z": ci_z,
                "require_ci_separation": require_ci_separation,
                "min_series_samples": min_series_samples,
                "ab_repeats": ab_repeats,
            },
        }
    )

    blocked_mutations_until: dict[str, int] = {}
    blocked_mutation_ttl = max(0, int(target.get("blocked_mutation_ttl", 10)))
    mutation_attempts = load_mutation_attempt_counts(args.target)

    for iteration in range(1, args.iterations + 1):
        current_source = source_path.read_text()

        mutation = ""
        diagnostics: dict[str, Any] = {}
        active_blocked = {name for name, until in blocked_mutations_until.items() if until >= iteration}
        if active_blocked:
            diagnostics["active_blocked_mutations"] = sorted(active_blocked)

        effective_rel_threshold, threshold_diag = resolve_min_improvement_rel(
            target=target,
            configured_rel=min_improvement_rel,
            baseline_series=best_metric_series,
        )
        diagnostics["threshold"] = threshold_diag

        if os.getenv("OPENAI_API_KEY"):
            prompt = build_prompt(
                template,
                target_name=args.target,
                metric_name=target["metric_name"],
                language=language,
                source_code=current_source,
            )
            candidate, llm_diagnostics = request_openai_candidate(
                model=args.model,
                system_prompt=f"Return only valid {language} source code.",
                user_prompt=prompt,
                temperature=args.temperature,
            )
            diagnostics.update(llm_diagnostics)
            if candidate is None:
                candidate, mutation, changed = heuristic_candidate(
                    current_source,
                    iteration,
                    language,
                    source_path,
                    blocked_mutations=active_blocked,
                    mutation_attempts=mutation_attempts,
                    target_config=target,
                )
                mutation = f"fallback_{mutation}"
            else:
                changed = candidate != current_source
                mutation = "openai_patch"
        else:
            candidate, mutation, changed = heuristic_candidate(
                current_source,
                iteration,
                language,
                source_path,
                blocked_mutations=active_blocked,
                mutation_attempts=mutation_attempts,
                target_config=target,
            )

        if not changed:
            prepare.append_result_row(
                target=args.target,
                iteration=iteration,
                status="skipped",
                metric_name=target["metric_name"],
                metric_value=best_metric,
                higher_is_better=bool(target["higher_is_better"]),
                check_s=0.0,
                info_or_bench_s=0.0,
                execute_s=0.0,
                notes=f"{mutation}:no_change;run={run_label}",
            )
            continue

        mutation_attempts[mutation] = mutation_attempts.get(mutation, 0) + 1

        source_path.write_text(candidate)
        try:
            result = prepare.evaluate_target(args.target)

            metric_value = result.get("metric_value")
            metric_for_row = float(metric_value) if metric_value is not None else None
            success = result.get("status") == "success" and metric_value is not None

            metric_series = parse_metric_series(result.get("debug", {}).get("metric_values"))
            rel_stdev = None
            if metric_series is not None and len(metric_series) >= 2:
                try:
                    rel_stdev = relative_stdev(metric_series)
                    diagnostics["rel_stdev"] = rel_stdev
                except Exception:  # noqa: BLE001
                    rel_stdev = None

            improved = bool(
                success
                and is_better(
                    float(metric_value),
                    best_metric,
                    higher_is_better=bool(target["higher_is_better"]),
                    min_improvement_abs=min_improvement_abs,
                    min_improvement_rel=effective_rel_threshold,
                )
            )

            notes = mutation
            if diagnostics:
                notes = f"{notes};diag={diagnostics.get('reason', 'n/a')}"

            if success and rel_stdev is not None and max_rel_stdev_f is not None and rel_stdev > max_rel_stdev_f:
                improved = False
                source_path.write_text(current_source)
                notes = f"rejected_high_variance:{notes};rel_stdev={rel_stdev:.6f}"

            if (
                improved
                and metric_series is not None
                and len(metric_series) >= min_series_samples
                and len(best_metric_series) >= min_series_samples
                and (min_effect_sigma > 0.0 or require_ci_separation)
            ):
                dist_ok, dist_reason, dist_stats = distribution_guard(
                    candidate_values=metric_series,
                    baseline_values=best_metric_series,
                    higher_is_better=bool(target["higher_is_better"]),
                    min_effect_sigma=min_effect_sigma,
                    ci_z=ci_z,
                    require_ci_separation=require_ci_separation,
                )
                diagnostics["distribution"] = dist_stats
                if not dist_ok:
                    improved = False
                    source_path.write_text(current_source)
                    notes = f"rejected_distribution_{dist_reason}:{notes}"

            best_before = best_metric
            if improved:
                confirmed_ok, confirmed_metric, confirmed_values, confirm_reason = confirm_improvement(
                    target_name=args.target,
                    best_metric=best_metric,
                    higher_is_better=bool(target["higher_is_better"]),
                    min_improvement_abs=min_improvement_abs,
                    min_improvement_rel=effective_rel_threshold,
                    repeats=confirm_repeats,
                )
                if confirmed_values:
                    diagnostics["confirm_values"] = confirmed_values
                if confirmed_ok:
                    ab_ok, ab_details, ab_reason = ab_validate_candidate(
                        source_path=source_path,
                        target_name=args.target,
                        candidate_source=candidate,
                        original_source=current_source,
                        higher_is_better=bool(target["higher_is_better"]),
                        min_improvement_abs=min_improvement_abs,
                        min_improvement_rel=effective_rel_threshold,
                        repeats=ab_repeats,
                    )
                    if ab_details:
                        diagnostics["ab_validation"] = ab_details

                    if ab_ok:
                        accepted += 1
                        blocked_mutations_until.clear()
                        best_metric = float(metric_value) if confirmed_metric is None else confirmed_metric
                        if confirmed_metric is not None:
                            metric_for_row = confirmed_metric
                        if metric_series is not None and metric_series:
                            best_metric_series = metric_series
                        elif confirmed_values:
                            best_metric_series = [float(v) for v in confirmed_values]
                        notes = f"accepted:{notes};{confirm_reason};{ab_reason}"
                    else:
                        improved = False
                        source_path.write_text(current_source)
                        notes = f"rejected_{ab_reason}:{notes};{confirm_reason}"
                else:
                    improved = False
                    source_path.write_text(current_source)
                    notes = f"rejected_{confirm_reason}:{notes}"
            else:
                source_path.write_text(current_source)
                if success:
                    raw_better = is_better(
                        float(metric_value),
                        best_metric,
                        higher_is_better=bool(target["higher_is_better"]),
                    )
                    if raw_better and (min_improvement_abs > 0.0 or effective_rel_threshold > 0.0):
                        notes = f"rejected_below_threshold:{notes}"
                    else:
                        notes = f"rejected_not_better:{notes}"
                else:
                    notes = f"rejected_eval_failed:{notes}"

            if not improved:
                blocked_mutations_until[mutation] = iteration + blocked_mutation_ttl

            should_write_artifact = args.artifacts == "all" or (args.artifacts == "accepted" and improved)
            if should_write_artifact:
                artifact_source = candidate if improved else current_source
                write_iteration_artifact(
                    target_name=args.target,
                    run_label=run_label,
                    iteration=iteration,
                    source_path=source_path,
                    language=language,
                    previous_source=current_source,
                    candidate_source=artifact_source,
                    mutation=mutation,
                    accepted=improved,
                    best_before=best_before,
                    best_after=best_metric,
                    diagnostics=diagnostics,
                    result=result,
                    runtime_env=run_env,
                )

            prepare.append_result_row(
                target=args.target,
                iteration=iteration,
                status=result.get("status", "failed") if success else "failed",
                metric_name=target["metric_name"],
                metric_value=metric_for_row,
                higher_is_better=bool(target["higher_is_better"]),
                check_s=float(result.get("check_s", 0.0)),
                info_or_bench_s=float(result.get("info_or_bench_s", 0.0)),
                execute_s=float(result.get("execute_s", 0.0)),
                notes=f"{notes};run={run_label}",
            )

            prepare.append_log(
                {
                    "event": "loop_iteration",
                    "timestamp": prepare.now_iso(),
                    "target": args.target,
                    "iteration": iteration,
                    "mutation": mutation,
                    "accepted": improved,
                    "best_metric": best_metric,
                    "effective_threshold": threshold_diag,
                    "result": {
                        "status": result.get("status"),
                        "metric_name": result.get("metric_name"),
                        "metric_value": result.get("metric_value"),
                        "notes": result.get("notes"),
                    },
                    "diagnostics": diagnostics,
                },
            )

            if args.max_accepted and accepted >= args.max_accepted:
                break
        except BaseException:
            source_path.write_text(current_source)
            raise

    print(
        json.dumps(
            {
                "target": args.target,
                "best_metric": best_metric,
                "accepted": accepted,
                "iterations_requested": args.iterations,
                "max_accepted": args.max_accepted,
            },
            indent=2,
            sort_keys=True,
        )
    )

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AutoPoseidon train loop (Karpathy-compatible entrypoint)")
    parser.add_argument("--target", default="cairo_poseidon_style_t8", help="Target from config/targets.json")
    parser.add_argument("--iterations", type=int, default=25, help="Max optimization iterations")
    parser.add_argument("--max-accepted", type=int, default=0, help="Stop after N accepted mutations (0 = no cap)")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5-mini"), help="Model for OpenAI mode")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature for OpenAI mode")
    parser.add_argument(
        "--artifacts",
        choices=["none", "accepted", "all"],
        default="accepted",
        help="Write iteration artifact bundles for accepted or all evaluated candidates",
    )
    parser.add_argument(
        "--confirm-repeats",
        type=int,
        default=0,
        help="Extra full evaluations required to confirm an apparent improvement (target config can override)",
    )
    parser.add_argument(
        "--target-overrides-json",
        default="",
        help="Optional JSON file with per-target acceptance overrides",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_loop(args)


if __name__ == "__main__":
    raise SystemExit(main())
