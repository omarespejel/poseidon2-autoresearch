#!/usr/bin/env python3
"""Karpathy-compatible autonomous optimization loop for AutoPoseidon targets.

This is the canonical loop entrypoint in compatibility mode:
- prepare.py: fixed evaluator + utilities
- train.py: editable optimization loop
- program.md: human-authored instructions
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import difflib
import hashlib
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
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
    "require_metric_series_for_stats",
    "ab_repeats",
    "blocked_mutation_ttl",
    "mutation_schedule",
    "mutation_ucb_explore",
}
DEFAULT_MUTATION_MEMORY_FILE = ROOT / "work" / "mutation_memory.json"
DEFAULT_MUTATION_MEMORY_MAX_ENTRIES = 512
DEFAULT_MUTATION_MEMORY_STALE_SECONDS = 14 * 24 * 60 * 60
DEFAULT_MUTATION_MEMORY_ACCEPTED_STALE_MULTIPLIER = 4.0
DEFAULT_MUTATION_MEMORY_MIN_ACCEPTED_TOTAL = 2
DEFAULT_MUTATION_MEMORY_MIN_SUCCESS_RATE = 0.30
DEFAULT_MUTATION_MEMORY_SEED_VERSION = 2


def configure_debug_environment(args: argparse.Namespace) -> None:
    if args.verbose > 0:
        existing = os.getenv("AUTORESEARCH_VERBOSE", "").strip()
        try:
            existing_level = int(existing) if existing else 0
        except ValueError:
            existing_level = 0
        os.environ["AUTORESEARCH_VERBOSE"] = str(max(existing_level, args.verbose))
    if args.debug_command_output:
        os.environ["AUTORESEARCH_DEBUG_COMMAND_OUTPUT"] = "1"
    os.environ["AUTORESEARCH_DEBUG_MAX_CHARS"] = str(max(256, int(args.debug_max_chars)))


def train_log(args: argparse.Namespace, message: str, *, level: int = 1) -> None:
    if int(getattr(args, "verbose", 0)) < level:
        return
    print(f"[train {prepare.now_iso()}] {message}", file=sys.stderr, flush=True)


def summarize_notes(notes: str, *, max_len: int = 220) -> str:
    if len(notes) <= max_len:
        return notes
    return f"{notes[:max_len]}... [truncated]"


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


def replace_nth_occurrence(source: str, old: str, new: str, occurrence: int) -> tuple[str, bool]:
    if occurrence < 1:
        return source, False
    start = 0
    idx = -1
    for _ in range(occurrence):
        idx = source.find(old, start)
        if idx == -1:
            return source, False
        start = idx + len(old)
    candidate = source[:idx] + new + source[idx + len(old) :]
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
    candidate, changed = replace_nth_occurrence(source, old, replacement, 1)
    if not changed:
        return source, "rust_pack_point_buffer:replace_failed", False
    return candidate, "rust_pack_point_buffer", True


def rust_mutator_pack_point_buffer_verify(source: str) -> tuple[str, str, bool]:
    old = "let packed_point = MultilinearPoint([betas.clone(), air_claims.point.0].concat());"
    replacement = (
        "let mut packed_point_coords = betas.clone();\n"
        "        packed_point_coords.extend_from_slice(&air_claims.point.0);\n"
        "        let packed_point = MultilinearPoint(packed_point_coords);"
    )
    candidate, changed = replace_nth_occurrence(source, old, replacement, 2)
    if not changed:
        return source, "rust_pack_point_buffer_verify:pattern_missing", False
    return candidate, "rust_pack_point_buffer_verify", True


def rust_mutator_pack_point_buffer_both(source: str) -> tuple[str, str, bool]:
    old = "let packed_point = MultilinearPoint([betas.clone(), air_claims.point.0].concat());"
    if source.count(old) < 2:
        return source, "rust_pack_point_buffer_both:pattern_missing", False
    replacement = (
        "let mut packed_point_coords = betas.clone();\n"
        "        packed_point_coords.extend_from_slice(&air_claims.point.0);\n"
        "        let packed_point = MultilinearPoint(packed_point_coords);"
    )
    candidate = source.replace(old, replacement, 2)
    return candidate, "rust_pack_point_buffer_both", True


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


def rust_mutator_x86_internal_for_loop(source: str) -> tuple[str, str, bool]:
    candidate, _, changed = rust_mutator_neon_internal_for_loop(source)
    if not changed:
        return source, "rust_x86_internal_for_loop:pattern_missing", False
    return candidate, "rust_x86_internal_for_loop", True


def rust_mutator_x86_internal_for_loop_w24(source: str) -> tuple[str, str, bool]:
    marker = "self.packed_internal_constants.iter().for_each(|&rc| {"
    first = source.find(marker)
    if first == -1:
        return source, "rust_x86_internal_for_loop_w24:pattern_missing", False
    second = source.find(marker, first + 1)
    if second == -1:
        # Fallback for files that only contain one internal loop.
        candidate, _, changed = rust_mutator_x86_internal_for_loop(source)
        if not changed:
            return source, "rust_x86_internal_for_loop_w24:pattern_missing", False
        return candidate, "rust_x86_internal_for_loop_w24:fallback_single", True
    candidate, _, changed = rust_mutator_neon_internal_for_loop_w24(source)
    if not changed:
        return source, "rust_x86_internal_for_loop_w24:pattern_missing", False
    return candidate, "rust_x86_internal_for_loop_w24", True


def rust_mutator_x86_internal_for_loop_both(source: str) -> tuple[str, str, bool]:
    marker = "self.packed_internal_constants.iter().for_each(|&rc| {"
    occurrences = source.count(marker)
    if occurrences <= 0:
        return source, "rust_x86_internal_for_loop_both:pattern_missing", False
    if occurrences == 1:
        candidate, _, changed = rust_mutator_x86_internal_for_loop(source)
        if not changed:
            return source, "rust_x86_internal_for_loop_both:pattern_missing", False
        return candidate, "rust_x86_internal_for_loop_both:fallback_single", True
    candidate, _, changed = rust_mutator_neon_internal_for_loop_both(source)
    if not changed:
        return source, "rust_x86_internal_for_loop_both:pattern_missing", False
    return candidate, "rust_x86_internal_for_loop_both", True


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


def _mutate_first_match(
    source: str,
    rules: list[tuple[str, str, str]],
    *,
    pattern_missing_note: str,
) -> tuple[str, str, bool]:
    for old, new, name in rules:
        if old in source:
            return source.replace(old, new, 1), name, True
    return source, pattern_missing_note, False


AVX2_SUM_VEC_OLD = (
    "                ILP::add_sum(\n"
    "                    &mut internal_state.s_hi,\n"
    "                    transmute::<PackedMontyField31AVX2<FP>, __m256i>(sum),\n"
    "                );"
)
AVX2_SUM_VEC_NEW = (
    "                let sum_vec = transmute::<PackedMontyField31AVX2<FP>, __m256i>(sum);\n"
    "                ILP::add_sum(&mut internal_state.s_hi, sum_vec);"
)
AVX512_SUM_VEC_OLD = (
    "                ILP::add_sum(\n"
    "                    &mut internal_state.s_hi,\n"
    "                    transmute::<PackedMontyField31AVX512<FP>, __m512i>(sum),\n"
    "                );"
)
AVX512_SUM_VEC_NEW = (
    "                let sum_vec = transmute::<PackedMontyField31AVX512<FP>, __m512i>(sum);\n"
    "                ILP::add_sum(&mut internal_state.s_hi, sum_vec);"
)


def rust_mutator_avx2_add_sum_loops(source: str) -> tuple[str, str, bool]:
    return _mutate_first_match(
        source,
        [
            (
                "input.as_mut()[..5]\n                .iter_mut()\n                .for_each(|x| *x = mm256_mod_add(sum, *x, PMP::PACKED_P));",
                "for x in input.as_mut()[..5].iter_mut() {\n            *x = mm256_mod_add(sum, *x, PMP::PACKED_P);\n        }",
                "rust_avx_add_sum_head_add",
            ),
            (
                "input.as_mut()[5..8]\n                .iter_mut()\n                .for_each(|x| *x = mm256_mod_sub(sum, *x, PMP::PACKED_P));",
                "for x in input.as_mut()[5..8].iter_mut() {\n            *x = mm256_mod_sub(sum, *x, PMP::PACKED_P);\n        }",
                "rust_avx_add_sum_head_sub",
            ),
            (
                "input.as_mut()[8..]\n                .iter_mut()\n                .for_each(|x| *x = signed_add_avx2::<PMP>(sum, *x));",
                "for x in input.as_mut()[8..].iter_mut() {\n            *x = signed_add_avx2::<PMP>(sum, *x);\n        }",
                "rust_avx_add_sum_tail_signed",
            ),
        ],
        pattern_missing_note="rust_avx2_add_sum:pattern_missing",
    )


def rust_mutator_avx2_add_sum_as_mut_hoist(source: str) -> tuple[str, str, bool]:
    marker = "unsafe fn add_sum(input: &mut Self::ArrayLike, sum: __m256i) {"
    idx = source.find(marker)
    if idx == -1:
        return source, "rust_avx2_add_sum_as_mut_hoist:pattern_missing", False

    end = source.find("\n}", idx)
    segment = source[idx : (end if end != -1 else len(source))]
    if "let input = input.as_mut();" in segment:
        return source, "rust_avx2_add_sum_as_mut_hoist:already_present", False

    old_header = "unsafe fn add_sum(input: &mut Self::ArrayLike, sum: __m256i) {\n        unsafe {\n"
    if old_header not in source:
        return source, "rust_avx2_add_sum_as_mut_hoist:header_missing", False

    candidate = source.replace(
        old_header,
        "unsafe fn add_sum(input: &mut Self::ArrayLike, sum: __m256i) {\n        unsafe {\n            let input = input.as_mut();\n",
        1,
    )
    candidate = candidate.replace("input.as_mut()[..5]", "input[..5]", 1)
    candidate = candidate.replace("input.as_mut()[5..8]", "input[5..8]", 1)
    candidate = candidate.replace("input.as_mut()[8..]", "input[8..]", 1)
    return candidate, "rust_avx2_add_sum_as_mut_hoist", True


def rust_mutator_avx512_add_sum_loops(source: str) -> tuple[str, str, bool]:
    return _mutate_first_match(
        source,
        [
            (
                "input.as_mut()[..5]\n            .iter_mut()\n            .for_each(|x| *x = mm512_mod_add(sum, *x, PMP::PACKED_P));",
                "for x in input.as_mut()[..5].iter_mut() {\n            *x = mm512_mod_add(sum, *x, PMP::PACKED_P);\n        }",
                "rust_avx512_add_sum_head_add",
            ),
            (
                "input.as_mut()[5..(8 + Self::NUM_POS)]\n            .iter_mut()\n            .for_each(|x| *x = mm512_mod_sub(sum, *x, PMP::PACKED_P));",
                "for x in input.as_mut()[5..(8 + Self::NUM_POS)].iter_mut() {\n            *x = mm512_mod_sub(sum, *x, PMP::PACKED_P);\n        }",
                "rust_avx512_add_sum_mid_sub",
            ),
            (
                "input.as_mut()[8 + Self::NUM_POS..]\n            .iter_mut()\n            .for_each(|x| *x = mm512_mod_add(sum, *x, PMP::PACKED_P));",
                "for x in input.as_mut()[8 + Self::NUM_POS..].iter_mut() {\n            *x = mm512_mod_add(sum, *x, PMP::PACKED_P);\n        }",
                "rust_avx512_add_sum_tail_add",
            ),
        ],
        pattern_missing_note="rust_avx512_add_sum:pattern_missing",
    )


def rust_mutator_avx512_add_sum_as_mut_hoist(source: str) -> tuple[str, str, bool]:
    marker = "unsafe fn add_sum(input: &mut Self::ArrayLike, sum: __m512i) {"
    idx = source.find(marker)
    if idx == -1:
        return source, "rust_avx512_add_sum_as_mut_hoist:pattern_missing", False

    end = source.find("\n}", idx)
    segment = source[idx : (end if end != -1 else len(source))]
    if "let input = input.as_mut();" in segment:
        return source, "rust_avx512_add_sum_as_mut_hoist:already_present", False

    old_header = "unsafe fn add_sum(input: &mut Self::ArrayLike, sum: __m512i) {\n"
    if old_header not in source:
        return source, "rust_avx512_add_sum_as_mut_hoist:header_missing", False

    candidate = source.replace(
        old_header,
        "unsafe fn add_sum(input: &mut Self::ArrayLike, sum: __m512i) {\n        let input = input.as_mut();\n",
        1,
    )
    candidate = candidate.replace("input.as_mut()[..5]", "input[..5]", 1)
    candidate = candidate.replace("input.as_mut()[5..(8 + Self::NUM_POS)]", "input[5..(8 + Self::NUM_POS)]", 1)
    candidate = candidate.replace("input.as_mut()[8 + Self::NUM_POS..]", "input[8 + Self::NUM_POS..]", 1)
    return candidate, "rust_avx512_add_sum_as_mut_hoist", True


def rust_mutator_avx2_sum_vec_hoist(source: str) -> tuple[str, str, bool]:
    return _mutate_first_match(
        source,
        [(AVX2_SUM_VEC_OLD, AVX2_SUM_VEC_NEW, "rust_avx_sum_vec_hoist")],
        pattern_missing_note="rust_avx_sum_vec_hoist:pattern_missing",
    )


def rust_mutator_avx512_sum_vec_hoist(source: str) -> tuple[str, str, bool]:
    return _mutate_first_match(
        source,
        [(AVX512_SUM_VEC_OLD, AVX512_SUM_VEC_NEW, "rust_avx512_sum_vec_hoist")],
        pattern_missing_note="rust_avx512_sum_vec_hoist:pattern_missing",
    )


def rust_mutator_avx2_sum_vec_hoist_w24(source: str) -> tuple[str, str, bool]:
    candidate, changed = replace_nth_occurrence(source, AVX2_SUM_VEC_OLD, AVX2_SUM_VEC_NEW, 2)
    if changed:
        return candidate, "rust_avx_sum_vec_hoist_w24", True
    return source, "rust_avx_sum_vec_hoist_w24:pattern_missing", False


def rust_mutator_avx512_sum_vec_hoist_w24(source: str) -> tuple[str, str, bool]:
    candidate, changed = replace_nth_occurrence(source, AVX512_SUM_VEC_OLD, AVX512_SUM_VEC_NEW, 2)
    if changed:
        return candidate, "rust_avx512_sum_vec_hoist_w24", True
    return source, "rust_avx512_sum_vec_hoist_w24:pattern_missing", False


def rust_mutator_avx2_sum_vec_hoist_both(source: str) -> tuple[str, str, bool]:
    if source.count(AVX2_SUM_VEC_OLD) < 2:
        return source, "rust_avx_sum_vec_hoist_both:pattern_missing", False
    return source.replace(AVX2_SUM_VEC_OLD, AVX2_SUM_VEC_NEW, 2), "rust_avx_sum_vec_hoist_both", True


def rust_mutator_avx512_sum_vec_hoist_both(source: str) -> tuple[str, str, bool]:
    if source.count(AVX512_SUM_VEC_OLD) < 2:
        return source, "rust_avx512_sum_vec_hoist_both:pattern_missing", False
    return (
        source.replace(AVX512_SUM_VEC_OLD, AVX512_SUM_VEC_NEW, 2),
        "rust_avx512_sum_vec_hoist_both",
        True,
    )


def rust_mutator_no_packing_internal_inline(source: str) -> tuple[str, str, bool]:
    marker = "    fn new_from_constants(internal_constants: Vec<MontyField31<FP>>) -> Self {"
    if marker not in source:
        return source, "rust_no_packing_internal_inline:pattern_missing", False
    if "#[inline(always)]\n    fn new_from_constants(internal_constants: Vec<MontyField31<FP>>) -> Self {" in source:
        return source, "rust_no_packing_internal_inline:already_present", False
    return source.replace(marker, "    #[inline(always)]\n" + marker, 1), "rust_no_packing_internal_inline", True


def rust_mutator_no_packing_external_inline(source: str) -> tuple[str, str, bool]:
    marker = "    fn new_from_constants(external_constants: ExternalLayerConstants<MontyField31<FP>, WIDTH>) -> Self {"
    if marker not in source:
        return source, "rust_no_packing_external_inline:pattern_missing", False
    if (
        "#[inline(always)]\n"
        "    fn new_from_constants(external_constants: ExternalLayerConstants<MontyField31<FP>, WIDTH>) -> Self {"
    ) in source:
        return source, "rust_no_packing_external_inline:already_present", False
    return source.replace(marker, "    #[inline(always)]\n" + marker, 1), "rust_no_packing_external_inline", True


def rust_mutator_monty_internal_for_loop(source: str) -> tuple[str, str, bool]:
    marker = "self.internal_constants.iter().for_each(|rc| {"
    idx = source.find(marker)
    if idx == -1:
        return source, "rust_monty_internal_for_loop:pattern_missing", False

    tail = source[idx:].replace(marker, "for rc in &self.internal_constants {", 1)
    close = tail.find("\n        })")
    if close == -1:
        return source, "rust_monty_internal_for_loop:closing_missing", False
    tail = tail[:close] + "\n        }" + tail[close + len("\n        })") :]
    return source[:idx] + tail, "rust_monty_internal_for_loop", True


def rust_mutator_monty_internal_for_copied(source: str) -> tuple[str, str, bool]:
    marker = "self.internal_constants.iter().for_each(|rc| {"
    idx = source.find(marker)
    if idx == -1:
        return source, "rust_monty_internal_for_copied:pattern_missing", False

    tail = source[idx:].replace(marker, "for rc in self.internal_constants.iter().copied() {", 1)
    close = tail.find("\n        })")
    if close == -1:
        return source, "rust_monty_internal_for_copied:closing_missing", False
    tail = tail[:close] + "\n        }" + tail[close + len("\n        })") :]
    tail = tail.replace("state[0] += *rc;", "state[0] += rc;", 1)
    return source[:idx] + tail, "rust_monty_internal_for_copied", True


def rust_mutator_monty_s0_cache_generic(source: str) -> tuple[str, str, bool]:
    old = (
        "        let part_sum: R = state[1..].iter().copied().sum();\n"
        "        let full_sum = part_sum + state[0];\n"
        "        state[0] = part_sum - state[0];"
    )
    new = (
        "        let part_sum: R = state[1..].iter().copied().sum();\n"
        "        let s0 = state[0];\n"
        "        let full_sum = part_sum + s0;\n"
        "        state[0] = part_sum - s0;"
    )
    if old not in source:
        return source, "rust_monty_s0_cache_generic:pattern_missing", False
    return source.replace(old, new, 1), "rust_monty_s0_cache_generic", True


def rust_mutator_monty_s0_cache_internal(source: str) -> tuple[str, str, bool]:
    old = (
        "            let part_sum: MontyField31<FP> = state[1..].iter().copied().sum();\n"
        "            let full_sum = part_sum + state[0];\n"
        "            state[0] = part_sum - state[0];"
    )
    new = (
        "            let part_sum: MontyField31<FP> = state[1..].iter().copied().sum();\n"
        "            let s0 = state[0];\n"
        "            let full_sum = part_sum + s0;\n"
        "            state[0] = part_sum - s0;"
    )
    if old not in source:
        return source, "rust_monty_s0_cache_internal:pattern_missing", False
    return source.replace(old, new, 1), "rust_monty_s0_cache_internal", True


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


def python_replace_first(
    source: str,
    replacements: list[tuple[str, str]],
    *,
    mutation: str,
) -> tuple[str, str, bool]:
    for old, new in replacements:
        if old in source:
            return source.replace(old, new, 1), mutation, True
    return source, f"{mutation}:pattern_missing", False


def python_replace_first_with_prefix(
    source: str,
    prefix: str,
    replacements: list[tuple[str, str]],
    *,
    mutation: str,
) -> tuple[str, str, bool]:
    return python_replace_first(
        source,
        [(f"{prefix}{old}", f"{prefix}{new}") for old, new in replacements],
        mutation=mutation,
    )


def python_mutator_diff_multi_delta_prob_up(source: str) -> tuple[str, str, bool]:
    return python_replace_first_with_prefix(
        source,
        "        delta[lane_idx] = rng.randrange(1, spec.modulus)\n        ",
        [
            ("if rng.random() < 0.35:", "if rng.random() < 0.50:"),
            ("if rng.random() < 0.30:", "if rng.random() < 0.45:"),
            ("if rng.random() < 0.25:", "if rng.random() < 0.35:"),
        ],
        mutation="python_trackb_diff_multi_delta_prob_up",
    )


def python_mutator_diff_multi_delta_prob_down(source: str) -> tuple[str, str, bool]:
    return python_replace_first_with_prefix(
        source,
        "        delta[lane_idx] = rng.randrange(1, spec.modulus)\n        ",
        [
            ("if rng.random() < 0.50:", "if rng.random() < 0.35:"),
            ("if rng.random() < 0.45:", "if rng.random() < 0.30:"),
            ("if rng.random() < 0.35:", "if rng.random() < 0.25:"),
        ],
        mutation="python_trackb_diff_multi_delta_prob_down",
    )


def python_mutator_diff_cross_lane_prob_up(source: str) -> tuple[str, str, bool]:
    return python_replace_first_with_prefix(
        source,
        "                if i == lane_idx:\n                    continue\n                ",
        [
            ("if rng.random() < 0.2:", "if rng.random() < 0.30:"),
            ("if rng.random() < 0.25:", "if rng.random() < 0.35:"),
            ("if rng.random() < 0.15:", "if rng.random() < 0.25:"),
        ],
        mutation="python_trackb_diff_cross_lane_prob_up",
    )


def python_mutator_diff_cross_lane_prob_down(source: str) -> tuple[str, str, bool]:
    return python_replace_first_with_prefix(
        source,
        "                if i == lane_idx:\n                    continue\n                ",
        [
            ("if rng.random() < 0.30:", "if rng.random() < 0.2:"),
            ("if rng.random() < 0.35:", "if rng.random() < 0.25:"),
            ("if rng.random() < 0.25:", "if rng.random() < 0.15:"),
        ],
        mutation="python_trackb_diff_cross_lane_prob_down",
    )


def python_mutator_mitm_bucket_cap_up(source: str) -> tuple[str, str, bool]:
    return python_replace_first_with_prefix(
        source,
        "        bucket = table.setdefault(key, [])\n        ",
        [
            ("if len(bucket) < 2:", "if len(bucket) < 4:"),
            ("if len(bucket) < 3:", "if len(bucket) < 5:"),
            ("if len(bucket) < 4:", "if len(bucket) < 6:"),
            ("if len(bucket) < 5:", "if len(bucket) < 7:"),
        ],
        mutation="python_trackb_mitm_bucket_cap_up",
    )


def python_mutator_mitm_bucket_cap_down(source: str) -> tuple[str, str, bool]:
    return python_replace_first_with_prefix(
        source,
        "        bucket = table.setdefault(key, [])\n        ",
        [
            ("if len(bucket) < 7:", "if len(bucket) < 5:"),
            ("if len(bucket) < 6:", "if len(bucket) < 4:"),
            ("if len(bucket) < 5:", "if len(bucket) < 3:"),
            ("if len(bucket) < 4:", "if len(bucket) < 2:"),
        ],
        mutation="python_trackb_mitm_bucket_cap_down",
    )


def python_mutator_algebraic_fit_gain_up(source: str) -> tuple[str, str, bool]:
    return python_replace_first(
        source,
        [
            (
                "        fit_gain = (2.5 * val_exact_rate) + (1.5 * train_exact_rate)",
                "        fit_gain = (3.0 * val_exact_rate) + (1.75 * train_exact_rate)",
            ),
            (
                "        fit_gain = (2.0 * val_exact_rate) + (1.25 * train_exact_rate)",
                "        fit_gain = (2.5 * val_exact_rate) + (1.5 * train_exact_rate)",
            ),
            (
                "        fit_gain = (1.5 * val_exact_rate) + (1.0 * train_exact_rate)",
                "        fit_gain = (2.0 * val_exact_rate) + (1.25 * train_exact_rate)",
            ),
        ],
        mutation="python_trackb_algebraic_fit_gain_up",
    )


def python_mutator_algebraic_fit_gain_down(source: str) -> tuple[str, str, bool]:
    return python_replace_first(
        source,
        [
            (
                "        fit_gain = (3.0 * val_exact_rate) + (1.75 * train_exact_rate)",
                "        fit_gain = (2.5 * val_exact_rate) + (1.5 * train_exact_rate)",
            ),
            (
                "        fit_gain = (2.5 * val_exact_rate) + (1.5 * train_exact_rate)",
                "        fit_gain = (2.0 * val_exact_rate) + (1.25 * train_exact_rate)",
            ),
            (
                "        fit_gain = (2.0 * val_exact_rate) + (1.25 * train_exact_rate)",
                "        fit_gain = (1.5 * val_exact_rate) + (1.0 * train_exact_rate)",
            ),
        ],
        mutation="python_trackb_algebraic_fit_gain_down",
    )


def python_heuristic_candidate(
    source: str,
    iteration: int,
    source_path: Path,
    blocked_mutations: set[str] | None = None,
    mutation_attempts: dict[str, int] | None = None,
    target_config: dict[str, Any] | None = None,
    preferred_mutations: list[str] | None = None,
    mutation_memory: dict[str, Any] | None = None,
    target_name: str = "",
) -> tuple[str, str, bool]:
    if source_path.name.lower() != "attack_harness.py":
        return source, "python_no_change", False

    operators: list[Any] = [
        python_mutator_diff_multi_delta_prob_up,
        python_mutator_diff_multi_delta_prob_down,
        python_mutator_diff_cross_lane_prob_up,
        python_mutator_diff_cross_lane_prob_down,
        python_mutator_mitm_bucket_cap_up,
        python_mutator_mitm_bucket_cap_down,
        python_mutator_algebraic_fit_gain_up,
        python_mutator_algebraic_fit_gain_down,
    ]

    shift = (iteration - 1) % len(operators)
    ordered = operators[shift:] + operators[:shift]
    preferred_rank = {name: idx for idx, name in enumerate(preferred_mutations or [])}

    def preference_for_label(label: str) -> tuple[int, int]:
        if label in preferred_rank:
            return (0, preferred_rank[label])
        if "+" in label:
            parts = [part.strip() for part in label.split("+") if part.strip()]
            part_ranks = [preferred_rank[part] for part in parts if part in preferred_rank]
            if part_ranks:
                return (1, min(part_ranks))
        return (2, 1_000_000)

    attempts = mutation_attempts or {}
    mutation_schedule = str((target_config or {}).get("mutation_schedule", "priority")).strip().lower()
    use_ucb_schedule = mutation_schedule == "ucb"
    mutation_ucb_explore = float((target_config or {}).get("mutation_ucb_explore", 0.75))
    scope_totals = mutation_memory_scope_totals(
        mutation_memory,
        target_name=target_name,
        language="python",
    )

    candidates: list[dict[str, Any]] = []
    seen_candidates: set[str] = set()
    for idx, operator in enumerate(ordered):
        candidate, mutation, changed = operator(source)
        if blocked_mutations and mutation in blocked_mutations:
            continue
        if not changed:
            continue
        if candidate in seen_candidates:
            continue
        seen_candidates.add(candidate)
        pref_class, pref_rank_value = preference_for_label(mutation)
        candidate_entry: dict[str, Any] = {
            "candidate": candidate,
            "mutation": mutation,
            "pref_class": pref_class,
            "pref_rank": pref_rank_value,
            "attempts": attempts.get(mutation, 0),
            "index": idx,
        }
        if use_ucb_schedule:
            accepted_hist, rejected_hist, history_scope = mutation_memory_counts(
                mutation_memory,
                mutation,
                target_name=target_name,
                language="python",
            )
            total_memory_observations = float(
                scope_totals.get(history_scope, scope_totals.get("global", 0.0))
            )
            candidate_entry["ucb_score"] = mutation_ucb_score(
                accepted=accepted_hist,
                rejected=rejected_hist,
                total_observations=total_memory_observations,
                explore=mutation_ucb_explore,
                preference_class=int(pref_class),
                schedule_tier=0,
                attempt_count=int(candidate_entry["attempts"]),
            )
        candidates.append(candidate_entry)

    if candidates:
        if use_ucb_schedule:
            candidates.sort(
                key=lambda item: (
                    -float(item.get("ucb_score", 0.0)),
                    int(item["pref_class"]),
                    int(item["pref_rank"]),
                    int(item["attempts"]),
                    int(item["index"]),
                )
            )
        else:
            candidates.sort(
                key=lambda item: (
                    int(item["pref_class"]),
                    int(item["pref_rank"]),
                    int(item["attempts"]),
                    int(item["index"]),
                )
            )
        chosen = candidates[0]
        return str(chosen["candidate"]), str(chosen["mutation"]), True
    return source, "python_no_change", False


def json_dump_stable(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def json_heuristic_candidate(
    source: str,
    iteration: int,
    source_path: Path,
    blocked_mutations: set[str] | None = None,
    mutation_attempts: dict[str, int] | None = None,
    target_config: dict[str, Any] | None = None,
    preferred_mutations: list[str] | None = None,
    mutation_memory: dict[str, Any] | None = None,
    target_name: str = "",
) -> tuple[str, str, bool]:
    path = str(source_path).replace("\\", "/").lower()
    if not path.endswith("config/track_b_attack_config.json"):
        return source, "json_no_change", False

    try:
        payload = json.loads(source)
    except json.JSONDecodeError:
        return source, "json_trackb_parse_failed", False
    if not isinstance(payload, dict):
        return source, "json_trackb_invalid_root", False

    def clone_obj() -> dict[str, Any]:
        return json.loads(json.dumps(payload))

    def target_profile_name(obj: dict[str, Any]) -> str:
        benchmark_command = None
        if isinstance(target_config, dict):
            benchmark_command = target_config.get("benchmark_command")
            if isinstance(benchmark_command, list):
                for idx, token in enumerate(benchmark_command[:-1]):
                    if token == "--profile":
                        candidate = benchmark_command[idx + 1]
                        if isinstance(candidate, str) and candidate.strip():
                            return candidate.strip()
        active_profile = obj.get("active_profile")
        if isinstance(active_profile, str) and active_profile.strip():
            return active_profile.strip()
        return ""

    def resolve_section(obj: dict[str, Any], section_name: str) -> dict[str, Any] | None:
        profile_name = target_profile_name(obj)
        if profile_name:
            profiles = obj.get("challenge_profiles")
            if isinstance(profiles, dict):
                profile_payload = profiles.get(profile_name)
                if isinstance(profile_payload, dict):
                    section = profile_payload.get(section_name)
                    if isinstance(section, dict):
                        return section
        section = obj.get(section_name)
        if isinstance(section, dict):
            return section
        return None

    def apply_delta(
        obj: dict[str, Any],
        *,
        section_name: str,
        key: str,
        delta: int,
        lo: int,
        hi: int,
    ) -> bool:
        section = resolve_section(obj, section_name)
        if section is None:
            return False
        try:
            current = int(section.get(key))
        except (TypeError, ValueError):
            return False
        updated = max(lo, min(hi, current + delta))
        if updated == current:
            return False
        section[key] = updated
        return True

    def total_rounds(obj: dict[str, Any]) -> int:
        poseidon_cfg = resolve_section(obj, "poseidon2")
        if poseidon_cfg is None:
            return 21
        full_rounds = max(2, int(poseidon_cfg.get("full_rounds", 8)))
        if full_rounds % 2 != 0:
            full_rounds += 1
        partial_rounds = max(1, int(poseidon_cfg.get("partial_rounds", 13)))
        return full_rounds + partial_rounds

    def apply_split_round(obj: dict[str, Any], delta: int) -> bool:
        analysis = resolve_section(obj, "analysis")
        if analysis is None:
            return False
        try:
            current = int(analysis.get("split_round"))
        except (TypeError, ValueError):
            return False
        rounds = total_rounds(obj)
        updated = max(1, min(max(2, rounds - 1), current + delta))
        if updated == current:
            return False
        analysis["split_round"] = updated
        return True

    def apply_algebraic_rounds(obj: dict[str, Any], delta: int) -> bool:
        analysis = resolve_section(obj, "analysis")
        if analysis is None:
            return False
        try:
            current = int(analysis.get("algebraic_rounds"))
        except (TypeError, ValueError):
            return False
        rounds = total_rounds(obj)
        updated = max(1, min(rounds, current + delta))
        if updated == current:
            return False
        analysis["algebraic_rounds"] = updated
        return True

    def op_search(key: str, delta: int, lo: int, hi: int) -> Any:
        return lambda obj: apply_delta(
            obj,
            section_name="search",
            key=key,
            delta=delta,
            lo=lo,
            hi=hi,
        )

    def op_analysis(key: str, delta: int, lo: int, hi: int) -> Any:
        return lambda obj: apply_delta(
            obj,
            section_name="analysis",
            key=key,
            delta=delta,
            lo=lo,
            hi=hi,
        )

    operators: list[tuple[str, Any]] = [
        ("json_trackb_diff_candidates_up", op_search("differential_candidates", +8, 4, 65536)),
        ("json_trackb_diff_candidates_down", op_search("differential_candidates", -8, 4, 65536)),
        (
            "json_trackb_diff_samples_up",
            op_search("differential_samples_per_candidate", +32, 8, 1_000_000),
        ),
        (
            "json_trackb_diff_samples_down",
            op_search("differential_samples_per_candidate", -32, 8, 1_000_000),
        ),
        ("json_trackb_mitm_forward_up", op_search("mitm_forward_states", +256, 64, 1_000_000)),
        ("json_trackb_mitm_forward_down", op_search("mitm_forward_states", -256, 64, 1_000_000)),
        ("json_trackb_mitm_backward_up", op_search("mitm_backward_states", +256, 64, 1_000_000)),
        ("json_trackb_mitm_backward_down", op_search("mitm_backward_states", -256, 64, 1_000_000)),
        ("json_trackb_collision_samples_up", op_search("collision_samples", +512, 64, 1_000_000)),
        ("json_trackb_collision_samples_down", op_search("collision_samples", -512, 64, 1_000_000)),
        ("json_trackb_split_round_up", lambda obj: apply_split_round(obj, +1)),
        ("json_trackb_split_round_down", lambda obj: apply_split_round(obj, -1)),
        ("json_trackb_algebraic_rounds_up", lambda obj: apply_algebraic_rounds(obj, +1)),
        ("json_trackb_algebraic_rounds_down", lambda obj: apply_algebraic_rounds(obj, -1)),
        ("json_trackb_algebraic_degree_up", op_analysis("algebraic_degree", +1, 1, 3)),
        ("json_trackb_algebraic_degree_down", op_analysis("algebraic_degree", -1, 1, 3)),
        ("json_trackb_algebraic_output_bits_up", op_analysis("algebraic_output_bits", +1, 4, 64)),
        ("json_trackb_algebraic_output_bits_down", op_analysis("algebraic_output_bits", -1, 4, 64)),
        ("json_trackb_middle_key_bits_up", op_analysis("middle_key_bits", +1, 6, 40)),
        ("json_trackb_middle_key_bits_down", op_analysis("middle_key_bits", -1, 6, 40)),
        ("json_trackb_truncated_bits_up", op_analysis("truncated_bits", +1, 8, 40)),
        ("json_trackb_truncated_bits_down", op_analysis("truncated_bits", -1, 8, 40)),
        ("json_trackb_alg_train_samples_up", op_search("algebraic_train_samples", +32, 16, 1_000_000)),
        ("json_trackb_alg_train_samples_down", op_search("algebraic_train_samples", -32, 16, 1_000_000)),
        ("json_trackb_alg_val_samples_up", op_search("algebraic_validation_samples", +16, 8, 1_000_000)),
        ("json_trackb_alg_val_samples_down", op_search("algebraic_validation_samples", -16, 8, 1_000_000)),
    ]

    shift = (iteration - 1) % len(operators)
    ordered = operators[shift:] + operators[:shift]
    preferred_rank = {name: idx for idx, name in enumerate(preferred_mutations or [])}

    def preference_for_label(label: str) -> tuple[int, int]:
        if label in preferred_rank:
            return (0, preferred_rank[label])
        if "+" in label:
            parts = [part.strip() for part in label.split("+") if part.strip()]
            part_ranks = [preferred_rank[part] for part in parts if part in preferred_rank]
            if part_ranks:
                return (1, min(part_ranks))
        return (2, 1_000_000)

    attempts = mutation_attempts or {}
    mutation_schedule = str((target_config or {}).get("mutation_schedule", "priority")).strip().lower()
    use_ucb_schedule = mutation_schedule == "ucb"
    mutation_ucb_explore = float((target_config or {}).get("mutation_ucb_explore", 0.75))
    scope_totals = mutation_memory_scope_totals(
        mutation_memory,
        target_name=target_name,
        language="json",
    )

    candidates: list[dict[str, Any]] = []
    for idx, (label, op) in enumerate(ordered):
        if blocked_mutations and label in blocked_mutations:
            continue
        candidate_obj = clone_obj()
        changed = bool(op(candidate_obj))
        if not changed:
            continue
        candidate_source = json_dump_stable(candidate_obj)
        if candidate_source == source:
            continue
        pref_class, pref_rank_value = preference_for_label(label)
        candidate_entry: dict[str, Any] = {
            "candidate": candidate_source,
            "mutation": label,
            "pref_class": pref_class,
            "pref_rank": pref_rank_value,
            "attempts": attempts.get(label, 0),
            "index": idx,
        }
        if use_ucb_schedule:
            accepted_hist, rejected_hist, history_scope = mutation_memory_counts(
                mutation_memory,
                label,
                target_name=target_name,
                language="json",
            )
            total_memory_observations = float(
                scope_totals.get(history_scope, scope_totals.get("global", 0.0))
            )
            candidate_entry["ucb_score"] = mutation_ucb_score(
                accepted=accepted_hist,
                rejected=rejected_hist,
                total_observations=total_memory_observations,
                explore=mutation_ucb_explore,
                preference_class=int(pref_class),
                schedule_tier=0,
                attempt_count=int(candidate_entry["attempts"]),
            )
        candidates.append(candidate_entry)

    if candidates:
        if use_ucb_schedule:
            candidates.sort(
                key=lambda item: (
                    -float(item.get("ucb_score", 0.0)),
                    int(item["pref_class"]),
                    int(item["pref_rank"]),
                    int(item["attempts"]),
                    int(item["index"]),
                )
            )
        else:
            candidates.sort(
                key=lambda item: (
                    int(item["pref_class"]),
                    int(item["pref_rank"]),
                    int(item["attempts"]),
                    int(item["index"]),
                )
            )
        chosen = candidates[0]
        return str(chosen["candidate"]), str(chosen["mutation"]), True

    return source, "json_trackb_no_change", False


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
    preferred_mutations: list[str] | None = None,
    mutation_memory: dict[str, Any] | None = None,
    target_name: str = "",
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
    if path.endswith("crates/backend/koala-bear/src/monty_31/x86_64_avx2/poseidon2.rs"):
        operators.extend(
            [
                rust_mutator_x86_internal_for_loop,
                rust_mutator_x86_internal_for_loop_w24,
                rust_mutator_x86_internal_for_loop_both,
                rust_mutator_avx2_add_sum_loops,
                rust_mutator_avx2_add_sum_as_mut_hoist,
                rust_mutator_avx2_sum_vec_hoist,
                rust_mutator_avx2_sum_vec_hoist_w24,
                rust_mutator_avx2_sum_vec_hoist_both,
            ]
        )
    if path.endswith("crates/backend/koala-bear/src/monty_31/x86_64_avx512/poseidon2.rs"):
        operators.extend(
            [
                rust_mutator_x86_internal_for_loop,
                rust_mutator_x86_internal_for_loop_w24,
                rust_mutator_x86_internal_for_loop_both,
                rust_mutator_avx512_add_sum_loops,
                rust_mutator_avx512_add_sum_as_mut_hoist,
                rust_mutator_avx512_sum_vec_hoist,
                rust_mutator_avx512_sum_vec_hoist_w24,
                rust_mutator_avx512_sum_vec_hoist_both,
            ]
        )
    if path.endswith("crates/backend/koala-bear/src/monty_31/no_packing/poseidon2.rs"):
        operators.extend(
            [
                rust_mutator_no_packing_internal_inline,
                rust_mutator_no_packing_external_inline,
            ]
        )
    if path.endswith("crates/backend/koala-bear/src/monty_31/poseidon2_monty.rs"):
        operators.extend(
            [
                rust_mutator_monty_internal_for_loop,
                rust_mutator_monty_internal_for_copied,
                rust_mutator_monty_s0_cache_generic,
                rust_mutator_monty_s0_cache_internal,
            ]
        )

    operators.extend(
        [
            rust_mutator_const_collects,
            rust_mutator_hoist_num_cols,
            rust_mutator_cache_trace_refs,
            rust_mutator_hoist_packed_cols,
            rust_mutator_pack_point_buffer,
            rust_mutator_pack_point_buffer_verify,
            rust_mutator_pack_point_buffer_both,
            rust_mutator_hoist_log_num_cols,
        ]
    )
    if not operators:
        return source, "rust_no_change", False

    shift = (iteration - 1) % len(operators)
    ordered = operators[shift:] + operators[:shift]
    preferred_rank = {name: idx for idx, name in enumerate(preferred_mutations or [])}

    def preference_for_label(label: str) -> tuple[int, int]:
        if label in preferred_rank:
            return (0, preferred_rank[label])
        if "+" in label:
            parts = [part.strip() for part in label.split("+") if part.strip()]
            part_ranks = [preferred_rank[part] for part in parts if part in preferred_rank]
            if part_ranks:
                return (1, min(part_ranks))
        return (2, 1_000_000)

    candidates: list[dict[str, Any]] = []
    single_candidates: list[tuple[int, str, str]] = []
    attempts = mutation_attempts or {}
    mutation_schedule = str((target_config or {}).get("mutation_schedule", "priority")).strip().lower()
    use_ucb_schedule = mutation_schedule == "ucb"
    mutation_ucb_explore = float((target_config or {}).get("mutation_ucb_explore", 0.75))
    scope_totals = mutation_memory_scope_totals(
        mutation_memory,
        target_name=target_name,
        language="rust",
    )

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
        pref_class, pref_rank_value = preference_for_label(mutation)
        candidates.append(
            {
                "tier": single_tier,
                "pref_class": pref_class,
                "pref_rank": pref_rank_value,
                "attempts": attempts.get(mutation, 0),
                "index": idx,
                "candidate": candidate,
                "mutation": mutation,
            }
        )

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
                pref_class, pref_rank_value = preference_for_label(combo_mutation)
                candidates.append(
                    {
                        "tier": compound_tier,
                        "pref_class": pref_class,
                        "pref_rank": pref_rank_value,
                        "attempts": combo_attempts,
                        "index": combo_idx,
                        "candidate": candidate_b,
                        "mutation": combo_mutation,
                    }
                )
                added += 1
                if added >= compound_limit:
                    break
            if added >= compound_limit:
                break

    if candidates:
        if use_ucb_schedule:
            for item in candidates:
                accepted_hist, rejected_hist, history_scope = mutation_memory_counts(
                    mutation_memory,
                    str(item["mutation"]),
                    target_name=target_name,
                    language="rust",
                )
                total_memory_observations = float(
                    scope_totals.get(history_scope, scope_totals.get("global", 0.0))
                )
                item["ucb_score"] = mutation_ucb_score(
                    accepted=accepted_hist,
                    rejected=rejected_hist,
                    total_observations=total_memory_observations,
                    explore=mutation_ucb_explore,
                    preference_class=int(item["pref_class"]),
                    schedule_tier=int(item["tier"]),
                    attempt_count=int(item["attempts"]),
                )
            candidates.sort(
                key=lambda item: (
                    -float(item.get("ucb_score", 0.0)),
                    int(item["pref_class"]),
                    int(item["pref_rank"]),
                    int(item["attempts"]),
                    int(item["index"]),
                )
            )
        else:
            candidates.sort(
                key=lambda item: (
                    int(item["tier"]),
                    int(item["pref_class"]),
                    int(item["pref_rank"]),
                    int(item["attempts"]),
                    int(item["index"]),
                )
            )
        chosen = candidates[0]
        return str(chosen["candidate"]), str(chosen["mutation"]), True
    return source, "rust_no_change", False


def heuristic_candidate(
    source: str,
    iteration: int,
    language: str,
    source_path: Path,
    blocked_mutations: set[str] | None = None,
    mutation_attempts: dict[str, int] | None = None,
    target_config: dict[str, Any] | None = None,
    preferred_mutations: list[str] | None = None,
    mutation_memory: dict[str, Any] | None = None,
    target_name: str = "",
) -> tuple[str, str, bool]:
    if language.lower() == "json":
        json_candidate, mutation, changed = json_heuristic_candidate(
            source,
            iteration,
            source_path,
            blocked_mutations=blocked_mutations,
            mutation_attempts=mutation_attempts,
            target_config=target_config,
            preferred_mutations=preferred_mutations,
            mutation_memory=mutation_memory,
            target_name=target_name,
        )
        if changed:
            return json_candidate, mutation, True
    if language.lower() == "rust":
        rust_candidate, mutation, changed = rust_heuristic_candidate(
            source,
            iteration,
            source_path,
            blocked_mutations=blocked_mutations,
            mutation_attempts=mutation_attempts,
            target_config=target_config,
            preferred_mutations=preferred_mutations,
            mutation_memory=mutation_memory,
            target_name=target_name,
        )
        if changed:
            return rust_candidate, mutation, True
    if language.lower() == "python":
        python_candidate, mutation, changed = python_heuristic_candidate(
            source,
            iteration,
            source_path,
            blocked_mutations=blocked_mutations,
            mutation_attempts=mutation_attempts,
            target_config=target_config,
            preferred_mutations=preferred_mutations,
            mutation_memory=mutation_memory,
            target_name=target_name,
        )
        if changed:
            return python_candidate, mutation, True
    return generic_heuristic_candidate(source, iteration)


def release_oldest_blocked_mutation(
    blocked_mutations_until: dict[str, int],
    active_blocked: set[str],
) -> str | None:
    if not active_blocked:
        return None
    oldest = min(active_blocked, key=lambda label: (blocked_mutations_until.get(label, 0), label))
    blocked_mutations_until.pop(oldest, None)
    return oldest


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


def git_checkpoint_commit(
    *,
    source_path: Path,
    target_name: str,
    iteration: int,
    mutation: str,
    best_metric: float,
    metric_value: float | None,
    prefix: str,
) -> dict[str, Any]:
    source_path = source_path.resolve()
    top = run_capture(["git", "-C", str(source_path.parent), "rev-parse", "--show-toplevel"])
    if not top:
        return {"status": "skipped", "reason": "source_not_in_git_repo"}
    root = Path(top.strip()).resolve()

    try:
        rel_path = source_path.relative_to(root)
    except ValueError:
        return {"status": "skipped", "reason": "source_outside_git_root", "git_root": str(root)}

    rel = str(rel_path)
    status = run_capture(["git", "-C", str(root), "status", "--porcelain", "--", rel]) or ""
    if not status.strip():
        return {"status": "skipped", "reason": "no_changes", "git_root": str(root), "path": rel}

    metric_text = "n/a" if metric_value is None else f"{float(metric_value):.6f}"
    commit_msg = (
        f"{prefix}: target={target_name} iter={iteration} mutation={mutation} "
        f"metric={metric_text} best={best_metric:.6f}"
    )
    proc = subprocess.run(
        ["git", "-C", str(root), "commit", "--only", "-m", commit_msg, "--", rel],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        return {
            "status": "failed",
            "reason": "commit_failed",
            "git_root": str(root),
            "path": rel,
            "message": summarize_notes(err, max_len=600),
        }

    head = run_capture(["git", "-C", str(root), "rev-parse", "HEAD"])
    branch = run_capture(["git", "-C", str(root), "rev-parse", "--abbrev-ref", "HEAD"])
    return {
        "status": "committed",
        "git_root": str(root),
        "path": rel,
        "commit": head.strip() if head else None,
        "branch": branch.strip() if branch else None,
        "message": commit_msg,
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


def normalize_mutation_label(token: str) -> str | None:
    token = token.strip()
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
    if token in {"no_change", "n/a", "rust_no_change", "python_no_change", "heuristic_no_change"}:
        return None
    return token


def is_retryable_no_change_mutation(mutation: str) -> bool:
    return mutation in {"json_trackb_no_change", "rust_no_change", "python_no_change", "heuristic_no_change"}


def extract_mutation_label_from_notes(notes: str) -> str | None:
    if not notes:
        return None
    token = notes.split(";", 1)[0].strip()
    if not token:
        return None
    return normalize_mutation_label(token)


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


def infer_mutation_language(*, mutation: str, target_language: str | None = None) -> str:
    candidate = (target_language or "").strip().lower()
    if candidate:
        return candidate
    if mutation.startswith("rust_"):
        return "rust"
    if mutation.startswith("json_"):
        return "json"
    if mutation.startswith("python_"):
        return "python"
    if mutation.startswith("heuristic_"):
        return "generic"
    return "unknown"


def parse_iso_to_epoch(value: str | None) -> float:
    if not value:
        return 0.0
    try:
        return dt.datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        try:
            dir_fd = os.open(str(path.parent), os.O_RDONLY)
        except OSError:
            dir_fd = None
        if dir_fd is not None:
            try:
                os.fsync(dir_fd)
            except OSError:
                # Directory fsync is best-effort for durability and can fail on some filesystems.
                pass
            finally:
                os.close(dir_fd)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


@contextmanager
def file_lock(lock_path: Path) -> Any:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = lock_path.open("a+", encoding="utf-8")
    try:
        if os.name == "nt":
            import msvcrt

            lock_file.seek(0, os.SEEK_END)
            if lock_file.tell() == 0:
                lock_file.write("\0")
                lock_file.flush()
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            return

        import fcntl

        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    finally:
        lock_file.close()


def load_mutation_memory(path: Path) -> dict[str, Any]:
    base: dict[str, Any] = {
        "version": 1,
        "updated_at": None,
        "seeded_from_results": False,
        "seeded_from_results_version": 0,
        "mutations": {},
    }
    if path.exists():
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return base
        if isinstance(payload, dict):
            payload.setdefault("version", 1)
            payload.setdefault("updated_at", None)
            payload.setdefault("seeded_from_results", False)
            payload.setdefault("seeded_from_results_version", 0)
            payload.setdefault("mutations", {})
            if isinstance(payload.get("mutations"), dict):
                return payload
    return base


def seed_mutation_memory_from_results(memory: dict[str, Any]) -> None:
    seed_version = int(memory.get("seeded_from_results_version", 0) or 0)
    if bool(memory.get("seeded_from_results")) and seed_version >= DEFAULT_MUTATION_MEMORY_SEED_VERSION:
        return

    # Rebuild from results.tsv whenever seed logic changes to avoid stale language coverage.
    memory["mutations"] = {}
    mutations = memory["mutations"]

    targets = prepare.load_targets()
    target_language_map: dict[str, str] = {}
    for target_name, cfg in targets.items():
        language = str(cfg.get("language", "")).strip().lower()
        if language:
            target_language_map[target_name] = language
            continue
        source_file = str(cfg.get("source_file", ""))
        if source_file.endswith(".rs"):
            target_language_map[target_name] = "rust"

    results_file = ROOT / "results.tsv"
    if not results_file.exists():
        memory["seeded_from_results"] = True
        memory["seeded_from_results_version"] = DEFAULT_MUTATION_MEMORY_SEED_VERSION
        return
    rows = results_file.read_text().splitlines()
    if len(rows) <= 1:
        memory["seeded_from_results"] = True
        memory["seeded_from_results_version"] = DEFAULT_MUTATION_MEMORY_SEED_VERSION
        return

    for row in rows[1:]:
        cols = row.split("\t")
        if len(cols) < 12:
            continue
        target_name = cols[1]
        try:
            iteration = int(cols[2])
        except Exception:  # noqa: BLE001
            iteration = 0
        if iteration <= 0:
            continue
        mutation = extract_mutation_label_from_notes(cols[11])
        if not mutation:
            continue
        status_token = cols[3].strip().lower()
        notes_token = cols[11].strip().lower()

        # Acceptance is encoded in notes (`accepted:` / `rejected_*:`), not in status.
        if notes_token.startswith("accepted:"):
            accepted = True
        elif notes_token.startswith("rejected_") or notes_token.startswith("rejected:"):
            accepted = False
        elif status_token == "accepted":
            accepted = True
        elif status_token == "rejected":
            accepted = False
        elif status_token in {"success", "failed", "skipped", "timeout", "error", "no_change", ""}:
            # Evaluation lifecycle statuses are not acceptance signals.
            continue
        else:
            # Skip unrecognized statuses instead of silently treating as rejection.
            continue
        update_mutation_memory(
            memory,
            mutation=mutation,
            accepted=accepted,
            target_name=target_name,
            language=infer_mutation_language(mutation=mutation, target_language=target_language_map.get(target_name)),
            timestamp=cols[0],
            metric_before=None,
            metric_after=None,
            compact=False,
        )
    compact_mutation_memory(mutations, now_epoch=time.time())
    memory["seeded_from_results"] = True
    memory["seeded_from_results_version"] = DEFAULT_MUTATION_MEMORY_SEED_VERSION


def save_mutation_memory(path: Path, memory: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    memory["updated_at"] = prepare.now_iso()
    atomic_write_text(path, json.dumps(memory, indent=2, sort_keys=True) + "\n")


def _increment_stat(container: dict[str, Any], key: str, accepted: bool) -> None:
    entry = container.setdefault(key, {"accepted": 0, "rejected": 0})
    if not isinstance(entry, dict):
        return
    field = "accepted" if accepted else "rejected"
    entry[field] = int(entry.get(field, 0)) + 1


def compact_mutation_memory(
    mutations: dict[str, Any],
    *,
    now_epoch: float,
    stale_seconds: int = DEFAULT_MUTATION_MEMORY_STALE_SECONDS,
    max_entries: int = DEFAULT_MUTATION_MEMORY_MAX_ENTRIES,
    accepted_stale_multiplier: float = DEFAULT_MUTATION_MEMORY_ACCEPTED_STALE_MULTIPLIER,
) -> None:
    if stale_seconds > 0:
        cutoff = now_epoch - float(stale_seconds)
        accepted_cutoff: float | None = None
        if accepted_stale_multiplier > 0:
            accepted_cutoff = now_epoch - (float(stale_seconds) * float(accepted_stale_multiplier))
        stale_keys: list[str] = []
        for mutation, raw in mutations.items():
            if not isinstance(raw, dict):
                stale_keys.append(mutation)
                continue
            accepted_total = int(raw.get("accepted_total", 0))
            last_seen_epoch = parse_iso_to_epoch(raw.get("last_seen_at"))
            if accepted_total > 0:
                if accepted_cutoff is not None and last_seen_epoch and last_seen_epoch < accepted_cutoff:
                    stale_keys.append(mutation)
                continue
            if last_seen_epoch and last_seen_epoch < cutoff:
                stale_keys.append(mutation)
        for key in stale_keys:
            mutations.pop(key, None)

    if max_entries <= 0 or len(mutations) <= max_entries:
        return

    ranked: list[tuple[int, int, float, float, str]] = []
    for mutation, raw in mutations.items():
        if not isinstance(raw, dict):
            ranked.append((0, 0, 0.0, 0.0, mutation))
            continue
        accepted_total = int(raw.get("accepted_total", 0))
        rejected_total = int(raw.get("rejected_total", 0))
        last_seen = parse_iso_to_epoch(raw.get("last_seen_at"))
        latest_gain = float(raw.get("latest_gain", 0.0) or 0.0)
        ranked.append((1 if accepted_total > 0 else 0, accepted_total - rejected_total, latest_gain, last_seen, mutation))

    ranked.sort(reverse=True)
    keep = {mutation for _, _, _, _, mutation in ranked[:max_entries]}
    for mutation in list(mutations.keys()):
        if mutation not in keep:
            mutations.pop(mutation, None)


def update_mutation_memory(
    memory: dict[str, Any],
    *,
    mutation: str,
    accepted: bool,
    target_name: str,
    language: str,
    timestamp: str,
    metric_before: float | None,
    metric_after: float | None,
    compact: bool = True,
) -> None:
    mutations = memory.setdefault("mutations", {})
    if not isinstance(mutations, dict):
        return

    entry = mutations.setdefault(
        mutation,
        {
            "accepted_total": 0,
            "rejected_total": 0,
            "last_seen_at": None,
            "last_accepted_at": None,
            "last_rejected_at": None,
            "languages": {},
            "targets": {},
            "latest_gain": None,
        },
    )
    if not isinstance(entry, dict):
        return

    if accepted:
        entry["accepted_total"] = int(entry.get("accepted_total", 0)) + 1
        entry["last_accepted_at"] = timestamp
    else:
        entry["rejected_total"] = int(entry.get("rejected_total", 0)) + 1
        entry["last_rejected_at"] = timestamp
    entry["last_seen_at"] = timestamp

    languages = entry.setdefault("languages", {})
    if isinstance(languages, dict):
        _increment_stat(languages, language, accepted)

    targets = entry.setdefault("targets", {})
    if isinstance(targets, dict):
        _increment_stat(targets, target_name, accepted)

    if metric_before is not None and metric_after is not None:
        entry["latest_gain"] = metric_after - metric_before

    if compact:
        compact_mutation_memory(mutations, now_epoch=parse_iso_to_epoch(timestamp) or time.time())


def update_and_save_mutation_memory(
    path: Path,
    current_memory: dict[str, Any],
    *,
    mutation: str,
    accepted: bool,
    target_name: str,
    language: str,
    timestamp: str,
    metric_before: float | None,
    metric_after: float | None,
) -> dict[str, Any]:
    lock_path = path.with_suffix(path.suffix + ".lock")
    with file_lock(lock_path):
        latest = load_mutation_memory(path)
        update_mutation_memory(
            latest,
            mutation=mutation,
            accepted=accepted,
            target_name=target_name,
            language=language,
            timestamp=timestamp,
            metric_before=metric_before,
            metric_after=metric_after,
        )
        save_mutation_memory(path, latest)

    current_memory.clear()
    current_memory.update(latest)
    return current_memory


def preferred_mutations_from_memory(
    memory: dict[str, Any],
    *,
    target_name: str,
    language: str,
    limit: int = 8,
    min_accepted_total: int = DEFAULT_MUTATION_MEMORY_MIN_ACCEPTED_TOTAL,
    min_success_rate: float = DEFAULT_MUTATION_MEMORY_MIN_SUCCESS_RATE,
) -> list[str]:
    mutations = memory.get("mutations")
    if not isinstance(mutations, dict):
        return []

    ranked: list[tuple[int, float, int, str]] = []
    for mutation, raw in mutations.items():
        if not isinstance(raw, dict):
            continue

        accepted_total = int(raw.get("accepted_total", 0))
        rejected_total = int(raw.get("rejected_total", 0))
        if accepted_total <= 0:
            continue

        language_ok = False
        raw_lang = raw.get("languages")
        if isinstance(raw_lang, dict):
            lang_entry = raw_lang.get(language)
            if isinstance(lang_entry, dict) and int(lang_entry.get("accepted", 0)) > 0:
                language_ok = True
        if not language_ok and language == "rust" and mutation.startswith("rust_"):
            language_ok = True
        if not language_ok and language == "json" and mutation.startswith("json_"):
            language_ok = True
        if not language_ok and language == "python" and mutation.startswith("python_"):
            language_ok = True
        if not language_ok:
            continue

        targets = raw.get("targets")
        has_current_target_success = False
        if isinstance(targets, dict):
            current = targets.get(target_name)
            if isinstance(current, dict) and int(current.get("accepted", 0)) > 0:
                has_current_target_success = True

        cross_target_tier = 0 if has_current_target_success else 1
        total = accepted_total + max(0, rejected_total)
        success_rate = accepted_total / total if total > 0 else 0.0
        if accepted_total < max(0, min_accepted_total) and success_rate < max(0.0, min_success_rate):
            continue
        ranked.append((cross_target_tier, -success_rate, -accepted_total, mutation))

    ranked.sort()
    return [mutation for _, _, _, mutation in ranked[:limit]]


def mutation_memory_scope_totals(
    memory: dict[str, Any] | None,
    *,
    target_name: str,
    language: str,
) -> dict[str, float]:
    totals = {"target": 0.0, "language": 0.0, "global": 0.0}
    if not isinstance(memory, dict):
        return totals
    raw_mutations = memory.get("mutations")
    if not isinstance(raw_mutations, dict):
        return totals

    for raw in raw_mutations.values():
        if not isinstance(raw, dict):
            continue
        accepted_total = max(0.0, float(raw.get("accepted_total", 0) or 0.0))
        rejected_total = max(0.0, float(raw.get("rejected_total", 0) or 0.0))
        totals["global"] += accepted_total + rejected_total

        raw_languages = raw.get("languages")
        if isinstance(raw_languages, dict):
            language_counts = raw_languages.get(language)
            if isinstance(language_counts, dict):
                language_accepted = max(0.0, float(language_counts.get("accepted", 0) or 0.0))
                language_rejected = max(0.0, float(language_counts.get("rejected", 0) or 0.0))
                totals["language"] += language_accepted + language_rejected

        raw_targets = raw.get("targets")
        if isinstance(raw_targets, dict):
            target_counts = raw_targets.get(target_name)
            if isinstance(target_counts, dict):
                target_accepted = max(0.0, float(target_counts.get("accepted", 0) or 0.0))
                target_rejected = max(0.0, float(target_counts.get("rejected", 0) or 0.0))
                totals["target"] += target_accepted + target_rejected

    if totals["language"] <= 0.0:
        totals["language"] = totals["global"]
    if totals["target"] <= 0.0:
        totals["target"] = totals["language"]
    return totals


def mutation_memory_counts(
    memory: dict[str, Any] | None,
    mutation: str,
    *,
    target_name: str,
    language: str,
) -> tuple[float, float, str]:
    if not isinstance(memory, dict):
        return (0.0, 0.0, "global")
    raw_mutations = memory.get("mutations")
    if not isinstance(raw_mutations, dict):
        return (0.0, 0.0, "global")

    def entry_counts(label: str) -> tuple[float, float, str] | None:
        raw = raw_mutations.get(label)
        if not isinstance(raw, dict):
            return None

        accepted_total = max(0.0, float(raw.get("accepted_total", 0) or 0.0))
        rejected_total = max(0.0, float(raw.get("rejected_total", 0) or 0.0))

        language_counts = None
        raw_languages = raw.get("languages")
        if isinstance(raw_languages, dict):
            language_counts = raw_languages.get(language)
        target_counts = None
        raw_targets = raw.get("targets")
        if isinstance(raw_targets, dict):
            target_counts = raw_targets.get(target_name)

        if isinstance(target_counts, dict):
            target_accepted = max(0.0, float(target_counts.get("accepted", 0) or 0.0))
            target_rejected = max(0.0, float(target_counts.get("rejected", 0) or 0.0))
            if target_accepted + target_rejected > 0.0:
                return (target_accepted, target_rejected, "target")
        if isinstance(language_counts, dict):
            language_accepted = max(0.0, float(language_counts.get("accepted", 0) or 0.0))
            language_rejected = max(0.0, float(language_counts.get("rejected", 0) or 0.0))
            if language_accepted + language_rejected > 0.0:
                return (language_accepted, language_rejected, "language")
        return (accepted_total, rejected_total, "global")

    direct = entry_counts(mutation)
    if direct is not None:
        return direct

    if "+" in mutation:
        parts = [part.strip() for part in mutation.split("+") if part.strip()]
        aggregates = [entry_counts(part) for part in parts]
        # Treat partially-known compounds as unknown to preserve exploration.
        if not aggregates or any(item is None for item in aggregates):
            return (0.0, 0.0, "global")

        accepted = sum(item[0] for item in aggregates if item is not None) / len(aggregates)
        rejected = sum(item[1] for item in aggregates if item is not None) / len(aggregates)
        scope_rank = {"target": 0, "language": 1, "global": 2}
        scope = max(
            (item[2] for item in aggregates if item is not None),
            key=lambda value: scope_rank.get(value, 2),
            default="global",
        )
        return (accepted, rejected, scope)

    return (0.0, 0.0, "global")


def mutation_ucb_score(
    *,
    accepted: float,
    rejected: float,
    total_observations: float,
    explore: float,
    preference_class: int,
    schedule_tier: int,
    attempt_count: int,
) -> float:
    observations = max(0.0, accepted + rejected)
    posterior_mean = (accepted + 1.0) / (observations + 2.0)
    bonus = max(0.0, explore) * math.sqrt(math.log(max(2.0, total_observations + 2.0)) / (observations + 1.0))
    preference_bonus = 0.0
    if preference_class == 0:
        preference_bonus = 0.05
    elif preference_class == 1:
        preference_bonus = 0.025
    tier_bonus = 0.02 if schedule_tier == 0 else 0.0
    attempt_penalty = min(0.25, 0.01 * max(0, attempt_count))
    return posterior_mean + bonus + preference_bonus + tier_bonus - attempt_penalty


def required_snippets_from_target(target_config: dict[str, Any]) -> list[str]:
    raw = target_config.get("required_snippets")
    if not isinstance(raw, list):
        return []
    seen: set[str] = set()
    out: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        token = item.strip()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def strip_rust_comments_and_literals(source: str) -> str:
    """Remove Rust comments and string/char literals for guardrail substring checks."""
    out: list[str] = []
    n = len(source)
    i = 0
    state = "code"
    block_depth = 0
    string_quote = ""
    raw_hashes = 0
    escape = False

    while i < n:
        ch = source[i]
        nxt = source[i + 1] if i + 1 < n else ""

        if state == "code":
            if ch == "/" and nxt == "/":
                out.extend([" ", " "])
                i += 2
                state = "line_comment"
                continue
            if ch == "/" and nxt == "*":
                out.extend([" ", " "])
                i += 2
                state = "block_comment"
                block_depth = 1
                continue
            if ch == "'" and i + 1 < n and (source[i + 1].isalpha() or source[i + 1] == "_"):
                # Preserve Rust lifetimes/labels (e.g. 'a, 'static, 'label:) as code.
                j = i + 1
                while j < n and (source[j].isalnum() or source[j] == "_"):
                    j += 1
                if not (j < n and source[j] == "'"):
                    out.append(ch)
                    i += 1
                    continue

            if ch in {"'", '"'}:
                out.append(ch)
                i += 1
                state = "string_or_char"
                string_quote = ch
                escape = False
                continue

            if ch == "r":
                j = i + 1
                while j < n and source[j] == "#":
                    j += 1
                if j < n and source[j] == '"':
                    hash_count = j - i - 1
                    out.extend(" " * (j - i + 1))
                    i = j + 1
                    state = "raw_string"
                    raw_hashes = hash_count
                    continue

            out.append(ch)
            i += 1
            continue

        if state == "line_comment":
            if ch == "\n":
                out.append("\n")
                state = "code"
            else:
                out.append(" ")
            i += 1
            continue

        if state == "block_comment":
            if ch == "/" and nxt == "*":
                out.extend([" ", " "])
                i += 2
                block_depth += 1
                continue
            if ch == "*" and nxt == "/":
                out.extend([" ", " "])
                i += 2
                block_depth -= 1
                if block_depth <= 0:
                    state = "code"
                continue
            out.append("\n" if ch == "\n" else " ")
            i += 1
            continue

        if state == "string_or_char":
            if escape:
                out.append(" " if ch != "\n" else "\n")
                escape = False
                i += 1
                continue
            if ch == "\\":
                out.append(" ")
                escape = True
                i += 1
                continue
            if ch == string_quote:
                out.append(ch)
                i += 1
                state = "code"
                continue
            out.append("\n" if ch == "\n" else " ")
            i += 1
            continue

        if state == "raw_string":
            if ch == '"':
                hashes = source[i + 1 : i + 1 + raw_hashes]
                if len(hashes) == raw_hashes and all(token == "#" for token in hashes):
                    out.append(" ")
                    out.extend(" " * raw_hashes)
                    i += 1 + raw_hashes
                    state = "code"
                    continue
            out.append("\n" if ch == "\n" else " ")
            i += 1
            continue

    return "".join(out)


def normalized_source_for_required_snippets(source: str, *, language: str) -> str:
    if language.lower() == "python":
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return ""
        defs = [
            f"def {node.name}("
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        return "\n".join(defs)
    if language.lower() == "rust":
        return strip_rust_comments_and_literals(source)
    return source


def required_snippet_id(snippet: str) -> str:
    return hashlib.sha256(snippet.encode("utf-8")).hexdigest()[:12]


def required_snippet_profile(source: str, snippets: list[str], *, language: str) -> dict[str, int]:
    normalized = normalized_source_for_required_snippets(source, language=language)
    profile: dict[str, int] = {}
    for snippet in snippets:
        profile[snippet] = normalized.count(snippet)
    return profile


def required_snippet_guard(candidate: str, profile: dict[str, int], *, language: str) -> tuple[bool, dict[str, Any]]:
    if not profile:
        return True, {"required_snippets": 0, "violations": []}

    normalized = normalized_source_for_required_snippets(candidate, language=language)
    violations: list[dict[str, Any]] = []
    for snippet, required in profile.items():
        if required <= 0:
            continue
        observed = normalized.count(snippet)
        if observed >= required:
            continue
        violations.append(
            {
                "id": required_snippet_id(snippet),
                "required": required,
                "observed": observed,
            }
        )

    return len(violations) == 0, {"required_snippets": len(profile), "violations": violations}


def parse_flag_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        if not token:
            return default
        if token in {"1", "true", "yes", "on"}:
            return True
        if token in {"0", "false", "no", "off"}:
            return False
        try:
            return int(token) != 0
        except ValueError:
            return default
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    return default
def resolved_objective_from_trackb_payload(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    top_level = payload.get("objective")
    resolved = top_level if isinstance(top_level, dict) else None
    source_key = "objective" if isinstance(top_level, dict) else None

    active_profile = payload.get("active_profile")
    profiles = payload.get("challenge_profiles")
    if isinstance(active_profile, str) and active_profile.strip() and isinstance(profiles, dict):
        profile_payload = profiles.get(active_profile.strip())
        if isinstance(profile_payload, dict):
            profile_objective = profile_payload.get("objective")
            if isinstance(profile_objective, dict):
                resolved = profile_objective
                source_key = f"challenge_profiles.{active_profile.strip()}.objective"

    if not isinstance(resolved, dict):
        return (None, None)
    return (json.loads(json.dumps(resolved)), source_key)


def resolved_objective_from_trackb_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    top_level = payload.get("objective")
    resolved = top_level if isinstance(top_level, dict) else None

    active_profile = payload.get("active_profile")
    profiles = payload.get("challenge_profiles")
    if isinstance(active_profile, str) and active_profile.strip() and isinstance(profiles, dict):
        profile_payload = profiles.get(active_profile.strip())
        if isinstance(profile_payload, dict):
            profile_objective = profile_payload.get("objective")
            if isinstance(profile_objective, dict):
                resolved = profile_objective

    if not isinstance(resolved, dict):
        return None
    return json.loads(json.dumps(resolved))


def objective_sections_from_trackb_payload(payload: dict[str, Any]) -> dict[str, Any]:
    sections: dict[str, Any] = {}
    top_level = payload.get("objective")
    if isinstance(top_level, dict):
        sections["objective"] = json.loads(json.dumps(top_level))

    profiles = payload.get("challenge_profiles")
    if isinstance(profiles, dict):
        for profile_name in sorted(profiles.keys()):
            profile_payload = profiles.get(profile_name)
            if not isinstance(profile_payload, dict):
                continue
            objective = profile_payload.get("objective")
            if isinstance(objective, dict):
                key = f"challenge_profiles.{profile_name}.objective"
                sections[key] = json.loads(json.dumps(objective))
    resolved = resolved_objective_from_trackb_payload(payload)
    if isinstance(resolved, dict) and resolved != sections.get("objective"):
        sections["resolved_objective"] = resolved
    return sections


def trackb_objective_guard(
    *,
    current_source: str,
    candidate_source: str,
    source_path: Path,
    target_config: dict[str, Any] | None,
) -> tuple[bool, dict[str, Any]]:
    path = str(source_path).replace("\\", "/").lower()
    if not path.endswith("config/track_b_attack_config.json"):
        return True, {"enabled": False}

    allow_mutation = parse_flag_bool((target_config or {}).get("json_allow_objective_mutations"), default=False)
    if allow_mutation:
        return True, {"enabled": True, "allow_objective_mutations": True}

    try:
        before_payload = json.loads(current_source)
    except json.JSONDecodeError:
        return False, {"enabled": True, "status": "baseline_parse_failed"}
    try:
        after_payload = json.loads(candidate_source)
    except json.JSONDecodeError:
        return False, {"enabled": True, "status": "candidate_parse_failed"}
    if not isinstance(before_payload, dict) or not isinstance(after_payload, dict):
        return False, {"enabled": True, "status": "invalid_root"}

    before_sections = objective_sections_from_trackb_payload(before_payload)
    after_sections = objective_sections_from_trackb_payload(after_payload)
    changed_paths: list[str] = []
    for key in sorted(set(before_sections.keys()) | set(after_sections.keys())):
        if before_sections.get(key) != after_sections.get(key):
            changed_paths.append(key)

    before_resolved, before_resolved_source = resolved_objective_from_trackb_payload(before_payload)
    after_resolved, after_resolved_source = resolved_objective_from_trackb_payload(after_payload)
    resolved_sources = {
        source_key
        for source_key in (before_resolved_source, after_resolved_source)
        if isinstance(source_key, str) and source_key
    }
    if before_resolved != after_resolved and not resolved_sources.intersection(changed_paths):
        changed_paths.append("resolved_objective")

    if changed_paths:
        return False, {"enabled": True, "status": "objective_modified", "paths": changed_paths}

    return True, {"enabled": True, "status": "ok"}


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
    configure_debug_environment(args)
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
    initial_source = source_path.read_text()

    template = load_prompt_template()
    if "language" in target:
        language = str(target["language"])
    elif target["type"] == "noir":
        language = "Noir"
    elif target["type"] == "cairo":
        language = "Cairo"
    else:
        language = "Source"
    language_norm = language.strip().lower()
    require_metric_series_for_stats = bool(target.get("require_metric_series_for_stats", False))

    git_checkpoint_mode = str(args.git_checkpoint_mode).strip().lower()
    if git_checkpoint_mode not in {"off", "accepted", "all"}:
        git_checkpoint_mode = "off"
    git_checkpoint_prefix = args.git_checkpoint_prefix.strip() or "autoresearch"

    mutation_memory_path = Path(args.mutation_memory_file)
    if not mutation_memory_path.is_absolute():
        mutation_memory_path = ROOT / mutation_memory_path
    mutation_memory: dict[str, Any] | None = None
    mutation_memory_enabled = (not args.disable_mutation_memory) and language_norm in {"rust", "json", "python"}
    if mutation_memory_enabled:
        # Persist seeded history to disk before iterations start so per-iteration
        # read/modify/write updates do not discard in-process seeded state.
        lock_path = mutation_memory_path.with_suffix(mutation_memory_path.suffix + ".lock")
        try:
            with file_lock(lock_path):
                mutation_memory = load_mutation_memory(mutation_memory_path)
                seed_mutation_memory_from_results(mutation_memory)
                save_mutation_memory(mutation_memory_path, mutation_memory)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[train] Warning: mutation memory bootstrap failed ({exc}); continuing without memory",
                file=sys.stderr,
            )
            mutation_memory = None

    max_iterations = args.iterations if args.iterations > 0 else None
    iterations_label = str(args.iterations) if max_iterations is not None else "infinite"

    train_log(
        args,
        (
            f"starting target={args.target} iterations={iterations_label} "
            f"max_accepted={args.max_accepted or 'none'} "
            f"max_runtime={args.max_runtime_seconds or 'none'}s "
            f"artifacts={args.artifacts} "
            f"git_checkpoint={git_checkpoint_mode}"
        ),
        level=1,
    )

    baseline = prepare.evaluate_target(args.target)
    if baseline["status"] != "success" or baseline["metric_value"] is None:
        print("Baseline evaluation failed; aborting optimization loop", file=sys.stderr)
        print(json.dumps(baseline, indent=2, sort_keys=True), file=sys.stderr)
        return 1

    run_label = make_run_label()
    run_env = runtime_fingerprint(target_name=args.target, target=target, source_path=source_path)
    loop_wall_start = time.perf_counter()
    stop_reason = "iterations_exhausted" if max_iterations is not None else "external_stop"

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
    train_log(
        args,
        (
            f"baseline metric={best_metric:.6f} "
            f"check_s={baseline['check_s']:.3f} info_or_bench_s={baseline['info_or_bench_s']:.3f} "
            f"execute_s={baseline['execute_s']:.3f}"
        ),
        level=1,
    )
    baseline_metric_series = parse_metric_series(baseline.get("debug", {}).get("metric_values"))
    if require_metric_series_for_stats and not baseline_metric_series:
        print(
            "Target requires metric series for statistical gates, but baseline evaluation did not provide debug.metric_values",
            file=sys.stderr,
        )
        return 1
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
    required_snippets = required_snippets_from_target(target)
    required_snippet_counts = required_snippet_profile(
        source_path.read_text(),
        required_snippets,
        language=language_norm,
    )
    missing_required_snippets = [snippet for snippet, count in required_snippet_counts.items() if count <= 0]
    if missing_required_snippets:
        missing_preview = ", ".join(required_snippet_id(snippet) for snippet in missing_required_snippets[:10])
        if len(missing_required_snippets) > 10:
            missing_preview = f"{missing_preview}, ..."
        message = (
            "required_snippets misconfigured: configured snippets are not present in the baseline source. "
            f"missing_ids={missing_preview}"
        )
        print(message, file=sys.stderr)
        prepare.append_log(
            {
                "event": "loop_start_failed",
                "timestamp": prepare.now_iso(),
                "target": args.target,
                "reason": "required_snippets_missing_in_baseline",
                "missing_required_snippets": [required_snippet_id(snippet) for snippet in missing_required_snippets],
            }
        )
        return 1

    prepare.append_log(
        {
            "event": "loop_start",
            "timestamp": prepare.now_iso(),
            "target": args.target,
            "best_metric": best_metric,
            "run_label": run_label,
            "runtime_env": run_env,
            "target_overrides": target_overrides,
            "iterations": max_iterations,
            "mode": "openai" if os.getenv("OPENAI_API_KEY") else "heuristic",
            "mutation_memory": {
                "enabled": mutation_memory is not None,
                "path": str(mutation_memory_path),
                "known_mutations": len((mutation_memory or {}).get("mutations", {}))
                if isinstance((mutation_memory or {}).get("mutations"), dict)
                else 0,
            },
            "compute_budget": {
                "max_iterations": max_iterations,
                "max_accepted": args.max_accepted if args.max_accepted > 0 else None,
                "max_runtime_seconds": args.max_runtime_seconds if args.max_runtime_seconds > 0 else None,
            },
            "git_checkpoint": {
                "mode": git_checkpoint_mode,
                "prefix": git_checkpoint_prefix,
            },
            "stats_gate": {
                "min_effect_sigma": min_effect_sigma,
                "ci_z": ci_z,
                "require_ci_separation": require_ci_separation,
                "min_series_samples": min_series_samples,
                "ab_repeats": ab_repeats,
                "require_metric_series_for_stats": require_metric_series_for_stats,
            },
            "mutation_strategy": {
                "schedule": str(target.get("mutation_schedule", "priority")),
                "ucb_explore": float(target.get("mutation_ucb_explore", 0.75)),
            },
            "source_guardrails": {
                "required_snippets": len(required_snippet_counts),
                "mode": "snippet_count_preservation",
            },
        }
    )

    blocked_mutations_until: dict[str, int] = {}
    blocked_mutation_ttl = max(0, int(target.get("blocked_mutation_ttl", 10)))
    mutation_attempts = load_mutation_attempt_counts(args.target)
    iterations_completed = 0

    iteration = 0
    while True:
        iteration += 1
        if max_iterations is not None and iteration > max_iterations:
            break

        if args.max_runtime_seconds > 0:
            elapsed = time.perf_counter() - loop_wall_start
            if elapsed >= args.max_runtime_seconds:
                stop_reason = "runtime_budget_exhausted"
                prepare.append_log(
                    {
                        "event": "budget_stop",
                        "timestamp": prepare.now_iso(),
                        "target": args.target,
                        "iteration": iteration,
                        "reason": stop_reason,
                        "elapsed_seconds": elapsed,
                        "max_runtime_seconds": args.max_runtime_seconds,
                    }
                )
                break

        iterations_completed = iteration
        train_log(
            args,
            (
                f"iter {iteration}/"
                f"{args.iterations if max_iterations is not None else 'inf'} "
                f"starting (accepted={accepted}, best={best_metric:.6f})"
            ),
            level=1,
        )
        current_source = source_path.read_text()

        mutation = ""
        diagnostics: dict[str, Any] = {}
        active_blocked = {name for name, until in blocked_mutations_until.items() if until >= iteration}
        if active_blocked:
            diagnostics["active_blocked_mutations"] = sorted(active_blocked)
        preferred_mutations: list[str] = []
        if mutation_memory is not None:
            preferred_mutations = preferred_mutations_from_memory(
                mutation_memory,
                target_name=args.target,
                language=language_norm,
            )
            if preferred_mutations:
                diagnostics["preferred_mutations"] = preferred_mutations[:5]

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
                    preferred_mutations=preferred_mutations,
                    mutation_memory=mutation_memory,
                    target_name=args.target,
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
                preferred_mutations=preferred_mutations,
                mutation_memory=mutation_memory,
                target_name=args.target,
            )

        if (
            not changed
            and is_retryable_no_change_mutation(mutation)
            and bool(target.get("recover_from_no_change", True))
            and active_blocked
        ):
            released = release_oldest_blocked_mutation(blocked_mutations_until, active_blocked)
            if released:
                diagnostics["released_blocked_mutation"] = released
                retry_blocked = {
                    name for name, until in blocked_mutations_until.items() if until >= iteration
                }
                if retry_blocked:
                    diagnostics["active_blocked_mutations"] = sorted(retry_blocked)
                else:
                    diagnostics.pop("active_blocked_mutations", None)
                candidate, mutation, changed = heuristic_candidate(
                    current_source,
                    iteration,
                    language,
                    source_path,
                    blocked_mutations=retry_blocked,
                    mutation_attempts=mutation_attempts,
                    target_config=target,
                    preferred_mutations=preferred_mutations,
                    mutation_memory=mutation_memory,
                    target_name=args.target,
                )

        if not changed:
            train_log(
                args,
                f"iter {iteration}: skipped (mutation={mutation}, reason=no_change)",
                level=1,
            )
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

        mutation_key = normalize_mutation_label(mutation) or mutation
        mutation_attempts[mutation_key] = mutation_attempts.get(mutation_key, 0) + 1
        train_log(args, f"iter {iteration}: evaluating mutation={mutation_key}", level=2)

        guard_ok, guard_details = required_snippet_guard(
            candidate,
            required_snippet_counts,
            language=language_norm,
        )
        if not guard_ok:
            diagnostics["required_snippets"] = guard_details
            blocked_mutations_until[mutation_key] = iteration + blocked_mutation_ttl

            violation_ids = ",".join(v.get("id", "") for v in guard_details.get("violations", []) if isinstance(v, dict))
            guard_notes = (
                f"rejected_guardrail_required_snippets:{mutation};"
                f"violations={violation_ids or 'unknown'}"
            )
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
                notes=f"{guard_notes};run={run_label}",
            )
            prepare.append_log(
                {
                    "event": "loop_iteration",
                    "timestamp": prepare.now_iso(),
                    "target": args.target,
                    "iteration": iteration,
                    "mutation": mutation,
                    "mutation_key": mutation_key,
                    "accepted": False,
                    "best_metric": best_metric,
                    "effective_threshold": threshold_diag,
                    "result": {
                        "status": "guardrail_rejected",
                        "metric_name": target["metric_name"],
                        "metric_value": best_metric,
                        "notes": guard_notes,
                    },
                    "diagnostics": diagnostics,
                },
            )
            train_log(
                args,
                f"iter {iteration}: rejected mutation={mutation_key} reason=required_snippet_guardrail",
                level=1,
            )
            continue

        objective_guard_ok, objective_guard_details = trackb_objective_guard(
            current_source=current_source,
            candidate_source=candidate,
            source_path=source_path,
            target_config=target,
        )
        diagnostics["trackb_objective_guard"] = objective_guard_details
        if not objective_guard_ok:
            reason = str(objective_guard_details.get("status", "objective_guard_failed"))
            changed_paths_raw = objective_guard_details.get("paths", [])
            changed_paths = ",".join(str(item) for item in changed_paths_raw) if isinstance(changed_paths_raw, list) else ""
            guard_notes = (
                f"rejected_guardrail_trackb_objective:{mutation};"
                f"reason={reason};paths={changed_paths or 'n/a'}"
            )
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
                notes=f"{guard_notes};run={run_label}",
            )
            prepare.append_log(
                {
                    "event": "loop_iteration",
                    "timestamp": prepare.now_iso(),
                    "target": args.target,
                    "iteration": iteration,
                    "mutation": mutation,
                    "mutation_key": mutation_key,
                    "accepted": False,
                    "best_metric": best_metric,
                    "effective_threshold": threshold_diag,
                    "result": {
                        "status": "guardrail_rejected",
                        "metric_name": target["metric_name"],
                        "metric_value": best_metric,
                        "notes": guard_notes,
                    },
                    "diagnostics": diagnostics,
                },
            )
            train_log(
                args,
                f"iter {iteration}: rejected mutation={mutation_key} reason=trackb_objective_guardrail",
                level=1,
            )
            continue

        source_path.write_text(candidate)
        try:
            result = prepare.evaluate_target(args.target)

            metric_value = result.get("metric_value")
            metric_for_row = float(metric_value) if metric_value is not None else None
            success = result.get("status") == "success" and metric_value is not None

            metric_series = parse_metric_series(result.get("debug", {}).get("metric_values"))
            missing_metric_series = False
            if success and require_metric_series_for_stats and metric_series is None:
                success = False
                missing_metric_series = True
                source_path.write_text(current_source)
                diagnostics["metric_series_guard"] = {
                    "required": True,
                    "status": "missing",
                    "reason": "debug.metric_values absent",
                }
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
            if missing_metric_series:
                notes = f"rejected_missing_metric_series:{notes}"

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
                        accepted_series = ab_details.get("candidate_values") if ab_details else None
                        if isinstance(accepted_series, list) and accepted_series:
                            best_metric_series = [float(v) for v in accepted_series]
                        elif confirmed_values:
                            best_metric_series = [float(v) for v in confirmed_values]
                        elif metric_series is not None and metric_series:
                            best_metric_series = metric_series
                        if required_snippets:
                            required_snippet_counts = required_snippet_profile(
                                candidate,
                                required_snippets,
                                language=language_norm,
                            )
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
                if notes.startswith("rejected_"):
                    pass
                elif success:
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
                blocked_mutations_until[mutation_key] = iteration + blocked_mutation_ttl

            if mutation_memory is not None:
                mutation_memory = update_and_save_mutation_memory(
                    mutation_memory_path,
                    mutation_memory,
                    mutation=mutation_key,
                    accepted=improved,
                    target_name=args.target,
                    language=infer_mutation_language(mutation=mutation_key, target_language=language_norm),
                    timestamp=prepare.now_iso(),
                    metric_before=best_before if improved else None,
                    metric_after=best_metric if improved else None,
                )

            should_write_artifact = args.artifacts == "all" or (args.artifacts == "accepted" and improved)
            if should_write_artifact:
                artifact_source = candidate if improved or args.artifacts == "all" else current_source
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

            should_git_checkpoint = git_checkpoint_mode == "all" or (
                git_checkpoint_mode == "accepted" and improved
            )
            if should_git_checkpoint:
                git_result = git_checkpoint_commit(
                    source_path=source_path,
                    target_name=args.target,
                    iteration=iteration,
                    mutation=mutation_key,
                    best_metric=best_metric,
                    metric_value=metric_for_row,
                    prefix=git_checkpoint_prefix,
                )
                diagnostics["git_checkpoint"] = git_result
                if git_result.get("status") == "failed":
                    train_log(
                        args,
                        f"iter {iteration}: git checkpoint failed ({git_result.get('message', git_result.get('reason', 'unknown'))})",
                        level=1,
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

            decision = "accepted" if improved else "rejected"
            metric_s = "n/a" if metric_for_row is None else f"{float(metric_for_row):.6f}"
            train_log(
                args,
                (
                    f"iter {iteration}: {decision} mutation={mutation_key} status={result.get('status')} "
                    f"metric={metric_s} best={best_metric:.6f} notes={summarize_notes(notes)}"
                ),
                level=1,
            )
            if diagnostics:
                train_log(args, f"iter {iteration} diagnostics={json.dumps(diagnostics, sort_keys=True)}", level=2)

            prepare.append_log(
                {
                    "event": "loop_iteration",
                    "timestamp": prepare.now_iso(),
                    "target": args.target,
                    "iteration": iteration,
                    "mutation": mutation,
                    "mutation_key": mutation_key,
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
                stop_reason = "max_accepted_reached"
                break
        except BaseException:
            source_path.write_text(current_source)
            raise

    elapsed_seconds = time.perf_counter() - loop_wall_start
    restore_on_no_accept = bool(target.get("restore_source_on_no_accept", True))
    if accepted == 0 and restore_on_no_accept:
        current_source = source_path.read_text()
        if current_source != initial_source:
            source_path.write_text(initial_source)
            train_log(
                args,
                "restored source to pre-run baseline because no mutations were accepted",
                level=1,
            )
            prepare.append_log(
                {
                    "event": "loop_restore_source",
                    "timestamp": prepare.now_iso(),
                    "target": args.target,
                    "run_label": run_label,
                    "reason": "no_accepted_mutations",
                }
            )
    train_log(
        args,
        (
            f"completed target={args.target} stop_reason={stop_reason} accepted={accepted} "
            f"iterations_completed={iterations_completed} best_metric={best_metric:.6f} elapsed_s={elapsed_seconds:.3f}"
        ),
        level=1,
    )
    prepare.append_log(
        {
            "event": "loop_end",
            "timestamp": prepare.now_iso(),
            "target": args.target,
            "run_label": run_label,
            "stop_reason": stop_reason,
            "iterations_completed": iterations_completed,
            "accepted": accepted,
            "best_metric": best_metric,
            "elapsed_seconds": elapsed_seconds,
            "compute_budget": {
                "max_iterations": max_iterations,
                "max_accepted": args.max_accepted if args.max_accepted > 0 else None,
                "max_runtime_seconds": args.max_runtime_seconds if args.max_runtime_seconds > 0 else None,
            },
        }
    )

    print(
        json.dumps(
            {
                "target": args.target,
                "best_metric": best_metric,
                "accepted": accepted,
                "iterations_requested": max_iterations,
                "infinite_iterations": max_iterations is None,
                "iterations_completed": iterations_completed,
                "max_accepted": args.max_accepted,
                "max_runtime_seconds": args.max_runtime_seconds if args.max_runtime_seconds > 0 else None,
                "git_checkpoint_mode": git_checkpoint_mode,
                "git_checkpoint_prefix": git_checkpoint_prefix if git_checkpoint_mode != "off" else None,
                "elapsed_seconds": elapsed_seconds,
                "stop_reason": stop_reason,
            },
            indent=2,
            sort_keys=True,
        )
    )

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AutoPoseidon train loop (Karpathy-compatible entrypoint)")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase stderr verbosity")
    parser.add_argument(
        "--debug-command-output",
        action="store_true",
        help="Print captured benchmark/build/test command stdout/stderr (truncated)",
    )
    parser.add_argument(
        "--debug-max-chars",
        type=int,
        default=4000,
        help="Max chars shown per command stream when debug output is enabled",
    )
    parser.add_argument("--target", default="cairo_poseidon_style_t8", help="Target from config/targets.json")
    parser.add_argument(
        "--iterations",
        type=int,
        default=25,
        help="Max optimization iterations (0 = run indefinitely until another budget cap triggers)",
    )
    parser.add_argument("--max-accepted", type=int, default=0, help="Stop after N accepted mutations (0 = no cap)")
    parser.add_argument(
        "--max-runtime-seconds",
        type=float,
        default=0.0,
        help="Stop loop when wall-clock runtime budget is reached (0 disables)",
    )
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
    parser.add_argument(
        "--mutation-memory-file",
        default=str(DEFAULT_MUTATION_MEMORY_FILE),
        help="Path to persisted mutation replay memory (used for cross-target replay)",
    )
    parser.add_argument(
        "--disable-mutation-memory",
        action="store_true",
        help="Disable mutation replay memory and use only per-run heuristics",
    )
    parser.add_argument(
        "--git-checkpoint-mode",
        choices=["off", "accepted", "all"],
        default="off",
        help="Create non-interactive git commits for loop checkpoints (Karpathy-style traceability)",
    )
    parser.add_argument(
        "--git-checkpoint-prefix",
        default="autoresearch",
        help="Commit message prefix used when --git-checkpoint-mode is enabled",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.iterations < 0:
        parser.error("--iterations must be >= 0")
    return run_loop(args)


if __name__ == "__main__":
    raise SystemExit(main())
