#!/usr/bin/env python3
"""Mutable Track B attack kernels.

This module isolates mutation-friendly kernel logic from the immutable
`attack_harness.py` evaluator so optimizer runs cannot modify the judge.
"""

from __future__ import annotations

import itertools
import math
import random
from typing import Any, Callable


def parse_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp_int(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def clamp_float(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def output_tag(value: int, bits: int) -> int:
    mask = (1 << bits) - 1
    return int(value) & mask


def middle_key(state: list[int], *, lanes: list[int], bits: int) -> tuple[int, ...]:
    mask = (1 << bits) - 1
    out: list[int] = []
    for lane in lanes:
        out.append(int(state[lane]) & mask)
    return tuple(out)


def monomial_exponents(num_vars: int, max_degree: int) -> list[tuple[int, ...]]:
    exps: list[tuple[int, ...]] = [tuple(0 for _ in range(num_vars))]
    for degree in range(1, max_degree + 1):
        for comb in itertools.combinations_with_replacement(range(num_vars), degree):
            out = [0] * num_vars
            for idx in comb:
                out[idx] += 1
            exps.append(tuple(out))
    return exps


def eval_monomial(values: list[int], exponents: tuple[int, ...], modulus: int) -> int:
    acc = 1
    for value, exp in zip(values, exponents):
        if exp <= 0:
            continue
        acc = (acc * pow(int(value) % modulus, int(exp), modulus)) % modulus
    return acc


def build_design_row(values: list[int], exponents: list[tuple[int, ...]], modulus: int) -> list[int]:
    return [eval_monomial(values, exp, modulus) for exp in exponents]


def egcd(a: int, b: int) -> tuple[int, int, int]:
    if b == 0:
        return (a, 1, 0)
    g, x1, y1 = egcd(b, a % b)
    return g, y1, x1 - (a // b) * y1


def modinv(a: int, m: int) -> int:
    g, x, _ = egcd(a % m, m)
    if g != 1:
        raise ValueError(f"mod inverse does not exist for a={a} mod m={m}")
    return x % m


def solve_linear_system_mod(
    matrix_rows: list[list[int]],
    rhs: list[int],
    *,
    modulus: int,
) -> dict[str, Any]:
    if not matrix_rows:
        return {
            "rank": 0,
            "nullity": 0,
            "inconsistent": False,
            "solution": [],
            "row_ops": 0,
            "pivot_columns": [],
        }

    n_rows = len(matrix_rows)
    n_cols = len(matrix_rows[0])
    aug: list[list[int]] = []
    for row, b in zip(matrix_rows, rhs):
        if len(row) != n_cols:
            raise ValueError("all matrix rows must have equal length")
        aug.append([int(v) % modulus for v in row] + [int(b) % modulus])

    row_ops = 0
    pivot_columns: list[int] = []
    pivot_row = 0

    for col in range(n_cols):
        pivot = None
        for r in range(pivot_row, n_rows):
            if aug[r][col] % modulus != 0:
                pivot = r
                break
        if pivot is None:
            continue

        if pivot != pivot_row:
            aug[pivot_row], aug[pivot] = aug[pivot], aug[pivot_row]
            row_ops += 1

        inv = modinv(aug[pivot_row][col], modulus)
        for c in range(col, n_cols + 1):
            aug[pivot_row][c] = (aug[pivot_row][c] * inv) % modulus
        row_ops += n_cols - col + 1

        for r in range(n_rows):
            if r == pivot_row:
                continue
            factor = aug[r][col] % modulus
            if factor == 0:
                continue
            for c in range(col, n_cols + 1):
                aug[r][c] = (aug[r][c] - factor * aug[pivot_row][c]) % modulus
            row_ops += n_cols - col + 1

        pivot_columns.append(col)
        pivot_row += 1
        if pivot_row >= n_rows:
            break

    inconsistent = False
    for r in range(n_rows):
        all_zero = True
        for c in range(n_cols):
            if aug[r][c] % modulus != 0:
                all_zero = False
                break
        if all_zero and aug[r][n_cols] % modulus != 0:
            inconsistent = True
            break

    solution = [0] * n_cols
    if not inconsistent:
        for r, c in enumerate(pivot_columns):
            solution[c] = aug[r][n_cols] % modulus

    rank = len(pivot_columns)
    return {
        "rank": rank,
        "nullity": max(0, n_cols - rank),
        "inconsistent": inconsistent,
        "solution": solution,
        "row_ops": row_ops,
        "pivot_columns": pivot_columns,
    }


def eval_linearized_polynomial(
    row: list[int],
    coeffs: list[int],
    *,
    modulus: int,
) -> int:
    acc = 0
    for a, b in zip(row, coeffs):
        acc = (acc + (int(a) * int(b))) % modulus
    return acc


def differential_kernel(
    *,
    spec: Any,
    analysis: dict[str, Any],
    search: dict[str, int],
    rng: random.Random,
    random_state_fn: Callable[[random.Random, Any], list[int]],
    poseidon2_permute_fn: Callable[[list[int], Any], list[int]],
) -> dict[str, Any]:
    # Rebind injected helpers to bare names so mutation operators can rewrite call sites uniformly.
    random_state = random_state_fn
    poseidon2_permute = poseidon2_permute_fn

    lane = int(analysis["target_lane"])
    bits = int(analysis["truncated_bits"])
    bucket_count = 1 << bits
    expected_prob = 1.0 / float(bucket_count)

    best_prob = 0.0
    best_bits = float("inf")
    best_delta: list[int] = [0] * spec.width
    best_advantage_bits = 0.0

    for _ in range(int(search["differential_candidates"])):
        delta = [0] * spec.width
        lane_idx = rng.randrange(spec.width)
        delta[lane_idx] = rng.randrange(1, spec.modulus)
        if rng.random() < 0.35:
            for i in range(spec.width):
                if i == lane_idx:
                    continue
                if rng.random() < 0.2:
                    delta[i] = rng.randrange(spec.modulus)

        counts: dict[int, int] = {}
        samples = int(search["differential_samples_per_candidate"])
        for _ in range(samples):
            x = random_state(rng, spec)
            x2 = [(a + b) % spec.modulus for a, b in zip(x, delta)]
            y1 = poseidon2_permute(x, spec)
            y2 = poseidon2_permute(x2, spec)
            d = output_tag((y2[lane] - y1[lane]) % spec.modulus, bits)
            counts[d] = counts.get(d, 0) + 1

        max_count = max(counts.values()) if counts else 0
        max_prob = float(max_count) / float(max(1, samples))
        complexity_bits = -math.log2(max(max_prob, 1e-15))
        advantage_bits = max(0.0, math.log2(max(max_prob / expected_prob, 1e-15)))

        if complexity_bits < best_bits:
            best_bits = complexity_bits
            best_prob = max_prob
            best_delta = delta
            best_advantage_bits = advantage_bits

    return {
        "best_probability": best_prob,
        "complexity_bits": best_bits,
        "advantage_bits": best_advantage_bits,
        "best_input_delta": best_delta,
        "expected_uniform_probability": expected_prob,
        "candidates": int(search["differential_candidates"]),
        "samples_per_candidate": int(search["differential_samples_per_candidate"]),
    }


def mitm_truncated_preimage_kernel(
    *,
    spec: Any,
    analysis: dict[str, Any],
    search: dict[str, int],
    rng: random.Random,
    random_state_fn: Callable[[random.Random, Any], list[int]],
    poseidon2_permute_fn: Callable[[list[int], Any], list[int]],
    poseidon2_prefix_fn: Callable[[list[int], Any, int], list[int]],
    poseidon2_invert_to_prefix_fn: Callable[[list[int], Any, int], list[int]],
) -> dict[str, Any]:
    # Rebind injected helpers to bare names so mutation operators can rewrite call sites uniformly.
    random_state = random_state_fn
    poseidon2_permute = poseidon2_permute_fn
    poseidon2_prefix = poseidon2_prefix_fn
    poseidon2_invert_to_prefix = poseidon2_invert_to_prefix_fn

    lane = int(analysis["target_lane"])
    trunc_bits = int(analysis["truncated_bits"])
    split_round = int(analysis["split_round"])
    key_bits = int(analysis["middle_key_bits"])
    key_lanes = list(analysis["middle_key_lanes"])

    target_input = random_state(rng, spec)
    target_output = poseidon2_permute(target_input, spec)
    target_tag = output_tag(target_output[lane], trunc_bits)

    forward_budget = int(search["mitm_forward_states"])
    backward_budget = int(search["mitm_backward_states"])

    table: dict[tuple[int, ...], list[list[int]]] = {}
    for _ in range(forward_budget):
        x = random_state(rng, spec)
        mid = poseidon2_prefix(x, spec, split_round)
        key = middle_key(mid, lanes=key_lanes, bits=key_bits)
        bucket = table.setdefault(key, [])
        if len(bucket) < 4:
            bucket.append(x)

    matches = 0
    verified = 0
    first_success_cost: int | None = None
    attempts = 0
    max_lane_value = spec.modulus - 1
    lane_hi_range = max(1, max_lane_value >> trunc_bits)

    for _ in range(backward_budget):
        attempts += 1
        y_guess = random_state(rng, spec)
        hi = rng.randrange(lane_hi_range)
        lane_value = ((hi << trunc_bits) | target_tag) % spec.modulus
        y_guess[lane] = lane_value

        mid_guess = poseidon2_invert_to_prefix(y_guess, spec, split_round)
        key = middle_key(mid_guess, lanes=key_lanes, bits=key_bits)
        candidates = table.get(key)
        if not candidates:
            continue
        matches += 1
        for cand in candidates:
            out = poseidon2_permute(cand, spec)
            if output_tag(out[lane], trunc_bits) == target_tag:
                if cand != target_input:
                    verified += 1
                    if first_success_cost is None:
                        first_success_cost = attempts + forward_budget
                    break

    total_work = forward_budget + backward_budget
    if first_success_cost is None:
        if matches > 0:
            complexity_bits = math.log2(max(2.0, float(total_work) / float(matches)))
            complexity_mode = "middle_state_match"
        else:
            complexity_bits = math.log2(max(2, total_work))
            complexity_mode = "no_match"
    else:
        complexity_bits = math.log2(max(2, first_success_cost))
        complexity_mode = "verified_preimage"

    return {
        "target_lane": lane,
        "target_tag": target_tag,
        "truncated_bits": trunc_bits,
        "split_round": split_round,
        "middle_key_bits": key_bits,
        "middle_key_lanes": key_lanes,
        "forward_budget": forward_budget,
        "backward_budget": backward_budget,
        "table_buckets": len(table),
        "table_entries": sum(len(v) for v in table.values()),
        "matches": matches,
        "match_rate": float(matches) / float(max(1, backward_budget)),
        "verified_hits": verified,
        "first_success_cost": first_success_cost,
        "complexity_bits": complexity_bits,
        "complexity_mode": complexity_mode,
        "attack_found": bool(verified > 0),
    }


def birthday_collision_kernel(
    *,
    spec: Any,
    analysis: dict[str, Any],
    search: dict[str, int],
    rng: random.Random,
    random_state_fn: Callable[[random.Random, Any], list[int]],
    poseidon2_permute_fn: Callable[[list[int], Any], list[int]],
) -> dict[str, Any]:
    # Rebind injected helpers to bare names so mutation operators can rewrite call sites uniformly.
    random_state = random_state_fn
    poseidon2_permute = poseidon2_permute_fn

    lane = int(analysis["target_lane"])
    bits = int(analysis["truncated_bits"])
    samples = int(search["collision_samples"])

    seen: dict[int, list[int]] = {}
    first_collision_at: int | None = None
    collisions = 0
    distinct = 0

    for idx in range(1, samples + 1):
        x = random_state(rng, spec)
        y = poseidon2_permute(x, spec)
        tag = output_tag(y[lane], bits)
        if tag in seen:
            collisions += 1
            if first_collision_at is None:
                first_collision_at = idx
        else:
            distinct += 1
            seen[tag] = x

    expected_random_bits = bits / 2.0
    if first_collision_at is None:
        observed_bits = math.log2(max(2, samples))
    else:
        observed_bits = math.log2(max(2, first_collision_at))

    advantage_bits = max(0.0, expected_random_bits - observed_bits)

    return {
        "target_lane": lane,
        "truncated_bits": bits,
        "samples": samples,
        "distinct_tags": distinct,
        "collisions": collisions,
        "first_collision_at": first_collision_at,
        "observed_complexity_bits": observed_bits,
        "expected_random_complexity_bits": expected_random_bits,
        "advantage_bits": advantage_bits,
    }


def algebraic_elimination_kernel(
    *,
    spec: Any,
    analysis: dict[str, Any],
    search: dict[str, int],
    rng: random.Random,
    random_state_fn: Callable[[random.Random, Any], list[int]],
    poseidon2_prefix_fn: Callable[[list[int], Any, int], list[int]],
    clamp_int_fn: Callable[[int, int, int], int] = clamp_int,
) -> dict[str, Any]:
    # Rebind injected helpers to bare names so mutation operators can rewrite call sites uniformly.
    random_state = random_state_fn
    poseidon2_prefix = poseidon2_prefix_fn
    clamp_int = clamp_int_fn

    rounds = clamp_int(int(analysis["algebraic_rounds"]), 1, spec.total_rounds)
    degree = clamp_int(int(analysis["algebraic_degree"]), 1, 3)
    target_lane = clamp_int(int(analysis["algebraic_target_lane"]), 0, spec.width - 1)
    output_bits = clamp_int(int(analysis["algebraic_output_bits"]), 4, 64)
    unknown_lanes = list(analysis["algebraic_unknown_lanes"])
    train_samples = int(search["algebraic_train_samples"])
    val_samples = int(search["algebraic_validation_samples"])

    # Fix non-symbolic lanes to a deterministic template and vary only selected unknown lanes.
    template = random_state(rng, spec)
    for lane in unknown_lanes:
        template[lane] = 0

    exponents = monomial_exponents(len(unknown_lanes), degree)
    term_count = len(exponents)
    train_matrix: list[list[int]] = []
    train_rhs: list[int] = []
    val_matrix: list[list[int]] = []
    val_rhs: list[int] = []

    def sample_point() -> tuple[list[int], int]:
        state = list(template)
        values: list[int] = []
        for lane in unknown_lanes:
            v = rng.randrange(spec.modulus)
            state[lane] = v
            values.append(v)
        out = poseidon2_prefix(state, spec, rounds)
        y = output_tag(out[target_lane], output_bits)
        return values, y

    for _ in range(train_samples):
        values, y = sample_point()
        train_matrix.append(build_design_row(values, exponents, spec.modulus))
        train_rhs.append(y % spec.modulus)

    for _ in range(val_samples):
        values, y = sample_point()
        val_matrix.append(build_design_row(values, exponents, spec.modulus))
        val_rhs.append(y % spec.modulus)

    solve = solve_linear_system_mod(train_matrix, train_rhs, modulus=spec.modulus)
    coeffs = list(solve["solution"])
    inconsistent = bool(solve["inconsistent"])
    rank = int(solve["rank"])
    nullity = int(solve["nullity"])

    train_exact = 0
    for row, y in zip(train_matrix, train_rhs):
        pred = eval_linearized_polynomial(row, coeffs, modulus=spec.modulus)
        if pred == y:
            train_exact += 1

    val_exact = 0
    for row, y in zip(val_matrix, val_rhs):
        pred = eval_linearized_polynomial(row, coeffs, modulus=spec.modulus)
        if pred == y:
            val_exact += 1

    train_exact_rate = float(train_exact) / float(max(1, train_samples))
    val_exact_rate = float(val_exact) / float(max(1, val_samples))
    base_bits = math.log2(max(2, term_count))
    if inconsistent:
        complexity_bits = base_bits + 4.0
    else:
        rank_gain = math.log2(max(1, nullity + 1))
        fit_gain = (2.5 * val_exact_rate) + (1.5 * train_exact_rate)
        complexity_bits = max(1.0, base_bits - rank_gain - fit_gain)

    return {
        "rounds": rounds,
        "degree": degree,
        "target_lane": target_lane,
        "output_bits": output_bits,
        "unknown_lanes": unknown_lanes,
        "train_samples": train_samples,
        "validation_samples": val_samples,
        "term_count": term_count,
        "rank": rank,
        "nullity": nullity,
        "inconsistent": inconsistent,
        "row_ops": int(solve["row_ops"]),
        "train_exact_rate": train_exact_rate,
        "validation_exact_rate": val_exact_rate,
        "complexity_bits": complexity_bits,
    }


def score(
    *,
    config: dict[str, Any],
    search: dict[str, int],
    differential: dict[str, Any],
    mitm_preimage: dict[str, Any],
    collision: dict[str, Any],
    algebraic: dict[str, Any],
    clamp_float_fn: Callable[[float, float, float], float] = clamp_float,
    parse_float_fn: Callable[[Any, float], float] = parse_float,
) -> dict[str, float]:
    # Rebind injected helpers to bare names so mutation operators can rewrite call sites uniformly.
    clamp_float = clamp_float_fn
    parse_float = parse_float_fn

    objective = config.get("objective", {})
    if not isinstance(objective, dict):
        objective = {}

    wd = clamp_float(parse_float(objective.get("weight_differential"), 0.35), 0.0, 1.0)
    wp = clamp_float(parse_float(objective.get("weight_preimage"), 0.45), 0.0, 1.0)
    wc = clamp_float(parse_float(objective.get("weight_collision"), 0.20), 0.0, 1.0)
    wa = clamp_float(parse_float(objective.get("weight_algebraic"), 0.0), 0.0, 1.0)
    tw = wd + wp + wc + wa
    if tw <= 0.0:
        wd, wp, wc, wa, tw = 0.35, 0.45, 0.20, 0.0, 1.0
    wd /= tw
    wp /= tw
    wc /= tw
    wa /= tw

    cap_bits = clamp_float(parse_float(objective.get("complexity_cap_bits"), 32.0), 8.0, 128.0)
    found_threshold_bits = clamp_float(parse_float(objective.get("attack_found_threshold_bits"), 16.0), 4.0, 64.0)
    found_bonus = clamp_float(parse_float(objective.get("attack_found_bonus"), 1.0), 0.0, 20.0)
    verified_bonus = clamp_float(parse_float(objective.get("verified_found_bonus"), found_bonus), 0.0, 40.0)
    cost_penalty_scale = clamp_float(parse_float(objective.get("cost_penalty_scale"), 0.08), 0.0, 3.0)

    diff_bits = float(differential["complexity_bits"])
    pre_bits = float(mitm_preimage["complexity_bits"])
    coll_bits = float(collision["observed_complexity_bits"])
    alg_bits = float(algebraic["complexity_bits"])

    diff_signal = max(0.0, cap_bits - diff_bits)
    pre_signal = max(0.0, cap_bits - pre_bits)
    coll_signal = max(0.0, float(collision["advantage_bits"]))
    alg_signal = max(0.0, cap_bits - alg_bits)

    heuristic_found = (
        bool(mitm_preimage.get("attack_found", False))
        or diff_bits <= found_threshold_bits
        or pre_bits <= found_threshold_bits
        or alg_bits <= found_threshold_bits
    )
    verified_found = bool(int(mitm_preimage.get("verified_hits", 0)) > 0)
    heuristic_component = found_bonus if heuristic_found else 0.0
    verified_component = verified_bonus if verified_found else 0.0

    search_cost = (
        int(search["differential_candidates"]) * int(search["differential_samples_per_candidate"])
        + int(search["mitm_forward_states"])
        + int(search["mitm_backward_states"])
        + int(search["collision_samples"])
    )
    cost_penalty = cost_penalty_scale * math.log2(max(2, search_cost))

    base_signal_score = (wd * diff_signal) + (wp * pre_signal) + (wc * coll_signal) + (wa * alg_signal) - cost_penalty
    attack_score_signal = base_signal_score + heuristic_component
    attack_score_verified = base_signal_score + verified_component
    attack_score_algebraic = alg_signal - cost_penalty

    return {
        # Backward-compatible default score: heuristic signal lane.
        "attack_score": attack_score_signal,
        "attack_score_signal": attack_score_signal,
        "attack_score_verified": attack_score_verified,
        "attack_found": 1.0 if verified_found else 0.0,
        "attack_score_algebraic": attack_score_algebraic,
        "heuristic_found": 1.0 if heuristic_found else 0.0,
        "verified_found": 1.0 if verified_found else 0.0,
        "best_attack_complexity_bits": min(diff_bits, pre_bits, coll_bits, alg_bits),
        "differential_complexity_bits": diff_bits,
        "preimage_complexity_bits": pre_bits,
        "collision_complexity_bits": coll_bits,
        "algebraic_complexity_bits": alg_bits,
        "differential_signal": diff_signal,
        "preimage_signal": pre_signal,
        "collision_signal": coll_signal,
        "algebraic_signal": alg_signal,
        "cost_penalty": cost_penalty,
    }
