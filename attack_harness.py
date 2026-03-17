#!/usr/bin/env python3
"""Deterministic reduced-round Poseidon2 cryptanalysis harness.

This harness is intended for autonomous search loops:
- deterministic constant generation and sampling
- reproducible reduced-round Poseidon2-style permutation
- attack kernels with explicit, benchmarkable metrics
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent


def parse_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp_int(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def clamp_float(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def is_prime(value: int) -> bool:
    if value < 2:
        return False

    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for prime in small_primes:
        if value == prime:
            return True
        if value % prime == 0:
            return False

    d = value - 1
    s = 0
    while d % 2 == 0:
        s += 1
        d //= 2

    if value.bit_length() <= 64:
        bases = (2, 325, 9375, 28178, 450775, 9780504, 1795265022)
    else:
        bases = small_primes

    for base in bases:
        if base % value == 0:
            continue
        x = pow(base, d, value)
        if x in (1, value - 1):
            continue
        for _ in range(s - 1):
            x = pow(x, 2, value)
            if x == value - 1:
                break
        else:
            return False
    return True


def deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in base.items():
        if isinstance(value, dict) or isinstance(value, list):
            out[key] = json.loads(json.dumps(value))
        else:
            out[key] = value
    for key, value in override.items():
        current = out.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            out[key] = deep_merge_dict(current, value)
            continue
        if isinstance(value, dict) or isinstance(value, list):
            out[key] = json.loads(json.dumps(value))
            continue
        out[key] = value
    return out


def resolve_profile_config(config: dict[str, Any], requested_profile: str) -> tuple[dict[str, Any], str]:
    selected = requested_profile.strip()
    if not selected:
        selected = str(config.get("active_profile", "")).strip()

    merged = json.loads(json.dumps(config))
    profiles = merged.get("challenge_profiles")
    if not isinstance(profiles, dict) or not selected:
        return merged, selected

    profile_payload = profiles.get(selected)
    if not isinstance(profile_payload, dict):
        valid = ", ".join(sorted(str(k) for k in profiles.keys()))
        raise ValueError(f"unknown challenge profile '{selected}' (available: {valid})")

    merged = deep_merge_dict(merged, profile_payload)
    return merged, selected


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


def hash_to_field(seed: str, domain: str, idx: int, modulus: int) -> int:
    payload = f"{seed}|{domain}|{idx}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest, "big") % modulus


def matrix_identity(n: int) -> list[list[int]]:
    out: list[list[int]] = []
    for i in range(n):
        row = [0] * n
        row[i] = 1
        out.append(row)
    return out


def matrix_mul_vec(matrix: list[list[int]], vec: list[int], modulus: int) -> list[int]:
    out: list[int] = [0] * len(matrix)
    for i, row in enumerate(matrix):
        acc = 0
        for a, b in zip(row, vec):
            acc = (acc + (a * b)) % modulus
        out[i] = acc
    return out


def matrix_inverse(matrix: list[list[int]], modulus: int) -> list[list[int]]:
    n = len(matrix)
    aug: list[list[int]] = []
    for i in range(n):
        aug.append([x % modulus for x in matrix[i]] + matrix_identity(n)[i])

    for col in range(n):
        pivot = None
        for r in range(col, n):
            if aug[r][col] % modulus != 0:
                pivot = r
                break
        if pivot is None:
            raise ValueError("matrix is singular over field")
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]

        inv_pivot = modinv(aug[col][col], modulus)
        for c in range(2 * n):
            aug[col][c] = (aug[col][c] * inv_pivot) % modulus

        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col] % modulus
            if factor == 0:
                continue
            for c in range(2 * n):
                aug[r][c] = (aug[r][c] - factor * aug[col][c]) % modulus

    inv: list[list[int]] = []
    for r in range(n):
        inv.append(aug[r][n:])
    return inv


def build_external_matrix(width: int) -> list[list[int]]:
    # M_E = 2I + J, a common efficient MDS-like construction.
    out: list[list[int]] = []
    for i in range(width):
        row = [1] * width
        row[i] = 2
        out.append(row)
    return out


def build_internal_matrix(width: int, diagonal: list[int]) -> list[list[int]]:
    if len(diagonal) != width:
        raise ValueError("internal diagonal length must equal width")
    out: list[list[int]] = []
    for i in range(width):
        row = [1] * width
        row[i] = (row[i] + diagonal[i])  # diag(v) + J
        out.append(row)
    return out


def sbox_forward(value: int, *, power: int, modulus: int) -> int:
    return pow(value % modulus, power, modulus)


def sbox_inverse(value: int, *, power_inv: int, modulus: int) -> int:
    return pow(value % modulus, power_inv, modulus)


@dataclass(frozen=True)
class Poseidon2Spec:
    modulus: int
    width: int
    full_rounds: int
    partial_rounds: int
    sbox_power: int
    sbox_power_inv: int
    external_matrix: list[list[int]]
    external_matrix_inv: list[list[int]]
    internal_matrix: list[list[int]]
    internal_matrix_inv: list[list[int]]
    external_round_constants: list[list[int]]
    internal_round_constants: list[int]

    @property
    def total_rounds(self) -> int:
        return self.full_rounds + self.partial_rounds


def build_spec(config: dict[str, Any], *, mode: str) -> Poseidon2Spec:
    poseidon2 = config.get("poseidon2", {})
    search = config.get("search", {})
    if not isinstance(poseidon2, dict):
        poseidon2 = {}
    if not isinstance(search, dict):
        search = {}

    mode_scale = 1.0
    if mode == "full":
        mode_scale = 2.0

    modulus = clamp_int(parse_int(poseidon2.get("field_modulus"), 2130706433), 1_000_003, (1 << 521) - 1)
    if not is_prime(modulus):
        raise ValueError(f"field_modulus must be prime (got {modulus})")
    width = clamp_int(parse_int(poseidon2.get("width"), 3), 2, 16)
    full_rounds = clamp_int(parse_int(poseidon2.get("full_rounds"), 8), 2, 32)
    if full_rounds % 2 != 0:
        full_rounds += 1
    partial_rounds = clamp_int(parse_int(poseidon2.get("partial_rounds"), 13), 1, 256)
    sbox_power = clamp_int(parse_int(poseidon2.get("sbox_power"), 5), 3, 11)
    if math.gcd(sbox_power, modulus - 1) != 1:
        raise ValueError(
            "sbox_power must be invertible modulo field_modulus-1 "
            f"(got power={sbox_power}, p={modulus})"
        )
    sbox_power_inv = modinv(sbox_power, modulus - 1)

    seed = str(poseidon2.get("constants_seed", "trackb-poseidon2-v1"))
    diagonal_raw = poseidon2.get("internal_diagonal")
    diagonal: list[int] = []
    if isinstance(diagonal_raw, list) and len(diagonal_raw) == width:
        for item in diagonal_raw:
            diagonal.append(int(item) % modulus)
    else:
        for i in range(width):
            diagonal.append((hash_to_field(seed, "internal_diag", i, modulus) % (modulus - 1)) + 1)

    external_matrix = build_external_matrix(width)
    internal_matrix = build_internal_matrix(width, diagonal)
    external_matrix_inv = matrix_inverse(external_matrix, modulus)
    internal_matrix_inv = matrix_inverse(internal_matrix, modulus)

    external_round_constants: list[list[int]] = []
    for r in range(full_rounds):
        row: list[int] = []
        for i in range(width):
            row.append(hash_to_field(seed, f"ext_rc_{r}", i, modulus))
        external_round_constants.append(row)
    internal_round_constants = [
        hash_to_field(seed, "int_rc", i, modulus)
        for i in range(partial_rounds)
    ]

    # Optional mode scaling can increase rounds for a stricter full profile while preserving reduced-round shape.
    if mode_scale > 1.0 and bool(parse_int(search.get("scale_rounds_in_full_mode"), 0)):
        extra = clamp_int(int(partial_rounds * (mode_scale - 1.0)), 0, 64)
        for i in range(extra):
            internal_round_constants.append(hash_to_field(seed, "int_rc_extra", i, modulus))
        partial_rounds += extra

    return Poseidon2Spec(
        modulus=modulus,
        width=width,
        full_rounds=full_rounds,
        partial_rounds=partial_rounds,
        sbox_power=sbox_power,
        sbox_power_inv=sbox_power_inv,
        external_matrix=external_matrix,
        external_matrix_inv=external_matrix_inv,
        internal_matrix=internal_matrix,
        internal_matrix_inv=internal_matrix_inv,
        external_round_constants=external_round_constants,
        internal_round_constants=internal_round_constants,
    )


def apply_full_round(state: list[int], spec: Poseidon2Spec, rc: list[int]) -> list[int]:
    s = [(x + c) % spec.modulus for x, c in zip(state, rc)]
    s = [sbox_forward(x, power=spec.sbox_power, modulus=spec.modulus) for x in s]
    return matrix_mul_vec(spec.external_matrix, s, spec.modulus)


def invert_full_round(state: list[int], spec: Poseidon2Spec, rc: list[int]) -> list[int]:
    s = matrix_mul_vec(spec.external_matrix_inv, state, spec.modulus)
    s = [sbox_inverse(x, power_inv=spec.sbox_power_inv, modulus=spec.modulus) for x in s]
    return [(x - c) % spec.modulus for x, c in zip(s, rc)]


def apply_internal_round(state: list[int], spec: Poseidon2Spec, rc: int) -> list[int]:
    s = list(state)
    s[0] = (s[0] + rc) % spec.modulus
    s[0] = sbox_forward(s[0], power=spec.sbox_power, modulus=spec.modulus)
    return matrix_mul_vec(spec.internal_matrix, s, spec.modulus)


def invert_internal_round(state: list[int], spec: Poseidon2Spec, rc: int) -> list[int]:
    s = matrix_mul_vec(spec.internal_matrix_inv, state, spec.modulus)
    s[0] = sbox_inverse(s[0], power_inv=spec.sbox_power_inv, modulus=spec.modulus)
    s[0] = (s[0] - rc) % spec.modulus
    return s


def poseidon2_permute(state: list[int], spec: Poseidon2Spec) -> list[int]:
    s = list(state)
    full_half = spec.full_rounds // 2

    for r in range(full_half):
        s = apply_full_round(s, spec, spec.external_round_constants[r])

    for r in range(spec.partial_rounds):
        s = apply_internal_round(s, spec, spec.internal_round_constants[r])

    for r in range(full_half):
        rc_idx = full_half + r
        s = apply_full_round(s, spec, spec.external_round_constants[rc_idx])

    return s


def poseidon2_prefix(state: list[int], spec: Poseidon2Spec, rounds: int) -> list[int]:
    rounds = clamp_int(rounds, 0, spec.total_rounds)
    s = list(state)
    full_half = spec.full_rounds // 2

    round_idx = 0
    for r in range(full_half):
        if round_idx >= rounds:
            return s
        s = apply_full_round(s, spec, spec.external_round_constants[r])
        round_idx += 1

    for r in range(spec.partial_rounds):
        if round_idx >= rounds:
            return s
        s = apply_internal_round(s, spec, spec.internal_round_constants[r])
        round_idx += 1

    for r in range(full_half):
        if round_idx >= rounds:
            return s
        rc_idx = full_half + r
        s = apply_full_round(s, spec, spec.external_round_constants[rc_idx])
        round_idx += 1

    return s


def poseidon2_invert_to_prefix(state: list[int], spec: Poseidon2Spec, prefix_rounds: int) -> list[int]:
    prefix_rounds = clamp_int(prefix_rounds, 0, spec.total_rounds)
    s = list(state)
    full_half = spec.full_rounds // 2

    round_types: list[tuple[str, int]] = []
    for r in range(full_half):
        round_types.append(("full", r))
    for r in range(spec.partial_rounds):
        round_types.append(("internal", r))
    for r in range(full_half):
        round_types.append(("full", full_half + r))

    for idx in range(spec.total_rounds - 1, prefix_rounds - 1, -1):
        kind, ridx = round_types[idx]
        if kind == "full":
            s = invert_full_round(s, spec, spec.external_round_constants[ridx])
        else:
            s = invert_internal_round(s, spec, spec.internal_round_constants[ridx])
    return s


def random_state(rng: random.Random, spec: Poseidon2Spec) -> list[int]:
    return [rng.randrange(spec.modulus) for _ in range(spec.width)]


def parse_search(config: dict[str, Any], *, mode: str) -> dict[str, int]:
    raw = config.get("search", {})
    if not isinstance(raw, dict):
        raw = {}

    scale = 1.0
    if mode == "full":
        scale = 2.0

    def scaled(key: str, default: int, lo: int, hi: int) -> int:
        base = parse_int(raw.get(key), default)
        return clamp_int(int(max(1, round(base * scale))), lo, hi)

    return {
        "seed": parse_int(raw.get("seed"), 1337),
        "differential_candidates": scaled("differential_candidates", 64, 4, 8192),
        "differential_samples_per_candidate": scaled("differential_samples_per_candidate", 192, 8, 262144),
        "mitm_forward_states": scaled("mitm_forward_states", 4096, 64, 1_000_000),
        "mitm_backward_states": scaled("mitm_backward_states", 4096, 64, 1_000_000),
        "collision_samples": scaled("collision_samples", 4096, 64, 1_000_000),
    }


def parse_analysis(config: dict[str, Any], spec: Poseidon2Spec) -> dict[str, Any]:
    raw = config.get("analysis", {})
    if not isinstance(raw, dict):
        raw = {}

    target_lane = clamp_int(parse_int(raw.get("target_lane"), 0), 0, spec.width - 1)
    truncated_bits = clamp_int(parse_int(raw.get("truncated_bits"), 24), 4, 64)
    split_round = clamp_int(parse_int(raw.get("split_round"), spec.total_rounds // 2), 1, spec.total_rounds - 1)
    middle_key_bits = clamp_int(parse_int(raw.get("middle_key_bits"), 18), 6, 64)

    key_lanes_raw = raw.get("middle_key_lanes")
    key_lanes: list[int] = []
    if isinstance(key_lanes_raw, list):
        for item in key_lanes_raw:
            idx = parse_int(item, -1)
            if 0 <= idx < spec.width and idx not in key_lanes:
                key_lanes.append(idx)
    if not key_lanes:
        key_lanes = [0]

    return {
        "target_lane": target_lane,
        "truncated_bits": truncated_bits,
        "split_round": split_round,
        "middle_key_bits": middle_key_bits,
        "middle_key_lanes": key_lanes,
    }


def output_tag(value: int, bits: int) -> int:
    mask = (1 << bits) - 1
    return int(value) & mask


def middle_key(state: list[int], *, lanes: list[int], bits: int) -> tuple[int, ...]:
    mask = (1 << bits) - 1
    out: list[int] = []
    for lane in lanes:
        out.append(int(state[lane]) & mask)
    return tuple(out)


def differential_kernel(
    *,
    spec: Poseidon2Spec,
    analysis: dict[str, Any],
    search: dict[str, int],
    rng: random.Random,
) -> dict[str, Any]:
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
    spec: Poseidon2Spec,
    analysis: dict[str, Any],
    search: dict[str, int],
    rng: random.Random,
) -> dict[str, Any]:
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
    spec: Poseidon2Spec,
    analysis: dict[str, Any],
    search: dict[str, int],
    rng: random.Random,
) -> dict[str, Any]:
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


def score(
    *,
    config: dict[str, Any],
    search: dict[str, int],
    differential: dict[str, Any],
    mitm_preimage: dict[str, Any],
    collision: dict[str, Any],
) -> dict[str, float]:
    objective = config.get("objective", {})
    if not isinstance(objective, dict):
        objective = {}

    wd = clamp_float(parse_float(objective.get("weight_differential"), 0.35), 0.0, 1.0)
    wp = clamp_float(parse_float(objective.get("weight_preimage"), 0.45), 0.0, 1.0)
    wc = clamp_float(parse_float(objective.get("weight_collision"), 0.20), 0.0, 1.0)
    tw = wd + wp + wc
    if tw <= 0.0:
        wd, wp, wc, tw = 0.35, 0.45, 0.20, 1.0
    wd /= tw
    wp /= tw
    wc /= tw

    cap_bits = clamp_float(parse_float(objective.get("complexity_cap_bits"), 32.0), 8.0, 128.0)
    found_threshold_bits = clamp_float(parse_float(objective.get("attack_found_threshold_bits"), 16.0), 4.0, 64.0)
    found_bonus = clamp_float(parse_float(objective.get("attack_found_bonus"), 1.0), 0.0, 20.0)
    verified_bonus = clamp_float(parse_float(objective.get("verified_found_bonus"), found_bonus), 0.0, 40.0)
    cost_penalty_scale = clamp_float(parse_float(objective.get("cost_penalty_scale"), 0.08), 0.0, 3.0)

    diff_bits = float(differential["complexity_bits"])
    pre_bits = float(mitm_preimage["complexity_bits"])
    coll_bits = float(collision["observed_complexity_bits"])

    diff_signal = max(0.0, cap_bits - diff_bits)
    pre_signal = max(0.0, cap_bits - pre_bits)
    coll_signal = max(0.0, float(collision["advantage_bits"]))

    heuristic_found = (
        bool(mitm_preimage.get("attack_found", False))
        or diff_bits <= found_threshold_bits
        or pre_bits <= found_threshold_bits
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

    base_signal_score = (wd * diff_signal) + (wp * pre_signal) + (wc * coll_signal) - cost_penalty
    attack_score_signal = base_signal_score + heuristic_component
    attack_score_verified = base_signal_score + verified_component

    return {
        # Backward-compatible default score: heuristic signal lane.
        "attack_score": attack_score_signal,
        "attack_score_signal": attack_score_signal,
        "attack_score_verified": attack_score_verified,
        "attack_found": 1.0 if verified_found else 0.0,
        "heuristic_found": 1.0 if heuristic_found else 0.0,
        "verified_found": 1.0 if verified_found else 0.0,
        "best_attack_complexity_bits": min(diff_bits, pre_bits, coll_bits),
        "differential_complexity_bits": diff_bits,
        "preimage_complexity_bits": pre_bits,
        "collision_complexity_bits": coll_bits,
        "differential_signal": diff_signal,
        "preimage_signal": pre_signal,
        "collision_signal": coll_signal,
        "cost_penalty": cost_penalty,
    }


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("config must be a JSON object")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deterministic reduced-round Poseidon2 cryptanalysis harness")
    parser.add_argument("--config", default="config/track_b_attack_config.json", help="Path to Track B config JSON")
    parser.add_argument(
        "--profile",
        default="",
        help="Optional challenge profile key from config.challenge_profiles",
    )
    parser.add_argument("--mode", choices=["fast", "full"], default="fast", help="Search budget mode")
    parser.add_argument("--output-format", choices=["json", "pretty"], default="json", help="Output encoding")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config_raw = load_config(config_path)
    config, selected_profile = resolve_profile_config(config_raw, args.profile)

    spec = build_spec(config, mode=args.mode)
    search = parse_search(config, mode=args.mode)
    analysis = parse_analysis(config, spec)
    base_seed = int(search["seed"])

    def kernel_rng(label: str) -> random.Random:
        material = f"{base_seed}:{args.mode}:{label}".encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
        return random.Random(seed)  # noqa: S311 - deterministic benchmarking harness, not crypto RNG.

    differential = differential_kernel(spec=spec, analysis=analysis, search=search, rng=kernel_rng("differential"))
    mitm_preimage = mitm_truncated_preimage_kernel(
        spec=spec,
        analysis=analysis,
        search=search,
        rng=kernel_rng("mitm_preimage"),
    )
    collision = birthday_collision_kernel(spec=spec, analysis=analysis, search=search, rng=kernel_rng("collision"))
    metrics = score(
        config=config,
        search=search,
        differential=differential,
        mitm_preimage=mitm_preimage,
        collision=collision,
    )

    payload = {
        "ok": True,
        "mode": args.mode,
        "config_path": str(config_path),
        "challenge_profile": selected_profile,
        "spec": {
            "field_modulus": spec.modulus,
            "width": spec.width,
            "full_rounds": spec.full_rounds,
            "partial_rounds": spec.partial_rounds,
            "sbox_power": spec.sbox_power,
            "total_rounds": spec.total_rounds,
        },
        "analysis": analysis,
        "search": search,
        "metrics": {
            "attack_score": metrics["attack_score"],
            "attack_score_signal": metrics["attack_score_signal"],
            "attack_score_verified": metrics["attack_score_verified"],
            "best_attack_complexity_bits": metrics["best_attack_complexity_bits"],
            "differential_complexity_bits": metrics["differential_complexity_bits"],
            "preimage_complexity_bits": metrics["preimage_complexity_bits"],
            "collision_complexity_bits": metrics["collision_complexity_bits"],
            "attack_found": int(metrics["attack_found"]),
            "heuristic_found": int(metrics["heuristic_found"]),
            "verified_found": int(metrics["verified_found"]),
        },
        "details": {
            "differential": differential,
            "mitm_preimage": mitm_preimage,
            "collision": collision,
            "signals": {
                "differential_signal": metrics["differential_signal"],
                "preimage_signal": metrics["preimage_signal"],
                "collision_signal": metrics["collision_signal"],
                "cost_penalty": metrics["cost_penalty"],
            },
        },
        "repro": {
            "command": (
                f"python3 attack_harness.py --config {config_path} "
                f"{f'--profile {selected_profile} ' if selected_profile else ''}"
                f"--mode {args.mode} --output-format json"
            )
        },
    }

    if args.output_format == "pretty":
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
