#!/usr/bin/env python3
"""Deterministic reduced-round Poseidon-style cryptanalysis harness.

This script intentionally evaluates a reduced-round, toy-field permutation to
enable rapid, reproducible attack-signal search loops.
"""

from __future__ import annotations

import argparse
import json
import math
import random
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


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("config must be a JSON object")
    return payload


def resolve_settings(config: dict[str, Any], *, mode: str) -> dict[str, Any]:
    challenge = config.get("challenge", {})
    search = config.get("search", {})
    objective = config.get("objective", {})

    if not isinstance(challenge, dict):
        challenge = {}
    if not isinstance(search, dict):
        search = {}
    if not isinstance(objective, dict):
        objective = {}

    mode_scale = 1.0
    if mode == "full":
        mode_scale = 2.0
    elif mode == "fast":
        mode_scale = 1.0

    full_rounds = parse_int(challenge.get("full_rounds"), 8)
    if full_rounds < 2:
        full_rounds = 2
    if full_rounds % 2 != 0:
        full_rounds += 1

    settings = {
        "challenge": {
            "field_modulus": clamp_int(parse_int(challenge.get("field_modulus"), 65537), 1021, 2**31 - 1),
            "width": clamp_int(parse_int(challenge.get("width"), 3), 2, 6),
            "full_rounds": full_rounds,
            "partial_rounds": clamp_int(parse_int(challenge.get("partial_rounds"), 22), 1, 64),
            "sbox_power": clamp_int(parse_int(challenge.get("sbox_power"), 5), 3, 11),
            "round_constants_seed": parse_int(challenge.get("round_constants_seed"), 1729),
            "target_lane": clamp_int(parse_int(challenge.get("target_lane"), 0), 0, 5),
            "preimage_target_bits": clamp_int(parse_int(challenge.get("preimage_target_bits"), 12), 4, 20),
        },
        "search": {
            "seed": parse_int(search.get("seed"), 1337),
            "delta_candidates": clamp_int(
                int(parse_int(search.get("delta_candidates"), 48) * mode_scale),
                4,
                2048,
            ),
            "samples_per_delta": clamp_int(
                int(parse_int(search.get("samples_per_delta"), 192) * mode_scale),
                16,
                65536,
            ),
            "preimage_attempts_per_trial": clamp_int(
                int(parse_int(search.get("preimage_attempts_per_trial"), 4096) * mode_scale),
                32,
                1_000_000,
            ),
            "preimage_trials": clamp_int(
                int(parse_int(search.get("preimage_trials"), 16) * mode_scale),
                2,
                2048,
            ),
        },
        "objective": {
            "weight_differential": clamp_float(
                parse_float(objective.get("weight_differential"), 0.70),
                0.0,
                1.0,
            ),
            "weight_preimage": clamp_float(
                parse_float(objective.get("weight_preimage"), 0.30),
                0.0,
                1.0,
            ),
            "complexity_cap_bits": clamp_float(
                parse_float(objective.get("complexity_cap_bits"), 24.0),
                4.0,
                128.0,
            ),
            "attack_found_threshold_bits": clamp_float(
                parse_float(objective.get("attack_found_threshold_bits"), 8.0),
                1.0,
                40.0,
            ),
            "attack_found_bonus": clamp_float(
                parse_float(objective.get("attack_found_bonus"), 1.0),
                0.0,
                16.0,
            ),
            "cost_penalty_scale": clamp_float(
                parse_float(objective.get("cost_penalty_scale"), 0.10),
                0.0,
                2.0,
            ),
        },
    }
    width = settings["challenge"]["width"]
    settings["challenge"]["target_lane"] = clamp_int(settings["challenge"]["target_lane"], 0, width - 1)
    return settings


def build_round_constants(*, rounds: int, width: int, modulus: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    out: list[list[int]] = []
    for _ in range(rounds):
        out.append([rng.randrange(modulus) for _ in range(width)])
    return out


def mds_mix(state: list[int], *, modulus: int) -> list[int]:
    width = len(state)
    out = [0] * width
    for i in range(width):
        acc = 0
        for j in range(width):
            coeff = 2 if i == j else 1
            acc += coeff * state[j]
        out[i] = acc % modulus
    return out


def sbox(value: int, *, power: int, modulus: int) -> int:
    return pow(value, power, modulus)


def permute(
    state: list[int],
    *,
    modulus: int,
    full_rounds: int,
    partial_rounds: int,
    sbox_power: int,
    round_constants: list[list[int]],
) -> list[int]:
    width = len(state)
    full_half = full_rounds // 2
    total_rounds = full_rounds + partial_rounds
    assert total_rounds == len(round_constants)

    s = list(state)
    round_idx = 0
    for _ in range(full_half):
        rc = round_constants[round_idx]
        round_idx += 1
        for i in range(width):
            s[i] = (s[i] + rc[i]) % modulus
            s[i] = sbox(s[i], power=sbox_power, modulus=modulus)
        s = mds_mix(s, modulus=modulus)

    for _ in range(partial_rounds):
        rc = round_constants[round_idx]
        round_idx += 1
        for i in range(width):
            s[i] = (s[i] + rc[i]) % modulus
        s[0] = sbox(s[0], power=sbox_power, modulus=modulus)
        s = mds_mix(s, modulus=modulus)

    for _ in range(full_half):
        rc = round_constants[round_idx]
        round_idx += 1
        for i in range(width):
            s[i] = (s[i] + rc[i]) % modulus
            s[i] = sbox(s[i], power=sbox_power, modulus=modulus)
        s = mds_mix(s, modulus=modulus)

    return s


def random_state(rng: random.Random, *, width: int, modulus: int) -> list[int]:
    return [rng.randrange(modulus) for _ in range(width)]


def vector_add_mod(a: list[int], b: list[int], modulus: int) -> list[int]:
    return [int((x + y) % modulus) for x, y in zip(a, b)]


def differential_scan(settings: dict[str, Any]) -> dict[str, Any]:
    challenge = settings["challenge"]
    search = settings["search"]
    modulus = int(challenge["field_modulus"])
    width = int(challenge["width"])
    total_rounds = int(challenge["full_rounds"] + challenge["partial_rounds"])
    lane = int(challenge["target_lane"])

    round_constants = build_round_constants(
        rounds=total_rounds,
        width=width,
        modulus=modulus,
        seed=int(challenge["round_constants_seed"]),
    )
    rng = random.Random(int(search["seed"]) + 11)

    best_bits = float("inf")
    best_prob = 0.0
    best_delta: list[int] = [0] * width

    delta_candidates = int(search["delta_candidates"])
    samples_per_delta = int(search["samples_per_delta"])
    uniform_prob = 1.0 / float(modulus)

    for _ in range(delta_candidates):
        delta = random_state(rng, width=width, modulus=modulus)
        if all(v == 0 for v in delta):
            delta[0] = 1
        counts: dict[int, int] = {}
        for _ in range(samples_per_delta):
            x = random_state(rng, width=width, modulus=modulus)
            x2 = vector_add_mod(x, delta, modulus)
            y1 = permute(
                x,
                modulus=modulus,
                full_rounds=int(challenge["full_rounds"]),
                partial_rounds=int(challenge["partial_rounds"]),
                sbox_power=int(challenge["sbox_power"]),
                round_constants=round_constants,
            )
            y2 = permute(
                x2,
                modulus=modulus,
                full_rounds=int(challenge["full_rounds"]),
                partial_rounds=int(challenge["partial_rounds"]),
                sbox_power=int(challenge["sbox_power"]),
                round_constants=round_constants,
            )
            out_delta = (y2[lane] - y1[lane]) % modulus
            counts[out_delta] = counts.get(out_delta, 0) + 1

        max_count = max(counts.values()) if counts else 0
        max_prob = float(max_count) / float(samples_per_delta)
        bits = -math.log2(max(max_prob, 1e-12))

        if bits < best_bits:
            best_bits = bits
            best_prob = max_prob
            best_delta = delta

    return {
        "best_probability": best_prob,
        "complexity_bits": best_bits,
        "best_input_delta": best_delta,
        "uniform_probability": uniform_prob,
        "bias_over_uniform": best_prob - uniform_prob,
        "delta_candidates": delta_candidates,
        "samples_per_delta": samples_per_delta,
    }


def preimage_scan(settings: dict[str, Any]) -> dict[str, Any]:
    challenge = settings["challenge"]
    search = settings["search"]
    modulus = int(challenge["field_modulus"])
    width = int(challenge["width"])
    target_bits = int(challenge["preimage_target_bits"])
    mask = (1 << target_bits) - 1
    total_rounds = int(challenge["full_rounds"] + challenge["partial_rounds"])
    lane = int(challenge["target_lane"])

    round_constants = build_round_constants(
        rounds=total_rounds,
        width=width,
        modulus=modulus,
        seed=int(challenge["round_constants_seed"]),
    )
    rng = random.Random(int(search["seed"]) + 97)
    attempts_cap = int(search["preimage_attempts_per_trial"])
    trials = int(search["preimage_trials"])

    attempts_on_success: list[int] = []
    successful = 0
    attempted = 0

    for _ in range(trials):
        target_state = random_state(rng, width=width, modulus=modulus)
        target_value = permute(
            target_state,
            modulus=modulus,
            full_rounds=int(challenge["full_rounds"]),
            partial_rounds=int(challenge["partial_rounds"]),
            sbox_power=int(challenge["sbox_power"]),
            round_constants=round_constants,
        )[lane] & mask

        found = False
        for attempt in range(1, attempts_cap + 1):
            attempted += 1
            guess = random_state(rng, width=width, modulus=modulus)
            guess_value = permute(
                guess,
                modulus=modulus,
                full_rounds=int(challenge["full_rounds"]),
                partial_rounds=int(challenge["partial_rounds"]),
                sbox_power=int(challenge["sbox_power"]),
                round_constants=round_constants,
            )[lane] & mask
            if guess_value == target_value:
                successful += 1
                attempts_on_success.append(attempt)
                found = True
                break
        if not found:
            continue

    success_rate = float(successful) / float(trials)
    if attempts_on_success:
        med_attempts = float(sorted(attempts_on_success)[len(attempts_on_success) // 2])
    else:
        med_attempts = float(attempts_cap + 1)
    complexity_bits = math.log2(max(2.0, med_attempts))

    return {
        "target_bits": target_bits,
        "attempts_cap": attempts_cap,
        "trials": trials,
        "successful_trials": successful,
        "success_rate": success_rate,
        "attempted_guesses": attempted,
        "median_attempts_on_success": med_attempts,
        "complexity_bits": complexity_bits,
    }


def score(settings: dict[str, Any], *, differential: dict[str, Any], preimage: dict[str, Any]) -> dict[str, float]:
    objective = settings["objective"]
    target_bits = float(settings["challenge"]["preimage_target_bits"])

    wd = float(objective["weight_differential"])
    wp = float(objective["weight_preimage"])
    total_weight = wd + wp
    if total_weight <= 0.0:
        wd = 0.7
        wp = 0.3
        total_weight = 1.0
    wd /= total_weight
    wp /= total_weight

    diff_bits = float(differential["complexity_bits"])
    pre_bits = float(preimage["complexity_bits"])
    cap_bits = float(objective["complexity_cap_bits"])
    threshold_bits = float(objective["attack_found_threshold_bits"])
    bonus = float(objective["attack_found_bonus"])
    cost_penalty_scale = float(objective["cost_penalty_scale"])

    diff_signal = max(0.0, cap_bits - diff_bits)
    pre_signal = max(0.0, target_bits - pre_bits)
    attack_found = 1 if (diff_bits <= threshold_bits or float(preimage["success_rate"]) >= 0.8) else 0

    search = settings["search"]
    search_cost = (
        int(search["delta_candidates"]) * int(search["samples_per_delta"])
        + int(search["preimage_attempts_per_trial"]) * int(search["preimage_trials"])
    )
    cost_penalty = cost_penalty_scale * math.log2(max(2, search_cost))

    attack_score = (wd * diff_signal) + (wp * pre_signal) + (bonus * attack_found) - cost_penalty

    return {
        "attack_score": attack_score,
        "differential_signal": diff_signal,
        "preimage_signal": pre_signal,
        "cost_penalty": cost_penalty,
        "attack_found": float(attack_found),
        "best_attack_complexity_bits": min(diff_bits, pre_bits),
        "differential_complexity_bits": diff_bits,
        "preimage_complexity_bits": pre_bits,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deterministic reduced-round Poseidon cryptanalysis harness")
    parser.add_argument(
        "--config",
        default="config/track_b_attack_config.json",
        help="Path to Track B config JSON",
    )
    parser.add_argument("--mode", choices=["fast", "full"], default="fast", help="Search budget mode")
    parser.add_argument(
        "--output-format",
        choices=["json", "pretty"],
        default="json",
        help="Output encoding",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_config(config_path)
    settings = resolve_settings(config, mode=args.mode)

    differential = differential_scan(settings)
    preimage = preimage_scan(settings)
    metrics = score(settings, differential=differential, preimage=preimage)

    payload = {
        "ok": True,
        "mode": args.mode,
        "config_path": str(config_path),
        "settings": settings,
        "metrics": {
            "attack_score": metrics["attack_score"],
            "best_attack_complexity_bits": metrics["best_attack_complexity_bits"],
            "differential_complexity_bits": metrics["differential_complexity_bits"],
            "preimage_complexity_bits": metrics["preimage_complexity_bits"],
            "attack_found": int(metrics["attack_found"]),
        },
        "details": {
            "differential": differential,
            "preimage": preimage,
            "signals": {
                "differential_signal": metrics["differential_signal"],
                "preimage_signal": metrics["preimage_signal"],
                "cost_penalty": metrics["cost_penalty"],
            },
        },
        "repro": {
            "command": f"python3 attack_harness.py --config {config_path} --mode {args.mode} --output-format json"
        },
    }

    if args.output_format == "pretty":
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
