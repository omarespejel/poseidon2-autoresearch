from __future__ import annotations

import json
import math
import random
import unittest
from pathlib import Path

import attack_harness


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "track_b_attack_config.json"


def tiny_config() -> dict:
    try:
        cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Track B config not found: {CONFIG_PATH}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Track B config is invalid JSON: {CONFIG_PATH}") from exc
    cfg["search"] = {
        "seed": 1337,
        "differential_candidates": 6,
        "differential_samples_per_candidate": 24,
        "mitm_forward_states": 128,
        "mitm_backward_states": 128,
        "collision_samples": 256,
        "scale_rounds_in_full_mode": 0,
    }
    cfg["analysis"] = {
        "target_lane": 0,
        "truncated_bits": 14,
        "split_round": 9,
        "middle_key_bits": 10,
        "middle_key_lanes": [0],
    }
    return cfg


def seeded_rng(seed: int) -> random.Random:
    return random.Random(seed)  # noqa: S311 - deterministic test fixture, not cryptographic use.


class AttackHarnessTests(unittest.TestCase):
    def test_poseidon2_inverse_roundtrip(self) -> None:
        cfg = tiny_config()
        spec = attack_harness.build_spec(cfg, mode="fast")
        boundary_states = [
            [0] * spec.width,
            [spec.modulus - 1] * spec.width,
            [1] + ([0] * (spec.width - 1)),
        ]
        for lane in range(1, spec.width):
            state = [0] * spec.width
            state[lane] = lane + 1
            boundary_states.append(state)

        for state in boundary_states:
            y = attack_harness.poseidon2_permute(state, spec)
            restored = attack_harness.poseidon2_invert_to_prefix(y, spec, 0)
            self.assertEqual(state, restored)

        prefix_rounds = min(3, spec.total_rounds - 1)
        probe = [idx + 2 for idx in range(spec.width)]
        y_probe = attack_harness.poseidon2_permute(probe, spec)
        prefix_state = attack_harness.poseidon2_invert_to_prefix(y_probe, spec, prefix_rounds)
        self.assertEqual(
            attack_harness.poseidon2_prefix(probe, spec, prefix_rounds),
            prefix_state,
        )

        num_iter = 12
        rng = seeded_rng(12345)
        for _ in range(num_iter):
            x = attack_harness.random_state(rng, spec)
            y = attack_harness.poseidon2_permute(x, spec)
            x_restored = attack_harness.poseidon2_invert_to_prefix(y, spec, 0)
            self.assertEqual(x, x_restored)

    def test_build_spec_handles_malformed_config_and_full_mode_scaling(self) -> None:
        default_spec = attack_harness.build_spec({}, mode="fast")
        malformed_spec = attack_harness.build_spec({"poseidon2": "bad", "search": "bad"}, mode="fast")
        self.assertEqual(default_spec.width, malformed_spec.width)
        self.assertEqual(default_spec.partial_rounds, malformed_spec.partial_rounds)

        composite_modulus_cfg = tiny_config()
        composite_modulus_cfg["poseidon2"]["field_modulus"] = 1_000_005
        with self.assertRaises(ValueError):
            attack_harness.build_spec(composite_modulus_cfg, mode="fast")

        cfg = tiny_config()
        cfg["search"]["scale_rounds_in_full_mode"] = 1
        fast_spec = attack_harness.build_spec(cfg, mode="fast")
        full_spec = attack_harness.build_spec(cfg, mode="full")
        fast_search = attack_harness.parse_search(cfg, mode="fast")
        full_search = attack_harness.parse_search(cfg, mode="full")

        self.assertGreater(full_spec.partial_rounds, fast_spec.partial_rounds)
        self.assertGreater(full_search["differential_candidates"], fast_search["differential_candidates"])
        self.assertGreater(full_search["collision_samples"], fast_search["collision_samples"])

    def test_kernels_are_deterministic_with_fixed_seed(self) -> None:
        cfg = tiny_config()
        spec = attack_harness.build_spec(cfg, mode="fast")
        search = attack_harness.parse_search(cfg, mode="fast")
        analysis = attack_harness.parse_analysis(cfg, spec)

        rng_a = seeded_rng(int(search["seed"]))
        d_a = attack_harness.differential_kernel(spec=spec, analysis=analysis, search=search, rng=rng_a)
        p_a = attack_harness.mitm_truncated_preimage_kernel(spec=spec, analysis=analysis, search=search, rng=rng_a)
        c_a = attack_harness.birthday_collision_kernel(spec=spec, analysis=analysis, search=search, rng=rng_a)
        a_a = attack_harness.algebraic_elimination_kernel(spec=spec, analysis=analysis, search=search, rng=rng_a)
        m_a = attack_harness.score(
            config=cfg,
            search=search,
            differential=d_a,
            mitm_preimage=p_a,
            collision=c_a,
            algebraic=a_a,
        )

        rng_b = seeded_rng(int(search["seed"]))
        d_b = attack_harness.differential_kernel(spec=spec, analysis=analysis, search=search, rng=rng_b)
        p_b = attack_harness.mitm_truncated_preimage_kernel(spec=spec, analysis=analysis, search=search, rng=rng_b)
        c_b = attack_harness.birthday_collision_kernel(spec=spec, analysis=analysis, search=search, rng=rng_b)
        a_b = attack_harness.algebraic_elimination_kernel(spec=spec, analysis=analysis, search=search, rng=rng_b)
        m_b = attack_harness.score(
            config=cfg,
            search=search,
            differential=d_b,
            mitm_preimage=p_b,
            collision=c_b,
            algebraic=a_b,
        )

        self.assertEqual(rng_a.getstate(), rng_b.getstate())
        self.assertEqual(d_a, d_b)
        self.assertEqual(p_a, p_b)
        self.assertEqual(c_a, c_b)
        self.assertEqual(a_a, a_b)
        self.assertEqual(m_a, m_b)

    def test_payload_shape(self) -> None:
        cfg = tiny_config()
        spec = attack_harness.build_spec(cfg, mode="fast")
        search = attack_harness.parse_search(cfg, mode="fast")
        analysis = attack_harness.parse_analysis(cfg, spec)
        rng = seeded_rng(int(search["seed"]))

        d = attack_harness.differential_kernel(spec=spec, analysis=analysis, search=search, rng=rng)
        p = attack_harness.mitm_truncated_preimage_kernel(spec=spec, analysis=analysis, search=search, rng=rng)
        c = attack_harness.birthday_collision_kernel(spec=spec, analysis=analysis, search=search, rng=rng)
        a = attack_harness.algebraic_elimination_kernel(spec=spec, analysis=analysis, search=search, rng=rng)
        m = attack_harness.score(config=cfg, search=search, differential=d, mitm_preimage=p, collision=c, algebraic=a)

        self.assertIn("attack_score", m)
        self.assertIn("attack_score_signal", m)
        self.assertIn("attack_score_verified", m)
        self.assertIn("attack_score_algebraic", m)
        self.assertIn("differential_complexity_bits", m)
        self.assertIn("preimage_complexity_bits", m)
        self.assertIn("collision_complexity_bits", m)
        self.assertIn("algebraic_complexity_bits", m)
        self.assertTrue(math_is_finite(float(m["attack_score"])))
        self.assertIn("attack_found", m)
        self.assertIn(m["attack_found"], (0.0, 1.0))
        self.assertGreaterEqual(float(m["differential_complexity_bits"]), 0.0)
        self.assertGreaterEqual(float(m["preimage_complexity_bits"]), 0.0)
        self.assertGreaterEqual(float(m["collision_complexity_bits"]), 0.0)
        self.assertGreaterEqual(float(m["differential_signal"]), 0.0)
        self.assertGreaterEqual(float(m["preimage_signal"]), 0.0)
        self.assertGreaterEqual(float(m["collision_signal"]), 0.0)

    def test_profile_overlay(self) -> None:
        cfg = tiny_config()
        cfg["active_profile"] = "poseidon64_bounty_shape"
        merged, selected = attack_harness.resolve_profile_config(cfg, "")
        self.assertEqual(selected, "poseidon64_bounty_shape")
        self.assertEqual(int(merged["poseidon2"]["field_modulus"]), 18446744069414584321)
        self.assertEqual(int(merged["poseidon2"]["sbox_power"]), 7)
        self.assertEqual(int(merged["analysis"]["truncated_bits"]), 28)

        with self.assertRaises(ValueError):
            attack_harness.resolve_profile_config(tiny_config(), "unknown_profile")

    def test_kernels_handle_zero_budgets(self) -> None:
        cfg = tiny_config()
        spec = attack_harness.build_spec(cfg, mode="fast")
        analysis = attack_harness.parse_analysis(cfg, spec)
        search = {
            "seed": 1337,
            "differential_candidates": 0,
            "differential_samples_per_candidate": 0,
            "mitm_forward_states": 0,
            "mitm_backward_states": 0,
            "collision_samples": 0,
            "algebraic_train_samples": 0,
            "algebraic_validation_samples": 0,
        }
        rng = seeded_rng(search["seed"])

        differential = attack_harness.differential_kernel(spec=spec, analysis=analysis, search=search, rng=rng)
        mitm_preimage = attack_harness.mitm_truncated_preimage_kernel(
            spec=spec,
            analysis=analysis,
            search=search,
            rng=rng,
        )
        collision = attack_harness.birthday_collision_kernel(spec=spec, analysis=analysis, search=search, rng=rng)
        algebraic = attack_harness.algebraic_elimination_kernel(spec=spec, analysis=analysis, search=search, rng=rng)
        metrics = attack_harness.score(
            config=cfg,
            search=search,
            differential=differential,
            mitm_preimage=mitm_preimage,
            collision=collision,
            algebraic=algebraic,
        )

        self.assertEqual(differential["best_probability"], 0.0)
        self.assertTrue(math.isinf(float(differential["complexity_bits"])))
        self.assertEqual(mitm_preimage["matches"], 0)
        self.assertEqual(mitm_preimage["verified_hits"], 0)
        self.assertEqual(collision["samples"], 0)
        self.assertEqual(collision["collisions"], 0)
        self.assertEqual(algebraic["train_samples"], 0)
        self.assertEqual(algebraic["validation_samples"], 0)
        self.assertTrue(math_is_finite(float(metrics["attack_score"])))


def math_is_finite(value: float) -> bool:
    return math.isfinite(value)


if __name__ == "__main__":
    unittest.main()
