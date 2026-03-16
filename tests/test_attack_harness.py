from __future__ import annotations

import json
import random
import unittest
from pathlib import Path

import attack_harness


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "track_b_attack_config.json"


def tiny_config() -> dict:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
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


class AttackHarnessTests(unittest.TestCase):
    def test_poseidon2_inverse_roundtrip(self) -> None:
        cfg = tiny_config()
        spec = attack_harness.build_spec(cfg, mode="fast")
        rng = random.Random(12345)
        for _ in range(12):
            x = attack_harness.random_state(rng, spec)
            y = attack_harness.poseidon2_permute(x, spec)
            x_restored = attack_harness.poseidon2_invert_to_prefix(y, spec, 0)
            self.assertEqual(x, x_restored)

    def test_kernels_are_deterministic_with_fixed_seed(self) -> None:
        cfg = tiny_config()
        spec = attack_harness.build_spec(cfg, mode="fast")
        search = attack_harness.parse_search(cfg, mode="fast")
        analysis = attack_harness.parse_analysis(cfg, spec)

        rng_a = random.Random(int(search["seed"]))
        d_a = attack_harness.differential_kernel(spec=spec, analysis=analysis, search=search, rng=rng_a)
        p_a = attack_harness.mitm_truncated_preimage_kernel(spec=spec, analysis=analysis, search=search, rng=rng_a)
        c_a = attack_harness.birthday_collision_kernel(spec=spec, analysis=analysis, search=search, rng=rng_a)
        m_a = attack_harness.score(
            config=cfg,
            search=search,
            differential=d_a,
            mitm_preimage=p_a,
            collision=c_a,
        )

        rng_b = random.Random(int(search["seed"]))
        d_b = attack_harness.differential_kernel(spec=spec, analysis=analysis, search=search, rng=rng_b)
        p_b = attack_harness.mitm_truncated_preimage_kernel(spec=spec, analysis=analysis, search=search, rng=rng_b)
        c_b = attack_harness.birthday_collision_kernel(spec=spec, analysis=analysis, search=search, rng=rng_b)
        m_b = attack_harness.score(
            config=cfg,
            search=search,
            differential=d_b,
            mitm_preimage=p_b,
            collision=c_b,
        )

        self.assertEqual(d_a, d_b)
        self.assertEqual(p_a, p_b)
        self.assertEqual(c_a, c_b)
        self.assertEqual(m_a, m_b)

    def test_payload_shape(self) -> None:
        cfg = tiny_config()
        spec = attack_harness.build_spec(cfg, mode="fast")
        search = attack_harness.parse_search(cfg, mode="fast")
        analysis = attack_harness.parse_analysis(cfg, spec)
        rng = random.Random(int(search["seed"]))

        d = attack_harness.differential_kernel(spec=spec, analysis=analysis, search=search, rng=rng)
        p = attack_harness.mitm_truncated_preimage_kernel(spec=spec, analysis=analysis, search=search, rng=rng)
        c = attack_harness.birthday_collision_kernel(spec=spec, analysis=analysis, search=search, rng=rng)
        m = attack_harness.score(config=cfg, search=search, differential=d, mitm_preimage=p, collision=c)

        self.assertIn("attack_score", m)
        self.assertIn("differential_complexity_bits", m)
        self.assertIn("preimage_complexity_bits", m)
        self.assertIn("collision_complexity_bits", m)
        self.assertTrue(math_is_finite(float(m["attack_score"])))


def math_is_finite(value: float) -> bool:
    try:
        return abs(value) < float("inf")
    except Exception:
        return False


if __name__ == "__main__":
    unittest.main()
