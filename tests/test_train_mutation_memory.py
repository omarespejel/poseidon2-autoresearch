from __future__ import annotations

import unittest

import train


def sample_memory() -> dict:
    return {
        "version": 1,
        "mutations": {
            "python_trackb_mitm_bucket_cap_up": {
                "accepted_total": 3,
                "rejected_total": 1,
                "languages": {"python": {"accepted": 3, "rejected": 1}},
                "targets": {
                    "namespace:kernel_fast_research": {"accepted": 3, "rejected": 1},
                },
            }
        },
    }


class MutationMemoryScopeTests(unittest.TestCase):
    def test_resolve_mutation_memory_target_scope_uses_namespace(self) -> None:
        scope, strict = train.resolve_mutation_memory_target_scope(
            "poseidon2_cryptanalysis_trackb_kernel_signal_fast",
            {"mutation_memory_namespace": "kernel_signal_poseidon64"},
        )
        self.assertEqual(scope, "namespace:kernel_signal_poseidon64")
        self.assertTrue(strict)

    def test_preferred_mutations_strict_scope_ignores_other_targets(self) -> None:
        memory = sample_memory()
        strict = train.preferred_mutations_from_memory(
            memory,
            target_name="namespace:kernel_signal_poseidon64",
            language="python",
            strict_target_scope=True,
            limit=5,
            min_accepted_total=1,
            min_success_rate=0.0,
        )
        loose = train.preferred_mutations_from_memory(
            memory,
            target_name="namespace:kernel_signal_poseidon64",
            language="python",
            strict_target_scope=False,
            limit=5,
            min_accepted_total=1,
            min_success_rate=0.0,
        )
        self.assertEqual(strict, [])
        self.assertIn("python_trackb_mitm_bucket_cap_up", loose)

    def test_mutation_memory_counts_strict_scope_returns_unknown_for_other_targets(self) -> None:
        memory = sample_memory()
        accepted, rejected, scope = train.mutation_memory_counts(
            memory,
            "python_trackb_mitm_bucket_cap_up",
            target_name="namespace:kernel_signal_poseidon64",
            language="python",
            strict_target_scope=True,
        )
        self.assertEqual((accepted, rejected, scope), (0.0, 0.0, "global"))

    def test_mutation_memory_scope_totals_strict_scope_only_counts_target_history(self) -> None:
        memory = sample_memory()
        memory["mutations"]["python_trackb_diff_multi_delta_prob_up"] = {
            "accepted_total": 2,
            "rejected_total": 1,
            "languages": {"python": {"accepted": 2, "rejected": 1}},
            "targets": {
                "namespace:kernel_signal_poseidon64": {"accepted": 2, "rejected": 1},
            },
        }
        totals = train.mutation_memory_scope_totals(
            memory,
            target_name="namespace:kernel_signal_poseidon64",
            language="python",
            strict_target_scope=True,
        )
        self.assertEqual(totals["target"], 3.0)
        self.assertEqual(totals["language"], 3.0)
        self.assertEqual(totals["global"], 3.0)


if __name__ == "__main__":
    unittest.main()
