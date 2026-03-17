from __future__ import annotations

import unittest
from pathlib import Path

import train


ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = ROOT / "attack_harness.py"


def harness_source() -> str:
    return HARNESS_PATH.read_text(encoding="utf-8")


class PythonMutationSelectionTests(unittest.TestCase):
    def test_priority_schedule_uses_first_available_operator(self) -> None:
        source = harness_source()
        candidate, mutation, changed = train.python_heuristic_candidate(
            source,
            1,
            HARNESS_PATH,
        )
        self.assertTrue(changed)
        self.assertNotEqual(candidate, source)
        self.assertEqual(mutation, "python_trackb_diff_multi_delta_prob_up")

    def test_blocked_operator_falls_back_to_next_candidate(self) -> None:
        source = harness_source()
        candidate, mutation, changed = train.python_heuristic_candidate(
            source,
            1,
            HARNESS_PATH,
            blocked_mutations={"python_trackb_diff_multi_delta_prob_up"},
        )
        self.assertTrue(changed)
        self.assertNotEqual(candidate, source)
        self.assertEqual(mutation, "python_trackb_diff_multi_delta_prob_down")

    def test_ucb_schedule_prefers_known_successful_operator(self) -> None:
        source = harness_source()
        memory = {
            "version": 1,
            "mutations": {
                "python_trackb_mitm_bucket_cap_up": {
                    "accepted_total": 9,
                    "rejected_total": 1,
                    "languages": {"python": {"accepted": 9, "rejected": 1}},
                    "targets": {
                        "poseidon2_cryptanalysis_trackb_kernel_fast": {
                            "accepted": 9,
                            "rejected": 1,
                        }
                    },
                }
            },
        }
        candidate, mutation, changed = train.python_heuristic_candidate(
            source,
            1,
            HARNESS_PATH,
            target_config={"mutation_schedule": "ucb", "mutation_ucb_explore": 0.0},
            mutation_memory=memory,
            target_name="poseidon2_cryptanalysis_trackb_kernel_fast",
        )
        self.assertTrue(changed)
        self.assertNotEqual(candidate, source)
        self.assertEqual(mutation, "python_trackb_mitm_bucket_cap_up")

    def test_compound_schedule_prefers_compound_mutation_on_trigger_iteration(self) -> None:
        source = harness_source()
        candidate, mutation, changed = train.python_heuristic_candidate(
            source,
            2,
            HARNESS_PATH,
            target_config={
                "compound_every": 2,
                "compound_limit": 6,
                "compound_second_window": 6,
                "mutation_schedule": "priority",
            },
        )
        self.assertTrue(changed)
        self.assertNotEqual(candidate, source)
        self.assertIn("+", mutation)

    def test_mutation_language_inference_supports_python_prefix(self) -> None:
        language = train.infer_mutation_language(mutation="python_trackb_mitm_bucket_cap_up")
        self.assertEqual(language, "python")


if __name__ == "__main__":
    unittest.main()
