from __future__ import annotations

import unittest
from pathlib import Path

import train


ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = ROOT / "attack_harness.py"


def harness_source() -> str:
    return HARNESS_PATH.read_text(encoding="utf-8")


class PythonMutationSelectionTests(unittest.TestCase):
    def test_required_snippet_guard_uses_python_ast(self) -> None:
        profile = {
            "def differential_kernel(": 1,
            "def mitm_truncated_preimage_kernel(": 1,
            "def algebraic_elimination_kernel(": 1,
            "def score(": 1,
        }
        fake_source = '''"""def differential_kernel(
def mitm_truncated_preimage_kernel(
def algebraic_elimination_kernel(
def score(
"""

def helper():
    return 1
'''
        ok, details = train.required_snippet_guard(fake_source, profile, language="Python")
        self.assertFalse(ok)
        self.assertEqual(len(details["violations"]), 4)

    def test_returns_no_change_for_non_harness_path(self) -> None:
        source = harness_source()
        candidate, mutation, changed = train.python_heuristic_candidate(
            source,
            1,
            Path("/tmp/some_other_file.py"),
        )
        self.assertFalse(changed)
        self.assertEqual(mutation, "python_no_change")
        self.assertEqual(candidate, source)

    def test_returns_no_change_when_no_patterns_match(self) -> None:
        candidate, mutation, changed = train.python_heuristic_candidate(
            "",
            1,
            HARNESS_PATH,
        )
        self.assertFalse(changed)
        self.assertEqual(mutation, "python_no_change")
        self.assertEqual(candidate, "")

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

    def test_diff_probability_mutations_target_distinct_sites(self) -> None:
        source = harness_source()
        multi_candidate, _, multi_changed = train.python_mutator_diff_multi_delta_prob_up(source)
        cross_candidate, _, cross_changed = train.python_mutator_diff_cross_lane_prob_up(source)
        self.assertTrue(multi_changed)
        self.assertTrue(cross_changed)
        self.assertNotEqual(multi_candidate, cross_candidate)

    def test_diff_probability_down_mutations_reverse_up_states(self) -> None:
        source = harness_source()
        multi_up, _, multi_up_changed = train.python_mutator_diff_multi_delta_prob_up(source)
        self.assertTrue(multi_up_changed)
        multi_down, _, multi_down_changed = train.python_mutator_diff_multi_delta_prob_down(multi_up)
        self.assertTrue(multi_down_changed)
        self.assertEqual(multi_down, source)

        cross_up, _, cross_up_changed = train.python_mutator_diff_cross_lane_prob_up(source)
        self.assertTrue(cross_up_changed)
        cross_down, _, cross_down_changed = train.python_mutator_diff_cross_lane_prob_down(cross_up)
        self.assertTrue(cross_down_changed)
        self.assertEqual(cross_down, source)

    def test_mutation_language_inference_supports_python_prefix(self) -> None:
        language = train.infer_mutation_language(mutation="python_trackb_mitm_bucket_cap_up")
        self.assertEqual(language, "python")

    def test_normalize_mutation_label_handles_python_no_change(self) -> None:
        self.assertIsNone(train.normalize_mutation_label("python_no_change"))

    def test_retryable_no_change_includes_python(self) -> None:
        self.assertTrue(train.is_retryable_no_change_mutation("python_no_change"))
        self.assertFalse(train.is_retryable_no_change_mutation("python_trackb_mitm_bucket_cap_up"))


if __name__ == "__main__":
    unittest.main()
