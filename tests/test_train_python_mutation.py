from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

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

    def test_algorithmic_diff_structure_mutator_applies(self) -> None:
        source = harness_source()
        candidate, mutation, changed = train.python_mutator_diff_secondary_lane_structure(source)
        self.assertTrue(changed)
        self.assertNotEqual(candidate, source)
        self.assertEqual(mutation, "python_trackb_diff_secondary_lane_structure")

    def test_operator_stats_demote_then_disable_on_no_signal_streak(self) -> None:
        stats = {"version": 1, "targets": {}}
        for _ in range(3):
            train.update_operator_stats(
                stats,
                target_name="poseidon2_cryptanalysis_trackb_kernel_fast",
                mutation="python_trackb_diff_multi_delta_prob_up",
                language="python",
                accepted=False,
                reward=0.0,
                runtime_s=1.0,
                timestamp="2026-03-17T00:00:00+00:00",
                reward_epsilon=1e-12,
                demote_streak=2,
                disable_streak=3,
            )
        _, penalties, rows = train.compute_operator_state(
            stats,
            target_name="poseidon2_cryptanalysis_trackb_kernel_fast",
            language="python",
            demote_streak=2,
            disable_streak=3,
        )
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["demoted"])
        self.assertTrue(rows[0]["disabled"])
        self.assertNotIn("python_trackb_diff_multi_delta_prob_up", penalties)

    def test_mutator_penalty_for_compound_uses_part_penalty(self) -> None:
        penalties = {"python_trackb_a": 0.05, "python_trackb_b": 0.11}
        score = train.mutator_penalty_for_label("python_trackb_a+python_trackb_b", penalties)
        self.assertAlmostEqual(score, 0.11, places=6)

    def test_validation_metric_passes_allows_equal_for_higher(self) -> None:
        ok, details = train.validation_metric_passes(
            candidate_value=10.0,
            baseline_value=10.0,
            higher_is_better=True,
            allow_drop_abs=0.0,
            allow_drop_rel=0.0,
        )
        self.assertTrue(ok)
        self.assertTrue(details["abs_ok"])
        self.assertTrue(details["rel_ok"])

    def test_validation_metric_passes_blocks_drop_without_tolerance(self) -> None:
        ok, _ = train.validation_metric_passes(
            candidate_value=9.9,
            baseline_value=10.0,
            higher_is_better=True,
            allow_drop_abs=0.0,
            allow_drop_rel=0.0,
        )
        self.assertFalse(ok)

    def test_validation_metric_passes_allows_drop_with_abs_tolerance(self) -> None:
        ok, _ = train.validation_metric_passes(
            candidate_value=9.95,
            baseline_value=10.0,
            higher_is_better=True,
            allow_drop_abs=0.1,
            allow_drop_rel=0.0,
        )
        self.assertTrue(ok)

    def test_validation_metric_passes_lower_is_better(self) -> None:
        ok, _ = train.validation_metric_passes(
            candidate_value=5.0,
            baseline_value=5.0,
            higher_is_better=False,
            allow_drop_abs=0.0,
            allow_drop_rel=0.0,
        )
        self.assertTrue(ok)
        ok2, _ = train.validation_metric_passes(
            candidate_value=5.2,
            baseline_value=5.0,
            higher_is_better=False,
            allow_drop_abs=0.05,
            allow_drop_rel=0.0,
        )
        self.assertFalse(ok2)

    def test_evaluate_validation_targets_success(self) -> None:
        with patch(
            "train.prepare.evaluate_target",
            return_value={"status": "success", "metric_value": 10.0, "metric_name": "attack_score"},
        ):
            ok, details, metrics, reason = train.evaluate_validation_targets(
                validation_targets=["v_lane"],
                validation_baselines={"v_lane": 10.0},
                targets_catalog={"v_lane": {"higher_is_better": True}},
                allow_drop_abs=0.0,
                allow_drop_rel=0.0,
            )
        self.assertTrue(ok)
        self.assertEqual(reason, "ok")
        self.assertEqual(metrics["v_lane"], 10.0)
        self.assertEqual(details["v_lane"]["status"], "ok")

    def test_evaluate_validation_targets_rejects_degradation(self) -> None:
        with patch(
            "train.prepare.evaluate_target",
            return_value={"status": "success", "metric_value": 9.0, "metric_name": "attack_score"},
        ):
            ok, details, metrics, reason = train.evaluate_validation_targets(
                validation_targets=["v_lane"],
                validation_baselines={"v_lane": 10.0},
                targets_catalog={"v_lane": {"higher_is_better": True}},
                allow_drop_abs=0.0,
                allow_drop_rel=0.0,
            )
        self.assertFalse(ok)
        self.assertEqual(reason, "degraded:v_lane")
        self.assertEqual(metrics["v_lane"], 9.0)
        self.assertEqual(details["v_lane"]["status"], "rejected")

    def test_algorithmic_mitm_key_mutator_applies(self) -> None:
        source = harness_source()
        candidate, mutation, changed = train.python_mutator_mitm_augmented_middle_key(source)
        self.assertTrue(changed)
        self.assertNotEqual(candidate, source)
        self.assertEqual(mutation, "python_trackb_mitm_augmented_middle_key")

    def test_algorithmic_algebraic_sampling_mutator_applies(self) -> None:
        source = harness_source()
        candidate, mutation, changed = train.python_mutator_algebraic_structured_unknown_samples(source)
        self.assertTrue(changed)
        self.assertNotEqual(candidate, source)
        self.assertEqual(mutation, "python_trackb_algebraic_structured_unknown_samples")

    def test_priority_can_force_algorithmic_mutator_when_simple_ones_blocked(self) -> None:
        source = harness_source()
        blocked = {
            "python_trackb_diff_multi_delta_prob_up",
            "python_trackb_diff_multi_delta_prob_down",
            "python_trackb_diff_cross_lane_prob_up",
            "python_trackb_diff_cross_lane_prob_down",
            "python_trackb_mitm_bucket_cap_up",
            "python_trackb_mitm_bucket_cap_down",
            "python_trackb_algebraic_fit_gain_up",
            "python_trackb_algebraic_fit_gain_down",
        }
        candidate, mutation, changed = train.python_heuristic_candidate(
            source,
            1,
            HARNESS_PATH,
            blocked_mutations=blocked,
            target_config={"mutation_schedule": "priority"},
        )
        self.assertTrue(changed)
        self.assertNotEqual(candidate, source)
        self.assertEqual(mutation, "python_trackb_diff_secondary_lane_structure")


if __name__ == "__main__":
    unittest.main()
