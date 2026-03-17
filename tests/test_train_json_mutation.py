from __future__ import annotations

import json
import unittest
from pathlib import Path

import train


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "track_b_attack_config.json"
TRACKB_MUTABLE_PATH = ROOT / "config" / "track_b_mutable_fast.json"


def stable_trackb_source() -> str:
    payload = json.loads(TRACKB_MUTABLE_PATH.read_text(encoding="utf-8"))
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def stable_trackb_base_source() -> str:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


class JsonMutationSelectionTests(unittest.TestCase):
    def test_priority_schedule_uses_first_available_operator(self) -> None:
        source = stable_trackb_source()
        candidate, mutation, changed = train.json_heuristic_candidate(
            source,
            1,
            TRACKB_MUTABLE_PATH,
        )
        self.assertTrue(changed)
        self.assertNotEqual(candidate, source)
        self.assertEqual(mutation, "json_trackb_diff_candidates_up")

    def test_blocked_operator_falls_back_to_next_candidate(self) -> None:
        source = stable_trackb_source()
        candidate, mutation, changed = train.json_heuristic_candidate(
            source,
            1,
            TRACKB_MUTABLE_PATH,
            blocked_mutations={"json_trackb_diff_candidates_up"},
        )
        self.assertTrue(changed)
        self.assertNotEqual(candidate, source)
        self.assertEqual(mutation, "json_trackb_diff_candidates_down")

    def test_operator_penalty_deprioritizes_first_candidate(self) -> None:
        source = stable_trackb_source()
        candidate, mutation, changed = train.json_heuristic_candidate(
            source,
            1,
            TRACKB_MUTABLE_PATH,
            operator_penalties={"json_trackb_diff_candidates_up": 0.3},
        )
        self.assertTrue(changed)
        self.assertNotEqual(candidate, source)
        self.assertEqual(mutation, "json_trackb_diff_candidates_down")

    def test_ucb_schedule_prefers_known_successful_operator(self) -> None:
        source = stable_trackb_source()
        memory = {
            "version": 1,
            "mutations": {
                "json_trackb_split_round_up": {
                    "accepted_total": 12,
                    "rejected_total": 1,
                    "languages": {"json": {"accepted": 12, "rejected": 1}},
                    "targets": {
                        "poseidon2_cryptanalysis_trackb_fast": {
                            "accepted": 12,
                            "rejected": 1,
                        }
                    },
                },
                "json_trackb_diff_candidates_up": {
                    "accepted_total": 0,
                    "rejected_total": 7,
                    "languages": {"json": {"accepted": 0, "rejected": 7}},
                    "targets": {
                        "poseidon2_cryptanalysis_trackb_fast": {
                            "accepted": 0,
                            "rejected": 7,
                        }
                    },
                },
            },
        }
        candidate, mutation, changed = train.json_heuristic_candidate(
            source,
            1,
            TRACKB_MUTABLE_PATH,
            target_config={"mutation_schedule": "ucb", "mutation_ucb_explore": 0.0},
            mutation_memory=memory,
            target_name="poseidon2_cryptanalysis_trackb_fast",
        )
        self.assertTrue(changed)
        self.assertNotEqual(candidate, source)
        self.assertEqual(mutation, "json_trackb_split_round_up")

    def test_preferred_mutations_accepts_json_prefix_fallback(self) -> None:
        memory = {
            "version": 1,
            "mutations": {
                "json_trackb_algebraic_degree_down": {
                    "accepted_total": 3,
                    "rejected_total": 1,
                    "targets": {"other_trackb": {"accepted": 3, "rejected": 1}},
                }
            },
        }
        preferred = train.preferred_mutations_from_memory(
            memory,
            target_name="poseidon2_cryptanalysis_algebraic_fast",
            language="json",
            limit=5,
            min_accepted_total=1,
            min_success_rate=0.0,
        )
        self.assertIn("json_trackb_algebraic_degree_down", preferred)


class TrackBObjectiveGuardTests(unittest.TestCase):
    def test_parse_flag_bool_fails_closed_for_unknown_values(self) -> None:
        self.assertFalse(train.parse_flag_bool({}, default=False))
        self.assertFalse(train.parse_flag_bool([False], default=False))
        self.assertFalse(train.parse_flag_bool("disabled", default=False))
        self.assertTrue(train.parse_flag_bool("disabled", default=True))
        self.assertTrue(train.parse_flag_bool(1.0, default=False))
        self.assertFalse(train.parse_flag_bool(0.0, default=True))

    def test_guard_allows_search_only_change(self) -> None:
        source = stable_trackb_base_source()
        payload = json.loads(source)
        payload["search"]["differential_candidates"] = int(payload["search"]["differential_candidates"]) + 8
        candidate = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        ok, details = train.trackb_objective_guard(
            current_source=source,
            candidate_source=candidate,
            source_path=CONFIG_PATH,
            target_config={},
        )
        self.assertTrue(ok)
        self.assertEqual(details.get("status"), "ok")

    def test_guard_rejects_objective_change(self) -> None:
        source = stable_trackb_base_source()
        payload = json.loads(source)
        payload["objective"]["weight_algebraic"] = round(float(payload["objective"]["weight_algebraic"]) + 0.05, 6)
        candidate = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        ok, details = train.trackb_objective_guard(
            current_source=source,
            candidate_source=candidate,
            source_path=CONFIG_PATH,
            target_config={},
        )
        self.assertFalse(ok)
        self.assertEqual(details.get("status"), "objective_modified")
        self.assertIn("objective", details.get("paths", []))
        self.assertNotIn("resolved_objective", details.get("paths", []))

    def test_guard_rejects_nested_profile_objective_change(self) -> None:
        source = stable_trackb_base_source()
        payload = json.loads(source)
        profiles = payload.setdefault("challenge_profiles", {})
        profiles["guard_a"] = {"objective": {"weight_algebraic": 0.15}}
        profiles["guard_b"] = {"objective": {"weight_algebraic": 0.35}}
        payload["active_profile"] = "guard_a"
        source = json.dumps(payload, indent=2, sort_keys=True) + "\n"

        payload["active_profile"] = "guard_b"
        candidate = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        ok, details = train.trackb_objective_guard(
            current_source=source,
            candidate_source=candidate,
            source_path=CONFIG_PATH,
            target_config={},
        )
        self.assertFalse(ok)
        self.assertEqual(details.get("status"), "objective_modified")
        self.assertIn("resolved_objective", details.get("paths", []))

    def test_guard_does_not_duplicate_active_profile_objective_change(self) -> None:
        source = stable_trackb_base_source()
        payload = json.loads(source)
        profiles = payload.setdefault("challenge_profiles", {})
        profiles["guard_active"] = {"objective": {"weight_algebraic": 0.15}}
        payload["active_profile"] = "guard_active"
        source = json.dumps(payload, indent=2, sort_keys=True) + "\n"

        profiles["guard_active"]["objective"]["weight_algebraic"] = 0.35
        candidate = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        ok, details = train.trackb_objective_guard(
            current_source=source,
            candidate_source=candidate,
            source_path=CONFIG_PATH,
            target_config={},
        )
        self.assertFalse(ok)
        self.assertEqual(details.get("status"), "objective_modified")
        self.assertIn("challenge_profiles.guard_active.objective", details.get("paths", []))
        self.assertNotIn("resolved_objective", details.get("paths", []))

    def test_guard_allows_objective_change_with_opt_in(self) -> None:
        source = stable_trackb_base_source()
        payload = json.loads(source)
        payload["objective"]["weight_algebraic"] = round(float(payload["objective"]["weight_algebraic"]) + 0.05, 6)
        candidate = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        ok, details = train.trackb_objective_guard(
            current_source=source,
            candidate_source=candidate,
            source_path=CONFIG_PATH,
            target_config={"json_allow_objective_mutations": True},
        )
        self.assertTrue(ok)
        self.assertEqual(details.get("allow_objective_mutations"), True)

    def test_guard_rejects_unparseable_baseline(self) -> None:
        ok, details = train.trackb_objective_guard(
            current_source="{",
            candidate_source=stable_trackb_base_source(),
            source_path=CONFIG_PATH,
            target_config={},
        )
        self.assertFalse(ok)
        self.assertEqual(details.get("status"), "baseline_parse_failed")

    def test_guard_rejects_unparseable_candidate(self) -> None:
        ok, details = train.trackb_objective_guard(
            current_source=stable_trackb_base_source(),
            candidate_source="{",
            source_path=CONFIG_PATH,
            target_config={},
        )
        self.assertFalse(ok)
        self.assertEqual(details.get("status"), "candidate_parse_failed")

    def test_guard_is_disabled_for_non_trackb_path(self) -> None:
        ok, details = train.trackb_objective_guard(
            current_source=stable_trackb_base_source(),
            candidate_source=stable_trackb_base_source(),
            source_path=Path("/tmp/other.json"),
            target_config={},
        )
        self.assertTrue(ok)
        self.assertEqual(details, {"enabled": False})

    def test_base_config_path_is_anchored_to_canonical_config_location(self) -> None:
        self.assertTrue(train.is_trackb_base_config_path(CONFIG_PATH))
        self.assertFalse(train.is_trackb_base_config_path(Path("/tmp/track_b_attack_config.json")))

    def test_mutable_config_path_is_anchored_to_canonical_config_directory(self) -> None:
        self.assertTrue(train.is_trackb_mutable_config_path(TRACKB_MUTABLE_PATH))
        self.assertFalse(train.is_trackb_mutable_config_path(Path("/tmp/track_b_mutable_fast.json")))

    def test_base_config_mutation_guard_blocks_by_default(self) -> None:
        ok, details = train.trackb_base_config_mutation_guard(
            source_path=CONFIG_PATH,
            target_config={},
        )
        self.assertFalse(ok)
        self.assertEqual(details.get("status"), "base_config_mutation_blocked")

    def test_base_config_mutation_guard_allows_opt_in(self) -> None:
        ok, details = train.trackb_base_config_mutation_guard(
            source_path=CONFIG_PATH,
            target_config={"allow_self_optimizing_config": True},
        )
        self.assertTrue(ok)
        self.assertEqual(details.get("allow_self_optimizing_config"), True)


if __name__ == "__main__":
    unittest.main()
