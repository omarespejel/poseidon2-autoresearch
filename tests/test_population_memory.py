from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import train


class PopulationMemoryTests(unittest.TestCase):
    def test_upsert_population_entry_tracks_accept_and_reject(self) -> None:
        memory = {"version": 1, "entries": []}
        source = "def f():\n    return 1\n"
        train.upsert_population_entry(
            memory,
            target_name="poseidon2_cryptanalysis_trackb_kernel_fast",
            language="python",
            source_code=source,
            metric_value=12.5,
            higher_is_better=True,
            accepted=True,
            timestamp="2026-03-17T00:00:00+00:00",
            notes="accepted:seed",
            max_entries=32,
        )
        train.upsert_population_entry(
            memory,
            target_name="poseidon2_cryptanalysis_trackb_kernel_fast",
            language="python",
            source_code=source,
            metric_value=12.2,
            higher_is_better=True,
            accepted=False,
            timestamp="2026-03-17T00:00:01+00:00",
            notes="rejected_not_better:seed",
            max_entries=32,
        )

        entries = memory.get("entries", [])
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual(entry["accepted_total"], 1)
        self.assertEqual(entry["rejected_total"], 1)
        self.assertEqual(entry["metric_value"], 12.5)

    def test_select_population_parent_uses_ranked_candidates(self) -> None:
        best_source = "def f():\n    return 0\n"
        alt_a = "def f():\n    return 1\n"
        alt_b = "def f():\n    return 2\n"
        memory = {
            "version": 1,
            "entries": [
                {
                    "target": "poseidon2_cryptanalysis_trackb_kernel_fast",
                    "language": "python",
                    "source_sha256": train.source_sha256(alt_a),
                    "source_code": alt_a,
                    "metric_value": 16.0,
                    "higher_is_better": True,
                    "accepted_total": 2,
                    "rejected_total": 1,
                    "sampled_total": 0,
                },
                {
                    "target": "poseidon2_cryptanalysis_trackb_kernel_fast",
                    "language": "python",
                    "source_sha256": train.source_sha256(alt_b),
                    "source_code": alt_b,
                    "metric_value": 14.0,
                    "higher_is_better": True,
                    "accepted_total": 0,
                    "rejected_total": 2,
                    "sampled_total": 2,
                },
            ],
        }
        rng = type(
            "DeterministicRng",
            (),
            {"choices": staticmethod(lambda population, weights, k: [0])},
        )()
        with patch("train.random.choices", side_effect=AssertionError("global random should not be used")):
            selected = train.select_population_parent(
                memory,
                target_name="poseidon2_cryptanalysis_trackb_kernel_fast",
                language="python",
                higher_is_better=True,
                best_metric=15.0,
                best_source=best_source,
                max_candidates=8,
                rng=rng,
            )
        self.assertIsNotNone(selected)
        self.assertEqual(selected["source_sha256"], train.source_sha256(alt_a))

    def test_mark_population_entry_sampled_updates_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "population.json"
            source = "def f():\n    return 3\n"
            base = {
                "version": 1,
                "entries": [
                    {
                        "target": "poseidon2_cryptanalysis_trackb_kernel_fast",
                        "language": "python",
                        "source_sha256": train.source_sha256(source),
                        "source_code": source,
                        "metric_value": 15.0,
                        "higher_is_better": True,
                        "accepted_total": 1,
                        "rejected_total": 0,
                        "sampled_total": 0,
                    }
                ],
            }
            train.save_population_memory(path, base)
            memory = train.load_population_memory(path)
            updated = train.mark_population_entry_sampled(
                path,
                memory,
                target_name="poseidon2_cryptanalysis_trackb_kernel_fast",
                language="python",
                source_hash=train.source_sha256(source),
                timestamp="2026-03-17T00:00:00+00:00",
            )
            entries = updated.get("entries", [])
            self.assertEqual(entries[0]["sampled_total"], 1)

    def test_save_mutator_stats_creates_parent_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "mutator_stats.json"
            train.save_mutator_stats(path, {"version": 1, "targets": {}})
            self.assertTrue(path.exists())
            saved = path.read_text(encoding="utf-8")
            self.assertIn('"version": 1', saved)

    def test_population_metric_component_is_bounded_near_zero_baseline(self) -> None:
        bounded = train.population_metric_component(
            metric_value=0.5,
            higher_is_better=True,
            best_metric=0.0,
        )
        self.assertGreater(bounded, 0.0)
        self.assertLessEqual(bounded, 1.0)

        lower_is_better = train.population_metric_component(
            metric_value=0.5,
            higher_is_better=False,
            best_metric=0.0,
        )
        self.assertLess(lower_is_better, 0.0)
        self.assertGreaterEqual(lower_is_better, -1.0)

    def test_population_rng_config_omits_budget_parameters(self) -> None:
        config = train.population_rng_config(
            language="python",
            higher_is_better=True,
            population_parent_sample_prob=0.35,
            population_max_entries=24,
        )
        self.assertEqual(
            config,
            {
                "language": "python",
                "higher_is_better": True,
                "population_parent_sample_prob": 0.35,
                "population_max_entries": 24,
            },
        )
        self.assertNotIn("max_iterations", config)
        self.assertNotIn("max_accepted", config)
        self.assertNotIn("max_runtime_seconds", config)

    def test_should_record_population_candidate(self) -> None:
        self.assertTrue(train.should_record_population_candidate(accepted=True, notes="accepted:x"))
        self.assertTrue(
            train.should_record_population_candidate(
                accepted=False,
                notes="rejected_below_threshold:python_trackb_x",
            )
        )
        self.assertTrue(
            train.should_record_population_candidate(
                accepted=False,
                notes="rejected_validation_degraded:v_lane:python_trackb_x",
            )
        )
        self.assertFalse(
            train.should_record_population_candidate(
                accepted=False,
                notes="rejected_not_better:python_trackb_x",
            )
        )

    def test_load_mutator_stats_handles_oserror(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "mutator-stats.json"
            path.write_text("{}")
            with patch.object(Path, "read_text", side_effect=OSError("boom")):
                stats = train.load_mutator_stats(path)
        self.assertEqual(stats, {"version": 1, "targets": {}})


if __name__ == "__main__":
    unittest.main()
