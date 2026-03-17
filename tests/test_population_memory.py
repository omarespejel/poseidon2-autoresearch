from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import train

ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = ROOT / "attack_harness.py"


def harness_source() -> str:
    return HARNESS_PATH.read_text(encoding="utf-8")


class DeterministicRng:
    def __init__(self, *, choices_result: int = 0, choice_result: str = "a", randint_values: list[int] | None = None) -> None:
        self._choices_result = choices_result
        self._choice_result = choice_result
        self._randint_values = list(randint_values or [])

    def choices(self, population: range, weights: list[float], k: int) -> list[int]:
        return [self._choices_result]

    def choice(self, population: list[str]) -> str:
        return self._choice_result

    def randint(self, low: int, high: int) -> int:
        if self._randint_values:
            return self._randint_values.pop(0)
        return low


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

    def test_upsert_population_entry_seed_does_not_increment_rejections(self) -> None:
        memory = {"version": 1, "entries": []}
        source = "def f():\n    return 1\n"
        train.upsert_population_entry(
            memory,
            target_name="poseidon2_cryptanalysis_trackb_kernel_fast",
            language="python",
            source_code=source,
            metric_value=12.5,
            higher_is_better=True,
            accepted=None,
            timestamp="2026-03-17T00:00:00+00:00",
            notes="population_seed:test",
            max_entries=32,
        )
        entry = memory["entries"][0]
        self.assertEqual(entry["accepted_total"], 0)
        self.assertEqual(entry["rejected_total"], 0)
        self.assertEqual(entry["seeded_total"], 1)

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
        rng = DeterministicRng(choices_result=0)
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

    def test_select_population_parent_requires_explicit_rng(self) -> None:
        with self.assertRaises(ValueError):
            train.select_population_parent(
                {"version": 1, "entries": []},
                target_name="poseidon2_cryptanalysis_trackb_kernel_fast",
                language="python",
                higher_is_better=True,
                best_metric=15.0,
                best_source="def f():\n    return 0\n",
                max_candidates=8,
                rng=None,
            )

    def test_select_population_parent_can_replay_cross_target_when_enabled(self) -> None:
        best_source = "def f():\n    return 0\n"
        local_alt = "def f():\n    return 1\n"
        cross_alt = "def f():\n    return 9\n"
        memory = {
            "version": 1,
            "entries": [
                {
                    "target": "poseidon2_cryptanalysis_trackb_kernel_fast",
                    "language": "python",
                    "source_sha256": train.source_sha256(local_alt),
                    "source_code": local_alt,
                    "metric_value": 0.1,
                    "higher_is_better": True,
                    "accepted_total": 0,
                    "rejected_total": 2,
                    "sampled_total": 0,
                },
                {
                    "target": "poseidon2_cryptanalysis_trackb_kernel_signal_fast",
                    "language": "python",
                    "source_sha256": train.source_sha256(cross_alt),
                    "source_code": cross_alt,
                    "metric_value": 99.0,
                    "higher_is_better": True,
                    "accepted_total": 5,
                    "rejected_total": 1,
                    "sampled_total": 0,
                },
            ],
        }
        rng = DeterministicRng(choices_result=0)
        with patch("train.random.choices", side_effect=AssertionError("global random should not be used")):
            selected = train.select_population_parent(
                memory,
                target_name="poseidon2_cryptanalysis_trackb_kernel_fast",
                language="python",
                higher_is_better=True,
                best_metric=15.0,
                best_source=best_source,
                allow_cross_target_replay=True,
                cross_target_min_accepted=2,
                cross_target_score_scale=0.7,
                max_candidates=8,
                rng=rng,
            )
        self.assertIsNotNone(selected)
        self.assertEqual(selected["target"], "poseidon2_cryptanalysis_trackb_kernel_signal_fast")
        self.assertEqual(selected["source_sha256"], train.source_sha256(cross_alt))

    def test_select_population_parent_respects_cross_target_min_accepted(self) -> None:
        best_source = "def f():\n    return 0\n"
        cross_alt = "def f():\n    return 9\n"
        memory = {
            "version": 1,
            "entries": [
                {
                    "target": "poseidon2_cryptanalysis_trackb_kernel_signal_fast",
                    "language": "python",
                    "source_sha256": train.source_sha256(cross_alt),
                    "source_code": cross_alt,
                    "metric_value": 99.0,
                    "higher_is_better": True,
                    "accepted_total": 1,
                    "rejected_total": 0,
                    "sampled_total": 0,
                },
            ],
        }
        selected = train.select_population_parent(
            memory,
            target_name="poseidon2_cryptanalysis_trackb_kernel_fast",
            language="python",
            higher_is_better=True,
            best_metric=15.0,
            best_source=best_source,
            allow_cross_target_replay=True,
            cross_target_min_accepted=2,
            cross_target_score_scale=0.7,
            max_candidates=8,
            rng=DeterministicRng(choices_result=0),
        )
        self.assertIsNone(selected)

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
            population_recombine_prob=0.15,
            population_recombine_max_lines=12,
            population_max_entries=24,
        )
        self.assertEqual(
            config,
            {
                "language": "python",
                "higher_is_better": True,
                "population_parent_sample_prob": 0.35,
                "population_recombine_prob": 0.15,
                "population_recombine_max_lines": 12,
                "population_max_entries": 24,
            },
        )
        self.assertNotIn("max_iterations", config)
        self.assertNotIn("max_accepted", config)
        self.assertNotIn("max_runtime_seconds", config)

    def test_mark_population_entry_sampled_can_mark_cross_target_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "population.json"
            source = "def f():\n    return 7\n"
            base = {
                "version": 1,
                "entries": [
                    {
                        "target": "poseidon2_cryptanalysis_trackb_kernel_signal_fast",
                        "language": "python",
                        "source_sha256": train.source_sha256(source),
                        "source_code": source,
                        "metric_value": 15.0,
                        "higher_is_better": True,
                        "accepted_total": 3,
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
                entry_target="poseidon2_cryptanalysis_trackb_kernel_signal_fast",
            )
            entries = updated.get("entries", [])
            self.assertEqual(entries[0]["sampled_total"], 1)

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

    def test_recombine_python_parent_block_swaps_selected_function(self) -> None:
        best = (
            "def a():\n"
            "    return 1\n\n"
            "def b():\n"
            "    return 2\n"
        )
        parent = (
            "def a():\n"
            "    return 9\n\n"
            "def b():\n"
            "    return 2\n"
        )
        rng = DeterministicRng(choice_result="a")
        with patch("train.random.choice", side_effect=AssertionError("global random should not be used")):
            candidate, details = train.recombine_python_parent_block(best, parent, rng=rng)
        self.assertIn("return 9", candidate)
        self.assertIn("def b()", candidate)
        self.assertEqual(details.get("status"), "ok")
        self.assertEqual(details.get("function"), "a")

    def test_recombine_python_parent_block_handles_syntax_error(self) -> None:
        candidate, details = train.recombine_python_parent_block("def a(:\n", "def a():\n    return 1\n")
        self.assertEqual(candidate, "def a(:\n")
        self.assertEqual(details.get("status"), "parse_failed:SyntaxError")

    def test_recombine_python_parent_block_handles_parser_exception(self) -> None:
        with patch("train.ast.parse", side_effect=ValueError("boom")):
            candidate, details = train.recombine_python_parent_block(
                "def a():\n    return 1\n",
                "def a():\n    return 2\n",
            )
        self.assertEqual(candidate, "def a():\n    return 1\n")
        self.assertEqual(details.get("status"), "parse_failed:ValueError")

    def test_recombine_python_parent_block_reports_no_differing_function(self) -> None:
        source = "def a():\n    return 1\n"
        candidate, details = train.recombine_python_parent_block(source, source)
        self.assertEqual(candidate, source)
        self.assertEqual(details.get("status"), "no_differing_function")

    def test_recombine_line_window_too_short(self) -> None:
        candidate, details = train.recombine_line_window("a\nb\nc\n", "a\nB\nC\n", max_lines=8)
        self.assertEqual(candidate, "a\nb\nc\n")
        self.assertEqual(details.get("status"), "too_short")

    def test_recombine_with_population_parent_same_source(self) -> None:
        source = "def a():\n    return 1\n"
        candidate, details = train.recombine_with_population_parent(
            best_source=source,
            parent_source=source,
            language="python",
            max_lines=8,
        )
        self.assertEqual(candidate, source)
        self.assertEqual(details.get("status"), "same_as_best")

    def test_recombine_with_population_parent_python_async_path(self) -> None:
        best = (
            "async def a():\n"
            "    return 1\n\n"
            "def b():\n"
            "    return 2\n"
        )
        parent = (
            "async def a():\n"
            "    return 9\n\n"
            "def b():\n"
            "    return 2\n"
        )
        candidate, details = train.recombine_with_population_parent(
            best_source=best,
            parent_source=parent,
            language="python",
            max_lines=8,
            rng=DeterministicRng(choice_result="a"),
        )
        self.assertIn("return 9", candidate)
        self.assertEqual(details.get("method"), "python_function_block")
        self.assertEqual(details.get("function"), "a")

    def test_recombine_with_population_parent_preserves_python_attempt_on_fallback(self) -> None:
        best = "class A:\n    def a(self):\n        return 1\n\n    def b(self):\n        return 2\n"
        parent = "class A:\n    def a(self):\n        return 9\n\n    def b(self):\n        return 2\n"
        candidate, details = train.recombine_with_population_parent(
            best_source=best,
            parent_source=parent,
            language="python",
            max_lines=4,
            rng=DeterministicRng(randint_values=[4, 0]),
        )
        self.assertNotEqual(candidate, best)
        self.assertEqual(details.get("method"), "line_window")
        self.assertEqual(details.get("python_attempt"), "no_shared_functions")

    def test_recombine_with_population_parent_fallback_line_window(self) -> None:
        best = "a\nb\nc\nd\ne\nf\n"
        parent = "a\nB\nC\nD\ne\nf\n"
        candidate, details = train.recombine_with_population_parent(
            best_source=best,
            parent_source=parent,
            language="rust",
            max_lines=4,
            rng=DeterministicRng(randint_values=[4, 1]),
        )
        self.assertEqual(candidate, "a\nB\nC\nD\ne\nf\n")
        self.assertEqual(details.get("method"), "line_window")
        self.assertEqual(details.get("status"), "ok")

    def test_generate_population_seed_candidates_returns_diverse_mutations(self) -> None:
        source = harness_source()
        seeds = train.generate_population_seed_candidates(
            source=source,
            source_path=HARNESS_PATH,
            language="python",
            target_name="poseidon2_cryptanalysis_trackb_kernel_fast",
            target_config={},
            mutation_memory=None,
            max_candidates=6,
            rng_seed="deadbeef01234567",
        )
        self.assertGreaterEqual(len(seeds), 3)
        mutations = [item["mutation"] for item in seeds]
        self.assertEqual(len(mutations), len(set(mutations)))
        self.assertTrue(all(item["source_code"] != source for item in seeds))

    def test_generate_population_seed_candidates_zero_max_returns_empty(self) -> None:
        seeds = train.generate_population_seed_candidates(
            source=harness_source(),
            source_path=HARNESS_PATH,
            language="python",
            target_name="poseidon2_cryptanalysis_trackb_kernel_fast",
            target_config={},
            mutation_memory=None,
            max_candidates=0,
        )
        self.assertEqual(seeds, [])

    def test_generate_population_seed_candidates_uses_seeded_random_state(self) -> None:
        source = "baseline\n"

        def fake_heuristic(
            source: str,
            iteration: int,
            language: str,
            source_path: Path,
            rng: random.Random | None = None,
            **_: object,
        ) -> tuple[str, str, bool]:
            chooser = rng if rng is not None else train.random
            return f"{source}{chooser.random():.8f}\n", f"mutation_{iteration}", True

        with patch("train.heuristic_candidate", side_effect=fake_heuristic):
            first = train.generate_population_seed_candidates(
                source=source,
                source_path=HARNESS_PATH,
                language="python",
                target_name="poseidon2_cryptanalysis_trackb_kernel_fast",
                target_config={},
                mutation_memory=None,
                max_candidates=3,
                rng_seed="00ff11aa22bb33cc",
            )
            second = train.generate_population_seed_candidates(
                source=source,
                source_path=HARNESS_PATH,
                language="python",
                target_name="poseidon2_cryptanalysis_trackb_kernel_fast",
                target_config={},
                mutation_memory=None,
                max_candidates=3,
                rng_seed="00ff11aa22bb33cc",
            )
        self.assertEqual(first, second)

    def test_seed_population_memory_from_heuristics_hits_min_entries(self) -> None:
        source = harness_source()
        with tempfile.TemporaryDirectory() as tmpdir:
            pop_path = Path(tmpdir) / "population.json"
            base_memory = {"version": 1, "entries": []}
            train.save_population_memory(pop_path, base_memory)
            loaded = train.load_population_memory(pop_path)
            loaded, _ = train.update_and_save_population_memory(
                pop_path,
                loaded,
                target_name="poseidon2_cryptanalysis_trackb_kernel_fast",
                language="python",
                source_code=source,
                metric_value=15.35,
                higher_is_better=True,
                accepted=True,
                timestamp="2026-03-17T00:00:00+00:00",
                notes="baseline_seed",
                max_entries=48,
            )
            seeded_memory, report = train.seed_population_memory_from_heuristics(
                population_memory_path=pop_path,
                population_memory=loaded,
                target_name="poseidon2_cryptanalysis_trackb_kernel_fast",
                language="python",
                source_path=HARNESS_PATH,
                baseline_source=source,
                baseline_metric=15.35,
                higher_is_better=True,
                target_config={},
                mutation_memory=None,
                min_entries=4,
                max_candidates=8,
                max_entries=48,
            )
            self.assertGreaterEqual(report.get("final_entries", 0), 4)
            self.assertGreater(report.get("seeded", 0), 0)
            self.assertIsInstance(report.get("rng_seed"), str)
            self.assertEqual(train.population_entries_count(seeded_memory), report.get("final_entries", 0))
            seeded_entries = [
                entry
                for entry in seeded_memory.get("entries", [])
                if isinstance(entry, dict) and str(entry.get("last_notes", "")).startswith("population_seed:")
            ]
            self.assertTrue(seeded_entries)
            self.assertTrue(all(int(entry.get("rejected_total", 0)) == 0 for entry in seeded_entries))
            self.assertTrue(all(int(entry.get("seeded_total", 0)) >= 1 for entry in seeded_entries))

    def test_seed_population_memory_from_heuristics_skips_when_already_sufficient(self) -> None:
        source = harness_source()
        with tempfile.TemporaryDirectory() as tmpdir:
            pop_path = Path(tmpdir) / "population.json"
            memory = {"version": 1, "entries": []}
            for idx in range(3):
                train.upsert_population_entry(
                    memory,
                    target_name="poseidon2_cryptanalysis_trackb_kernel_fast",
                    language="python",
                    source_code=f"{source}# {idx}\n",
                    metric_value=15.35 + idx,
                    higher_is_better=True,
                    accepted=True,
                    timestamp=f"2026-03-17T00:00:0{idx}+00:00",
                    notes=f"accepted:{idx}",
                    max_entries=48,
                )
            train.save_population_memory(pop_path, memory)
            loaded = train.load_population_memory(pop_path)
            seeded_memory, report = train.seed_population_memory_from_heuristics(
                population_memory_path=pop_path,
                population_memory=loaded,
                target_name="poseidon2_cryptanalysis_trackb_kernel_fast",
                language="python",
                source_path=HARNESS_PATH,
                baseline_source=source,
                baseline_metric=15.35,
                higher_is_better=True,
                target_config={},
                mutation_memory=None,
                min_entries=3,
                max_candidates=8,
                max_entries=48,
            )
            self.assertEqual(seeded_memory, loaded)
            self.assertEqual(report["status"], "skipped")
            self.assertEqual(report["reason"], "already_sufficient")
            self.assertEqual(report["initial_entries"], 3)
            self.assertEqual(report["final_entries"], 3)
            self.assertEqual(report["seeded"], 0)
            self.assertIsInstance(report.get("rng_seed"), str)

    def test_seed_population_memory_from_heuristics_propagates_write_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pop_path = Path(tmpdir) / "population.json"
            memory = {"version": 1, "entries": []}
            with patch(
                "train.generate_population_seed_candidates",
                return_value=[{"mutation": "population_seed:test", "source_code": "candidate"}],
            ), patch(
                "train.update_and_save_population_memory",
                side_effect=OSError("read only"),
            ):
                with self.assertRaises(OSError):
                    train.seed_population_memory_from_heuristics(
                        population_memory_path=pop_path,
                        population_memory=memory,
                        target_name="poseidon2_cryptanalysis_trackb_kernel_fast",
                        language="python",
                        source_path=HARNESS_PATH,
                        baseline_source="baseline",
                        baseline_metric=15.35,
                        higher_is_better=True,
                        target_config={},
                        mutation_memory=None,
                        min_entries=2,
                        max_candidates=1,
                        max_entries=48,
                    )


if __name__ == "__main__":
    unittest.main()
