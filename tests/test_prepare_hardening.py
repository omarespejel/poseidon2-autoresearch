from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import prepare


def node_hash_for(record: dict[str, object]) -> str:
    material = prepare.stable_json(
        {
            "seq": record["seq"],
            "timestamp": record["timestamp"],
            "event_type": record["event_type"],
            "prev_hash": record["prev_hash"],
            "payload_hash": record["payload_hash"],
        }
    )
    return prepare.sha256_text(material)


class PrepareHardeningTests(unittest.TestCase):
    def test_file_lock_reuses_in_process_mutex_for_same_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock_a = root / "shared.lock"
            lock_b = root / "other.lock"
            key_a = str(lock_a.resolve())
            key_b = str(lock_b.resolve())
            prepare._THREAD_FILE_LOCKS.clear()

            with prepare.file_lock(lock_a):
                first = prepare._THREAD_FILE_LOCKS[key_a]
            with prepare.file_lock(lock_a):
                second = prepare._THREAD_FILE_LOCKS[key_a]
            with prepare.file_lock(lock_b):
                third = prepare._THREAD_FILE_LOCKS[key_b]

        self.assertIs(first, second)
        self.assertIsNot(first, third)

    def test_parse_sandbox_prefix_supports_string_list_none_and_empty(self) -> None:
        self.assertEqual(prepare.parse_sandbox_prefix("docker run --rm"), ["docker", "run", "--rm"])
        self.assertEqual(prepare.parse_sandbox_prefix(["bwrap", "--ro-bind", "/", "/"]), ["bwrap", "--ro-bind", "/", "/"])
        self.assertEqual(prepare.parse_sandbox_prefix(""), [])
        self.assertEqual(prepare.parse_sandbox_prefix(None), [])

    def test_parse_sandbox_prefix_rejects_malformed_shell_string(self) -> None:
        with self.assertRaises(ValueError):
            prepare.parse_sandbox_prefix("bwrap --ro-bind '/ /")

    def test_run_cmd_enforces_required_sandbox(self) -> None:
        with patch.dict(os.environ, {"AUTORESEARCH_REQUIRE_SANDBOX": "1", "AUTORESEARCH_SANDBOX_PREFIX": ""}, clear=False):
            result = prepare.run_cmd([sys.executable, "-c", "print('ok')"], prepare.ROOT)
        self.assertEqual(result.code, 126)
        self.assertIn("AUTORESEARCH_SANDBOX_PREFIX", result.stderr)

    def test_run_cmd_explicit_empty_sandbox_prefix_does_not_fall_back_to_env(self) -> None:
        completed = subprocess.CompletedProcess(args=["python3"], returncode=0, stdout="ok\n", stderr="")
        with patch.dict(
            os.environ,
            {"AUTORESEARCH_SANDBOX_PREFIX": "bwrap --ro-bind / /"},
            clear=False,
        ), patch("prepare.subprocess.run", return_value=completed) as run_mock:
            result = prepare.run_cmd(
                [sys.executable, "-c", "print('ok')"],
                prepare.ROOT,
                sandbox_prefix=[],
            )
        self.assertEqual(result.code, 0)
        self.assertEqual(run_mock.call_args.args[0][:2], [sys.executable, "-c"])

    def test_append_provenance_event_builds_verifiable_hash_chain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            provenance_file = root / "provenance_chain.jsonl"
            state_file = root / "state.json"
            with patch.object(prepare, "PROVENANCE_FILE", provenance_file), patch.object(
                prepare,
                "PROVENANCE_STATE_FILE",
                state_file,
            ):
                prepare.append_provenance_event("event_a", {"value": 1})
                prepare.append_provenance_event("event_b", {"value": 2})

            lines = provenance_file.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            first = json.loads(lines[0])
            second = json.loads(lines[1])
            self.assertEqual(first["payload"], {"value": 1})
            self.assertEqual(second["payload"], {"value": 2})
            self.assertEqual(first["payload_hash"], prepare.sha256_text(prepare.stable_json(first["payload"])))
            self.assertEqual(second["payload_hash"], prepare.sha256_text(prepare.stable_json(second["payload"])))
            self.assertEqual(first["node_hash"], node_hash_for(first))
            self.assertEqual(second["node_hash"], node_hash_for(second))
            self.assertEqual(second["prev_hash"], first["node_hash"])
            self.assertEqual(second["seq"], 2)

    def test_append_provenance_event_rejects_corrupt_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            provenance_file = root / "provenance_chain.jsonl"
            state_file = root / "state.json"
            state_file.write_text("{broken", encoding="utf-8")
            with patch.object(prepare, "PROVENANCE_FILE", provenance_file), patch.object(
                prepare,
                "PROVENANCE_STATE_FILE",
                state_file,
            ):
                with self.assertRaises(prepare.ToolError):
                    prepare.append_provenance_event("event_a", {"value": 1})

    def test_append_result_row_records_verifiable_tsv_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results_file = root / "results.tsv"
            log_file = root / "agent_log.jsonl"
            provenance_file = root / "provenance_chain.jsonl"
            state_file = root / "state.json"
            with patch.object(prepare, "RESULTS_FILE", results_file), patch.object(
                prepare,
                "LOG_FILE",
                log_file,
            ), patch.object(
                prepare,
                "PROVENANCE_FILE",
                provenance_file,
            ), patch.object(
                prepare,
                "PROVENANCE_STATE_FILE",
                state_file,
            ):
                prepare.append_result_row(
                    target="poseidon2_cryptanalysis_trackb_fast",
                    iteration=1,
                    status="success",
                    metric_name="attack_score",
                    metric_value=12.3456789,
                    higher_is_better=True,
                    check_s=0.1,
                    info_or_bench_s=0.2,
                    execute_s=0.3,
                    notes="accepted:test",
                )

            result_lines = results_file.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(result_lines), 2)
            row_tsv = result_lines[1]
            record = json.loads(provenance_file.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(record["payload"]["row_tsv"], row_tsv)
            self.assertEqual(record["payload_hash"], prepare.sha256_text(prepare.stable_json(record["payload"])))
            self.assertEqual(record["node_hash"], node_hash_for(record))


if __name__ == "__main__":
    unittest.main()
