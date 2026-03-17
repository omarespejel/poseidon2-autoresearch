from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import prepare


class PrepareHardeningTests(unittest.TestCase):
    def test_parse_sandbox_prefix_supports_string_and_list(self) -> None:
        self.assertEqual(prepare.parse_sandbox_prefix("docker run --rm"), ["docker", "run", "--rm"])
        self.assertEqual(prepare.parse_sandbox_prefix(["bwrap", "--ro-bind", "/", "/"]), ["bwrap", "--ro-bind", "/", "/"])
        self.assertEqual(prepare.parse_sandbox_prefix(""), [])

    def test_run_cmd_enforces_required_sandbox(self) -> None:
        with patch.dict(os.environ, {"AUTORESEARCH_REQUIRE_SANDBOX": "1", "AUTORESEARCH_SANDBOX_PREFIX": ""}, clear=False):
            result = prepare.run_cmd([sys.executable, "-c", "print('ok')"], prepare.ROOT)
        self.assertEqual(result.code, 126)
        self.assertIn("AUTORESEARCH_SANDBOX_PREFIX", result.stderr)

    def test_append_provenance_event_builds_hash_chain(self) -> None:
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
            self.assertEqual(second["prev_hash"], first["node_hash"])
            self.assertEqual(second["seq"], 2)


if __name__ == "__main__":
    unittest.main()
