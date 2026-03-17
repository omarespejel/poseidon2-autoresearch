from __future__ import annotations

import unittest
from unittest.mock import patch

import campaign


class CampaignJsonContractTests(unittest.TestCase):
    def test_must_run_json_accepts_object_with_ok_true(self) -> None:
        result = campaign.RunResult(argv=["echo"], code=0, stdout='{"ok": true, "value": 3}', stderr="")
        with patch("campaign.must_run", return_value=result):
            payload = campaign.must_run_json(["noop"], label="unit")
        self.assertEqual(payload["value"], 3)

    def test_must_run_json_rejects_malformed_payload(self) -> None:
        result = campaign.RunResult(argv=["echo"], code=0, stdout="{broken", stderr="")
        with patch("campaign.must_run", return_value=result):
            with self.assertRaises(SystemExit):
                campaign.must_run_json(["noop"], label="unit")

    def test_must_run_json_rejects_missing_ok_field(self) -> None:
        result = campaign.RunResult(argv=["echo"], code=0, stdout='{"value": 3}', stderr="")
        with patch("campaign.must_run", return_value=result):
            with self.assertRaises(SystemExit):
                campaign.must_run_json(["noop"], label="unit")

    def test_must_run_json_rejects_ok_false(self) -> None:
        result = campaign.RunResult(argv=["echo"], code=0, stdout='{"ok": false, "error": "boom"}', stderr="")
        with patch("campaign.must_run", return_value=result):
            with self.assertRaises(SystemExit):
                campaign.must_run_json(["noop"], label="unit")


if __name__ == "__main__":
    unittest.main()
