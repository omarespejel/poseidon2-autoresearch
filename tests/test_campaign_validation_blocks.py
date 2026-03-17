from __future__ import annotations

import unittest

import campaign


class CampaignValidationBlockTests(unittest.TestCase):
    def test_extract_mutation_from_validation_note(self) -> None:
        token = "rejected_validation_degraded:poseidon2_cryptanalysis_trackb_kernel_signal_fast:python_trackb_diff_cross_lane_prob_up+python_trackb_mitm_bucket_cap_up"
        mutation = campaign.extract_mutation_from_note_token(token)
        self.assertEqual(
            mutation,
            "python_trackb_diff_cross_lane_prob_up+python_trackb_mitm_bucket_cap_up",
        )

    def test_summarize_validation_blocks_groups_by_mutation(self) -> None:
        rows = [
            {
                "target": "poseidon2_cryptanalysis_trackb_kernel_fast",
                "notes": "rejected_validation_degraded:poseidon2_cryptanalysis_trackb_kernel_signal_fast:python_trackb_diff_cross_lane_prob_up;diag=n/a;run=run1",
            },
            {
                "target": "poseidon2_cryptanalysis_trackb_kernel_fast",
                "notes": "rejected_validation_degraded:poseidon2_cryptanalysis_trackb_kernel_signal_fast:python_trackb_diff_cross_lane_prob_up;diag=n/a;run=run1",
            },
            {
                "target": "poseidon2_cryptanalysis_trackb_kernel_signal_fast",
                "notes": "rejected_validation_eval_failed:poseidon2_cryptanalysis_trackb_kernel_fast:python_trackb_mitm_bucket_cap_up;diag=n/a;run=run2",
            },
            {
                "target": "poseidon2_cryptanalysis_trackb_kernel_fast",
                "notes": "accepted:python_trackb_diff_cross_lane_prob_up;run=run3",
            },
        ]
        summary = campaign.summarize_validation_blocks(rows)
        self.assertEqual(summary["total"], 3)
        self.assertEqual(len(summary["rows"]), 2)
        self.assertEqual(summary["rows"][0]["mutation"], "python_trackb_diff_cross_lane_prob_up")
        self.assertEqual(summary["rows"][0]["count"], 2)
        self.assertIn("poseidon2_cryptanalysis_trackb_kernel_fast", summary["rows"][0]["targets"])


if __name__ == "__main__":
    unittest.main()
