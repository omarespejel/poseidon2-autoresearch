from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import io
import unittest
from unittest.mock import patch

import campaign


class FixedDateTime(dt.datetime):
    @classmethod
    def now(cls, tz: dt.tzinfo | None = None) -> dt.datetime:
        fixed = cls(2026, 3, 17, 7, 35, 0, tzinfo=dt.timezone.utc)
        if tz is None:
            return fixed.replace(tzinfo=None)
        return fixed.astimezone(tz)


class CampaignLogTests(unittest.TestCase):
    def test_campaign_log_skips_when_verbose_zero(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            campaign.campaign_log(argparse.Namespace(verbose=0), "quiet")
        self.assertEqual(stderr.getvalue(), "")

    def test_campaign_log_skips_when_verbose_missing_or_none(self) -> None:
        for args in (argparse.Namespace(), argparse.Namespace(verbose=None)):
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                campaign.campaign_log(args, "quiet")
            self.assertEqual(stderr.getvalue(), "")

    def test_campaign_log_writes_timestamped_line(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), patch.object(campaign.dt, "datetime", FixedDateTime):
            campaign.campaign_log(argparse.Namespace(verbose=1), "hello")
        self.assertEqual(stderr.getvalue(), "[campaign 2026-03-17T07:35:00+00:00] hello\n")


class ResolveLoopTargetTests(unittest.TestCase):
    def test_cryptanalysis_uses_kernel_first_default_for_sentinel(self) -> None:
        resolved = campaign.resolve_loop_target("cryptanalysis", campaign.DEFAULT_LOOP_TARGET_SENTINEL)
        self.assertEqual(resolved, campaign.DEFAULT_CRYPTANALYSIS_LOOP_TARGET)

    def test_explicit_loop_target_is_preserved(self) -> None:
        resolved = campaign.resolve_loop_target("cryptanalysis", campaign.LEGACY_CRYPTANALYSIS_LOOP_TARGET)
        self.assertEqual(resolved, campaign.LEGACY_CRYPTANALYSIS_LOOP_TARGET)

    def test_parser_help_documents_legacy_pin(self) -> None:
        help_text = campaign.build_parser().format_help()
        self.assertIn(campaign.DEFAULT_CRYPTANALYSIS_LOOP_TARGET, help_text)
        self.assertIn(campaign.LEGACY_CRYPTANALYSIS_LOOP_TARGET, help_text)


if __name__ == "__main__":
    unittest.main()
