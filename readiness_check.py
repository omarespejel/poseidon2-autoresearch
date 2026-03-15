#!/usr/bin/env python3
"""Submission-readiness checker for AutoPoseidon artifacts and timeline."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
RESULTS_FILE = ROOT / "results.tsv"
EVIDENCE_MANIFEST = ROOT / "evidence" / "manifest.json"
EVIDENCE_SUMMARY = ROOT / "evidence" / "summary.md"
PORTFOLIO_JSON = ROOT / "portfolio_report.json"
DEFAULT_REPORT = ROOT / "readiness_report.md"

# Public Synthesis timeline marker observed on-site.
BUILD_CLOSE_UTC = dt.datetime(2026, 3, 22, 17, 0, 0, tzinfo=dt.timezone.utc)


@dataclass
class CheckResult:
    name: str
    ok: bool
    details: str


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def parse_iso(raw: str | None) -> dt.datetime | None:
    if not raw:
        return None
    try:
        return dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def read_results_rows() -> list[dict[str, str]]:
    if not RESULTS_FILE.exists():
        return []
    with RESULTS_FILE.open("r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def latest_result_timestamp(rows: list[dict[str, str]]) -> dt.datetime | None:
    out: dt.datetime | None = None
    for row in rows:
        ts = parse_iso(row.get("timestamp", ""))
        if ts and (out is None or ts > out):
            out = ts
    return out


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(data, dict):
        return data
    return None


def build_checks(
    *,
    manifest: dict[str, Any] | None,
    portfolio: dict[str, Any] | None,
    rows: list[dict[str, str]],
) -> list[CheckResult]:
    checks: list[CheckResult] = []

    checks.append(
        CheckResult(
            name="evidence_manifest_present",
            ok=manifest is not None,
            details="evidence/manifest.json exists and parses" if manifest is not None else "missing or invalid manifest",
        )
    )
    checks.append(
        CheckResult(
            name="portfolio_report_present",
            ok=portfolio is not None,
            details="portfolio_report.json exists and parses" if portfolio is not None else "missing or invalid report",
        )
    )
    checks.append(
        CheckResult(
            name="evidence_summary_present",
            ok=EVIDENCE_SUMMARY.exists(),
            details="evidence/summary.md exists" if EVIDENCE_SUMMARY.exists() else "missing evidence summary",
        )
    )

    accepted_total = int(manifest.get("accepted_total", 0)) if manifest else 0
    retained_count = 0
    if manifest and isinstance(manifest.get("accepted"), list):
        for item in manifest["accepted"]:
            if isinstance(item, dict) and bool(item.get("retained")):
                retained_count += 1

    checks.append(
        CheckResult(
            name="accepted_history_nonempty",
            ok=accepted_total > 0,
            details=f"accepted_total={accepted_total}",
        )
    )
    checks.append(
        CheckResult(
            name="retained_patch_exists",
            ok=retained_count > 0,
            details=f"retained_count={retained_count}",
        )
    )

    ts = latest_result_timestamp(rows)
    recent_ok = False
    details = "no results rows"
    if ts is not None:
        age_h = (now_utc() - ts).total_seconds() / 3600.0
        recent_ok = age_h <= 24.0
        details = f"latest_result={ts.isoformat()} age_hours={age_h:.2f}"
    checks.append(CheckResult(name="recent_activity_24h", ok=recent_ok, details=details))

    remaining_h = (BUILD_CLOSE_UTC - now_utc()).total_seconds() / 3600.0
    checks.append(
        CheckResult(
            name="before_build_close",
            ok=remaining_h > 0.0,
            details=f"hours_to_build_close={remaining_h:.2f} (deadline={BUILD_CLOSE_UTC.isoformat()})",
        )
    )

    return checks


def write_markdown_report(path: Path, checks: list[CheckResult]) -> None:
    ready = all(c.ok for c in checks if c.name != "recent_activity_24h")
    now = now_utc().isoformat()

    lines: list[str] = []
    lines.append("# AutoPoseidon Readiness Report")
    lines.append("")
    lines.append(f"- generated_at: `{now}`")
    lines.append(f"- build_close_utc: `{BUILD_CLOSE_UTC.isoformat()}`")
    lines.append(f"- overall_ready: `{'yes' if ready else 'no'}`")
    lines.append("")
    lines.append("| check | ok | details |")
    lines.append("|---|---|---|")
    for c in checks:
        lines.append(f"| {c.name} | {'yes' if c.ok else 'no'} | {c.details} |")

    blockers = [c for c in checks if not c.ok]
    lines.append("")
    lines.append("## Blockers")
    lines.append("")
    if not blockers:
        lines.append("- none")
    else:
        for b in blockers:
            lines.append(f"- `{b.name}`: {b.details}")

    path.write_text("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check AutoPoseidon submission readiness")
    parser.add_argument("--report", default=str(DEFAULT_REPORT), help="Path to write markdown report")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    manifest = load_json(EVIDENCE_MANIFEST)
    portfolio = load_json(PORTFOLIO_JSON)
    rows = read_results_rows()
    checks = build_checks(manifest=manifest, portfolio=portfolio, rows=rows)

    report_path = Path(args.report)
    write_markdown_report(report_path, checks)

    payload = {
        "ok": True,
        "generated_at": now_utc().isoformat(),
        "report": str(report_path),
        "build_close_utc": BUILD_CLOSE_UTC.isoformat(),
        "checks": [{"name": c.name, "ok": c.ok, "details": c.details} for c in checks],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
