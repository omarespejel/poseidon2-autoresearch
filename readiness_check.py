#!/usr/bin/env python3
"""Submission-readiness checker for AutoPoseidon artifacts and timeline."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
RESULTS_FILE = ROOT / "results.tsv"
EVIDENCE_MANIFEST = ROOT / "evidence" / "manifest.json"
EVIDENCE_SUMMARY = ROOT / "evidence" / "summary.md"
PORTFOLIO_JSON = ROOT / "portfolio_report.json"
SUBMISSION_DIR = ROOT / "submission"
AGENT_MANIFEST = SUBMISSION_DIR / "agent.json"
AGENT_LOG_JSON = SUBMISSION_DIR / "agent_log.json"
SUBMISSION_RECEIPTS = SUBMISSION_DIR / "submission_receipts.json"
DEFAULT_REPORT = ROOT / "readiness_report.md"

# Public Synthesis timeline marker observed on-site.
DEFAULT_BUILD_CLOSE_UTC = dt.datetime(2026, 3, 22, 17, 0, 0, tzinfo=dt.timezone.utc)
INFORMATIONAL_CHECKS: frozenset[str] = frozenset(
    {
        "recent_activity_24h",
        "submission_multi_tool_orchestration",
        "submission_receipts_additional_evidence",
        # Kept informational for backward compatibility with older agent_log schemas.
        "submission_safety_guardrails_populated",
        "submission_compute_budget_usage",
    }
)


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


def looks_like_onchain_receipt(value: str) -> bool:
    token = value.strip()
    if not token:
        return False
    if re.fullmatch(r"0x[0-9a-fA-F]{64}", token):
        return True
    return token.startswith("https://")


def overall_ready_from_checks(checks: list[CheckResult], *, informational: frozenset[str] = INFORMATIONAL_CHECKS) -> bool:
    return all(c.ok for c in checks if c.name not in informational)


def retained_entry_path(item: dict[str, Any]) -> Path | None:
    for key in ("retained_artifact_dir", "artifact_iter_dir", "metadata_file"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return Path(value)
    return None


def resolve_build_close_utc(raw: str) -> dt.datetime:
    parsed = parse_iso(raw)
    if parsed is None:
        raise ValueError(f"invalid ISO-8601 timestamp: {raw}")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def build_checks(
    *,
    manifest: dict[str, Any] | None,
    portfolio: dict[str, Any] | None,
    rows: list[dict[str, str]],
    build_close_utc: dt.datetime,
) -> list[CheckResult]:
    checks: list[CheckResult] = []
    submission_manifest = load_json(AGENT_MANIFEST)
    submission_log = load_json(AGENT_LOG_JSON)
    submission_receipts = load_json(SUBMISSION_RECEIPTS)

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
    checks.append(
        CheckResult(
            name="submission_agent_manifest_present",
            ok=submission_manifest is not None,
            details="submission/agent.json exists and parses"
            if submission_manifest is not None
            else "missing or invalid submission/agent.json",
        )
    )
    checks.append(
        CheckResult(
            name="submission_agent_log_present",
            ok=submission_log is not None,
            details="submission/agent_log.json exists and parses"
            if submission_log is not None
            else "missing or invalid submission/agent_log.json",
        )
    )
    checks.append(
        CheckResult(
            name="submission_receipts_present",
            ok=submission_receipts is not None,
            details="submission/submission_receipts.json exists and parses"
            if submission_receipts is not None
            else "missing or invalid submission/submission_receipts.json",
        )
    )

    if submission_manifest is not None:
        required_manifest_keys = [
            "name",
            "operator_wallet",
            "erc8004_identity",
            "erc8004_registration_tx",
            "supported_tools",
            "tech_stacks",
            "compute_constraints",
            "task_categories",
        ]
        missing_keys = [key for key in required_manifest_keys if key not in submission_manifest]
        checks.append(
            CheckResult(
                name="submission_manifest_required_keys",
                ok=not missing_keys,
                details="ok" if not missing_keys else f"missing keys: {', '.join(missing_keys)}",
            )
        )

        missing_identity_values = []
        for key in ("operator_wallet", "erc8004_identity", "erc8004_registration_tx"):
            value = submission_manifest.get(key)
            if not isinstance(value, str) or not value.strip():
                missing_identity_values.append(key)
        checks.append(
            CheckResult(
                name="submission_manifest_identity_values",
                ok=not missing_identity_values,
                details="ok"
                if not missing_identity_values
                else f"missing identity values: {', '.join(missing_identity_values)}",
            )
        )

        compute_constraints = submission_manifest.get("compute_constraints")
        has_compute = isinstance(compute_constraints, dict) and any(
            compute_constraints.get(key) is not None
            for key in ("max_iterations", "max_accepted", "max_runtime_seconds", "max_model_calls")
        )
        checks.append(
            CheckResult(
                name="submission_compute_budget_defined",
                ok=has_compute,
                details="compute constraints populated" if has_compute else "compute constraints missing or empty",
            )
        )

    if submission_log is not None:
        required_sections = [
            "stages",
            "conversation_log",
            "decisions",
            "tool_calls",
            "failures",
            "final_outputs",
            "compute_budget",
            "safety_guardrails",
        ]
        missing_sections = [section for section in required_sections if section not in submission_log]
        checks.append(
            CheckResult(
                name="submission_log_required_sections",
                ok=not missing_sections,
                details="ok" if not missing_sections else f"missing sections: {', '.join(missing_sections)}",
            )
        )

        stage_counts = submission_log.get("stages")
        missing_stages: list[str] = []
        if isinstance(stage_counts, dict):
            for stage in ("discover", "plan", "execute", "verify", "submit"):
                count = stage_counts.get(stage)
                if not isinstance(count, int) or count < 1:
                    missing_stages.append(stage)
        else:
            missing_stages = ["discover", "plan", "execute", "verify", "submit"]
        checks.append(
            CheckResult(
                name="submission_log_stage_coverage",
                ok=not missing_stages,
                details="ok" if not missing_stages else f"missing stage coverage: {', '.join(missing_stages)}",
            )
        )

        decisions = submission_log.get("decisions")
        decision_count = len(decisions) if isinstance(decisions, list) else 0
        checks.append(
            CheckResult(
                name="submission_autonomous_decision_trace",
                ok=decision_count > 0,
                details=f"decision_count={decision_count}",
            )
        )

        conversation_log = submission_log.get("conversation_log")
        conversation_count = len(conversation_log) if isinstance(conversation_log, list) else 0
        checks.append(
            CheckResult(
                name="submission_conversation_log_present",
                ok=conversation_count > 0,
                details=f"conversation_events={conversation_count}",
            )
        )
        seen_stages: set[str] = set()
        if isinstance(conversation_log, list):
            for event in conversation_log:
                if not isinstance(event, dict):
                    continue
                stage = event.get("stage")
                if isinstance(stage, str) and stage.strip():
                    seen_stages.add(stage.strip())
        missing_trace_stages = [
            stage for stage in ("discover", "plan", "execute", "verify", "submit") if stage not in seen_stages
        ]
        checks.append(
            CheckResult(
                name="submission_conversation_stage_trace",
                ok=not missing_trace_stages,
                details="ok" if not missing_trace_stages else f"missing stages: {', '.join(missing_trace_stages)}",
            )
        )

        tool_calls = submission_log.get("tool_calls")
        distinct_tools: set[str] = set()
        if isinstance(tool_calls, list):
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                tool = call.get("tool")
                if isinstance(tool, str) and tool.strip():
                    distinct_tools.add(tool.strip())
        checks.append(
            CheckResult(
                name="submission_tool_calls_present",
                ok=bool(distinct_tools),
                details=f"distinct_tools={len(distinct_tools)} [{', '.join(sorted(distinct_tools)[:8])}]",
            )
        )
        checks.append(
            CheckResult(
                name="submission_multi_tool_orchestration",
                ok=len(distinct_tools) >= 2,
                details=f"distinct_tools={len(distinct_tools)}",
            )
        )

        safety_guardrails = submission_log.get("safety_guardrails")
        has_safety_payload = False
        if isinstance(safety_guardrails, dict):
            policy = safety_guardrails.get("policy")
            counters = safety_guardrails.get("counters")
            has_safety_payload = (
                isinstance(policy, list)
                and len(policy) > 0
                and isinstance(counters, dict)
                and len(counters) > 0
            )
        checks.append(
            CheckResult(
                name="submission_safety_guardrails_populated",
                ok=has_safety_payload,
                details="safety_guardrails has policy and counters" if has_safety_payload else "missing safety policy/counters",
            )
        )

        compute_budget = submission_log.get("compute_budget")
        has_budget_usage = False
        if isinstance(compute_budget, dict):
            consumed = compute_budget.get("consumed")
            has_budget_usage = isinstance(consumed, dict) and any(
                consumed.get(key) is not None
                for key in ("iterations", "accepted", "estimated_model_calls", "evaluation_seconds_total")
            )
        checks.append(
            CheckResult(
                name="submission_compute_budget_usage",
                ok=has_budget_usage,
                details="compute_budget.consumed populated" if has_budget_usage else "missing compute budget usage details",
            )
        )

    if submission_receipts is not None:
        erc8004 = submission_receipts.get("erc8004")
        reg_tx = ""
        additional_count = 0
        if isinstance(erc8004, dict):
            raw = erc8004.get("registration_tx", "")
            if isinstance(raw, str):
                reg_tx = raw.strip()
            raw_additional = erc8004.get("additional_receipts")
            if isinstance(raw_additional, list):
                additional_count = sum(
                    1
                    for item in raw_additional
                    if isinstance(item, str) and bool(item.strip()) and looks_like_onchain_receipt(item)
                )
        checks.append(
            CheckResult(
                name="submission_receipts_registration_tx",
                ok=bool(reg_tx),
                details="ok" if reg_tx else "missing erc8004.registration_tx in submission receipts",
            )
        )
        if reg_tx:
            receipt_ok = looks_like_onchain_receipt(reg_tx)
            checks.append(
                CheckResult(
                    name="submission_receipts_registration_tx_format",
                    ok=receipt_ok,
                    details="onchain receipt format detected" if receipt_ok else "registration tx is not hash/url-like",
                )
            )
        checks.append(
            CheckResult(
                name="submission_receipts_additional_evidence",
                ok=additional_count > 0,
                details=f"additional_onchain_receipts={additional_count}",
            )
        )

    accepted_total = int(manifest.get("accepted_total", 0)) if manifest else 0
    retained_count = 0
    retained_files_count = 0
    missing_count = 0
    if manifest and isinstance(manifest.get("accepted"), list):
        for item in manifest["accepted"]:
            if isinstance(item, dict) and bool(item.get("retained")):
                retained_count += 1
                retained_path = retained_entry_path(item)
                if retained_path is not None and retained_path.exists():
                    retained_files_count += 1
                else:
                    missing_count += 1

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
            ok=retained_files_count > 0,
            details=f"retained_files={retained_files_count}, missing={missing_count}, flagged={retained_count}",
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

    remaining_h = (build_close_utc - now_utc()).total_seconds() / 3600.0
    checks.append(
        CheckResult(
            name="before_build_close",
            ok=remaining_h > 0.0,
            details=f"hours_to_build_close={remaining_h:.2f} (deadline={build_close_utc.isoformat()})",
        )
    )

    return checks


def write_markdown_report(path: Path, checks: list[CheckResult], *, build_close_utc: dt.datetime) -> None:
    # Fresh activity is informative but not a hard blocker once the evidence pack is assembled.
    ready = overall_ready_from_checks(checks)
    now = now_utc().isoformat()

    lines: list[str] = []
    lines.append("# AutoPoseidon Readiness Report")
    lines.append("")
    lines.append(f"- generated_at: `{now}`")
    lines.append(f"- build_close_utc: `{build_close_utc.isoformat()}`")
    lines.append(f"- overall_ready: `{'yes' if ready else 'no'}`")
    lines.append("")
    lines.append("| check | ok | details |")
    lines.append("|---|---|---|")
    for c in checks:
        lines.append(f"| {c.name} | {'yes' if c.ok else 'no'} | {c.details} |")

    blockers = [c for c in checks if not c.ok and c.name not in INFORMATIONAL_CHECKS]
    warnings = [c for c in checks if not c.ok and c.name in INFORMATIONAL_CHECKS]
    lines.append("")
    lines.append("## Blockers")
    lines.append("")
    if not blockers:
        lines.append("- none")
    else:
        for b in blockers:
            lines.append(f"- `{b.name}`: {b.details}")

    if warnings:
        lines.append("")
        lines.append("## Warnings")
        lines.append("")
        for warning in warnings:
            lines.append(f"- `{warning.name}`: {warning.details}")

    path.write_text("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check AutoPoseidon submission readiness")
    parser.add_argument("--report", default=str(DEFAULT_REPORT), help="Path to write markdown report")
    parser.add_argument(
        "--build-close-utc",
        default=DEFAULT_BUILD_CLOSE_UTC.isoformat(),
        help="ISO-8601 build-close timestamp in UTC",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        build_close_utc = resolve_build_close_utc(args.build_close_utc)
    except ValueError as exc:
        print(f"Invalid --build-close-utc: {exc}", file=sys.stderr)
        return 2

    manifest = load_json(EVIDENCE_MANIFEST)
    portfolio = load_json(PORTFOLIO_JSON)
    rows = read_results_rows()
    checks = build_checks(manifest=manifest, portfolio=portfolio, rows=rows, build_close_utc=build_close_utc)
    overall_ready = overall_ready_from_checks(checks)

    report_path = Path(args.report)
    write_markdown_report(report_path, checks, build_close_utc=build_close_utc)

    payload = {
        "ok": True,
        "generated_at": now_utc().isoformat(),
        "report": str(report_path),
        "build_close_utc": build_close_utc.isoformat(),
        "overall_ready": overall_ready,
        "informational_checks": sorted(INFORMATIONAL_CHECKS),
        "checks": [{"name": c.name, "ok": c.ok, "details": c.details} for c in checks],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
