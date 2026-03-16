#!/usr/bin/env python3
"""Reproducible campaign runner for AutoPoseidon.

Runs a dual-track execution:
- Track A: autonomous optimization loop and Synthesis-ready artifacts
- Real stack: Lean/Poseidon2 baseline + optional source-level optimization
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
PREPARE = ROOT / "prepare.py"
LOOP = ROOT / "train.py"
PORTFOLIO = ROOT / "portfolio_loop.py"
EVIDENCE_PACK = ROOT / "evidence_pack.py"
SUBMISSION_PACK = ROOT / "submission_pack.py"
READINESS_CHECK = ROOT / "readiness_check.py"
RESULTS = ROOT / "results.tsv"
LOG = ROOT / "agent_log.jsonl"
REPORT = ROOT / "campaign_report.md"
PORTFOLIO_MD = ROOT / "portfolio_report.md"
PORTFOLIO_JSON = ROOT / "portfolio_report.json"
READINESS_REPORT = ROOT / "readiness_report.md"
EVIDENCE_DIR = ROOT / "evidence"
SUBMISSION_DIR = ROOT / "submission"

DEFAULT_REAL_OPTIMIZE_TARGETS = (
    "leanmultisig_poseidon16_src_fast,"
    "leanmultisig_poseidon16_table_src_fast,"
    "leanmultisig_poseidon2_neon_src_fast"
)


@dataclass
class RunResult:
    argv: list[str]
    code: int
    stdout: str
    stderr: str


def run(argv: list[str], *, stream_stderr: bool = False) -> RunResult:
    if stream_stderr:
        proc = subprocess.run(
            argv,
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=None,
            check=False,
        )
        return RunResult(argv=argv, code=proc.returncode, stdout=proc.stdout, stderr="")

    proc = subprocess.run(argv, cwd=str(ROOT), text=True, capture_output=True, check=False)
    return RunResult(argv=argv, code=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)


def must_run(argv: list[str], *, label: str, stream_stderr: bool = False) -> RunResult:
    result = run(argv, stream_stderr=stream_stderr)
    if result.code != 0:
        sys.stderr.write(f"Command failed ({result.code}) [{label}]: {' '.join(argv)}\n")
        if result.stdout:
            sys.stderr.write(result.stdout + "\n")
        if result.stderr:
            sys.stderr.write(result.stderr + "\n")
        raise SystemExit(result.code)
    return result


def must_run_json(argv: list[str], *, label: str, stream_stderr: bool = False) -> dict[str, Any]:
    result = must_run(argv, label=label, stream_stderr=stream_stderr)
    raw = result.stdout.strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {"parse_error": True, "stdout_tail": raw[-2000:]}
    return payload if isinstance(payload, dict) else {"parse_error": True, "stdout_tail": raw[-2000:]}


def reset_outputs() -> None:
    RESULTS.write_text(
        "timestamp\ttarget\titeration\tstatus\tmetric_name\tmetric_value\tbest_value\tdelta\tcheck_s\tinfo_or_bench_s\texecute_s\tnotes\n"
    )
    LOG.write_text("")
    for path in (REPORT, PORTFOLIO_MD, PORTFOLIO_JSON, READINESS_REPORT):
        if path.exists():
            path.unlink()
    for path in (EVIDENCE_DIR, SUBMISSION_DIR):
        if path.exists():
            shutil.rmtree(path)


def parse_results() -> list[dict[str, str]]:
    if not RESULTS.exists():
        return []
    with RESULTS.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def summarize(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    by_target: dict[str, dict[str, str]] = {}
    for row in rows:
        by_target[row["target"]] = row
    return by_target


def count_accepts(rows: list[dict[str, str]], target: str) -> int:
    n = 0
    for row in rows:
        if row["target"] != target:
            continue
        if row["notes"].startswith("accepted:"):
            n += 1
    return n


def add_debug_flags(argv: list[str], args: argparse.Namespace) -> list[str]:
    out = list(argv)
    if args.verbose > 0:
        out.extend(["-v"] * args.verbose)
    if args.debug_command_output:
        out.append("--debug-command-output")
    out.extend(["--debug-max-chars", str(max(256, int(args.debug_max_chars)))])
    return out


def prepare_cmd(args: argparse.Namespace, *tail: str) -> list[str]:
    base = [sys.executable, str(PREPARE)]
    return add_debug_flags(base, args) + list(tail)


def train_cmd(args: argparse.Namespace, *tail: str) -> list[str]:
    base = [sys.executable, str(LOOP)]
    return add_debug_flags(base, args) + list(tail)


def portfolio_cmd(args: argparse.Namespace, *tail: str) -> list[str]:
    base = [sys.executable, str(PORTFOLIO)]
    return add_debug_flags(base, args) + list(tail)


def write_report(
    *,
    rows: list[dict[str, str]],
    loop_target: str,
    real_targets: list[str],
    loop_iterations: int,
    loop_accepts: int,
    loop_payload: dict[str, Any],
    portfolio_payload: dict[str, Any] | None,
    evidence_payload: dict[str, Any] | None,
    submission_payload: dict[str, Any] | None,
    readiness_payload: dict[str, Any] | None,
    synthesis_cook: bool,
) -> None:
    latest = summarize(rows)

    lines: list[str] = []
    lines.append("# AutoPoseidon Campaign Report")
    lines.append("")
    lines.append("## Run Summary")
    lines.append("")
    lines.append(f"- Synthesis cook mode: `{str(synthesis_cook).lower()}`")
    lines.append(f"- Loop target: `{loop_target}`")
    lines.append(f"- Loop iterations requested: `{loop_iterations if loop_iterations > 0 else 'infinite'}`")
    lines.append(f"- Loop accepted improvements: `{loop_accepts}`")
    lines.append(f"- Real baseline targets: `{', '.join(real_targets) if real_targets else 'none'}`")
    if portfolio_payload:
        lines.append(f"- Real optimization rounds executed: `{portfolio_payload.get('rounds_executed', 'n/a')}`")
        lines.append(f"- Real optimization total accepted: `{portfolio_payload.get('total_accepted', 'n/a')}`")
    lines.append("")
    lines.append("## Latest Metrics")
    lines.append("")
    lines.append("| target | status | metric | value | best | notes |")
    lines.append("|---|---|---|---:|---:|---|")

    for target, row in latest.items():
        lines.append(
            "| {target} | {status} | {metric} | {value} | {best} | {notes} |".format(
                target=target,
                status=row.get("status", ""),
                metric=row.get("metric_name", ""),
                value=row.get("metric_value", ""),
                best=row.get("best_value", ""),
                notes=row.get("notes", ""),
            )
        )

    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append("- `results.tsv`: append-only metrics table")
    lines.append("- `agent_log.jsonl`: machine-readable loop + evaluation events")
    lines.append("- `campaign_report.md`: this report")
    if evidence_payload:
        lines.append("- `evidence/manifest.json`: retained accepted improvements")
    if submission_payload:
        lines.append("- `submission/agent.json`, `submission/agent_log.json`, `submission/submission_receipts.json`")
    if readiness_payload:
        lines.append("- `readiness_report.md`: readiness checklist")

    lines.append("")
    lines.append("## Pipeline Outputs")
    lines.append("")
    lines.append("| step | ok | key output |")
    lines.append("|---|---|---|")
    lines.append(
        "| loop | yes | {value} |".format(
            value=loop_payload.get("stop_reason", "completed") if loop_payload else "completed"
        )
    )
    if portfolio_payload is not None:
        lines.append(
            "| real_optimize | {ok} | accepted={acc} |".format(
                ok="yes" if portfolio_payload.get("ok", True) else "no",
                acc=portfolio_payload.get("total_accepted", "n/a"),
            )
        )
    if evidence_payload is not None:
        lines.append(
            "| evidence_pack | {ok} | {out} |".format(
                ok="yes" if evidence_payload.get("ok", True) else "no",
                out=evidence_payload.get("out_dir", evidence_payload.get("manifest", "evidence/")),
            )
        )
    if submission_payload is not None:
        lines.append(
            "| submission_pack | {ok} | {out} |".format(
                ok="yes" if submission_payload.get("ok", True) else "no",
                out=submission_payload.get("out_dir", "submission/"),
            )
        )
    if readiness_payload is not None:
        readiness_ok = "yes"
        checks = readiness_payload.get("checks")
        if isinstance(checks, list):
            required_failed = 0
            for check in checks:
                if not isinstance(check, dict):
                    continue
                name = str(check.get("name", ""))
                ok = bool(check.get("ok", False))
                if name != "recent_activity_24h" and not ok:
                    required_failed += 1
            readiness_ok = "yes" if required_failed == 0 else "no"
        lines.append(
            "| readiness_check | {ok} | {report} |".format(
                ok=readiness_ok,
                report=readiness_payload.get("report", "readiness_report.md"),
            )
        )

    lines.append("")
    lines.append("## Reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append("python3 campaign.py --fresh --synthesis-cook --real-profile fast")
    lines.append("```")

    REPORT.write_text("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a full AutoPoseidon campaign")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase stderr verbosity")
    parser.add_argument(
        "--debug-command-output",
        action="store_true",
        help="Print captured benchmark/build/test command stdout/stderr (truncated)",
    )
    parser.add_argument(
        "--debug-max-chars",
        type=int,
        default=6000,
        help="Max chars shown per command stream when debug output is enabled",
    )
    parser.add_argument("--fresh", action="store_true", help="Reset results.tsv and agent_log.jsonl before run")
    parser.add_argument(
        "--synthesis-cook",
        action="store_true",
        help="Enable a Synthesis-aligned pipeline (real optimize + evidence + submission + readiness)",
    )
    parser.add_argument(
        "--real-profile",
        choices=["none", "fast", "full"],
        default="fast",
        help="Real benchmark profile for Lean baseline targets",
    )
    parser.add_argument("--no-bootstrap", action="store_true", help="Skip bootstrap step for real targets")
    parser.add_argument("--loop-target", default="cairo_poseidon_style_t8", help="Target for the optimization loop")
    parser.add_argument("--loop-iterations", type=int, default=25, help="Loop iteration budget (0 = infinite)")
    parser.add_argument("--max-accepted", type=int, default=5, help="Stop loop after N accepted improvements")
    parser.add_argument(
        "--loop-artifacts",
        choices=["none", "accepted", "all"],
        default="accepted",
        help="Artifact mode passed to train.py",
    )
    parser.add_argument(
        "--real-optimize-rounds",
        type=int,
        default=0,
        help="Portfolio optimization rounds for real Lean source targets (0 = disabled)",
    )
    parser.add_argument(
        "--real-optimize-targets",
        default=DEFAULT_REAL_OPTIMIZE_TARGETS,
        help="Comma-separated source targets for real optimization",
    )
    parser.add_argument(
        "--real-optimize-batch-iterations",
        type=int,
        default=6,
        help="Iterations per target batch in portfolio optimization",
    )
    parser.add_argument(
        "--real-optimize-batch-max-accepted",
        type=int,
        default=1,
        help="Max accepted per batch in portfolio optimization",
    )
    parser.add_argument(
        "--real-optimize-schedule",
        choices=["round_robin", "ucb"],
        default="ucb",
        help="Scheduling strategy for real-source portfolio optimization",
    )
    parser.add_argument(
        "--real-optimize-ucb-explore",
        type=float,
        default=0.75,
        help="UCB exploration coefficient for real-source optimization",
    )
    parser.add_argument(
        "--build-evidence",
        action="store_true",
        help="Run evidence_pack.py after loop/portfolio",
    )
    parser.add_argument(
        "--build-submission",
        action="store_true",
        help="Run submission_pack.py after loop/portfolio",
    )
    parser.add_argument(
        "--strict-submission",
        action="store_true",
        help="Pass --strict to submission_pack.py",
    )
    parser.add_argument(
        "--build-readiness",
        action="store_true",
        help="Run readiness_check.py after loop/portfolio",
    )
    parser.add_argument(
        "--agent-name",
        default="AutoPoseidon",
        help="Agent name passed to submission_pack.py",
    )
    parser.add_argument(
        "--readiness-build-close-utc",
        default="",
        help="Optional override for readiness_check --build-close-utc",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.loop_iterations < 0:
        raise SystemExit("--loop-iterations must be >= 0")
    if args.real_optimize_rounds < 0:
        raise SystemExit("--real-optimize-rounds must be >= 0")

    if args.synthesis_cook:
        if args.real_optimize_rounds == 0:
            args.real_optimize_rounds = 2
        if not args.build_evidence:
            args.build_evidence = True
        if not args.build_submission:
            args.build_submission = True
        if not args.build_readiness:
            args.build_readiness = True

    start_row_count = 0 if args.fresh or not RESULTS.exists() else len(parse_results())

    if args.fresh:
        reset_outputs()

    # Baseline and loop target optimization.
    must_run(
        prepare_cmd(args, "baseline", "--target", args.loop_target, "--notes", "campaign_baseline"),
        label="baseline_loop_target",
        stream_stderr=args.verbose > 0,
    )
    loop_payload = must_run_json(
        train_cmd(
            args,
            "--target",
            args.loop_target,
            "--iterations",
            str(args.loop_iterations),
            "--max-accepted",
            str(args.max_accepted),
            "--artifacts",
            args.loop_artifacts,
        ),
        label="train_loop",
        stream_stderr=args.verbose > 0,
    )

    # Real baseline targets (active Lean stack context).
    real_targets: list[str] = []
    if args.real_profile == "fast":
        real_targets = ["leanmultisig_poseidon16_fast", "leanmultisig_xmss_fast"]
    elif args.real_profile == "full":
        real_targets = ["leanmultisig_poseidon16_full", "leanmultisig_xmss_fast"]

    if real_targets and not args.no_bootstrap:
        for target in real_targets:
            must_run(
                prepare_cmd(args, "bootstrap", "--target", target),
                label=f"bootstrap_{target}",
                stream_stderr=args.verbose > 0,
            )

    for target in real_targets:
        must_run(
            prepare_cmd(args, "baseline", "--target", target, "--notes", "campaign_real"),
            label=f"baseline_{target}",
            stream_stderr=args.verbose > 0,
        )

    portfolio_payload: dict[str, Any] | None = None
    if args.real_optimize_rounds > 0:
        portfolio_payload = must_run_json(
            portfolio_cmd(
                args,
                "--targets",
                args.real_optimize_targets,
                "--rounds",
                str(args.real_optimize_rounds),
                "--batch-iterations",
                str(args.real_optimize_batch_iterations),
                "--batch-max-accepted",
                str(args.real_optimize_batch_max_accepted),
                "--schedule",
                args.real_optimize_schedule,
                "--ucb-explore",
                str(args.real_optimize_ucb_explore),
                "--artifacts",
                args.loop_artifacts,
            ),
            label="portfolio_real_optimize",
            stream_stderr=args.verbose > 0,
        )

    evidence_payload: dict[str, Any] | None = None
    if args.build_evidence:
        evidence_payload = must_run_json(
            [sys.executable, str(EVIDENCE_PACK)],
            label="evidence_pack",
            stream_stderr=args.verbose > 0,
        )

    submission_payload: dict[str, Any] | None = None
    if args.build_submission:
        submission_argv = [
            sys.executable,
            str(SUBMISSION_PACK),
            "--agent-name",
            args.agent_name,
            "--task-category",
            "cryptographic_optimization",
            "--task-category",
            "autonomous_research",
        ]
        if args.strict_submission:
            submission_argv.append("--strict")
        submission_payload = must_run_json(
            submission_argv,
            label="submission_pack",
            stream_stderr=args.verbose > 0,
        )

    readiness_payload: dict[str, Any] | None = None
    if args.build_readiness:
        readiness_argv = [sys.executable, str(READINESS_CHECK)]
        if args.readiness_build_close_utc.strip():
            readiness_argv.extend(["--build-close-utc", args.readiness_build_close_utc.strip()])
        readiness_payload = must_run_json(
            readiness_argv,
            label="readiness_check",
            stream_stderr=args.verbose > 0,
        )

    rows = parse_results()
    run_rows = rows[start_row_count:]
    accepts = count_accepts(run_rows, args.loop_target)

    write_report(
        rows=run_rows,
        loop_target=args.loop_target,
        real_targets=real_targets,
        loop_iterations=args.loop_iterations,
        loop_accepts=accepts,
        loop_payload=loop_payload,
        portfolio_payload=portfolio_payload,
        evidence_payload=evidence_payload,
        submission_payload=submission_payload,
        readiness_payload=readiness_payload,
        synthesis_cook=args.synthesis_cook,
    )

    print(
        json.dumps(
            {
                "ok": True,
                "report": str(REPORT),
                "rows": len(run_rows),
                "accepted": accepts,
                "loop_payload": loop_payload,
                "portfolio_payload": portfolio_payload,
                "evidence_payload": evidence_payload,
                "submission_payload": submission_payload,
                "readiness_payload": readiness_payload,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
