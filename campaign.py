#!/usr/bin/env python3
"""Reproducible campaign runner for AutoPoseidon.

Runs a dual-track execution:
- Autonomous optimization loop (Cairo target by default)
- Real EF-adjacent benchmark targets (Lean stack)
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PREPARE = ROOT / "prepare.py"
LOOP = ROOT / "train.py"
RESULTS = ROOT / "results.tsv"
LOG = ROOT / "agent_log.jsonl"
REPORT = ROOT / "campaign_report.md"


@dataclass
class RunResult:
    argv: list[str]
    code: int
    stdout: str
    stderr: str


def run(argv: list[str]) -> RunResult:
    proc = subprocess.run(argv, cwd=str(ROOT), text=True, capture_output=True, check=False)
    return RunResult(argv=argv, code=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)


def must_run(argv: list[str]) -> RunResult:
    result = run(argv)
    if result.code != 0:
        sys.stderr.write(f"Command failed ({result.code}): {' '.join(argv)}\n")
        if result.stdout:
            sys.stderr.write(result.stdout + "\n")
        if result.stderr:
            sys.stderr.write(result.stderr + "\n")
        raise SystemExit(result.code)
    return result


def reset_outputs() -> None:
    RESULTS.write_text(
        "timestamp\ttarget\titeration\tstatus\tmetric_name\tmetric_value\tbest_value\tdelta\tcheck_s\tinfo_or_bench_s\texecute_s\tnotes\n"
    )
    LOG.write_text("")


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


def write_report(
    *,
    rows: list[dict[str, str]],
    loop_target: str,
    real_targets: list[str],
    loop_iterations: int,
    loop_accepts: int,
) -> None:
    latest = summarize(rows)

    lines: list[str] = []
    lines.append("# AutoPoseidon Campaign Report")
    lines.append("")
    lines.append("## Run Summary")
    lines.append("")
    lines.append(f"- Loop target: `{loop_target}`")
    lines.append(f"- Loop iterations requested: `{loop_iterations}`")
    lines.append(f"- Loop accepted improvements: `{loop_accepts}`")
    lines.append(f"- Real targets: `{', '.join(real_targets) if real_targets else 'none'}`")
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
    lines.append("")
    lines.append("## Reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append("python3 campaign.py --fresh --real-profile fast --loop-iterations 25")
    lines.append("```")

    REPORT.write_text("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a full AutoPoseidon campaign")
    parser.add_argument("--fresh", action="store_true", help="Reset results.tsv and agent_log.jsonl before run")
    parser.add_argument(
        "--real-profile",
        choices=["none", "fast", "full"],
        default="fast",
        help="Real benchmark profile for Lean targets",
    )
    parser.add_argument("--no-bootstrap", action="store_true", help="Skip bootstrap step for real targets")
    parser.add_argument("--loop-target", default="cairo_poseidon_style_t8", help="Target for the optimization loop")
    parser.add_argument("--loop-iterations", type=int, default=25, help="Loop iteration budget")
    parser.add_argument("--max-accepted", type=int, default=5, help="Stop loop after N accepted improvements")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    start_row_count = 0 if args.fresh or not RESULTS.exists() else len(parse_results())

    if args.fresh:
        reset_outputs()

    # Baseline loop target and run optimization loop.
    must_run([sys.executable, str(PREPARE), "baseline", "--target", args.loop_target, "--notes", "campaign_baseline"])
    loop_res = must_run(
        [
            sys.executable,
            str(LOOP),
            "--target",
            args.loop_target,
            "--iterations",
            str(args.loop_iterations),
            "--max-accepted",
            str(args.max_accepted),
        ]
    )

    try:
        loop_payload = json.loads(loop_res.stdout.strip() or "{}")
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"Warning: failed to parse train.py stdout as JSON: {exc}; stdout={loop_res.stdout!r}\n")
        loop_payload = {}

    # Real targets (EF-adjacent stack).
    real_targets: list[str] = []
    if args.real_profile == "fast":
        real_targets = ["leanmultisig_poseidon16_fast", "leanmultisig_xmss_fast"]
    elif args.real_profile == "full":
        real_targets = ["leanmultisig_poseidon16_full", "leanmultisig_xmss_fast"]

    if real_targets and not args.no_bootstrap:
        for target in real_targets:
            must_run([sys.executable, str(PREPARE), "bootstrap", "--target", target])

    for target in real_targets:
        must_run([sys.executable, str(PREPARE), "baseline", "--target", target, "--notes", "campaign_real"])

    rows = parse_results()
    run_rows = rows[start_row_count:]
    accepts = count_accepts(run_rows, args.loop_target)
    loop_accepts_reported = loop_payload.get("accepted")
    write_report(
        rows=run_rows,
        loop_target=args.loop_target,
        real_targets=real_targets,
        loop_iterations=args.loop_iterations,
        loop_accepts=accepts,
    )

    print(
        json.dumps(
            {
                "ok": True,
                "report": str(REPORT),
                "rows": len(run_rows),
                "accepted": accepts,
                "loop_accepts_reported": loop_accepts_reported,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
