#!/usr/bin/env python3
"""Adaptive multi-target orchestration for AutoPoseidon.

Runs short train.py batches across multiple targets so optimization can
continue even if one target plateaus.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
TRAIN = ROOT / "train.py"
REPORT_MD = ROOT / "portfolio_report.md"
REPORT_JSON = ROOT / "portfolio_report.json"


@dataclass
class BatchResult:
    target: str
    round_index: int
    code: int
    accepted: int
    best_metric: float | None
    payload: dict[str, Any]
    stderr: str


def run_batch(
    *,
    target: str,
    round_index: int,
    iterations: int,
    max_accepted: int,
    artifacts: str,
    confirm_repeats: int,
    target_overrides_json: str = "",
) -> BatchResult:
    argv = [
        sys.executable,
        str(TRAIN),
        "--target",
        target,
        "--iterations",
        str(iterations),
        "--max-accepted",
        str(max_accepted),
        "--artifacts",
        artifacts,
        "--confirm-repeats",
        str(confirm_repeats),
    ]
    if target_overrides_json:
        argv.extend(["--target-overrides-json", target_overrides_json])
    proc = subprocess.run(argv, cwd=str(ROOT), text=True, capture_output=True, check=False)

    payload: dict[str, Any] = {}
    accepted = 0
    best_metric: float | None = None

    stdout = proc.stdout.strip()
    if stdout:
        try:
            payload = json.loads(stdout)
            accepted = int(payload.get("accepted", 0))
            metric = payload.get("best_metric")
            if metric is not None:
                best_metric = float(metric)
        except json.JSONDecodeError:
            payload = {"stdout_parse_error": True, "stdout_tail": stdout[-1000:]}

    return BatchResult(
        target=target,
        round_index=round_index,
        code=proc.returncode,
        accepted=accepted,
        best_metric=best_metric,
        payload=payload,
        stderr=proc.stderr,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Adaptive multi-target AutoPoseidon runner")
    parser.add_argument(
        "--targets",
        default="leanmultisig_poseidon16_src_fast,leanmultisig_poseidon16_table_src_fast,leanmultisig_poseidon2_neon_src_fast",
        help="Comma-separated target list to cycle through",
    )
    parser.add_argument("--rounds", type=int, default=4, help="Number of portfolio rounds")
    parser.add_argument("--batch-iterations", type=int, default=6, help="Iterations per target batch")
    parser.add_argument("--batch-max-accepted", type=int, default=1, help="Max accepted changes per target batch")
    parser.add_argument(
        "--artifacts",
        choices=["none", "accepted", "all"],
        default="accepted",
        help="Artifact mode passed to train.py",
    )
    parser.add_argument(
        "--confirm-repeats",
        type=int,
        default=0,
        help="Confirmation re-runs passed to train.py (target config may override in train.py)",
    )
    parser.add_argument(
        "--plateau-threshold",
        type=int,
        default=2,
        help="Skip a target after this many consecutive zero-accept batches",
    )
    parser.add_argument(
        "--stop-after-total-accepted",
        type=int,
        default=0,
        help="Stop early after this many accepted changes (0 disables)",
    )
    parser.add_argument(
        "--target-overrides-json",
        default="",
        help="Optional JSON file with per-target acceptance overrides",
    )
    return parser


def write_reports(
    *,
    rows: list[BatchResult],
    totals: dict[str, dict[str, Any]],
    rounds: int,
    batch_iterations: int,
    batch_max_accepted: int,
    target_overrides_json: str,
) -> None:
    REPORT_JSON.write_text(
        json.dumps(
            {
                "rounds": rounds,
                "batch_iterations": batch_iterations,
                "batch_max_accepted": batch_max_accepted,
                "target_overrides_json": target_overrides_json,
                "totals": totals,
                "rows": [
                    {
                        "round": row.round_index,
                        "target": row.target,
                        "code": row.code,
                        "accepted": row.accepted,
                        "best_metric": row.best_metric,
                    }
                    for row in rows
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )

    lines: list[str] = []
    lines.append("# AutoPoseidon Portfolio Report")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- rounds: `{rounds}`")
    lines.append(f"- batch iterations: `{batch_iterations}`")
    lines.append(f"- batch max accepted: `{batch_max_accepted}`")
    if target_overrides_json:
        lines.append(f"- target overrides: `{target_overrides_json}`")
    lines.append("")
    lines.append("## Per Target Totals")
    lines.append("")
    lines.append("| target | batches | accepted | last best metric | consecutive zero-accept batches |")
    lines.append("|---|---:|---:|---:|---:|")
    for target, item in totals.items():
        metric = item.get("best_metric")
        metric_text = "n/a" if metric is None else f"{metric:.6f}"
        lines.append(
            f"| {target} | {item['batches']} | {item['accepted']} | {metric_text} | {item['zero_streak']} |"
        )

    lines.append("")
    lines.append("## Batch Timeline")
    lines.append("")
    lines.append("| round | target | exit | accepted | best metric |")
    lines.append("|---:|---|---:|---:|---:|")
    for row in rows:
        metric_text = "n/a" if row.best_metric is None else f"{row.best_metric:.6f}"
        lines.append(f"| {row.round_index} | {row.target} | {row.code} | {row.accepted} | {metric_text} |")

    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append("- `portfolio_report.md`")
    lines.append("- `portfolio_report.json`")
    lines.append("- `results.tsv`")
    lines.append("- `agent_log.jsonl`")
    lines.append("")
    lines.append("## Reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append(
        "python3 portfolio_loop.py --rounds 4 --batch-iterations 6 --batch-max-accepted 1 --artifacts accepted"
    )
    lines.append("```")
    REPORT_MD.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    if not targets:
        print("No targets configured", file=sys.stderr)
        return 2

    totals: dict[str, dict[str, Any]] = {
        t: {"batches": 0, "accepted": 0, "best_metric": None, "plateau_streak": 0, "zero_streak": 0} for t in targets
    }
    rows: list[BatchResult] = []
    total_accepted = 0

    for round_index in range(1, args.rounds + 1):
        made_progress = False
        for target in targets:
            state = totals[target]
            if state["plateau_streak"] >= args.plateau_threshold:
                continue

            result = run_batch(
                target=target,
                round_index=round_index,
                iterations=args.batch_iterations,
                max_accepted=args.batch_max_accepted,
                artifacts=args.artifacts,
                confirm_repeats=args.confirm_repeats,
                target_overrides_json=args.target_overrides_json,
            )
            rows.append(result)

            state["batches"] += 1
            state["accepted"] += result.accepted
            if result.best_metric is not None:
                state["best_metric"] = result.best_metric

            if result.accepted > 0:
                state["plateau_streak"] = 0
                state["zero_streak"] = 0
                made_progress = True
            else:
                state["plateau_streak"] += 1
                state["zero_streak"] += 1

            total_accepted += result.accepted
            if args.stop_after_total_accepted and total_accepted >= args.stop_after_total_accepted:
                write_reports(
                    rows=rows,
                    totals=totals,
                    rounds=round_index,
                    batch_iterations=args.batch_iterations,
                    batch_max_accepted=args.batch_max_accepted,
                    target_overrides_json=args.target_overrides_json,
                )
                print(
                    json.dumps(
                        {
                            "ok": True,
                            "rounds_executed": round_index,
                            "total_accepted": total_accepted,
                            "target_overrides_json": args.target_overrides_json,
                            "report_md": str(REPORT_MD),
                            "report_json": str(REPORT_JSON),
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
                return 0

        # If everything plateaued in this round, reset streaks to force a fresh sweep.
        if not made_progress:
            for state in totals.values():
                state["plateau_streak"] = 0

    write_reports(
        rows=rows,
        totals=totals,
        rounds=args.rounds,
        batch_iterations=args.batch_iterations,
        batch_max_accepted=args.batch_max_accepted,
        target_overrides_json=args.target_overrides_json,
    )

    print(
        json.dumps(
            {
                "ok": True,
                "rounds_executed": args.rounds,
                "total_accepted": total_accepted,
                "target_overrides_json": args.target_overrides_json,
                "report_md": str(REPORT_MD),
                "report_json": str(REPORT_JSON),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
