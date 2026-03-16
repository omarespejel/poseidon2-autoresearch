#!/usr/bin/env python3
"""Adaptive multi-target orchestration for AutoPoseidon.

Runs short train.py batches across multiple targets so optimization can
continue even if one target plateaus.
"""

from __future__ import annotations

import argparse
import json
import math
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


def default_target_state() -> dict[str, Any]:
    return {"batches": 0, "accepted": 0, "best_metric": None, "plateau_streak": 0, "zero_streak": 0}


def portfolio_log(verbose: int, message: str, *, level: int = 1) -> None:
    if verbose < level:
        return
    print(f"[portfolio] {message}", file=sys.stderr, flush=True)


def resolve_state_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    return path


def load_totals_state(state_path: Path | None, targets: list[str]) -> dict[str, dict[str, Any]]:
    totals = {target: default_target_state() for target in targets}
    if state_path is None or not state_path.exists():
        return totals

    try:
        data = json.loads(state_path.read_text())
    except (OSError, json.JSONDecodeError):
        return totals
    raw_totals = data.get("totals")
    if not isinstance(raw_totals, dict):
        return totals

    for target in targets:
        raw_state = raw_totals.get(target)
        if not isinstance(raw_state, dict):
            continue
        loaded = default_target_state()
        try:
            loaded["batches"] = max(0, int(raw_state.get("batches", 0)))
        except (TypeError, ValueError):
            loaded["batches"] = 0
        try:
            loaded["accepted"] = max(0, int(raw_state.get("accepted", 0)))
        except (TypeError, ValueError):
            loaded["accepted"] = 0
        best_metric = raw_state.get("best_metric")
        if best_metric is not None:
            try:
                loaded["best_metric"] = float(best_metric)
            except (TypeError, ValueError):
                loaded["best_metric"] = None
        totals[target] = loaded
    return totals


def save_totals_state(state_path: Path | None, totals: dict[str, dict[str, Any]]) -> None:
    if state_path is None:
        return
    state_path.parent.mkdir(parents=True, exist_ok=True)
    persisted_totals = {
        target: {
            "batches": int(state.get("batches", 0)),
            "accepted": int(state.get("accepted", 0)),
            "best_metric": state.get("best_metric"),
        }
        for target, state in totals.items()
    }
    tmp_path = state_path.with_name(f"{state_path.name}.tmp")
    tmp_path.write_text(json.dumps({"totals": persisted_totals}, indent=2, sort_keys=True) + "\n")
    tmp_path.replace(state_path)


def run_batch(
    *,
    target: str,
    round_index: int,
    iterations: int,
    max_accepted: int,
    artifacts: str,
    confirm_repeats: int,
    verbose: int,
    debug_command_output: bool,
    debug_max_chars: int,
    target_overrides_json: str = "",
    mutation_memory_file: str = "",
    disable_mutation_memory: bool = False,
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
    if verbose > 0:
        argv.extend(["--verbose"] * verbose)
    if debug_command_output:
        argv.append("--debug-command-output")
    argv.extend(["--debug-max-chars", str(max(256, debug_max_chars))])
    if target_overrides_json:
        argv.extend(["--target-overrides-json", target_overrides_json])
    if mutation_memory_file:
        argv.extend(["--mutation-memory-file", mutation_memory_file])
    if disable_mutation_memory:
        argv.append("--disable-mutation-memory")
    if verbose > 0:
        proc = subprocess.run(
            argv,
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=None,
            check=False,
        )
    else:
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
        stderr=proc.stderr if proc.stderr is not None else "",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Adaptive multi-target AutoPoseidon runner")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase stderr verbosity")
    parser.add_argument(
        "--debug-command-output",
        action="store_true",
        help="Forward command stdout/stderr debugging to train.py/prepare.py",
    )
    parser.add_argument(
        "--debug-max-chars",
        type=int,
        default=4000,
        help="Max chars shown per command stream when debug output is enabled",
    )
    parser.add_argument(
        "--targets",
        default="leanmultisig_poseidon16_src_fast,leanmultisig_poseidon16_table_src_fast,leanmultisig_poseidon2_neon_src_fast",
        help="Comma-separated target list to cycle through",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=4,
        help="Number of portfolio rounds (0 = run indefinitely until stop-after-total-accepted is reached)",
    )
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
        "--schedule",
        choices=["round_robin", "ucb"],
        default="round_robin",
        help="Target scheduling strategy across rounds",
    )
    parser.add_argument(
        "--ucb-explore",
        type=float,
        default=0.75,
        help="Exploration strength used when --schedule ucb",
    )
    parser.add_argument(
        "--target-overrides-json",
        default="",
        help="Optional JSON file with per-target acceptance overrides",
    )
    parser.add_argument(
        "--state-json",
        default="",
        help="Optional JSON file used to persist per-target scheduling totals across invocations",
    )
    parser.add_argument(
        "--mutation-memory-file",
        default="",
        help="Optional path forwarded to train.py --mutation-memory-file",
    )
    parser.add_argument(
        "--disable-mutation-memory",
        action="store_true",
        help="Disable cross-target mutation replay in child train.py runs",
    )
    return parser


def write_reports(
    *,
    rows: list[BatchResult],
    totals: dict[str, dict[str, Any]],
    rounds: int,
    batch_iterations: int,
    batch_max_accepted: int,
    schedule: str,
    ucb_explore: float,
    target_overrides_json: str,
    state_json: str,
    mutation_memory_file: str,
    disable_mutation_memory: bool,
) -> None:
    REPORT_JSON.write_text(
        json.dumps(
            {
                "rounds": rounds,
                "batch_iterations": batch_iterations,
                "batch_max_accepted": batch_max_accepted,
                "schedule": schedule,
                "ucb_explore": ucb_explore,
                "target_overrides_json": target_overrides_json,
                "state_json": state_json,
                "mutation_memory_file": mutation_memory_file,
                "disable_mutation_memory": disable_mutation_memory,
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
    lines.append(f"- schedule: `{schedule}`")
    if schedule == "ucb":
        lines.append(f"- ucb explore: `{ucb_explore}`")
    if target_overrides_json:
        lines.append(f"- target overrides: `{target_overrides_json}`")
    if state_json:
        lines.append(f"- state json: `{state_json}`")
    if mutation_memory_file:
        lines.append(f"- mutation memory file: `{mutation_memory_file}`")
    lines.append(f"- mutation memory disabled: `{str(disable_mutation_memory).lower()}`")
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


def ucb_score(state: dict[str, Any], *, total_batches: int, explore: float, max_reward: int) -> float:
    batches = int(state.get("batches", 0))
    accepted = int(state.get("accepted", 0))
    if batches <= 0:
        return float("inf")
    mean_reward = (accepted / batches) / max(1, max_reward)
    bonus = max(0.0, explore) * math.sqrt(math.log(max(2, total_batches)) / batches)
    return mean_reward + bonus


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.rounds < 0:
        print("--rounds must be >= 0", file=sys.stderr)
        return 2
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    if not targets:
        print("No targets configured", file=sys.stderr)
        return 2
    if args.ucb_explore < 0.0:
        print("--ucb-explore must be >= 0", file=sys.stderr)
        return 2

    state_path = resolve_state_path(args.state_json) if args.state_json else None
    totals = load_totals_state(state_path, targets)
    rows: list[BatchResult] = []
    total_accepted = 0
    rounds_executed = 0
    max_rounds = args.rounds if args.rounds > 0 else None
    rounds_label = str(args.rounds) if max_rounds is not None else "infinite"

    round_index = 0
    while True:
        round_index += 1
        if max_rounds is not None and round_index > max_rounds:
            break
        rounds_executed = round_index
        made_progress = False
        portfolio_log(args.verbose, f"starting round {round_index}/{rounds_label}", level=1)
        remaining_targets = [target for target in targets if totals[target]["plateau_streak"] < args.plateau_threshold]

        while remaining_targets:
            if args.schedule == "ucb":
                total_batches = sum(int(item["batches"]) for item in totals.values())
                target = max(
                    remaining_targets,
                    key=lambda name: ucb_score(
                        totals[name],
                        total_batches=total_batches,
                        explore=args.ucb_explore,
                        max_reward=args.batch_max_accepted,
                    ),
                )
                remaining_targets.remove(target)
            else:
                target = remaining_targets.pop(0)

            portfolio_log(
                args.verbose,
                (
                    f"dispatch target={target} schedule={args.schedule} "
                    f"plateau_streak={totals[target]['plateau_streak']}"
                ),
                level=2,
            )

            state = totals[target]

            result = run_batch(
                target=target,
                round_index=round_index,
                iterations=args.batch_iterations,
                max_accepted=args.batch_max_accepted,
                artifacts=args.artifacts,
                confirm_repeats=args.confirm_repeats,
                verbose=args.verbose,
                debug_command_output=args.debug_command_output,
                debug_max_chars=args.debug_max_chars,
                target_overrides_json=args.target_overrides_json,
                mutation_memory_file=args.mutation_memory_file,
                disable_mutation_memory=args.disable_mutation_memory,
            )
            rows.append(result)
            portfolio_log(
                args.verbose,
                (
                    f"round={round_index} target={target} exit={result.code} "
                    f"accepted={result.accepted} best_metric="
                    f"{'n/a' if result.best_metric is None else f'{result.best_metric:.6f}'}"
                ),
                level=1,
            )
            if result.code != 0 and result.stderr.strip():
                portfolio_log(args.verbose, f"stderr tail:\n{result.stderr[-2000:]}", level=1)

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

            save_totals_state(state_path, totals)

            total_accepted += result.accepted
            if args.stop_after_total_accepted and total_accepted >= args.stop_after_total_accepted:
                write_reports(
                    rows=rows,
                    totals=totals,
                    rounds=round_index,
                    batch_iterations=args.batch_iterations,
                    batch_max_accepted=args.batch_max_accepted,
                    schedule=args.schedule,
                    ucb_explore=args.ucb_explore,
                    target_overrides_json=args.target_overrides_json,
                    state_json=args.state_json,
                    mutation_memory_file=args.mutation_memory_file,
                    disable_mutation_memory=args.disable_mutation_memory,
                )
                print(
                    json.dumps(
                        {
                            "ok": True,
                            "rounds_executed": round_index,
                            "infinite_rounds": max_rounds is None,
                            "total_accepted": total_accepted,
                            "schedule": args.schedule,
                            "ucb_explore": args.ucb_explore,
                            "target_overrides_json": args.target_overrides_json,
                            "state_json": args.state_json,
                            "mutation_memory_file": args.mutation_memory_file,
                            "disable_mutation_memory": args.disable_mutation_memory,
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
                state["zero_streak"] = 0

    write_reports(
        rows=rows,
        totals=totals,
        rounds=rounds_executed,
        batch_iterations=args.batch_iterations,
        batch_max_accepted=args.batch_max_accepted,
        schedule=args.schedule,
        ucb_explore=args.ucb_explore,
        target_overrides_json=args.target_overrides_json,
        state_json=args.state_json,
        mutation_memory_file=args.mutation_memory_file,
        disable_mutation_memory=args.disable_mutation_memory,
    )
    save_totals_state(state_path, totals)

    print(
        json.dumps(
            {
                "ok": True,
                "rounds_executed": rounds_executed,
                "infinite_rounds": max_rounds is None,
                "total_accepted": total_accepted,
                "schedule": args.schedule,
                "ucb_explore": args.ucb_explore,
                "target_overrides_json": args.target_overrides_json,
                "state_json": args.state_json,
                "mutation_memory_file": args.mutation_memory_file,
                "disable_mutation_memory": args.disable_mutation_memory,
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
