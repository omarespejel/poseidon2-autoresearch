#!/usr/bin/env python3
"""Checkpointed autoresearch runner.

Runs repeated cycles of:
1) target calibration snapshots
2) one portfolio optimization sweep
3) optional cross-target regression check
4) evidence/readiness refresh
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
PREPARE = ROOT / "prepare.py"
PORTFOLIO = ROOT / "portfolio_loop.py"
EVIDENCE = ROOT / "evidence_pack.py"
READINESS = ROOT / "readiness_check.py"
REPORT_JSON = ROOT / "checkpoint_report.json"
REPORT_MD = ROOT / "checkpoint_report.md"
DEFAULT_OVERRIDES_PATH = ROOT / "work" / "checkpoint_target_overrides.json"
WORK_DIR = DEFAULT_OVERRIDES_PATH.parent


@dataclass
class CmdResult:
    argv: list[str]
    cwd: Path
    code: int
    stdout: str
    stderr: str


def run_cmd(argv: list[str], *, extra_env: dict[str, str] | None = None) -> CmdResult:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run(argv, cwd=str(ROOT), text=True, capture_output=True, env=env, check=False)
    return CmdResult(argv=argv, cwd=ROOT, code=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)


def must_run(argv: list[str], *, extra_env: dict[str, str] | None = None) -> CmdResult:
    result = run_cmd(argv, extra_env=extra_env)
    if result.code != 0:
        sys.stderr.write(f"Command failed ({result.code}): {' '.join(argv)}\n")
        if result.stdout:
            sys.stderr.write(result.stdout + "\n")
        if result.stderr:
            sys.stderr.write(result.stderr + "\n")
        raise SystemExit(result.code)
    return result


def must_json(argv: list[str], *, extra_env: dict[str, str] | None = None) -> tuple[dict[str, Any], CmdResult]:
    result = must_run(argv, extra_env=extra_env)
    try:
        payload = json.loads(result.stdout.strip() or "{}")
    except json.JSONDecodeError:
        sys.stderr.write(f"JSON parse error for command: {' '.join(argv)}\n")
        if result.stdout:
            sys.stderr.write(result.stdout + "\n")
        raise SystemExit(1) from None
    return payload, result


def resolve_overrides_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    return path


def clamp_optional(value: float, *, floor: float = 0.0, cap: float | None = None) -> float:
    out = max(floor, value)
    if cap is not None and cap > 0.0:
        out = min(out, cap)
    return out


def build_cycle_overrides(
    *,
    calibration: dict[str, dict[str, Any]],
    rel_floor: float,
    rel_cap: float | None,
    noise_multiplier: float,
    noise_floor: float,
    noise_cap: float | None,
    force_fixed_mode: bool,
) -> dict[str, dict[str, Any]]:
    overrides: dict[str, dict[str, Any]] = {}
    for target, stats in calibration.items():
        row: dict[str, Any] = {}

        rec_rel = stats.get("recommended_min_improvement_rel")
        try:
            rec_rel_f = float(rec_rel)
        except Exception:  # noqa: BLE001
            rec_rel_f = None
        if rec_rel_f is not None:
            row["min_improvement_rel"] = clamp_optional(rec_rel_f, floor=max(0.0, rel_floor), cap=rel_cap)
            if force_fixed_mode:
                row["min_improvement_rel_mode"] = "fixed"

        rel_stdev = stats.get("rel_stdev")
        try:
            rel_stdev_f = float(rel_stdev)
        except Exception:  # noqa: BLE001
            rel_stdev_f = None
        if rel_stdev_f is not None:
            guard = clamp_optional(
                rel_stdev_f * max(0.0, noise_multiplier),
                floor=max(0.0, noise_floor),
                cap=noise_cap,
            )
            row["max_rel_stdev"] = guard

        if row:
            overrides[target] = row

    return overrides


def write_reports(payload: dict[str, Any]) -> None:
    REPORT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True))

    lines: list[str] = []
    lines.append("# AutoPoseidon Checkpoint Report")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    cfg = payload["config"]
    lines.append(f"- cycles: `{cfg['cycles']}`")
    lines.append(f"- targets: `{', '.join(cfg['targets'])}`")
    lines.append(f"- calibration samples: `{cfg['calibration_samples']}`")
    lines.append(f"- batch iterations: `{cfg['batch_iterations']}`")
    lines.append(f"- batch max accepted: `{cfg['batch_max_accepted']}`")
    lines.append(f"- schedule: `{cfg['schedule']}`")
    if cfg.get("schedule") == "ucb":
        lines.append(f"- ucb explore: `{cfg['ucb_explore']}`")
    lines.append(f"- auto threshold overrides: `{cfg['auto_threshold_overrides']}`")
    if cfg.get("auto_threshold_overrides"):
        lines.append(f"- overrides path: `{cfg['overrides_path']}`")
    lines.append("")
    lines.append("## Cycle Summary")
    lines.append("")
    lines.append("| cycle | accepted (portfolio) | cross-target status | max rel stdev | overrides |")
    lines.append("|---:|---:|---|---:|---:|")
    for cycle in payload["cycles"]:
        max_rel = cycle.get("max_rel_stdev")
        max_rel_text = "n/a" if max_rel is None else f"{max_rel:.6f}"
        override_count = len(cycle.get("target_overrides", {}))
        lines.append(
            f"| {cycle['cycle']} | {cycle.get('portfolio_accepted', 0)} | {cycle.get('cross_target_status', 'n/a')} | {max_rel_text} | {override_count} |"
        )

    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    lines.append("- `checkpoint_report.md`")
    lines.append("- `checkpoint_report.json`")
    lines.append("- `portfolio_report.md` / `portfolio_report.json` (last cycle)")
    lines.append("- `evidence/manifest.json` / `evidence/summary.md`")
    lines.append("- `readiness_report.md`")
    lines.append("")
    lines.append("## Reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append(
        "python3 checkpoint_loop.py --cycles 3 --calibration-samples 3 --batch-iterations 6 --batch-max-accepted 1"
    )
    lines.append("```")
    REPORT_MD.write_text("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run checkpointed autoresearch cycles")
    parser.add_argument(
        "--targets",
        default="leanmultisig_poseidon16_src_fast,leanmultisig_poseidon16_table_src_fast,leanmultisig_poseidon2_neon_src_fast",
        help="Comma-separated optimization targets",
    )
    parser.add_argument("--cycles", type=int, default=3, help="Number of checkpoint cycles")
    parser.add_argument("--calibration-samples", type=int, default=3, help="Samples per calibrate call")
    parser.add_argument("--sigma-multiplier", type=float, default=2.0, help="Calibrate sigma multiplier")
    parser.add_argument("--batch-iterations", type=int, default=6, help="Train iterations per target in each cycle")
    parser.add_argument("--batch-max-accepted", type=int, default=1, help="Max accepted edits per target per cycle")
    parser.add_argument(
        "--schedule",
        choices=["round_robin", "ucb"],
        default="round_robin",
        help="Scheduling strategy passed to portfolio_loop.py",
    )
    parser.add_argument("--ucb-explore", type=float, default=0.75, help="UCB exploration strength for portfolio scheduling")
    parser.add_argument(
        "--artifacts",
        choices=["none", "accepted", "all"],
        default="accepted",
        help="Artifact mode passed to portfolio loop",
    )
    parser.add_argument("--confirm-repeats", type=int, default=0, help="Confirm repeats passed to portfolio loop")
    parser.add_argument(
        "--cross-target",
        default="leanmultisig_xmss_fast",
        help="Cross-target regression target checked every cycle (empty disables)",
    )
    parser.add_argument(
        "--auto-threshold-overrides",
        action="store_true",
        help="Generate per-target acceptance threshold overrides from calibration each cycle",
    )
    parser.add_argument("--override-rel-floor", type=float, default=0.0, help="Lower bound for relative threshold overrides")
    parser.add_argument("--override-rel-cap", type=float, default=0.05, help="Upper cap for relative threshold overrides")
    parser.add_argument(
        "--override-noise-multiplier",
        type=float,
        default=2.0,
        help="Multiply rel_stdev to derive max_rel_stdev override",
    )
    parser.add_argument("--override-noise-floor", type=float, default=0.0, help="Lower bound for max_rel_stdev overrides")
    parser.add_argument("--override-noise-cap", type=float, default=0.25, help="Upper cap for max_rel_stdev overrides")
    parser.add_argument(
        "--override-force-fixed-mode",
        action="store_true",
        help="Force min_improvement_rel_mode=fixed in generated overrides",
    )
    parser.add_argument(
        "--overrides-path",
        default=str(DEFAULT_OVERRIDES_PATH),
        help="Path for generated target-overrides JSON file",
    )
    parser.add_argument("--skip-evidence", action="store_true", help="Skip evidence/readiness refresh")
    parser.add_argument("--nice", default="", help="AUTORESEARCH_NICE value for subprocesses")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    if not targets:
        print("No optimization targets configured", file=sys.stderr)
        return 2

    extra_env: dict[str, str] = {}
    if args.nice:
        extra_env["AUTORESEARCH_NICE"] = args.nice
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    bootstrap_targets = list(dict.fromkeys([*targets, args.cross_target.strip()] if args.cross_target.strip() else targets))
    for target in bootstrap_targets:
        must_run([sys.executable, str(PREPARE), "bootstrap", "--target", target], extra_env=extra_env)

    cycles: list[dict[str, Any]] = []
    total_accepted = 0

    for cycle_idx in range(1, args.cycles + 1):
        cycle_data: dict[str, Any] = {"cycle": cycle_idx, "calibration": {}}

        max_rel_stdev: float | None = None
        for target in targets:
            calib_payload, _ = must_json(
                [
                    sys.executable,
                    str(PREPARE),
                    "calibrate",
                    "--target",
                    target,
                    "--samples",
                    str(args.calibration_samples),
                    "--sigma-multiplier",
                    str(args.sigma_multiplier),
                ],
                extra_env=extra_env,
            )
            stats = calib_payload.get("stats", {})
            rel_stdev = stats.get("rel_stdev")
            try:
                rel_val = float(rel_stdev)
            except Exception:  # noqa: BLE001
                rel_val = None
            cycle_data["calibration"][target] = {
                "samples_success": calib_payload.get("samples_success"),
                "samples_failed": calib_payload.get("samples_failed"),
                "median": stats.get("median"),
                "rel_stdev": rel_val,
                "recommended_min_improvement_rel": calib_payload.get("recommendation", {}).get("min_improvement_rel"),
                "configured_min_improvement_rel": calib_payload.get("configured", {}).get("min_improvement_rel"),
                "configured_max_rel_stdev": calib_payload.get("configured", {}).get("max_rel_stdev"),
            }
            if rel_val is not None:
                max_rel_stdev = rel_val if max_rel_stdev is None else max(max_rel_stdev, rel_val)

        cycle_data["max_rel_stdev"] = max_rel_stdev

        cycle_overrides: dict[str, dict[str, Any]] = {}
        overrides_path = resolve_overrides_path(args.overrides_path)
        if args.auto_threshold_overrides:
            rel_cap = args.override_rel_cap if args.override_rel_cap > 0.0 else None
            noise_cap = args.override_noise_cap if args.override_noise_cap > 0.0 else None
            cycle_overrides = build_cycle_overrides(
                calibration=cycle_data["calibration"],
                rel_floor=args.override_rel_floor,
                rel_cap=rel_cap,
                noise_multiplier=args.override_noise_multiplier,
                noise_floor=args.override_noise_floor,
                noise_cap=noise_cap,
                force_fixed_mode=args.override_force_fixed_mode,
            )
            overrides_path.parent.mkdir(parents=True, exist_ok=True)
            overrides_path.write_text(json.dumps(cycle_overrides, indent=2, sort_keys=True) + "\n")
            cycle_data["target_overrides"] = cycle_overrides
            cycle_data["target_overrides_path"] = str(overrides_path)

        portfolio_argv = [
            sys.executable,
            str(PORTFOLIO),
            "--targets",
            ",".join(targets),
            "--rounds",
            "1",
            "--batch-iterations",
            str(args.batch_iterations),
            "--batch-max-accepted",
            str(args.batch_max_accepted),
            "--schedule",
            args.schedule,
            "--ucb-explore",
            str(args.ucb_explore),
            "--artifacts",
            args.artifacts,
            "--confirm-repeats",
            str(args.confirm_repeats),
        ]
        if args.auto_threshold_overrides:
            portfolio_argv.extend(["--target-overrides-json", str(overrides_path)])
        portfolio_payload, _ = must_json(portfolio_argv, extra_env=extra_env)
        accepted = int(portfolio_payload.get("total_accepted", 0))
        total_accepted += accepted
        cycle_data["portfolio_accepted"] = accepted
        cycle_data["portfolio_report"] = portfolio_payload.get("report_json")

        cross_target = args.cross_target.strip()
        if cross_target:
            cross_payload, _ = must_json(
                [
                    sys.executable,
                    str(PREPARE),
                    "evaluate",
                    "--target",
                    cross_target,
                    "--iteration",
                    str(cycle_idx),
                    "--notes",
                    f"checkpoint_cycle_{cycle_idx}",
                ],
                extra_env=extra_env,
            )
            cycle_data["cross_target"] = cross_target
            cycle_data["cross_target_status"] = cross_payload.get("status")
            cycle_data["cross_target_metric"] = cross_payload.get("metric_value")
        else:
            cycle_data["cross_target_status"] = "skipped"

        cycles.append(cycle_data)

    evidence_payload: dict[str, Any] | None = None
    readiness_payload: dict[str, Any] | None = None
    if not args.skip_evidence:
        evidence_payload, _ = must_json([sys.executable, str(EVIDENCE)], extra_env=extra_env)
        readiness_payload, _ = must_json([sys.executable, str(READINESS)], extra_env=extra_env)

    payload = {
        "ok": True,
        "config": {
            "targets": targets,
            "cycles": args.cycles,
            "calibration_samples": args.calibration_samples,
            "sigma_multiplier": args.sigma_multiplier,
            "batch_iterations": args.batch_iterations,
            "batch_max_accepted": args.batch_max_accepted,
            "schedule": args.schedule,
            "ucb_explore": args.ucb_explore,
            "artifacts": args.artifacts,
            "confirm_repeats": args.confirm_repeats,
            "cross_target": args.cross_target,
            "auto_threshold_overrides": args.auto_threshold_overrides,
            "override_rel_floor": args.override_rel_floor,
            "override_rel_cap": args.override_rel_cap,
            "override_noise_multiplier": args.override_noise_multiplier,
            "override_noise_floor": args.override_noise_floor,
            "override_noise_cap": args.override_noise_cap,
            "override_force_fixed_mode": args.override_force_fixed_mode,
            "overrides_path": str(resolve_overrides_path(args.overrides_path)),
            "nice": args.nice,
        },
        "total_accepted": total_accepted,
        "cycles": cycles,
        "evidence": evidence_payload,
        "readiness": readiness_payload,
        "report_json": str(REPORT_JSON),
        "report_md": str(REPORT_MD),
    }
    write_reports(payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
