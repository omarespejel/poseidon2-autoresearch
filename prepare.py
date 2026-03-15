#!/usr/bin/env python3
"""AutoPoseidon preparation and evaluation harness."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
TARGETS_FILE = ROOT / "config" / "targets.json"
RESULTS_FILE = ROOT / "results.tsv"
LOG_FILE = ROOT / "agent_log.jsonl"


@dataclass
class CommandResult:
    argv: list[str]
    cwd: Path
    code: int
    stdout: str
    stderr: str
    seconds: float


class ToolError(RuntimeError):
    pass


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def command_result_to_json(result: CommandResult) -> dict[str, Any]:
    return {
        "argv": result.argv,
        "cwd": str(result.cwd),
        "code": result.code,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "seconds": result.seconds,
    }


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def run_cmd(argv: list[str], cwd: Path) -> CommandResult:
    cpu_affinity = os.getenv("AUTORESEARCH_CPU_AFFINITY", "").strip()
    nice_level = os.getenv("AUTORESEARCH_NICE", "").strip()

    final_argv = list(argv)
    if cpu_affinity and shutil.which("taskset"):
        final_argv = ["taskset", "-c", cpu_affinity, *final_argv]
    if nice_level:
        final_argv = ["nice", "-n", nice_level, *final_argv]

    start = time.perf_counter()
    proc = subprocess.run(final_argv, cwd=str(cwd), text=True, capture_output=True)
    return CommandResult(
        argv=final_argv,
        cwd=cwd,
        code=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        seconds=time.perf_counter() - start,
    )


def load_targets() -> dict[str, dict[str, Any]]:
    data = json.loads(TARGETS_FILE.read_text())
    return data["targets"]


def ensure_outputs() -> None:
    if not RESULTS_FILE.exists():
        RESULTS_FILE.write_text(
            "timestamp\ttarget\titeration\tstatus\tmetric_name\tmetric_value\tbest_value\tdelta\tcheck_s\tinfo_or_bench_s\texecute_s\tnotes\n"
        )
    if not LOG_FILE.exists():
        LOG_FILE.write_text("")


def append_log(payload: dict[str, Any]) -> None:
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def best_metric_for_target(target: str, higher_is_better: bool) -> float | None:
    if not RESULTS_FILE.exists():
        return None

    best: float | None = None
    for line in RESULTS_FILE.read_text().splitlines()[1:]:
        cols = line.split("\t")
        if len(cols) < 6:
            continue
        if cols[1] != target or cols[3] != "success":
            continue
        try:
            value = float(cols[5])
        except ValueError:
            continue

        if best is None:
            best = value
        elif higher_is_better and value > best:
            best = value
        elif (not higher_is_better) and value < best:
            best = value

    return best


def append_result_row(
    *,
    target: str,
    iteration: int,
    status: str,
    metric_name: str,
    metric_value: float | None,
    higher_is_better: bool,
    check_s: float,
    info_or_bench_s: float,
    execute_s: float,
    notes: str,
) -> None:
    ensure_outputs()

    prev_best = best_metric_for_target(target, higher_is_better)
    value_s = ""
    best_s = ""
    delta_s = ""

    if metric_value is not None:
        value_s = f"{metric_value:.6f}"
        best_value = metric_value if prev_best is None else prev_best

        if status == "success":
            if prev_best is None:
                best_value = metric_value
            elif higher_is_better:
                best_value = max(prev_best, metric_value)
            else:
                best_value = min(prev_best, metric_value)

        best_s = f"{best_value:.6f}"
        if prev_best is not None:
            delta_s = f"{(metric_value - prev_best):.6f}"

    row = [
        now_iso(),
        target,
        str(iteration),
        status,
        metric_name,
        value_s,
        best_s,
        delta_s,
        f"{check_s:.4f}",
        f"{info_or_bench_s:.4f}",
        f"{execute_s:.4f}",
        notes.replace("\t", " "),
    ]

    with RESULTS_FILE.open("a", encoding="utf-8") as f:
        f.write("\t".join(row) + "\n")


def parse_nargo_info_acir(text: str, function_name: str = "main") -> int:
    lines = text.splitlines()
    header = None

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and "ACIR Opcodes" in stripped and "Function" in stripped:
            header = [c.strip() for c in stripped.strip("|").split("|")]
            break

    if header is None:
        raise ToolError("Could not parse nargo info table header")

    try:
        function_idx = header.index("Function")
        acir_idx = header.index("ACIR Opcodes")
    except ValueError as exc:
        raise ToolError("Could not locate Function / ACIR columns in nargo output") from exc

    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        if "ACIR Opcodes" in stripped or "====" in stripped or "----" in stripped:
            continue

        cols = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cols) <= max(function_idx, acir_idx):
            continue
        if cols[function_idx] != function_name:
            continue

        try:
            return int(cols[acir_idx])
        except ValueError as exc:
            raise ToolError(f"Invalid ACIR value in row: {cols}") from exc

    raise ToolError(f"Function '{function_name}' not found in nargo info output")


def ensure_tool_exists(tool: str) -> None:
    if shutil.which(tool) is None:
        raise ToolError(f"Required tool '{tool}' not found in PATH")


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


def extract_metric(haystack: str, metric_regex: str) -> float | None:
    cleaned = strip_ansi(haystack)
    matches = re.findall(metric_regex, cleaned)
    if not matches:
        return None
    match = matches[-1]
    if isinstance(match, tuple):
        if not match:
            return None
        token = match[0]
    else:
        token = match
    try:
        return float(token)
    except ValueError:
        return None


def aggregate_metric(values: list[float], mode: str) -> float:
    if not values:
        raise ToolError("No metric values to aggregate")
    if mode == "median":
        return float(statistics.median(values))
    if mode == "mean":
        return float(sum(values) / len(values))
    if mode == "min":
        return float(min(values))
    if mode == "max":
        return float(max(values))
    if mode == "last":
        return float(values[-1])
    raise ToolError(f"Unknown aggregate mode '{mode}'")


def resolve_cairo_artifact(project_dir: Path, target: dict[str, Any]) -> Path:
    explicit = target.get("artifact_file")
    if explicit:
        artifact = project_dir / explicit
        if artifact.exists():
            return artifact
        raise ToolError(f"Cairo artifact not found: {artifact}")

    pattern = target.get("artifact_glob", "target/dev/*.sierra.json")
    matches = sorted(project_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise ToolError(f"No Cairo artifacts found for pattern '{pattern}' in {project_dir}")
    return matches[0]


def parse_cairo_metric(project_dir: Path, target: dict[str, Any]) -> tuple[float, Path]:
    artifact = resolve_cairo_artifact(project_dir, target)
    mode = target.get("metric_mode", "sierra_program_len")

    if mode == "file_size_bytes":
        return float(artifact.stat().st_size), artifact

    if mode == "sierra_program_len":
        try:
            data = json.loads(artifact.read_text())
        except Exception as exc:  # noqa: BLE001
            raise ToolError(f"Failed to parse Cairo artifact JSON: {artifact}") from exc

        if isinstance(data, dict):
            program = data.get("sierra_program")
            if program is None:
                program = data.get("program")
            if isinstance(program, list):
                return float(len(program)), artifact
            statements = data.get("statements")
            if isinstance(statements, list):
                return float(len(statements)), artifact

        raise ToolError(f"Could not extract Sierra program length from {artifact}")

    raise ToolError(f"Unknown Cairo metric mode '{mode}'")


def bootstrap_target(target_name: str, target: dict[str, Any]) -> None:
    bootstrap = target.get("bootstrap")
    if not bootstrap:
        return

    git_url = bootstrap["git_url"]
    repo_rel = bootstrap["path"]
    ref = bootstrap.get("ref", "main")
    repo_dir = ROOT / repo_rel

    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    ensure_tool_exists("git")

    if repo_dir.exists() and (repo_dir / ".git").exists():
        run_cmd(["git", "fetch", "--all", "--tags", "--prune"], repo_dir)
        run_cmd(["git", "checkout", ref], repo_dir)
        run_cmd(["git", "pull", "--ff-only"], repo_dir)
        return

    if repo_dir.exists() and not (repo_dir / ".git").exists():
        raise ToolError(f"Bootstrap path exists but is not a git repo: {repo_dir}")

    clone = run_cmd(["git", "clone", "--depth", "1", "--branch", ref, git_url, str(repo_dir)], ROOT)
    if clone.code != 0:
        raise ToolError(f"Failed to clone {git_url}: {clone.stderr or clone.stdout}")


def evaluate_noir(target_name: str, target: dict[str, Any]) -> dict[str, Any]:
    ensure_tool_exists("nargo")
    project_dir = ROOT / target["project_dir"]
    if not project_dir.exists():
        raise ToolError(f"Noir project not found for target '{target_name}': {project_dir}")

    check = run_cmd(["nargo", "check"], project_dir)
    if check.code != 0:
        return {
            "status": "failed",
            "metric_name": target["metric_name"],
            "metric_value": None,
            "check_s": check.seconds,
            "info_or_bench_s": 0.0,
            "execute_s": 0.0,
            "notes": f"nargo check failed: {(check.stderr or check.stdout).strip()[:400]}",
            "debug": {"check": command_result_to_json(check)},
        }

    info = run_cmd(["nargo", "info"], project_dir)
    if info.code != 0:
        return {
            "status": "failed",
            "metric_name": target["metric_name"],
            "metric_value": None,
            "check_s": check.seconds,
            "info_or_bench_s": info.seconds,
            "execute_s": 0.0,
            "notes": f"nargo info failed: {(info.stderr or info.stdout).strip()[:400]}",
            "debug": {
                "check": command_result_to_json(check),
                "info": command_result_to_json(info),
            },
        }

    try:
        acir = parse_nargo_info_acir(info.stdout, function_name="main")
    except ToolError as exc:
        return {
            "status": "failed",
            "metric_name": target["metric_name"],
            "metric_value": None,
            "check_s": check.seconds,
            "info_or_bench_s": info.seconds,
            "execute_s": 0.0,
            "notes": str(exc),
            "debug": {
                "check": command_result_to_json(check),
                "info": command_result_to_json(info),
            },
        }

    execute = run_cmd(["nargo", "execute"], project_dir)
    if execute.code != 0:
        return {
            "status": "failed",
            "metric_name": target["metric_name"],
            "metric_value": float(acir),
            "check_s": check.seconds,
            "info_or_bench_s": info.seconds,
            "execute_s": execute.seconds,
            "notes": f"nargo execute failed: {(execute.stderr or execute.stdout).strip()[:400]}",
            "debug": {
                "check": command_result_to_json(check),
                "info": command_result_to_json(info),
                "execute": command_result_to_json(execute),
            },
        }

    return {
        "status": "success",
        "metric_name": target["metric_name"],
        "metric_value": float(acir),
        "check_s": check.seconds,
        "info_or_bench_s": info.seconds,
        "execute_s": execute.seconds,
        "notes": "ok",
        "debug": {
            "check": command_result_to_json(check),
            "info": command_result_to_json(info),
            "execute": command_result_to_json(execute),
        },
    }


def evaluate_cairo(target_name: str, target: dict[str, Any]) -> dict[str, Any]:
    ensure_tool_exists("scarb")
    project_dir = ROOT / target["project_dir"]
    if not project_dir.exists():
        raise ToolError(f"Cairo project not found for target '{target_name}': {project_dir}")

    build_cmd = target.get("build_command", ["scarb", "build"])
    test_cmd = target.get("test_command", ["scarb", "test"])

    build = run_cmd(build_cmd, project_dir)
    if build.code != 0:
        return {
            "status": "failed",
            "metric_name": target["metric_name"],
            "metric_value": None,
            "check_s": build.seconds,
            "info_or_bench_s": 0.0,
            "execute_s": 0.0,
            "notes": f"scarb build failed: {(build.stderr or build.stdout).strip()[:400]}",
            "debug": {"build": command_result_to_json(build)},
        }

    try:
        metric_value, artifact = parse_cairo_metric(project_dir, target)
    except ToolError as exc:
        return {
            "status": "failed",
            "metric_name": target["metric_name"],
            "metric_value": None,
            "check_s": build.seconds,
            "info_or_bench_s": 0.0,
            "execute_s": 0.0,
            "notes": str(exc),
            "debug": {"build": command_result_to_json(build)},
        }

    tests = run_cmd(test_cmd, project_dir)
    if tests.code != 0:
        return {
            "status": "failed",
            "metric_name": target["metric_name"],
            "metric_value": metric_value,
            "check_s": build.seconds,
            "info_or_bench_s": 0.0,
            "execute_s": tests.seconds,
            "notes": f"scarb test failed: {(tests.stderr or tests.stdout).strip()[:400]}",
            "debug": {
                "build": command_result_to_json(build),
                "test": command_result_to_json(tests),
            },
        }

    return {
        "status": "success",
        "metric_name": target["metric_name"],
        "metric_value": metric_value,
        "check_s": build.seconds,
        "info_or_bench_s": 0.0,
        "execute_s": tests.seconds,
        "notes": f"ok ({artifact.name})",
        "debug": {
            "build": command_result_to_json(build),
            "test": command_result_to_json(tests),
        },
    }


def evaluate_command(target_name: str, target: dict[str, Any]) -> dict[str, Any]:
    benchmark_command = target["benchmark_command"]
    metric_regex = target["metric_regex"]
    project_dir = ROOT / target["project_dir"]
    warmup_runs = int(target.get("warmup_runs", 0))
    runs = int(target.get("runs", 1))
    aggregate = str(target.get("aggregate", "median"))
    trim_extremes = int(target.get("trim_extremes", 0))

    if not project_dir.exists():
        return {
            "status": "failed",
            "metric_name": target["metric_name"],
            "metric_value": None,
            "check_s": 0.0,
            "info_or_bench_s": 0.0,
            "execute_s": 0.0,
            "notes": f"project dir not found: {project_dir}",
            "debug": {},
        }

    if runs < 1:
        return {
            "status": "failed",
            "metric_name": target["metric_name"],
            "metric_value": None,
            "check_s": 0.0,
            "info_or_bench_s": 0.0,
            "execute_s": 0.0,
            "notes": "runs must be >= 1",
            "debug": {},
        }

    total_seconds = 0.0
    metric_values_raw: list[float] = []
    bench_runs: list[dict[str, Any]] = []
    total_invocations = warmup_runs + runs

    for invocation in range(total_invocations):
        bench = run_cmd(benchmark_command, project_dir)
        total_seconds += bench.seconds
        bench_runs.append(command_result_to_json(bench))

        if bench.code != 0:
            return {
                "status": "failed",
                "metric_name": target["metric_name"],
                "metric_value": None,
                "check_s": 0.0,
                "info_or_bench_s": total_seconds,
                "execute_s": 0.0,
                "notes": f"benchmark command failed at run {invocation + 1}: {(bench.stderr or bench.stdout).strip()[:400]}",
                "debug": {"bench_runs": bench_runs},
            }

        value = extract_metric(bench.stdout + "\n" + bench.stderr, metric_regex)
        if value is None:
            return {
                "status": "failed",
                "metric_name": target["metric_name"],
                "metric_value": None,
                "check_s": 0.0,
                "info_or_bench_s": total_seconds,
                "execute_s": 0.0,
                "notes": f"metric regex did not match benchmark output at run {invocation + 1}",
                "debug": {"bench_runs": bench_runs},
            }

        if invocation >= warmup_runs:
            metric_values_raw.append(value)

    metric_values = metric_values_raw
    if trim_extremes > 0:
        if len(metric_values_raw) <= trim_extremes * 2:
            return {
                "status": "failed",
                "metric_name": target["metric_name"],
                "metric_value": None,
                "check_s": 0.0,
                "info_or_bench_s": total_seconds,
                "execute_s": 0.0,
                "notes": f"trim_extremes={trim_extremes} too large for runs={len(metric_values_raw)}",
                "debug": {"bench_runs": bench_runs, "metric_values_raw": metric_values_raw},
            }
        sorted_values = sorted(metric_values_raw)
        metric_values = sorted_values[trim_extremes : len(sorted_values) - trim_extremes]

    try:
        metric_value = aggregate_metric(metric_values, aggregate)
    except ToolError as exc:
        return {
            "status": "failed",
            "metric_name": target["metric_name"],
            "metric_value": None,
            "check_s": 0.0,
            "info_or_bench_s": total_seconds,
            "execute_s": 0.0,
                "notes": str(exc),
                "debug": {
                    "bench_runs": bench_runs,
                    "metric_values": metric_values,
                    "metric_values_raw": metric_values_raw,
                    "trim_extremes": trim_extremes,
                },
            }

    values_s = ",".join(f"{v:.2f}" for v in metric_values)
    return {
        "status": "success",
        "metric_name": target["metric_name"],
        "metric_value": metric_value,
        "check_s": 0.0,
        "info_or_bench_s": total_seconds,
        "execute_s": 0.0,
        "notes": (
            f"ok runs={runs} warmup={warmup_runs} aggregate={aggregate} "
            f"trim_extremes={trim_extremes} values=[{values_s}]"
        ),
        "debug": {
            "bench_runs": bench_runs,
            "metric_values": metric_values,
            "metric_values_raw": metric_values_raw,
            "aggregate": aggregate,
            "trim_extremes": trim_extremes,
        },
    }


def evaluate_target(target_name: str) -> dict[str, Any]:
    targets = load_targets()
    if target_name not in targets:
        raise ToolError(f"Unknown target '{target_name}'")

    target = targets[target_name]
    ttype = target["type"]
    if ttype == "noir":
        return evaluate_noir(target_name, target)
    if ttype == "cairo":
        return evaluate_cairo(target_name, target)
    if ttype == "command":
        return evaluate_command(target_name, target)
    raise ToolError(f"Unknown target type '{ttype}' for '{target_name}'")


def cmd_bootstrap(args: argparse.Namespace) -> int:
    targets = load_targets()
    selected = [args.target] if args.target else sorted(targets.keys())

    for target_name in selected:
        if target_name not in targets:
            print(f"Unknown target: {target_name}", file=sys.stderr)
            return 2

        try:
            bootstrap_target(target_name, targets[target_name])
            print(f"bootstrapped: {target_name}")
        except ToolError as exc:
            print(f"bootstrap failed for {target_name}: {exc}", file=sys.stderr)
            return 1

    return 0


def cmd_list_targets(_: argparse.Namespace) -> int:
    targets = load_targets()
    for name, cfg in sorted(targets.items()):
        print(f"{name}: {cfg.get('description', '')}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    targets = load_targets()
    if args.target not in targets:
        print(f"Unknown target: {args.target}", file=sys.stderr)
        return 2

    ensure_outputs()
    target = targets[args.target]

    try:
        result = evaluate_target(args.target)
    except ToolError as exc:
        print(f"evaluate failed: {exc}", file=sys.stderr)
        return 1

    append_result_row(
        target=args.target,
        iteration=args.iteration,
        status=result["status"],
        metric_name=result["metric_name"],
        metric_value=result["metric_value"],
        higher_is_better=bool(target["higher_is_better"]),
        check_s=result["check_s"],
        info_or_bench_s=result["info_or_bench_s"],
        execute_s=result["execute_s"],
        notes=args.notes if args.notes else result["notes"],
    )

    append_log(
        {
            "event": "evaluate",
            "timestamp": now_iso(),
            "target": args.target,
            "iteration": args.iteration,
            "result": {
                "status": result["status"],
                "metric_name": result["metric_name"],
                "metric_value": result["metric_value"],
                "notes": result["notes"],
            },
        }
    )

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "success" else 1


def cmd_baseline(args: argparse.Namespace) -> int:
    return cmd_evaluate(args)


def cmd_calibrate(args: argparse.Namespace) -> int:
    targets = load_targets()
    if args.target not in targets:
        print(f"Unknown target: {args.target}", file=sys.stderr)
        return 2

    target = targets[args.target]
    ensure_outputs()

    values: list[float] = []
    runs: list[dict[str, Any]] = []
    failures = 0

    for sample_index in range(1, args.samples + 1):
        try:
            result = evaluate_target(args.target)
        except ToolError as exc:
            failures += 1
            runs.append(
                {
                    "sample": sample_index,
                    "status": "failed",
                    "error": str(exc),
                }
            )
            continue

        metric_value = result.get("metric_value")
        status = result.get("status", "failed")
        run_payload = {
            "sample": sample_index,
            "status": status,
            "metric_name": result.get("metric_name"),
            "metric_value": metric_value,
            "notes": result.get("notes"),
            "debug": result.get("debug", {}),
        }
        runs.append(run_payload)

        if args.record:
            append_result_row(
                target=args.target,
                iteration=sample_index,
                status=status,
                metric_name=result["metric_name"],
                metric_value=float(metric_value) if metric_value is not None else None,
                higher_is_better=bool(target["higher_is_better"]),
                check_s=float(result.get("check_s", 0.0)),
                info_or_bench_s=float(result.get("info_or_bench_s", 0.0)),
                execute_s=float(result.get("execute_s", 0.0)),
                notes=f"{args.notes}:sample_{sample_index}",
            )

        if status == "success" and metric_value is not None:
            values.append(float(metric_value))
        else:
            failures += 1

    if not values:
        payload = {
            "ok": False,
            "target": args.target,
            "samples": args.samples,
            "failures": failures,
            "message": "No successful samples",
            "runs": runs,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    mean = float(statistics.fmean(values))
    median = float(statistics.median(values))
    stdev = float(statistics.pstdev(values)) if len(values) >= 2 else 0.0
    rel_stdev = float("inf") if mean == 0.0 else abs(stdev / mean)
    min_value = float(min(values))
    max_value = float(max(values))

    configured_rel = float(target.get("min_improvement_rel", 0.0))
    configured_noise = target.get("max_rel_stdev")
    recommended_rel = max(configured_rel, rel_stdev * args.sigma_multiplier)

    payload = {
        "ok": True,
        "target": args.target,
        "samples_requested": args.samples,
        "samples_success": len(values),
        "samples_failed": failures,
        "metric_name": target["metric_name"],
        "higher_is_better": bool(target["higher_is_better"]),
        "stats": {
            "min": min_value,
            "max": max_value,
            "mean": mean,
            "median": median,
            "stdev": stdev,
            "rel_stdev": rel_stdev,
        },
        "configured": {
            "min_improvement_rel": configured_rel,
            "max_rel_stdev": configured_noise,
        },
        "recommendation": {
            "sigma_multiplier": args.sigma_multiplier,
            "min_improvement_rel": recommended_rel,
            "max_rel_stdev_floor": rel_stdev,
        },
        "runs": runs,
    }

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AutoPoseidon prep and evaluation harness")
    sub = parser.add_subparsers(dest="cmd", required=True)

    list_targets = sub.add_parser("list-targets", help="List configured targets")
    list_targets.set_defaults(func=cmd_list_targets)

    bootstrap = sub.add_parser("bootstrap", help="Bootstrap external repos for target(s)")
    bootstrap.add_argument("--target", help="Target name (default: all)")
    bootstrap.set_defaults(func=cmd_bootstrap)

    evaluate = sub.add_parser("evaluate", help="Evaluate one target and append logs")
    evaluate.add_argument("--target", required=True, help="Target name")
    evaluate.add_argument("--iteration", type=int, default=0, help="Iteration index")
    evaluate.add_argument("--notes", default="", help="Optional note for results.tsv")
    evaluate.set_defaults(func=cmd_evaluate)

    baseline = sub.add_parser("baseline", help="Alias for evaluate")
    baseline.add_argument("--target", required=True, help="Target name")
    baseline.add_argument("--iteration", type=int, default=0, help="Iteration index")
    baseline.add_argument("--notes", default="baseline", help="Optional note for results.tsv")
    baseline.set_defaults(func=cmd_baseline)

    calibrate = sub.add_parser("calibrate", help="Measure metric stability over repeated evaluations")
    calibrate.add_argument("--target", required=True, help="Target name")
    calibrate.add_argument("--samples", type=int, default=7, help="Number of repeated evaluations")
    calibrate.add_argument(
        "--sigma-multiplier",
        type=float,
        default=2.0,
        help="Multiplier applied to relative stdev for suggested min_improvement_rel",
    )
    calibrate.add_argument("--record", action="store_true", help="Append each sample to results.tsv")
    calibrate.add_argument("--notes", default="calibration", help="Note prefix used when --record is enabled")
    calibrate.set_defaults(func=cmd_calibrate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
