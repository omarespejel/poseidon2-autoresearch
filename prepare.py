#!/usr/bin/env python3
"""AutoPoseidon preparation and evaluation harness."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from target_config import load_targets_file

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
GIT_OID_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def is_verbose(level: int = 1) -> bool:
    return env_int("AUTORESEARCH_VERBOSE", 0) >= level


def debug_max_chars() -> int:
    return max(256, env_int("AUTORESEARCH_DEBUG_MAX_CHARS", 4000))


def debug_command_output() -> bool:
    return os.getenv("AUTORESEARCH_DEBUG_COMMAND_OUTPUT", "").strip().lower() in {"1", "true", "yes", "on"}


def debug_log(message: str, *, level: int = 1) -> None:
    if not is_verbose(level):
        return
    print(f"[prepare {now_iso()}] {message}", file=sys.stderr, flush=True)


def trim_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    trimmed = text[:limit]
    return f"{trimmed}\n... [truncated {len(text) - limit} chars]"


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

    pretty = " ".join(shlex.quote(arg) for arg in final_argv)
    debug_log(f"run: (cd {cwd} && {pretty})", level=1)
    start = time.perf_counter()
    try:
        proc = subprocess.run(final_argv, cwd=str(cwd), text=True, capture_output=True, check=False)
    except OSError as exc:
        return CommandResult(
            argv=final_argv,
            cwd=cwd,
            code=127,
            stdout="",
            stderr=str(exc),
            seconds=time.perf_counter() - start,
        )
    result = CommandResult(
        argv=final_argv,
        cwd=cwd,
        code=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        seconds=time.perf_counter() - start,
    )
    debug_log(f"done: exit={result.code} seconds={result.seconds:.3f}", level=1)
    show_output = debug_command_output() or (result.code != 0 and is_verbose(1))
    if show_output:
        cap = debug_max_chars()
        if result.stdout.strip():
            debug_log("stdout:\n" + trim_text(result.stdout, cap), level=1)
        if result.stderr.strip():
            debug_log("stderr:\n" + trim_text(result.stderr, cap), level=1)
    return result


def load_targets() -> dict[str, dict[str, Any]]:
    return load_targets_file(TARGETS_FILE)


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


def sanitize_notes(notes: str) -> str:
    return notes.replace("\t", " ").replace("\r", " ").replace("\n", " ")


def best_metric_for_target(target: str, *, higher_is_better: bool) -> float | None:
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

        if best is None or (higher_is_better and value > best) or ((not higher_is_better) and value < best):
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

    prev_best = best_metric_for_target(target, higher_is_better=higher_is_better)
    value_s = ""
    best_s = ""
    delta_s = ""
    best_value = prev_best

    if metric_value is not None:
        value_s = f"{metric_value:.6f}"

        if status == "success":
            if prev_best is None:
                best_value = metric_value
            elif higher_is_better:
                best_value = max(prev_best, metric_value)
            else:
                best_value = min(prev_best, metric_value)

        if best_value is not None:
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
        sanitize_notes(notes),
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


def is_git_oid(ref: str) -> bool:
    return bool(GIT_OID_RE.fullmatch(ref.strip()))


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


def aggregate_weighted_metric(values: list[float], weights: list[float], mode: str) -> float:
    if not values:
        raise ToolError("No profile metrics to aggregate")
    if len(values) != len(weights):
        raise ToolError("values/weights length mismatch")

    clean: list[tuple[float, float]] = []
    for value, weight in zip(values, weights):
        if not math.isfinite(value):
            raise ToolError(f"Non-finite profile metric: {value}")
        if not math.isfinite(weight) or weight <= 0.0:
            raise ToolError(f"Invalid profile weight: {weight}")
        clean.append((float(value), float(weight)))

    total_weight = sum(weight for _, weight in clean)
    if total_weight <= 0.0:
        raise ToolError("Total profile weight must be positive")

    if mode == "weighted_mean":
        return sum(value * weight for value, weight in clean) / total_weight

    if mode == "weighted_geomean":
        log_sum = 0.0
        for value, weight in clean:
            if value <= 0.0:
                raise ToolError("weighted_geomean requires all profile metrics to be > 0")
            log_sum += weight * math.log(value)
        return math.exp(log_sum / total_weight)

    raise ToolError(f"Unknown profile aggregate mode '{mode}'")


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

    if repo_dir.exists() and not (repo_dir / ".git").exists():
        raise ToolError(f"Bootstrap path exists but is not a git repo: {repo_dir}")

    if not repo_dir.exists():
        clone_argv = ["git", "clone", "--depth", "1", git_url, str(repo_dir)]
        if not is_git_oid(ref):
            clone_argv = ["git", "clone", "--depth", "1", "--branch", ref, git_url, str(repo_dir)]
        clone = run_cmd(clone_argv, ROOT)
        if clone.code != 0:
            raise ToolError(f"Failed to clone {git_url}: {clone.stderr or clone.stdout}")

    if is_git_oid(ref):
        fetch = run_cmd(["git", "fetch", "--depth", "1", "origin", ref], repo_dir)
        if fetch.code != 0:
            raise ToolError(f"Failed to fetch pinned ref {ref}: {fetch.stderr or fetch.stdout}")
        checkout = run_cmd(["git", "checkout", "--detach", "FETCH_HEAD"], repo_dir)
        if checkout.code != 0:
            raise ToolError(f"Failed to checkout pinned ref {ref}: {checkout.stderr or checkout.stdout}")
        return

    fetch = run_cmd(["git", "fetch", "--all", "--tags", "--prune"], repo_dir)
    if fetch.code != 0:
        raise ToolError(f"Failed to fetch {repo_dir}: {fetch.stderr or fetch.stdout}")
    checkout = run_cmd(["git", "checkout", ref], repo_dir)
    if checkout.code != 0:
        raise ToolError(f"Failed to checkout {ref}: {checkout.stderr or checkout.stdout}")
    pull = run_cmd(["git", "pull", "--ff-only"], repo_dir)
    if pull.code != 0:
        raise ToolError(f"Failed to pull {ref}: {pull.stderr or pull.stdout}")


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


def evaluate_command_profile(
    *,
    metric_name: str,
    project_dir: Path,
    profile_name: str,
    benchmark_command: list[str],
    metric_regex: str,
    warmup_runs: int,
    runs: int,
    aggregate: str,
    trim_extremes: int,
) -> dict[str, Any]:
    if runs < 1:
        return {
            "status": "failed",
            "metric_name": metric_name,
            "metric_value": None,
            "check_s": 0.0,
            "info_or_bench_s": 0.0,
            "execute_s": 0.0,
            "notes": f"profile={profile_name} runs must be >= 1",
            "debug": {"profile": profile_name},
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
                "metric_name": metric_name,
                "metric_value": None,
                "check_s": 0.0,
                "info_or_bench_s": total_seconds,
                "execute_s": 0.0,
                "notes": (
                    f"profile={profile_name} benchmark command failed at run {invocation + 1}: "
                    f"{(bench.stderr or bench.stdout).strip()[:400]}"
                ),
                "debug": {
                    "profile": profile_name,
                    "bench_runs": bench_runs,
                    "benchmark_command": benchmark_command,
                },
            }

        value = extract_metric(bench.stdout + "\n" + bench.stderr, metric_regex)
        if value is None:
            return {
                "status": "failed",
                "metric_name": metric_name,
                "metric_value": None,
                "check_s": 0.0,
                "info_or_bench_s": total_seconds,
                "execute_s": 0.0,
                "notes": f"profile={profile_name} metric regex did not match output at run {invocation + 1}",
                "debug": {
                    "profile": profile_name,
                    "bench_runs": bench_runs,
                    "metric_regex": metric_regex,
                },
            }

        if invocation >= warmup_runs:
            metric_values_raw.append(value)

    metric_values = metric_values_raw
    if trim_extremes > 0:
        if len(metric_values_raw) <= trim_extremes * 2:
            return {
                "status": "failed",
                "metric_name": metric_name,
                "metric_value": None,
                "check_s": 0.0,
                "info_or_bench_s": total_seconds,
                "execute_s": 0.0,
                "notes": (
                    f"profile={profile_name} trim_extremes={trim_extremes} "
                    f"too large for runs={len(metric_values_raw)}"
                ),
                "debug": {
                    "profile": profile_name,
                    "bench_runs": bench_runs,
                    "metric_values_raw": metric_values_raw,
                },
            }
        sorted_values = sorted(metric_values_raw)
        metric_values = sorted_values[trim_extremes : len(sorted_values) - trim_extremes]

    try:
        metric_value = aggregate_metric(metric_values, aggregate)
    except ToolError as exc:
        return {
            "status": "failed",
            "metric_name": metric_name,
            "metric_value": None,
            "check_s": 0.0,
            "info_or_bench_s": total_seconds,
            "execute_s": 0.0,
            "notes": f"profile={profile_name} {exc}",
            "debug": {
                "profile": profile_name,
                "bench_runs": bench_runs,
                "metric_values": metric_values,
                "metric_values_raw": metric_values_raw,
                "trim_extremes": trim_extremes,
                "aggregate": aggregate,
            },
        }

    values_s = ",".join(f"{v:.2f}" for v in metric_values)
    return {
        "status": "success",
        "metric_name": metric_name,
        "metric_value": metric_value,
        "check_s": 0.0,
        "info_or_bench_s": total_seconds,
        "execute_s": 0.0,
        "notes": (
            f"profile={profile_name} ok runs={runs} warmup={warmup_runs} aggregate={aggregate} "
            f"trim_extremes={trim_extremes} values=[{values_s}]"
        ),
        "debug": {
            "profile": profile_name,
            "benchmark_command": benchmark_command,
            "metric_regex": metric_regex,
            "bench_runs": bench_runs,
            "metric_values": metric_values,
            "metric_values_raw": metric_values_raw,
            "aggregate": aggregate,
            "trim_extremes": trim_extremes,
            "runs": runs,
            "warmup_runs": warmup_runs,
        },
    }


def evaluate_command(target_name: str, target: dict[str, Any]) -> dict[str, Any]:
    project_dir = ROOT / target["project_dir"]
    metric_name = str(target["metric_name"])

    if not project_dir.exists():
        return {
            "status": "failed",
            "metric_name": metric_name,
            "metric_value": None,
            "check_s": 0.0,
            "info_or_bench_s": 0.0,
            "execute_s": 0.0,
            "notes": f"project dir not found: {project_dir}",
            "debug": {},
        }

    default_command = target.get("benchmark_command")
    default_regex = str(target.get("metric_regex", ""))
    default_warmup = int(target.get("warmup_runs", 0))
    default_runs = int(target.get("runs", 1))
    default_aggregate = str(target.get("aggregate", "median"))
    default_trim = int(target.get("trim_extremes", 0))

    profiles_raw = target.get("benchmark_profiles")
    if not isinstance(profiles_raw, list) or not profiles_raw:
        if not isinstance(default_command, list) or not default_command:
            return {
                "status": "failed",
                "metric_name": metric_name,
                "metric_value": None,
                "check_s": 0.0,
                "info_or_bench_s": 0.0,
                "execute_s": 0.0,
                "notes": "benchmark_command must be a non-empty list",
                "debug": {},
            }
        return evaluate_command_profile(
            metric_name=metric_name,
            project_dir=project_dir,
            profile_name="default",
            benchmark_command=default_command,
            metric_regex=default_regex,
            warmup_runs=default_warmup,
            runs=default_runs,
            aggregate=default_aggregate,
            trim_extremes=default_trim,
        )

    profile_values: list[float] = []
    profile_weights: list[float] = []
    profile_reports: list[dict[str, Any]] = []
    total_seconds = 0.0

    for idx, profile_raw in enumerate(profiles_raw, start=1):
        if not isinstance(profile_raw, dict):
            return {
                "status": "failed",
                "metric_name": metric_name,
                "metric_value": None,
                "check_s": 0.0,
                "info_or_bench_s": total_seconds,
                "execute_s": 0.0,
                "notes": f"benchmark_profiles[{idx}] must be an object",
                "debug": {"profile_reports": profile_reports},
            }

        profile_name = str(profile_raw.get("name", f"profile_{idx}")).strip() or f"profile_{idx}"
        command = profile_raw.get("benchmark_command", default_command)
        metric_regex = str(profile_raw.get("metric_regex", default_regex))
        warmup_runs = int(profile_raw.get("warmup_runs", default_warmup))
        runs = int(profile_raw.get("runs", default_runs))
        aggregate = str(profile_raw.get("aggregate", default_aggregate))
        trim_extremes = int(profile_raw.get("trim_extremes", default_trim))
        weight = float(profile_raw.get("weight", 1.0))

        if not isinstance(command, list) or not command:
            return {
                "status": "failed",
                "metric_name": metric_name,
                "metric_value": None,
                "check_s": 0.0,
                "info_or_bench_s": total_seconds,
                "execute_s": 0.0,
                "notes": f"profile={profile_name} benchmark_command must be a non-empty list",
                "debug": {"profile_reports": profile_reports},
            }
        if not metric_regex:
            return {
                "status": "failed",
                "metric_name": metric_name,
                "metric_value": None,
                "check_s": 0.0,
                "info_or_bench_s": total_seconds,
                "execute_s": 0.0,
                "notes": f"profile={profile_name} metric_regex must be non-empty",
                "debug": {"profile_reports": profile_reports},
            }

        report = evaluate_command_profile(
            metric_name=metric_name,
            project_dir=project_dir,
            profile_name=profile_name,
            benchmark_command=command,
            metric_regex=metric_regex,
            warmup_runs=warmup_runs,
            runs=runs,
            aggregate=aggregate,
            trim_extremes=trim_extremes,
        )
        total_seconds += float(report.get("info_or_bench_s", 0.0))

        metric_value = report.get("metric_value")
        if report.get("status") != "success" or metric_value is None:
            debug_payload = report.get("debug", {})
            profile_reports.append(
                {
                    "name": profile_name,
                    "weight": weight,
                    "status": report.get("status"),
                    "metric_value": metric_value,
                    "notes": report.get("notes"),
                    "debug": debug_payload if isinstance(debug_payload, dict) else {},
                }
            )
            return {
                "status": "failed",
                "metric_name": metric_name,
                "metric_value": None,
                "check_s": 0.0,
                "info_or_bench_s": total_seconds,
                "execute_s": 0.0,
                "notes": str(report.get("notes", "profile evaluation failed")),
                "debug": {"profile_reports": profile_reports},
            }

        if not math.isfinite(weight) or weight <= 0.0:
            return {
                "status": "failed",
                "metric_name": metric_name,
                "metric_value": None,
                "check_s": 0.0,
                "info_or_bench_s": total_seconds,
                "execute_s": 0.0,
                "notes": f"profile={profile_name} weight must be > 0",
                "debug": {"profile_reports": profile_reports},
            }

        debug_payload = report.get("debug", {})
        profile_reports.append(
            {
                "name": profile_name,
                "weight": weight,
                "status": "success",
                "metric_value": float(metric_value),
                "notes": report.get("notes"),
                "debug": debug_payload if isinstance(debug_payload, dict) else {},
            }
        )
        profile_values.append(float(metric_value))
        profile_weights.append(weight)

    profiles_aggregate = str(target.get("profiles_aggregate", "weighted_geomean")).strip().lower()
    try:
        combined_metric = aggregate_weighted_metric(profile_values, profile_weights, profiles_aggregate)
    except ToolError as exc:
        return {
            "status": "failed",
            "metric_name": metric_name,
            "metric_value": None,
            "check_s": 0.0,
            "info_or_bench_s": total_seconds,
            "execute_s": 0.0,
            "notes": str(exc),
            "debug": {"profile_reports": profile_reports, "profiles_aggregate": profiles_aggregate},
        }

    profile_series: list[list[float]] = []
    for report in profile_reports:
        debug_payload = report.get("debug", {})
        values = debug_payload.get("metric_values") if isinstance(debug_payload, dict) else None
        if not isinstance(values, list):
            profile_series = []
            break
        parsed: list[float] = []
        for item in values:
            if not isinstance(item, (int, float)):
                parsed = []
                break
            parsed.append(float(item))
        if not parsed:
            profile_series = []
            break
        profile_series.append(parsed)

    composite_series: list[float]
    if profile_series and len({len(series) for series in profile_series}) == 1:
        composite_series = []
        for idx in range(len(profile_series[0])):
            step_values = [series[idx] for series in profile_series]
            try:
                composite_series.append(
                    aggregate_weighted_metric(step_values, profile_weights, profiles_aggregate)
                )
            except ToolError:
                composite_series = []
                break
    else:
        composite_series = []

    if not composite_series:
        composite_series = [float(v) for v in profile_values]

    profile_values_note = ",".join(
        f"{report['name']}={float(report['metric_value']):.2f}" for report in profile_reports
    )
    return {
        "status": "success",
        "metric_name": metric_name,
        "metric_value": combined_metric,
        "check_s": 0.0,
        "info_or_bench_s": total_seconds,
        "execute_s": 0.0,
        "notes": (
            f"ok profiles={len(profile_reports)} aggregate={profiles_aggregate} "
            f"values=[{profile_values_note}]"
        ),
        "debug": {
            "profile_reports": profile_reports,
            "profiles_aggregate": profiles_aggregate,
            "metric_values": composite_series,
            "metric_values_raw": profile_values,
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
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase stderr verbosity")
    parser.add_argument(
        "--debug-command-output",
        action="store_true",
        help="Print captured command stdout/stderr to stderr (truncated)",
    )
    parser.add_argument(
        "--debug-max-chars",
        type=int,
        default=4000,
        help="Max chars shown per command stream when debug output is enabled",
    )
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
    if args.verbose > 0:
        os.environ["AUTORESEARCH_VERBOSE"] = str(args.verbose)
    if args.debug_command_output:
        os.environ["AUTORESEARCH_DEBUG_COMMAND_OUTPUT"] = "1"
    os.environ["AUTORESEARCH_DEBUG_MAX_CHARS"] = str(max(256, int(args.debug_max_chars)))
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
