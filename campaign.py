#!/usr/bin/env python3
"""Reproducible campaign runner for AutoPoseidon.

Runs a dual-track execution:
- Track A: autonomous optimization loop and Synthesis-ready artifacts
- Real stack: Lean/Poseidon2 baseline + optional source-level optimization
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import readiness_check

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
MUTATION_MEMORY = ROOT / "work" / "mutation_memory.json"

DEFAULT_INFORMATIONAL_READINESS_CHECKS: frozenset[str] = readiness_check.INFORMATIONAL_CHECKS


def resolve_poseidon_arch() -> str:
    override = os.getenv("POSEIDON_ARCH", "").strip().lower()
    if override in {"arm64", "aarch64", "arm", "neon"}:
        return "arm64"
    if override in {"x86_64", "amd64", "x86", "avx2", "avx512"}:
        return "x86_64"

    machine = platform.machine().lower()
    processor = platform.processor().lower()
    if any(token in machine for token in ("arm64", "aarch64", "arm")) or any(
        token in processor for token in ("arm", "apple")
    ):
        return "arm64"
    if machine in {"x86_64", "amd64", "x86", "i386", "i686"} or any(
        token in processor for token in ("intel", "x86")
    ):
        return "x86_64"
    return "unknown"


def default_real_optimize_targets() -> str:
    """Select SOTA Poseidon source targets that match the current host backend."""
    targets = [
        "leanmultisig_poseidon16_src_fast",
        "leanmultisig_poseidon16_table_src_fast",
        "leanmultisig_poseidon2_monty_core_src_fast",
    ]
    arch = resolve_poseidon_arch()
    if arch == "arm64":
        targets.append("leanmultisig_poseidon2_neon_src_fast")
    elif arch == "x86_64":
        targets.append("leanmultisig_poseidon2_avx2_src_fast")
    else:
        # Unknown host: hedge by including wrapper fallback and x86 backend source targets.
        targets.append("leanmultisig_poseidon2_no_packing_src_fast")
        targets.append("leanmultisig_poseidon2_avx2_src_fast")
    return ",".join(targets)


def default_crypto_optimize_targets() -> str:
    return ",".join(
        [
            "poseidon2_cryptanalysis_trackb_kernel_fast",
            "poseidon2_cryptanalysis_trackb_kernel_signal_fast",
            "poseidon2_cryptanalysis_trackb_fast",
            "poseidon2_cryptanalysis_trackb_verified_fast",
            "poseidon2_cryptanalysis_algebraic_fast",
            "poseidon2_cryptanalysis_poseidon64_signal_fast",
            "poseidon2_cryptanalysis_poseidon64_algebraic_fast",
        ]
    )


def parse_targets_csv(raw: str) -> list[str]:
    out: list[str] = []
    for token in str(raw).split(","):
        clean = token.strip()
        if clean:
            out.append(clean)
    return out


@dataclass
class RunResult:
    argv: list[str]
    code: int
    stdout: str
    stderr: str


def campaign_log(args: argparse.Namespace, message: str) -> None:
    if int(getattr(args, "verbose", 0)) <= 0:
        return
    stamp = dt.datetime.now(dt.timezone.utc).isoformat()
    print(f"[campaign {stamp}] {message}", file=sys.stderr, flush=True)


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


def fail_json_contract(*, label: str, reason: str, stdout: str) -> None:
    tail = stdout[-2000:] if stdout else ""
    sys.stderr.write(f"Invalid JSON payload [{label}]: {reason}\n")
    if tail:
        sys.stderr.write(tail + "\n")
    raise SystemExit(2)


def must_run_json(
    argv: list[str],
    *,
    label: str,
    stream_stderr: bool = False,
    require_ok: bool = True,
) -> dict[str, Any]:
    result = must_run(argv, label=label, stream_stderr=stream_stderr)
    raw = result.stdout.strip()
    if not raw:
        fail_json_contract(label=label, reason="empty stdout", stdout="")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        fail_json_contract(label=label, reason="malformed JSON", stdout=raw)
    if not isinstance(payload, dict):
        fail_json_contract(label=label, reason="JSON root must be an object", stdout=raw)
    if require_ok and "ok" not in payload:
        fail_json_contract(label=label, reason="missing required 'ok' field", stdout=raw)
    if require_ok and not bool(payload.get("ok")):
        fail_json_contract(label=label, reason="step reported ok=false", stdout=raw)
    return payload


def reset_outputs(mutation_memory_path: Path = MUTATION_MEMORY) -> None:
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
    for path in (
        mutation_memory_path,
        mutation_memory_path.with_suffix(mutation_memory_path.suffix + ".lock"),
    ):
        if path.exists():
            path.unlink()


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


def extract_mutation_from_note_token(token: str) -> str:
    text = str(token or "").strip()
    if not text:
        return "unknown"
    prefixes = ["python_", "rust_", "json_", "openai_patch", "fallback_", "heuristic_"]
    hits = [text.find(prefix) for prefix in prefixes if text.find(prefix) >= 0]
    if hits:
        start = min(hits)
        text = text[start:]
    elif ":" in text:
        text = text.split(":")[-1].strip()
    text = re.sub(r":no_change$", "", text).strip()
    return text or "unknown"


def summarize_validation_blocks(rows: list[dict[str, str]]) -> dict[str, Any]:
    total = 0
    by_mutation: dict[str, dict[str, Any]] = {}
    for row in rows:
        notes = str(row.get("notes", "")).strip()
        if not notes:
            continue
        token = notes.split(";", 1)[0].strip()
        if not token.startswith("rejected_validation_"):
            continue
        total += 1
        mutation = extract_mutation_from_note_token(token)
        entry = by_mutation.setdefault(
            mutation,
            {
                "mutation": mutation,
                "count": 0,
                "targets": set(),
            },
        )
        entry["count"] = int(entry.get("count", 0)) + 1
        target = str(row.get("target", "")).strip()
        if target:
            entry_targets = entry.get("targets")
            if isinstance(entry_targets, set):
                entry_targets.add(target)

    rows_out: list[dict[str, Any]] = []
    for item in by_mutation.values():
        targets = sorted(str(v) for v in (item.get("targets") or set()))
        rows_out.append(
            {
                "mutation": str(item.get("mutation", "unknown")),
                "count": int(item.get("count", 0)),
                "targets": targets,
            }
        )
    rows_out.sort(key=lambda item: (-int(item["count"]), str(item["mutation"])))
    return {"total": total, "rows": rows_out}


def add_debug_flags(argv: list[str], args: argparse.Namespace) -> list[str]:
    out = list(argv)
    if args.verbose > 0:
        out.extend(["-v"] * args.verbose)
    if args.debug_command_output:
        out.append("--debug-command-output")
    out.extend(["--debug-max-chars", str(max(256, int(args.debug_max_chars)))])
    return out


def add_mutation_memory_flags(argv: list[str], args: argparse.Namespace) -> list[str]:
    out = list(argv)
    if args.mutation_memory_file:
        out.extend(["--mutation-memory-file", args.mutation_memory_file])
    if args.disable_mutation_memory:
        out.append("--disable-mutation-memory")
    return out


def add_git_checkpoint_flags(argv: list[str], args: argparse.Namespace) -> list[str]:
    out = list(argv)
    mode = str(getattr(args, "git_checkpoint_mode", "off")).strip().lower()
    if mode in {"accepted", "all"}:
        out.extend(["--git-checkpoint-mode", mode])
        prefix = str(getattr(args, "git_checkpoint_prefix", "")).strip()
        if prefix:
            out.extend(["--git-checkpoint-prefix", prefix])
    return out


def prepare_cmd(args: argparse.Namespace, *tail: str) -> list[str]:
    base = [sys.executable, str(PREPARE)]
    return add_debug_flags(base, args) + list(tail)


def train_cmd(args: argparse.Namespace, *tail: str) -> list[str]:
    base = [sys.executable, str(LOOP)]
    return add_git_checkpoint_flags(add_mutation_memory_flags(add_debug_flags(base, args), args), args) + list(tail)


def portfolio_cmd(args: argparse.Namespace, *tail: str) -> list[str]:
    base = [sys.executable, str(PORTFOLIO)]
    return add_mutation_memory_flags(add_debug_flags(base, args), args) + list(tail)


def write_report(
    *,
    track: str,
    rows: list[dict[str, str]],
    loop_target: str,
    real_targets: list[str],
    crypto_targets: list[str],
    loop_iterations: int,
    loop_accepts: int,
    loop_payload: dict[str, Any],
    portfolio_payload: dict[str, Any] | None,
    crypto_payload: dict[str, Any] | None,
    evidence_payload: dict[str, Any] | None,
    submission_payload: dict[str, Any] | None,
    readiness_payload: dict[str, Any] | None,
    synthesis_cook: bool,
) -> None:
    latest = summarize(rows)
    validation_blocks = summarize_validation_blocks(rows)

    lines: list[str] = []
    lines.append("# AutoPoseidon Campaign Report")
    lines.append("")
    lines.append("## Run Summary")
    lines.append("")
    lines.append(f"- Campaign track: `{track}`")
    lines.append(f"- Synthesis cook mode: `{str(synthesis_cook).lower()}`")
    lines.append(f"- Loop target: `{loop_target}`")
    lines.append(f"- Loop iterations requested: `{loop_iterations if loop_iterations > 0 else 'infinite'}`")
    lines.append(f"- Loop accepted improvements: `{loop_accepts}`")
    lines.append(f"- Real baseline targets: `{', '.join(real_targets) if real_targets else 'none'}`")
    lines.append(f"- Cryptanalysis targets: `{', '.join(crypto_targets) if crypto_targets else 'none'}`")
    if portfolio_payload:
        lines.append(f"- Real optimization rounds executed: `{portfolio_payload.get('rounds_executed', 'n/a')}`")
        lines.append(f"- Real optimization total accepted: `{portfolio_payload.get('total_accepted', 'n/a')}`")
    if crypto_payload:
        lines.append(f"- Cryptanalysis optimization rounds executed: `{crypto_payload.get('rounds_executed', 'n/a')}`")
        lines.append(f"- Cryptanalysis optimization total accepted: `{crypto_payload.get('total_accepted', 'n/a')}`")
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
    lines.append("## Validation Blocks")
    lines.append("")
    lines.append(
        f"- Total rejected after primary acceptance checks due to validation gate: `{validation_blocks['total']}`"
    )
    validation_rows = list(validation_blocks.get("rows", []))
    if validation_rows:
        lines.append("")
        lines.append("| mutation | blocked_count | targets |")
        lines.append("|---|---:|---|")
        for item in validation_rows:
            targets = ", ".join(item.get("targets", [])) or "n/a"
            lines.append(
                "| {mutation} | {count} | {targets} |".format(
                    mutation=item.get("mutation", "unknown"),
                    count=item.get("count", 0),
                    targets=targets,
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
    if crypto_payload is not None:
        lines.append(
            "| cryptanalysis_optimize | {ok} | accepted={acc} |".format(
                ok="yes" if crypto_payload.get("ok", True) else "no",
                acc=crypto_payload.get("total_accepted", "n/a"),
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
        overall_ready = readiness_payload.get("overall_ready")
        if isinstance(overall_ready, bool):
            readiness_ok = "yes" if overall_ready else "no"
        else:
            checks = readiness_payload.get("checks")
            informational_raw = readiness_payload.get("informational_checks")
            informational_checks = set(DEFAULT_INFORMATIONAL_READINESS_CHECKS)
            if isinstance(informational_raw, list):
                informational_checks = {
                    str(item).strip()
                    for item in informational_raw
                    if isinstance(item, str) and str(item).strip()
                } or informational_checks
            if isinstance(checks, list):
                required_failed = 0
                for check in checks:
                    if not isinstance(check, dict):
                        continue
                    name = str(check.get("name", ""))
                    ok = bool(check.get("ok", False))
                    if name not in informational_checks and not ok:
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
    if track == "performance":
        lines.append("python3 campaign.py --fresh --track performance --synthesis-cook --real-profile fast")
    elif track == "cryptanalysis":
        lines.append("python3 campaign.py --fresh --track cryptanalysis --loop-iterations 25 --crypto-optimize-rounds 2")
    else:
        lines.append(
            "python3 campaign.py --fresh --track hybrid --synthesis-cook --real-profile fast --crypto-optimize-rounds 2"
        )
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
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Reset results/logs/reports and work/mutation_memory.json before run",
    )
    parser.add_argument(
        "--synthesis-cook",
        action="store_true",
        help="Enable a Synthesis-aligned pipeline (real optimize + evidence + submission + readiness)",
    )
    parser.add_argument(
        "--track",
        choices=["performance", "cryptanalysis", "hybrid"],
        default="performance",
        help="Campaign track: implementation performance, cryptanalysis, or both",
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
        default=default_real_optimize_targets(),
        help="Comma-separated source targets for real optimization (host-aware default)",
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
        "--crypto-optimize-rounds",
        type=int,
        default=0,
        help="Portfolio optimization rounds for cryptanalysis targets (0 = disabled)",
    )
    parser.add_argument(
        "--crypto-optimize-targets",
        default=default_crypto_optimize_targets(),
        help="Comma-separated cryptanalysis targets for Track B optimization",
    )
    parser.add_argument(
        "--crypto-optimize-batch-iterations",
        type=int,
        default=8,
        help="Iterations per target batch in cryptanalysis portfolio optimization",
    )
    parser.add_argument(
        "--crypto-optimize-batch-max-accepted",
        type=int,
        default=2,
        help="Max accepted per batch in cryptanalysis portfolio optimization",
    )
    parser.add_argument(
        "--crypto-optimize-schedule",
        choices=["round_robin", "ucb"],
        default="ucb",
        help="Scheduling strategy for cryptanalysis portfolio optimization",
    )
    parser.add_argument(
        "--crypto-optimize-ucb-explore",
        type=float,
        default=0.70,
        help="UCB exploration coefficient for cryptanalysis optimization",
    )
    parser.add_argument(
        "--mutation-memory-file",
        default="",
        help="Optional path forwarded to train.py/portfolio_loop.py --mutation-memory-file",
    )
    parser.add_argument(
        "--disable-mutation-memory",
        action="store_true",
        help="Disable cross-target mutation replay in train.py and portfolio_loop.py",
    )
    parser.add_argument(
        "--git-checkpoint-mode",
        choices=["off", "accepted", "all"],
        default="off",
        help="Forward Karpathy-style checkpoint commits to train.py loop runs",
    )
    parser.add_argument(
        "--git-checkpoint-prefix",
        default="autoresearch",
        help="Commit message prefix for train.py checkpoint commits",
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
        "--operator-wallet",
        default=os.getenv("AUTORESEARCH_OPERATOR_WALLET", ""),
        help="Operator wallet forwarded to submission_pack.py",
    )
    parser.add_argument(
        "--erc8004-identity",
        default=os.getenv("AUTORESEARCH_ERC8004_IDENTITY", ""),
        help="ERC-8004 identity forwarded to submission_pack.py",
    )
    parser.add_argument(
        "--erc8004-registration-tx",
        default=os.getenv("AUTORESEARCH_ERC8004_TX", ""),
        help="ERC-8004 registration tx hash/url forwarded to submission_pack.py",
    )
    parser.add_argument(
        "--additional-receipt",
        action="append",
        default=[],
        help="Additional onchain receipt forwarded to submission_pack.py (repeatable)",
    )
    parser.add_argument(
        "--submission-project-url",
        default="",
        help="Optional project URL forwarded to submission_pack.py",
    )
    parser.add_argument(
        "--submission-notes",
        default="",
        help="Optional submission notes forwarded to submission_pack.py",
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
    if args.crypto_optimize_rounds < 0:
        raise SystemExit("--crypto-optimize-rounds must be >= 0")

    if args.synthesis_cook:
        if args.track in {"performance", "hybrid"} and args.real_optimize_rounds == 0:
            args.real_optimize_rounds = 2
        if args.track in {"cryptanalysis", "hybrid"} and args.crypto_optimize_rounds == 0:
            args.crypto_optimize_rounds = 2
        if not args.build_evidence:
            args.build_evidence = True
        if not args.build_submission:
            args.build_submission = True
        if not args.build_readiness:
            args.build_readiness = True

    if args.track == "cryptanalysis" and args.loop_target == "cairo_poseidon_style_t8":
        # Track B defaults to cryptanalysis objective unless user explicitly overrides.
        args.loop_target = "poseidon2_cryptanalysis_trackb_kernel_fast"

    start_row_count = 0 if args.fresh or not RESULTS.exists() else len(parse_results())

    if args.fresh:
        memory_path = Path(args.mutation_memory_file) if args.mutation_memory_file else MUTATION_MEMORY
        if not memory_path.is_absolute():
            memory_path = ROOT / memory_path
        reset_outputs(memory_path)

    # Baseline and loop target optimization.
    campaign_log(args, f"baseline loop target={args.loop_target}")
    must_run(
        prepare_cmd(args, "baseline", "--target", args.loop_target, "--notes", "campaign_baseline"),
        label="baseline_loop_target",
        stream_stderr=args.verbose > 0,
    )
    campaign_log(
        args,
        (
            f"start train loop target={args.loop_target} iterations={args.loop_iterations} "
            f"max_accepted={args.max_accepted}"
        ),
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

    # Performance track baseline targets (active Lean stack context).
    real_targets: list[str] = []
    if args.track in {"performance", "hybrid"}:
        if args.real_profile == "fast":
            real_targets = ["leanmultisig_poseidon16_fast", "leanmultisig_xmss_fast"]
        elif args.real_profile == "full":
            real_targets = ["leanmultisig_poseidon16_full", "leanmultisig_xmss_fast"]

    if real_targets and not args.no_bootstrap:
        for target in real_targets:
            campaign_log(args, f"bootstrap real target={target}")
            must_run(
                prepare_cmd(args, "bootstrap", "--target", target),
                label=f"bootstrap_{target}",
                stream_stderr=args.verbose > 0,
            )

    for target in real_targets:
        campaign_log(args, f"baseline real target={target}")
        must_run(
            prepare_cmd(args, "baseline", "--target", target, "--notes", "campaign_real"),
            label=f"baseline_{target}",
            stream_stderr=args.verbose > 0,
        )

    portfolio_payload: dict[str, Any] | None = None
    if args.track in {"performance", "hybrid"} and args.real_optimize_rounds > 0:
        campaign_log(
            args,
            (
                f"start performance portfolio rounds={args.real_optimize_rounds} "
                f"targets={args.real_optimize_targets}"
            ),
        )
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

    crypto_targets: list[str] = []
    if args.track in {"cryptanalysis", "hybrid"}:
        crypto_targets = parse_targets_csv(args.crypto_optimize_targets)
        if not crypto_targets:
            crypto_targets = parse_targets_csv(default_crypto_optimize_targets())

    for target in crypto_targets:
        if target == args.loop_target:
            continue
        campaign_log(args, f"baseline cryptanalysis target={target}")
        must_run(
            prepare_cmd(args, "baseline", "--target", target, "--notes", "campaign_crypto"),
            label=f"baseline_{target}",
            stream_stderr=args.verbose > 0,
        )

    crypto_payload: dict[str, Any] | None = None
    if args.track in {"cryptanalysis", "hybrid"} and args.crypto_optimize_rounds > 0:
        campaign_log(
            args,
            (
                f"start cryptanalysis portfolio rounds={args.crypto_optimize_rounds} "
                f"targets={','.join(crypto_targets)}"
            ),
        )
        crypto_payload = must_run_json(
            portfolio_cmd(
                args,
                "--targets",
                ",".join(crypto_targets),
                "--rounds",
                str(args.crypto_optimize_rounds),
                "--batch-iterations",
                str(args.crypto_optimize_batch_iterations),
                "--batch-max-accepted",
                str(args.crypto_optimize_batch_max_accepted),
                "--schedule",
                args.crypto_optimize_schedule,
                "--ucb-explore",
                str(args.crypto_optimize_ucb_explore),
                "--artifacts",
                args.loop_artifacts,
            ),
            label="portfolio_crypto_optimize",
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
        task_categories = ["autonomous_research"]
        if args.track in {"performance", "hybrid"}:
            task_categories.extend(["cryptographic_optimization", "zk_proving_performance"])
        if args.track in {"cryptanalysis", "hybrid"}:
            task_categories.extend(["cryptanalysis", "hash_security_analysis"])
        seen_categories: set[str] = set()
        deduped_categories: list[str] = []
        for category in task_categories:
            if category in seen_categories:
                continue
            seen_categories.add(category)
            deduped_categories.append(category)

        submission_argv = [
            sys.executable,
            str(SUBMISSION_PACK),
            "--agent-name",
            args.agent_name,
            "--operator-wallet",
            args.operator_wallet.strip(),
            "--erc8004-identity",
            args.erc8004_identity.strip(),
            "--erc8004-registration-tx",
            args.erc8004_registration_tx.strip(),
        ]
        for category in deduped_categories:
            submission_argv.extend(["--task-category", category])
        if args.submission_project_url.strip():
            submission_argv.extend(["--project-url", args.submission_project_url.strip()])
        if args.submission_notes.strip():
            submission_argv.extend(["--submission-notes", args.submission_notes.strip()])
        for receipt in args.additional_receipt:
            receipt_value = str(receipt).strip()
            if receipt_value:
                submission_argv.extend(["--additional-receipt", receipt_value])
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
        track=args.track,
        rows=run_rows,
        loop_target=args.loop_target,
        real_targets=real_targets,
        crypto_targets=crypto_targets,
        loop_iterations=args.loop_iterations,
        loop_accepts=accepts,
        loop_payload=loop_payload,
        portfolio_payload=portfolio_payload,
        crypto_payload=crypto_payload,
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
                "track": args.track,
                "loop_payload": loop_payload,
                "portfolio_payload": portfolio_payload,
                "crypto_payload": crypto_payload,
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
