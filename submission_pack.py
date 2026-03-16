#!/usr/bin/env python3
"""Build canonical submission artifacts from AutoPoseidon logs.

Outputs:
- submission/agent.json
- submission/agent_log.json
- submission/submission_receipts.json
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from target_config import load_targets_file

ROOT = Path(__file__).resolve().parent
TARGETS_FILE = ROOT / "config" / "targets.json"
RESULTS_FILE = ROOT / "results.tsv"
LOG_FILE = ROOT / "agent_log.jsonl"
EVIDENCE_MANIFEST = ROOT / "evidence" / "manifest.json"
DEFAULT_OUT = ROOT / "submission"


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def parse_float(raw: str | None, default: float = 0.0) -> float:
    if raw is None:
        return default
    token = str(raw).strip()
    if not token:
        return default
    try:
        return float(token)
    except ValueError:
        return default


def read_results() -> list[dict[str, str]]:
    if not RESULTS_FILE.exists():
        return []
    with RESULTS_FILE.open("r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        token = line.strip()
        if not token:
            continue
        try:
            payload = json.loads(token)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            out.append(payload)
    return out


def read_json_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def load_targets() -> dict[str, dict[str, Any]]:
    return load_targets_file(TARGETS_FILE)


def infer_target_commands(target: dict[str, Any]) -> list[list[str]]:
    ttype = target.get("type")
    if ttype == "cairo":
        return [
            list(target.get("build_command", ["scarb", "build"])),
            list(target.get("test_command", ["scarb", "test"])),
        ]
    if ttype == "noir":
        return [["nargo", "check"], ["nargo", "info"], ["nargo", "execute"]]
    if ttype == "command":
        out: list[list[str]] = []
        seen: set[tuple[str, ...]] = set()
        default_cmd = target.get("benchmark_command")
        if isinstance(default_cmd, list):
            default_cmd = [str(part) for part in default_cmd]

        profiles = target.get("benchmark_profiles")
        if isinstance(profiles, list) and profiles:
            for profile in profiles:
                if not isinstance(profile, dict):
                    continue
                cmd = profile.get("benchmark_command")
                if cmd is None:
                    cmd = default_cmd
                if not isinstance(cmd, list) or not cmd:
                    continue
                normalized = tuple(str(part) for part in cmd)
                if normalized in seen:
                    continue
                seen.add(normalized)
                out.append([*normalized])
            return out

        cmd = default_cmd if isinstance(default_cmd, list) else []
        if isinstance(cmd, list) and cmd:
            normalized = tuple(str(part) for part in cmd)
            if normalized not in seen:
                out.append([*normalized])
        return out
    return []


def derive_tool_calls(rows: list[dict[str, str]], targets: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        target_name = row.get("target", "")
        target_cfg = targets.get(target_name, {})
        commands = infer_target_commands(target_cfg)
        if not commands:
            continue

        total_s = (
            parse_float(row.get("check_s"), 0.0)
            + parse_float(row.get("info_or_bench_s"), 0.0)
            + parse_float(row.get("execute_s"), 0.0)
        )
        per_call_s = total_s / max(1, len(commands))

        for jdx, cmd in enumerate(commands, start=1):
            tool = cmd[0] if cmd else "unknown"
            calls.append(
                {
                    "id": f"tc_{idx}_{jdx}",
                    "timestamp": row.get("timestamp"),
                    "target": target_name,
                    "iteration": int(row.get("iteration", "0") or 0),
                    "tool": tool,
                    "command": " ".join(cmd),
                    "duration_s_estimate": per_call_s,
                    "status": row.get("status", "unknown"),
                    "source": "results.tsv",
                }
            )
    return calls


def collect_loop_events(events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    loop_start = [e for e in events if e.get("event") == "loop_start"]
    loop_iter = [e for e in events if e.get("event") == "loop_iteration"]
    loop_end = [e for e in events if e.get("event") == "loop_end"]
    return loop_start, loop_iter, loop_end


def derive_decisions(loop_iter: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for event in loop_iter:
        result = event.get("result") if isinstance(event.get("result"), dict) else {}
        diag = event.get("diagnostics") if isinstance(event.get("diagnostics"), dict) else {}

        out.append(
            {
                "timestamp": event.get("timestamp"),
                "target": event.get("target"),
                "iteration": int(event.get("iteration", 0) or 0),
                "mutation": event.get("mutation", ""),
                "accepted": bool(event.get("accepted", False)),
                "status": result.get("status"),
                "metric_name": result.get("metric_name"),
                "metric_value": result.get("metric_value"),
                "guardrails": {
                    "evaluation_success": result.get("status") == "success",
                    "strict_improvement_gate": True,
                    "distribution_check": "distribution" in diag,
                    "ab_validation": "ab_validation" in diag,
                    "variance_check": "rel_stdev" in diag,
                },
            }
        )
    return out


def derive_decisions_from_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        iteration = int(row.get("iteration", "0") or 0)
        if iteration <= 0:
            continue
        notes = row.get("notes", "")
        mutation = notes.split(";", 1)[0] if notes else "row_fallback"
        accepted = mutation.startswith("accepted:")
        out.append(
            {
                "timestamp": row.get("timestamp"),
                "target": row.get("target"),
                "iteration": iteration,
                "mutation": mutation,
                "accepted": accepted,
                "status": row.get("status"),
                "metric_name": row.get("metric_name"),
                "metric_value": parse_float(row.get("metric_value"), default=float("nan")),
                "guardrails": {
                    "evaluation_success": row.get("status") == "success",
                    "strict_improvement_gate": True,
                    "distribution_check": "rejected_distribution" in notes,
                    "ab_validation": "rejected_ab" not in notes,
                    "variance_check": "rejected_high_variance" in notes,
                },
            }
        )
    return out


def derive_retries(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    attempts = Counter(str(d.get("mutation", "")) for d in decisions if d.get("mutation"))
    retries: list[dict[str, Any]] = []
    for mutation, count in sorted(attempts.items()):
        if count > 1:
            retries.append({"mutation": mutation, "attempts": count, "retries": count - 1})
    return retries


def derive_failures(rows: list[dict[str, str]], decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for row in rows:
        status = row.get("status", "")
        notes = row.get("notes", "")
        if status != "failed" and "rejected_" not in notes:
            continue
        failures.append(
            {
                "timestamp": row.get("timestamp"),
                "target": row.get("target"),
                "iteration": int(row.get("iteration", "0") or 0),
                "status": status,
                "notes": notes,
            }
        )

    # Include loop-level failures not captured in results rows.
    for d in decisions:
        if d.get("status") != "success":
            failures.append(
                {
                    "timestamp": d.get("timestamp"),
                    "target": d.get("target"),
                    "iteration": d.get("iteration"),
                    "status": d.get("status"),
                    "notes": "loop_iteration_non_success",
                }
            )
    return failures


def collect_final_outputs(rows: list[dict[str, str]], targets: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    by_target: dict[str, dict[str, Any]] = {}

    for row in rows:
        target = row.get("target", "")
        if row.get("status") != "success":
            continue
        value = parse_float(row.get("metric_value"), default=float("nan"))
        if value != value:  # NaN
            continue

        cfg = targets.get(target, {})
        higher_is_better = bool(cfg.get("higher_is_better", False))
        current = by_target.get(target)
        if current is None:
            by_target[target] = {
                "target": target,
                "metric_name": row.get("metric_name"),
                "metric_value": value,
                "higher_is_better": higher_is_better,
                "timestamp": row.get("timestamp"),
            }
            continue

        best_val = float(current["metric_value"])
        better = value > best_val if higher_is_better else value < best_val
        if better:
            current.update(
                {
                    "metric_name": row.get("metric_name"),
                    "metric_value": value,
                    "timestamp": row.get("timestamp"),
                }
            )

    manifest = read_json_dict(EVIDENCE_MANIFEST)
    accepted = manifest.get("accepted", []) if isinstance(manifest.get("accepted"), list) else []
    retained = [a for a in accepted if isinstance(a, dict) and bool(a.get("retained"))]

    outputs = sorted(by_target.values(), key=lambda item: item["target"])
    outputs.append(
        {
            "target": "_retained_patches",
            "metric_name": "retained_count",
            "metric_value": len(retained),
            "higher_is_better": True,
            "timestamp": now_iso(),
        }
    )
    return outputs


def derive_stages(
    *,
    rows: list[dict[str, str]],
    loop_start: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
) -> dict[str, int]:
    baseline_rows = [r for r in rows if int(r.get("iteration", "0") or 0) == 0]
    non_baseline_rows = [r for r in rows if int(r.get("iteration", "0") or 0) > 0]
    decision_count = len(decisions)
    non_baseline_count = len(non_baseline_rows)
    has_activity = bool(rows or loop_start or decisions)

    discover_count = max(
        1 if baseline_rows else 0,
        len(loop_start),
        1 if decision_count > 0 else 0,
        1 if non_baseline_count > 0 else 0,
    )
    plan_count = max(len(loop_start), 1 if has_activity else 0)
    execute_count = max(decision_count, non_baseline_count)
    verify_count = max(decision_count, non_baseline_count)
    submit_count = 1 if has_activity else 0

    return {
        "discover": discover_count,
        "plan": plan_count,
        "execute": execute_count,
        "verify": verify_count,
        "submit": submit_count,
    }


def derive_conversation_log(
    *,
    rows: list[dict[str, str]],
    loop_start: list[dict[str, Any]],
    loop_end: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seq = 0

    def push(
        *,
        stage: str,
        timestamp: str | None,
        summary: str,
        target: str | None = None,
        iteration: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        nonlocal seq
        seq += 1
        entry: dict[str, Any] = {
            "id": f"cl_{seq:05d}",
            "stage": stage,
            "timestamp": timestamp or now_iso(),
            "summary": summary,
        }
        if target:
            entry["target"] = target
        if iteration is not None:
            entry["iteration"] = int(iteration)
        if details:
            entry["details"] = details
        out.append(entry)

    for row in rows:
        try:
            iteration = int(row.get("iteration", "0") or 0)
        except ValueError:
            iteration = 0
        if iteration != 0:
            continue
        push(
            stage="discover",
            timestamp=row.get("timestamp"),
            target=row.get("target"),
            iteration=0,
            summary="Collected baseline target metric",
            details={
                "metric_name": row.get("metric_name"),
                "metric_value": parse_float(row.get("metric_value"), default=float("nan")),
                "status": row.get("status"),
            },
        )

    if not out:
        first_start = loop_start[0] if loop_start else {}
        first_row = rows[0] if rows else {}
        first_decision = decisions[0] if decisions else {}
        fallback_ts = (
            first_start.get("timestamp")
            or first_row.get("timestamp")
            or first_decision.get("timestamp")
            or now_iso()
        )
        fallback_target = (
            str(first_start.get("target", ""))
            or str(first_row.get("target", ""))
            or str(first_decision.get("target", ""))
        )
        if rows or loop_start or decisions:
            push(
                stage="discover",
                timestamp=fallback_ts,
                target=fallback_target,
                iteration=0,
                summary="Initialized target context and baseline assumptions",
            )

    for event in loop_start:
        compute_budget = event.get("compute_budget") if isinstance(event.get("compute_budget"), dict) else {}
        push(
            stage="plan",
            timestamp=event.get("timestamp"),
            target=str(event.get("target", "")),
            summary="Selected optimization policy and compute budget",
            details={
                "mode": event.get("mode"),
                "iterations": event.get("iterations"),
                "max_iterations": compute_budget.get("max_iterations"),
                "max_accepted": compute_budget.get("max_accepted"),
                "max_runtime_seconds": compute_budget.get("max_runtime_seconds"),
            },
        )

    if not loop_start and (rows or decisions):
        first_row = rows[0] if rows else {}
        first_decision = decisions[0] if decisions else {}
        push(
            stage="plan",
            timestamp=first_row.get("timestamp") or first_decision.get("timestamp") or now_iso(),
            target=str(first_row.get("target", "")) or str(first_decision.get("target", "")),
            summary="Recovered planning context from persisted run artifacts",
            details={
                "mode": "reconstructed",
                "source": "results.tsv_and_decisions",
                "reason": "loop_start_missing",
            },
        )

    for decision in decisions:
        ts = decision.get("timestamp")
        target = str(decision.get("target", ""))
        iteration = int(decision.get("iteration", 0) or 0)
        mutation = str(decision.get("mutation", ""))
        push(
            stage="execute",
            timestamp=ts,
            target=target,
            iteration=iteration,
            summary="Applied candidate mutation and executed benchmark pipeline",
            details={"mutation": mutation},
        )
        push(
            stage="verify",
            timestamp=ts,
            target=target,
            iteration=iteration,
            summary="Validated candidate against strict acceptance guardrails",
            details={
                "accepted": bool(decision.get("accepted", False)),
                "status": decision.get("status"),
                "metric_name": decision.get("metric_name"),
                "metric_value": decision.get("metric_value"),
            },
        )

    for event in loop_end:
        push(
            stage="submit",
            timestamp=event.get("timestamp"),
            target=str(event.get("target", "")),
            summary="Finalized run outputs and prepared submission artifacts",
            details={
                "accepted": event.get("accepted"),
                "best_metric": event.get("best_metric"),
                "stop_reason": event.get("stop_reason"),
                "elapsed_seconds": event.get("elapsed_seconds"),
            },
        )

    if not loop_end and decisions:
        push(
            stage="submit",
            timestamp=decisions[-1].get("timestamp"),
            target=str(decisions[-1].get("target", "")),
            summary="Finalized run outputs and prepared submission artifacts",
        )

    return out


def derive_budget(
    *,
    loop_start: list[dict[str, Any]],
    loop_end: list[dict[str, Any]],
    rows: list[dict[str, str]],
    decisions: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    budget_from_log: dict[str, Any] = {}
    if loop_start:
        latest = loop_start[-1]
        maybe_budget = latest.get("compute_budget")
        if isinstance(maybe_budget, dict):
            budget_from_log = maybe_budget

    total_eval_s = 0.0
    for row in rows:
        total_eval_s += parse_float(row.get("check_s"), 0.0)
        total_eval_s += parse_float(row.get("info_or_bench_s"), 0.0)
        total_eval_s += parse_float(row.get("execute_s"), 0.0)

    elapsed_wall_s = None
    if loop_end:
        last = loop_end[-1]
        try:
            elapsed_wall_s = float(last.get("elapsed_seconds"))
        except Exception:  # noqa: BLE001
            elapsed_wall_s = None

    max_iterations = args.compute_max_iterations
    if max_iterations is None:
        raw = budget_from_log.get("max_iterations")
        try:
            max_iterations = int(raw) if raw is not None else None
        except Exception:  # noqa: BLE001
            max_iterations = None

    max_accepted = args.compute_max_accepted
    if max_accepted is None:
        raw = budget_from_log.get("max_accepted")
        try:
            max_accepted = int(raw) if raw is not None else None
        except Exception:  # noqa: BLE001
            max_accepted = None

    max_runtime_s = args.compute_max_runtime_seconds
    if max_runtime_s is None:
        raw = budget_from_log.get("max_runtime_seconds")
        try:
            max_runtime_s = float(raw) if raw is not None else None
        except Exception:  # noqa: BLE001
            max_runtime_s = None

    accepted_count = sum(1 for d in decisions if bool(d.get("accepted")))
    model_calls = sum(1 for d in decisions if str(d.get("mutation", "")).startswith("openai_patch"))

    utilization: dict[str, float] = {}
    if max_iterations and max_iterations > 0:
        utilization["iterations_pct"] = (len(decisions) / max_iterations) * 100.0
    if max_accepted and max_accepted > 0:
        utilization["accepted_pct"] = (accepted_count / max_accepted) * 100.0
    if max_runtime_s and max_runtime_s > 0 and elapsed_wall_s is not None:
        utilization["runtime_pct"] = (elapsed_wall_s / max_runtime_s) * 100.0

    return {
        "constraints": {
            "max_iterations": max_iterations,
            "max_accepted": max_accepted,
            "max_runtime_seconds": max_runtime_s,
            "max_model_calls": args.compute_max_model_calls,
        },
        "consumed": {
            "iterations": len(decisions),
            "accepted": accepted_count,
            "estimated_model_calls": model_calls,
            "evaluation_seconds_total": total_eval_s,
            "wall_seconds_total": elapsed_wall_s,
        },
        "utilization": utilization,
    }


def derive_safety(rows: list[dict[str, str]], decisions: list[dict[str, Any]]) -> dict[str, Any]:
    rejected = 0
    eval_failed = 0
    dist_reject = 0
    variance_reject = 0

    for row in rows:
        notes = row.get("notes", "")
        if "rejected_" in notes:
            rejected += 1
        if "rejected_eval_failed" in notes or row.get("status") == "failed":
            eval_failed += 1
        if "rejected_distribution" in notes:
            dist_reject += 1
        if "rejected_high_variance" in notes:
            variance_reject += 1

    revert_count = sum(1 for d in decisions if not bool(d.get("accepted")))

    return {
        "policy": [
            "Reject non-improving candidates and automatically revert source state",
            "Accept only strict metric improvements under configurable relative/absolute thresholds",
            "Require successful compile/test/benchmark before considering acceptance",
            "Optional distribution, variance, and A/B confirmation gates block unsafe wins",
        ],
        "counters": {
            "rejected_candidates": rejected,
            "evaluation_failures": eval_failed,
            "distribution_rejections": dist_reject,
            "variance_rejections": variance_reject,
            "automatic_reverts": revert_count,
        },
    }


def normalize_task_categories(raw: list[str]) -> list[str]:
    if raw:
        clean = [token.strip() for token in raw if token.strip()]
        if clean:
            return clean
    return ["cryptographic_optimization", "zk_proving_performance", "autonomous_research"]


def build_agent_manifest(
    *,
    args: argparse.Namespace,
    targets_seen: list[str],
    targets: dict[str, dict[str, Any]],
    budget: dict[str, Any],
) -> dict[str, Any]:
    tools: set[str] = {"python"}
    stacks: set[str] = {"python"}

    for target_name in targets_seen:
        cfg = targets.get(target_name, {})
        for cmd in infer_target_commands(cfg):
            if cmd:
                tools.add(cmd[0])
        language = cfg.get("language")
        if isinstance(language, str) and language.strip():
            stacks.add(language.strip().lower())
        ttype = cfg.get("type")
        if isinstance(ttype, str):
            stacks.add(ttype)

    manifest = {
        "schema_version": "1.0",
        "generated_at": now_iso(),
        "name": args.agent_name,
        "description": args.description,
        "operator_wallet": args.operator_wallet.strip(),
        "erc8004_identity": args.erc8004_identity.strip(),
        "erc8004_registration_tx": args.erc8004_registration_tx.strip(),
        "supported_tools": sorted(tools),
        "tech_stacks": sorted(stacks),
        "compute_constraints": budget.get("constraints", {}),
        "task_categories": normalize_task_categories(args.task_category),
        "targets_seen": targets_seen,
    }
    return manifest


def build_submission_receipts(
    *,
    args: argparse.Namespace,
    stages: dict[str, int],
    output_paths: dict[str, str],
    warnings: list[str],
) -> dict[str, Any]:
    additional = [token.strip() for token in args.additional_receipt if token.strip()]
    return {
        "schema_version": "1.0",
        "generated_at": now_iso(),
        "erc8004": {
            "identity": args.erc8004_identity.strip(),
            "registration_tx": args.erc8004_registration_tx.strip(),
            "additional_receipts": additional,
        },
        "autonomous_stage_counts": stages,
        "deliverables": output_paths,
        "submission": {
            "project_url": args.project_url.strip(),
            "notes": args.submission_notes.strip(),
        },
        "warnings": warnings,
    }


def required_identity_missing(manifest: dict[str, Any]) -> list[str]:
    required = ["operator_wallet", "erc8004_identity", "erc8004_registration_tx"]
    missing: list[str] = []
    for key in required:
        value = manifest.get(key)
        if not isinstance(value, str) or not value.strip():
            missing.append(key)
    return missing


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate canonical submission artifacts from AutoPoseidon logs")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output directory for submission artifacts")
    parser.add_argument("--agent-name", default="AutoPoseidon", help="Agent display name")
    parser.add_argument(
        "--description",
        default="Autonomous optimization agent for cryptographic circuit and prover performance",
        help="Agent description",
    )
    parser.add_argument("--operator-wallet", default=os.getenv("AUTORESEARCH_OPERATOR_WALLET", ""), help="Operator wallet")
    parser.add_argument("--erc8004-identity", default=os.getenv("AUTORESEARCH_ERC8004_IDENTITY", ""), help="ERC-8004 identity")
    parser.add_argument(
        "--erc8004-registration-tx",
        default=os.getenv("AUTORESEARCH_ERC8004_TX", ""),
        help="ERC-8004 registration transaction hash or explorer URL",
    )
    parser.add_argument(
        "--additional-receipt",
        action="append",
        default=[],
        help="Additional onchain receipt (tx hash or explorer URL)",
    )
    parser.add_argument("--project-url", default="", help="Submission project URL (optional)")
    parser.add_argument("--submission-notes", default="", help="Short submission note")
    parser.add_argument(
        "--task-category",
        action="append",
        default=[],
        help="Task category label (repeatable)",
    )
    parser.add_argument("--compute-max-iterations", type=int, default=None, help="Override compute max iterations")
    parser.add_argument("--compute-max-accepted", type=int, default=None, help="Override compute max accepted")
    parser.add_argument(
        "--compute-max-runtime-seconds",
        type=float,
        default=None,
        help="Override compute max runtime in seconds",
    )
    parser.add_argument("--compute-max-model-calls", type=int, default=None, help="Optional max model call budget")
    parser.add_argument("--strict", action="store_true", help="Fail if required ERC-8004/operator fields are missing")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = load_targets()
    rows = read_results()
    events = read_jsonl(LOG_FILE)
    loop_start, loop_iter, loop_end = collect_loop_events(events)

    decisions = derive_decisions(loop_iter)
    if not decisions:
        decisions = derive_decisions_from_rows(rows)
    retries = derive_retries(decisions)
    failures = derive_failures(rows, decisions)
    tool_calls = derive_tool_calls(rows, targets)

    targets_seen = sorted({row.get("target", "") for row in rows if row.get("target")})
    stages = derive_stages(rows=rows, loop_start=loop_start, decisions=decisions)
    conversation_log = derive_conversation_log(
        rows=rows,
        loop_start=loop_start,
        loop_end=loop_end,
        decisions=decisions,
    )
    budget = derive_budget(
        loop_start=loop_start,
        loop_end=loop_end,
        rows=rows,
        decisions=decisions,
        args=args,
    )
    safety = derive_safety(rows, decisions)
    final_outputs = collect_final_outputs(rows, targets)

    manifest = build_agent_manifest(
        args=args,
        targets_seen=targets_seen,
        targets=targets,
        budget=budget,
    )

    warnings: list[str] = []
    missing_identity = required_identity_missing(manifest)
    if missing_identity:
        warnings.append(f"missing required identity fields: {', '.join(missing_identity)}")

    if args.strict and missing_identity:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "missing_identity_fields",
                    "missing": missing_identity,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    agent_log = {
        "schema_version": "1.0",
        "generated_at": now_iso(),
        "source_files": {
            "results": str(RESULTS_FILE),
            "raw_event_log": str(LOG_FILE),
        },
        "stages": stages,
        "summary": {
            "targets_seen": targets_seen,
            "rows_total": len(rows),
            "decisions_total": len(decisions),
            "accepted_total": sum(1 for d in decisions if bool(d.get("accepted"))),
            "rejected_total": sum(1 for d in decisions if not bool(d.get("accepted"))),
            "tool_calls_total": len(tool_calls),
            "failures_total": len(failures),
            "conversation_events_total": len(conversation_log),
        },
        "compute_budget": budget,
        "safety_guardrails": safety,
        "conversation_log": conversation_log,
        "decisions": decisions,
        "tool_calls": tool_calls,
        "retries": retries,
        "failures": failures,
        "final_outputs": final_outputs,
    }

    output_paths = {
        "agent_json": str(out_dir / "agent.json"),
        "agent_log_json": str(out_dir / "agent_log.json"),
        "submission_receipts_json": str(out_dir / "submission_receipts.json"),
    }
    receipts = build_submission_receipts(args=args, stages=stages, output_paths=output_paths, warnings=warnings)

    (out_dir / "agent.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "agent_log.json").write_text(json.dumps(agent_log, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "submission_receipts.json").write_text(
        json.dumps(receipts, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    payload = {
        "ok": True,
        "out_dir": str(out_dir),
        "files": output_paths,
        "warnings": warnings,
        "stages": stages,
        "accepted_total": agent_log["summary"]["accepted_total"],
        "tool_calls_total": agent_log["summary"]["tool_calls_total"],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
