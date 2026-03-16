#!/usr/bin/env python3
"""Generate a reproducible evidence bundle from AutoPoseidon outputs."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from target_config import load_targets_file

ROOT = Path(__file__).resolve().parent
TARGETS_FILE = ROOT / "config" / "targets.json"
RESULTS_FILE = ROOT / "results.tsv"
LOG_FILE = ROOT / "agent_log.jsonl"
ARTIFACTS_DIR = ROOT / "artifacts"
DEFAULT_OUT = ROOT / "evidence"
MAX_ARTIFACT_MATCH_AGE_S = 15 * 60
REPRO_PY_COMPILE_FILES = [
    "campaign.py",
    "checkpoint_loop.py",
    "evidence_pack.py",
    "portfolio_loop.py",
    "prepare.py",
    "readiness_check.py",
    "target_config.py",
    "train.py",
]


@dataclass
class AcceptedRecord:
    timestamp: str
    target: str
    iteration: int
    metric_name: str
    metric_value: float | None
    best_value: float | None
    notes: str
    run_label: str | None
    source_file: str | None
    artifact_iter_dir: str | None
    metadata_file: str | None
    metadata_timestamp: str | None
    env_file: str | None
    git_commit: str | None
    match_age_s: float | None
    confidence: str
    baseline_median: float | None
    candidate_median: float | None
    gain_abs: float | None
    gain_pct: float | None
    effect_sigma: float | None
    rel_stdev: float | None
    ab_gain_pct: float | None
    confirm_n: int | None
    retained: bool
    retained_artifact_dir: str | None


def load_targets() -> dict[str, dict[str, Any]]:
    return load_targets_file(TARGETS_FILE)


def read_results() -> list[dict[str, str]]:
    if not RESULTS_FILE.exists():
        return []
    with RESULTS_FILE.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def parse_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    token = raw.strip()
    if not token:
        return None
    try:
        return float(token)
    except ValueError:
        return None


def parse_int(raw: str, default: int = 0) -> int:
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def parse_run_label(notes: str) -> str | None:
    match = re.search(r"(?:^|;)run=([^;]+)", notes)
    if not match:
        return None
    return match.group(1).strip() or None


def parse_iso_timestamp(raw: str | None) -> dt.datetime | None:
    if not raw:
        return None
    token = raw.strip()
    if not token:
        return None
    try:
        return dt.datetime.fromisoformat(token.replace("Z", "+00:00"))
    except ValueError:
        return None


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(data, dict):
        return data
    return None


def metadata_timestamp(metadata_file: Path) -> dt.datetime | None:
    data = read_json(metadata_file)
    if not data:
        return None
    return parse_iso_timestamp(str(data.get("timestamp", "")))


def metadata_run_label(metadata_file: Path) -> str | None:
    data = read_json(metadata_file)
    if not data:
        return None
    value = data.get("run_label")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def artifact_match_score(metadata_file: Path, row_ts: dt.datetime | None) -> float | None:
    if row_ts is None:
        return 0.0
    ts = metadata_timestamp(metadata_file)
    if ts is None:
        try:
            return abs(metadata_file.stat().st_mtime - row_ts.timestamp())
        except OSError:
            return None
    return abs((ts - row_ts).total_seconds())


def find_artifact_metadata(
    target: str,
    iteration: int,
    run_label: str | None,
    row_timestamp: str,
) -> tuple[Path | None, Path | None, float | None]:
    iter_name = f"iter_{iteration:05d}"
    target_dir = (ARTIFACTS_DIR / target).resolve()
    try:
        target_dir.relative_to(ARTIFACTS_DIR.resolve())
    except ValueError:
        return None, None, None
    if not target_dir.exists():
        return None, None, None

    if run_label:
        explicit = target_dir / run_label / iter_name / "metadata.json"
        if explicit.exists():
            return explicit.parent, explicit, 0.0
        return None, None, None

    row_ts = parse_iso_timestamp(row_timestamp)
    candidates = list(target_dir.glob(f"**/{iter_name}/metadata.json"))
    if not candidates:
        return None, None, None

    ranked = sorted(
        (
            (score, candidate)
            for candidate in candidates
            for score in [artifact_match_score(candidate, row_ts)]
            if score is not None
        ),
        key=lambda item: item[0],
    )
    if not ranked:
        return None, None, None

    score, metadata = ranked[0]
    if row_ts is not None and score > MAX_ARTIFACT_MATCH_AGE_S:
        return None, None, score

    return metadata.parent, metadata, score


def format_confidence(metadata_file: Path | None) -> str:
    if metadata_file is None or not metadata_file.exists():
        return "n/a"
    data = read_json(metadata_file)
    if not data:
        return "n/a"
    diag = data.get("diagnostics", {})
    if not isinstance(diag, dict):
        return "n/a"

    parts: list[str] = []

    rel = diag.get("rel_stdev")
    if isinstance(rel, (int, float)):
        parts.append(f"rstd={float(rel) * 100:.2f}%")

    confirm_values = diag.get("confirm_values")
    if isinstance(confirm_values, list):
        parts.append(f"confirm_n={len(confirm_values)}")

    dist = diag.get("distribution")
    if isinstance(dist, dict):
        delta = dist.get("delta")
        effect = dist.get("effect_sigma")
        if isinstance(delta, (int, float)):
            parts.append(f"dMed={float(delta):.2f}")
        if isinstance(effect, (int, float)):
            parts.append(f"sEff={float(effect):.2f}")

    ab = diag.get("ab_validation")
    if isinstance(ab, dict):
        cand = ab.get("candidate_median")
        base = ab.get("original_median")
        if isinstance(cand, (int, float)) and isinstance(base, (int, float)) and float(base) != 0.0:
            pct = ((float(cand) / float(base)) - 1.0) * 100.0
            parts.append(f"ab={pct:+.2f}%")

    if not parts:
        return "n/a"
    return "; ".join(parts)


def summarize_diagnostics(metadata_file: Path | None) -> dict[str, Any]:
    out = {
        "baseline_median": None,
        "candidate_median": None,
        "gain_abs": None,
        "gain_pct": None,
        "effect_sigma": None,
        "rel_stdev": None,
        "ab_gain_pct": None,
        "confirm_n": None,
    }
    if metadata_file is None or not metadata_file.exists():
        return out

    data = read_json(metadata_file)
    if not data:
        return out
    diag = data.get("diagnostics", {})
    if not isinstance(diag, dict):
        return out

    rel = diag.get("rel_stdev")
    if isinstance(rel, (int, float)):
        out["rel_stdev"] = float(rel)

    confirm_values = diag.get("confirm_values")
    if isinstance(confirm_values, list):
        out["confirm_n"] = len(confirm_values)

    dist = diag.get("distribution")
    if isinstance(dist, dict):
        base = dist.get("baseline_median")
        cand = dist.get("candidate_median")
        eff = dist.get("effect_sigma")
        if isinstance(base, (int, float)):
            out["baseline_median"] = float(base)
        if isinstance(cand, (int, float)):
            out["candidate_median"] = float(cand)
        if isinstance(eff, (int, float)):
            out["effect_sigma"] = float(eff)

    ab = diag.get("ab_validation")
    if isinstance(ab, dict):
        cand = ab.get("candidate_median")
        base = ab.get("original_median")
        if isinstance(cand, (int, float)):
            out["candidate_median"] = float(cand)
        if isinstance(base, (int, float)):
            out["baseline_median"] = float(base)
        if isinstance(cand, (int, float)) and isinstance(base, (int, float)) and float(base) != 0.0:
            out["ab_gain_pct"] = ((float(cand) / float(base)) - 1.0) * 100.0

    base = out["baseline_median"]
    cand = out["candidate_median"]
    if isinstance(base, float) and isinstance(cand, float):
        out["gain_abs"] = cand - base
        if base != 0.0:
            out["gain_pct"] = ((cand / base) - 1.0) * 100.0

    return out


def read_env_commit(env_file: Path | None) -> str | None:
    if env_file is None or not env_file.exists():
        return None
    try:
        data = json.loads(env_file.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    git_rows = data.get("git")
    if not isinstance(git_rows, list):
        return None
    for row in git_rows:
        if isinstance(row, dict) and row.get("label") == "target_project_dir":
            commit = row.get("commit")
            if isinstance(commit, str) and commit.strip():
                return commit.strip()
    for row in git_rows:
        if isinstance(row, dict):
            commit = row.get("commit")
            if isinstance(commit, str) and commit.strip():
                return commit.strip()
    return None


def normalize_text_bytes(raw: bytes) -> bytes:
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    return raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def is_retained(*, source_file: Path | None, artifact_iter_dir: Path | None) -> bool:
    if source_file is None or artifact_iter_dir is None or not source_file.exists():
        return False
    after_candidates = sorted(artifact_iter_dir.glob("after.*"))
    if not after_candidates:
        return False
    try:
        source_bytes = normalize_text_bytes(source_file.read_bytes())
        after_bytes = normalize_text_bytes(after_candidates[0].read_bytes())
        return source_bytes == after_bytes
    except OSError:
        return False


def collect_accepted_rows(targets: dict[str, dict[str, Any]], rows: list[dict[str, str]]) -> list[AcceptedRecord]:
    accepted: list[AcceptedRecord] = []

    for row in rows:
        notes = row.get("notes", "")
        if not notes.startswith("accepted:"):
            continue

        target = row.get("target", "")
        iteration = parse_int(row.get("iteration", "0"), default=0)
        run_label = parse_run_label(notes)
        cfg = targets.get(target, {})
        source_rel = cfg.get("source_file")
        source_file = ROOT / source_rel if isinstance(source_rel, str) else None

        timestamp = row.get("timestamp", "")
        artifact_iter_dir, metadata_file, match_age_s = find_artifact_metadata(target, iteration, run_label, timestamp)
        env_file = artifact_iter_dir / "environment.json" if artifact_iter_dir else None
        git_commit = read_env_commit(env_file)
        mtime = metadata_timestamp(metadata_file) if metadata_file else None
        retained = is_retained(source_file=source_file, artifact_iter_dir=artifact_iter_dir)
        confidence = format_confidence(metadata_file)
        diag_summary = summarize_diagnostics(metadata_file)
        artifact_run = metadata_run_label(metadata_file) if metadata_file else None
        effective_run = run_label or artifact_run

        accepted.append(
            AcceptedRecord(
                timestamp=timestamp,
                target=target,
                iteration=iteration,
                metric_name=row.get("metric_name", ""),
                metric_value=parse_float(row.get("metric_value", "")),
                best_value=parse_float(row.get("best_value", "")),
                notes=notes,
                run_label=effective_run,
                source_file=str(source_file) if source_file else None,
                artifact_iter_dir=str(artifact_iter_dir) if artifact_iter_dir else None,
                metadata_file=str(metadata_file) if metadata_file else None,
                metadata_timestamp=mtime.isoformat() if mtime else None,
                env_file=str(env_file) if env_file else None,
                git_commit=git_commit,
                match_age_s=match_age_s,
                confidence=confidence,
                baseline_median=diag_summary["baseline_median"],
                candidate_median=diag_summary["candidate_median"],
                gain_abs=diag_summary["gain_abs"],
                gain_pct=diag_summary["gain_pct"],
                effect_sigma=diag_summary["effect_sigma"],
                rel_stdev=diag_summary["rel_stdev"],
                ab_gain_pct=diag_summary["ab_gain_pct"],
                confirm_n=diag_summary["confirm_n"],
                retained=retained,
                retained_artifact_dir=None,
            )
        )

    return accepted


def write_manifest(*, out_dir: Path, accepted: list[AcceptedRecord], total_rows: int) -> Path:
    manifest = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "results_file": str(RESULTS_FILE),
        "agent_log_file": str(LOG_FILE),
        "rows_total": total_rows,
        "accepted_total": len(accepted),
        "accepted": [
            {
                "timestamp": r.timestamp,
                "target": r.target,
                "iteration": r.iteration,
                "metric_name": r.metric_name,
                "metric_value": r.metric_value,
                "best_value": r.best_value,
                "notes": r.notes,
                "run_label": r.run_label,
                "source_file": r.source_file,
                "artifact_iter_dir": r.artifact_iter_dir,
                "metadata_file": r.metadata_file,
                "metadata_timestamp": r.metadata_timestamp,
                "env_file": r.env_file,
                "git_commit": r.git_commit,
                "match_age_s": r.match_age_s,
                "confidence": r.confidence,
                "baseline_median": r.baseline_median,
                "candidate_median": r.candidate_median,
                "gain_abs": r.gain_abs,
                "gain_pct": r.gain_pct,
                "effect_sigma": r.effect_sigma,
                "rel_stdev": r.rel_stdev,
                "ab_gain_pct": r.ab_gain_pct,
                "confirm_n": r.confirm_n,
                "retained": r.retained,
                "retained_artifact_dir": r.retained_artifact_dir,
            }
            for r in accepted
        ],
    }
    out = out_dir / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return out


def short_metric(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    return f"{value:.2f}"


def short_pct(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    return f"{value:+.2f}%"


def short_small(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    return f"{value:.3f}"


def unique_retained_records(accepted: list[AcceptedRecord]) -> list[AcceptedRecord]:
    seen: set[tuple[str | None, str | None]] = set()
    out: list[AcceptedRecord] = []
    for record in accepted:
        if not record.retained:
            continue
        key = (record.source_file, record.artifact_iter_dir)
        if key in seen:
            continue
        seen.add(key)
        out.append(record)
    return out


def write_summary(*, out_dir: Path, accepted: list[AcceptedRecord], total_rows: int) -> Path:
    retained = unique_retained_records(accepted)
    lines: list[str] = []
    lines.append("# AutoPoseidon Evidence Pack")
    lines.append("")
    lines.append("## Snapshot")
    lines.append("")
    lines.append(f"- results rows: `{total_rows}`")
    lines.append(f"- accepted rows: `{len(accepted)}`")
    lines.append(f"- currently retained accepted patches: `{len(retained)}`")
    lines.append(f"- source tables: `{RESULTS_FILE}` and `{LOG_FILE}`")
    lines.append("")

    lines.append("## Accepted History")
    lines.append("")
    lines.append("| timestamp | target | run | iter | metric | retained | commit | confidence | notes | artifact |")
    lines.append("|---|---|---|---:|---:|---:|---|---|---|---|")
    for r in accepted:
        artifact = r.artifact_iter_dir or "n/a"
        commit = r.git_commit[:12] if r.git_commit else "n/a"
        lines.append(
            f"| {r.timestamp} | {r.target} | {r.run_label or 'n/a'} | {r.iteration} | {short_metric(r.metric_value)} | {'yes' if r.retained else 'no'} | {commit} | {r.confidence} | {r.notes} | {artifact} |"
        )

    lines.append("")
    lines.append("## Retained Patches")
    lines.append("")
    if retained:
        for r in retained:
            source = r.source_file or "n/a"
            artifact = r.artifact_iter_dir or "n/a"
            commit = r.git_commit[:12] if r.git_commit else "n/a"
            lines.append(
                f"- `{r.target}` iteration `{r.iteration}` -> `{source}` (commit: `{commit}`, artifact: `{artifact}`)"
            )
    else:
        lines.append("- none")

    lines.append("")
    lines.append("## Retained Confidence Deltas")
    lines.append("")
    lines.append("| target | iter | baseline median | candidate median | gain % | gain abs | effect sigma | rel stdev | A/B gain % | confirm_n |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    if retained:
        for r in retained:
            lines.append(
                "| "
                f"{r.target} | {r.iteration} | {short_metric(r.baseline_median)} | {short_metric(r.candidate_median)} | "
                f"{short_pct(r.gain_pct)} | {short_metric(r.gain_abs)} | {short_small(r.effect_sigma)} | "
                f"{short_pct((r.rel_stdev * 100.0) if r.rel_stdev is not None else None)} | {short_pct(r.ab_gain_pct)} | "
                f"{r.confirm_n if r.confirm_n is not None else 'n/a'} |"
            )
    else:
        lines.append("| none | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")

    lines.append("")
    lines.append("## Reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append(f"python3 -m py_compile {' '.join(REPRO_PY_COMPILE_FILES)}")
    lines.append("python3 portfolio_loop.py --rounds 4 --batch-iterations 6 --batch-max-accepted 1 --artifacts accepted")
    lines.append("python3 evidence_pack.py")
    lines.append("```")

    out = out_dir / "summary.md"
    out.write_text("\n".join(lines) + "\n")
    return out


def copy_retained_artifacts(*, out_dir: Path, accepted: list[AcceptedRecord]) -> list[str]:
    retained_dir = out_dir / "retained_artifacts"
    retained_dir.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    retained = unique_retained_records(accepted)
    for index, r in enumerate(retained, start=1):
        if not r.artifact_iter_dir:
            continue
        src = Path(r.artifact_iter_dir)
        if not src.exists():
            continue

        ts = re.sub(r"[^0-9A-Za-z]+", "", r.timestamp)
        tag = f"{index:03d}__{r.target}__iter_{r.iteration:05d}__{ts}"
        dst = retained_dir / tag
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        r.retained_artifact_dir = str(dst)
        copied.append(str(dst))

    return copied


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an evidence pack from AutoPoseidon logs")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output directory for evidence files")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = load_targets()
    rows = read_results()
    accepted = collect_accepted_rows(targets, rows)

    copied = copy_retained_artifacts(out_dir=out_dir, accepted=accepted)
    manifest = write_manifest(out_dir=out_dir, accepted=accepted, total_rows=len(rows))
    summary = write_summary(out_dir=out_dir, accepted=accepted, total_rows=len(rows))

    payload = {
        "ok": True,
        "out_dir": str(out_dir),
        "manifest": str(manifest),
        "summary": str(summary),
        "rows_total": len(rows),
        "accepted_total": len(accepted),
        "retained_artifacts": copied,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
