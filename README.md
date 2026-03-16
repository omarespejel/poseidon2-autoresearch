# AutoPoseidon

AutoPoseidon is a reproducible autoresearch harness for cryptographic circuit optimization.

This scaffold focuses on what is quickly verifiable:

- autonomous iteration loop
- measurable metric deltas
- deterministic accept/reject policy
- machine-readable audit trail (`agent_log.jsonl`)

## What Is Included

- `prepare.py`: setup + target evaluation + baseline recording (fixed/read-only)
- `train.py`: autonomous optimization loop (editable research loop file)
- `program.md`: human-authored instructions for the agent
- `run_loop.py`: backward-compatible alias to `train.py`
- `portfolio_loop.py`: adaptive multi-target runner (avoids single-target plateaus)
- `evidence_pack.py`: generate submission-grade evidence bundle from logs/artifacts
- `submission_pack.py`: generate canonical `agent.json`, `agent_log.json`, and receipts bundle
- `readiness_check.py`: submission-readiness checker (artifacts + activity + timeline gate)
- `config/targets.json`: benchmark targets and commands
- `examples/cairo_poseidon_style`: primary Cairo optimization sandbox
- `examples/noir_poseidon2_style_t8`: optional Noir optimization sandbox
- `examples/noir_poseidon2_style_t16`: optional larger Noir sandbox
- `results.tsv`: append-only experiment table
- `agent_log.jsonl`: structured event log for every run
- `artifacts/<target>/<run_label>/iter_*`: per-iteration source/diff/debug bundles (run-scoped)
  - includes `environment.json` (runtime controls, tool versions, git snapshots)

## Fast Start

From repository root:

```bash
python3 prepare.py list-targets
python3 prepare.py baseline --target cairo_poseidon_style_t8 --notes baseline
python3 train.py --target cairo_poseidon_style_t8 --iterations 12 --max-accepted 3
```

## Karpathy-Compatible Mode

The repo now supports the same top-level workflow shape:

- `prepare.py` (fixed)
- `train.py` (agent loop / editable)
- `program.md` (human instructions)

This keeps compatibility with prompts and workflows written for Karpathy-style autoresearch projects.
Operational instructions are in `codex_instructions.md`.
Forward roadmap is in `NEXT_STEPS.md`.
Date-anchored execution checklist is in `SPRINT_PLAN_2026-03-15_to_2026-03-22.md`.
Deep external validation notes are in `DEEP_RESEARCH_2026-03-15.md`.
Noise calibration snapshots are tracked in `calibration_report.md`.

## Adaptive Portfolio Mode

For broader search across multiple Rust hotspots:

```bash
python3 portfolio_loop.py --rounds 4 --batch-iterations 6 --batch-max-accepted 1 --artifacts accepted
```

To inject per-target acceptance overrides (for example, from a calibration pass):

```bash
python3 portfolio_loop.py --rounds 2 --batch-iterations 4 --target-overrides-json work/checkpoint_target_overrides.json
```

This writes:

- `portfolio_report.md`
- `portfolio_report.json`

To build a reproducibility/evidence bundle:

```bash
python3 evidence_pack.py
```

This writes:

- `evidence/manifest.json`
- `evidence/summary.md`
- `evidence/retained_artifacts/*`
  - accepted rows include commit hints when available (`environment.json` -> git metadata)
  - legacy accepted rows are matched to artifacts by timestamp proximity to avoid iteration-collision misattribution
  - confidence signals from metadata diagnostics are summarized in `evidence/summary.md`

To generate a readiness checkpoint report:

```bash
python3 readiness_check.py
```

To generate canonical submission artifacts:

```bash
python3 submission_pack.py --agent-name AutoPoseidon
```

To run full checkpoint cycles with calibration-driven threshold overrides:

```bash
python3 checkpoint_loop.py --cycles 2 --calibration-samples 3 --auto-threshold-overrides --override-force-fixed-mode
```

## Optional: Active Lean Stack Baseline

These targets track active Lean/Poseidon2 runtime throughput:

```bash
python3 prepare.py bootstrap --target leanmultisig_poseidon16_fast
python3 prepare.py baseline --target leanmultisig_poseidon16_fast --notes lean_baseline
python3 prepare.py baseline --target leanmultisig_xmss_fast --notes lean_xmss
```

Lean command targets use warmup + repeated runs with median aggregation for more stable throughput metrics.
Targets can optionally trim outliers before aggregation with `trim_extremes`.
For noisy throughput metrics, acceptance also uses relative improvement thresholds from `config/targets.json`.
Thresholds can be static (`fixed`) or noise-adaptive (`adaptive` / `floor`) via:

- `min_improvement_rel_mode`
- `min_improvement_rel_sigma`
- `min_improvement_rel_min`
- `min_improvement_rel_max`
Command targets can also require confirmation evaluations (`confirm_repeats`) before accepting a mutation.
Optional variance guards (`max_rel_stdev`) can reject noisy benchmark batches automatically.
Source-level command targets can additionally require distribution separation
(`min_effect_sigma`, `ci_z`, `require_ci_separation`) before accepting a run.
They can also run post-accept A/B replay (`ab_repeats`) to confirm patched vs original.
Rejected mutations can be temporarily cooled down (`blocked_mutation_ttl`) to encourage broader exploration.
Optional runtime controls are supported during benchmark commands:

- `AUTORESEARCH_NICE=<level>` (process niceness)
- `AUTORESEARCH_CPU_AFFINITY=<cpu-list>` (uses `taskset -c` when available)

### Calibration Pass (Recommended)

Before long runs, measure target noise empirically:

```bash
python3 prepare.py calibrate --target leanmultisig_poseidon16_src_fast --samples 5
python3 prepare.py calibrate --target leanmultisig_poseidon16_table_src_fast --samples 5
python3 prepare.py calibrate --target leanmultisig_poseidon2_neon_src_fast --samples 5
```

For direct Rust autoresearch against Lean source code:

```bash
python3 prepare.py baseline --target leanmultisig_poseidon16_src_fast --notes lean_src_baseline
python3 train.py --target leanmultisig_poseidon16_src_fast --iterations 5 --max-accepted 1 --artifacts accepted
```

Additional source-level targets are also available for deeper Poseidon2 hotspots:

- `leanmultisig_poseidon16_table_src_fast`
- `leanmultisig_poseidon2_neon_src_fast`

Reference implementation context can be inspected in:

- `work/leanMultisig`
- `work/leanSig`

## LLM Mode (Optional)

`train.py` defaults to deterministic heuristics. To enable model-proposed patches:

```bash
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-5-mini
python3 train.py --target cairo_poseidon_style_t8 --iterations 25
```

The loop falls back to heuristics automatically if an API call fails.

## Cross-Target Mutation Replay

Rust targets now share accepted mutation history through persisted replay memory:

- default file: `work/mutation_memory.json`
- memory is seeded from historical `results.tsv` accepted/rejected rows
- successful mutations on one Rust target are prioritized on other Rust targets

Controls:

```bash
# default behavior (memory enabled)
python3 train.py --target leanmultisig_poseidon16_src_fast --iterations 12

# custom memory file
python3 train.py --target leanmultisig_poseidon2_neon_src_fast --mutation-memory-file work/custom_memory.json

# disable replay memory
python3 train.py --target leanmultisig_poseidon2_neon_src_fast --disable-mutation-memory
```

## Acceptance Policy

A candidate is accepted only when:

- evaluation succeeds (`scarb build` + `scarb test` for Cairo targets, or the equivalent target-specific checks)
- metric strictly improves relative to best-so-far
- optional noise guards pass (`max_rel_stdev`, distribution/effect-size gate for command targets)
- optional confirmation runs (`confirm_repeats`) still beat the current best

Otherwise the file is reverted.

## Notes

- The included Cairo/Noir targets are optimization sandboxes, not audited production cryptography.
- For production or research publication, replace sandbox targets with audited production implementations.
