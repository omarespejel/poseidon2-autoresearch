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
- `attack_harness.py`: deterministic reduced-round cryptanalysis harness (Track B signal/verified/algebraic targets)
- `program.md`: human-authored instructions for the agent
- `run_loop.py`: backward-compatible alias to `train.py`
- `portfolio_loop.py`: adaptive multi-target runner (avoids single-target plateaus)
- `RESEARCH_POSEIDON_VULNERABILITIES.md`: current public cryptanalysis snapshot + source links
- `evidence_pack.py`: generate submission-grade evidence bundle from logs/artifacts
- `submission_pack.py`: generate canonical `agent.json`, `agent_log.json`, and receipts bundle
- `readiness_check.py`: submission-readiness checker (artifacts + activity + timeline gate)
- `config/targets.json`: benchmark targets and commands
- `config/track_b_attack_config.json`: immutable Track B base config (objective + canonical profiles)
- `config/track_b_mutable_*.json`: per-target mutable Track B search/analysis overlays
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
# Track B evaluation harness example
python3 prepare.py evaluate --target poseidon2_cryptanalysis_trackb_fast
# verify/export provenance for later attestation
python3 prepare.py provenance-verify
python3 prepare.py provenance-export --output work/provenance_export.json
```

## Verbose and Debug Runs

Use verbosity when you need to see real-time progress and command-level diagnostics:

```bash
# loop-level progress (iteration start/result) to stderr
python3 train.py --target leanmultisig_poseidon16_src_fast --iterations 12 -v

# include captured build/bench stdout+stderr for debugging (truncated)
python3 train.py --target leanmultisig_poseidon16_src_fast --iterations 4 -vv \
  --debug-command-output --debug-max-chars 8000

# same debugging flags are available in prepare.py
python3 prepare.py --verbose --debug-command-output evaluate --target leanmultisig_poseidon16_src_fast
```

## Sandbox Requirement

Command targets now require a sandbox by default. Set a sandbox wrapper before running command-backed evaluations or loops:

```bash
export AUTORESEARCH_SANDBOX_PREFIX="your-sandbox-wrapper ..."
python3 prepare.py evaluate --target poseidon2_cryptanalysis_trackb_kernel_fast
```

Per-target `require_sandbox: false` is still available as an explicit opt-out, but the checked-in default is fail-closed for command targets.

## Karpathy-Compatible Mode

The repo now supports the same top-level workflow shape:

- `prepare.py` (fixed)
- `train.py` (agent loop / editable)
- `program.md` (human instructions)

This keeps compatibility with prompts and workflows written for Karpathy-style autoresearch projects.
Use `--iterations 0` to run indefinitely until `--max-accepted` or `--max-runtime-seconds` stops the loop.
Use `--git-checkpoint-mode accepted` to create per-accept non-interactive git commits for source changes.

## End-To-End Campaign

To run a full track-aligned pipeline (autonomous loop + real Lean baselines + real-source optimization + evidence + submission/readiness artifacts):

```bash
python3 campaign.py --fresh --synthesis-cook --real-profile fast -v
```

Useful controls:

```bash
# add extra real-source optimization rounds
python3 campaign.py --synthesis-cook --real-optimize-rounds 4

# run cryptanalysis-only campaign lane
python3 campaign.py --track cryptanalysis --real-profile none --loop-iterations 25 --crypto-optimize-rounds 2

# run both lanes in one campaign
python3 campaign.py --track hybrid --synthesis-cook --crypto-optimize-rounds 2

# show the host-aware default source target set (ARM NEON vs x86 AVX2)
python3 campaign.py --help | rg real-optimize-targets -n
python3 campaign.py --help | rg crypto-optimize-targets -n

# run loop indefinitely (stopped by max accepted)
python3 campaign.py --synthesis-cook --loop-iterations 0 --max-accepted 3

# Karpathy-style accepted commits for loop target
python3 campaign.py --synthesis-cook --git-checkpoint-mode accepted --git-checkpoint-prefix autoresearch

# enforce identity fields for canonical submission artifacts
python3 campaign.py --synthesis-cook --strict-submission

# fill ERC-8004 identity + receipts in one pass (required for strict readiness)
python3 campaign.py --synthesis-cook --strict-submission \
  --operator-wallet 0xYOUR_OPERATOR_WALLET \
  --erc8004-identity your-erc8004-identity \
  --erc8004-registration-tx https://basescan.org/tx/0xYOUR_REGISTRATION_TX \
  --additional-receipt https://basescan.org/tx/0xYOUR_AGENT_ACTION_TX
```

Track requirement mapping is documented in `SYNTHESIS_ALIGNMENT.md`.

## Adaptive Portfolio Mode

For broader search across multiple Rust hotspots:

```bash
python3 portfolio_loop.py --rounds 4 --batch-iterations 6 --batch-max-accepted 1 --artifacts accepted
```

Set `--rounds 0` to run indefinitely until `--stop-after-total-accepted` is reached.

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
They can also enforce `require_metric_series_for_stats=true` so statistical gates cannot run with missing `debug.metric_values`.
They can also run post-accept A/B replay (`ab_repeats`) to confirm patched vs original.
Shared-source targets can add held-out verifier gates (`holdout_targets`) plus reward-audit lanes
(`reward_audit_targets`) that must remain non-regressed before a candidate is accepted and promoted.
Rejected mutations can be temporarily cooled down (`blocked_mutation_ttl`) to encourage broader exploration.
If the search stalls with `no_change`, the loop can release the oldest blocked mutation and retry selection in-place (`recover_from_no_change`, default `true`).
Mutation selection can use a UCB bandit scheduler (`mutation_schedule=ucb`) with configurable exploration strength (`mutation_ucb_explore`).
Sensitive crypto targets can enforce snippet-preservation guardrails (`required_snippets`) to reject candidates that drop critical permutation steps before benchmarking.
Source-level Poseidon targets can run workload profiles (`benchmark_profiles`) and aggregate them with `profiles_aggregate`
to avoid overfitting to one benchmark shape (for example, combining `--log-n-perms 10` and `11` in one accept/reject metric).
Optional runtime controls are supported during benchmark commands:

- `AUTORESEARCH_NICE=<level>` (process niceness)
- `AUTORESEARCH_CPU_AFFINITY=<cpu-list>` (uses `taskset -c` when available)

### Calibration Pass (Recommended)

Before long runs, measure target noise empirically:

```bash
python3 prepare.py calibrate --target leanmultisig_poseidon16_src_fast --samples 5
python3 prepare.py calibrate --target leanmultisig_poseidon16_table_src_fast --samples 5
python3 prepare.py calibrate --target leanmultisig_poseidon2_monty_core_src_fast --samples 5
python3 prepare.py calibrate --target leanmultisig_poseidon2_neon_src_fast --samples 5
python3 prepare.py calibrate --target leanmultisig_poseidon2_avx2_src_fast --samples 5
python3 prepare.py calibrate --target leanmultisig_poseidon2_no_packing_src_fast --samples 5
```

For direct Rust autoresearch against Lean source code:

```bash
python3 prepare.py baseline --target leanmultisig_poseidon16_src_fast --notes lean_src_baseline
python3 train.py --target leanmultisig_poseidon16_src_fast --iterations 5 --max-accepted 1 --artifacts accepted \
  --git-checkpoint-mode accepted --git-checkpoint-prefix autoresearch
```

Additional source-level targets are also available for deeper Poseidon2 hotspots:

- `leanmultisig_poseidon16_table_src_fast`
- `leanmultisig_poseidon2_monty_core_src_fast`
- `leanmultisig_poseidon2_neon_src_fast`
- `leanmultisig_poseidon2_avx2_src_fast`
- `leanmultisig_poseidon2_no_packing_src_fast` (wrapper fallback target)

Reference implementation context can be inspected in:

- `work/leanMultisig`
- `work/leanSig`

## Track B Cryptanalysis Targets

Track B uses command targets that optimize a deterministic `attack_score` extracted from JSON output.
The scoring objective stays in the immutable base config, while JSON mutation targets now edit per-target
`config/track_b_mutable_*.json` overlays that only carry `search` and `analysis` fields.
The harness runs reduced-round Poseidon2-style kernels over a prime field:

- differential bias search on truncated output deltas
- meet-in-the-middle truncated preimage search (prefix/suffix split with inversion)
- birthday-style truncated collision search
- algebraic elimination pressure lane over prefix rounds (rank/nullity/fit-derived complexity signal)

- `poseidon2_cryptanalysis_trackb_fast`
- `poseidon2_cryptanalysis_trackb_full`
- `poseidon2_cryptanalysis_trackb_verified_fast` (strict verified-hit lane)
- `poseidon2_cryptanalysis_algebraic_fast` (algebraic complexity lane)
- `poseidon2_cryptanalysis_poseidon64_signal_fast` (profile lane)
- `poseidon2_cryptanalysis_poseidon64_algebraic_fast` (profile algebraic lane)
- `poseidon2_cryptanalysis_poseidon256_signal_fast` (profile lane)
- `poseidon2_cryptanalysis_koalabear16_signal_fast` (profile lane)

Kernel-first Track B now uses a staged verifier stack:

- primary metric on `poseidon2_cryptanalysis_trackb_kernel_fast` or `..._kernel_signal_fast`
- same-source holdouts on verified-hit and algebraic lanes
- reward-audit lanes on `poseidon256_bounty_shape` and `koalabear_w16_shape`
- replay/population promotion only after the candidate survives all gates

Quick start:

```bash
python3 attack_harness.py --config config/track_b_attack_config.json --config-override config/track_b_mutable_fast.json --mode fast --output-format pretty
python3 attack_harness.py --config config/track_b_attack_config.json --config-override config/track_b_mutable_poseidon64_signal_fast.json --profile poseidon64_bounty_shape --mode fast --output-format pretty
python3 prepare.py baseline --target poseidon2_cryptanalysis_trackb_fast --notes trackb_baseline
python3 train.py --target poseidon2_cryptanalysis_trackb_fast --iterations 12 --max-accepted 2 -v
python3 train.py --target poseidon2_cryptanalysis_trackb_verified_fast --iterations 12 --max-accepted 1 -v
python3 prepare.py evaluate --target poseidon2_cryptanalysis_algebraic_fast
python3 prepare.py evaluate --target poseidon2_cryptanalysis_poseidon64_algebraic_fast
python3 train.py --target poseidon2_cryptanalysis_algebraic_fast --iterations 12 --max-accepted 2 -v
```

## LLM Mode (Optional)

`train.py` defaults to deterministic heuristics. To enable model-proposed patches with the OpenAI API:

```bash
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-5-mini
python3 train.py --target cairo_poseidon_style_t8 --iterations 25
```

To use the logged-in Codex desktop/CLI session instead of an API key:

```bash
codex login status
python3 train.py --target poseidon2_cryptanalysis_trackb_kernel_fast --iterations 12 --llm-backend codex --model gpt-5-codex
```

Controls:

- `--llm-backend auto|heuristic|openai|codex`
- `--model ...` for the selected backend
- `--codex-reasoning-effort low|medium|high` for Codex CLI mode
- `--codex-timeout-seconds 180` to adjust a single Codex proposal budget
- `AUTORESEARCH_CODEX_BIN=/path/to/codex` to override the CLI path
- `AUTORESEARCH_CODEX_TIMEOUT_SECONDS=180` to raise/lower the Codex request timeout

For guarded Python targets such as `attack_kernels.py`, Codex mode uses structured function-level edits in an isolated temporary workspace instead of handing the whole repo to the agent. This reduces prompt surface, avoids repository exploration during proposal generation, biases edits toward the attack kernels instead of `score()`, and skips purely structural/refactor-only patches before benchmark spend.

Kernel-first Track B targets add three extra search policies on top of that:

- adaptive timeout retry: keep the normal Codex budget first, then retry once at a larger timeout for `attack_kernels.py` edits if the first call times out
- single-kernel focus: present one attack family per Codex request, with only the relevant helper summaries, instead of prompting over the entire kernel module
- family scheduler + novelty filter: allocate Codex attempts across `differential`, `mitm`, `collision`, and `algebraic` kernels using bandit-style scoring, and reject duplicate semantic signatures before benchmark spend

The primary kernel targets expose these knobs directly in [`config/targets.json`](config/targets.json):

- `codex_timeout_retry_count`
- `codex_timeout_retry_seconds`
- `codex_kernel_family_explore`
- `codex_kernel_family_novelty_weight`
- `codex_kernel_family_timeout_penalty`

Cross-run family scheduler state is persisted in `work/codex_focus_stats.json`, so short Codex-backed runs can accumulate timeout/reward/duplicate statistics instead of restarting the bandit from scratch every time.

The loop falls back to heuristics automatically if the OpenAI or Codex request path fails.
The Codex CLI path uses tagged flat-text sections because `codex exec` does not expose a separate system-role channel; treat prompt-injection protections on that path as weaker than the OpenAI API path and prefer pinned model IDs for reproducible experiments.

## Cross-Target Mutation Replay

Rust and Track B JSON targets share accepted mutation history through persisted replay memory:

- default file: `work/mutation_memory.json`
- memory is seeded from historical `results.tsv` accepted/rejected rows
- successful mutations on one target are prioritized on sibling targets of the same language

Controls:

```bash
# default behavior (memory enabled)
python3 train.py --target leanmultisig_poseidon16_src_fast --iterations 12

# custom memory file
python3 train.py --target leanmultisig_poseidon2_neon_src_fast --mutation-memory-file work/custom_memory.json

# Track B JSON lane with operator-level UCB + replay
python3 train.py --target poseidon2_cryptanalysis_algebraic_fast --iterations 12 -v

# disable replay memory
python3 train.py --target leanmultisig_poseidon2_neon_src_fast --disable-mutation-memory
```

## Provenance Export And Attestation

Local runs maintain a verifiable hash chain in `provenance_chain.jsonl` plus `work/provenance_state.json`.
You can verify and export a manifest covering the chain head and key outputs:

```bash
python3 prepare.py provenance-verify
python3 prepare.py provenance-export --output work/provenance_export.json
```

For stronger off-box attestation, run the GitHub Actions workflow in
`.github/workflows/provenance-attestation.yml`, which exports the manifest and signs it with GitHub Artifact Attestations.

## Acceptance Policy

A candidate is accepted only when:

- evaluation succeeds (`scarb build` + `scarb test` for Cairo targets, or the equivalent target-specific checks)
- metric strictly improves relative to best-so-far
- optional noise guards pass (`max_rel_stdev`, distribution/effect-size gate for command targets)
- optional confirmation runs (`confirm_repeats`) still beat the current best
- optional shared-source validation gates pass (`validation_targets`)
- optional holdout verifier gates pass (`holdout_targets`)
- optional reward-audit lanes pass (`reward_audit_targets`)

Otherwise the file is reverted.

## Notes

- The included Cairo/Noir targets are optimization sandboxes, not audited production cryptography.
- For production or research publication, replace sandbox targets with audited production implementations.
