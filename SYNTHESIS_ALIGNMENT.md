# Synthesis Alignment (Track A)

Last verified: 2026-03-16

Primary source for required capabilities:
- https://synthesis.devfolio.co/catalog?track=cook&limit=100
- https://synthesis.devfolio.co/catalog?track=pl-genesis-agents-with-receipts-8004&limit=100

Supporting research sources:
- https://github.com/karpathy/autoresearch/blob/master/program.md
- https://github.com/leanEthereum/leanMultisig
- https://github.com/Plonky3/Plonky3
- https://www.poseidon-initiative.info/

## Requirement Mapping

| Required capability | Current implementation |
|---|---|
| Autonomous execution loop (discover -> plan -> execute -> verify -> submit) | `prepare.py` + `train.py` + `submission_pack.py` stage derivation (`stages` in `submission/agent_log.json`) |
| Agent identity (ERC-8004 operator + registration tx) | `submission_pack.py` fields: `operator_wallet`, `erc8004_identity`, `erc8004_registration_tx` |
| Agent capability manifest (`agent.json`) | `submission_pack.py` writes `submission/agent.json` with tools, stack, compute constraints, task categories |
| Structured execution logs (`agent_log.json`) | `submission_pack.py` writes `submission/agent_log.json` with decisions, retries, failures, tool calls, outputs |
| Tool use and orchestration | `train.py`, `prepare.py`, `portfolio_loop.py`, `campaign.py` (multi-step orchestration) |
| Safety guardrails | strict improvement gate, evaluation success requirement, optional variance/distribution/A-B confirmations, required-snippet preservation on sensitive Poseidon targets, auto-revert |
| Verification rigor on real workloads | command-target workload profiles (`benchmark_profiles`) aggregate multiple Poseidon throughput shapes into one acceptance metric (`profiles_aggregate`) |
| Compute budget awareness | `train.py` budgets (`max_iterations`, `max_accepted`, `max_runtime_seconds`) + submission budget accounting |
| Karpathy-style iterative evidence | `train.py --iterations 0` infinite mode + optional `--git-checkpoint-mode accepted` per-accept source commit trail |

## Track Fit

- Primary competitive track: `Let the Agent Cook` (autonomy, multi-tool orchestration, safety, compute budgeting).
- Secondary additive fit: `Agents With Receipts — ERC-8004` (same artifacts plus onchain verifiability depth).
- Optional upside: `Synthesis Open Track` for broader judge preference beyond sponsor-specific criteria.

## Real Poseidon2 Improvement Track

Active targets for practical performance improvements:
- `leanmultisig_poseidon16_src_fast`
- `leanmultisig_poseidon16_table_src_fast`
- `leanmultisig_poseidon2_monty_core_src_fast`
- `leanmultisig_poseidon2_neon_src_fast`

These target Lean/Poseidon2 Rust source hotspots and optimize `poseidons_per_s` with repeated benchmark gates.

Current active-code signal (snapshot 2026-03-16 UTC):
- `leanEthereum/leanMultisig`: pushed `2026-03-15T22:19:57Z` (latest commit `6f06c02182`).
- `Plonky3/Plonky3`: pushed `2026-03-16T14:49:49Z` (latest commit `b4dcde4694`).

Security context:
- EF-backed Poseidon Cryptanalysis Initiative rewards reduced-round attack progress, which is complementary to this repo's performance optimization lane and should be treated as a separate Track B research direction.

## Recommended Run

```bash
python3 campaign.py --fresh --synthesis-cook --real-profile fast --real-optimize-rounds 2 -v
```

Outputs:
- `campaign_report.md`
- `evidence/manifest.json`
- `submission/agent.json`
- `submission/agent_log.json`
- `submission/submission_receipts.json`
- `readiness_report.md`
