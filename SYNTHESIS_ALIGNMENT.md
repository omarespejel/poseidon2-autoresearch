# Synthesis Alignment (Track A)

Last verified: 2026-03-16

Primary source for required capabilities:
- https://synthesis.devfolio.co/catalog?track=cook&limit=100

## Requirement Mapping

| Required capability | Current implementation |
|---|---|
| Autonomous execution loop (discover -> plan -> execute -> verify -> submit) | `prepare.py` + `train.py` + `submission_pack.py` stage derivation (`stages` in `submission/agent_log.json`) |
| Agent identity (ERC-8004 operator + registration tx) | `submission_pack.py` fields: `operator_wallet`, `erc8004_identity`, `erc8004_registration_tx` |
| Agent capability manifest (`agent.json`) | `submission_pack.py` writes `submission/agent.json` with tools, stack, compute constraints, task categories |
| Structured execution logs (`agent_log.json`) | `submission_pack.py` writes `submission/agent_log.json` with decisions, retries, failures, tool calls, outputs |
| Tool use and orchestration | `train.py`, `prepare.py`, `portfolio_loop.py`, `campaign.py` (multi-step orchestration) |
| Safety guardrails | strict improvement gate, evaluation success requirement, optional variance/distribution/A-B confirmations, auto-revert |
| Compute budget awareness | `train.py` budgets (`max_iterations`, `max_accepted`, `max_runtime_seconds`) + submission budget accounting |

## Real Poseidon2 Improvement Track

Active targets for practical performance improvements:
- `leanmultisig_poseidon16_src_fast`
- `leanmultisig_poseidon16_table_src_fast`
- `leanmultisig_poseidon2_monty_core_src_fast`
- `leanmultisig_poseidon2_neon_src_fast`

These target Lean/Poseidon2 Rust source hotspots and optimize `poseidons_per_s` with repeated benchmark gates.

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
