# Research Brief (2026-03-16)

## 1) What Synthesis Actually Scores (verified)

Primary track requirements from the live catalog:

- Let the Agent Cook requires autonomous loop (discover -> plan -> execute -> verify -> submit), ERC-8004 identity + registration tx, machine-readable `agent.json`, structured `agent_log.json`, real tool use, safety guardrails, and compute-budget awareness.
- Agents With Receipts requires ERC-8004 onchain verifiability and DevSpot-compatible `agent.json` / `agent_log.json`.

Sources:
- https://synthesis.devfolio.co/catalog?track=cook&limit=100
- https://synthesis.devfolio.co/catalog?track=receipts&limit=100
- https://synthesis.devfolio.co/skill.md

## 2) Karpathy Autoresearch Constraints (verified)

The loop philosophy to preserve:

- fixed evaluator
- editable optimization loop
- accept/revert discipline
- continuous operation ("never stop" style)

Source:
- https://github.com/karpathy/autoresearch/blob/master/program.md

## 3) Practical SOTA Poseidon Targets (verified)

For real performance impact, the highest-value open source paths are Rust proving stacks (not only sandbox circuits):

- Lean multisig prover throughput (`poseidons_per_s`, `xmss_per_s`)
  - https://github.com/leanEthereum/leanMultisig
- Poseidon2 proving objective in Plonky3 on KoalaBear
  - https://github.com/Plonky3/Plonky3
- Noir Poseidon/Poseidon2 constraint baselines (useful sandbox baseline, not final bottleneck)
  - https://github.com/TaceoLabs/noir-poseidon

## 4) Decision

Track A execution remains:

- Keep Synthesis deliverables first-class (`agent.json`, `agent_log.json`, readiness/evidence packs).
- Run optimization directly on Lean Poseidon source targets to pursue practical throughput gains.
- Keep Noir/Cairo as fast fallback sandboxes only.

## 5) Changes Landed In This PR Branch

- Operator-level UCB mutation selection added for Rust loop.
- Mutation memory now records objective gain (positive = better) even for non-accepted candidates, so ranking is not binary-only.
- Real Lean source targets configured to use UCB mutation selection.

This increases probability of finding real accepted improvements while preserving guardrails required by Synthesis scoring.
