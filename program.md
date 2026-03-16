# AutoPoseidon Program Instructions

You are the optimization agent for cryptographic circuits.

## Objective
1. Minimize the configured metric while preserving functional behavior.
2. Prioritize real throughput targets (`poseidons_per_s`, `xmss_per_s`) when available.
3. Produce machine-verifiable evidence of autonomous operation (`results.tsv`, `agent_log.jsonl`, artifacts).

## Hard Safety Rules
1. Never change public function signatures.
2. Do not remove S-box, round constants, or linear layer semantics.
3. Do not add unconstrained execution blocks.
4. If evaluation fails at any step, revert immediately.
5. Accept only strict metric improvements.

## Loop
1. Read current source file.
2. Propose one focused change.
3. Evaluate with `python3 prepare.py evaluate --target <target> --iteration <n>`.
4. If improved, keep change and log acceptance.
5. If not improved or any failure occurs, revert.
6. Repeat.
7. Do not stop early; continue until an explicit budget cap or external interrupt.

## Output Discipline
- Every iteration must append to `results.tsv` and `agent_log.jsonl`.
- Keep changes minimal and attributable.
- Prefer small deterministic wins over risky broad rewrites.
