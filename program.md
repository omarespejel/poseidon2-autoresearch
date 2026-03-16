# AutoPoseidon Program Instructions

You are the optimization agent for cryptographic circuits.

## Objective
Minimize the configured metric (for example `sierra_program_len` on Cairo targets) while preserving functional behavior.

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

## Output Discipline
- Every iteration must append to `results.tsv` and `agent_log.jsonl`.
- Keep changes minimal and attributable.
- Prefer small deterministic wins over risky broad rewrites.
