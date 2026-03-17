# Track B Kernel Signal Lane

Date: 2026-03-17

## Hypothesis

The default cryptanalysis portfolio should expose only one poseidon64 signal evaluator arm. Replacing the config-mutation default arm with a kernel-first arm keeps the same benchmark signal while constraining edits to `attack_harness.py`, where `required_snippets` can guard the kernel entrypoints.

## Methodology

Commands run from repository root:

```bash
python3 prepare.py evaluate --target poseidon2_cryptanalysis_trackb_kernel_signal_fast
python3 prepare.py evaluate --target poseidon2_cryptanalysis_poseidon64_signal_fast
```

Configuration changes in this PR:

- keep `poseidon2_cryptanalysis_trackb_kernel_signal_fast` in `campaign.default_crypto_optimize_targets()`
- remove `poseidon2_cryptanalysis_poseidon64_signal_fast` from the default cryptanalysis portfolio
- set `poseidon2_cryptanalysis_trackb_kernel_signal_fast.runs` to `1` because fast mode is deterministic
- add target-scoped mutation-memory namespaces for the `research_fast` and `poseidon64_bounty_shape` kernel lanes

## Results

- `poseidon2_cryptanalysis_trackb_kernel_signal_fast`
  - benchmark command: `python3 attack_harness.py --config config/track_b_attack_config.json --profile poseidon64_bounty_shape --mode fast --output-format json`
  - runs: `1`
  - metric: `attack_score_signal = 17.086093854526005`
  - series: `[17.086093854526005]`
- `poseidon2_cryptanalysis_poseidon64_signal_fast`
  - benchmark command: `python3 attack_harness.py --config config/track_b_attack_config.json --profile poseidon64_bounty_shape --mode fast --output-format json`
  - runs: `2`
  - metric: `attack_score_signal = 17.086093854526005`
  - series: `[17.086093854526005, 17.086093854526005]`

The evaluator command and observed metric are identical. The portfolio distinction is therefore the mutation surface, not a different scoring function.

## Determinism And Replay Scope

`attack_harness.py` fast mode is deterministic for this target shape because the evaluation uses a fixed config seed (`search.seed = 1337`) and fixed search budgets, so repeated runs return the same metric. One run is therefore sufficient for this lane.

The train-loop UCB scheduler does not estimate empirical benchmark variance; `mutation_ucb_score` uses accepted/rejected mutation history counts plus an exploration bonus. Deterministic benchmark output therefore does not create a narrower score-variance term inside the scheduler.

Replay history is isolated between the two kernel arms. This PR adds distinct `mutation_memory_namespace` values, and `train.py` now treats namespaced targets as strict target scopes when reading preferred mutations and UCB history. A mutation accepted on `poseidon2_cryptanalysis_trackb_kernel_fast` is not replayed into `poseidon2_cryptanalysis_trackb_kernel_signal_fast` unless it has target-local history for that namespace.

## Decision

Accept the kernel-first signal lane, but keep only that lane in the default cryptanalysis portfolio. Leave the config-mutation poseidon64 signal target available for explicit runs, not as a competing default UCB arm.
