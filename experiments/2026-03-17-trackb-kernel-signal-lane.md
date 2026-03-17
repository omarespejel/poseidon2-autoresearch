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
- raise `poseidon2_cryptanalysis_trackb_kernel_signal_fast.runs` from `2` to `3` so `aggregate: "median"` is a real median

## Results

- `poseidon2_cryptanalysis_trackb_kernel_signal_fast`
  - benchmark command: `python3 attack_harness.py --config config/track_b_attack_config.json --profile poseidon64_bounty_shape --mode fast --output-format json`
  - runs: `3`
  - metric: `attack_score_signal = 17.086093854526005`
  - series: `[17.086093854526005, 17.086093854526005, 17.086093854526005]`
- `poseidon2_cryptanalysis_poseidon64_signal_fast`
  - benchmark command: `python3 attack_harness.py --config config/track_b_attack_config.json --profile poseidon64_bounty_shape --mode fast --output-format json`
  - runs: `2`
  - metric: `attack_score_signal = 17.086093854526005`
  - series: `[17.086093854526005, 17.086093854526005]`

The evaluator command and observed metric are identical. The portfolio distinction is therefore the mutation surface, not a different scoring function.

## Decision

Accept the kernel-first signal lane, but keep only that lane in the default cryptanalysis portfolio. Leave the config-mutation poseidon64 signal target available for explicit runs, not as a competing default UCB arm.
