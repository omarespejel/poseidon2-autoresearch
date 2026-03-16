# Poseidon2 Autoresearch — Review Rules

## Cryptographic Correctness

- All changes to Poseidon2 parameters (round constants, MDS matrices, S-box definitions) must reference the original paper or a peer-reviewed specification.
- Hash function implementations must be constant-time where applicable to prevent timing side-channels.
- Any new permutation variant must include test vectors validated against a reference implementation.

## Rust Code Quality

- No `.unwrap()` or `.expect()` in non-test code. Use `?` operator or explicit error handling.
- All `unsafe` blocks require a `// SAFETY:` comment documenting the invariants being upheld.
- Public APIs must have doc comments with examples where practical.
- Prefer `#[must_use]` on functions that return values that should not be silently discarded.
- Use `clippy::pedantic` level compliance.

## Performance

- Avoid unnecessary heap allocations in hot paths (hash computation, permutation rounds).
- Prefer stack-allocated arrays over `Vec` for fixed-size data.
- Clone only when ownership transfer is not possible — explain why in a comment if cloning in performance-critical code.
- Benchmark any optimization claim. No performance PR is accepted without before/after numbers.

## Experiment Discipline

- Every optimization experiment must be logged with: hypothesis, methodology, results, and accept/revert decision.
- Experiment traces must be reproducible — pin all dependencies, document the hardware/OS, and seed any RNG.
- Reverted experiments should remain documented (not deleted) so the team can learn from them.

## Testing

- All new functionality requires unit tests covering the happy path and at least one error case.
- Cryptographic primitives require property-based tests (e.g., round-trip, collision resistance spot checks).
- Edge cases: empty input, maximum-length input, and malformed input must be tested.

## Dependencies

- New dependencies must be justified. Prefer the standard library or well-audited crates.
- Pin dependency versions in `Cargo.toml`. No wildcard versions.
- `Cargo.lock` must be committed for binary/application targets.
