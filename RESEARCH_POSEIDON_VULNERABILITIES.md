# Poseidon / Poseidon2 Vulnerability Research Snapshot

Date: 2026-03-17 (UTC)

This note tracks the current public attack surface on reduced-round Poseidon-family permutations and how it maps to Track B.

## 1) Program Status (What Is Actually Rewarded)

- Ethereum's historical zk-hash bounty (`crypto.ethereum.org/bounties/zk-hash`) is archived and marked **ended on November 15, 2022**. It describes collision/preimage attack objectives for zk-friendly hashes (including Poseidon), but is no longer an active deadline program.
  - Source: <https://crypto.ethereum.org/bounties/zk-hash>
- The active Poseidon cryptanalysis work is organized through the Poseidon Initiative, with explicit reduced-round challenges and claimed/unclaimed entries listed publicly.
  - Source: <https://www.poseidon-initiative.info/>

## 2) Current Public Cryptanalysis Direction

The strongest public work keeps improving attacks on **reduced-round** variants rather than full-round standard parameters.

- Poseidon2 paper (2023): introduces Poseidon2 and states known attacks can be countered by conservative round choices.
  - Source: <https://eprint.iacr.org/2023/323>
- Cheap/Free-Lunch style attacks (AOH family, including Poseidon in reduced settings) showed new linear-system style attack strategies.
  - Sources:
    - <https://eprint.iacr.org/2024/310>
    - IACR summary: <https://iacr.org/news/index.php?next=21681>
- Recent Poseidon-focused updates include:
  - differential attacks on arithmetization-oriented hashes
  - advanced interpolation-style attacks on reduced rounds
  - subspace-trail investigations on Poseidon2
  - Sources:
    - <https://eprint.iacr.org/2025/950>
    - <https://eprint.iacr.org/2025/937>
    - <https://eprint.iacr.org/2025/954>
- Earlier baseline algebraic cryptanalysis reference:
  - Source: <https://tosc.iacr.org/index.php/ToSC/article/view/9709>

## 3) Practical Takeaways for This Repo

- Full-round production parameters remain unbroken publicly; attack progress is concentrated in reduced-round zones and specific parameter regimes.
- The useful engineering target is not "break full Poseidon2 now", but:
  - rapidly stress reduced-round profiles,
  - quantify attack signal movement under parameter and budget changes,
  - produce reproducible evidence when a verified hit appears.

## 4) Mapping to Implemented Track B Profiles

Track B now includes profile-backed shapes in `config/track_b_attack_config.json`:

- `research_fast` (default): deterministic, fast iteration profile for autoresearch loops.
- `poseidon64_bounty_shape`: reduced-round shape aligned to public Poseidon-64-style bounty context.
- `poseidon256_bounty_shape`: reduced-round shape aligned to public Poseidon-256-style bounty context.
- `koalabear_w16_shape`: wider-state profile for Lean/KoalaBear-style stress shape.

These profiles are intentionally attack-search harness profiles, not claims of exact full canonical parameter replication.

## 5) Metric Lanes (Why Two Scores)

Track B now exposes two optimization lanes:

- `attack_score_signal`: rewards strong reduced-round signal movement (useful for broad exploration and autonomous-search throughput).
- `attack_score_verified`: rewards concrete verified-hit events (stricter lane for evidence quality).

Both are emitted by `attack_harness.py` under `metrics.*` and can be selected via `metric_json_path` targets.

## 6) Suggested Ongoing Research Discipline

- Re-check sources weekly (papers and challenge status change quickly).
- Keep profile configs versioned and dated.
- Distinguish clearly in reports:
  - heuristic signal gains,
  - verified reduced-round hits,
  - any claim about full-round security (should require external review).
