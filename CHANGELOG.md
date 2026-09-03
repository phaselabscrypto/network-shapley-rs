# Changelog

All notable changes to this fork are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [phase-2026.09] - 2026-09-02

- `#1`: Replaces the hand-written simplex with HiGHS (`highs = "2.1"`) and removes `microlp`. Adds Monte Carlo permutation sampling through `compute_sampled`, cancellation and progress counters through `ComputeControl`, and seed-cache (B3) reuse.
- `#2`: Adds progress counters to the exact compute path. Adds `coalition_count()`.
- `#3`: Warm-starts the exact coalition solve through a crate-private `WarmCoalitionSolver`, about 3 times faster than the prior cold solve.
- `#4`: Adds `network_link_estimate` for a per-link Shapley estimate. Each focus-owned link becomes its own player, every other operator collapses to `Others`, uptime is forced to 1.0, and the solve runs once over `2^(links+1)` coalitions.
- `#5`: Consolidates the API onto `ComputeOptions`, adds a constants module, and adds a sparse-matrix helper.
- `#6`: Exempts link estimation from the operator cap. Adds `MAX_LINK_PLAYERS = 31` (a `u32` mask with bit 31 reserved) and `MAX_OPERATORS_PARTIAL_UPTIME = 15`.
- `#7`: Adds a cold retry for when HiGHS returns `Unknown` on a warm solve.

### Behaviour notes

- HiGHS computes the results. Upstream `v0.6.0` used a hand-written simplex.
- A warm-started solve is stable to floating-point rounding but is not bit-identical run to run (see the comment near the top of `src/shapley.rs`).
- The operator and link-player limits, and the default LP time limit, live in crate-private constants: `MAX_OPERATORS` (20), `MAX_OPERATORS_PARTIAL_UPTIME` (15), `MAX_LINK_PLAYERS` (31), `DEFAULT_LP_TIME_LIMIT_SECS` (60 seconds). Code outside the crate reaches them only through a `TooManyOperators` error, an `LpSolver` error, or the `SHAPLEY_LP_TIME_LIMIT_SECS` environment variable.
