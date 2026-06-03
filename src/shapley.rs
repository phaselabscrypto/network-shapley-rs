use std::{
    cell::RefCell,
    collections::{BTreeMap, HashMap, HashSet},
    fmt::{Display, Formatter},
    sync::Arc,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};

use rand::Rng;
use rand::seq::SliceRandom;
use rayon::prelude::*;
#[cfg(feature = "serde")]
use {
    serde::{Deserialize, Serialize},
    tabled::Tabled,
};

use crate::{
    consolidation::{consolidate_demand, consolidate_links},
    error::{Result, ShapleyError},
    lp_builder::LpBuilderInput,
    solver::{CoalitionBuffers, PrecomputedRows, SolveStatus, solve_coalition},
    types::{Demands, Devices, PrivateLinks, PublicLinks},
    utils::factorial,
    validation::check_inputs,
};

/// Sentinel bit for operators that are always included in every coalition
/// (Public, Private, empty). Set in bit 31 so it never collides with
/// operator index bits 0..19.
const ALWAYS_BIT: u32 = 1 << 31;

// For clarity
pub type Operator = String;

// Since shapley value is per operator, we just use a hashmap
pub type ShapleyOutput = BTreeMap<Operator, ShapleyValue>;

/// Input parameters for Shapley computation
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
pub struct ShapleyInput {
    pub private_links: PrivateLinks,
    pub devices: Devices,
    pub demands: Demands,
    pub public_links: PublicLinks,
    pub operator_uptime: f64,
    pub contiguity_bonus: f64,
    pub demand_multiplier: f64,
}

impl ShapleyInput {
    pub fn compute(&self) -> Result<ShapleyOutput> {
        let shapley = Shapley::new(
            self.private_links.clone(),
            self.devices.clone(),
            self.demands.clone(),
            self.public_links.clone(),
            self.operator_uptime,
            self.contiguity_bonus,
            self.demand_multiplier,
        );

        let output = shapley.compute()?;
        Ok(output)
    }

    /// Approximate Shapley values via adaptive Monte Carlo permutation sampling.
    ///
    /// Instead of evaluating all 2^N coalitions, samples random permutations
    /// of operators and averages marginal contributions. Uses a three-pass
    /// approach: (1) generate permutations and collect needed coalition masks,
    /// (2) solve all unique coalitions in parallel with rayon, (3) compute
    /// marginal contributions from cached values.
    ///
    /// Adaptive: runs `config.min_samples` first, then adds batches of
    /// `config.batch_size` until all operators converge (relative SE ≤
    /// `config.target_se`) or `config.max_samples` is reached.
    pub fn compute_sampled(&self, config: SamplingConfig) -> Result<SampledOutput> {
        let shapley = Shapley::new(
            self.private_links.clone(),
            self.devices.clone(),
            self.demands.clone(),
            self.public_links.clone(),
            self.operator_uptime,
            self.contiguity_bonus,
            self.demand_multiplier,
        );
        shapley.compute_sampled(config)
    }

    /// Compute sampled Shapley values, pre-populating the coalition cache.
    ///
    /// Coalitions already present in `existing_cache` won't be re-solved.
    /// This is the key optimisation for simulation workflows: reuse
    /// baseline coalition values and only solve coalitions affected by
    /// topology changes.
    pub fn compute_sampled_with_cache(
        &self,
        config: SamplingConfig,
        existing_cache: HashMap<u32, Option<f64>>,
    ) -> Result<SampledOutput> {
        let shapley = Shapley::new(
            self.private_links.clone(),
            self.devices.clone(),
            self.demands.clone(),
            self.public_links.clone(),
            self.operator_uptime,
            self.contiguity_bonus,
            self.demand_multiplier,
        );
        shapley.compute_sampled_with_cache(config, existing_cache)
    }

    /// Like [`Self::compute_sampled`] but cooperatively cancellable, and
    /// reports live progress via `control` (drives a progress bar; set
    /// `control.cancel` to stop early — returns [`ShapleyError::Cancelled`]).
    pub fn compute_sampled_cancellable(
        &self,
        config: SamplingConfig,
        control: &ComputeControl,
    ) -> Result<SampledOutput> {
        let shapley = Shapley::new(
            self.private_links.clone(),
            self.devices.clone(),
            self.demands.clone(),
            self.public_links.clone(),
            self.operator_uptime,
            self.contiguity_bonus,
            self.demand_multiplier,
        );
        shapley.compute_sampled_cancellable(config, control)
    }

    /// Like [`Self::compute_sampled_with_cache`] but cooperatively cancellable
    /// with live progress via `control`.
    pub fn compute_sampled_with_cache_cancellable(
        &self,
        config: SamplingConfig,
        existing_cache: HashMap<u32, Option<f64>>,
        control: &ComputeControl,
    ) -> Result<SampledOutput> {
        let shapley = Shapley::new(
            self.private_links.clone(),
            self.devices.clone(),
            self.demands.clone(),
            self.public_links.clone(),
            self.operator_uptime,
            self.contiguity_bonus,
            self.demand_multiplier,
        );
        shapley.compute_sampled_with_cache_cancellable(config, existing_cache, control)
    }

    /// Sampled Shapley with SOUND topology-aware reuse (B3). Seeds the coalition
    /// cache from a baseline run on a *different* topology, but reuses a seeded
    /// value only for coalitions that exclude every operator in
    /// `changed_operators` (the operators owning a primitive that differs from
    /// the seed's topology — the caller derives these from changed links'/devices'
    /// endpoint operators). Coalitions touching a changed operator are re-solved,
    /// so the result is identical to a full fresh recompute.
    ///
    /// SAFETY (caller-enforced): only call when the operator SET, `public_links`,
    /// `demands`, `demand_multiplier`, `contiguity_bonus`, and `operator_uptime`
    /// are unchanged vs the seed's topology — otherwise fall back to
    /// [`Self::compute_sampled`]. Pass `changed_operators == []` only when the
    /// topology is identical (equivalent to [`Self::compute_sampled_with_cache`]).
    pub fn compute_sampled_with_reuse(
        &self,
        config: SamplingConfig,
        seed_cache: HashMap<u32, Option<f64>>,
        changed_operators: Vec<String>,
    ) -> Result<SampledOutput> {
        let shapley = Shapley::new(
            self.private_links.clone(),
            self.devices.clone(),
            self.demands.clone(),
            self.public_links.clone(),
            self.operator_uptime,
            self.contiguity_bonus,
            self.demand_multiplier,
        );
        shapley.compute_sampled_with_reuse(config, seed_cache, &changed_operators)
    }

    /// Like [`Self::compute_sampled_with_reuse`] but cancellable with live
    /// progress via `control`.
    pub fn compute_sampled_with_reuse_cancellable(
        &self,
        config: SamplingConfig,
        seed_cache: HashMap<u32, Option<f64>>,
        changed_operators: Vec<String>,
        control: &ComputeControl,
    ) -> Result<SampledOutput> {
        let shapley = Shapley::new(
            self.private_links.clone(),
            self.devices.clone(),
            self.demands.clone(),
            self.public_links.clone(),
            self.operator_uptime,
            self.contiguity_bonus,
            self.demand_multiplier,
        );
        shapley.compute_sampled_with_reuse_cancellable(
            config,
            seed_cache,
            &changed_operators,
            control,
        )
    }

    /// Exact Shapley with SOUND topology-aware reuse (B3, small-N path). Same
    /// reuse rule as [`Self::compute_sampled_with_reuse`] but for the exact
    /// `compute()` enumeration: a coalition's value is taken from `seed_cache`
    /// when the coalition excludes every changed operator, else solved fresh.
    /// Byte-identical to [`Self::compute`] on the modified topology.
    pub fn compute_with_reuse(
        &self,
        seed_cache: HashMap<u32, Option<f64>>,
        changed_operators: Vec<String>,
    ) -> Result<ShapleyOutput> {
        let shapley = Shapley::new(
            self.private_links.clone(),
            self.devices.clone(),
            self.demands.clone(),
            self.public_links.clone(),
            self.operator_uptime,
            self.contiguity_bonus,
            self.demand_multiplier,
        );
        shapley.compute_inner(seed_cache, &changed_operators)
    }
}

/// Individual Shapley value for an operator
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize, Tabled))]
#[derive(Debug, Clone, PartialEq)]
pub struct ShapleyValue {
    pub value: f64,
    pub proportion: f64,
}

impl Display for ShapleyValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "value: {}, proportion: {}", self.value, self.proportion)
    }
}

/// Configuration for Monte Carlo permutation sampling.
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Minimum permutations before checking convergence.
    pub min_samples: usize,
    /// Maximum permutations (hard cap).
    pub max_samples: usize,
    /// Target relative standard error per operator.
    pub target_se: f64,
    /// Batch size for adaptive rounds after initial min_samples.
    pub batch_size: usize,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            min_samples: 100,
            max_samples: 500,
            target_se: 0.05,
            batch_size: 50,
        }
    }
}

impl SamplingConfig {
    /// Create config tuned for the given problem size.
    /// Reduces sample count for expensive LPs (high demand count).
    pub fn for_problem(_n_operators: usize, n_demands: usize) -> Self {
        let (base, cap) = if n_demands > 2000 {
            (40, 150)
        } else {
            (50, 300)
        };
        Self {
            min_samples: base,
            max_samples: cap,
            ..Default::default()
        }
    }

    /// Relaxed config for what-if simulations where directional accuracy
    /// matters more than precision. Uses 10% target SE (vs 5% default)
    /// and caps at 200 samples, yielding ~2–3× faster convergence.
    pub fn for_simulation(_n_operators: usize, n_demands: usize) -> Self {
        let (base, cap) = if n_demands > 2000 {
            (30, 120)
        } else {
            (40, 200)
        };
        Self {
            min_samples: base,
            max_samples: cap,
            target_se: 0.10,
            batch_size: 50,
        }
    }
}

/// Result of sampled Shapley computation with convergence diagnostics.
#[derive(Debug)]
pub struct SampledOutput {
    /// Shapley values in the same format as exact `compute()`.
    pub values: ShapleyOutput,
    /// Number of permutation samples actually used.
    pub samples_used: usize,
    /// Per-operator standard error of the mean.
    pub standard_errors: BTreeMap<String, f64>,
    /// Whether all operators converged within `target_se`.
    pub converged: bool,
    /// Coalition value cache: maps coalition bitmask → LP objective value.
    /// Can be reused across computations with the same network topology.
    pub coalition_cache: HashMap<u32, Option<f64>>,
    /// How many seeded coalition values were admitted + reused this run (the
    /// gate-passing subset of a `compute_sampled_with_reuse` seed). 0 for a cold
    /// run. `coalition_cache.len() - coalitions_reused` were solved fresh.
    pub coalitions_reused: usize,
}

/// Cooperative cancellation + progress for a sampled compute.
///
/// Pass a shared `ComputeControl` into
/// [`ShapleyInput::compute_sampled_cancellable`]: set `cancel` from another
/// thread to stop early (the call returns [`ShapleyError::Cancelled`]), and
/// read `progress` to drive a progress bar.
#[derive(Debug, Default, Clone)]
pub struct ComputeControl {
    pub cancel: Arc<AtomicBool>,
    pub progress: Arc<ComputeProgress>,
}

/// Live progress counters for a sampled compute (monotonically increasing).
#[derive(Debug, Default)]
pub struct ComputeProgress {
    /// Coalition LPs solved so far across all batches.
    pub coalitions_solved: AtomicUsize,
    /// Permutation samples drawn so far. Advances only between batches, so on
    /// its own it makes a coarse, step-shaped progress bar.
    pub samples_done: AtomicUsize,
    /// Configured sample cap (denominator for a rough progress fraction).
    pub max_samples: AtomicUsize,
    /// Number of samples in the IN-FLIGHT batch (`min_samples` for the first
    /// batch, then `batch_size`). With the three batch fields below, a poller
    /// can interpolate WITHIN a batch — `percent = (samples_done + batch_samples
    /// · batch_solved/batch_total) / max_samples` — for a smooth, monotonic bar
    /// that moves as each LP solves rather than jumping once per batch.
    pub batch_samples: AtomicUsize,
    /// Coalition LPs the in-flight batch must solve (known after its Pass 1).
    /// 0 before Pass 1 completes; with B3 reuse this excludes reused coalitions,
    /// so it reflects real work.
    pub batch_total: AtomicUsize,
    /// Coalition LPs solved so far WITHIN the in-flight batch (reset each batch).
    pub batch_solved: AtomicUsize,
}

impl ComputeProgress {
    /// Zero every counter. Used when a single [`ComputeControl`] is reused across
    /// two sequential compute phases (e.g. baseline then modified) so the second
    /// phase's progress starts at 0 instead of inheriting the first phase's final
    /// counts. Relaxed stores are fine: a concurrent reader (the progress bridge)
    /// only renders a fraction, never coordinates on these values.
    pub fn reset(&self) {
        self.coalitions_solved.store(0, Ordering::Relaxed);
        self.samples_done.store(0, Ordering::Relaxed);
        self.max_samples.store(0, Ordering::Relaxed);
        self.batch_samples.store(0, Ordering::Relaxed);
        self.batch_total.store(0, Ordering::Relaxed);
        self.batch_solved.store(0, Ordering::Relaxed);
    }
}

#[derive(Debug)]
struct Shapley {
    pub private_links: PrivateLinks,
    pub devices: Devices,
    pub demands: Demands,
    pub public_links: PublicLinks,
    pub operator_uptime: f64,
    pub contiguity_bonus: f64,
    pub demand_multiplier: f64,
}

impl Shapley {
    fn new(
        private_links: PrivateLinks,
        devices: Devices,
        demands: Demands,
        public_links: PublicLinks,
        operator_uptime: f64,
        contiguity_bonus: f64,
        demand_multiplier: f64,
    ) -> Self {
        Self {
            private_links,
            devices,
            demands,
            public_links,
            operator_uptime,
            contiguity_bonus,
            demand_multiplier,
        }
    }

    fn compute(&self) -> Result<ShapleyOutput> {
        self.compute_inner(HashMap::new(), &[])
    }

    /// Exact compute with optional SOUND coalition reuse (B3). `seed_cache`
    /// holds baseline coalition values (masks include `ALWAYS_BIT`); a coalition
    /// is reused iff it excludes every operator in `changed_operators`, else
    /// solved fresh. `seed_cache` empty + `changed_operators` empty == plain
    /// `compute()`.
    fn compute_inner(
        &self,
        seed_cache: HashMap<u32, Option<f64>>,
        changed_operators: &[String],
    ) -> Result<ShapleyOutput> {
        // Validate inputs
        check_inputs(
            &self.private_links,
            &self.devices,
            &self.demands,
            &self.public_links,
            self.operator_uptime,
        )?;

        // Enumerate all operators (excluding "Private" and "Public")
        let mut operators: Vec<String> = self
            .devices
            .iter()
            .map(|d| d.operator.clone())
            .filter(|op| op != "Private" && op != "Public")
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        operators.sort();

        let n_operators = operators.len();
        if n_operators == 0 {
            return Ok(ShapleyOutput::new());
        }

        // Add hard limit to prevent computationally infeasible problems
        const MAX_OPERATORS: usize = 20;
        if n_operators > MAX_OPERATORS {
            return Err(ShapleyError::TooManyOperators {
                count: n_operators,
                limit: MAX_OPERATORS,
            });
        }

        // Consolidate demands and links
        let full_demand = consolidate_demand(&self.demands, self.demand_multiplier)?;
        let full_map = consolidate_links(
            &self.private_links,
            &self.devices,
            &full_demand,
            &self.public_links,
            self.contiguity_bonus,
        )?;

        // Build LP primitives
        let primitives = LpBuilderInput::new(&full_map, &full_demand).build()?;

        // Pre-compute row-oriented constraint data (once, before the coalition loop)
        let precomputed = PrecomputedRows::new(&primitives);

        // Pre-compute operator bitmasks (once, before the parallel loop)
        let op_index: HashMap<&str, u8> = operators
            .iter()
            .enumerate()
            .map(|(i, op)| (op.as_str(), i as u8))
            .collect();

        // B3 reuse gate (same rule as run_sampling): a coalition may take its
        // value from `seed_cache` iff it excludes every changed operator. Empty
        // when `changed_operators` is empty (no reuse).
        let reuse_gate: u32 = changed_operators
            .iter()
            .filter_map(|op| op_index.get(op.as_str()).map(|&idx| 1u32 << idx))
            .fold(0u32, |acc, b| acc | b);

        let operator_mask = |op: &str| -> u32 {
            if op == "Public" || op == "Private" || op.is_empty() {
                ALWAYS_BIT
            } else if let Some(&idx) = op_index.get(op) {
                1u32 << idx
            } else {
                0
            }
        };

        let col_op1_mask: Vec<u32> = primitives
            .col_op1
            .iter()
            .map(|s| operator_mask(s))
            .collect();
        let col_op2_mask: Vec<u32> = primitives
            .col_op2
            .iter()
            .map(|s| operator_mask(s))
            .collect();
        let row_op1_mask: Vec<u32> = primitives
            .row_op1
            .iter()
            .map(|s| operator_mask(s))
            .collect();
        let row_op2_mask: Vec<u32> = primitives
            .row_op2
            .iter()
            .map(|s| operator_mask(s))
            .collect();

        let n_coalitions = 1 << n_operators;
        let n_cols = col_op1_mask.len();
        eprintln!(
            "[shapley] LP dims: {} cols, {} eq-rows, {} ub-rows; {} operators, {} coalitions (exact)",
            n_cols,
            primitives.b_eq.len(),
            primitives.b_ub.len(),
            n_operators,
            n_coalitions,
        );

        thread_local! {
            static BUFFERS: RefCell<Option<CoalitionBuffers>> = const { RefCell::new(None) };
        }

        // Solve LP for each coalition
        let coalition_values: Vec<Option<f64>> = (0..n_coalitions)
            .into_par_iter()
            .map(|coalition_idx| {
                let coalition_mask = (coalition_idx as u32) | ALWAYS_BIT;

                // B3 reuse: a coalition excluding every changed operator has an
                // identical sub-LP (column gating), so its seeded value is exact.
                if (coalition_idx as u32) & reuse_gate == 0
                    && let Some(&seeded) = seed_cache.get(&coalition_mask)
                {
                    return seeded;
                }

                BUFFERS.with(|cell| {
                    let mut borrow = cell.borrow_mut();
                    let buf = borrow.get_or_insert_with(|| CoalitionBuffers::new(n_cols));

                    match solve_coalition(
                        &primitives,
                        &precomputed,
                        buf,
                        coalition_mask,
                        &col_op1_mask,
                        &col_op2_mask,
                        &row_op1_mask,
                        &row_op2_mask,
                    ) {
                        Ok(result) => {
                            if matches!(result.status, SolveStatus::Solved) {
                                Some(-result.objective_value) // Negative because we minimize
                            } else {
                                None // Infeasible coalition
                            }
                        }
                        Err(_) => None,
                    }
                })
            })
            .collect();

        // Compute expected values with operator uptime
        let expected_values = if self.operator_uptime < 1.0 {
            compute_expected_values(&coalition_values, n_operators, self.operator_uptime)?
        } else {
            coalition_values
                .iter()
                .map(|&v| v.unwrap_or(f64::NEG_INFINITY))
                .collect()
        };

        // Compute Shapley values
        let shapley_values = compute_shapley_values(&expected_values, n_operators);

        // Convert to output format
        let total_value: f64 = shapley_values.iter().map(|v| v.max(0.0)).sum();

        let output = operators
            .into_iter()
            .zip(shapley_values)
            .map(|(operator, value)| {
                let proportion = if total_value > 0.0 {
                    (value.max(0.0) / total_value * 100.0) / 100.0
                } else {
                    0.0
                };

                (operator, ShapleyValue { value, proportion })
            })
            .collect();

        Ok(output)
    }

    fn compute_sampled(&self, config: SamplingConfig) -> Result<SampledOutput> {
        self.run_sampling(config, HashMap::new(), &[], None)
    }

    fn compute_sampled_cancellable(
        &self,
        config: SamplingConfig,
        control: &ComputeControl,
    ) -> Result<SampledOutput> {
        self.run_sampling(config, HashMap::new(), &[], Some(control))
    }

    fn compute_sampled_with_reuse(
        &self,
        config: SamplingConfig,
        seed_cache: HashMap<u32, Option<f64>>,
        changed_operators: &[String],
    ) -> Result<SampledOutput> {
        self.run_sampling(config, seed_cache, changed_operators, None)
    }

    fn compute_sampled_with_reuse_cancellable(
        &self,
        config: SamplingConfig,
        seed_cache: HashMap<u32, Option<f64>>,
        changed_operators: &[String],
        control: &ComputeControl,
    ) -> Result<SampledOutput> {
        self.run_sampling(config, seed_cache, changed_operators, Some(control))
    }

    /// Shared adaptive Monte-Carlo permutation sampler.
    ///
    /// `coalition_cache` seeds previously-solved coalition values (empty for a
    /// cold run; populated by [`Shapley::compute_sampled_with_cache`]).
    /// Coalitions already present are not re-solved, so callers MUST only seed
    /// values that are valid for *this* topology.
    ///
    /// Honours `operator_uptime`. When uptime < 1.0 each sampled permutation
    /// also draws an independent "up" mask (each operator up with probability
    /// `operator_uptime`); the per-operator marginal is
    /// `uptime * (v(U ∪ {i}) - v(U))` where `U` is the set of *up* predecessors.
    /// This is an unbiased estimator of the same uptime-weighted Shapley value
    /// the exact path produces via [`compute_expected_values`]:
    ///   `E[v(S∪i) under uptime] - E[v(S) under uptime] = uptime · E_U[v(U∪i) - v(U)]`.
    /// At uptime == 1.0 every operator is always up, so `up_prefix` is the full
    /// permutation prefix and this reduces exactly to the plain marginal.
    fn run_sampling(
        &self,
        config: SamplingConfig,
        seed_cache: HashMap<u32, Option<f64>>,
        changed_operators: &[String],
        control: Option<&ComputeControl>,
    ) -> Result<SampledOutput> {
        // ── Setup (identical to compute()) ──────────────────────────
        check_inputs(
            &self.private_links,
            &self.devices,
            &self.demands,
            &self.public_links,
            self.operator_uptime,
        )?;

        let mut operators: Vec<String> = self
            .devices
            .iter()
            .map(|d| d.operator.clone())
            .filter(|op| op != "Private" && op != "Public")
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        operators.sort();

        let n = operators.len();
        if n == 0 {
            return Ok(SampledOutput {
                values: ShapleyOutput::new(),
                samples_used: 0,
                standard_errors: BTreeMap::new(),
                converged: true,
                coalition_cache: HashMap::new(),
                coalitions_reused: 0,
            });
        }

        const MAX_OPERATORS: usize = 20;
        if n > MAX_OPERATORS {
            return Err(ShapleyError::TooManyOperators {
                count: n,
                limit: MAX_OPERATORS,
            });
        }

        // Consolidate demands and links
        let full_demand = consolidate_demand(&self.demands, self.demand_multiplier)?;
        let full_map = consolidate_links(
            &self.private_links,
            &self.devices,
            &full_demand,
            &self.public_links,
            self.contiguity_bonus,
        )?;

        // Build LP primitives
        let primitives = LpBuilderInput::new(&full_map, &full_demand).build()?;
        let precomputed = PrecomputedRows::new(&primitives);

        // Build operator bitmasks (identical to compute())
        let op_index: HashMap<&str, u8> = operators
            .iter()
            .enumerate()
            .map(|(i, op)| (op.as_str(), i as u8))
            .collect();

        // ── B3 selective reuse: admit only SOUND seed entries ───────────────
        // `changed_operators` are the operators owning a primitive that differs
        // between the seed's topology and this one (derived by the caller from
        // COLUMN-side link/device endpoints only — never row tags). A cached
        // coalition value is reusable iff the coalition excludes every changed
        // operator: column gating (solver.rs) then guarantees its sub-LP — kept
        // columns + costs + kept rows' surviving coefficients + all eq-rows — is
        // byte-identical across the two topologies, so v(S) (and infeasibility)
        // is unchanged. We map names→bits with THIS run's op_index (so a caller
        // never has to know our sorted-bit assignment), and admit only the
        // gate-passing subset: gated-out masks are simply absent, so the Pass-1
        // `contains_key` check re-solves them and the Pass-3 reads never miss.
        let reuse_gate: u32 = changed_operators
            .iter()
            .filter_map(|op| op_index.get(op.as_str()).map(|&idx| 1u32 << idx))
            .fold(0u32, |acc, b| acc | b);
        let mut coalition_cache: HashMap<u32, Option<f64>> =
            HashMap::with_capacity(seed_cache.len());
        for (mask, val) in seed_cache {
            if mask & reuse_gate == 0 {
                coalition_cache.insert(mask, val);
            }
        }
        let coalitions_reused = coalition_cache.len();

        let operator_mask = |op: &str| -> u32 {
            if op == "Public" || op == "Private" || op.is_empty() {
                ALWAYS_BIT
            } else if let Some(&idx) = op_index.get(op) {
                1u32 << idx
            } else {
                0
            }
        };

        let col_op1_mask: Vec<u32> = primitives
            .col_op1
            .iter()
            .map(|s| operator_mask(s))
            .collect();
        let col_op2_mask: Vec<u32> = primitives
            .col_op2
            .iter()
            .map(|s| operator_mask(s))
            .collect();
        let row_op1_mask: Vec<u32> = primitives
            .row_op1
            .iter()
            .map(|s| operator_mask(s))
            .collect();
        let row_op2_mask: Vec<u32> = primitives
            .row_op2
            .iter()
            .map(|s| operator_mask(s))
            .collect();
        let n_cols = col_op1_mask.len();
        eprintln!(
            "[shapley] LP dims: {} cols, {} eq-rows, {} ub-rows; {} operators (sampled)",
            n_cols,
            primitives.b_eq.len(),
            primitives.b_ub.len(),
            n,
        );

        if let Some(c) = control {
            c.progress
                .max_samples
                .store(config.max_samples, Ordering::Relaxed);
        }

        // ── Adaptive sampling loop ──────────────────────────────────
        // `apply_uptime` switches on the Bernoulli active-set estimator; at
        // uptime == 1.0 every operator is always up and `up_prefix` tracks the
        // full permutation prefix (the original estimator).
        let uptime = self.operator_uptime;
        let apply_uptime = uptime < 1.0;
        let mut rng = rand::rng();
        let mut all_marginals: Vec<Vec<f64>> = Vec::new();
        let mut total_samples: usize = 0;

        if !coalition_cache.is_empty() {
            eprintln!(
                "[shapley] starting with {} pre-cached coalitions",
                coalition_cache.len(),
            );
        }

        loop {
            let batch = if total_samples == 0 {
                config.min_samples
            } else {
                config.batch_size
            };

            // Reset the in-flight-batch progress counters (batch_total is filled
            // once Pass 1 has collected the unique masks for this batch).
            if let Some(c) = control {
                c.progress.batch_samples.store(batch, Ordering::Relaxed);
                c.progress.batch_total.store(0, Ordering::Relaxed);
                c.progress.batch_solved.store(0, Ordering::Relaxed);
            }

            // ── Pass 1: sample permutations (+ up-masks), collect needed masks ──
            let mut needed_masks: HashSet<u32> = HashSet::new();
            let mut batch_perms: Vec<(Vec<usize>, Vec<bool>)> = Vec::with_capacity(batch);

            for _ in 0..batch {
                let mut perm: Vec<usize> = (0..n).collect();
                perm.shuffle(&mut rng);

                // Per-operator "up" draw (all-up when uptime >= 1.0).
                let up: Vec<bool> = if apply_uptime {
                    (0..n).map(|_| rng.random::<f64>() < uptime).collect()
                } else {
                    vec![true; n]
                };

                // Walk the permutation, accumulating the up-predecessor
                // coalition `U` and collecting the masks `U` and `U ∪ {i}`.
                let mut up_prefix: u32 = 0;
                for &i in &perm {
                    let u_full = up_prefix | ALWAYS_BIT;
                    let ui_full = up_prefix | (1u32 << i) | ALWAYS_BIT;
                    if !coalition_cache.contains_key(&u_full) {
                        needed_masks.insert(u_full);
                    }
                    if !coalition_cache.contains_key(&ui_full) {
                        needed_masks.insert(ui_full);
                    }
                    if up[i] {
                        up_prefix |= 1u32 << i;
                    }
                }
                batch_perms.push((perm, up));
            }

            // ── Pass 2: Solve new coalitions in parallel (rayon) ────
            let new_masks: Vec<u32> = needed_masks.into_iter().collect();
            let new_count = new_masks.len();
            // Publish this batch's solve target so a poller can interpolate.
            if let Some(c) = control {
                c.progress.batch_total.store(new_count, Ordering::Relaxed);
            }
            let cached_count = coalition_cache.len();
            eprintln!(
                "[shapley] batch {}: {} perms, {} new coalitions to solve ({} cached)",
                total_samples / batch.max(1) + 1,
                batch,
                new_count,
                cached_count,
            );

            thread_local! {
                static SAMP_BUFFERS: RefCell<Option<CoalitionBuffers>> =
                    const { RefCell::new(None) };
            }

            let solve_start = std::time::Instant::now();
            let new_values: Vec<(u32, Option<f64>)> = new_masks
                .par_iter()
                .map(|&mask| {
                    // Cooperative cancellation: once flagged, drain the
                    // remaining masks without solving (rayon has no clean
                    // early-abort, so we make each remaining item cheap).
                    if control.is_some_and(|c| c.cancel.load(Ordering::Relaxed)) {
                        return (mask, None);
                    }
                    SAMP_BUFFERS.with(|cell| {
                        let mut borrow = cell.borrow_mut();
                        let buf = borrow.get_or_insert_with(|| CoalitionBuffers::new(n_cols));

                        let val = match solve_coalition(
                            &primitives,
                            &precomputed,
                            buf,
                            mask,
                            &col_op1_mask,
                            &col_op2_mask,
                            &row_op1_mask,
                            &row_op2_mask,
                        ) {
                            Ok(result) => {
                                if matches!(result.status, SolveStatus::Solved) {
                                    Some(-result.objective_value)
                                } else {
                                    None
                                }
                            }
                            Err(_) => None,
                        };
                        if let Some(c) = control {
                            c.progress.coalitions_solved.fetch_add(1, Ordering::Relaxed);
                            c.progress.batch_solved.fetch_add(1, Ordering::Relaxed);
                        }
                        (mask, val)
                    })
                })
                .collect();

            for (mask, val) in new_values {
                coalition_cache.insert(mask, val);
            }
            let solve_elapsed = solve_start.elapsed();
            eprintln!(
                "[shapley] batch {} solved {} coalitions in {:.1}s ({:.0}ms/lp)",
                total_samples / batch.max(1) + 1,
                new_count,
                solve_elapsed.as_secs_f64(),
                if new_count > 0 {
                    solve_elapsed.as_millis() as f64 / new_count as f64
                } else {
                    0.0
                },
            );

            // ── Pass 3: Compute marginal contributions ──────────────
            // Replays the same up-mask walk and reads cached values. The
            // `uptime` factor + up-predecessor coalition `U` implement the
            // uptime-weighted marginal; at uptime == 1.0 this is `v(S∪i)-v(S)`.
            for (perm, up) in &batch_perms {
                let mut marginals = vec![0.0f64; n];
                let mut up_prefix: u32 = 0;
                for &i in perm {
                    let u_full = up_prefix | ALWAYS_BIT;
                    let ui_full = up_prefix | (1u32 << i) | ALWAYS_BIT;
                    let v_before = coalition_cache[&u_full].unwrap_or(0.0);
                    let v_with = coalition_cache[&ui_full].unwrap_or(0.0);
                    marginals[i] = uptime * (v_with - v_before);
                    if up[i] {
                        up_prefix |= 1u32 << i;
                    }
                }
                all_marginals.push(marginals);
            }
            total_samples += batch;

            // Publish progress + honour cancellation between batches.
            if let Some(c) = control {
                c.progress
                    .samples_done
                    .store(total_samples, Ordering::Relaxed);
                if c.cancel.load(Ordering::Relaxed) {
                    return Err(ShapleyError::Cancelled);
                }
            }

            // ── Check convergence ───────────────────────────────────
            if total_samples >= config.min_samples {
                let (means, ses) = welford_statistics(&all_marginals, n);
                let max_relative_se = means
                    .iter()
                    .zip(&ses)
                    .filter(|(m, _)| m.abs() > 1e-10)
                    .map(|(m, s)| s / m.abs())
                    .fold(0.0f64, f64::max);

                if max_relative_se <= config.target_se || total_samples >= config.max_samples {
                    let converged = max_relative_se <= config.target_se;

                    // Build output (same format as compute())
                    let total_value: f64 = means.iter().map(|v| v.max(0.0)).sum();
                    let values: ShapleyOutput = operators
                        .iter()
                        .enumerate()
                        .map(|(i, op)| {
                            let proportion = if total_value > 0.0 {
                                (means[i].max(0.0) / total_value * 100.0) / 100.0
                            } else {
                                0.0
                            };
                            (
                                op.clone(),
                                ShapleyValue {
                                    value: means[i],
                                    proportion,
                                },
                            )
                        })
                        .collect();

                    let standard_errors: BTreeMap<String, f64> = operators
                        .iter()
                        .enumerate()
                        .map(|(i, op)| (op.clone(), ses[i]))
                        .collect();

                    return Ok(SampledOutput {
                        values,
                        samples_used: total_samples,
                        standard_errors,
                        converged,
                        coalition_cache,
                        coalitions_reused,
                    });
                }
            }

            // Safety: hard cap (should not reach here due to check above)
            if total_samples >= config.max_samples {
                break;
            }
        }

        // Fallback (unreachable in practice)
        Err(ShapleyError::LpSolver(
            "sampling loop exited unexpectedly".to_string(),
        ))
    }

    /// Like [`compute_sampled`] but seeds the coalition cache from
    /// `existing_cache` so previously-solved coalitions are not re-solved.
    ///
    /// SAFETY: `existing_cache` must hold coalition values computed for the
    /// SAME topology (same private_links / costs). Coalition masks key only by
    /// operator membership, not topology, so seeding values from a different
    /// topology silently reuses stale objectives. The simulate handler keys
    /// its epoch cache by a topology hash to enforce this.
    fn compute_sampled_with_cache(
        &self,
        config: SamplingConfig,
        existing_cache: HashMap<u32, Option<f64>>,
    ) -> Result<SampledOutput> {
        // Whole-cache reuse (no changed operators) — caller guarantees the cache
        // is for the SAME topology (see SAFETY note above).
        self.run_sampling(config, existing_cache, &[], None)
    }

    fn compute_sampled_with_cache_cancellable(
        &self,
        config: SamplingConfig,
        existing_cache: HashMap<u32, Option<f64>>,
        control: &ComputeControl,
    ) -> Result<SampledOutput> {
        self.run_sampling(config, existing_cache, &[], Some(control))
    }
}

/// Compute expected values considering operator uptime.
///
/// For each coalition S, computes:
///   evalue[S] = Σ_{T⊆S} uptime^|T| × (1-uptime)^(|S\T|) × svalue[T]
///
/// Uses Gosper's subset iteration (`t = (t-1) & s`) for O(3^n) total work
/// instead of O(4^n) dense matrix operations.
fn compute_expected_values(
    svalue: &[Option<f64>],
    n_operators: usize,
    operator_uptime: f64,
) -> Result<Vec<f64>> {
    let n_coal = 1 << n_operators;
    let downtime = 1.0 - operator_uptime;

    let svalue_vec: Vec<f64> = svalue
        .iter()
        .map(|&v| v.unwrap_or(f64::NEG_INFINITY))
        .collect();

    let mut evalue = vec![0.0; n_coal];

    for (s, ev) in evalue.iter_mut().enumerate() {
        let s_size = (s as u32).count_ones() as i32;
        let mut sum = 0.0;

        // Iterate over all subsets t of s (including empty set)
        let mut t = s;
        loop {
            let val = svalue_vec[t];
            if val.is_finite() {
                let t_size = (t as u32).count_ones() as i32;
                let prob = operator_uptime.powi(t_size) * downtime.powi(s_size - t_size);
                sum += prob * val;
            }
            if t == 0 {
                break;
            }
            t = (t - 1) & s;
        }

        *ev = sum;
    }

    // Preserve empty coalition value
    if let Some(v) = svalue[0]
        && v.is_finite()
    {
        evalue[0] = v;
    }

    Ok(evalue)
}

/// Compute Shapley values from coalition values
fn compute_shapley_values(coalition_values: &[f64], n_operators: usize) -> Vec<f64> {
    let mut shapley_values = vec![0.0; n_operators];
    let fact_n = factorial(n_operators);

    for (k, sv) in shapley_values.iter_mut().enumerate() {
        let mut value = 0.0;

        // Find coalitions with this operator
        for (coalition_idx, &with_value) in coalition_values.iter().enumerate() {
            if (coalition_idx >> k) & 1 == 1 {
                // Coalition without operator (remove bit k)
                let without_idx = coalition_idx ^ (1 << k);
                let without_value = coalition_values[without_idx];

                // Coalition size
                let coalition_size = (coalition_idx as u32).count_ones() as usize;

                // Weight calculation
                let weight = factorial(coalition_size - 1)
                    * factorial(n_operators - coalition_size)
                    / fact_n;

                value += weight * (with_value - without_value);
            }
        }

        *sv = value;
    }

    shapley_values
}

/// Compute per-operator mean and standard error using Welford's online algorithm.
/// Avoids numerical instability from computing Σx² directly.
fn welford_statistics(marginals: &[Vec<f64>], n: usize) -> (Vec<f64>, Vec<f64>) {
    let m = marginals.len() as f64;
    let mut means = vec![0.0; n];
    let mut m2 = vec![0.0; n];

    for (k, sample) in marginals.iter().enumerate() {
        let k1 = (k + 1) as f64;
        for i in 0..n {
            let delta = sample[i] - means[i];
            means[i] += delta / k1;
            let delta2 = sample[i] - means[i];
            m2[i] += delta * delta2;
        }
    }

    let ses: Vec<f64> = m2
        .iter()
        .map(|v| (v / (m * (m - 1.0).max(1.0))).sqrt())
        .collect();

    (means, ses)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Demand, Device, PrivateLink, PublicLink};

    #[test]
    fn test_shapley_computation() {
        // Create simple test data following the example format
        let private_links = vec![
            PrivateLink::new(
                "NYC1".to_string(),
                "LON1".to_string(),
                10.0,
                100.0,
                1.0,
                Some(1),
            ),
            PrivateLink::new(
                "LON1".to_string(),
                "PAR1".to_string(),
                10.0,
                100.0,
                1.0,
                Some(2),
            ),
        ];

        let devices = vec![
            Device::new("NYC1".to_string(), 1, "Operator1".to_string()),
            Device::new("LON1".to_string(), 1, "Operator1".to_string()),
            Device::new("PAR1".to_string(), 1, "Operator2".to_string()),
        ];

        let demands = vec![Demand::new(
            "NYC".to_string(),
            "PAR".to_string(),
            1,
            50.0,
            1.0,
            1,
            false,
        )];

        let public_links = vec![PublicLink::new("NYC".to_string(), "PAR".to_string(), 100.0)];

        let shapley = Shapley::new(private_links, devices, demands, public_links, 1.0, 5.0, 1.0);

        let result = shapley.compute();
        assert!(result.is_ok(), "Error in test: {result:?}");

        let values = result.expect("Shapley computation should succeed in tests");
        assert_eq!(values.len(), 2); // Two operators
    }

    #[test]
    fn test_compute_expected_values_simple() {
        // Test with 2 operators, uptime = 0.9
        let n_ops = 2;
        let uptime = 0.9; // 0.9

        // svalue for coalitions: {}, {B}, {A}, {A,B}
        let svalue = vec![Some(100.0), Some(120.0), Some(150.0), Some(200.0)];

        let evalue = compute_expected_values(&svalue, n_ops, uptime)
            .expect("Expected value computation should succeed in tests");

        // These expected values are derived from running the reference Python code
        // with the same inputs.
        let expected_evalue = vec![100.0, 118.0, 145.0, 187.3];

        for (i, (val, exp)) in evalue.iter().zip(expected_evalue).enumerate() {
            assert!(
                (val - exp).abs() < 1e-9,
                "Mismatch at index {i}: got {val}, expected {exp}",
            );
        }
    }

    #[test]
    fn test_sampled_matches_exact_2_operators() {
        let private_links = vec![
            PrivateLink::new("NYC1".into(), "LON1".into(), 10.0, 100.0, 1.0, Some(1)),
            PrivateLink::new("LON1".into(), "PAR1".into(), 10.0, 100.0, 1.0, Some(2)),
        ];
        let devices = vec![
            Device::new("NYC1".into(), 1, "Operator1".into()),
            Device::new("LON1".into(), 1, "Operator1".into()),
            Device::new("PAR1".into(), 1, "Operator2".into()),
        ];
        let demands = vec![Demand::new(
            "NYC".into(),
            "PAR".into(),
            1,
            50.0,
            1.0,
            1,
            false,
        )];
        let public_links = vec![PublicLink::new("NYC".into(), "PAR".into(), 100.0)];

        let input = ShapleyInput {
            private_links,
            devices,
            demands,
            public_links,
            operator_uptime: 1.0,
            contiguity_bonus: 5.0,
            demand_multiplier: 1.0,
        };

        let exact = input.compute().expect("exact compute should succeed");
        let sampled = input
            .compute_sampled(SamplingConfig {
                min_samples: 500,
                max_samples: 500,
                target_se: 0.001,
                batch_size: 500,
            })
            .expect("sampled compute should succeed");

        // With only 2 operators (4 coalitions), 500 samples should be very close
        for (op, ev) in &exact {
            let sv = sampled.values.get(op).unwrap();
            let err = (ev.proportion - sv.proportion).abs();
            assert!(
                err < 0.05,
                "{op}: exact={:.4} sampled={:.4} err={:.4}",
                ev.proportion,
                sv.proportion,
                err
            );
        }
    }

    #[test]
    fn test_sampled_matches_exact_with_uptime() {
        // Regression for the bug where compute_sampled ignored operator_uptime
        // and therefore computed a different quantity than exact compute() for
        // the production default uptime < 1.0. With the Bernoulli active-set
        // estimator the sampled proportions must converge to exact.
        let private_links = vec![
            PrivateLink::new("NYC1".into(), "LON1".into(), 10.0, 100.0, 1.0, Some(1)),
            PrivateLink::new("LON1".into(), "PAR1".into(), 10.0, 100.0, 1.0, Some(2)),
        ];
        let devices = vec![
            Device::new("NYC1".into(), 1, "Operator1".into()),
            Device::new("LON1".into(), 1, "Operator1".into()),
            Device::new("PAR1".into(), 1, "Operator2".into()),
        ];
        let demands = vec![Demand::new(
            "NYC".into(),
            "PAR".into(),
            1,
            50.0,
            1.0,
            1,
            false,
        )];
        let public_links = vec![PublicLink::new("NYC".into(), "PAR".into(), 100.0)];

        let input = ShapleyInput {
            private_links,
            devices,
            demands,
            public_links,
            operator_uptime: 0.9, // < 1.0 — exercises the uptime expectation
            contiguity_bonus: 5.0,
            demand_multiplier: 1.0,
        };

        let exact = input.compute().expect("exact compute should succeed");
        let sampled = input
            .compute_sampled(SamplingConfig {
                min_samples: 4000,
                max_samples: 4000,
                target_se: 0.0001,
                batch_size: 4000,
            })
            .expect("sampled compute should succeed");

        for (op, ev) in &exact {
            let sv = sampled
                .values
                .get(op)
                .unwrap_or_else(|| panic!("operator {op} missing from sampled output"));
            let err = (ev.proportion - sv.proportion).abs();
            assert!(
                err < 0.06,
                "{op}: exact={:.4} sampled={:.4} err={:.4} (uptime path must match exact)",
                ev.proportion,
                sv.proportion,
                err
            );
        }
    }

    #[test]
    fn test_sampled_adaptive_converges() {
        let private_links = vec![
            PrivateLink::new("NYC1".into(), "LON1".into(), 10.0, 100.0, 1.0, Some(1)),
            PrivateLink::new("LON1".into(), "PAR1".into(), 10.0, 100.0, 1.0, Some(2)),
        ];
        let devices = vec![
            Device::new("NYC1".into(), 1, "Operator1".into()),
            Device::new("LON1".into(), 1, "Operator1".into()),
            Device::new("PAR1".into(), 1, "Operator2".into()),
        ];
        let demands = vec![Demand::new(
            "NYC".into(),
            "PAR".into(),
            1,
            50.0,
            1.0,
            1,
            false,
        )];
        let public_links = vec![PublicLink::new("NYC".into(), "PAR".into(), 100.0)];

        let input = ShapleyInput {
            private_links,
            devices,
            demands,
            public_links,
            operator_uptime: 1.0,
            contiguity_bonus: 5.0,
            demand_multiplier: 1.0,
        };

        let result = input
            .compute_sampled(SamplingConfig::default())
            .expect("sampled compute should succeed");

        assert!(
            result.converged,
            "should converge for simple 2-operator case"
        );
        assert!(
            result.samples_used >= 100,
            "should use at least min_samples"
        );
        assert!(result.samples_used <= 500, "should not exceed max_samples");
        assert_eq!(result.values.len(), 2, "should have 2 operators");
    }

    #[test]
    fn test_welford_statistics() {
        // Known values: 3 samples of 2 operators
        let marginals = vec![vec![10.0, 20.0], vec![12.0, 18.0], vec![11.0, 22.0]];
        let (means, ses) = welford_statistics(&marginals, 2);

        assert!((means[0] - 11.0).abs() < 1e-10, "mean[0] should be 11.0");
        assert!((means[1] - 20.0).abs() < 1e-10, "mean[1] should be 20.0");

        // SE = sqrt(variance / n) where variance = Σ(x-μ)²/(n-1)
        // For op0: var = ((10-11)² + (12-11)² + (11-11)²) / 2 = 1.0
        // SE = sqrt(1.0 / 3) ≈ 0.5774
        assert!(
            (ses[0] - (1.0f64 / 3.0).sqrt()).abs() < 1e-10,
            "SE[0] mismatch: {}",
            ses[0]
        );
    }

    #[test]
    fn test_sampling_config_for_problem() {
        let config = SamplingConfig::for_problem(14, 1148);
        assert_eq!(config.min_samples, 50);
        assert_eq!(config.max_samples, 300);

        let config_heavy = SamplingConfig::for_problem(14, 3000);
        assert_eq!(config_heavy.min_samples, 40);
        assert_eq!(config_heavy.max_samples, 150);
    }
}
