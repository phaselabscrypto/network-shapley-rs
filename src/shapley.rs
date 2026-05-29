use std::{
    cell::RefCell,
    collections::{BTreeMap, HashMap, HashSet},
    fmt::{Display, Formatter},
};

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
        let (base, cap) = if n_demands > 2000 { (80, 300) } else { (100, 500) };
        Self {
            min_samples: base,
            max_samples: cap,
            ..Default::default()
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

        thread_local! {
            static BUFFERS: RefCell<Option<CoalitionBuffers>> = const { RefCell::new(None) };
        }

        // Solve LP for each coalition
        let coalition_values: Vec<Option<f64>> = (0..n_coalitions)
            .into_par_iter()
            .map(|coalition_idx| {
                BUFFERS.with(|cell| {
                    let mut borrow = cell.borrow_mut();
                    let buf = borrow.get_or_insert_with(|| CoalitionBuffers::new(n_cols));

                    let coalition_mask = (coalition_idx as u32) | ALWAYS_BIT;

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

        let operator_mask = |op: &str| -> u32 {
            if op == "Public" || op == "Private" || op.is_empty() {
                ALWAYS_BIT
            } else if let Some(&idx) = op_index.get(op) {
                1u32 << idx
            } else {
                0
            }
        };

        let col_op1_mask: Vec<u32> = primitives.col_op1.iter().map(|s| operator_mask(s)).collect();
        let col_op2_mask: Vec<u32> = primitives.col_op2.iter().map(|s| operator_mask(s)).collect();
        let row_op1_mask: Vec<u32> = primitives.row_op1.iter().map(|s| operator_mask(s)).collect();
        let row_op2_mask: Vec<u32> = primitives.row_op2.iter().map(|s| operator_mask(s)).collect();
        let n_cols = col_op1_mask.len();

        // ── Adaptive sampling loop ──────────────────────────────────
        let mut rng = rand::rng();
        let mut all_marginals: Vec<Vec<f64>> = Vec::new();
        let mut coalition_cache: HashMap<u32, Option<f64>> = HashMap::new();
        let mut total_samples: usize = 0;

        loop {
            let batch = if total_samples == 0 {
                config.min_samples
            } else {
                config.batch_size
            };

            // ── Pass 1: Generate permutations, collect needed masks ──
            let mut needed_masks: HashSet<u32> = HashSet::new();
            let mut batch_perms: Vec<Vec<usize>> = Vec::with_capacity(batch);

            for _ in 0..batch {
                let mut perm: Vec<usize> = (0..n).collect();
                perm.shuffle(&mut rng);

                let mut mask: u32 = 0;
                // Empty coalition
                let empty_full = mask | ALWAYS_BIT;
                if !coalition_cache.contains_key(&empty_full) {
                    needed_masks.insert(empty_full);
                }
                for &i in &perm {
                    mask |= 1u32 << i;
                    let full = mask | ALWAYS_BIT;
                    if !coalition_cache.contains_key(&full) {
                        needed_masks.insert(full);
                    }
                }
                batch_perms.push(perm);
            }

            // ── Pass 2: Solve new coalitions in parallel (rayon) ────
            let new_masks: Vec<u32> = needed_masks.into_iter().collect();

            thread_local! {
                static SAMP_BUFFERS: RefCell<Option<CoalitionBuffers>> =
                    const { RefCell::new(None) };
            }

            let new_values: Vec<(u32, Option<f64>)> = new_masks
                .par_iter()
                .map(|&mask| {
                    SAMP_BUFFERS.with(|cell| {
                        let mut borrow = cell.borrow_mut();
                        let buf =
                            borrow.get_or_insert_with(|| CoalitionBuffers::new(n_cols));

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
                        (mask, val)
                    })
                })
                .collect();

            for (mask, val) in new_values {
                coalition_cache.insert(mask, val);
            }

            // ── Pass 3: Compute marginal contributions ──────────────
            for perm in &batch_perms {
                let mut marginals = vec![0.0f64; n];
                let mut mask: u32 = 0;
                for &i in perm {
                    let v_before = coalition_cache[&(mask | ALWAYS_BIT)].unwrap_or(0.0);
                    mask |= 1u32 << i;
                    let v_with = coalition_cache[&(mask | ALWAYS_BIT)].unwrap_or(0.0);
                    marginals[i] = v_with - v_before;
                }
                all_marginals.push(marginals);
            }
            total_samples += batch;

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
                            (op.clone(), ShapleyValue { value: means[i], proportion })
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
                    });
                }
            }

            // Safety: hard cap (should not reach here due to check above)
            if total_samples >= config.max_samples {
                break;
            }
        }

        // Fallback (unreachable in practice)
        Err(ShapleyError::LpSolver("sampling loop exited unexpectedly".to_string()))
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
        let demands = vec![Demand::new("NYC".into(), "PAR".into(), 1, 50.0, 1.0, 1, false)];
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
                ev.proportion, sv.proportion, err
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
        let demands = vec![Demand::new("NYC".into(), "PAR".into(), 1, 50.0, 1.0, 1, false)];
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

        assert!(result.converged, "should converge for simple 2-operator case");
        assert!(
            result.samples_used >= 100,
            "should use at least min_samples"
        );
        assert!(
            result.samples_used <= 500,
            "should not exceed max_samples"
        );
        assert_eq!(result.values.len(), 2, "should have 2 operators");
    }

    #[test]
    fn test_welford_statistics() {
        // Known values: 3 samples of 2 operators
        let marginals = vec![
            vec![10.0, 20.0],
            vec![12.0, 18.0],
            vec![11.0, 22.0],
        ];
        let (means, ses) = welford_statistics(&marginals, 2);

        assert!((means[0] - 11.0).abs() < 1e-10, "mean[0] should be 11.0");
        assert!((means[1] - 20.0).abs() < 1e-10, "mean[1] should be 20.0");

        // SE = sqrt(variance / n) where variance = Σ(x-μ)²/(n-1)
        // For op0: var = ((10-11)² + (12-11)² + (11-11)²) / 2 = 1.0
        // SE = sqrt(1.0 / 3) ≈ 0.5774
        assert!((ses[0] - (1.0f64 / 3.0).sqrt()).abs() < 1e-10, "SE[0] mismatch: {}", ses[0]);
    }

    #[test]
    fn test_sampling_config_for_problem() {
        let config = SamplingConfig::for_problem(14, 1148);
        assert_eq!(config.min_samples, 100);
        assert_eq!(config.max_samples, 500);

        let config_heavy = SamplingConfig::for_problem(14, 3000);
        assert_eq!(config_heavy.min_samples, 80);
        assert_eq!(config_heavy.max_samples, 300);
    }
}
