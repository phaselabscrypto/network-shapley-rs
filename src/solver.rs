use std::sync::Arc;

use crate::{
    error::{Result, ShapleyError},
    lp_builder::LpPrimitives,
    sparse::CscMatrix,
};

/// Pre-computed row-oriented representation of the LP constraint matrices.
/// Built once from the full primitives, then reused for every coalition.
pub(crate) struct PrecomputedRows {
    /// Equality constraint rows: each entry is (original_col_index, coefficient)
    eq_rows: Vec<Vec<(usize, f64)>>,
    /// Inequality constraint rows: each entry is (original_col_index, coefficient)
    ub_rows: Vec<Vec<(usize, f64)>>,
}

impl PrecomputedRows {
    /// Build from the full (unfiltered) LP primitives. Call once before the coalition loop.
    pub(crate) fn new(primitives: &LpPrimitives) -> Self {
        Self {
            eq_rows: rows_from_csc(&primitives.a_eq),
            ub_rows: rows_from_csc(&primitives.a_ub),
        }
    }
}

/// Reusable per-thread buffers for coalition LP construction.
pub(crate) struct CoalitionBuffers {
    pub col_remap: Vec<usize>,
    pub cost: Vec<f64>,
    pub keep_rows: Vec<usize>,
}

impl CoalitionBuffers {
    pub fn new(n_cols: usize) -> Self {
        Self {
            col_remap: vec![usize::MAX; n_cols],
            cost: Vec::with_capacity(n_cols),
            keep_rows: Vec::with_capacity(256),
        }
    }

    pub fn reset(&mut self) {
        self.col_remap.fill(usize::MAX);
        self.cost.clear();
        self.keep_rows.clear();
    }
}

/// Solver termination status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SolveStatus {
    Solved,
    Infeasible,
}

/// LP solver wrapper for HiGHS (used in tests)
#[cfg(test)]
pub(crate) struct LpSolver {
    pb: highs::RowProblem,
}

/// Result of solving an LP (used in tests)
#[cfg(test)]
#[derive(Debug)]
#[allow(dead_code)]
pub(crate) struct LpSolution {
    pub status: SolveStatus,
    pub objective_value: f64,
}

/// Collect all entries from a CSC matrix into row-oriented form.
/// Returns a Vec indexed by row, each containing (col_index, value) pairs.
fn rows_from_csc(matrix: &CscMatrix<f64>) -> Vec<Vec<(usize, f64)>> {
    let mut rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); matrix.m];
    for col in 0..matrix.n {
        let start = matrix.colptr[col];
        let end = matrix.colptr[col + 1];
        for idx in start..end {
            rows[matrix.rowval[idx]].push((col, matrix.nzval[idx]));
        }
    }
    rows
}

#[cfg(test)]
#[allow(dead_code)]
impl LpSolver {
    /// Create a new LP solver from individual components
    pub(crate) fn new(
        cost: &[f64],
        a_eq: &CscMatrix<f64>,
        b_eq: &[f64],
        a_ub: &CscMatrix<f64>,
        b_ub: &[f64],
    ) -> Result<Self> {
        let mut pb = highs::RowProblem::default();

        // Add variables with cost coefficients and non-negativity bounds
        let vars: Vec<highs::Col> = cost.iter().map(|&c| pb.add_column(c, 0.0..)).collect();

        // Add equality constraints (A_eq * x = b_eq)
        let eq_rows = rows_from_csc(a_eq);
        for (row_idx, entries) in eq_rows.iter().enumerate() {
            let terms: Vec<(highs::Col, f64)> =
                entries.iter().map(|&(col, val)| (vars[col], val)).collect();
            pb.add_row(b_eq[row_idx]..=b_eq[row_idx], &terms);
        }

        // Add inequality constraints (A_ub * x <= b_ub)
        let ub_rows = rows_from_csc(a_ub);
        for (row_idx, entries) in ub_rows.iter().enumerate() {
            let terms: Vec<(highs::Col, f64)> =
                entries.iter().map(|&(col, val)| (vars[col], val)).collect();
            pb.add_row(..=b_ub[row_idx], &terms);
        }

        Ok(Self { pb })
    }

    /// Solve the LP problem
    pub(crate) fn solve(self) -> Result<LpSolution> {
        let solved = self.pb.optimise(highs::Sense::Minimise).solve();
        match solved.status() {
            highs::HighsModelStatus::Optimal => Ok(LpSolution {
                status: SolveStatus::Solved,
                objective_value: solved.objective_value(),
            }),
            highs::HighsModelStatus::Infeasible => Ok(LpSolution {
                status: SolveStatus::Infeasible,
                objective_value: 0.0,
            }),
            other => Err(ShapleyError::LpSolver(format!(
                "HiGHS solver failed: {:?}",
                other
            ))),
        }
    }
}

/// Solve result from the coalition solver.
pub(crate) struct CoalitionResult {
    pub status: SolveStatus,
    pub objective_value: f64,
}

/// Create and solve an LP for a specific coalition using pre-computed
/// row-oriented constraint data. Avoids rebuilding CSC matrices per coalition.
///
/// `coalition_mask` has bit i set for each operator i in the coalition,
/// plus `ALWAYS_BIT` so that Public/Private/empty operators always match.
///
/// This is the *fresh-build* coalition solve: it constructs a new `RowProblem`
/// holding only the coalition's active columns (subset/remapped via `col_remap`)
/// and kept ub-rows (`keep_rows`), then re-runs presolve. It backs the sampling
/// path — which solves a dedup'd mask set in arbitrary `par_iter` order, so there
/// is no basis locality to exploit — and serves as the parity oracle for the
/// warm-start path.
///
/// The exact path ([`crate::shapley::Shapley::compute`]) instead uses
/// [`WarmCoalitionSolver`]. Earlier this warm-start was thought infeasible because
/// each coalition's column/row *set* differs. But gating is purely operator-based,
/// so a coalition is just the SAME full-size model with the excluded operators'
/// columns pinned to `[0, 0]` — identical variable/constraint structure across
/// coalitions. HiGHS dual simplex then warm-starts from the retained basis
/// (presolve off) instead of paying a cold presolve per solve. Equivalence with
/// this fresh-build objective is asserted per-coalition by the `warm_matches_fresh`
/// test, so the speedup never costs accuracy.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_coalition(
    primitives: &LpPrimitives,
    precomputed: &PrecomputedRows,
    buffers: &mut CoalitionBuffers,
    coalition_mask: u32,
    col_op1_mask: &[u32],
    col_op2_mask: &[u32],
    row_op1_mask: &[u32],
    row_op2_mask: &[u32],
) -> Result<CoalitionResult> {
    let n_cols = col_op1_mask.len();

    buffers.reset();

    // Ensure col_remap is large enough (may grow between calls if n_cols changes)
    if buffers.col_remap.len() < n_cols {
        buffers.col_remap.resize(n_cols, usize::MAX);
    }

    // Step 1: Compute keep_cols and build a remap array
    let mut new_col = 0usize;

    for i in 0..n_cols {
        if (col_op1_mask[i] & coalition_mask) != 0 && (col_op2_mask[i] & coalition_mask) != 0 {
            buffers.col_remap[i] = new_col;
            buffers.cost.push(primitives.cost[i]);
            new_col += 1;
        }
    }

    if new_col == 0 {
        return Err(ShapleyError::MatrixConstructionError(
            "No columns selected for coalition".to_string(),
        ));
    }

    // Step 2: Compute keep_rows for A_ub
    for i in 0..row_op1_mask.len() {
        if (row_op1_mask[i] & coalition_mask) != 0 && (row_op2_mask[i] & coalition_mask) != 0 {
            buffers.keep_rows.push(i);
        }
    }

    // ── Build the HiGHS model directly from the precomputed rows ─────────
    // No intermediate TriMatI/CSR: feed `add_row` straight from
    // precomputed.eq_rows + the kept ub_rows, remapping each original column
    // to the coalition's kept column. Output-identical to the old CSR path
    // (column subsetting and an empty "0 (cmp) rhs" row are LP-equivalent),
    // but skips a full triplet build + sort + read-back on every solve.
    let mut pb = highs::RowProblem::default();

    // Variables: cost coefficients, non-negative bounds [0, ∞).
    let vars: Vec<highs::Col> = buffers
        .cost
        .iter()
        .map(|&c| pb.add_column(c, 0.0..))
        .collect();

    // Per-row scratch, reused across rows to avoid a per-row allocation.
    let mut entries: Vec<(highs::Col, f64)> = Vec::with_capacity(64);

    // Equality constraints — all rows, remapped (rhs ..= rhs).
    for (row_idx, row_entries) in precomputed.eq_rows.iter().enumerate() {
        entries.clear();
        for &(old_col, val) in row_entries {
            let nc = buffers.col_remap[old_col];
            if nc != usize::MAX {
                entries.push((vars[nc], val));
            }
        }
        let b = primitives.b_eq[row_idx];
        pb.add_row(b..=b, &entries);
    }

    // Inequality constraints — only kept rows, remapped (..= rhs).
    for &row_idx in &buffers.keep_rows {
        entries.clear();
        for &(old_col, val) in &precomputed.ub_rows[row_idx] {
            let nc = buffers.col_remap[old_col];
            if nc != usize::MAX {
                entries.push((vars[nc], val));
            }
        }
        pb.add_row(..=primitives.b_ub[row_idx], &entries);
    }

    // Solve — tune HiGHS for repeated medium LPs in a rayon pool:
    //  - threads=1: avoid contention with rayon (1 LP per rayon thread)
    //  - simplex_strategy=1: dual serial simplex (fastest for sparse medium LPs)
    //  - presolve=on: reduces ~1148-row constraint matrices significantly
    let mut model = pb
        .try_optimise(highs::Sense::Minimise)
        .map_err(|_| ShapleyError::LpSolver("HiGHS model construction failed".to_string()))?;
    model.set_option("threads", 1_i32);
    model.set_option("simplex_strategy", 1_i32);
    model.set_option("presolve", "on");
    let solved = model.solve();

    match solved.status() {
        highs::HighsModelStatus::Optimal => Ok(CoalitionResult {
            status: SolveStatus::Solved,
            objective_value: solved.objective_value(),
        }),
        highs::HighsModelStatus::Infeasible | highs::HighsModelStatus::UnboundedOrInfeasible => {
            Ok(CoalitionResult {
                status: SolveStatus::Infeasible,
                objective_value: 0.0,
            })
        }
        other => Err(ShapleyError::LpSolver(format!(
            "HiGHS solver failed: {:?}",
            other
        ))),
    }
}

/// Full-size, coalition-independent description of the LP, built once per
/// `compute()` and shared (via `Arc`) across rayon workers. Each worker turns it
/// into its own persistent [`highs::Model`] for warm-started re-solving.
///
/// Unlike the fresh-build path, columns are NOT subset per coalition: the model
/// always contains every column and every row (eq + ub) at their original
/// indices. A coalition is expressed purely by pinning excluded columns to
/// `[0, 0]` (see [`WarmCoalitionSolver::solve`]).
pub(crate) struct LpTemplate {
    cost: Vec<f64>,
    eq_rows: Vec<Vec<(usize, f64)>>,
    ub_rows: Vec<Vec<(usize, f64)>>,
    b_eq: Vec<f64>,
    b_ub: Vec<f64>,
}

impl LpTemplate {
    /// Build the template from the full (unfiltered) LP primitives.
    pub(crate) fn from_primitives(primitives: &LpPrimitives) -> Self {
        Self {
            cost: primitives.cost.clone(),
            eq_rows: rows_from_csc(&primitives.a_eq),
            ub_rows: rows_from_csc(&primitives.a_ub),
            b_eq: primitives.b_eq.clone(),
            b_ub: primitives.b_ub.clone(),
        }
    }

    /// Construct a fresh HiGHS model holding every column (bounds `[0, ∞)`) and
    /// every constraint, tuned for warm-started re-solving. Returns the model and
    /// the column handles (indexed identically to `cost` / the mask vectors).
    fn build_model(&self) -> Result<(highs::Model, Vec<highs::Col>)> {
        let mut pb = highs::RowProblem::default();

        let vars: Vec<highs::Col> = self.cost.iter().map(|&c| pb.add_column(c, 0.0..)).collect();

        let mut entries: Vec<(highs::Col, f64)> = Vec::with_capacity(64);

        // Equality (flow-conservation) rows — all kept, full column set.
        for (row_idx, row) in self.eq_rows.iter().enumerate() {
            entries.clear();
            for &(col, val) in row {
                entries.push((vars[col], val));
            }
            let b = self.b_eq[row_idx];
            pb.add_row(b..=b, &entries);
        }

        // Inequality (bandwidth) rows — all kept, full column set. Rows touching
        // only excluded (pinned-to-zero) columns collapse to `0 <= rhs`, so they
        // never need per-coalition mutation.
        for (row_idx, row) in self.ub_rows.iter().enumerate() {
            entries.clear();
            for &(col, val) in row {
                entries.push((vars[col], val));
            }
            pb.add_row(..=self.b_ub[row_idx], &entries);
        }

        let mut model = pb.try_optimise(highs::Sense::Minimise).map_err(|_| {
            ShapleyError::LpSolver("HiGHS warm model construction failed".to_string())
        })?;
        // Warm-start tuning:
        //  - solver=simplex + simplex_strategy=1 (dual): dual simplex restores
        //    optimality fastest after a variable-bound change.
        //  - presolve=off: presolve would discard the basis between runs, which is
        //    exactly what we want to retain.
        //  - threads=1: one LP per rayon worker, no nested parallelism.
        model.set_option("threads", 1_i32);
        model.set_option("solver", "simplex");
        model.set_option("simplex_strategy", 1_i32);
        model.set_option("presolve", "off");
        model.make_quiet();
        Ok((model, vars))
    }
}

/// A persistent, per-thread HiGHS model that solves successive coalitions by
/// toggling column bounds and re-solving from the retained simplex basis, instead
/// of rebuilding the model and re-running presolve per coalition.
///
/// `epoch` identifies the `compute()` call this model was built for: a thread-local
/// solver from a previous (different) problem must be rebuilt rather than reused.
pub(crate) struct WarmCoalitionSolver {
    epoch: u64,
    template: Arc<LpTemplate>,
    /// `Some` between solves; momentarily `None` while a solve is in flight (the
    /// HiGHS `Model` is consumed by `try_solve` and handed back afterwards). Left
    /// `None` after a hard solver error so the next call rebuilds from `template`.
    model: Option<highs::Model>,
    vars: Vec<highs::Col>,
    /// Current active/excluded state of each column, so we only issue
    /// `change_column_bounds` for columns that actually flip.
    col_active: Vec<bool>,
}

impl WarmCoalitionSolver {
    /// Build a solver for the given problem `epoch` from a shared template.
    pub(crate) fn from_template(epoch: u64, template: Arc<LpTemplate>) -> Result<Self> {
        let (model, vars) = template.build_model()?;
        let col_active = vec![true; vars.len()];
        Ok(Self {
            epoch,
            template,
            model: Some(model),
            vars,
            col_active,
        })
    }

    /// The `compute()` epoch this solver was built for.
    pub(crate) fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Solve one coalition. A column is active iff BOTH its endpoint operators are
    /// in `coalition_mask` (identical gate to [`solve_coalition`]); excluded
    /// columns are pinned to `[0, 0]`. Rows are never mutated — see [`LpTemplate`].
    pub(crate) fn solve(
        &mut self,
        coalition_mask: u32,
        col_op1_mask: &[u32],
        col_op2_mask: &[u32],
    ) -> Result<CoalitionResult> {
        // Recover from a prior hard solver error by rebuilding the model.
        if self.model.is_none() {
            let (model, vars) = self.template.build_model()?;
            self.vars = vars;
            self.col_active = vec![true; self.vars.len()];
            self.model = Some(model);
        }

        let model = self.model.as_mut().expect("warm model present");

        let n = self.vars.len();
        let mut active_count = 0usize;
        for i in 0..n {
            let active =
                (col_op1_mask[i] & coalition_mask) != 0 && (col_op2_mask[i] & coalition_mask) != 0;
            if active {
                active_count += 1;
            }
            if active != self.col_active[i] {
                if active {
                    model.change_column_bounds(self.vars[i], 0.0..);
                } else {
                    model.change_column_bounds(self.vars[i], 0.0..=0.0);
                }
                self.col_active[i] = active;
            }
        }

        // No active columns ⇒ empty sub-LP, matching `solve_coalition`'s
        // "No columns selected" early return (mapped to `None` by the caller).
        if active_count == 0 {
            return Ok(CoalitionResult {
                status: SolveStatus::Infeasible,
                objective_value: 0.0,
            });
        }

        // Warm re-solve on the same HiGHS object: dual simplex restarts from the
        // retained basis (presolve is off). `try_solve` consumes the model; on
        // success we take it back via `From<SolvedModel>` for the next coalition.
        let model = self.model.take().expect("warm model present");
        match model.try_solve() {
            Ok(solved) => {
                let status = solved.status();
                let objective_value = solved.objective_value();
                self.model = Some(highs::Model::from(solved));
                match status {
                    highs::HighsModelStatus::Optimal => Ok(CoalitionResult {
                        status: SolveStatus::Solved,
                        objective_value,
                    }),
                    highs::HighsModelStatus::Infeasible
                    | highs::HighsModelStatus::UnboundedOrInfeasible => Ok(CoalitionResult {
                        status: SolveStatus::Infeasible,
                        objective_value: 0.0,
                    }),
                    other => Err(ShapleyError::LpSolver(format!(
                        "HiGHS solver failed: {:?}",
                        other
                    ))),
                }
            }
            // `try_solve` only errors on a structurally invalid model, which should
            // never happen here. Leave `model` None so the next call rebuilds.
            Err(_) => Err(ShapleyError::LpSolver(
                "HiGHS warm solve failed".to_string(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        consolidation::{consolidate_demand, consolidate_links},
        lp_builder::LpBuilderInput,
        types::{ConsolidatedDemand, ConsolidatedLink, Demand, Device, PrivateLink, PublicLink},
    };

    /// Same sentinel as `shapley::ALWAYS_BIT`: bit for always-included operators.
    const TEST_ALWAYS_BIT: u32 = 1 << 31;

    /// A small 3-operator network: three parallel SRC→DST links owned by A, B, C
    /// with different latencies and bandwidths, plus one demand that needs more
    /// capacity than the cheapest link alone. This yields coalitions that are
    /// infeasible (no/insufficient capacity) and feasible with distinct optima —
    /// exactly the variety the warm/fresh equivalence check should cover.
    fn three_op_network() -> (Vec<ConsolidatedLink>, Vec<ConsolidatedDemand>) {
        let link = |op: &str, latency: f64, bandwidth: f64, shared: u32| ConsolidatedLink {
            device1: "SRC".to_string(),
            device2: "DST".to_string(),
            latency,
            bandwidth,
            operator1: op.to_string(),
            operator2: op.to_string(),
            shared,
            link_type: 0,
        };
        let links = vec![
            link("A", 10.0, 10.0, 1),
            link("B", 20.0, 10.0, 2),
            link("C", 5.0, 3.0, 3), // cheapest but too small to carry the demand alone
        ];
        let demands = vec![ConsolidatedDemand {
            start: "SRC".to_string(),
            end: "DST".to_string(),
            receivers: 1,
            traffic: 5.0,
            priority: 1.0,
            kind: 1,
            multicast: false,
            original: 1,
        }];
        (links, demands)
    }

    /// Map an operator string to its coalition bit, mirroring `shapley.rs`.
    fn operator_bit(op: &str, ordered_ops: &[&str]) -> u32 {
        if op == "Public" || op == "Private" || op.is_empty() {
            TEST_ALWAYS_BIT
        } else {
            ordered_ops
                .iter()
                .position(|o| *o == op)
                .map(|idx| 1u32 << idx)
                .unwrap_or(0)
        }
    }

    /// The parity safety net: over *every* coalition of `links`/`demands`, the
    /// warm-start solver must produce the identical coalition value as the
    /// fresh-build `solve_coalition`. If any structural assumption behind
    /// warm-start (e.g. a bandwidth or multicast row touching a column outside its
    /// operators) were violated, this would fail.
    fn assert_warm_matches_fresh(links: &[ConsolidatedLink], demands: &[ConsolidatedDemand]) {
        let primitives = LpBuilderInput::new(links, demands)
            .build()
            .expect("LP build should succeed");
        let precomputed = PrecomputedRows::new(&primitives);

        // Derive operator order exactly as `compute_inner` does: sorted unique
        // operators across all link endpoints, excluding the always-in sentinels.
        let ordered_ops: Vec<String> = links
            .iter()
            .flat_map(|l| [l.operator1.clone(), l.operator2.clone()])
            .filter(|o| o != "Public" && o != "Private" && !o.is_empty())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        let ops: Vec<&str> = ordered_ops.iter().map(String::as_str).collect();
        let n_ops = ops.len();
        assert!(
            (1..=20).contains(&n_ops),
            "unexpected operator count {n_ops}"
        );

        let mask_of =
            |tags: &[String]| -> Vec<u32> { tags.iter().map(|s| operator_bit(s, &ops)).collect() };
        let col_op1_mask = mask_of(&primitives.col_op1);
        let col_op2_mask = mask_of(&primitives.col_op2);
        let row_op1_mask = mask_of(&primitives.row_op1);
        let row_op2_mask = mask_of(&primitives.row_op2);

        let template = Arc::new(LpTemplate::from_primitives(&primitives));
        let mut warm = WarmCoalitionSolver::from_template(1, template).expect("warm build");
        let mut buf = CoalitionBuffers::new(primitives.cost.len());

        // Map a solve result to the coalition value exactly as `compute_inner` does.
        let to_value = |r: Result<CoalitionResult>| -> Option<f64> {
            r.ok().and_then(|res| match res.status {
                SolveStatus::Solved => Some(-res.objective_value),
                SolveStatus::Infeasible => None,
            })
        };

        let mut feasible_seen = 0;
        for coalition_idx in 0..(1u32 << n_ops) {
            let mask = coalition_idx | TEST_ALWAYS_BIT;

            let fresh = to_value(solve_coalition(
                &primitives,
                &precomputed,
                &mut buf,
                mask,
                &col_op1_mask,
                &col_op2_mask,
                &row_op1_mask,
                &row_op2_mask,
            ));
            let warmed = to_value(warm.solve(mask, &col_op1_mask, &col_op2_mask));

            match (fresh, warmed) {
                (Some(f), Some(w)) => {
                    feasible_seen += 1;
                    assert!(
                        (f - w).abs() < 1e-9,
                        "coalition {coalition_idx:b}: fresh={f} warm={w}"
                    );
                }
                (None, None) => {}
                (f, w) => panic!(
                    "coalition {coalition_idx:b}: feasibility differs fresh={f:?} warm={w:?}"
                ),
            }
        }

        assert!(
            feasible_seen > 0,
            "fixture produced no feasible coalitions — test is vacuous"
        );
    }

    /// Parity over a unicast network with shared-bandwidth rows.
    #[test]
    fn warm_matches_fresh() {
        let (links, demands) = three_op_network();
        assert_warm_matches_fresh(&links, &demands);
    }

    /// Parity over a MULTICAST network — exercises the within-group constraint rows
    /// and multicast auxiliary columns, the trickiest case for the "excluded
    /// columns zero out their rows" argument. Built from the known-valid
    /// `simple_test` fixture and consolidated exactly as `compute()` does.
    #[test]
    fn warm_matches_fresh_multicast() {
        let private_links = vec![
            PrivateLink::new("SIN1".into(), "FRA1".into(), 50.0, 10.0, 1.0, None),
            PrivateLink::new("FRA1".into(), "AMS1".into(), 3.0, 10.0, 1.0, None),
            PrivateLink::new("FRA1".into(), "LON1".into(), 5.0, 10.0, 1.0, None),
        ];
        let devices = vec![
            Device::new("SIN1".into(), 1, "Alpha".into()),
            Device::new("FRA1".into(), 1, "Alpha".into()),
            Device::new("AMS1".into(), 1, "Beta".into()),
            Device::new("LON1".into(), 1, "Beta".into()),
        ];
        let public_links = vec![
            PublicLink::new("SIN".into(), "FRA".into(), 100.0),
            PublicLink::new("SIN".into(), "AMS".into(), 102.0),
            PublicLink::new("FRA".into(), "LON".into(), 7.0),
            PublicLink::new("FRA".into(), "AMS".into(), 5.0),
        ];
        let demands = vec![
            Demand::new("SIN".into(), "AMS".into(), 1, 1.0, 1.0, 1, true),
            Demand::new("SIN".into(), "LON".into(), 5, 1.0, 2.0, 1, true),
            Demand::new("AMS".into(), "LON".into(), 2, 3.0, 1.0, 2, false),
            Demand::new("AMS".into(), "FRA".into(), 1, 3.0, 1.0, 2, false),
        ];

        let consolidated_demands = consolidate_demand(&demands, 1.0).expect("consolidate demand");
        let consolidated_links = consolidate_links(
            &private_links,
            &devices,
            &consolidated_demands,
            &public_links,
            5.0,
        )
        .expect("consolidate links");

        assert_warm_matches_fresh(&consolidated_links, &consolidated_demands);
    }

    /// Re-solving the same coalition twice (the warm path) must be idempotent.
    #[test]
    fn warm_repeated_solve_is_stable() {
        let (links, demands) = three_op_network();
        let primitives = LpBuilderInput::new(&links, &demands).build().unwrap();
        let ordered_ops = ["A", "B", "C"];
        let col_op1_mask: Vec<u32> = primitives
            .col_op1
            .iter()
            .map(|s| operator_bit(s, &ordered_ops))
            .collect();
        let col_op2_mask: Vec<u32> = primitives
            .col_op2
            .iter()
            .map(|s| operator_bit(s, &ordered_ops))
            .collect();

        let template = Arc::new(LpTemplate::from_primitives(&primitives));
        let mut warm = WarmCoalitionSolver::from_template(7, template).unwrap();
        assert_eq!(warm.epoch(), 7);

        // Grand coalition then back to a sub-coalition then grand again.
        let grand = (1u32 << ordered_ops.len()) - 1 | TEST_ALWAYS_BIT;
        let first = warm.solve(grand, &col_op1_mask, &col_op2_mask).unwrap();
        let _ = warm.solve(0b001 | TEST_ALWAYS_BIT, &col_op1_mask, &col_op2_mask);
        let again = warm.solve(grand, &col_op1_mask, &col_op2_mask).unwrap();
        assert_eq!(first.status, again.status);
        assert!((first.objective_value - again.objective_value).abs() < 1e-9);
    }

    fn simple_links() -> Vec<ConsolidatedLink> {
        vec![ConsolidatedLink {
            device1: "A".to_string(),
            device2: "B".to_string(),
            latency: 1.0,
            bandwidth: 10.0,
            operator1: "Op1".to_string(),
            operator2: "Op1".to_string(),
            shared: 1,
            link_type: 0,
        }]
    }

    fn simple_demands() -> Vec<ConsolidatedDemand> {
        vec![ConsolidatedDemand {
            start: "A".to_string(),
            end: "B".to_string(),
            receivers: 1,
            traffic: 5.0,
            priority: 1.0,
            kind: 1,
            multicast: false,
            original: 1,
        }]
    }

    #[test]
    fn test_solver_creation() {
        let links = simple_links();
        let demands = simple_demands();
        let lp_builder = LpBuilderInput::new(&links, &demands);
        let primitives = lp_builder.build().expect("LP builder should succeed");
        let solver = LpSolver::new(
            &primitives.cost,
            &primitives.a_eq,
            &primitives.b_eq,
            &primitives.a_ub,
            &primitives.b_ub,
        );
        assert!(solver.is_ok());
    }

    #[test]
    fn test_rows_from_csc() {
        // 2x3 matrix: [[1, 0, 2], [0, 3, 0]]
        let matrix = CscMatrix::new(
            2,
            3,
            vec![0, 1, 2, 3],    // colptr
            vec![0, 1, 0],       // rowval
            vec![1.0, 3.0, 2.0], // nzval
        );
        let rows = rows_from_csc(&matrix);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0], vec![(0, 1.0), (2, 2.0)]); // row 0: col 0 = 1, col 2 = 2
        assert_eq!(rows[1], vec![(1, 3.0)]); // row 1: col 1 = 3
    }

    #[test]
    fn test_rows_from_csc_empty() {
        let matrix = CscMatrix::new(3, 2, vec![0, 0, 0], vec![], vec![]);
        let rows = rows_from_csc(&matrix);
        assert_eq!(rows.len(), 3);
        assert!(rows[0].is_empty());
        assert!(rows[1].is_empty());
        assert!(rows[2].is_empty());
    }

    #[test]
    fn test_precomputed_rows() {
        let links = simple_links();
        let demands = simple_demands();
        let lp_builder = LpBuilderInput::new(&links, &demands);
        let primitives = lp_builder.build().expect("LP builder should succeed");
        let precomputed = PrecomputedRows::new(&primitives);

        // Should have rows matching the matrix dimensions
        assert_eq!(precomputed.eq_rows.len(), primitives.a_eq.m);
        assert_eq!(precomputed.ub_rows.len(), primitives.a_ub.m);
    }

    #[test]
    fn test_coalition_buffers_new_and_reset() {
        let mut buf = CoalitionBuffers::new(10);

        assert_eq!(buf.col_remap.len(), 10);
        assert!(buf.col_remap.iter().all(|&v| v == usize::MAX));
        assert!(buf.cost.is_empty());

        // Simulate use
        buf.col_remap[0] = 0;
        buf.col_remap[5] = 1;
        buf.cost.push(1.0);
        buf.cost.push(2.0);
        buf.keep_rows.push(0);

        // Reset should clear everything
        buf.reset();
        assert!(buf.col_remap.iter().all(|&v| v == usize::MAX));
        assert!(buf.cost.is_empty());
        assert!(buf.keep_rows.is_empty());

        // Capacity should be preserved (no reallocation)
        assert!(buf.cost.capacity() >= 10);
    }

    #[test]
    fn test_solve_coalition_empty_columns() {
        let links = simple_links();
        let demands = simple_demands();
        let lp_builder = LpBuilderInput::new(&links, &demands);
        let primitives = lp_builder.build().expect("LP builder should succeed");
        let precomputed = PrecomputedRows::new(&primitives);
        let mut buffers = CoalitionBuffers::new(primitives.cost.len());

        // Coalition mask 0 (no operators) — should fail with no columns
        let col_masks = vec![0u32; primitives.cost.len()];
        let row_masks = vec![0u32; primitives.b_ub.len()];

        let result = solve_coalition(
            &primitives,
            &precomputed,
            &mut buffers,
            0, // empty coalition
            &col_masks,
            &col_masks,
            &row_masks,
            &row_masks,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_solve_coalition_grand_coalition() {
        let links = simple_links();
        let demands = simple_demands();
        let lp_builder = LpBuilderInput::new(&links, &demands);
        let primitives = lp_builder.build().expect("LP builder should succeed");
        let precomputed = PrecomputedRows::new(&primitives);
        let mut buffers = CoalitionBuffers::new(primitives.cost.len());

        // All bits set — grand coalition, everything included
        let all_bits = u32::MAX;
        let col_masks = vec![all_bits; primitives.cost.len()];
        let row_masks = vec![all_bits; primitives.b_ub.len()];

        let result = solve_coalition(
            &primitives,
            &precomputed,
            &mut buffers,
            all_bits,
            &col_masks,
            &col_masks,
            &row_masks,
            &row_masks,
        );

        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.status, SolveStatus::Solved);
        // Objective should be finite and non-zero for a feasible problem
        assert!(result.objective_value.is_finite());
    }
}
