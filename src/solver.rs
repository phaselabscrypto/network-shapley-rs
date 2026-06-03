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
/// B2 (HiGHS warm-start) — investigated and intentionally NOT done. Each call
/// builds a FRESH `RowProblem` whose column set+count (subset/remapped via
/// `col_remap`) and ub-row set (`keep_rows`) are coalition-specific, so a HiGHS
/// basis/dual warm-start — valid only when variable + constraint structure match
/// the prior solve — never applies. The sampler also solves a dedup'd mask set in
/// arbitrary `par_iter` order, so adjacent solves on a worker are unrelated
/// coalitions with no basis locality. The thread-local `CoalitionBuffers` already
/// amortize the reusable allocations, and `presolve=on` (where the time goes) is
/// re-run per call. A real warm-start would need a fixed full-size model with
/// per-coalition bound toggling — a large rewrite risking changed objectives —
/// for no expected gain. Fewer LPs (B3 selective reuse) is the better lever.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        lp_builder::LpBuilderInput,
        types::{ConsolidatedDemand, ConsolidatedLink},
    };

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
