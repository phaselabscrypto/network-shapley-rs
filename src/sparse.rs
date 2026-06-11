/// Local CSC (Compressed Sparse Column) matrix type, replacing clarabel::algebra::CscMatrix.
///
/// Fields match the Clarabel naming convention used throughout the codebase:
/// `m` (rows), `n` (cols), `colptr`, `rowval`, `nzval`.
#[derive(Debug, Clone)]
pub(crate) struct CscMatrix<T = f64> {
    /// Number of rows.
    pub m: usize,
    /// Number of columns.
    pub n: usize,
    /// Column pointers (length `n + 1`).
    pub colptr: Vec<usize>,
    /// Row indices of non-zero entries.
    pub rowval: Vec<usize>,
    /// Non-zero values.
    pub nzval: Vec<T>,
}

impl<T: Clone> CscMatrix<T> {
    pub fn new(m: usize, n: usize, colptr: Vec<usize>, rowval: Vec<usize>, nzval: Vec<T>) -> Self {
        debug_assert_eq!(colptr.len(), n + 1, "colptr length must be n + 1");
        debug_assert_eq!(
            rowval.len(),
            nzval.len(),
            "rowval and nzval must have equal length"
        );
        Self {
            m,
            n,
            colptr,
            rowval,
            nzval,
        }
    }

    /// Number of structural non-zeros.
    pub fn nnz(&self) -> usize {
        self.nzval.len()
    }
}

use crate::error::Result;

/// Build a CSC matrix from (row, col, value) triplets, sorting in place.
///
/// The triplets slice is sorted by (col, row) as a side effect.
pub(crate) fn from_triplets(
    triplets: &mut [(usize, usize, f64)],
    n_rows: usize,
    n_cols: usize,
) -> Result<CscMatrix<f64>> {
    if triplets.is_empty() {
        return Ok(CscMatrix::new(
            n_rows,
            n_cols,
            vec![0; n_cols + 1],
            vec![],
            vec![],
        ));
    }

    triplets.sort_unstable_by_key(|&(r, c, _)| (c, r));

    let mut col_ptr = vec![0];
    let mut row_ind = Vec::with_capacity(triplets.len());
    let mut values = Vec::with_capacity(triplets.len());

    let mut current_col = 0;

    for &(row, col, val) in triplets.iter() {
        while current_col < col {
            col_ptr.push(row_ind.len());
            current_col += 1;
        }

        row_ind.push(row);
        values.push(val);
    }

    while current_col < n_cols {
        col_ptr.push(row_ind.len());
        current_col += 1;
    }

    Ok(CscMatrix::new(n_rows, n_cols, col_ptr, row_ind, values))
}

/// Construct a dense-to-CSC matrix from a 2D array reference (test helper).
/// Mirrors `CscMatrix::<f64>::from(&[[1.0, 2.0], [3.0, 4.0]])` from Clarabel.
impl<const COLS: usize, const ROWS: usize> From<&[[f64; COLS]; ROWS]> for CscMatrix<f64> {
    fn from(rows: &[[f64; COLS]; ROWS]) -> Self {
        let m = ROWS;
        let n = COLS;
        let mut colptr = Vec::with_capacity(n + 1);
        let mut rowval = Vec::new();
        let mut nzval = Vec::new();

        colptr.push(0);
        for col in 0..n {
            for (row, row_data) in rows.iter().enumerate() {
                let val = row_data[col];
                if val != 0.0 {
                    rowval.push(row);
                    nzval.push(val);
                }
            }
            colptr.push(rowval.len());
        }

        Self {
            m,
            n,
            colptr,
            rowval,
            nzval,
        }
    }
}
