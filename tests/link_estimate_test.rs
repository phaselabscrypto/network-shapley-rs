//! Parity test: `ShapleyInput::network_link_estimate` vs the Python
//! `network_linkestimate` reference, plus link-estimate edge cases.
//!
//! Requires Python 3 with pandas + scipy and the `network-shapley` repo accessible
//! (see tests/python_parity.py). Run with:
//!   cargo test --features serde --test link_estimate_test
//! The parity test SKIPs (passes) when Python or its deps are unavailable.

use std::{collections::BTreeMap, fs::File, process::Command};

use network_shapley::{
    error::ShapleyError,
    link_estimate::LinkEstimate,
    shapley::ShapleyInput,
    types::{Device, PrivateLink},
};

/// Per-link Shapley `value` is a raw LP objective delta; match Python within the
/// same absolute tolerance the operator-level parity test uses.
const VALUE_TOLERANCE: f64 = 0.01;
/// `percent` is a 0–1 share; tighter tolerance.
const PERCENT_TOLERANCE: f64 = 1e-4;

fn read_csv<T: serde::de::DeserializeOwned>(path: &str) -> Vec<T> {
    let file = File::open(path).unwrap_or_else(|e| panic!("Failed to open {path}: {e}"));
    csv::Reader::from_reader(file)
        .deserialize()
        .map(|r| r.unwrap())
        .collect()
}

fn fixture_input(demand_file: &str, operator_uptime: f64, multiplier: f64) -> ShapleyInput {
    ShapleyInput {
        private_links: read_csv("tests/private_links.csv"),
        devices: read_csv("tests/devices.csv"),
        demands: read_csv(demand_file),
        public_links: read_csv("tests/public_links.csv"),
        operator_uptime,
        contiguity_bonus: 5.0,
        demand_multiplier: multiplier,
    }
}

#[derive(serde::Deserialize, Debug)]
struct PyLink {
    device1: String,
    device2: String,
    value: f64,
    percent: f64,
}

/// Runs `python_parity.py link-estimate`; returns None (SKIP) if Python/deps absent.
fn run_python_link_estimate() -> Option<BTreeMap<String, Vec<PyLink>>> {
    let output = match Command::new("python3")
        .args(["tests/python_parity.py", "link-estimate"])
        .output()
    {
        Ok(o) => o,
        Err(_) => {
            eprintln!("SKIP: python3 not found");
            return None;
        }
    };

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if stderr.contains("ModuleNotFoundError") {
            eprintln!("SKIP: Python deps not installed (pandas/scipy)");
            return None;
        }
        panic!("Python link-estimate script failed:\n{stderr}");
    }

    let stdout = String::from_utf8(output.stdout).expect("Invalid UTF-8 from Python");
    Some(serde_json::from_str(&stdout).expect("Failed to parse Python JSON output"))
}

/// Index link rows by the canonical (device1, device2) pair for order-independent
/// comparison (both sides emit one row per link in `device1 < device2` form).
fn rust_by_pair(rows: &[LinkEstimate]) -> BTreeMap<(String, String), &LinkEstimate> {
    rows.iter()
        .map(|r| ((r.device1.clone(), r.device2.clone()), r))
        .collect()
}

fn compare(scenario: &str, rust: &[LinkEstimate], python: &[PyLink]) {
    assert_eq!(
        rust.len(),
        python.len(),
        "{scenario}: link count mismatch: Rust={}, Python={}",
        rust.len(),
        python.len()
    );

    let by_pair = rust_by_pair(rust);
    for py in python {
        let key = (py.device1.clone(), py.device2.clone());
        let rv = by_pair
            .get(&key)
            .unwrap_or_else(|| panic!("{scenario}: link {key:?} missing from Rust output"));

        let vdiff = (rv.value - py.value).abs();
        assert!(
            vdiff < VALUE_TOLERANCE,
            "{scenario}: value mismatch for {key:?}: Rust={:.6}, Python={:.6}, diff={vdiff:.6}",
            rv.value,
            py.value,
        );
        let pdiff = (rv.percent - py.percent).abs();
        assert!(
            pdiff < PERCENT_TOLERANCE,
            "{scenario}: percent mismatch for {key:?}: Rust={:.6}, Python={:.6}, diff={pdiff:.6}",
            rv.percent,
            py.percent,
        );
    }
    eprintln!("{scenario}: PASS ({} links)", rust.len());
}

#[test]
fn test_link_estimate_python_parity() {
    let python = match run_python_link_estimate() {
        Some(p) => p,
        None => {
            eprintln!("Skipping link-estimate parity test — Python or deps not available");
            return;
        }
    };

    let scenarios = [
        ("linkest_demand1_Alpha_1x", "tests/demand1.csv", "Alpha"),
        ("linkest_demand2_Alpha_1x", "tests/demand2.csv", "Alpha"),
        ("linkest_demand1_Theta_1x", "tests/demand1.csv", "Theta"),
    ];

    for (key, demand_file, focus) in scenarios {
        let rust = fixture_input(demand_file, 0.98, 1.0)
            .network_link_estimate(focus)
            .expect("network_link_estimate failed");
        let py = python
            .get(key)
            .unwrap_or_else(|| panic!("Missing {key} from Python output"));
        compare(key, &rust, py);
    }
}

/// `network_link_estimate` forces `operator_uptime = 1.0` internally (only the
/// per-link `Uptime` penalty applies). The `ShapleyInput.operator_uptime` field
/// must therefore have NO effect on the result. If it were honoured, 0.98 would
/// trigger the coalition-level expectation pass and diverge wildly — so a tight
/// tolerance cleanly proves it is ignored.
#[test]
fn test_link_estimate_ignores_operator_uptime() {
    let a = fixture_input("tests/demand1.csv", 0.98, 1.0)
        .network_link_estimate("Alpha")
        .unwrap();
    let b = fixture_input("tests/demand1.csv", 1.0, 1.0)
        .network_link_estimate("Alpha")
        .unwrap();

    assert_eq!(a.len(), b.len(), "link count differs across uptime inputs");
    let bb = rust_by_pair(&b);
    for ra in &a {
        let rb = bb
            .get(&(ra.device1.clone(), ra.device2.clone()))
            .expect("link present in both");
        // Warm-start values are FP-stable, not bit-identical run-to-run, so use a
        // small tolerance (≪ any uptime-driven divergence).
        assert!(
            (ra.value - rb.value).abs() < 1e-6,
            "uptime changed value for {}-{}: {} vs {}",
            ra.device1,
            ra.device2,
            ra.value,
            rb.value
        );
    }
}

/// A focus operator that owns no links yields an empty result (Python drops all
/// rows after the retag filter).
#[test]
fn test_link_estimate_unknown_focus_is_empty() {
    let rows = fixture_input("tests/demand1.csv", 1.0, 1.0)
        .network_link_estimate("NoSuchOperator")
        .expect("should not error");
    assert!(rows.is_empty(), "expected no links, got {rows:?}");
}

/// More than 20 operators trips the `n_ops < 21` cap (mirrors Python's assert;
/// enforced via `check_inputs`). The post-retag player cap returns the identical
/// `TooManyOperators` error.
#[test]
fn test_link_estimate_too_many_operators() {
    let devices: Vec<Device> = (0..21)
        .map(|i| Device::new(format!("DEV{i}"), 10, format!("Op{i}")))
        .collect();

    let input = ShapleyInput {
        private_links: vec![PrivateLink::new(
            "DEV0".into(),
            "DEV1".into(),
            1.0,
            10.0,
            1.0,
            None,
        )],
        devices,
        demands: Vec::new(),
        public_links: Vec::new(),
        operator_uptime: 1.0,
        contiguity_bonus: 5.0,
        demand_multiplier: 1.0,
    };

    let err = input
        .network_link_estimate("Op0")
        .expect_err("21 operators must be rejected");
    assert!(
        matches!(err, ShapleyError::TooManyOperators { limit: 20, count } if count >= 21),
        "expected TooManyOperators{{limit:20}}, got {err:?}"
    );
}
