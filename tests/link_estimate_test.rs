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
    shapley::{ComputeControl, ShapleyInput},
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
    bandwidth: f64,
    latency: f64,
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

/// Sort rows by (device pair, bandwidth, latency) for positional comparison.
/// Parallel links — same device pair, different bandwidth/latency — are LEGAL
/// input (only exact (pair, bandwidth, latency) duplicates are rejected), so a
/// pair-only map key would silently collapse them. Cross-language float bit-keys
/// would be fragile (pandas and Rust compute the uptime-adjusted bandwidth
/// independently), so we order by value and compare fields within tolerances.
fn sorted_rust(rows: &[LinkEstimate]) -> Vec<&LinkEstimate> {
    let mut v: Vec<&LinkEstimate> = rows.iter().collect();
    v.sort_by(|a, b| {
        (a.device1.as_str(), a.device2.as_str())
            .cmp(&(b.device1.as_str(), b.device2.as_str()))
            .then(a.bandwidth.total_cmp(&b.bandwidth))
            .then(a.latency.total_cmp(&b.latency))
    });
    v
}

fn sorted_python(rows: &[PyLink]) -> Vec<&PyLink> {
    let mut v: Vec<&PyLink> = rows.iter().collect();
    v.sort_by(|a, b| {
        (a.device1.as_str(), a.device2.as_str())
            .cmp(&(b.device1.as_str(), b.device2.as_str()))
            .then(a.bandwidth.total_cmp(&b.bandwidth))
            .then(a.latency.total_cmp(&b.latency))
    });
    v
}

fn compare(scenario: &str, rust: &[LinkEstimate], python: &[PyLink]) {
    assert_eq!(
        rust.len(),
        python.len(),
        "{scenario}: link count mismatch: Rust={}, Python={}",
        rust.len(),
        python.len()
    );

    for (rv, py) in sorted_rust(rust).into_iter().zip(sorted_python(python)) {
        let key = (py.device1.as_str(), py.device2.as_str());
        assert_eq!(
            (rv.device1.as_str(), rv.device2.as_str()),
            key,
            "{scenario}: row pairing mismatch after sort",
        );
        // Bandwidth/latency are part of the row identity (parallel links) — the
        // uptime-adjusted bandwidth is computed independently on each side, so
        // assert closeness, which also pins the consolidation parity.
        assert!(
            (rv.bandwidth - py.bandwidth).abs() < 1e-6,
            "{scenario}: bandwidth mismatch for {key:?}: Rust={}, Python={}",
            rv.bandwidth,
            py.bandwidth,
        );
        assert!(
            (rv.latency - py.latency).abs() < 1e-9,
            "{scenario}: latency mismatch for {key:?}: Rust={}, Python={}",
            rv.latency,
            py.latency,
        );

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
    for (ra, rb) in sorted_rust(&a).into_iter().zip(sorted_rust(&b)) {
        assert_eq!(
            (ra.device1.as_str(), ra.device2.as_str()),
            (rb.device1.as_str(), rb.device2.as_str()),
            "row pairing mismatch after sort",
        );
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

/// Minimal two-operator input for exercising the link-estimate pre-checks
/// (which run BEFORE `check_inputs`, so demands/public links can stay empty).
fn precheck_input(private_links: Vec<PrivateLink>) -> ShapleyInput {
    ShapleyInput {
        private_links,
        devices: vec![
            Device::new("AAA1".into(), 10, "Alpha".into()),
            Device::new("BBB1".into(), 10, "Alpha".into()),
            Device::new("CCC1".into(), 10, "Beta".into()),
        ],
        demands: Vec::new(),
        public_links: Vec::new(),
        operator_uptime: 1.0,
        contiguity_bonus: 5.0,
        demand_multiplier: 1.0,
    }
}

fn expect_validation(input: ShapleyInput, focus: &str, needle: &str) {
    let err = input
        .network_link_estimate(focus)
        .expect_err("pre-check must reject this input");
    assert!(
        matches!(&err, ShapleyError::Validation(msg) if msg.contains(needle)),
        "expected Validation containing {needle:?}, got {err:?}"
    );
}

/// Python pre-check parity (network_linkestimate.py:97-101): a shared-group id
/// appearing on more than one focus-owned link is rejected.
#[test]
fn test_shared_group_on_focus_links_rejected() {
    let input = precheck_input(vec![
        PrivateLink::new("AAA1".into(), "BBB1".into(), 1.0, 10.0, 1.0, Some(7)),
        PrivateLink::new("AAA1".into(), "CCC1".into(), 2.0, 10.0, 1.0, Some(7)),
    ]);
    expect_validation(input, "Alpha", "Shared groups");
}

/// Python pre-check parity (network_linkestimate.py:103-106): identical
/// (unordered device pair, bandwidth, latency) links are rejected.
#[test]
fn test_duplicate_links_rejected() {
    let input = precheck_input(vec![
        PrivateLink::new("AAA1".into(), "BBB1".into(), 1.0, 10.0, 1.0, None),
        PrivateLink::new("BBB1".into(), "AAA1".into(), 1.0, 10.0, 1.0, None),
    ]);
    expect_validation(input, "Alpha", "Duplicate links");
}

/// Signed-zero regression: pandas value-equality treats +0.0 and -0.0 bandwidth
/// as duplicates, and so must the Rust key — otherwise the pair slips past the
/// guard and confuses the `==`-based symmetric-pair match in retag_links.
#[test]
fn test_signed_zero_duplicate_links_rejected() {
    let input = precheck_input(vec![
        PrivateLink::new("AAA1".into(), "BBB1".into(), 1.0, 0.0, 1.0, None),
        PrivateLink::new("AAA1".into(), "BBB1".into(), 1.0, -0.0, 1.0, None),
    ]);
    expect_validation(input, "Alpha", "Duplicate links");
}

/// A NaN latency makes the symmetric reverse row unfindable (`NaN != NaN`).
/// Python raises IndexError there; the Rust port must fail loudly too, not
/// silently half-tag the pair and return plausible-looking wrong values.
/// (Only reachable via the direct crate API — serde_json/CSV reject NaN.)
#[test]
fn test_nan_latency_fails_loudly_instead_of_half_tagging() {
    let mut input = fixture_input("tests/demand1.csv", 1.0, 1.0);
    let alpha_link = input
        .private_links
        .iter_mut()
        .find(|l| l.device1 == "AMS1" || l.device2 == "AMS1")
        .expect("fixture has an Alpha-owned link");
    alpha_link.latency = f64::NAN;

    let err = input
        .network_link_estimate("Alpha")
        .expect_err("missing symmetric edge must be a loud error");
    assert!(
        matches!(&err, ShapleyError::Validation(msg) if msg.contains("no symmetric reverse link")),
        "expected missing-sym Validation, got {err:?}"
    );
}

/// The cancellable variant must produce the same values as the plain call and
/// drive the progress counters (denominator = 2^players, fully consumed).
#[test]
fn test_cancellable_matches_plain_and_reports_progress() {
    use std::sync::atomic::Ordering;

    let control = ComputeControl::default();
    let a = fixture_input("tests/demand1.csv", 1.0, 1.0)
        .network_link_estimate_cancellable("Alpha", &control)
        .expect("cancellable run failed");
    let b = fixture_input("tests/demand1.csv", 1.0, 1.0)
        .network_link_estimate("Alpha")
        .expect("plain run failed");

    assert_eq!(a.len(), b.len());
    for (ra, rb) in sorted_rust(&a).into_iter().zip(sorted_rust(&b)) {
        assert!(
            (ra.value - rb.value).abs() < 1e-6,
            "cancellable diverged for {}-{}: {} vs {}",
            ra.device1,
            ra.device2,
            ra.value,
            rb.value
        );
    }

    let total = control.progress.max_samples.load(Ordering::Relaxed);
    assert!(
        total > 0 && total.is_power_of_two(),
        "progress denominator should be 2^players, got {total}"
    );
    assert_eq!(
        control.progress.coalitions_solved.load(Ordering::Relaxed),
        total,
        "every coalition should be counted as solved"
    );
}

/// A pre-cancelled control aborts before any aggregation and surfaces as
/// `ShapleyError::Cancelled` (the worker maps this to a cancelled job).
#[test]
fn test_pre_cancelled_control_returns_cancelled() {
    use std::sync::atomic::Ordering;

    let control = ComputeControl::default();
    control.cancel.store(true, Ordering::Relaxed);
    let err = fixture_input("tests/demand1.csv", 1.0, 1.0)
        .network_link_estimate_cancellable("Alpha", &control)
        .expect_err("pre-cancelled control must abort");
    assert!(
        matches!(err, ShapleyError::Cancelled),
        "expected Cancelled, got {err:?}"
    );
}
