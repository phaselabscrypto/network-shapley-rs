//! The per-LP time limit must surface as a LOUD error — never a hang and never
//! silently mis-valued coalitions. With an absurdly small limit both the warm
//! solve and the cold rescue exceed it, so the whole computation must fail
//! with an `LpSolver` error naming the limit.
//!
//! Run with: cargo test --features serde --test lp_time_limit

use std::fs::File;

use network_shapley::{error::ShapleyError, shapley::ShapleyInput};

fn read_csv<T: serde::de::DeserializeOwned>(path: &str) -> Vec<T> {
    let file = File::open(path).unwrap_or_else(|e| panic!("Failed to open {path}: {e}"));
    csv::Reader::from_reader(file)
        .deserialize()
        .map(|r| r.unwrap())
        .collect()
}

#[test]
fn absurd_time_limit_fails_loudly_instead_of_hanging_or_corrupting() {
    // Own test binary → own process; set before the limit's OnceLock is read.
    // SAFETY: no other threads are running this early in the test process.
    unsafe { std::env::set_var("SHAPLEY_LP_TIME_LIMIT_SECS", "0.000001") };

    let input = ShapleyInput {
        private_links: read_csv("tests/private_links.csv"),
        devices: read_csv("tests/devices.csv"),
        demands: read_csv("tests/demand1.csv"),
        public_links: read_csv("tests/public_links.csv"),
        operator_uptime: 1.0,
        contiguity_bonus: 5.0,
        demand_multiplier: 1.0,
    };

    let err = input
        .compute()
        .expect_err("a time-limited solve must error, not return values");
    assert!(
        matches!(&err, ShapleyError::LpSolver(msg) if msg.contains("time limit")),
        "expected a loud time-limit LpSolver error, got {err:?}"
    );
}
