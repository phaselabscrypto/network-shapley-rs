//! Exact-path progress + cancellation via `ComputeOptions`.
//!
//! Guards three properties of the exact-path instrumentation:
//!   1. `compute_with(opts)` with a control returns BYTE-IDENTICAL output to `compute()`
//!      (the instrumentation must not change the math).
//!   2. After a full run, `ctrl.progress.coalitions_solved == input.coalition_count()`
//!      (so a caller can drive `solved / coalition_count` as a progress bar, and
//!      aggregate across many parallel calls).
//!   3. A pre-set `cancel` flag makes the call return `ShapleyError::Cancelled`.

use network_shapley::{
    error::ShapleyError,
    shapley::{ComputeControl, ComputeOptions, ShapleyInput},
    types::{Demand, Device, PrivateLink, PublicLink},
};
use std::sync::atomic::Ordering;

/// Small 2-operator (Alpha/Beta) fixture — same as `simple_test.rs`.
fn small_input() -> ShapleyInput {
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
    ShapleyInput {
        private_links,
        devices,
        demands,
        public_links,
        operator_uptime: 0.98,
        contiguity_bonus: 5.0,
        demand_multiplier: 1.0,
    }
}

#[test]
fn cancellable_matches_plain_and_counts_coalitions() {
    let input = small_input();

    // coalition_count = 2^operators (Alpha, Beta) = 4.
    assert_eq!(input.coalition_count(), 4, "2 operators -> 2^2 coalitions");

    let plain = input.compute().expect("compute");

    let ctrl = ComputeControl::default();
    let cancellable = input
        .compute_with(ComputeOptions {
            control: Some(Box::new(ctrl.clone())),
            ..Default::default()
        })
        .expect("compute_with");

    // (1) byte-identical output — instrumentation didn't touch the math.
    assert_eq!(plain, cancellable, "compute_with must equal compute()");

    // (2) every coalition reported exactly once -> bar reaches 100%.
    assert_eq!(
        ctrl.progress.coalitions_solved.load(Ordering::Relaxed),
        input.coalition_count(),
        "coalitions_solved must reach coalition_count()"
    );
    assert_eq!(
        ctrl.progress.batch_solved.load(Ordering::Relaxed),
        input.coalition_count(),
        "batch_solved tracks coalitions for the smooth-percent denominator"
    );
}

#[test]
fn pre_cancelled_returns_cancelled() {
    let input = small_input();
    let ctrl = ComputeControl::default();
    ctrl.cancel.store(true, Ordering::Relaxed);
    match input.compute_with(ComputeOptions {
        control: Some(Box::new(ctrl)),
        ..Default::default()
    }) {
        Err(ShapleyError::Cancelled) => {}
        other => panic!("expected Cancelled, got {other:?}"),
    }
}
