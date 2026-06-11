//! B3 soundness: topology-aware selective coalition reuse must produce values
//! identical to a full fresh recompute on the modified topology.
//!
//! The core proof uses the EXACT path (`compute_with` + reuse opts vs `compute`), which
//! is deterministic — HiGHS solves the same LP to the same objective, so a reused
//! coalition value (baseline solve of an unchanged sub-LP) is bit-comparable to a
//! fresh solve. The sampled path is checked via the weaker but RNG-independent
//! property that every coalition mask present in both a reuse run and a fresh run
//! carries the same value.
//!
//! Reuse rule (conservative, column-gated): a coalition may take its baseline
//! value iff it excludes EVERY operator that owns a changed primitive. The caller
//! derives `changed_operators` from changed links'/devices' endpoint operators —
//! NEVER from LP row tags (shared-bandwidth rows are tagged first-occurrence-only
//! and would miss the real owner; see `shared_id_change_*`).

use network_shapley::{
    shapley::{ComputeOptions, SamplingConfig, ShapleyInput},
    types::{Demand, Device, PrivateLink, PublicLink},
};
use std::collections::HashMap;

const EPS: f64 = 1e-9;

/// Three operators A,B,C over FRA/AMS/LON, fully connected by public links so
/// every coalition (incl. empty) is feasible. Each operator owns one private
/// link that lowers cost. `a_latency` parameterizes A's link so a "modified"
/// topology can change A's link only.
fn topology(a_latency: f64, a_shared: Option<u32>, b_shared: Option<u32>) -> ShapleyInput {
    // Device names: first 3 chars are the metro (consolidation slices `[..3]`
    // and must match a public-link city). Suffix disambiguates per operator.
    let devices = vec![
        Device::new("FRAa".into(), 10, "A".into()),
        Device::new("AMSa".into(), 10, "A".into()),
        Device::new("FRAb".into(), 10, "B".into()),
        Device::new("AMSb".into(), 10, "B".into()),
        Device::new("LONc".into(), 10, "C".into()),
        Device::new("AMSc".into(), 10, "C".into()),
    ];
    let private_links = vec![
        PrivateLink::new(
            "FRAa".into(),
            "AMSa".into(),
            a_latency,
            100.0,
            1.0,
            a_shared,
        ),
        PrivateLink::new("FRAb".into(), "AMSb".into(), 6.0, 100.0, 1.0, b_shared),
        PrivateLink::new("LONc".into(), "AMSc".into(), 7.0, 100.0, 1.0, None),
    ];
    let public_links = vec![
        PublicLink::new("FRA".into(), "AMS".into(), 50.0),
        PublicLink::new("AMS".into(), "LON".into(), 50.0),
        PublicLink::new("FRA".into(), "LON".into(), 80.0),
    ];
    // Same demand `kind` requires a single (start, traffic, multicast) per the
    // crate's validation (validation.rs) — keep them consistent; only `end` varies.
    let demands = vec![
        Demand::new("FRA".into(), "AMS".into(), 1, 10.0, 1.0, 1, false),
        Demand::new("FRA".into(), "LON".into(), 1, 10.0, 1.0, 1, false),
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

/// Build a baseline coalition-value cache by running sampling with a high cap
/// (small N ⇒ permutation prefixes cover the masks; uncovered masks just get
/// solved fresh in the reuse run, so partial coverage is still sound).
fn baseline_cache(input: &ShapleyInput) -> HashMap<u32, Option<f64>> {
    let cfg = SamplingConfig {
        min_samples: 400,
        max_samples: 400,
        target_se: 0.0, // force the full max so coverage is broad
        batch_size: 50,
    };
    input
        .compute_sampled(cfg)
        .expect("baseline sampling")
        .coalition_cache
}

fn assert_outputs_eq(
    a: &network_shapley::shapley::ShapleyOutput,
    b: &network_shapley::shapley::ShapleyOutput,
    ctx: &str,
) {
    assert_eq!(a.len(), b.len(), "{ctx}: operator count differs");
    for (op, va) in a {
        let vb = b
            .get(op)
            .unwrap_or_else(|| panic!("{ctx}: missing operator {op}"));
        assert!(
            (va.value - vb.value).abs() < EPS,
            "{ctx}: value mismatch for {op}: reuse={} fresh={}",
            va.value,
            vb.value
        );
        assert!(
            (va.proportion - vb.proportion).abs() < EPS,
            "{ctx}: proportion mismatch for {op}: reuse={} fresh={}",
            va.proportion,
            vb.proportion
        );
    }
}

/// D1 (exact): changed link owned by A ⇒ reuse(["A"]) == fresh recompute, byte-tight.
#[test]
fn d1_exact_reuse_equals_fresh() {
    let baseline = topology(5.0, None, None);
    let modified = topology(3.0, None, None); // A's link latency changed

    let seed = baseline_cache(&baseline);
    assert!(!seed.is_empty(), "seed must be populated");

    let reuse = modified
        .compute_with(ComputeOptions {
            seed_cache: seed,
            changed_operators: vec!["A".to_string()],
            ..Default::default()
        })
        .expect("compute_with reuse");
    let fresh = modified.compute().expect("compute");

    assert_outputs_eq(&reuse, &fresh, "d1_exact");

    // The change must actually move A's value vs the baseline (proves we didn't
    // silently reuse A's coalitions — the C1 collapsed-delta bug).
    let base = baseline.compute().expect("baseline compute");
    assert!(
        (base["A"].value - fresh["A"].value).abs() > EPS,
        "A's value should change when A's link changes (base={}, mod={})",
        base["A"].value,
        fresh["A"].value
    );
}

/// D1 (sampled): every coalition mask present in BOTH a reuse run and a fresh run
/// must carry the same value, and the reuse run must actually reuse some masks.
#[test]
fn d1_sampled_common_masks_match_and_reuse_happens() {
    let baseline = topology(5.0, None, None);
    let modified = topology(3.0, None, None);

    let seed = baseline_cache(&baseline);
    let cfg = SamplingConfig {
        min_samples: 300,
        max_samples: 300,
        target_se: 0.0,
        batch_size: 50,
    };

    let reuse = modified
        .compute_sampled_with(
            cfg.clone(),
            ComputeOptions {
                seed_cache: seed,
                changed_operators: vec!["A".to_string()],
                ..Default::default()
            },
        )
        .expect("sampled reuse");
    let fresh = modified.compute_sampled(cfg).expect("sampled fresh");

    assert!(
        reuse.coalitions_reused > 0,
        "expected some coalitions to be reused"
    );

    // bit(A) == 1 << 0 (operators sort to [A,B,C]); ALWAYS_BIT == 1<<31.
    let a_bit = 1u32 << 0;
    let mut common = 0;
    for (mask, rv) in &reuse.coalition_cache {
        if let Some(fv) = fresh.coalition_cache.get(mask) {
            common += 1;
            match (rv, fv) {
                (Some(r), Some(f)) => assert!(
                    (r - f).abs() < EPS,
                    "mask {mask:#x}: reuse={r} fresh={f} (a_bit set: {})",
                    mask & a_bit != 0
                ),
                (None, None) => {}
                _ => panic!("mask {mask:#x}: feasibility mismatch reuse={rv:?} fresh={fv:?}"),
            }
        }
    }
    assert!(common > 0, "no common masks to compare");
}

/// D2 (shared-id trap): two private links share a bandwidth id but belong to
/// different operators. Changing A's link with the CORRECT column-derived set
/// (["A"]) must still equal a fresh recompute — the {A,B} coalition (which carries
/// the shared row) contains A and is therefore re-solved, not reused. A row-tag
/// derived set could be {B} or {} and would wrongly reuse {A,B}; this guards that.
#[test]
fn d2_shared_id_change_reuse_equals_fresh() {
    let baseline = topology(5.0, Some(5), Some(5)); // A & B links share bandwidth id 5
    let modified = topology(3.0, Some(5), Some(5));

    let seed = baseline_cache(&baseline);
    let reuse = modified
        .compute_with(ComputeOptions {
            seed_cache: seed,
            changed_operators: vec!["A".to_string()],
            ..Default::default()
        })
        .expect("compute_with reuse");
    let fresh = modified.compute().expect("compute");
    assert_outputs_eq(&reuse, &fresh, "d2_shared_id");
}

/// D2b (negative guard): using the WRONG changed set (omitting A, as a row-tag
/// derivation might) MUST diverge from a fresh recompute — proving the changed
/// set has to include the changed link's actual endpoint operators.
#[test]
fn d2_wrong_changed_set_diverges() {
    let baseline = topology(5.0, None, None);
    let modified = topology(3.0, None, None);

    let seed = baseline_cache(&baseline);
    // Wrongly claim nothing changed ⇒ whole-cache reuse across topologies (the
    // C1 bug). A's coalitions get stale baseline values ⇒ A's value is wrong.
    let wrong = modified
        .compute_with(ComputeOptions {
            seed_cache: seed,
            ..Default::default()
        })
        .expect("compute_with wrong set");
    let fresh = modified.compute().expect("compute");

    let diff = (wrong["A"].value - fresh["A"].value).abs();
    assert!(
        diff > EPS,
        "wrong changed-set should distort A's value (diff={diff}); if this fails the test topology isn't sensitive enough"
    );
}

/// D6: empty changed set on the SAME topology = whole-cache reuse = fresh.
#[test]
fn d6_empty_set_same_topology_equals_fresh() {
    let input = topology(5.0, None, None);
    let seed = baseline_cache(&input);
    let reuse = input
        .compute_with(ComputeOptions {
            seed_cache: seed,
            ..Default::default()
        })
        .expect("compute_with reuse");
    let fresh = input.compute().expect("compute");
    assert_outputs_eq(&reuse, &fresh, "d6_empty_same_topology");
}

/// D7: every operator changed ⇒ only the empty coalition is reusable; no panic,
/// result still equals a fresh recompute.
#[test]
fn d7_all_changed_degrades_to_fresh() {
    let baseline = topology(5.0, None, None);
    let modified = topology(3.0, None, None);
    let seed = baseline_cache(&baseline);
    let reuse = modified
        .compute_with(ComputeOptions {
            seed_cache: seed,
            changed_operators: vec!["A".to_string(), "B".to_string(), "C".to_string()],
            ..Default::default()
        })
        .expect("compute_with reuse");
    let fresh = modified.compute().expect("compute");
    assert_outputs_eq(&reuse, &fresh, "d7_all_changed");
}
