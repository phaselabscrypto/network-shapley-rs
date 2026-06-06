//! End-to-end benchmark of the exact Shapley `compute()` path on the bundled
//! 8-operator reference fixture (256 coalition LP solves). This is the same shape
//! as the per-city exact reward solve, so it is the right place to measure the
//! warm-start win.
//!
//! Run on this branch and on `origin/main` (or use a saved baseline) to compare
//! fresh-build vs warm-start. The `--bench coalition` selector is required: a bare
//! `cargo bench` also runs the lib's default test harness, which rejects
//! criterion's `--save-baseline`/`--baseline` flags.
//!
//! ```text
//! cargo bench --features serde --bench coalition -- --save-baseline warmstart
//! # then, on origin/main (or after a change):
//! cargo bench --features serde --bench coalition -- --baseline warmstart
//! ```

use std::fs::File;
use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use network_shapley::{
    shapley::ShapleyInput,
    types::{Demand, Device, PrivateLink, PublicLink},
};

fn read_csv<T: serde::de::DeserializeOwned>(path: &str) -> Vec<T> {
    let file = File::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    csv::Reader::from_reader(file)
        .deserialize()
        .map(|row| row.expect("deserialize csv row"))
        .collect()
}

fn reference_input(demand_file: &str) -> ShapleyInput {
    ShapleyInput {
        private_links: read_csv::<PrivateLink>("tests/private_links.csv"),
        devices: read_csv::<Device>("tests/devices.csv"),
        public_links: read_csv::<PublicLink>("tests/public_links.csv"),
        demands: read_csv::<Demand>(demand_file),
        operator_uptime: 0.98,
        contiguity_bonus: 5.0,
        demand_multiplier: 1.2,
    }
}

fn bench_exact_compute(c: &mut Criterion) {
    let mut group = c.benchmark_group("exact_compute");
    group.sample_size(20);
    for demand_file in ["tests/demand1.csv", "tests/demand2.csv"] {
        let input = reference_input(demand_file);
        let label = demand_file.rsplit('/').next().unwrap_or(demand_file);
        group.bench_function(label, |b| {
            b.iter(|| black_box(input.compute().expect("compute")));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_exact_compute);
criterion_main!(benches);
