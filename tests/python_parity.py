#!/usr/bin/env python3
"""Run the Python network_shapley on test fixtures and output JSON for comparison.

Used by the Rust parity test (python_parity_test.rs) to validate that
both implementations produce identical results from the same inputs.

Requires: pip install pandas scipy
"""
import json
import sys
import os

# Find the Python network_shapley module — check common locations
SEARCH_PATHS = [
    os.path.join(os.path.dirname(__file__), "..", "..", "network-shapley-py"),
    os.path.join(os.path.dirname(__file__), "..", "..", "network-shapley"),
    os.environ.get("NETWORK_SHAPLEY_PY_PATH", ""),
]

for path in SEARCH_PATHS:
    if path and os.path.isfile(os.path.join(path, "network_shapley.py")):
        sys.path.insert(0, path)
        break

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from network_shapley import network_shapley

TEST_DIR = os.path.join(os.path.dirname(__file__))


def run_scenario(demand_file: str, multiplier: float) -> dict:
    devices = pd.read_csv(os.path.join(TEST_DIR, "devices.csv"))
    devices.columns = ["Device", "Edge", "Operator"]

    private_links = pd.read_csv(os.path.join(TEST_DIR, "private_links.csv"))
    private_links.columns = ["Device1", "Device2", "Latency", "Bandwidth", "Uptime", "Shared"]

    public_links = pd.read_csv(os.path.join(TEST_DIR, "public_links.csv"))
    public_links.columns = ["City1", "City2", "Latency"]

    demand = pd.read_csv(os.path.join(TEST_DIR, demand_file))
    demand.columns = ["Start", "End", "Receivers", "Traffic", "Priority", "Type", "Multicast"]

    result = network_shapley(
        private_links=private_links,
        devices=devices,
        demand=demand,
        public_links=public_links,
        operator_uptime=0.98,
        contiguity_bonus=5.0,
        demand_multiplier=multiplier,
    )

    return {row["Operator"]: row["Value"] for _, row in result.iterrows()}


def _load_network_linkestimate():
    """Load the reference `network_linkestimate` with its missing imports pre-seeded.

    The reference module omits imports for names it uses (`pd`, `np`, `_assert`,
    `consolidate_*`, `lp_primitives`, ...) — they live in `network_shapley`. It
    even ANNOTATES its defs with `pd.DataFrame`, and on Python <= 3.13
    annotations are evaluated eagerly at def time, so a plain `import` raises
    NameError before any post-import injection could run (PEP 649 defers this
    only from 3.14). Pre-seed a fresh module namespace with network_shapley's
    globals and exec the file body into it — works on every supported Python.
    """
    import importlib.util

    import network_shapley as _ns

    path = os.path.join(os.path.dirname(_ns.__file__), "network_linkestimate.py")
    spec = importlib.util.spec_from_file_location("network_linkestimate", path)
    mod = importlib.util.module_from_spec(spec)
    for _name in dir(_ns):
        if not _name.startswith("__"):
            setattr(mod, _name, getattr(_ns, _name))
    spec.loader.exec_module(mod)
    return mod.network_linkestimate


def run_link_estimate(demand_file: str, focus: str, multiplier: float) -> list:
    network_linkestimate = _load_network_linkestimate()

    devices = pd.read_csv(os.path.join(TEST_DIR, "devices.csv"))
    devices.columns = ["Device", "Edge", "Operator"]

    private_links = pd.read_csv(os.path.join(TEST_DIR, "private_links.csv"))
    private_links.columns = ["Device1", "Device2", "Latency", "Bandwidth", "Uptime", "Shared"]

    public_links = pd.read_csv(os.path.join(TEST_DIR, "public_links.csv"))
    public_links.columns = ["City1", "City2", "Latency"]

    demand = pd.read_csv(os.path.join(TEST_DIR, demand_file))
    demand.columns = ["Start", "End", "Receivers", "Traffic", "Priority", "Type", "Multicast"]

    result = network_linkestimate(
        private_links=private_links,
        devices=devices,
        demand=demand,
        public_links=public_links,
        operator_focus=focus,
        contiguity_bonus=5.0,
        demand_multiplier=multiplier,
    )

    return [
        {
            "device1": str(row["Device1"]),
            "device2": str(row["Device2"]),
            "bandwidth": float(row["Bandwidth"]),
            "latency": float(row["Latency"]),
            "value": float(row["Value"]),
            "percent": float(row["Percent"]),
        }
        for _, row in result.iterrows()
    ]


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "shapley"

    if mode == "link-estimate":
        # Focus operators chosen from the fixture: Alpha owns 3 intra-operator
        # links + 1 cross-operator link; Theta spans LAX/SIN/TYO.
        output = {
            "linkest_demand1_Alpha_1x": run_link_estimate("demand1.csv", "Alpha", 1.0),
            "linkest_demand2_Alpha_1x": run_link_estimate("demand2.csv", "Alpha", 1.0),
            "linkest_demand1_Theta_1x": run_link_estimate("demand1.csv", "Theta", 1.0),
        }
        print(json.dumps(output))
    else:
        output = {
            "demand1_1x": run_scenario("demand1.csv", 1.0),
            "demand1_1.2x": run_scenario("demand1.csv", 1.2),
            "demand2_1x": run_scenario("demand2.csv", 1.0),
            "demand2_1.2x": run_scenario("demand2.csv", 1.2),
        }
        print(json.dumps(output))
