//! Per-link Shapley value-add for a single focus operator — a faithful port of
//! the Python reference `network_linkestimate` (network-shapley/network_linkestimate.py).
//!
//! Where [`ShapleyInput::compute`] gives a Shapley value per *operator*, this
//! retags each focus-owned link as its own integer *pseudo-operator* (collapsing
//! every other operator to `"Others"` and on/off-ramp helper edges to `"Private"`),
//! then runs ONE exact 2^n coalition Shapley over those link-players. Each focus
//! link's `value` is its Shapley value; `percent` is its share of the positive
//! total. This is single-shot over the whole demand set — the per-source-city +
//! stake-weighted aggregation is the *reward* methodology (`compute()` callers),
//! not link estimation.
//!
//! Faithfulness notes vs the Python reference:
//! - `operator_uptime` is forced to `1.0` (the per-link `Uptime` bandwidth penalty
//!   in [`consolidate_links`] still applies; only the coalition-level expectation
//!   pass is skipped).
//! - link-ownership is OR semantics (a link is focus-owned iff *either* endpoint's
//!   operator is the focus operator).
//! - the same `< 21` player cap is enforced (both on raw operators, via
//!   [`check_inputs`], and on the post-retag players).

use std::collections::{HashMap, HashSet};
use std::sync::atomic::Ordering;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{
    consolidation::{consolidate_demand, consolidate_links},
    constants::{CITY_PREFIX_LEN, MAX_OPERATORS, OP_OTHERS, OP_PRIVATE, OP_PUBLIC},
    error::{Result, ShapleyError},
    shapley::{ComputeControl, ShapleyInput, compute_shapley_values, solve_coalitions_over_map},
    types::ConsolidatedLink,
    validation::check_inputs,
};

/// Operator tags that never correspond to a focus link in the output.
const DROP_TAGS: [&str; 3] = [OP_PUBLIC, OP_PRIVATE, OP_OTHERS];

/// Per-link value-add for one focus operator. One row per focus-owned link, in the
/// canonical `device1 < device2` orientation.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct LinkEstimate {
    pub device1: String,
    pub device2: String,
    pub bandwidth: f64,
    pub latency: f64,
    /// The link's Shapley value (signed; may be negative).
    pub value: f64,
    /// `max(value, 0) / Σ max(value, 0)` over the returned links — a 0–1 fraction
    /// (0 when the positive total is 0).
    pub percent: f64,
}

impl ShapleyInput {
    /// Per-link Shapley value-add for `operator_focus` (faithful port of Python
    /// `network_linkestimate`). See module docs.
    ///
    /// Returns [`ShapleyError::TooManyOperators`] when the network has more than 20
    /// operators, or when the focus operator resolves to more than 20 link-players
    /// (both mirror the Python `n_ops < 21` asserts).
    pub fn network_link_estimate(&self, operator_focus: &str) -> Result<Vec<LinkEstimate>> {
        self.link_estimate_inner(operator_focus, None)
    }

    /// Like [`Self::network_link_estimate`] but cooperatively cancellable with live
    /// per-coalition progress via `control` (set `control.cancel` from another
    /// thread to stop early → [`ShapleyError::Cancelled`]; read `control.progress`
    /// to drive a bar — the denominator is `2^(player count)`).
    pub fn network_link_estimate_cancellable(
        &self,
        operator_focus: &str,
        control: &ComputeControl,
    ) -> Result<Vec<LinkEstimate>> {
        self.link_estimate_inner(operator_focus, Some(control))
    }

    fn link_estimate_inner(
        &self,
        operator_focus: &str,
        control: Option<&ComputeControl>,
    ) -> Result<Vec<LinkEstimate>> {
        // Python fixes operator_uptime = 1.0 for link estimation. The per-link
        // `Uptime` penalty inside `consolidate_links` still applies; only the
        // coalition-level expectation pass is skipped (evalue == svalue).
        let operator_uptime = 1.0;

        // Python `network_linkestimate` pre-checks (network_linkestimate.py:97–106),
        // ahead of the shared `check_inputs`.
        self.check_link_estimate_inputs(operator_focus)?;
        check_inputs(
            &self.private_links,
            &self.devices,
            &self.demands,
            &self.public_links,
            operator_uptime,
        )?;

        // Consolidate over the full demand set, then retag focus links as
        // pseudo-operators (network_linkestimate.py:112–116).
        let full_demand = consolidate_demand(&self.demands, self.demand_multiplier)?;
        let mut full_map = consolidate_links(
            &self.private_links,
            &self.devices,
            &full_demand,
            &self.public_links,
            self.contiguity_bonus,
        )?;
        retag_links(&mut full_map, operator_focus)?;

        // Players = unique edge operators minus the always-in sentinels. "Others"
        // counts as a player (it can carry value), matching Python.
        let mut operators: Vec<String> = full_map
            .iter()
            .flat_map(|l| [l.operator1.clone(), l.operator2.clone()])
            .filter(|op| op != OP_PRIVATE && op != OP_PUBLIC)
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        operators.sort();

        if operators.len() > MAX_OPERATORS {
            return Err(ShapleyError::TooManyOperators {
                count: operators.len(),
                limit: MAX_OPERATORS,
            });
        }
        if operators.is_empty() {
            // Focus operator owns no links (and there is nothing else to value).
            return Ok(Vec::new());
        }

        // Size the progress denominator (2^players) for a cancellable run so a
        // poller can render `coalitions_solved / total`.
        if let Some(c) = control {
            let n_coalitions = 1usize << operators.len();
            c.progress.reset();
            c.progress
                .max_samples
                .store(n_coalitions, Ordering::Relaxed);
            c.progress
                .batch_samples
                .store(n_coalitions, Ordering::Relaxed);
            c.progress
                .batch_total
                .store(n_coalitions, Ordering::Relaxed);
        }

        // One exact 2^n coalition Shapley over the link-players, reusing the
        // warm-start solve core. uptime == 1.0 → no seed cache / changed operators.
        let expected = solve_coalitions_over_map(
            &operators,
            &full_map,
            &full_demand,
            operator_uptime,
            HashMap::new(),
            &[],
            control,
        )?;
        let shapley = compute_shapley_values(&expected, operators.len());

        let value_of: HashMap<&str, f64> = operators
            .iter()
            .map(|o| o.as_str())
            .zip(shapley.iter().copied())
            .collect();

        // Output rows (network_linkestimate.py:171–181): drop edges whose BOTH
        // operators are sentinels, keep the canonical `device1 < device2`
        // orientation, and map the non-dropped operator to its Shapley value.
        let mut links: Vec<LinkEstimate> = Vec::new();
        for l in &full_map {
            let drop1 = DROP_TAGS.contains(&l.operator1.as_str());
            let drop2 = DROP_TAGS.contains(&l.operator2.as_str());
            if (drop1 && drop2) || l.device1 >= l.device2 {
                continue;
            }
            let op = if drop1 {
                l.operator2.as_str()
            } else {
                l.operator1.as_str()
            };
            links.push(LinkEstimate {
                device1: l.device1.clone(),
                device2: l.device2.clone(),
                bandwidth: l.bandwidth,
                latency: l.latency,
                value: value_of.get(op).copied().unwrap_or(0.0),
                percent: 0.0,
            });
        }

        let total: f64 = links.iter().map(|r| r.value.max(0.0)).sum();
        for r in &mut links {
            r.percent = if total > 0.0 {
                r.value.max(0.0) / total
            } else {
                0.0
            };
        }

        Ok(links)
    }

    /// The two `network_linkestimate`-specific pre-checks (network_linkestimate.py:97–106):
    /// no shared-group reuse among the focus operator's links, and no duplicate
    /// links. Both protect the symmetric-pair retag, which assumes a focus link is
    /// uniquely identified by `(unordered devices, bandwidth, latency)`.
    fn check_link_estimate_inputs(&self, operator_focus: &str) -> Result<()> {
        let device_op: HashMap<&str, &str> = self
            .devices
            .iter()
            .map(|d| (d.device.as_str(), d.operator.as_str()))
            .collect();
        let owns = |l: &crate::types::PrivateLink| {
            device_op.get(l.device1.as_str()).copied() == Some(operator_focus)
                || device_op.get(l.device2.as_str()).copied() == Some(operator_focus)
        };

        // No shared-group id may appear on more than one focus-owned link.
        let mut seen_shared: HashSet<u32> = HashSet::new();
        for l in self.private_links.iter().filter(|l| owns(l)) {
            if let Some(group) = l.shared
                && !seen_shared.insert(group)
            {
                return Err(ShapleyError::Validation(
                    "Shared groups are not allowed for links by operator_focus.".to_string(),
                ));
            }
        }

        // No two links may share (unordered devices, bandwidth, latency).
        let mut seen_links: HashSet<(String, String, u64, u64)> = HashSet::new();
        for l in self.private_links.iter() {
            let (a, b) = if l.device1 <= l.device2 {
                (l.device1.clone(), l.device2.clone())
            } else {
                (l.device2.clone(), l.device1.clone())
            };
            let key = (a, b, float_key(l.bandwidth), float_key(l.latency));
            if !seen_links.insert(key) {
                return Err(ShapleyError::Validation(
                    "Duplicate links found.".to_string(),
                ));
            }
        }

        Ok(())
    }
}

/// Key an `f64` for duplicate detection by VALUE equality, matching the Python
/// reference (pandas `==`): `+0.0` and `-0.0` collapse to one key. Plain
/// `to_bits` would let a signed-zero pair slip past the duplicate guard and
/// then confuse the `==`-based symmetric-pair match in [`retag_links`]. NaN is
/// unreachable via serde/CSV inputs, so bit-keying every other value is exact.
fn float_key(x: f64) -> u64 {
    (if x == 0.0 { 0.0f64 } else { x }).to_bits()
}

/// Retag operators so the coalition Shapley values per *link* of `operator_focus`.
/// Faithful port of `network_linkestimate.py:retag_links` (lines 4–59), operating
/// in place on the consolidated edge list (the analogue of the pandas `full_map`).
///
/// # Errors
///
/// Returns [`ShapleyError::Validation`] when a focus-owned link has no symmetric
/// reverse edge. `consolidate_links` always emits the reverse twin with
/// bit-identical floats, so a miss means a broken invariant (e.g. NaN smuggled
/// in via the direct crate API). The Python reference raises `IndexError` here;
/// silently half-tagging the pair would return plausible-looking wrong values.
fn retag_links(links: &mut [ConsolidatedLink], operator_focus: &str) -> Result<()> {
    // 1) Collapse every non-focus, non-public operator to "Others".
    for l in links.iter_mut() {
        if l.operator1 != OP_PUBLIC && l.operator1 != operator_focus {
            l.operator1 = OP_OTHERS.to_string();
        }
        if l.operator2 != OP_PUBLIC && l.operator2 != operator_focus {
            l.operator2 = OP_OTHERS.to_string();
        }
    }

    // 2) Tag links to process: those with the focus operator on either endpoint.
    let mut tag: Vec<bool> = links
        .iter()
        .map(|l| l.operator1 == operator_focus || l.operator2 == operator_focus)
        .collect();

    // 3) Walk tagged links in natural (insertion) order — matching Python's
    //    `full_map` row order — assigning each focus link-pair a fresh integer
    //    pseudo-operator. The symmetric reverse edge gets the same number.
    let mut counter: u32 = 0;
    while let Some(idx) = tag.iter().position(|&t| t) {
        let d1 = links[idx].device1.clone();
        let d2 = links[idx].device2.clone();

        if is_real_device(&d1) && is_real_device(&d2) {
            // Symmetric reverse edge: swapped devices, equal bandwidth + latency.
            // `f64 ==` is exact here — reverse rows clone the same floats and
            // uptime == 1.0 leaves the bandwidth penalty deterministic.
            let bw = links[idx].bandwidth;
            let lat = links[idx].latency;
            let Some(s) = links.iter().position(|l| {
                l.device1 == d2 && l.device2 == d1 && l.bandwidth == bw && l.latency == lat
            }) else {
                return Err(ShapleyError::Validation(format!(
                    "no symmetric reverse link for {d1}->{d2} \
                     (bandwidth {bw}, latency {lat})"
                )));
            };

            counter += 1;
            let tag_n = counter.to_string();

            // Two independent ifs: an intra-focus link (both endpoints focus)
            // assigns the same counter to all four operator slots.
            if links[idx].operator1 == operator_focus {
                links[idx].operator1 = tag_n.clone();
                links[s].operator2 = tag_n.clone();
            }
            if links[idx].operator2 == operator_focus {
                links[idx].operator2 = tag_n.clone();
                links[s].operator1 = tag_n.clone();
            }

            tag[idx] = false;
            tag[s] = false;
        } else {
            // Edge/ramp connection (a non-real-device endpoint): route through the
            // fixed "Private" pathway.
            links[idx].operator1 = OP_PRIVATE.to_string();
            links[idx].operator2 = OP_PRIVATE.to_string();
            tag[idx] = false;
        }
    }

    Ok(())
}

/// Match Python's real-device regex `^[A-Z]{3}([1-9][0-9]*|0[1-9])$`: exactly three
/// uppercase ASCII letters followed by a non-zero-leading integer or a two-char
/// `0[1-9]`. Distinguishes operator switches (`FRA1`, `LON01`) from the crate's
/// consolidated helper nodes — public switches `"{city}00"` and bare cities
/// `"{city}"` — which it rejects.
fn is_real_device(s: &str) -> bool {
    if !s.is_ascii() || s.len() <= CITY_PREFIX_LEN {
        return false;
    }
    let bytes = s.as_bytes();
    if !bytes[..CITY_PREFIX_LEN].iter().all(u8::is_ascii_uppercase) {
        return false;
    }
    let suffix = &bytes[CITY_PREFIX_LEN..];
    if !suffix.iter().all(u8::is_ascii_digit) {
        return false;
    }
    if suffix[0] != b'0' {
        true // [1-9][0-9]* — non-zero leading digit, rest are digits
    } else {
        suffix.len() == 2 && suffix[1] != b'0' // 0[1-9]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_device_matcher_matches_python_regex() {
        for ok in ["NYC1", "LON01", "PAR12", "AAA9", "FRA123"] {
            assert!(is_real_device(ok), "{ok} should be a real device");
        }
        for bad in [
            "NYC00", "NYC", "NYC0", "nyc1", "NY1", "NYCA", "FRA", "FRA1A", "ABCD", "AB12",
        ] {
            assert!(!is_real_device(bad), "{bad} should NOT be a real device");
        }
    }
}
