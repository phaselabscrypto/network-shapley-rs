//! Crate-wide constants — single source of truth for magic values.

/// Maximum operators the coalition solver supports (2^N coalitions).
pub(crate) const MAX_OPERATORS: usize = 20;

/// Sentinel operator labels used in coalition membership.
pub(crate) const OP_PUBLIC: &str = "Public";
pub(crate) const OP_PRIVATE: &str = "Private";
pub(crate) const OP_OTHERS: &str = "Others";

/// City code prefix length (e.g. "FRA" from "FRA1").
pub(crate) const CITY_PREFIX_LEN: usize = 3;

/// Public switch device suffix (e.g. "FRA00").
pub(crate) const PUBLIC_SWITCH_SUFFIX: &str = "00";

/// Default LP time limit for HiGHS solver (seconds).
pub(crate) const DEFAULT_LP_TIME_LIMIT_SECS: f64 = 60.0;

/// Priority precision multiplier for integer rounding in LP objectives.
pub(crate) const PRIORITY_PRECISION: f64 = 100.0;

/// Quadratic uptime → effective-availability penalty curve.
///
/// Fitted to operational SLA data: maps raw uptime ∈ \[0, 1\] to an
/// effective fraction ∈ \[0, 1\], heavily penalising below 98%.
///
/// Key points: 100% → 1.0, 99% → ~0.66, 98% → ~0.0, <98% → 0.0.
pub(crate) mod uptime_penalty {
    pub const A: f64 = -1578.9474;
    pub const B: f64 = 3176.3158;
    pub const C: f64 = -1596.3684;

    /// Compute the effective-availability factor for the given raw uptime.
    #[inline]
    pub fn factor(uptime: f64) -> f64 {
        (A * uptime.powi(2) + B * uptime + C).clamp(0.0, 1.0)
    }
}

/// Extract the city-code prefix from a device name, if long enough.
///
/// Returns `None` for device names shorter than [`CITY_PREFIX_LEN`] chars,
/// preventing index-out-of-bounds panics on malformed input.
pub(crate) fn city_prefix(device: &str) -> Option<&str> {
    (device.len() >= CITY_PREFIX_LEN).then(|| &device[..CITY_PREFIX_LEN])
}
