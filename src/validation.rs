use std::collections::HashSet;

use crate::{
    constants::{MAX_OPERATORS, MAX_OPERATORS_PARTIAL_UPTIME, OP_PUBLIC},
    error::{Result, ShapleyError},
    types::{Demands, Devices, PrivateLinks, PublicLinks},
    utils::has_digit,
};

/// Validate that the operator count is within the coalition solver's limit.
///
/// The exact and sampled Shapley paths enumerate/sample over `2^n` coalitions, so
/// the number of operators is capped. Partial uptime (`operator_uptime < 1.0`) runs
/// the more expensive expectation pass and is held to the tighter
/// [`MAX_OPERATORS_PARTIAL_UPTIME`]; full uptime is allowed up to [`MAX_OPERATORS`].
///
/// `n_operators` must already exclude the `Private`/`Public` sentinel operators.
///
/// # Errors
///
/// Returns [`ShapleyError::TooManyOperators`] when `n_operators` exceeds the limit
/// for the given `operator_uptime`.
pub(crate) fn check_operator_limit(n_operators: usize, operator_uptime: f64) -> Result<()> {
    let limit = if operator_uptime < 1.0 {
        MAX_OPERATORS_PARTIAL_UPTIME
    } else {
        MAX_OPERATORS
    };
    if n_operators > limit {
        return Err(ShapleyError::TooManyOperators {
            count: n_operators,
            limit,
        });
    }
    Ok(())
}

/// Validate the structural correctness of network shapley inputs.
///
/// This covers everything except the operator-count cap, which is enforced
/// separately by [`check_operator_limit`] (the compute paths) so that
/// `network_link_estimate` — whose cost depends on a focus operator's link count,
/// not the raw network operator count — is not gated on the number of operators.
pub(crate) fn check_inputs(
    private_links: &PrivateLinks,
    devices: &Devices,
    demands: &Demands,
    public_links: &PublicLinks,
) -> Result<()> {
    // Check for "Public" operator name before filtering
    for device in devices {
        if device.operator == OP_PUBLIC {
            return Err(ShapleyError::Validation(
                "Public is a protected keyword for operator names; choose another.".to_string(),
            ));
        }
    }

    // Check that private links table is labeled correctly
    if private_links.is_empty() {
        return Err(ShapleyError::Validation(
            "There must be at least one private link for this simulation.".to_string(),
        ));
    }

    // Check that public links table is labeled correctly
    for link in public_links {
        if has_digit(&link.city1) {
            return Err(ShapleyError::InvalidCityLabel(format!(
                "City {} should not contain a digit",
                link.city1
            )));
        }
        if has_digit(&link.city2) {
            return Err(ShapleyError::InvalidCityLabel(format!(
                "City {} should not contain a digit",
                link.city2
            )));
        }
    }

    // Check that demand points are labeled correctly
    for demand in demands {
        if has_digit(&demand.start) {
            return Err(ShapleyError::InvalidCityLabel(format!(
                "City {} should not contain a digit",
                demand.start
            )));
        }
        if has_digit(&demand.end) {
            return Err(ShapleyError::InvalidCityLabel(format!(
                "City {} should not contain a digit",
                demand.end
            )));
        }
    }

    // Check that for a given demand type, there is a single origin, size, and multicast flag
    use std::collections::HashMap;
    let mut type_info: HashMap<u32, (&str, f64, bool)> = HashMap::new();

    for demand in demands {
        match type_info.get(&demand.kind) {
            Some(&(start, traffic, multicast)) => {
                if start != demand.start.as_str()
                    || traffic != demand.traffic
                    || multicast != demand.multicast
                {
                    return Err(ShapleyError::DataInconsistency(format!(
                        "Demand type {} has inconsistent properties",
                        demand.kind
                    )));
                }
            }
            None => {
                type_info.insert(
                    demand.kind,
                    (demand.start.as_str(), demand.traffic, demand.multicast),
                );
            }
        }
    }

    // Check there are no duplicate devices
    let device_names: Vec<&str> = devices.iter().map(|d| d.device.as_str()).collect();
    let unique_devices: HashSet<&str> = device_names.iter().cloned().collect();
    if device_names.len() != unique_devices.len() {
        return Err(ShapleyError::DataInconsistency(
            "There are duplicated devices in the list.".to_string(),
        ));
    }

    // Check that every device in private_links appears in devices
    let device_set: HashSet<&str> = devices.iter().map(|d| d.device.as_str()).collect();
    for link in private_links {
        if !device_set.contains(link.device1.as_str()) {
            return Err(ShapleyError::MissingDevice(link.device1.clone()));
        }
        if !device_set.contains(link.device2.as_str()) {
            return Err(ShapleyError::MissingDevice(link.device2.clone()));
        }
    }

    // Check that all demand nodes are reachable by the public network
    let public_nodes: HashSet<&str> = public_links
        .iter()
        .flat_map(|link| [link.city1.as_str(), link.city2.as_str()])
        .collect();

    for demand in demands {
        if !public_nodes.contains(demand.start.as_str()) {
            return Err(ShapleyError::UnreachableDemandNode(demand.start.clone()));
        }
        if !public_nodes.contains(demand.end.as_str()) {
            return Err(ShapleyError::UnreachableDemandNode(demand.end.clone()));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Demand, Device, PrivateLink, PublicLink};

    #[test]
    fn test_valid_inputs() {
        let private_links = vec![PrivateLink::new(
            "SIN1".to_string(),
            "FRA1".to_string(),
            50.0,
            10.0,
            1.0,
            None,
        )];

        let devices = vec![
            Device::new("SIN1".to_string(), 1, "Alpha".to_string()),
            Device::new("FRA1".to_string(), 1, "Beta".to_string()),
        ];

        let public_links = vec![PublicLink::new("SIN".to_string(), "FRA".to_string(), 100.0)];

        let demands = vec![Demand::new(
            "SIN".to_string(),
            "FRA".to_string(),
            1,
            1.0,
            1.0,
            1,
            false,
        )];

        assert!(check_inputs(&private_links, &devices, &demands, &public_links).is_ok());
    }

    #[test]
    fn test_operator_limit_full_uptime() {
        // At full uptime the cap is MAX_OPERATORS (20): 20 is fine, 21 is too many.
        assert!(check_operator_limit(20, 1.0).is_ok());
        assert!(matches!(
            check_operator_limit(21, 1.0),
            Err(ShapleyError::TooManyOperators {
                count: 21,
                limit: 20
            })
        ));
    }

    #[test]
    fn test_operator_limit_partial_uptime() {
        // Below full uptime the cap tightens to MAX_OPERATORS_PARTIAL_UPTIME (15).
        assert!(check_operator_limit(15, 0.9).is_ok());
        assert!(matches!(
            check_operator_limit(16, 0.9),
            Err(ShapleyError::TooManyOperators {
                count: 16,
                limit: 15
            })
        ));
    }
}
