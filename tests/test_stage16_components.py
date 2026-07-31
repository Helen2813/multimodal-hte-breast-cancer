import numpy as np

from scripts._stage16_utils import aipw_components


def test_aipw_component_sum_matches_estimate():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    a = np.array([0, 1, 0, 1])
    e = np.array([0.4, 0.5, 0.6, 0.5])
    mu0 = np.array([1.5, 1.5, 2.5, 2.5])
    mu1 = np.array([2.0, 2.0, 3.0, 3.0])
    result = aipw_components(y, a, e, mu0, mu1)
    summary = result["summary"]
    total = (
        summary["plugin_component_days"]
        + summary["treated_residual_component_days"]
        + summary["control_residual_component_days"]
    )
    assert abs(total - summary["estimate_days"]) < 1e-12
    assert abs(
        result["patient"]["normalized_contribution_days"].sum()
        - summary["estimate_days"]
    ) < 1e-12
